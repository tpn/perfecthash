/*++

Copyright (c) 2020-2024 Trent Nelson <trent@trent.me>

Module Name:

    Graph.cu

Abstract:

    CUDA graph implementation.

--*/

#include <iostream>
#include <mutex>
#include <thread>
#include <fstream>

#include <cuda.h>
#include <cuda_device_runtime_api.h>

#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>

#define PH_CU
#define HAS_CUDA

#include <PerfectHash.h>

EXTERN_C_BEGIN
#include "../PerfectHash/CuDeviceAttributes.h"
#include "../PerfectHash/stdafx.h"
#include "../PerfectHash/Graph.h"
EXTERN_C_END

#include "Graph.cuh"

#include "../PerfectHash/PerfectHashTableHashExCpp.hpp"

#include <stdio.h>

#include "CuDeviceAttributes.cuh"

namespace cg = cooperative_groups;

#define DEFAULT_BLOCK_SIZE 128

//
// Define helper macros.
//

#undef ASSERT

#define ASSERT(x) if (!(x)) {                                  \
    printf("Assertion failed at %s:%d\n", __FILE__, __LINE__); \
    assert(x);                                                 \
}

#undef EMPTY
#undef IsEmpty

#define EMPTY (-1)
#define IsEmpty(Value) (Value == ((decltype(Value))EMPTY))

template<typename GraphType>
GLOBAL
VOID
GraphCuHashKeysKernel(
    GraphType* Graph
    )
{
    using KeyType = typename GraphType::KeyType;
    using VertexType = typename GraphType::VertexType;
    using VertexPairType = typename GraphType::VertexPairType;
    using ResultType = VertexPairType;

    bool VertexCollision;
    bool WarpCollision;
    uint32_t Index;
    KeyType Key;
    KeyType *Keys;
    VertexPairType Hash;
    VertexPairType *VertexPairs;
    VertexType Mask = Graph->NumberOfVertices - 1;
    uint32_t NumberOfKeys = Graph->NumberOfKeys;
    const int32_t Stride = gridDim.x * blockDim.x;

    auto HashFunction = GraphGetHashFunction(Graph);

    VertexPairs = (VertexPairType *)Graph->VertexPairs;
    Keys = (KeyType *)Graph->DeviceKeys;

    Index = GlobalThreadIndex();

    while (Index < NumberOfKeys) {

        Key = Keys[Index];
        Hash = HashFunction(Key, Mask);

        //
        // For each active thread in the group, check if any thread has
        // the same vertex value as any other thread.  We consider this
        // to be a "warp" vertex collision.
        //

        VertexCollision = false;
        WarpCollision = false;
        auto Group = cg::coalesced_threads();
        auto Rank = Group.thread_rank();

        for (auto Inner = 0; Inner < Group.size(); Inner++) {

            if (Rank == Inner) {

                //
                // Check for an edge-level vertex collision.
                //

                VertexCollision = (Hash.Vertex1 == Hash.Vertex2);
                if (VertexCollision) {
                    AtomicAggIncCGV(&Graph->CuVertexCollisionFailures);
                }

            } else {

                auto Vertex1 = Group.shfl(Hash.Vertex1, Inner);
                auto Vertex2 = Group.shfl(Hash.Vertex2, Inner);

                WarpCollision = (
                    (Vertex1 == Hash.Vertex1 && Vertex2 == Hash.Vertex2) ||
                    (Vertex1 == Hash.Vertex2 && Vertex2 == Hash.Vertex1)
                );

                if (WarpCollision) {
                    AtomicAggIncCGV(&Graph->CuWarpVertexCollisionFailures);
                }
            }
        }

        VertexPairs[Index] = Hash;

        Index += Stride;
    }
}

//
// AddHashedKeys Logic.
//

template<typename GraphType>
DEVICE
bool
GraphCuAddEdge1(
    GraphType* Graph,
    typename GraphType::EdgeType Edge,
    typename GraphType::VertexType VertexIndex
    )
{
    using EdgeType = typename GraphType::EdgeType;
    using DegreeType = typename GraphType::DegreeType;
    using VertexType = typename GraphType::VertexType;
    using Vertex3Type = typename GraphType::Vertex3Type;
    using AtomicCASType = typename GraphType::AtomicVertex3CASType;

    bool Retry = false;
    Vertex3Type *Vertex;
    Vertex3Type *Vertices3;

    PLOCK VertexLocks = (PLOCK)Graph->CuVertexLocks;
    PLOCK EdgeLocks = (PLOCK)Graph->CuEdgeLocks;

    if (!VertexLocks[VertexIndex].TryLock()) {
        Retry = true;
        goto End;
    }

    if (!EdgeLocks[Edge].TryLock()) {
        VertexLocks[VertexIndex].Unlock();
        Retry = true;
        goto End;
    }

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;
    Vertex = &Vertices3[VertexIndex];

    if constexpr (sizeof(VertexType) == sizeof(uint32_t) ||
                  sizeof(VertexType) == sizeof(uint16_t))
    {

        Vertex3Type PrevVertex3;
        EdgeType NextEdges;
        DegreeType NextDegree;
        Vertex3Type NextVertex3;
        AtomicCASType PrevAtomic;
        AtomicCASType NextAtomic;
        AtomicCASType *Address = reinterpret_cast<AtomicCASType*>(Vertex);

        do {
            PrevVertex3 = *Vertex;

            NextDegree = PrevVertex3.Degree + 1;
            NextDegree += 1;

            NextEdges = PrevVertex3.Edges;
            NextEdges ^= Edge;

            NextVertex3.Degree = NextDegree;
            NextVertex3.Edges = NextEdges;

            PrevAtomic = PrevVertex3.Combined.AsLargestIntegral;
            NextAtomic = NextVertex3.Combined.AsLargestIntegral;

        } while (atomicCAS(Address, PrevAtomic, NextAtomic) != PrevAtomic);

    } else if constexpr (sizeof(VertexType) == sizeof(uint8_t)) {

        //
        // Not yet implemented.
        //
    }

    //
    // Unlock edge and vertex locks in the reverse order in which they were
    // acquired.
    //

    EdgeLocks[Edge].Unlock();
    VertexLocks[VertexIndex].Unlock();

    //
    // Intentional fall-through.
    //

End:

    return Retry;
}

template<typename GraphType,
         typename AtomicType = uint32_t>
FORCEINLINE
DEVICE
void GraphCuAddEdge(
    GraphType* Graph,
    typename GraphType::EdgeType Edge,
    typename GraphType::VertexType VertexIndex
    )
{
    using EdgeType = typename GraphType::EdgeType;
    using DegreeType = typename GraphType::DegreeType;
    using VertexType = typename GraphType::VertexType;
    using Vertex3Type = typename GraphType::Vertex3Type;
    using AtomicCASType = typename GraphType::AtomicVertex3CASType;

    Vertex3Type *Vertex;
    Vertex3Type *Vertices3;

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;
    Vertex = &Vertices3[VertexIndex];

    cg::coalesced_group Group = cg::coalesced_threads();
    auto LabeledGroup = cg::labeled_partition(Group, VertexIndex);

    //
    // XOR reduction across all threads in the labeled group.
    //

    auto CumulativeXOR =
        cg::reduce(LabeledGroup, Edge, cg::bit_xor<EdgeType>());

    auto GroupSize = LabeledGroup.num_threads();

    if (LabeledGroup.thread_rank() == 0) {

        //
        // Atomically perform the XOR and increment operations.
        //

        if constexpr (sizeof(VertexType) == sizeof(uint32_t) ||
                      sizeof(VertexType) == sizeof(uint16_t))
        {

#if 0
            //
            // For 32-bit VertexType, use separate atomic operations.
            //

            atomicXor((uint32_t *)&(Vertex->Edges), (uint32_t)Edge);
            atomicAdd((uint32_t *)&(Vertex->Degree), GroupSize);
#endif

            Vertex3Type PrevVertex3;
            EdgeType NextEdges;
            DegreeType NextDegree;
            Vertex3Type NextVertex3;
            AtomicCASType PrevAtomic;
            AtomicCASType NextAtomic;
            AtomicCASType *Address = reinterpret_cast<AtomicCASType*>(Vertex);

            do {
                PrevVertex3 = *Vertex;

                NextDegree = PrevVertex3.Degree;
                NextDegree += GroupSize;

                NextEdges = PrevVertex3.Edges;
                NextEdges ^= CumulativeXOR;

                NextVertex3.Degree = NextDegree;
                NextVertex3.Edges = NextEdges;

                PrevAtomic = PrevVertex3.Combined.AsLargestIntegral;
                NextAtomic = NextVertex3.Combined.AsLargestIntegral;

            } while (atomicCAS(Address, PrevAtomic, NextAtomic) != PrevAtomic);

        } else if constexpr (sizeof(VertexType) == sizeof(uint8_t)) {

            //
            // N.B. Above code is untested and almost certainly incorrect.
            //

            AtomicType PrevEdge;
            AtomicType PrevAtomic;
            AtomicType PrevDegree;
            AtomicType NextAtomic;
            AtomicType NextDegree;

            AtomicType *Address = reinterpret_cast<AtomicType*>(Vertex);

            //
            // Index to pick the appropriate byte permutation.
            //

            uint8_t Index = sizeof(Edge);
            constexpr AtomicType ValueMask = ~0;
            constexpr AtomicType Selection[] = {
                0x3214, 0x3240, 0x3410, 0x4210
            };
            AtomicType Selector = Selection[Index];

            const AtomicType Mask = ~0x3;
            const AtomicType Bucket = ((VertexIndex >> 2) & Mask);
            const uint8_t Index2 = static_cast<uint8_t>(VertexIndex & 0x3);
            const uint32_t Selector2 = Selection[Index];
            uintptr_t Base = reinterpret_cast<uintptr_t>(Graph->Vertices3);
            Base += VertexIndex * sizeof(Vertex3Type);
            Base &= ~0x3;
            AtomicType *Address2 = reinterpret_cast<AtomicType*>(Base);
            AtomicType PrevAtomic2 = *Address2;

            do {
                PrevAtomic = *Address;
                PrevEdge = (PrevAtomic >> (Index * 8)) & ValueMask;
                PrevEdge ^= CumulativeXOR;
                PrevDegree = PrevAtomic & ValueMask;
                NextDegree = PrevDegree + 1;

                AtomicType Combined = (NextDegree << 8) | PrevEdge;
                NextAtomic = __byte_perm(PrevAtomic, Combined, Selector);

            } while (atomicCAS(Address, PrevAtomic, NextAtomic) != PrevAtomic);

        }
    }

    LabeledGroup.sync();
    return;
}

template<typename GraphType>
GLOBAL
VOID
GraphCuAddHashedKeysKernel(
    GraphType* Graph
    )
{
    uint32_t Index;
    uint32_t NumberOfKeys = Graph->NumberOfKeys;
    typename GraphType::EdgeType Edge;
    typename GraphType::VertexPairType VertexPair;
    typename GraphType::VertexPairType *VertexPairs;

    const int32_t Stride = gridDim.x * blockDim.x;

    VertexPairs = (typename GraphType::VertexPairType *)Graph->VertexPairs;

    Index = GlobalThreadIndex();

    while (Index < NumberOfKeys) {

        VertexPair = VertexPairs[Index];
        Edge = Index;

        GraphCuAddEdge(Graph,
                       Edge,
                       VertexPair.Vertex1);

        GraphCuAddEdge(Graph,
                       Edge,
                       VertexPair.Vertex2);

        Index += Stride;
    }
}

template<typename GraphType>
GLOBAL
VOID
GraphCuAddHashedKeysKernel1(
    GraphType* Graph
    )
{
    uint32_t Index;
    uint32_t NumberOfKeys = Graph->NumberOfKeys;
    typename GraphType::EdgeType Edge;
    typename GraphType::VertexPairType VertexPair;
    typename GraphType::VertexPairType *VertexPairs;

    const int32_t Stride = gridDim.x * blockDim.x;

    bool Retry1 = true;
    bool Retry2 = true;

    VertexPairs = (typename GraphType::VertexPairType *)Graph->VertexPairs;

    Index = GlobalThreadIndex();

    while (Index < NumberOfKeys) {

        VertexPair = VertexPairs[Index];
        Edge = Index;

        do {
            if (Retry1) {
                Retry1 = GraphCuAddEdge1(Graph,
                                        Edge,
                                        VertexPair.Vertex1);
            }

            if (Retry2) {
                Retry2 = GraphCuAddEdge1(Graph,
                                        Edge,
                                        VertexPair.Vertex2);
            }
        } while (Retry1 || Retry2);

        Index += Stride;
    }
}

HOST
HRESULT
GetKernelConfig(
    _In_ PGRAPH Graph,
    _In_ PVOID Kernel,
    _In_ ULONG& BlocksPerGrid,
    _In_ ULONG& ThreadsPerBlock,
    _In_ ULONG& SharedMemory
    )
{
    int NumberOfBlocksPerSm;
    ULONG LocalBlocksPerGrid;
    cudaDeviceProp DeviceProperties;
    CUfunction Function = (CUfunction)Kernel;

    if (ThreadsPerBlock == 0) {
        ThreadsPerBlock = DEFAULT_BLOCK_SIZE;
    }

    if (BlocksPerGrid == 0) {
        BlocksPerGrid = (
            ((Graph->NumberOfKeys + ThreadsPerBlock) - 1) / ThreadsPerBlock
        );
    }

    CUDA_CALL(
        cudaGetDeviceProperties(
            &DeviceProperties,
            0
            //Graph->CuDeviceIndex-1
        )
    );

    CUDA_CALL(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &NumberOfBlocksPerSm,
            Function,
            (int)BlocksPerGrid,
            (size_t)SharedMemory
        )
    );

    LocalBlocksPerGrid = (
        DeviceProperties.multiProcessorCount * NumberOfBlocksPerSm
    );
    BlocksPerGrid = max(BlocksPerGrid, LocalBlocksPerGrid);

    return S_OK;
}

EXTERN_C
HOST
HRESULT
GraphCuAddKeys(
    _In_ PGRAPH Graph,
    _In_ ULONG BlocksPerGrid,
    _In_ ULONG ThreadsPerBlock,
    _In_ ULONG SharedMemoryInBytes
    )
{
    HRESULT Result;
    PGRAPH DeviceGraph;
    CUstream_st* Stream;
    PPH_CU_SOLVE_CONTEXT SolveContext;
    ULONG SharedMemory = SharedMemoryInBytes;
    union {
        struct {
            ULONG CuVertexCollisionFailures;
            ULONG CuWarpVertexCollisionFailures;
        };
        ULARGE_INTEGER CuHashKeysCollisionFailures;
    };

    //
    // Verify the requested hash function is supported on GPU.
    //

    if (!GraphIsHashFunctionSupported(Graph)) {
        return PH_E_HASH_FUNCTION_NOT_SUPPORTED_ON_GPU;
    }

    //
    // Get suitable launch parameters for the HashKeys() kernel.
    //

    if (IsUsingAssigned16(Graph)) {
        Result = GetKernelConfig(Graph,
                                 (PVOID)GraphCuHashKeysKernel<GRAPH16>,
                                 BlocksPerGrid,
                                 ThreadsPerBlock,
                                 SharedMemory);
    } else {
        Result = GetKernelConfig(Graph,
                                 (PVOID)GraphCuHashKeysKernel<GRAPH32>,
                                 BlocksPerGrid,
                                 ThreadsPerBlock,
                                 SharedMemory);
    }

    if (FAILED(Result)) {
        goto End;
    }

    //
    // Copy the seeds to constant memory.
    //

    CUDA_CALL(cudaMemcpyToSymbol(c_GraphSeeds,
                                 &Graph->GraphSeeds,
                                 sizeof(Graph->GraphSeeds),
                                 0,
                                 cudaMemcpyHostToDevice));

    //
    // Initialize aliases.
    //

    SolveContext = Graph->CuSolveContext;
    DeviceGraph = SolveContext->DeviceGraph;
    Stream = (CUstream_st *)SolveContext->Stream;

    //
    // Launch the kernel.
    //

    if (IsUsingAssigned16(Graph)) {
        GraphCuHashKeysKernel<<<BlocksPerGrid,
                                ThreadsPerBlock,
                                SharedMemory,
                                Stream>>>((PGRAPH16)DeviceGraph);
    } else {
        GraphCuHashKeysKernel<<<BlocksPerGrid,
                                ThreadsPerBlock,
                                SharedMemory,
                                Stream>>>((PGRAPH32)DeviceGraph);
    }

    CUDA_CALL(
        cudaMemcpyAsync((PVOID)&CuHashKeysCollisionFailures.QuadPart,
                        &DeviceGraph->CuHashKeysCollisionFailures.QuadPart,
                        sizeof(CuHashKeysCollisionFailures.QuadPart),
                        cudaMemcpyDeviceToHost,
                        Stream)
    );

    CUDA_CALL(cudaStreamSynchronize(Stream));

    if (CuWarpVertexCollisionFailures > 0) {
        Result = PH_E_GRAPH_GPU_WARP_VERTEX_COLLISION_FAILURE;
    } else if (CuVertexCollisionFailures > 0) {
        Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
    } else {
        Result = S_OK;
    }

    Graph->CuHashKeysResult = Result;

    if (FAILED(Result)) {
        goto End;
    }

    //
    // All of the keys were hashed without any vertex collisions.  Proceed to
    // add the keys to the graph.
    //

    //
    // Get suitable launch parameters for the AddHashedKeys() kernel.
    //

    if (IsUsingAssigned16(Graph)) {
        Result = GetKernelConfig(Graph,
                                 (PVOID)GraphCuAddHashedKeysKernel<GRAPH16>,
                                 BlocksPerGrid,
                                 ThreadsPerBlock,
                                 SharedMemory);
    } else {
        Result = GetKernelConfig(Graph,
                                 (PVOID)GraphCuAddHashedKeysKernel<GRAPH32>,
                                 BlocksPerGrid,
                                 ThreadsPerBlock,
                                 SharedMemory);
    }

    if (FAILED(Result)) {
        goto End;
    }

    //
    // Launch the kernel.
    //

    if (IsUsingAssigned16(Graph)) {
        GraphCuAddHashedKeysKernel<GRAPH16><<<BlocksPerGrid,
                                              ThreadsPerBlock,
                                              SharedMemory,
                                              Stream>>>((PGRAPH16)DeviceGraph);
    } else {
        GraphCuAddHashedKeysKernel<GRAPH32><<<BlocksPerGrid,
                                              ThreadsPerBlock,
                                              SharedMemory,
                                              Stream>>>((PGRAPH32)DeviceGraph);
    }

    CUDA_CALL(cudaStreamSynchronize(Stream));

    //
    // Intentional follow-on to End.
    //

    Result = S_OK;

End:

    return Result;
}


//
// IsAcyclic logic.
//

template<typename GraphType>
DEVICE
bool
GraphCuRemoveEdgeVertex(
    _In_ GraphType* Graph,
    _In_ typename GraphType::VertexType VertexIndex,
    _In_ typename GraphType::EdgeType Edge,
    _Out_ bool &Removed
    )
{
    using EdgeType = typename GraphType::EdgeType;
    using DegreeType = typename GraphType::DegreeType;
    using VertexType = typename GraphType::VertexType;
    using Vertex3Type = typename GraphType::Vertex3Type;
    using AtomicCASType = typename GraphType::AtomicVertex3CASType;

    bool Retry = false;
    bool DidNotRemove = false;
    Vertex3Type *Vertex;
    Vertex3Type *Vertices3;
    PLOCK VertexLocks = (PLOCK)Graph->CuVertexLocks;
    PLOCK EdgeLocks = (PLOCK)Graph->CuEdgeLocks;

    Removed = false;

    if (!VertexLocks[VertexIndex].TryLock()) {
        Retry = true;
        goto End;
    }

    if (!EdgeLocks[Edge].TryLock()) {
        VertexLocks[VertexIndex].Unlock();
        Retry = true;
        goto End;
    }

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;
    Vertex = (decltype(Vertex))&Vertices3[VertexIndex];
    if (Vertex->Degree == 0) {
        goto Unlock;
    }

    if constexpr (sizeof(VertexType) == sizeof(uint32_t) ||
                  sizeof(VertexType) == sizeof(uint16_t))
    {
        Vertex3Type PrevVertex3;
        EdgeType NextEdges;
        DegreeType NextDegree;
        Vertex3Type NextVertex3;
        AtomicCASType PrevAtomic;
        AtomicCASType NextAtomic;
        AtomicCASType *Address = reinterpret_cast<AtomicCASType*>(Vertex);

        do {
            PrevVertex3 = *Vertex;

            if (PrevVertex3.Degree == 0) {
                DidNotRemove = true;
                break;
            }

            NextDegree = PrevVertex3.Degree;
            NextDegree -= 1;

            NextEdges = PrevVertex3.Edges;
            NextEdges ^= Edge;

            NextVertex3.Degree = NextDegree;
            NextVertex3.Edges = NextEdges;

            PrevAtomic = PrevVertex3.Combined.AsLargestIntegral;
            NextAtomic = NextVertex3.Combined.AsLargestIntegral;

        } while (atomicCAS(Address, PrevAtomic, NextAtomic) != PrevAtomic);

        if (!DidNotRemove) {
            Removed = true;
        }

    } else if constexpr (sizeof(VertexType) == sizeof(uint8_t)) {

        //
        // Not yet implemented.
        //
    }

Unlock:

    //
    // Unlock edge and vertex locks in the reverse order in which they were
    // acquired.
    //

    EdgeLocks[Edge].Unlock();
    VertexLocks[VertexIndex].Unlock();

    //
    // Intentional fall-through.
    //

End:

    return Retry;
}

template<typename GraphType>
DEVICE
bool
GraphCuRemoveVertex(
    _In_ GraphType* Graph,
    _In_ typename GraphType::VertexType VertexIndex
    )
{
    using EdgeType = typename GraphType::EdgeType;
    using DegreeType = typename GraphType::DegreeType;
    using Edge3Type = typename GraphType::Edge3Type;
    using OrderType = typename GraphType::OrderType;
    using VertexType = typename GraphType::VertexType;
    using Vertex3Type = typename GraphType::Vertex3Type;
    using VertexPairType = typename GraphType::VertexPairType;
    using AtomicVertex3Type = typename GraphType::AtomicVertex3Type;
    using OrderIndexType = int32_t;

    bool Retry = false;
    bool Retry1 = true;
    bool Retry2 = true;
    bool Removed1 = false;
    bool Removed2 = false;
    EdgeType Edge;
    //DegreeType Degree;
    Edge3Type Edge3;
    Edge3Type *Edges3 = (decltype(Edges3))Graph->Edges3;
    OrderType *Order = (decltype(Order))Graph->Order;
    OrderType *OrderAddress;
    //OrderIndexType OrderIndex;
    OrderIndexType *GraphOrderIndex =
        (decltype(GraphOrderIndex))&Graph->OrderIndex;

    Vertex3Type Vertex;
    Vertex3Type *Vertices3;

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;

    Vertex = Vertices3[VertexIndex];

    if (Vertex.Degree != 1) {
        goto End;
    }

    Edge = Vertex.Edges;
    Edge3 = Edges3[Edge];

    do {

        if (Retry1) {
            Retry1 = GraphCuRemoveEdgeVertex(Graph,
                                             Edge3.Vertex1,
                                             Edge,
                                             Removed1);
        }

        if (Retry2) {
            Retry2 = GraphCuRemoveEdgeVertex(Graph,
                                             Edge3.Vertex2,
                                             Edge,
                                             Removed2);
        }

    } while (Retry1 || Retry2);

    if (Removed1 || Removed2) {
        AtomicAggIncCGV(&Graph->DeletedEdgeCount);
        AtomicAggSubCGV(GraphOrderIndex);
#if 0
        OrderIndex = AtomicAggSubCG(GraphOrderIndex);
        OrderAddress = &Order[OrderIndex];

        if (OrderIndex >= 0) {
            ASSERT(*OrderAddress == 0);
            ASSERT(Order[OrderIndex] == 0);
            Order[OrderIndex] = Edge;
            ASSERT(Order[OrderIndex] == Edge);
            ASSERT(*OrderAddress == Edge);
        }
#endif
    }

End:
    return Retry;
}


template<typename GraphType>
GLOBAL
VOID
GraphCuIsAcyclicPhase1Kernel(
    GraphType* Graph
    )
{
    using VertexType = typename GraphType::VertexType;

    int32_t Index;

    const int32_t Stride = gridDim.x * blockDim.x;
    const uint32_t NumberOfVertices = Graph->NumberOfVertices;

    Index = GlobalThreadIndex();

    while (Index < NumberOfVertices) {
        bool Retry = true;

        while (Retry) {
            Retry = GraphCuRemoveVertex(Graph, (VertexType)Index);
        }

        Index += Stride;
    }

    return;
}

EXTERN_C
HOST
HRESULT
GraphCuIsAcyclic(
    _In_ PGRAPH Graph,
    _In_ ULONG BlocksPerGrid,
    _In_ ULONG ThreadsPerBlock,
    _In_ ULONG SharedMemoryInBytes
    )
{
    BOOLEAN IsAcyclic = FALSE;
    ULONG Attempts = 0;
    LONG OrderIndexDelta = 0;
    LONG PreviousOrderIndex = 0;
    LONG DeviceOrderIndex = 0;

    HRESULT Result;
    PGRAPH DeviceGraph;
    CUstream_st* Stream;
    PPH_CU_SOLVE_CONTEXT SolveContext;
    ULONG SharedMemory = SharedMemoryInBytes;

    //
    // Get suitable launch parameters for the IsAcyclicPhase1() kernel.
    //

    if (IsUsingAssigned16(Graph)) {
        Result = GetKernelConfig(Graph,
                                 (PVOID)GraphCuIsAcyclicPhase1Kernel<GRAPH16>,
                                 BlocksPerGrid,
                                 ThreadsPerBlock,
                                 SharedMemory);
    } else {
        Result = GetKernelConfig(Graph,
                                 (PVOID)GraphCuIsAcyclicPhase1Kernel<GRAPH32>,
                                 BlocksPerGrid,
                                 ThreadsPerBlock,
                                 SharedMemory);
    }

    if (FAILED(Result)) {
        goto End;
    }

    //
    // Initialize aliases.
    //

    SolveContext = Graph->CuSolveContext;
    DeviceGraph = SolveContext->DeviceGraph;
    Stream = (CUstream_st *)SolveContext->Stream;

    //
    // Enter the kernel launch loop.
    //

    while (TRUE) {

        ++Attempts;

        //
        // Dispatch the appropriate kernel and wait for completion.
        //

        if (IsUsingAssigned16(Graph)) {
            GraphCuIsAcyclicPhase1Kernel<<<BlocksPerGrid,
                                           ThreadsPerBlock,
                                           SharedMemory,
                                           Stream>>>((PGRAPH16)DeviceGraph);
        } else {
            GraphCuIsAcyclicPhase1Kernel<<<BlocksPerGrid,
                                           ThreadsPerBlock,
                                           SharedMemory,
                                           Stream>>>((PGRAPH32)DeviceGraph);
        }

        CUDA_CALL(cudaMemcpyAsync((PVOID)&DeviceOrderIndex,
                                  (PVOID)&DeviceGraph->OrderIndex,
                                  sizeof(DeviceOrderIndex),
                                  cudaMemcpyDeviceToHost,
                                  Stream));

        CUDA_CALL(cudaStreamSynchronize(Stream));

        //
        // Check to see if our OrderIndex has reached 0, this is indicative
        // of an acyclic graph.  (We use <= 0 because we may see -1 or 0 in
        // some cases.)
        //

        if (DeviceOrderIndex <= 0) {

            //
            // We were able to delete all vertices with degree 1, therefore,
            // our graph is acyclic.
            //

            IsAcyclic = TRUE;
            break;
        }

        //
        // If this is our first pass, capture the OrderIndex as previous and
        // continue.
        //

        if (Attempts == 1) {
            PreviousOrderIndex = DeviceOrderIndex;
            continue;
        }

        //
        // Calculate the delta between the current OrderIndex and what we saw
        // on the last pass.  If they haven't changed, it means we weren't able
        // to find any more vertices with degree 1 to delete, which means the
        // graph isn't acyclic.
        //

        OrderIndexDelta = PreviousOrderIndex - DeviceOrderIndex;
        ASSERT(OrderIndexDelta >= 0);
        if (OrderIndexDelta == 0) {
            break;
        }

        //
        // Update previous value and continue for another pass.
        //

        PreviousOrderIndex = DeviceOrderIndex;
    }

    //
    // Capture how many attempts were made to determine if the graph was
    // acyclic.
    //

    Graph->CuIsAcyclicPhase1Attempts = Attempts;

    //
    // Make a note that we're acyclic if applicable in the graph's flags.
    // This is checked by GraphAssign() to ensure we only operate on acyclic
    // graphs.
    //

    if (IsAcyclic) {

        Graph->Flags.IsAcyclic = TRUE;

    } else {

        ULONG HighestDeletedEdges;
        ULONG NumberOfEdgesDeleted;
        PPERFECT_HASH_CONTEXT Context;

        Context = Graph->Info->Context;
        NumberOfEdgesDeleted = Graph->DeletedEdgeCount;

        if (NumberOfEdgesDeleted > Context->HighestDeletedEdgesCount) {

            //
            // Register as the highest deleted edges count if applicable.
            //

            while (TRUE) {

                HighestDeletedEdges = Context->HighestDeletedEdgesCount;

                if (NumberOfEdgesDeleted <= HighestDeletedEdges) {
                    break;
                }

                InterlockedCompareExchange(
                    (PLONG)&Context->HighestDeletedEdgesCount,
                    NumberOfEdgesDeleted,
                    HighestDeletedEdges
                );

            }
        }
    }

End:
    return (IsAcyclic ? S_OK : PH_E_GRAPH_CYCLIC_FAILURE);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab filetype=cuda formatoptions=croql   :
