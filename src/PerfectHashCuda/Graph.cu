/*++

Copyright (c) 2020-2023 Trent Nelson <trent@trent.me>

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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/inner_product.h>

#define PH_CU

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

//#include <limits>

#if defined(__CUDA_VERSION__) && (__CUDA_VERSION__ >= 12000)
// __viaddmin_u32
#error lkasjdflaskdfj
#else
// __viaddmin_u32
__host__ __device__ unsigned int __viaddmin_u32_(unsigned int a,
                                                 unsigned int b,
                                                 unsigned int c) {
    return min(a + b, c);
}
#endif

#undef ASSERT

#define ASSERT(x) if (!(x)) {                                  \
    printf("Assertion failed at %s:%d\n", __FILE__, __LINE__); \
    assert(x);                                                 \
}

//
// Define helper macros.
//

#undef EMPTY
#undef IsEmpty

#define EMPTY ((ULONG)-1)
#define IsEmpty(Value) (Value == ((decltype(Value))EMPTY))


EXTERN_C
DEVICE
BOOLEAN
GraphCuShouldWeContinueTryingToSolve(
    _In_ PGRAPH Graph,
    _Out_ PHRESULT Result
    )
{
    ULONG Target;
    ULONG Timeout;
    ULONG ClockRate;
    BOOLEAN ContinueSolving;
    BOOLEAN CheckTimeout;
    ULONGLONG Delta;
    ULONGLONG ThisClock;
    PCU_DEVICE_ATTRIBUTES Attributes;

    if (Graph->Attempt++ == 0) {
        Graph->CuStartClock = clock64();
    } else {
        ThisClock = clock64();
        Delta = ThisClock - Graph->CuStartClock;
        Attributes = (PCU_DEVICE_ATTRIBUTES)Graph->CuDeviceAttributes;
        ClockRate = Attributes->ClockRate;

        Graph->CuEndClock = ThisClock;
        Graph->CuCycles = Delta;
        Graph->CuElapsedMilliseconds = Delta / ClockRate;

        CheckTimeout = (
            Attributes->KernelExecTimeout > 0 ||
            AlwaysRespectCuKernelRuntimeLimit(Graph)
        );

        if (CheckTimeout) {

            if (Attributes->KernelExecTimeout > 0) {

                //
                // There's a kernel timeout for this device.  Convert it to
                // milliseconds, then use whatever is the smaller value between
                // it and the user-specified kernel runtime limit.
                //
                // N.B. We subtract 10 milliseconds just so we're not *too*
                //      close to the Windows-enforced driver timeout limit.
                //

                Timeout = (Attributes->KernelExecTimeout * 1000) - 10;

                Target = min(Timeout,
                             Graph->CuKernelRuntimeTargetInMilliseconds);
            } else {

                Target = Graph->CuKernelRuntimeTargetInMilliseconds;
            }

            if (Graph->CuElapsedMilliseconds >= Target) {
                *Result = PH_S_CU_KERNEL_RUNTIME_TARGET_REACHED;
                return FALSE;
            }
        }
    }

    *Result = S_OK;

    //return TRUE;
    ContinueSolving = (Graph->Attempt <= 2);
    return ContinueSolving;
}

EXTERN_C
DEVICE
HRESULT
GraphCuApplySeedMasks(
    _In_ PGRAPH Graph
    )
{
    BYTE Index;
    BYTE NumberOfSeeds;
    LONG Mask;
    ULONG NewSeed;
    PULONG Seed;
    PULONG Seeds;
    const LONG *Masks;

    if (!HasSeedMasks(Graph)) {

        //
        // No seed masks are available for this hash routine.
        //

        return S_FALSE;
    }

    //
    // Validation complete.  Loop through the masks and apply those with a value
    // greater than zero to the seed at the corresponding offset.
    //

    Seeds = &Graph->FirstSeed;
    Masks = &Graph->SeedMasks.Mask1;

    NumberOfSeeds = (BYTE)Graph->NumberOfSeeds;

    for (Index = 0; Index < NumberOfSeeds; Index++) {

        Mask = *Masks++;

        if (Mask != -1 && Mask != 0) {

            //
            // Valid mask found, apply it to the seed data at this slot.
            //

            Seed = Seeds + Index;
            NewSeed = *Seed & Mask;
            *Seed = NewSeed;
        }
    }

    return S_OK;
}

EXTERN_C
DEVICE
HRESULT
GraphCuLoadNewSeeds(
    _In_ PGRAPH Graph
    )
/*++

Routine Description:

    Loads new seed data for a graph instance.  This is called prior to each
    solving attempt.

Arguments:

    Graph - Supplies a pointer to the graph instance for which the new seed
        data will be loaded.

Return Value:

    S_OK - Success.

--*/
{
    BYTE Index;
    PULONG Seed;
    BYTE NumberOfSeeds;
    HRESULT Result;
    PCU_KERNEL_CONTEXT Ctx;
    curandStatePhilox4_32_10_t *State;

    if (HasUserSeeds(Graph)) {

        //
        // The user has supplied seeds, so skip the curand() calls.
        //

        goto End;
    }

    Ctx = Graph->CuKernelContext;
    State = &Ctx->RngState.Philox4;

    Seed = &Graph->FirstSeed;
    NumberOfSeeds = (BYTE)Graph->NumberOfSeeds;

    for (Index = 0; Index < NumberOfSeeds; Index++) {
        *Seed++ = curand(State);
    }

End:

    Result = S_OK;

    if (HasSeedMasks(Graph)) {
        Result = GraphCuApplySeedMasks(Graph);
    }

    return Result;
}

template<typename GraphType>
DEVICE
VOID
GraphCuHashKeys(
    GraphType* Graph
    )
{
    using KeyType = typename GraphType::KeyType;
    using VertexType = typename GraphType::VertexType;
    using VertexPairType = typename GraphType::VertexPairType;
    using ResultType = VertexPairType;

    bool Collision;
    uint32_t Index;
    KeyType Key;
    KeyType *Keys;
    VertexPairType Hash;
    VertexPairType *VertexPairs;
    VertexType Mask = Graph->NumberOfVertices - 1;
    uint32_t NumberOfKeys = Graph->NumberOfKeys;

    auto HashFunction = GraphGetHashFunction(Graph);

    VertexPairs = (VertexPairType *)Graph->VertexPairs;
    Keys = (KeyType *)Graph->DeviceKeys;

    Index = GlobalThreadIndex();

#if 0
    if (Index == 0) {
        printf("GraphCuHashKeys: Graph: %p\n", Graph);
        auto OrderByVertex = Graph->CuSolveContext->HostGraph->OrderByVertex;
        printf("GraphCuHashKeys: OrderByVertex: %p\n",
               OrderByVertex);
        Graph->OrderByVertex = OrderByVertex;
        printf("GraphCuHashKeys: Graph->OrderByVertex: %p\n",
               Graph->OrderByVertex);
    }
#endif

    Collision = false;

    while (Index < NumberOfKeys) {

        Key = Keys[Index];
        Hash = HashFunction(Key, Mask);

#if 0
        //
        // For each active thread in the group, check if any thread has
        // the same vertex value as any other thread.
        //

        auto g = cg::coalesced_threads();
        auto t = g.thread_rank();

        for (auto i = 0; i < g.size(); i++) {

            if (t == i) {
                if (Hash.Vertex1 == Hash.Vertex2) {
                    Collision = true;
                }
            }

            auto v1 = g.shfl(Hash.Vertex1, i);
            auto v2 = g.shfl(Hash.Vertex2, i);

            if (t != i) {
                Collision = (
                    Collision ||
                    (v1 == Hash.Vertex1 && v2 == Hash.Vertex2) ||
                    (v1 == Hash.Vertex2 && v2 == Hash.Vertex1)
                );
            }

        }

        if (g.any(Collision)) {
            Collision = true;
        }

        if (Collision) {
            break;
        }
#endif

        VertexPairs[Index] = Hash;

        Index += gridDim.x * blockDim.x;

#if 0
        if (Hash.Vertex1 > Hash.Vertex2) {
            auto Temp = Hash.Vertex1;
            Hash.Vertex1 = Hash.Vertex2;
            Hash.Vertex2 = Temp;
        }
#endif

#if 0
        Vertex1Count = AtomicAggIncMultiCGEx(Counts, Hash.Vertex1);
        Vertex2Count = AtomicAggIncMultiCGEx(Counts, Hash.Vertex2);

        Graph->KeyCounts[Index].Vertex1Count = Vertex1Count;
        Graph->KeyCounts[Index].Vertex2Count = Vertex2Count;
        Graph->KeyCounts[Index].Max = max(Vertex1Count, Vertex2Count);
        Graph->KeyCounts[Index].Sum = __viaddmin_u32(Vertex1Count,
                                                     Vertex2Count,
                                                     MaxCount);
        Graph->KeyCountIndices[Index] = Index;

        auto l1 = cg::labeled_partition(g, Hash.Vertex1);
        if (l1.thread_rank() == 0) {
            atomicAdd(&Graph->VertexCounts[Hash.Vertex1], l1.size());
        }

        auto l2 = cg::labeled_partition(g, Hash.Vertex2);
        if (l2.thread_rank() == 0) {
            atomicAdd(&Graph->VertexCounts[Hash.Vertex2], l2.size());
        }

        Graph->SortedVertexPairs[Index] = Hash;
        Graph->VertexPairIndices[Index] = Index;
#endif

    }

    __syncthreads();

    //
    // Determine if any threads in the block encountered a collision.
    //

    if (Collision) {
        auto g = cg::coalesced_threads();
        if (g.thread_rank() == 0) {
            Graph->CuHashKeysResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
        }
    }
}

template<typename GraphType>
DEVICE
VOID
GraphCuAddKeys(
    GraphType* Graph
    )
{
    using KeyType = typename GraphType::KeyType;
    using VertexType = typename GraphType::VertexType;
    using VertexPairType = typename GraphType::VertexPairType;
    using ResultType = VertexPairType;

    const int32_t Stride = gridDim.x * blockDim.x;

    bool Collision;
    uint32_t Index;
    KeyType Key;
    KeyType *Keys;
    VertexPairType Hash;
    VertexPairType *VertexPairs;
    VertexType Mask = Graph->NumberOfVertices - 1;
    uint32_t NumberOfKeys = Graph->NumberOfKeys;

    auto HashFunction = GraphGetHashFunction(Graph);

    VertexPairs = (VertexPairType *)Graph->VertexPairs;
    Keys = (KeyType *)Graph->DeviceKeys;

    Index = GlobalThreadIndex();

#if 0
    if (Index == 0) {
        printf("GraphCuHashKeys: Graph: %p\n", Graph);
        auto OrderByVertex = Graph->CuSolveContext->HostGraph->OrderByVertex;
        printf("GraphCuHashKeys: OrderByVertex: %p\n",
               OrderByVertex);
        Graph->OrderByVertex = OrderByVertex;
        printf("GraphCuHashKeys: Graph->OrderByVertex: %p\n",
               Graph->OrderByVertex);
    }
#endif

    Collision = false;

    while (Index < NumberOfKeys) {

        Key = Keys[Index];
        Hash = HashFunction(Key, Mask);

#if 0
        //
        // For each active thread in the group, check if any thread has
        // the same vertex value as any other thread.
        //

        auto g = cg::coalesced_threads();
        auto t = g.thread_rank();

        for (auto i = 0; i < g.size(); i++) {

            if (t == i) {
                if (Hash.Vertex1 == Hash.Vertex2) {
                    Collision = true;
                }
            }

            auto v1 = g.shfl(Hash.Vertex1, i);
            auto v2 = g.shfl(Hash.Vertex2, i);

            if (t != i) {
                Collision = (
                    Collision ||
                    (v1 == Hash.Vertex1 && v2 == Hash.Vertex2) ||
                    (v1 == Hash.Vertex2 && v2 == Hash.Vertex1)
                );
            }

        }

        if (g.any(Collision)) {
            Collision = true;
        }

        if (Collision) {
            break;
        }
#endif

        VertexPairs[Index] = Hash;

        Index += gridDim.x * blockDim.x;

#if 0
        if (Hash.Vertex1 > Hash.Vertex2) {
            auto Temp = Hash.Vertex1;
            Hash.Vertex1 = Hash.Vertex2;
            Hash.Vertex2 = Temp;
        }
#endif

#if 0
        Vertex1Count = AtomicAggIncMultiCGEx(Counts, Hash.Vertex1);
        Vertex2Count = AtomicAggIncMultiCGEx(Counts, Hash.Vertex2);

        Graph->KeyCounts[Index].Vertex1Count = Vertex1Count;
        Graph->KeyCounts[Index].Vertex2Count = Vertex2Count;
        Graph->KeyCounts[Index].Max = max(Vertex1Count, Vertex2Count);
        Graph->KeyCounts[Index].Sum = __viaddmin_u32(Vertex1Count,
                                                     Vertex2Count,
                                                     MaxCount);
        Graph->KeyCountIndices[Index] = Index;

        auto l1 = cg::labeled_partition(g, Hash.Vertex1);
        if (l1.thread_rank() == 0) {
            atomicAdd(&Graph->VertexCounts[Hash.Vertex1], l1.size());
        }

        auto l2 = cg::labeled_partition(g, Hash.Vertex2);
        if (l2.thread_rank() == 0) {
            atomicAdd(&Graph->VertexCounts[Hash.Vertex2], l2.size());
        }

        Graph->SortedVertexPairs[Index] = Hash;
        Graph->VertexPairIndices[Index] = Index;
#endif

    }

    __syncthreads();

    //
    // Determine if any threads in the block encountered a collision.
    //

    if (Collision) {
        auto g = cg::coalesced_threads();
        if (g.thread_rank() == 0) {
            Graph->CuHashKeysResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
        }
    }
}

EXTERN_C
GLOBAL
VOID
HashKeys(
    _In_ PGRAPH Graph
    )
{
#if 0
    Graph->CuHashKeysResult = S_OK;
    if (GlobalThreadIndex() == 0) {
        printf("Graph->CuHashKeysResult: %d\n", Graph->CuHashKeysResult);
        printf("Graph->Seeds[0]: 0x%08x\n", Graph->Seeds[0]);
        printf("Graph->Seeds[1]: 0x%08x\n", Graph->Seeds[1]);
        printf("Graph->Seeds[2]: 0x%08x\n", Graph->Seeds[2]);
    }

    if (GlobalThreadIndex() == 0) {
        printf("HashKeys(): Graph: %p\n", Graph);
    }
#endif

    if (IsUsingAssigned16(Graph)) {
        GraphCuHashKeys<GRAPH16>((PGRAPH16)Graph);
    } else {
        GraphCuHashKeys<GRAPH32>((PGRAPH32)Graph);
    }

#if 0
    if (GlobalThreadIndex() == 0) {
        printf("Graph->CuHashKeysResult: %d\n", Graph->CuHashKeysResult);
    }
#endif
    return;
}

EXTERN_C
HOST
VOID
HashKeysHost(
    _In_ PGRAPH Graph,
    _In_ ULONG BlocksPerGrid,
    _In_ ULONG ThreadsPerBlock,
    _In_ ULONG SharedMemoryInBytes
    )
{
    PPH_CU_SOLVE_CONTEXT SolveContext;

    SolveContext = Graph->CuSolveContext;

    printf("HashKeysHost: Graph: %p\n", Graph);

    if (!GraphIsHashFunctionSupported(Graph)) {
        printf("Unsupported hash function ID on GPU: %d.\n",
               Graph->HashFunctionId);
        return;
    }

    CUDA_CALL(cudaMemcpyToSymbol(c_GraphSeeds,
                                 &Graph->GraphSeeds,
                                 sizeof(Graph->GraphSeeds),
                                 0,
                                 cudaMemcpyHostToDevice));

    HashKeys<<<BlocksPerGrid,
               ThreadsPerBlock,
               SharedMemoryInBytes,
               (CUstream_st *)SolveContext->Stream>>>(Graph);
}


//
// AddHashedKeys Logic.
//

template<typename GraphType>
HOST
VOID
GraphCuPrintOneVertices(
    GraphType* Graph,
    uint32_t Index
    )
{
    using Vertex3Type = typename GraphType::Vertex3Type;
    typename GraphType::Vertex3Type *Vertices3;

    Vertices3 = (typename GraphType::Vertex3Type *)Graph->Vertices3;
    Vertex3Type Vertex3 = Vertices3[Index];
    printf("Vertex %d has degree %d, edges: %x\n",
           Index,
           Vertex3.Degree,
           Vertex3.Edges);
}


template<typename GraphType,
         typename AtomicType = uint32_t>
FORCEINLINE
DEVICE
void GraphCuAddEdge2(
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

            //__threadfence();


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
FORCEINLINE
DEVICE
void GraphCuAddEdge(
    GraphType* Graph,
    typename GraphType::EdgeType Edge,
    typename GraphType::VertexType VertexIndex
    )
{
    using EdgeType = typename GraphType::EdgeType;
    using VertexType = typename GraphType::VertexType;
    using Vertex3Type = typename GraphType::Vertex3Type;
    using AtomicVertex3Type = typename GraphType::AtomicVertex3Type;

    AtomicVertex3Type *Vertex;
    AtomicVertex3Type *Vertices3;

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;
    Vertex = &Vertices3[VertexIndex];

    cg::coalesced_group Group = cg::coalesced_threads();
    auto LabeledGroup = cg::labeled_partition(Group, VertexIndex);

    //
    // XOR reduction across all threads in the labeled group.
    //

    auto CumulativeXOR =
        cg::reduce(LabeledGroup, Edge, cg::bit_xor<EdgeType>());

    if (LabeledGroup.size() > 1) {
        for (int i = 0; i < LabeledGroup.size(); i++) {
            if (i == LabeledGroup.thread_rank()) {
                printf("[%d,%d,%d] LabeledGroup[%d/%d]->Edge: %x\n",
                        blockIdx.x,
                        threadIdx.x,
                        threadIdx.y,
                        i,
                        LabeledGroup.size(),
                        Edge);
            }
        }
        LabeledGroup.sync();
    }

    if (LabeledGroup.thread_rank() == 0) {
        bool print = false;

        //
        // Print the size of the labeled group, and then print the cumulative
        // XOR value, the vertex index, and the vertex's edges and degree
        // values before and after the atomic operations.  Prefix each line
        // with the thread/block dimensions.
        //

#if 0
        if (LabeledGroup.size() > 1 ||
            Vertex->Degree.load() != 0 ||
            Vertex->Edges.load() != 0) {
            print = true;
        }
#endif
        if (LabeledGroup.size() > 1) {
            print = true;
        }
        if (print) {
            printf("[%d,%d,%d] LabeledGroup.size(): %d\n",
                   blockIdx.x,
                   threadIdx.x,
                   threadIdx.y,
                   LabeledGroup.size());
            printf("[%d,%d,%d] Edge: %x\n",
                   blockIdx.x,
                   threadIdx.x,
                   threadIdx.y,
                   Edge);
            printf("[%d,%d,%d] CumulativeXOR: %x\n",
                   blockIdx.x,
                   threadIdx.x,
                   threadIdx.y,
                   CumulativeXOR);
            printf("[%d,%d,%d] VertexIndex: %d\n",
                    blockIdx.x,
                    threadIdx.x,
                    threadIdx.y,
                    VertexIndex);
            printf("[%d,%d,%d] Before: Vertex->Edges: %x\n",
                    blockIdx.x,
                    threadIdx.x,
                    threadIdx.y,
                    Vertex->Edges.load());
            printf("[%d,%d,%d] Before: Vertex->Degree: %x\n",
                    blockIdx.x,
                    threadIdx.x,
                    threadIdx.y,
                    Vertex->Degree.load());

        }

        Vertex->Degree += LabeledGroup.size();
        Vertex->Edges ^= CumulativeXOR;

        if (print) {

            printf("[%d,%d,%d] After: Vertex->Edges: %x\n",
                    blockIdx.x,
                    threadIdx.x,
                    threadIdx.y,
                    Vertex->Edges.load());
            printf("[%d,%d,%d] After: Vertex->Degree: %x\n",
                    blockIdx.x,
                    threadIdx.x,
                    threadIdx.y,
                    Vertex->Degree.load());
        }
    }

    LabeledGroup.sync();
    return;
}

template<typename GraphType>
FORCEINLINE
DEVICE
void GraphCuAddEdge3(
    GraphType* Graph,
    typename GraphType::EdgeType Edge,
    typename GraphType::VertexType VertexIndex
    )
{
    using EdgeType = typename GraphType::EdgeType;
    using VertexType = typename GraphType::VertexType;
    using Vertex3Type = typename GraphType::Vertex3Type;
    using AtomicVertex3Type = typename GraphType::AtomicVertex3Type;

    AtomicVertex3Type *Vertex;
    AtomicVertex3Type *Vertices3;

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;
    Vertex = &Vertices3[VertexIndex];

    cg::coalesced_group Group = cg::coalesced_threads();
    auto LabeledGroup = cg::labeled_partition(Group, VertexIndex);

    int32_t *Degrees = (int32_t *)Graph->_Degrees;
    int32_t *Edges = (int32_t *)Graph->_Edges;

    //
    // XOR reduction across all threads in the labeled group.
    //

    auto CumulativeXOR =
        cg::reduce(LabeledGroup, Edge, cg::bit_xor<EdgeType>());

    if (LabeledGroup.size() > 1) {
        for (int i = 0; i < LabeledGroup.size(); i++) {
            if (i == LabeledGroup.thread_rank()) {
                printf("[%d,%d,%d] LabeledGroup[%d/%d]->Edge: %x\n",
                        blockIdx.x,
                        threadIdx.x,
                        threadIdx.y,
                        i,
                        LabeledGroup.size(),
                        Edge);
            }
        }
        LabeledGroup.sync();
    }

    if (LabeledGroup.thread_rank() == 0) {
        bool print = false;

        //
        // Print the size of the labeled group, and then print the cumulative
        // XOR value, the vertex index, and the vertex's edges and degree
        // values before and after the atomic operations.  Prefix each line
        // with the thread/block dimensions.
        //

#if 0
        if (LabeledGroup.size() > 1 ||
            Vertex->Degree.load() != 0 ||
            Vertex->Edges.load() != 0) {
            print = true;
        }
#endif
        if (LabeledGroup.size() > 1) {
            print = true;
        }
        if (print) {
            printf("[%d,%d,%d] LabeledGroup.size(): %d\n",
                   blockIdx.x,
                   threadIdx.x,
                   threadIdx.y,
                   LabeledGroup.size());
            printf("[%d,%d,%d] Edge: %x\n",
                   blockIdx.x,
                   threadIdx.x,
                   threadIdx.y,
                   Edge);
            printf("[%d,%d,%d] CumulativeXOR: %x\n",
                   blockIdx.x,
                   threadIdx.x,
                   threadIdx.y,
                   CumulativeXOR);
            printf("[%d,%d,%d] VertexIndex: %d\n",
                    blockIdx.x,
                    threadIdx.x,
                    threadIdx.y,
                    VertexIndex);
            printf("[%d,%d,%d] Before: Vertex->Edges: %x\n",
                    blockIdx.x,
                    threadIdx.x,
                    threadIdx.y,
                    Vertex->Edges.load());
            printf("[%d,%d,%d] Before: Vertex->Degree: %x\n",
                    blockIdx.x,
                    threadIdx.x,
                    threadIdx.y,
                    Vertex->Degree.load());

        }

        atomicAdd(&Degrees[VertexIndex], LabeledGroup.size());
        //Vertex->Degree += LabeledGroup.size();
        atomicXor(&Edges[VertexIndex], Edge);
        //Vertex->Edges ^= CumulativeXOR;

        if (print) {

            printf("[%d,%d,%d] After: Vertex->Edges: %x\n",
                    blockIdx.x,
                    threadIdx.x,
                    threadIdx.y,
                    Vertex->Edges.load());
            printf("[%d,%d,%d] After: Vertex->Degree: %x\n",
                    blockIdx.x,
                    threadIdx.x,
                    threadIdx.y,
                    Vertex->Degree.load());
        }
    }

    LabeledGroup.sync();
    return;
}

template<typename GraphType>
HOST
VOID
GraphCuPrintVertices(
    GraphType* Graph
    )
{
#if 0
    typename GraphType::Vertex3Type *Vertices3;

    Vertices3 = (typename GraphType::Vertex3Type *)Graph->Vertices3;

    for (auto i = 0; i < 10; i++) {
        printf("Vertex %d has degree %d, edges: %x\n",
               i,
               Vertices3[i].Degree,
               Vertices3[i].Edges);
    }
#endif
}

template<typename GraphType>
DEVICE
VOID
GraphAddEdge3(
    GraphType* Graph,
    typename GraphType::EdgeType Edge,
    typename GraphType::VertexType Vertex1Index,
    typename GraphType::VertexType Vertex2Index
    )
/*++

Routine Description:

    This routine adds an edge to the hypergraph for two vertices.

Arguments:

    Graph - Supplies a pointer to the graph for which the edge is to be added.

    Edge - Supplies the edge to add to the graph.

    Vertex1 - Supplies the first vertex.

    Vertex2 - Supplies the second vertex.

Return Value:

    None.

--*/
{
#ifdef _DEBUG
    //
    // Invariant checks:
    //
    //      - Vertex1Index should be less than the number of vertices.
    //      - Vertex2Index should be less than the number of vertices.
    //      - Edge should be less than the number of edges.
    //      - The graph must not have started deletions.
    //

    ASSERT(Vertex1Index < Graph->NumberOfVertices);
    ASSERT(Vertex2Index < Graph->NumberOfVertices);
    ASSERT(Edge < Graph->NumberOfEdges);
    ASSERT(!Graph->Flags.Shrinking);
#endif

    //
    // Insert the first edge.
    //

    GraphCuAddEdge(Graph, Edge, Vertex1Index);

    //
    // Insert the second edge.
    //

    GraphCuAddEdge(Graph, Edge, Vertex2Index);
}

template<typename GraphType>
DEVICE
VOID
GraphCuAddHashedKeys(
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
#if 0
    if (Index == 0) {
        printf("GraphCuAddHashedKeys: Graph: %p\n", Graph);
    }
#endif

    while (Index < NumberOfKeys) {

        VertexPair = VertexPairs[Index];
        Edge = Index;

        GraphCuAddEdge2(Graph,
                        Edge,
                        VertexPair.Vertex1);

        GraphCuAddEdge2(Graph,
                        Edge,
                        VertexPair.Vertex2);

        Index += Stride;
    }
}


#define BLOCK_SIZE 128

template<typename GraphType>
DEVICE
VOID
GraphCuAddHashedKeys2(
    GraphType* Graph
    )
{
    using KeyType = typename GraphType::KeyType;
    using EdgeType = typename GraphType::EdgeType;
    using VertexType = typename GraphType::VertexType;
    using VertexPairType = typename GraphType::VertexPairType;

    uint32_t Index;
    uint32_t Batch;
    uint32_t BlockBatchIndex;
    uint32_t NumberOfKeys = Graph->NumberOfKeys;
    VertexType Mask = Graph->NumberOfVertices - 1;

    KeyType Key;
    EdgeType Edge;
    KeyType *Keys;
    VertexPairType VertexPair;
    VertexPairType *VertexPairs;

    const int32_t Stride = gridDim.x * blockDim.x;

    __shared__ KeyType SharedKeys[BLOCK_SIZE];

    auto Grid = cg::this_grid();
    auto Block = cg::this_thread_block();
    auto GroupIndexX = Block.group_index().x;
    auto BlockSize = Block.size();
    auto GridSize = Grid.size();
    auto LocalIndex = Block.thread_rank();
    auto const BlockCopySize = sizeof(KeyType) * BlockSize;

    Keys = (KeyType *)Graph->DeviceKeys;
    VertexPairs = (VertexPairType *)Graph->VertexPairs;

    auto HashFunction = GraphGetHashFunction(Graph);

    Index = GlobalThreadIndex();

    Batch = 0;

    while (Index < NumberOfKeys) {

        BlockBatchIndex = (Block.group_index().x * Block.size());
        BlockBatchIndex += (Grid.size() * Batch);

        cg::memcpy_async(Block,
                         SharedKeys,
                         Keys + BlockBatchIndex,
                         BlockCopySize);

        cg::wait(Block);

        Key = SharedKeys[LocalIndex];
        VertexPair = HashFunction(Key, Mask);

        Edge = Index;

        GraphCuAddEdge2(Graph,
                        Edge,
                        VertexPair.Vertex1);

        GraphCuAddEdge2(Graph,
                        Edge,
                        VertexPair.Vertex2);

        Index += Stride;

        Block.sync();

        VertexPairs[Index] = VertexPair;
    }
}


GLOBAL
VOID
AddHashedKeys(
    _In_ PGRAPH Graph
    )
{
#if 0
    if (GlobalThreadIndex() == 0) {
        PPH_CU_SOLVE_CONTEXT SolveContext;
        PPH_CU_DEVICE_CONTEXT DeviceContext;
        PCU_DEVICE_ATTRIBUTES Attributes;

        SolveContext = Graph->CuSolveContext;
        DeviceContext = SolveContext->DeviceContext;
        Attributes = (PCU_DEVICE_ATTRIBUTES)DeviceContext->DeviceAttributes;

        PrintCuDeviceAttributes<<<1, 1, 1>>>(Attributes);
    }
#endif

    if (IsUsingAssigned16(Graph)) {
        GraphCuAddHashedKeys<GRAPH16>((PGRAPH16)Graph);
    } else {
        GraphCuAddHashedKeys<GRAPH32>((PGRAPH32)Graph);
    }
}

template<typename GraphType>
GLOBAL
VOID
CheckHashedKeys(
    GraphType* Graph
    )
{
    using EdgeType = typename GraphType::EdgeType;
    using VertexType = typename GraphType::VertexType;
    using Vertex3Type = typename GraphType::Vertex3Type;
    using VertexPairType = typename GraphType::VertexPairType;
    using AtomicVertex3Type = typename GraphType::AtomicVertex3Type;

    typename GraphType::VertexPairType Hash;
    typename GraphType::KeyType Key;
    typename GraphType::KeyType *Keys;
    PULONG Seeds = Graph->Seeds;
    Keys = (typename GraphType::KeyType *)Graph->DeviceKeys;
    auto Mask = Graph->NumberOfVertices - 1;

    uint32_t Index;
    uint32_t NumberOfKeys = Graph->NumberOfKeys;

    EdgeType Edge;
    Vertex3Type *Vertex;
    Vertex3Type *Vertices3;
    VertexPairType VertexPair;
    VertexPairType *VertexPairs;

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;
    VertexPairs = (decltype(VertexPairs))Graph->VertexPairs;

    auto HashFunction = GraphGetHashFunction(Graph);

    int32_t *Degrees = (int32_t *)Graph->_Degrees;
    int32_t *Edges = (int32_t *)Graph->_Edges;

    int32_t V1Degree1;
    int32_t V1Degree2;
    int32_t V1Edge1;
    int32_t V1Edge2;

    int32_t V2Degree1;
    int32_t V2Degree2;
    int32_t V2Edge1;
    int32_t V2Edge2;

    Index = GlobalThreadIndex();
    if (Index == 0) {
        printf("CheckAddHashedKeys: Graph: %p\n", Graph);
    }

    while (Index < NumberOfKeys) {

        VertexPair = VertexPairs[Index];
        Edge = Index;

        Key = Keys[Index];
        Hash = HashFunction(Key, Seeds, Mask);

#if 0
        if (Hash.Vertex1 != VertexPair.Vertex1) {
            printf("[%d] Vertex1: %d != %d\n",
                   Index,
                   Hash.Vertex1,
                   VertexPair.Vertex1);
        }

        if (Hash.Vertex2 != VertexPair.Vertex2) {
            printf("[%d] Vertex2: %d != %d\n",
                   Index,
                   Hash.Vertex2,
                   VertexPair.Vertex2);
        }
#endif

#if 0
        V1Degree1 = (int32_t)Vertices3[VertexPair.Vertex1].Degree;
        V1Degree2 = Degrees[VertexPair.Vertex1];
        if (V1Degree1 != V1Degree2) {
            printf("[%d] V1Degree Vertex: %d, %d != %d\n",
                   Index,
                   VertexPair.Vertex1,
                   V1Degree1,
                   V1Degree2);
        }

        V2Degree1 = (int32_t)Vertices3[VertexPair.Vertex2].Degree;
        V2Degree2 = Degrees[VertexPair.Vertex2];
        if (V2Degree1 != V2Degree2) {
            printf("[%d] V2Degree Vertex: %d, %d != %d\n",
                   Index,
                   VertexPair.Vertex2,
                   V2Degree1,
                   V2Degree2);
        }

        V1Edge1 = (int32_t)Vertices3[VertexPair.Vertex1].Edges;
        V1Edge2 = Edges[VertexPair.Vertex1];
        if (V1Edge1 != V1Edge2) {
            printf("[%d] V1Edge Vertex: %d, %x != %x\n",
                   Index,
                   VertexPair.Vertex1,
                   V1Edge1,
                   V1Edge2);
        }

        V2Edge1 = (int32_t)Vertices3[VertexPair.Vertex2].Edges;
        V2Edge2 = Edges[VertexPair.Vertex2];
        if (V2Edge1 != V2Edge2) {
            printf("[%d] V2Edge Vertex: %d, %x != %x\n",
                   Index,
                   VertexPair.Vertex2,
                   V2Edge1,
                   V2Edge2);
        }
#elif 0
        V1Degree1 = (int32_t)Vertices3[VertexPair.Vertex1].Degree;
        V1Degree2 = Degrees[VertexPair.Vertex1];
        if (V1Degree1 == V1Degree2) {
            printf("[%d] V1Degree Vertex: %d, %d == %d\n",
                   Index,
                   VertexPair.Vertex1,
                   V1Degree1,
                   V1Degree2);
        }

        V2Degree1 = (int32_t)Vertices3[VertexPair.Vertex2].Degree;
        V2Degree2 = Degrees[VertexPair.Vertex2];
        if (V2Degree1 == V2Degree2) {
            printf("[%d] V2Degree Vertex: %d, %d == %d\n",
                   Index,
                   VertexPair.Vertex2,
                   V2Degree1,
                   V2Degree2);
        }

        V1Edge1 = (int32_t)Vertices3[VertexPair.Vertex1].Edges;
        V1Edge2 = Edges[VertexPair.Vertex1];
        if (V1Edge1 == V1Edge2) {
            printf("[%d] V1Edge Vertex: %d, %x == %x\n",
                   Index,
                   VertexPair.Vertex1,
                   V1Edge1,
                   V1Edge2);
        }

        V2Edge1 = (int32_t)Vertices3[VertexPair.Vertex2].Edges;
        V2Edge2 = Edges[VertexPair.Vertex2];
        if (V2Edge1 == V2Edge2) {
            printf("[%d] V2Edge Vertex: %d, %x == %x\n",
                   Index,
                   VertexPair.Vertex2,
                   V2Edge1,
                   V2Edge2);
        }
#endif

        Index += gridDim.x * blockDim.x;
    }
}

EXTERN_C
HOST
VOID
AddHashedKeysHost(
    _In_ PGRAPH Graph,
    _In_ ULONG BlocksPerGrid,
    _In_ ULONG ThreadsPerBlock,
    _In_ ULONG SharedMemoryInBytes
    )
{
    PPH_CU_SOLVE_CONTEXT SolveContext;

    SolveContext = Graph->CuSolveContext;
#if 0
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    PCU_DEVICE_ATTRIBUTES Attributes;
    DeviceContext = SolveContext->DeviceContext;
    Attributes = (PCU_DEVICE_ATTRIBUTES)DeviceContext->DeviceAttributes;

    PrintCuDeviceAttributes(Attributes);
#endif

#if 0
    size_t AllocSize = sizeof(int32_t) * Graph->NumberOfVertices;
    CUDA_CALL(cudaMallocManaged((void **)&Graph->_Edges, AllocSize));
    CUDA_CALL(cudaMallocManaged((void **)&Graph->_Degrees, AllocSize));

    CUDA_CALL(cudaMemset(Graph->_Edges, 0, AllocSize));
    CUDA_CALL(cudaMemset(Graph->_Degrees, 0, AllocSize));

    SolveContext->HostGraph->_Edges = Graph->_Edges;
    SolveContext->HostGraph->_Degrees = Graph->_Degrees;
#endif

    printf("AddHashedKeysHost: Graph: %p\n", Graph);

    if (!GraphIsHashFunctionSupported(Graph)) {
        printf("Unsupported hash function ID on GPU: %d.\n",
               Graph->HashFunctionId);
        return;
    }

    CUDA_CALL(cudaMemcpyToSymbol(c_GraphSeeds,
                                 &Graph->GraphSeeds,
                                 sizeof(Graph->GraphSeeds),
                                 0,
                                 cudaMemcpyHostToDevice));

#if PH_WINDOWS
    int NumberOfBlocksPerSm;
    cudaDeviceProp DeviceProperties;
    CUDA_CALL(cudaGetDeviceProperties(&DeviceProperties, 0));
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &NumberOfBlocksPerSm,
        AddHashedKeys,
        (int)ThreadsPerBlock,
        (size_t)SharedMemoryInBytes
    ));

    ULONG LocalBlocksPerGrid = (
        DeviceProperties.multiProcessorCount * NumberOfBlocksPerSm
    );
    BlocksPerGrid = min(BlocksPerGrid, LocalBlocksPerGrid);

    printf("DeviceProperties.multiProcessorCount: %d\n",
           DeviceProperties.multiProcessorCount);
    printf("NumberOfBlocksPerSm: %d\n", NumberOfBlocksPerSm);
#endif
    printf("BlocksPerGrid: %d\n", BlocksPerGrid);
    printf("ThreadsPerBlock: %d\n", ThreadsPerBlock);

    dim3 GridDim(BlocksPerGrid, 1, 1);
    dim3 BlockDim(ThreadsPerBlock, 1, 1);

    AddHashedKeys<<<GridDim,
                    BlockDim,
                    SharedMemoryInBytes,
                    (CUstream_st *)SolveContext->Stream>>>(Graph);

    //
    // Sync the stream to ensure the kernel has completed.
    //

#if 0
    cudaStreamSynchronize((CUstream_st *)SolveContext->Stream);

    if (IsUsingAssigned16(Graph)) {

        CheckHashedKeys<GRAPH16><<<1,
                                   1,
                                   SharedMemoryInBytes,
                                   (CUstream_st *)SolveContext->Stream>>>(
                                        (PGRAPH16)Graph);

    } else {

        CheckHashedKeys<GRAPH32><<<BlocksPerGrid,
                                   ThreadsPerBlock,
                                   SharedMemoryInBytes,
                                   (CUstream_st *)SolveContext->Stream>>>(
                                        (PGRAPH32)Graph);

    }
#endif

    cudaStreamSynchronize((CUstream_st *)SolveContext->Stream);

    //
    // Copy Vertices3 to SortedVertices3.
    //

    CUDA_CALL(cudaMemcpy(Graph->SortedVertices3,
                         Graph->Vertices3,
                         Graph->Info->Vertices3SizeInBytes,
                         cudaMemcpyDeviceToDevice));

}

//
// IsAcyclic v2 logic.
//

template<typename GraphType>
DEVICE
bool
GraphCuRemoveEdge(
    _In_ GraphType* Graph,
    _In_ typename GraphType::Vertex3Type* Vertex,
    _In_ typename GraphType::EdgeType Edge
    )
{
    using EdgeType = typename GraphType::EdgeType;
    using DegreeType = typename GraphType::DegreeType;
    using VertexType = typename GraphType::VertexType;
    using Vertex3Type = typename GraphType::Vertex3Type;
    using AtomicCASType = typename GraphType::AtomicVertex3CASType;

    if constexpr ((sizeof(VertexType) == sizeof(uint32_t)) ||
                  (sizeof(VertexType) == sizeof(uint16_t)))
    {

        Vertex3Type PrevVertex3;
        EdgeType NextEdges;
        DegreeType NextDegree;
        Vertex3Type NextVertex3;
        AtomicCASType PrevAtomic;
        AtomicCASType NextAtomic;
        AtomicCASType *Address = reinterpret_cast<AtomicCASType*>(Vertex);
        int Attempt = 0;

        do {
            ++Attempt;

            if (Vertex->Degree == 0) {
                printf("Degree == 0, Edges: %x, Edge: %x, Attempt: %d\n",
                       Vertex->Edges,
                       Edge,
                       Attempt);
                break;
            }
            ASSERT(Vertex != NULL);
            ASSERT(Vertex->Degree > 0);

            PrevVertex3 = *Vertex;

            NextDegree = PrevVertex3.Degree - 1;

            NextEdges = PrevVertex3.Edges;
            NextEdges ^= Edge;

            NextVertex3.Degree = NextDegree;
            NextVertex3.Edges = NextEdges;

            PrevAtomic = PrevVertex3.Combined.AsLargestIntegral;
            NextAtomic = NextVertex3.Combined.AsLargestIntegral;

        } while (atomicCAS(Address, PrevAtomic, NextAtomic) != PrevAtomic);

        //__threadfence();

    } else if constexpr (sizeof(VertexType) == sizeof(uint8_t)) {

        //
        // Not yet implemented.
        //
    }
}

template<typename GraphType>
DEVICE
bool
GraphCuRemoveEdgeVertex(
    _In_ GraphType* Graph,
    _In_ typename GraphType::VertexType VertexIndex,
    _In_ typename GraphType::EdgeType Edge,
    _In_ PLOCK Locks,
    _Out_ bool &Removed
    )
{
    bool Retry = false;
    using EdgeType = typename GraphType::EdgeType;
    using DegreeType = typename GraphType::DegreeType;
    using VertexType = typename GraphType::VertexType;
    using Vertex3Type = typename GraphType::Vertex3Type;
    using AtomicCASType = typename GraphType::AtomicVertex3CASType;

    Vertex3Type *Vertex;
    Vertex3Type *Vertices3;

    Removed = false;

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;

    cg::coalesced_group Group = cg::coalesced_threads();
    auto LabeledGroup = cg::labeled_partition(Group, VertexIndex);
    auto CumulativeXOR = cg::reduce(LabeledGroup,
                                    Edge,
                                    cg::bit_xor<EdgeType>());

    auto GroupSize = LabeledGroup.num_threads();

    if (LabeledGroup.thread_rank() == 0) {

        if (!Locks[VertexIndex].TryLock()) {
            Retry = true;
            goto End;
        }

        Vertex = (decltype(Vertex))&Vertices3[VertexIndex];
        if (Vertex->Degree == 0) {
            Locks[VertexIndex].Unlock();
            goto End;
        }

        if constexpr (sizeof(VertexType) == sizeof(uint32_t) ||
                      sizeof(VertexType) == sizeof(uint16_t))
        {

#if 0
            atomicXor((uint32_t *)&(Vertex->Edges), (uint32_t)Edge);
            atomicSub((uint32_t *)&(Vertex->Degree), GroupSize);
            Removed = true;
            __threadfence();
#endif

            Vertex3Type PrevVertex3;
            EdgeType NextEdges;
            DegreeType NextDegree;
            Vertex3Type NextVertex3;
            AtomicCASType PrevAtomic;
            AtomicCASType NextAtomic;
            AtomicCASType *Address = reinterpret_cast<AtomicCASType*>(Vertex);
            intptr_t Delta;

            do {
                PrevVertex3 = *Vertex;

                NextDegree = PrevVertex3.Degree;

                Delta = ((intptr_t)NextDegree - GroupSize);
                if (Delta < 0) {
                    Retry = true;
                    break;
                } else {
                    NextDegree -= GroupSize;
                }

                NextEdges = PrevVertex3.Edges;
                NextEdges ^= CumulativeXOR;

                NextVertex3.Degree = NextDegree;
                NextVertex3.Edges = NextEdges;

                PrevAtomic = PrevVertex3.Combined.AsLargestIntegral;
                NextAtomic = NextVertex3.Combined.AsLargestIntegral;

            } while (atomicCAS(Address, PrevAtomic, NextAtomic) != PrevAtomic);

            Removed = true;
            //__threadfence();

        } else if constexpr (sizeof(VertexType) == sizeof(uint8_t)) {

            //
            // Not yet implemented.
            //
        }

        Locks[VertexIndex].Unlock();
    }

End:
    LabeledGroup.sync();
    //
    // Don't propagate the 'Removed' value back to other threads.
    //
    //Removed = LabeledGroup.shfl(Removed, 0);
    return Retry;
}

template<typename GraphType>
DEVICE
VOID
GraphCuRemoveVertexOld(
    _In_ GraphType* Graph,
    _In_ typename GraphType::VertexType VertexIndex,
    _In_ PLOCK Locks
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

    EdgeType Edge;
    DegreeType Degree;
    Edge3Type *Edge3;
    Edge3Type *Edges3 = (decltype(Edges3))Graph->Edges3;
    OrderType OrderIndex;
    Vertex3Type *Vertex1;
    Vertex3Type *Vertex2;

    Vertex3Type *Vertex;
    Vertex3Type *Vertices3;

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;
    Vertex = &Vertices3[VertexIndex];

    if (!Locks[VertexIndex].TryLock()) {
        printf("Failed to acquire lock for VertexIndex: %d\n", VertexIndex);
        return;
    }

    Degree = Vertex->Degree;
    Locks[VertexIndex].Unlock();
    if (Degree != 1) {
        return;
    }

    Edge = Vertex->Edges;
    Edge3 = &Edges3[Edge];

    if (IsEmpty(Edge3->Vertex1)) {
#if 0
        if (!IsEmpty(Edge3->Vertex2)) {
            printf("A VertexIndex: %d\n", VertexIndex);
            printf("Edge3->Vertex1: %x\n", Edge3->Vertex1);
            printf("Edge3->Vertex2: %x\n", Edge3->Vertex2);
        }
#endif
        ASSERT(IsEmpty(Edge3->Vertex2));
        return;
    } else if (IsEmpty(Edge3->Vertex2)) {
#if 0
        if (!IsEmpty(Edge3->Vertex1)) {
            printf("B VertexIndex: %d\n", VertexIndex);
            printf("Edge3->Vertex2: %x\n", Edge3->Vertex2);
            printf("Edge3->Vertex1: %x\n", Edge3->Vertex1);
        }
#endif
        ASSERT(IsEmpty(Edge3->Vertex1));
        return;
    }

    cg::coalesced_group Group1 = cg::coalesced_threads();
    auto Vertex1Group = cg::labeled_partition(Group1, Edge3->Vertex1);
    auto CumulativeXOR1 = cg::reduce(Vertex1Group,
                                     Edge,
                                     cg::bit_xor<EdgeType>());

    if (Vertex1Group.thread_rank() == 0) {
        Vertex1 = (decltype(Vertex1))&Vertices3[Edge3->Vertex1];
        if (Vertex1->Degree >= 1) {
            Locks[Edge3->Vertex1].TryLock();
            if (Vertex1->Degree >= 1) {
                Vertex1->Edges ^= CumulativeXOR1;
                Vertex1->Degree -= Vertex1Group.num_threads();
#if 0
                auto NextDegree = (intptr_t)Vertex1->Degree - Vertex1Group.num_threads();
                if (NextDegree < 0) {
                    printf("Vertex1->Degree: %d\n", Vertex1->Degree);
                    printf("Vertex1Group.num_threads(): %d\n", Vertex1Group.num_threads());
                    printf("NextDegree: %d\n", NextDegree);
                    Vertex1->Degree = 0;
                } else {
                    Vertex1->Degree -= Vertex1Group.num_threads();
                }
#endif
            }
            Locks[Edge3->Vertex1].Unlock();
        }
    }
    Vertex1Group.sync();

    cg::coalesced_group Group2 = cg::coalesced_threads();
    auto Vertex2Group = cg::labeled_partition(Group1, Edge3->Vertex2);
    auto CumulativeXOR2 = cg::reduce(Vertex2Group,
                                     Edge,
                                     cg::bit_xor<EdgeType>());

    if (Vertex2Group.thread_rank() == 0) {
        Vertex2 = (decltype(Vertex2))&Vertices3[Edge3->Vertex2];
        if (Vertex2->Degree >= 1) {
            Locks[Edge3->Vertex2].TryLock();
            if (Vertex2->Degree >= 1) {
                Vertex2->Edges ^= CumulativeXOR2;
                Vertex2->Degree -= Vertex2Group.num_threads();
#if 0
                auto NextDegree = (intptr_t)Vertex2->Degree - Vertex2Group.num_threads();
                if (NextDegree < 0) {
                    printf("Vertex2->Degree: %d\n", Vertex2->Degree);
                    printf("Vertex2Group.num_threads(): %d\n", Vertex2Group.num_threads());
                    printf("NextDegree: %d\n", NextDegree);
                    Vertex2->Degree = 0;
                } else {
                    Vertex2->Degree -= Vertex2Group.num_threads();
                }
#endif
            }
            Locks[Edge3->Vertex2].Unlock();
        }
    }
    Vertex2Group.sync();

    AtomicAggIncCGV(&Graph->DeletedEdgeCount);
    OrderIndex = AtomicAggSubCG(&Graph->OrderIndex);
#ifdef _DEBUG
    ASSERT(Graph->DeletedEdgeCount <= Graph->NumberOfEdges);
    ASSERT(OrderIndex < Graph->NumberOfEdges);
#endif
    Graph->Order[OrderIndex] = Edge;
}

template<typename GraphType>
DEVICE
bool
GraphCuRemoveVertex(
    _In_ GraphType* Graph,
    _In_ typename GraphType::VertexType VertexIndex,
    _In_ PLOCK Locks
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
    DegreeType Degree;
    Edge3Type *Edge3;
    Edge3Type *Edges3 = (decltype(Edges3))Graph->Edges3;
    OrderType *Order = (decltype(Order))Graph->Order;
    OrderType *SortedOrder = (decltype(Order))Graph->SortedOrder;
    OrderType *OrderAddress;
    OrderIndexType *GraphOrderIndex = (decltype(GraphOrderIndex))&Graph->OrderIndex;
    OrderIndexType OrderIndex;

    Vertex3Type *Vertex;
    Vertex3Type *Vertices3;

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;
    Vertex = &Vertices3[VertexIndex];

    if (Vertex->Degree == 0) {
        goto End;
    }

    if (!Locks[VertexIndex].TryLock()) {
        Retry = true;
        goto End;
    }
    Degree = Vertex->Degree;
    Locks[VertexIndex].Unlock();
    if (Degree != 1) {
        goto End;
    }

    Edge = Vertex->Edges;
    Edge3 = &Edges3[Edge];

#if 1
    if (IsEmpty(Edge3->Vertex1)) {
#if 0
        if (!IsEmpty(Edge3->Vertex2)) {
            printf("A VertexIndex: %d\n", VertexIndex);
            printf("Edge3->Vertex1: %x\n", Edge3->Vertex1);
            printf("Edge3->Vertex2: %x\n", Edge3->Vertex2);
        }
#endif
        ASSERT(IsEmpty(Edge3->Vertex2));
        goto End;
    } else if (IsEmpty(Edge3->Vertex2)) {
#if 0
        if (!IsEmpty(Edge3->Vertex1)) {
            printf("B VertexIndex: %d\n", VertexIndex);
            printf("Edge3->Vertex2: %x\n", Edge3->Vertex2);
            printf("Edge3->Vertex1: %x\n", Edge3->Vertex1);
        }
#endif
        ASSERT(IsEmpty(Edge3->Vertex1));
        goto End;
    }
#endif

    do {
        if (Retry1) {
            Retry1 = GraphCuRemoveEdgeVertex(Graph,
                                             Edge3->Vertex1,
                                             Edge,
                                             Locks,
                                             Removed1);
        }
        if (Retry2) {
            Retry2 = GraphCuRemoveEdgeVertex(Graph,
                                             Edge3->Vertex2,
                                             Edge,
                                             Locks,
                                             Removed2);
        }
    } while (Retry1 || Retry2);

#if 0
    AtomicAggIncCGV(&Graph->DeletedEdgeCount);
    OrderIndex = AtomicAggSubCG(&Graph->OrderIndex);
    Graph->Order[OrderIndex] = Edge;
    Graph->OrderByVertex[GlobalThreadIndex()] = Edge;
#else
#ifdef _DEBUG
    //ASSERT(Graph->DeletedEdgeCount <= Graph->NumberOfEdges);
    //ASSERT(OrderIndex < Graph->NumberOfEdges);
#endif

    if (Removed1 || Removed2) {
        AtomicAggIncCGV(&Graph->OrderIndexEither);
    }
    if (Removed1 && Removed2) {
        AtomicAggIncCGV(&Graph->OrderIndexBoth);
    }
    if (Removed1) {
        AtomicAggIncCGV(&Graph->OrderIndex1);
    }
    if (Removed2) {
        AtomicAggIncCGV(&Graph->OrderIndex2);
    }
    if (!Removed1 && !Removed2) {
        AtomicAggIncCGV(&Graph->OrderIndexNone);
    }

    if (Removed1 || Removed2) {
        AtomicAggIncCGV(&Graph->DeletedEdgeCount);
        OrderIndex = AtomicAggSubCG(GraphOrderIndex);
        OrderAddress = &Order[OrderIndex];
#if 0
        if (OrderIndex < 0) {
            ASSERT(false);
        }
#endif
        if (OrderIndex >= 0) {
            ASSERT(*OrderAddress == 0);
            ASSERT(Order[OrderIndex] == 0);
            Order[OrderIndex] = Edge;
            ASSERT(Order[OrderIndex] == Edge);
            ASSERT(*OrderAddress == Edge);
            SortedOrder[OrderIndex] = Edge;
        }
        //Graph->OrderByThreadOrder[ThreadOrderIndex] = Edge;
    }

#if 0
    Graph->OrderByThread[GlobalThreadIndex()] = Edge;
    Graph->OrderByVertex[VertexIndex] = GlobalThreadIndex();
    Graph->OrderByEdge[Edge] = GlobalThreadIndex();
    Graph->OrderByThreadOrder[OrderIndex] = GlobalThreadIndex();
#endif
    //__threadfence();
#endif
End:
    return Retry;
}

template<typename GraphType>
DEVICE
bool
GraphCuRemoveVertexNew(
    _In_ GraphType* Graph,
    _In_ typename GraphType::VertexType VertexIndex,
    _In_ PLOCK Locks
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
    EdgeType Edge;
    DegreeType Degree;
    Edge3Type *Edge3;
    Edge3Type *Edges3 = (decltype(Edges3))Graph->Edges3;
    OrderType *Order = (decltype(Order))Graph->Order;
    OrderType *OrderAddress;
    OrderIndexType *GraphOrderIndex = (decltype(GraphOrderIndex))&Graph->OrderIndex;
    OrderIndexType OrderIndex;
    PLOCK SecondLock = nullptr;

    Vertex3Type *Vertex;
    Vertex3Type *SecondVertex;
    Vertex3Type *Vertices3;

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;
    Vertex = &Vertices3[VertexIndex];

    //__threadfence();

    if (Vertex->Degree == 0) {
        return Retry;
    }

    Locks[VertexIndex].Lock();
    Degree = Vertex->Degree;
    if (Degree != 1) {
        Locks[VertexIndex].Unlock();
        return Retry;
    }

    Edge = Vertex->Edges;
    Edge3 = &Edges3[Edge];

    ASSERT(
        (Edge3->Vertex1 == VertexIndex) ||
        (Edge3->Vertex2 == VertexIndex)
    );

    if (Edge3->Vertex1 == VertexIndex) {
        SecondVertex = &Vertices3[Edge3->Vertex2];
        ASSERT(SecondVertex != nullptr);
        SecondLock = &Locks[Edge3->Vertex2];
    } else {
        SecondVertex = &Vertices3[Edge3->Vertex1];
        ASSERT(SecondVertex != nullptr);
        SecondLock = &Locks[Edge3->Vertex1];
    }

    GraphCuRemoveEdge(Graph, Vertex, Edge);

    Locks[VertexIndex].Unlock();

    //__threadfence();

    if (SecondVertex->Degree == 0) {
        return Retry;
    }

    SecondLock->Lock();
    if (SecondVertex->Degree > 0) {
        GraphCuRemoveEdge(Graph, SecondVertex, Edge);
    }
    SecondLock->Unlock();

    cg::coalesced_group Group = cg::coalesced_threads();
    auto Size = Group.num_threads();
    OrderIndexType Prev;
    OrderIndexType Next;
    if (Group.thread_rank() == 0) {
        OrderIndexType *Address = GraphOrderIndex;
        do {

            Prev = *Address;
            ASSERT(Prev >= 0);

            if (Prev == 0) {
                Next = -1;
                break;
            } else if (Prev >= Size) {
                Next = Prev - Size;
            } else {
                Next = 0;
            }
            ASSERT(Next >= 0);
        } while (atomicCAS(Address, Prev, Next) != Prev);
    }
    Group.sync();
    Next = Group.shfl(Next, 0);
    if (Next != -1) {
        OrderIndex = Group.shfl(Prev, 0) - Group.thread_rank();
        if (OrderIndex >= 0) {
            OrderAddress = &Order[OrderIndex];
            ASSERT(OrderIndex >= 0);
            ASSERT(*OrderAddress == 0);
            *OrderAddress = Edge;
        }
    }

End:

    return Retry;
}

template<typename GraphType>
DEVICE
VOID
GraphCuRemoveVertexUnsafe(
    _In_ GraphType* Graph,
    _In_ typename GraphType::VertexType VertexIndex
    )
{
    using EdgeType = typename GraphType::EdgeType;
    using Edge3Type = typename GraphType::Edge3Type;
    using OrderType = typename GraphType::OrderType;
    using VertexType = typename GraphType::VertexType;
    using Vertex3Type = typename GraphType::Vertex3Type;
    using VertexPairType = typename GraphType::VertexPairType;
    using AtomicVertex3Type = typename GraphType::AtomicVertex3Type;

    EdgeType Edge;
    Edge3Type *Edge3;
    Edge3Type *Edges3 = (decltype(Edges3))Graph->Edges3;
    OrderType OrderIndex;
    Vertex3Type *Vertex1;
    Vertex3Type *Vertex2;

    Vertex3Type *Vertex;
    Vertex3Type *Vertices3;

    Vertices3 = (decltype(Vertices3))Graph->Vertices3;
    Vertex = &Vertices3[VertexIndex];

    if (Vertex->Degree != 1) {
        return;
    }

    Edge = Vertex->Edges;
    Edge3 = &Edges3[Edge];

    if (IsEmpty(Edge3->Vertex1)) {
        ASSERT(IsEmpty(Edge3->Vertex2));
        return;
    } else if (IsEmpty(Edge3->Vertex2)) {
        ASSERT(IsEmpty(Edge3->Vertex1));
        return;
    }

    Vertex1 = &Vertices3[Edge3->Vertex1];
    if (Vertex1->Degree >= 1) {
        Vertex1->Edges ^= Edge;
        --Vertex1->Degree;
    }

    Vertex2 = &Vertices3[Edge3->Vertex2];
    if (Vertex2->Degree >= 1) {
        Vertex2->Edges ^= Edge;
        --Vertex2->Degree;
    }

    Graph->DeletedEdgeCount++;
    OrderIndex = --Graph->OrderIndex;
#ifdef _DEBUG
    ASSERT(Graph->DeletedEdgeCount <= Graph->NumberOfEdges);
    ASSERT(OrderIndex < Graph->NumberOfEdges);
#endif
    Graph->Order[OrderIndex] = Edge;
}

template<typename GraphType>
GLOBAL
VOID
GraphCuIsAcyclicPhase1x4(
    GraphType* Graph,
    PLOCK Locks
    )
{
    using VertexType = typename GraphType::VertexType;

    int32_t Index1;
    int32_t Index2;
    int32_t Index3;
    int32_t Index4;

    const int32_t Stride = gridDim.x * blockDim.x;
    const uint32_t NumberOfVertices = Graph->NumberOfVertices;

    Index1 = GlobalThreadIndex();

    Index2 = Index1 + Stride;
    Index3 = Index1 + Stride + Stride;
    Index4 = Index1 + Stride + Stride + Stride;

    uint8_t LocalRetries;
    uint32_t ThreadRetries = 0;

    while (Index1 < NumberOfVertices) {
        bool Retry1 = true;
        bool Retry2 = (Index2 < NumberOfVertices);
        bool Retry3 = (Index3 < NumberOfVertices);
        bool Retry4 = (Index4 < NumberOfVertices);
        LocalRetries = 0;

        do {
            if (Retry1) {
                Retry1 = GraphCuRemoveVertex(Graph,
                                             (VertexType)Index1,
                                             Locks);
            }

            if (Retry2) {
                Retry2 = GraphCuRemoveVertex(Graph,
                                             (VertexType)Index2,
                                             Locks);
            }

            if (Retry3) {
                Retry3 = GraphCuRemoveVertex(Graph,
                                             (VertexType)Index3,
                                             Locks);
            }

            if (Retry4) {
                Retry4 = GraphCuRemoveVertex(Graph,
                                             (VertexType)Index4,
                                             Locks);
            }

            if (Retry1) {
                LocalRetries++;
            }
            if (Retry2) {
                LocalRetries++;
            }
            if (Retry3) {
                LocalRetries++;
            }
            if (Retry4) {
                LocalRetries++;
            }
            if (LocalRetries) {
                ThreadRetries += LocalRetries;
            }

        } while (Retry1 || Retry2 || Retry3 || Retry4);

        Index1 = Index4 + Stride;
        Index2 = Index4 + Stride + Stride;
        Index3 = Index4 + Stride + Stride + Stride;
        Index4 = Index4 + Stride + Stride + Stride + Stride;
    }

    return;
}

template<typename GraphType>
GLOBAL
VOID
GraphCuIsAcyclicPhase1(
    GraphType* Graph,
    PLOCK Locks
    )
{
    using VertexType = typename GraphType::VertexType;

    int32_t Index;

    const int32_t Stride = gridDim.x * blockDim.x;
    const uint32_t NumberOfVertices = Graph->NumberOfVertices;

    Index = GlobalThreadIndex();

    uint8_t LocalRetries;
    uint32_t ThreadRetries = 0;

    while (Index < NumberOfVertices) {
        bool Retry1 = true;
        LocalRetries = 0;

        do {
            if (Retry1) {
                Retry1 = GraphCuRemoveVertex(Graph,
                                             (VertexType)Index,
                                             Locks);
            }

            if (Retry1) {
                LocalRetries++;
            }
            if (LocalRetries) {
                ThreadRetries += LocalRetries;
            }

        } while (Retry1);

        Index += Stride;
    }

    return;
}

template<typename GraphType>
GLOBAL
VOID
GraphCuIsAcyclicPhase1Old(
    GraphType* Graph,
    PLOCK Locks
    )
{
    using VertexType = typename GraphType::VertexType;

    int32_t Index;
    uint32_t NumberOfVertices = Graph->NumberOfVertices;
    VertexType Vertex;

    Index = GlobalThreadIndex();

    bool Retry;

    while (Index < NumberOfVertices) {
        Vertex = (VertexType)Index;
        Retry = true;
        do {
            Retry = GraphCuRemoveVertex(Graph, Vertex, Locks);
        } while (Retry);
        Index += gridDim.x * blockDim.x;
    }

    return;
}

template <typename T>
struct IS_NOT_EMPTY {
    __host__ __device__
    bool operator()(const T &Value) const {
        return (Value != static_cast<T>(-1));
    }
};

EXTERN_C
HOST
SIZE_T
CountNonEmptyHost(
    PGRAPH Graph,
    PLONG Values
    )
{
    using ValueType = std::remove_reference_t<decltype(Values[0])>;

    SIZE_T Count;

    thrust::device_ptr<ValueType> ValuesPtr(Values);
    Count = thrust::count_if(ValuesPtr,
                             ValuesPtr + Graph->NumberOfVertices,
                             IS_NOT_EMPTY<ValueType>());
    return Count;
}



template<typename GraphType>
GLOBAL
VOID
GraphCuIsAcyclicPhase2(
    GraphType* Graph,
    PLOCK Locks
    )
{
    using EdgeType = typename GraphType::EdgeType;
    using Edge3Type = typename GraphType::Edge3Type;
    using OrderType = typename GraphType::OrderType;
    using VertexType = typename GraphType::VertexType;

    bool Retry1;
    bool Retry2;
    int32_t Index;
    uint32_t NumberOfKeys = Graph->NumberOfKeys;
    OrderType *Order = (decltype(Order))Graph->Order;
    Edge3Type *Edges3 = (decltype(Edges3))Graph->Edges3;
    OrderType EdgeIndex;
    Edge3Type *OtherEdge;

    const int32_t Stride = gridDim.x * blockDim.x;

    //auto Grid = cg::this_grid();
    uint8_t LocalRetries;
    uint32_t ThreadRetries = 0;

    for (Index = NumberOfKeys - GlobalThreadIndex();
         Graph->OrderIndex > 0 &&
         Index > (Graph->OrderIndex <= 0 ? 0 : Graph->OrderIndex);
         NOTHING)
    {
        Retry1 = true;
        Retry2 = true;
        LocalRetries = 0;

        EdgeIndex = Order[Index];
        OtherEdge = &Edges3[EdgeIndex];
        do {
            if (Retry1) {
                Retry1 = GraphCuRemoveVertex(Graph,
                                             OtherEdge->Vertex1,
                                             Locks);
            }
            if (Retry2) {
                Retry2 = GraphCuRemoveVertex(Graph,
                                             OtherEdge->Vertex2,
                                             Locks);
            }
            if (Retry1) {
                LocalRetries++;
            }
            if (Retry2) {
                LocalRetries++;
            }
            if (LocalRetries) {
                ThreadRetries += LocalRetries;
            }
        } while (Retry1 || Retry2);
        Index -= Stride;
    }

#if 0
    if (GlobalThreadIndex() == 0) {
        for (Index = (int32_t)NumberOfKeys;
             Graph->OrderIndex > 0 && Index > Graph->OrderIndex;
             NOTHING)
        {
            EdgeIndex = Order[--Index];
            OtherEdge = &Edges3[EdgeIndex];
            GraphCuRemoveVertex(Graph, OtherEdge->Vertex1, Locks);
            GraphCuRemoveVertex(Graph, OtherEdge->Vertex2, Locks);
        }

        ASSERT(Graph->OrderIndex >= 0);
        NumberOfEdgesDeleted = Graph->DeletedEdgeCount;
        IsAcyclic = (NumberOfEdgesDeleted == NumberOfKeys);
        if (IsAcyclic) {
            ASSERT(Graph->OrderIndex == 0);
            Graph->CuIsAcyclicResult = S_OK;
        } else {
            ASSERT(Graph->OrderIndex > 0);
            Graph->CuIsAcyclicResult = PH_E_GRAPH_CYCLIC_FAILURE;
        }
    }
#elif 0

    while (Index < NumberOfKeys) {
        EdgeIndex = Order[Index];
        OtherEdge = &Edges3[EdgeIndex];
        GraphCuRemoveVertex(Graph, OtherEdge->Vertex1, Locks);
        GraphCuRemoveVertex(Graph, OtherEdge->Vertex2, Locks);
        Index += gridDim.x * blockDim.x;
    }

    __syncthreads();

    if (GlobalThreadIndex() == 0) {
        ASSERT(Graph->OrderIndex >= 0);
        NumberOfEdgesDeleted = Graph->DeletedEdgeCount;
        IsAcyclic = (NumberOfEdgesDeleted == NumberOfKeys);
        if (IsAcyclic) {
            ASSERT(Graph->OrderIndex == 0);
            Graph->CuIsAcyclicResult = S_OK;
        } else {
            ASSERT(Graph->OrderIndex > 0);
            Graph->CuIsAcyclicResult = PH_E_GRAPH_CYCLIC_FAILURE;
        }
    }
#endif

    return;
}

EXTERN_C
HOST
VOID
IsGraphAcyclicPhase1Host(
    _In_ PGRAPH Graph,
    _In_ ULONG BlocksPerGrid,
    _In_ ULONG ThreadsPerBlock,
    _In_ ULONG SharedMemoryInBytes
    )
{
    PGRAPH HostGraph;
    PLOCK Locks;
    PPH_CU_SOLVE_CONTEXT SolveContext;
    PVOID Kernel;

    SolveContext = Graph->CuSolveContext;

    printf("IsGraphAcyclicHost: Graph: %p\n", Graph);
    //Graph->OrderIndex = Graph->NumberOfKeys;
    printf("IsGraphAcyclicHost: Graph->OrderIndex: %d\n", Graph->OrderIndex);

    HostGraph = (PGRAPH)SolveContext->HostGraph;
    ASSERT(HostGraph != Graph);
    ASSERT(SolveContext->DeviceGraph == Graph);

    size_t AllocSize = sizeof(*Locks) * Graph->NumberOfVertices;

    if (HostGraph->VertexLocks == NULL) {
        printf("Allocating %Iu bytes for HostGraph->VertexLocks...\n",
               AllocSize);

        CUDA_CALL(cudaMalloc((void **)&Locks, AllocSize));
        printf("cudaMalloc: Locks: %p\n", Locks);
        HostGraph->VertexLocks = (PVOID)Locks;

        CUDA_CALL(cudaMemcpy(&Graph->VertexLocks,
                             HostGraph->VertexLocks,
                             sizeof(Graph->VertexLocks),
                             cudaMemcpyHostToDevice));

    } else {
        Locks = (PLOCK)HostGraph->VertexLocks;
    }

    CUDA_CALL(cudaMemset(Locks, -1, AllocSize));

    if (IsUsingAssigned16(Graph)) {
        Kernel = (PVOID)GraphCuIsAcyclicPhase1<GRAPH16>;
    } else {
        Kernel = (PVOID)GraphCuIsAcyclicPhase1<GRAPH32>;
    }

    printf("Kernel: %p\n", Kernel);
    printf("Graph: %p\n", Graph);
    printf("Locks: %p\n", Locks);

#if PH_WINDOWS
    int NumberOfBlocksPerSm;
    cudaDeviceProp DeviceProperties;
    CUDA_CALL(cudaGetDeviceProperties(&DeviceProperties, 0));
    CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &NumberOfBlocksPerSm,
        Kernel,
        (int)ThreadsPerBlock,
        (size_t)SharedMemoryInBytes
    ));

    ULONG LocalBlocksPerGrid = (
        DeviceProperties.multiProcessorCount * NumberOfBlocksPerSm
    );
    BlocksPerGrid = min(BlocksPerGrid, LocalBlocksPerGrid);

    printf("DeviceProperties.multiProcessorCount: %d\n",
           DeviceProperties.multiProcessorCount);
    printf("NumberOfBlocksPerSm: %d\n", NumberOfBlocksPerSm);
#endif
    printf("BlocksPerGrid: %d\n", BlocksPerGrid);
    printf("ThreadsPerBlock: %d\n", ThreadsPerBlock);

    dim3 GridDim(BlocksPerGrid, 1, 1);
    dim3 BlockDim(ThreadsPerBlock, 1, 1);
    //PVOID KernelArgs[] = { Graph, Locks };

    if (IsUsingAssigned16(Graph)) {
        GraphCuIsAcyclicPhase1<GRAPH16><<<GridDim,
                                          BlockDim,
                                          SharedMemoryInBytes,
                                          (CUstream_st *)SolveContext->Stream>>>(
            (PGRAPH16)Graph,
            Locks
        );
    } else {
        GraphCuIsAcyclicPhase1<GRAPH32><<<GridDim,
                                          BlockDim,
                                          SharedMemoryInBytes,
                                          (CUstream_st *)SolveContext->Stream>>>(
            (PGRAPH32)Graph,
            Locks
        );
    }

#if 0
        CUDA_CALL(cudaLaunchCooperativeKernel(
            (void *)GraphCuIsAcyclic<GRAPH16>,
            GridDim,
            BlockDim,
            KernelArgs,
            SharedMemoryInBytes,
            (CUstream_st *)SolveContext->Stream
        ));
    } else {
        CUDA_CALL(cudaLaunchCooperativeKernel(
            (void *)GraphCuIsAcyclic<GRAPH32>,
            GridDim,
            BlockDim,
            KernelArgs,
            SharedMemoryInBytes,
            (CUstream_st *)SolveContext->Stream
        ));
    }
#endif

    //cudaStreamSynchronize((CUstream_st *)SolveContext->Stream);

    //CUDA_CALL(cudaFree(Locks));
    return;
}

EXTERN_C
HOST
VOID
IsGraphAcyclicPhase2Host(
    _In_ PGRAPH Graph,
    _In_ ULONG BlocksPerGrid,
    _In_ ULONG ThreadsPerBlock,
    _In_ ULONG SharedMemoryInBytes
    )
{
    PGRAPH HostGraph;
    PLOCK Locks;
    PPH_CU_SOLVE_CONTEXT SolveContext;

    SolveContext = Graph->CuSolveContext;

    HostGraph = (PGRAPH)SolveContext->HostGraph;
    ASSERT(HostGraph != Graph);
    ASSERT(SolveContext->DeviceGraph == Graph);

    size_t AllocSize = sizeof(*Locks) * Graph->NumberOfVertices;

    Locks = (PLOCK)HostGraph->VertexLocks;
    ASSERT(Locks != NULL);

    CUDA_CALL(cudaMemset(Locks, -1, AllocSize));

    if (IsUsingAssigned16(Graph)) {
        GraphCuIsAcyclicPhase2<GRAPH16><<<BlocksPerGrid,
                                          ThreadsPerBlock,
                                          SharedMemoryInBytes,
                                          (CUstream_st *)SolveContext->Stream>>>(
            (PGRAPH16)Graph,
            Locks
        );
    } else {
        GraphCuIsAcyclicPhase2<GRAPH32><<<BlocksPerGrid,
                                          ThreadsPerBlock,
                                          SharedMemoryInBytes,
                                          (CUstream_st *)SolveContext->Stream>>>(
            (PGRAPH32)Graph,
            Locks
        );

    }

#if 0
        CUDA_CALL(cudaLaunchCooperativeKernel(
            (void *)GraphCuIsAcyclic<GRAPH16>,
            GridDim,
            BlockDim,
            KernelArgs,
            SharedMemoryInBytes,
            (CUstream_st *)SolveContext->Stream
        ));
    } else {
        CUDA_CALL(cudaLaunchCooperativeKernel(
            (void *)GraphCuIsAcyclic<GRAPH32>,
            GridDim,
            BlockDim,
            KernelArgs,
            SharedMemoryInBytes,
            (CUstream_st *)SolveContext->Stream
        ));
    }
#endif

    //cudaStreamSynchronize((CUstream_st *)SolveContext->Stream);

    //CUDA_CALL(cudaFree(Locks));
    return;
}

//
// VertexPair/IsAcyclic Logic.
//

template<typename VertexPairType>
DEVICE
bool
VertexPairNotEqual(
    const VertexPairType Left,
    const VertexPairType Right
    )
{
    return (
        Left.Vertex1 != Right.Vertex1 &&
        Left.Vertex2 != Right.Vertex2
    );
    //return (Left.AsULongLong != Right.AsULongLong);
}

template<typename VertexPairType>
DEVICE
bool
VertexPairEqual(
    const VertexPairType Left,
    const VertexPairType Right
    )
{
    return (
        Left.Vertex1 == Right.Vertex1 &&
        Left.Vertex2 == Right.Vertex2
    );
    //return (Left.AsULongLong == Right.AsULongLong);
}

template<typename GraphType>
HOST
VOID
GraphCuIsAcyclicOld(
    GraphType* Graph
    )
{
    //uint32_t UniqueCount;
    typename GraphType::VertexPairType *VertexPairs;
    typename GraphType::VertexPairType *VertexPairsEnd;
    typename GraphType::VertexPairType *VertexPairsUnique;
    uint32_t NumberOfKeys = Graph->NumberOfKeys;

    Graph->CuIsAcyclicResult = S_OK;


    VertexPairs = (typename GraphType::VertexPairType *)Graph->VertexPairs;
    VertexPairsEnd = VertexPairs + NumberOfKeys;

    printf("111 Starting thrust::unique...\n");
    VertexPairsUnique = thrust::unique(
        thrust::device,
        VertexPairs,
        VertexPairsEnd
    );
    printf("111 Finished thrust::unique...\n");

    if (VertexPairsUnique != VertexPairsEnd) {
        printf("111 VertexPairsUnique != VertexPairsEnd!\n");
        Graph->CuIsAcyclicResult = PH_E_GRAPH_CYCLIC_FAILURE;
    } else {
        printf("111 VertexPairsUnique == VertexPairsEnd!\n");
    }

    //printf("VertexPairs[0].Vertex1: %d\n", VertexPairs[0].Vertex1);
    //printf("VertexPairs[0].Vertex2: %d\n", VertexPairs[0].Vertex2);
    //printf("VertexPairs[100].Vertex1: %d\n", VertexPairs[100].Vertex1);
    //printf("VertexPairs[100].Vertex2: %d\n", VertexPairs[100].Vertex2);
    //printf("VertexPairsEnd[0].Vertex1: %d\n", VertexPairsEnd[0].Vertex1);
    //printf("VertexPairsEnd[0].Vertex2: %d\n", VertexPairsEnd[0].Vertex2);
    //printf("Number of keys: %d\n", NumberOfKeys);

    printf("Starting thrust::sort...\n");
    thrust::sort(thrust::device, VertexPairs, VertexPairsEnd);
    printf("Finished thrust sort...\n");
    //printf("VertexPairs[0].Vertex1: %d\n", VertexPairs[0].Vertex1);
    //printf("VertexPairs[0].Vertex2: %d\n", VertexPairs[0].Vertex2);
    //printf("VertexPairs[100].Vertex1: %d\n", VertexPairs[100].Vertex1);
    //printf("VertexPairs[100].Vertex2: %d\n", VertexPairs[100].Vertex2);
    //printf("VertexPairsEnd[0].Vertex1: %d\n", VertexPairsEnd[0].Vertex1);
    //printf("VertexPairsEnd[0].Vertex2: %d\n", VertexPairsEnd[0].Vertex2);

#if 0
    for (uint32_t i = 0; i < NumberOfKeys; i++) {
        printf("%d,%d\n",
               VertexPairs[i].Vertex1,
               VertexPairs[i].Vertex2);
    }
#endif

    printf("Starting thrust::unique...\n");
    VertexPairsUnique = thrust::unique(
        thrust::device,
        VertexPairs,
        VertexPairsEnd
    );
    printf("Finished thrust::unique...\n");

    if (VertexPairsUnique != VertexPairsEnd) {
        printf("VertexPairsUnique != VertexPairsEnd!\n");
        Graph->CuIsAcyclicResult = PH_E_GRAPH_CYCLIC_FAILURE;
    } else {
        printf("VertexPairsUnique == VertexPairsEnd!\n");
    }

#if 0
    UniqueCount = thrust::inner_product(
        thrust::device,
        VertexPairs,
        VertexPairsEnd - 1,
        VertexPairs + 1,
        ULONG(1),
        thrust::plus<ULONG>(),
        VertexPairNotEqual<typename GraphType::VertexPairType>
    );

    printf("VertexPair unique count: %d, num keys: %d\n",
           UniqueCount, NumberOfKeys);
#endif

    auto diff = thrust::distance(VertexPairs, VertexPairsEnd);
    printf("diff: %td\n", diff);

}

EXTERN_C
HOST
VOID
IsGraphAcyclicOldHost(
    _In_ PGRAPH Graph
    )
{
    if (IsUsingAssigned16(Graph)) {
        GraphCuIsAcyclicOld<GRAPH16>((PGRAPH16)Graph);
    } else {
        GraphCuIsAcyclicOld<GRAPH32>((PGRAPH32)Graph);
    }
    return;
}

//
// PostAddHashedKeys
//

template <typename T>
struct VERTEX3_LESS_THAN {
    __host__ __device__
    bool operator()(const T &Left, const T &Right) const {
        return (Left.Degree < Right.Degree);
    }
};

template <typename T>
struct VERTEX3_GREATER_THAN {
    __host__ __device__
    bool operator()(const T &Left, const T &Right) const {
        return (Left.Degree > Right.Degree);
    }
};

template <typename T>
struct VERTEX3_NON_EMPTY {
    __host__ __device__
    bool operator()(const T &Value) const {
        return (Value.Degree != 0);
    }
};


template<typename GraphType>
HOST
VOID
GraphCuPostAddHashedKeys(
    _In_ GraphType* Graph
    )
{
    using Vertex3Type = typename GraphType::Vertex3Type;

    PPH_CU_SOLVE_CONTEXT SolveContext;

    SolveContext = Graph->CuSolveContext;

    CUstream_st *Stream = (CUstream_st *)SolveContext->Stream2;

    thrust::sequence(thrust::cuda::par.on(Stream),
                     Graph->SortedVertices3Indices,
                     Graph->SortedVertices3Indices + Graph->NumberOfVertices);

    // Step 1: Copy non-zero degree vertices to SortedVertices3
    auto End = thrust::copy_if(
        thrust::cuda::par.on(Stream),
        (Vertex3Type *)Graph->Vertices3,
        (Vertex3Type *)(Graph->Vertices3 + Graph->NumberOfVertices),
        (Vertex3Type *)Graph->SortedVertices3,
        VERTEX3_NON_EMPTY<Vertex3Type>()
    );

    Graph->NumberOfNonZeroVertices = (
        thrust::distance(
            (Vertex3Type *)Graph->SortedVertices3,
            End
        )
    );

    // Step 2: Fill SortedVertices3Indices with original indices of non-zero degree vertices
    thrust::transform_if(
        thrust::cuda::par.on(Stream),
        thrust::counting_iterator<int32_t>(0),
        thrust::counting_iterator<int32_t>(Graph->NumberOfVertices),
        Graph->SortedVertices3Indices,
        [] __device__ (int Index) { return Index; },
        [=] __device__ (int Index) {
            return VERTEX3_NON_EMPTY<Vertex3Type>()(
                ((Vertex3Type *)Graph->Vertices3)[Index]
            );
        }
    );

    // Step 3: Sort SortedVertices3 and also rearrange SortedVertices3Indices accordingly
    thrust::stable_sort_by_key(
        thrust::cuda::par.on(Stream),
        (Vertex3Type *)Graph->SortedVertices3,
        (Vertex3Type *)(Graph->SortedVertices3 + Graph->NumberOfNonZeroVertices),
        Graph->SortedVertices3Indices,
        VERTEX3_GREATER_THAN<Vertex3Type>()
    );

}

EXTERN_C
HOST
VOID
GraphPostAddHashedKeysHost(
    _In_ PGRAPH Graph,
    _In_ ULONG BlocksPerGrid,
    _In_ ULONG ThreadsPerBlock,
    _In_ ULONG SharedMemoryInBytes
    )
{
    if (IsUsingAssigned16(Graph)) {
        GraphCuPostAddHashedKeys<GRAPH16>((PGRAPH16)Graph);
    } else {
        GraphCuPostAddHashedKeys<GRAPH32>((PGRAPH32)Graph);
    }

#if 0
    if (IsUsingAssigned16(Graph)) {
        GraphCuPostAddHashedKeys<GRAPH16><<<BlocksPerGrid,
                                            ThreadsPerBlock,
                                            SharedMemoryInBytes,
                                            Stream>>>((PGRAPH16)Graph);
    } else {
        GraphCuPostAddHashedKeys<GRAPH32><<<BlocksPerGrid,
                                            ThreadsPerBlock,
                                            SharedMemoryInBytes,
                                            Stream>>>((PGRAPH32)Graph);
    }
#endif

    return;
}

bool g_WriteDisabled = false;

std::mutex g_WriteMutex{};

EXTERN_C
HOST
bool
WriteDataToFileSync(
    const char* FilenameCStr,
    const char* DirectoryCStr,
    const char* Buffer,
    size_t NumberOfBytesytes)
{
    const std::string Filename(FilenameCStr);
    const std::string Directory(DirectoryCStr);

    bool Success = false;

    if (g_WriteDisabled) {
        return true;
    }

    std::string Path = Directory + "/" + Filename;

    // Open file for writing in binary mode
    std::ofstream File(Path, std::ios::binary);

    std::cout << "Writing " << NumberOfBytesytes << " bytes to file " <<
        Path << "." << std::endl;

    // Check if file was opened successfully
    if (!File.is_open()) {
        std::unique_lock<std::mutex> lock(g_WriteMutex);
        std::cerr << "\nFailed to open file " << Path <<
            " for writing." << std::endl;
        return false;
    }

    // Write data from Buffer to file
    File.write(Buffer, NumberOfBytesytes);

    // Check if write was successful
    if (File.bad()) {
        std::unique_lock<std::mutex> lock(g_WriteMutex);
        std::cerr << "\nFailed to write data to file " << Path <<
            "." << std::endl;

        std::cerr << "Error code: " << strerror(errno) << std::endl;
    } else {
        Success = true;
    }

    // Close file
    File.close();

    return Success;
}

EXTERN_C
HOST
void
WriteDataToFileAsync(
    const char* FilenameCStr,
    const char* DirectoryCStr,
    const char* Buffer,
    size_t NumberOfBytesytes)
{
    std::thread Thread(WriteDataToFileSync,
                       FilenameCStr,
                       DirectoryCStr,
                       Buffer,
                       NumberOfBytesytes);
    Thread.detach();
}

//
// Scratch
//

template<typename GraphType>
HOST
VOID
GraphCuSortOrder(
    _In_ GraphType* Graph
    )
{
    using OrderType = typename GraphType::OrderType;

    OrderType Start;
    OrderType *Begin;
    OrderType *End;
    OrderType *Unique;

    PPH_CU_SOLVE_CONTEXT SolveContext;

    SolveContext = Graph->CuSolveContext;

    CUstream_st *Stream = (CUstream_st *)SolveContext->Stream3;

    Start = (Graph->OrderIndex <= 0 ? 0 : Graph->OrderIndex);
    printf("Start: %d\n", Start);

    Begin = ((OrderType *)Graph->SortedOrder) + Start;
    printf("Begin: %p\n", Begin);
    End = Begin + (Graph->NumberOfKeys - Start);
    printf("End: %p\n", End);
    printf("*End: %d\n", *End);
    printf("*(End-1): %d\n", *(End-1));
    printf("*(End-2): %d\n", *(End-2));
    printf("Distance: %td\n", thrust::distance(Begin, End));
    printf("NumberOfKeys: %d\n", Graph->NumberOfKeys);

    //
    // Copy Graph->Order to Graph->SortedOrder using thrust.
    //

#if 0
    thrust::copy(thrust::cuda::par.on(Stream),
                 Begin,
                 End,
                 (OrderType *)Graph->Order);
#endif

    printf("sorting...\n");
    thrust::sort(thrust::cuda::par.on(Stream), Begin, End);
    printf("sorted!\n");

    printf("*End: %d\n", *End);
    printf("*(End-1): %d\n", *(End-1));
    printf("*(End-2): %d\n", *(End-2));

    Unique = thrust::unique(thrust::cuda::par.on(Stream), Begin, End);
    printf("Unique: %p\n", Unique);
    printf("*Unique: %d\n", *Unique);
    printf("*(Unique-1): %d\n", *(Unique-1));
    printf("*(Unique-2): %d\n", *(Unique-2));
    printf("*(Unique+1): %d\n", *(Unique+1));
    printf("*(Unique+2): %d\n", *(Unique+2));
    if (Unique != End) {
        printf("Unique != End! %p != %p\n", Unique, End);
    } else {
        printf("Unique == End!\n");
    }

    auto Diff = thrust::distance(Begin, End);
    printf("Diff1: %td\n", Diff);
    printf("Distance unique: %td\n", thrust::distance(Begin, Unique));
}


EXTERN_C
HOST
VOID
GraphScratchHost(
    _In_ PGRAPH Graph,
    _In_ ULONG BlocksPerGrid,
    _In_ ULONG ThreadsPerBlock,
    _In_ ULONG SharedMemoryInBytes
    )
{
    switch (Graph->CuScratch) {
        case 1: {
            if (IsUsingAssigned16(Graph)) {
                GraphCuSortOrder<GRAPH16>((PGRAPH16)Graph);
            } else {
                GraphCuSortOrder<GRAPH32>((PGRAPH32)Graph);
            }

            break;
        }
        default:
            break;
    }
}
// vim:set ts=8 sw=4 sts=4 tw=80 expandtab filetype=cuda formatoptions=croql   :
