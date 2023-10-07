/*++

Copyright (c) 2020-2023 Trent Nelson <trent@trent.me>

Module Name:

    Graph.cu

Abstract:

    CUDA graph implementation.

--*/

#define PH_CU

#include <PerfectHash.h>

EXTERN_C_BEGIN
#include "../PerfectHash/CuDeviceAttributes.h"
#include "../PerfectHash/stdafx.h"
#include "../PerfectHash/Graph.h"
EXTERN_C_END
#include "../PerfectHash/PerfectHashTableHashExCpp.hpp"

#include <cuda.h>
#include <cuda_device_runtime_api.h>

#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/inner_product.h>

#include "Graph.cuh"

#include <stdio.h>

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
    bool Collision;
    uint32_t Index;
    typename GraphType::KeyType Key;
    typename GraphType::KeyType *Keys;
    typename GraphType::VertexPairType *VertexPairs;
    typename GraphType::VertexPairType Hash;
    auto Mask = Graph->NumberOfVertices - 1;
    uint32_t NumberOfKeys = Graph->NumberOfKeys;
    PULONG Seeds = Graph->Seeds;

    auto HashFunction = GetHashFunctionForId<
        std::remove_reference_t<decltype(VertexPairs[0])>,
        std::remove_reference_t<decltype(Keys[0])>,
        decltype(Mask)
    >(Graph->HashFunctionId);

    VertexPairs = (typename GraphType::VertexPairType *)Graph->VertexPairs;
    Keys = (typename GraphType::KeyType *)Graph->DeviceKeys;

    Index = GlobalThreadIndex();

    Collision = false;

    while (Index < NumberOfKeys) {

        Key = Keys[Index];
        Hash = HashFunction(Key, Seeds, Mask);

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
    Graph->CuHashKeysResult = S_OK;
    if (GlobalThreadIndex() == 0) {
        printf("Graph->CuHashKeysResult: %d\n", Graph->CuHashKeysResult);
        printf("Graph->Seeds[0]: 0x%08x\n", Graph->Seeds[0]);
        printf("Graph->Seeds[1]: 0x%08x\n", Graph->Seeds[1]);
        printf("Graph->Seeds[2]: 0x%08x\n", Graph->Seeds[2]);
    }

    if (IsUsingAssigned16(Graph)) {
        GraphCuHashKeys<GRAPH16>((PGRAPH16)Graph);
    } else {
        GraphCuHashKeys<GRAPH32>((PGRAPH32)Graph);
    }
    if (GlobalThreadIndex() == 0) {
        printf("Graph->CuHashKeysResult: %d\n", Graph->CuHashKeysResult);
    }
    return;
}


EXTERN_C
GLOBAL
VOID
AddKeysToGraph(
    _In_ PGRAPH Graph
    )
{
    return;
}


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
GraphCuIsAcyclic(
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

    printf("VertexPairs[0].Vertex1: %d\n", VertexPairs[0].Vertex1);
    printf("VertexPairs[0].Vertex2: %d\n", VertexPairs[0].Vertex2);
    printf("VertexPairs[100].Vertex1: %d\n", VertexPairs[100].Vertex1);
    printf("VertexPairs[100].Vertex2: %d\n", VertexPairs[100].Vertex2);
    printf("VertexPairsEnd[0].Vertex1: %d\n", VertexPairsEnd[0].Vertex1);
    printf("VertexPairsEnd[0].Vertex2: %d\n", VertexPairsEnd[0].Vertex2);
    printf("Number of keys: %d\n", NumberOfKeys);

    printf("Starting thrust::sort...\n");
    thrust::sort(thrust::device, VertexPairs, VertexPairsEnd);
    printf("Finished thrust sort...\n");
    printf("VertexPairs[0].Vertex1: %d\n", VertexPairs[0].Vertex1);
    printf("VertexPairs[0].Vertex2: %d\n", VertexPairs[0].Vertex2);
    printf("VertexPairs[100].Vertex1: %d\n", VertexPairs[100].Vertex1);
    printf("VertexPairs[100].Vertex2: %d\n", VertexPairs[100].Vertex2);
    printf("VertexPairsEnd[0].Vertex1: %d\n", VertexPairsEnd[0].Vertex1);
    printf("VertexPairsEnd[0].Vertex2: %d\n", VertexPairsEnd[0].Vertex2);

    for (uint32_t i = 0; i < NumberOfKeys; i++) {
        printf("%d,%d\n",
               VertexPairs[i].Vertex1,
               VertexPairs[i].Vertex2);
    }

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
IsGraphAcyclicHost(
    _In_ PGRAPH Graph
    )
{
    if (IsUsingAssigned16(Graph)) {
        GraphCuIsAcyclic<GRAPH16>((PGRAPH16)Graph);
    } else {
        GraphCuIsAcyclic<GRAPH32>((PGRAPH32)Graph);
    }
    return;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab filetype=cuda formatoptions=croql   :
