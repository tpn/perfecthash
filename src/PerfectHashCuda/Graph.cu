/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    Graph.cu

Abstract:

    CUDA graph implementation.

--*/

#include <PerfectHashCuda.h>

EXTERN_C_BEGIN
#include "../PerfectHash/CuDeviceAttributes.h"
#include "../PerfectHash/Graph.h"

#include <cuda.h>
#include <cuda_device_runtime_api.h>
EXTERN_C_END

#include <curand_kernel.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/unique.h>

#include "Graph.cuh"

//
// Shared memory.
//

extern SHARED ULONG SharedRaw[];

//
// Error handling.
//

EXTERN_C
DEVICE
VOID
PerfectHashPrintCuError(
    PCSZ FunctionName,
    PCSZ FileName,
    ULONG LineNumber,
    ULONG Error
    )
{
    PCSZ ErrorName;
    PCSZ ErrorString;

    ErrorName = cudaGetErrorName((CU_RESULT)Error);
    ErrorString = cudaGetErrorString((CU_RESULT)Error);

    //
    // Error message format:
    //
    //      <FileName>:<LineNumber>: <Name> failed with error <Code>: \
    //          <ErrorName>: <ErrorString>.
    //

    printf("%s:%d %s failed with error 0x%x: %s: %s.\n",
           FileName,
           LineNumber,
           FunctionName,
           Error,
           ErrorName,
           ErrorString);
}

EXTERN_C
DEVICE
VOID
PerfectHashPrintError(
    PCSZ FunctionName,
    PCSZ FileName,
    ULONG LineNumber,
    ULONG Result
    )
{
    printf("%s:%d %s failed with error 0x%x.\n",
           FileName,
           LineNumber,
           FunctionName,
           Result);
}

EXTERN_C
GLOBAL
VOID
HashAllMultiplyShiftRKernel(
    _In_reads_(NumberOfKeys) PKEY Keys,
    _In_ ULONG NumberOfKeys,
    _Out_writes_(NumberOfKeys) PVERTEX_PAIR VertexPairs,
    _In_ ULONG Mask,
    _In_ PULONG Seeds,
    _Out_ PHRESULT GlobalResult
    )
{
    KEY Key;
    ULONG Index;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    PHRESULT BlockResult;
    PINT2 Output = (PINT2)VertexPairs;
    PGRAPH_SHARED Shared = (PGRAPH_SHARED)SharedRaw;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    //
    // If this is thread 0 in the block, initialize the shared memory and set
    // the global result to S_OK.
    //

    if (ThreadIndex.x == 0) {

        Shared->HashKeysBlockResults = (PHRESULT)(
            RtlOffsetToPointer(
                SharedRaw,
                sizeof(GRAPH_SHARED)
            )
        );


        *GlobalResult = S_OK;
    }

    __syncthreads();

    BlockResult = &Shared->HashKeysBlockResults[BlockIndex.x];

    FOR_EACH_1D(Index, NumberOfKeys) {

        //
        // Block-level fast-path exit if we've already detected a vertex
        // collision.  I haven't profiled things to determine if it makes
        // sense to either do: a) this, or b) an additional global memory
        // read of `*GlobalResult` (currently not being done).
        //

        if (*BlockResult != S_OK) {
            goto End;
        }

        Key = Keys[Index];

        Vertex1 = (((Key * SEED1) >> SEED3_BYTE1) & Mask);
        Vertex2 = (((Key * SEED2) >> SEED3_BYTE2) & Mask);

        if (Vertex1 == Vertex2) {

            //
            // Set the block-level and global-level results to indicate
            // collision, then jump to the end.
            //

            *BlockResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            *GlobalResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        }

        //
        // Store the vertex pairs.
        //

        Output[Index].x = Vertex1;
        Output[Index].y = Vertex2;
    }

End:
    return;
}

EXTERN_C
GLOBAL
VOID
HashAllMultiplyShiftRKernel2(
    _In_reads_(NumberOfKeys) PKEY Keys,
    _In_ ULONG NumberOfKeys,
    _Out_writes_(NumberOfKeys) PVERTEX Vertices1,
    _Out_writes_(NumberOfKeys) PVERTEX Vertices2,
    _Out_writes_(NumberOfKeys) PULONG Vertices1Index,
    _In_ ULONG Mask,
    _In_ PULONG Seeds,
    _Out_ PHRESULT GlobalResult
    )
{
    KEY Key;
    ULONG Index;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    VERTEX Vertex1;
    VERTEX Vertex2;
    PHRESULT BlockResult;
    PGRAPH_SHARED Shared = (PGRAPH_SHARED)SharedRaw;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    //
    // If this is thread 0 in the block, initialize the shared memory and set
    // the global result to S_OK.
    //

    if (ThreadIndex.x == 0) {

        Shared->HashKeysBlockResults = (PHRESULT)(
            RtlOffsetToPointer(
                SharedRaw,
                sizeof(GRAPH_SHARED)
            )
        );


        *GlobalResult = S_OK;
    }

    __syncthreads();

    BlockResult = &Shared->HashKeysBlockResults[BlockIndex.x];

    FOR_EACH_1D(Index, NumberOfKeys) {

        //
        // Block-level fast-path exit if we've already detected a vertex
        // collision.  I haven't profiled things to determine if it makes
        // sense to either do: a) this, or b) an additional global memory
        // read of `*GlobalResult` (currently not being done).
        //

        if (*BlockResult != S_OK) {
            goto End;
        }

        Key = Keys[Index];

        Vertex1 = (((Key * SEED1) >> SEED3_BYTE1) & Mask);
        Vertex2 = (((Key * SEED2) >> SEED3_BYTE2) & Mask);

        if (Vertex1 == Vertex2) {

            //
            // Set the block-level and global-level results to indicate
            // collision, then jump to the end.
            //

            *BlockResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            *GlobalResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        }

        //
        // Store the vertex pairs.
        //

        //Output[Index].x = Vertex1;
        //Output[Index].y = Vertex2;
        Vertices1[Index] = Vertex1;
        Vertices2[Index] = Vertex2;
        Vertices1Index[Index] = Index;

    }

End:
    return;
}

EXTERN_C
GLOBAL
VOID
HashAllMultiplyShiftRKernel3(
    _In_reads_(NumberOfKeys) PKEY Keys,
    _In_ ULONG NumberOfKeys,
    _Out_writes_(NumberOfKeys) PVERTEX Vertices1,
    _Out_writes_(NumberOfKeys) PVERTEX Vertices2,
    _Out_writes_(NumberOfKeys) PVERTEX_PAIR VertexPairs,
    _Out_writes_(NumberOfKeys) PULONG Vertices1Index,
    _Out_writes_(NumberOfKeys) PULONG Vertices2Index,
    _Out_writes_(NumberOfKeys) PULONG VertexPairsIndex,
    _In_ ULONG Mask,
    _In_ PULONG Seeds,
    _Out_ PHRESULT GlobalResult
    )
{
    KEY Key;
    ULONG Index;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    VERTEX Vertex1;
    VERTEX Vertex2;
    PHRESULT BlockResult;
    PGRAPH_SHARED Shared = (PGRAPH_SHARED)SharedRaw;
    PINT2 Output = (PINT2)VertexPairs;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

#if 0
    //
    // If this is thread 0 in the block, initialize the shared memory and set
    // the global result to S_OK.
    //

    if (ThreadIndex.x == 0) {

        Shared->HashKeysBlockResults = (PHRESULT)(
            RtlOffsetToPointer(
                SharedRaw,
                sizeof(GRAPH_SHARED)
            )
        );


        *GlobalResult = S_OK;
    }

    __syncthreads();

    BlockResult = &Shared->HashKeysBlockResults[BlockIndex.x];
#endif

    FOR_EACH_1D(Index, NumberOfKeys) {

        //
        // Block-level fast-path exit if we've already detected a vertex
        // collision.  I haven't profiled things to determine if it makes
        // sense to either do: a) this, or b) an additional global memory
        // read of `*GlobalResult` (currently not being done).
        //

#if 0
        if (*BlockResult != S_OK) {
            goto End;
        }
#endif

        Key = Keys[Index];

        Vertex1 = (((Key * SEED1) >> SEED3_BYTE1) & Mask);
        Vertex2 = (((Key * SEED2) >> SEED3_BYTE2) & Mask);

        if (Vertex1 == Vertex2) {

            //
            // Set the block-level and global-level results to indicate
            // collision, then jump to the end.
            //

            //*BlockResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            *GlobalResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        }

        //
        // Store the vertex pairs.
        //

        //Output[Index].x = Vertex1;
        //Output[Index].y = Vertex2;
        VertexPairs[Index].Vertex1 = Vertex1;
        VertexPairs[Index].Vertex2 = Vertex2;
        Vertices1[Index] = Vertex1;
        Vertices2[Index] = Vertex2;
        Vertices1Index[Index] = Index;
        Vertices2Index[Index] = Index;
        VertexPairsIndex[Index] = Index;

    }

End:
    return;
}

KERNEL
VOID
GraphCuSortVertices1Kernel(
    _In_ PGRAPH Graph
    )
{
    ULONG UniqueCount;
    thrust::device_ptr<ULONG> Vertices1(Graph->Vertices1);
    thrust::device_ptr<ULONG> Index(Graph->Vertices1Index);

#if 0
    thrust::stable_sort_by_key(thrust::device,
                               Vertices1,
                               Vertices1 + Graph->NumberOfKeys,
                               Index);
#endif

    thrust::stable_sort(
        thrust::device,
        Vertices1,
        Vertices1 + Graph->NumberOfKeys
    );

    UniqueCount = thrust::inner_product(
        thrust::device,
        Vertices1,
        Vertices1 + Graph->NumberOfKeys - 1,
        Vertices1 + 1,
        ULONG(1),
        thrust::plus<ULONG>(),
        thrust::not_equal_to<ULONG>()
    );

    printf("Vertices1:\n\tUniqueCount: %u\n\tKeys: %u\n\t"
           "Seed1: %u\n\tSeed2: %u\n\tSeed3: %u\n",
           UniqueCount,
           Graph->NumberOfKeys,
           Graph->Seed1,
           Graph->Seed2,
           Graph->Seed3);

}

KERNEL
VOID
GraphCuSortVertices2Kernel(
    _In_ PGRAPH Graph
    )
{
    thrust::device_ptr<ULONG> Vertices2(Graph->Vertices2);
    thrust::device_ptr<ULONG> Index(Graph->Vertices2Index);

    thrust::stable_sort_by_key(thrust::device,
                               Vertices2,
                               Vertices2 + Graph->NumberOfKeys,
                               Index);
}

DEVICE
bool
VertexPairLessThan(
    const VERTEX_PAIR Left,
    const VERTEX_PAIR Right
    )
{
    if (Left.Vertex1 < Right.Vertex1) {
        return true;
    } else if (Left.Vertex1 == Right.Vertex1) {
        return (Left.Vertex2 < Right.Vertex2);
    } else {
        return false;
    }
}

DEVICE
bool
VertexPairNotEqual(
    const VERTEX_PAIR Left,
    const VERTEX_PAIR Right
    )
{
    return (
        Left.Vertex1 != Right.Vertex1 &&
        Left.Vertex2 != Right.Vertex2
    );
    //return (Left.AsULongLong != Right.AsULongLong);
}

DEVICE
bool
VertexPairEqual(
    const VERTEX_PAIR Left,
    const VERTEX_PAIR Right
    )
{
    return (
        Left.Vertex1 == Right.Vertex1 &&
        Left.Vertex2 == Right.Vertex2
    );
    //return (Left.AsULongLong == Right.AsULongLong);
}


KERNEL
VOID
GraphCuSortVertexPairsKernel(
    _In_ PGRAPH Graph
    )
{
    ULONG UniqueCount;
    thrust::device_ptr<VERTEX_PAIR> VertexPairs(Graph->VertexPairs);
    thrust::device_ptr<VERTEX_PAIR> VertexPairsEnd;
    thrust::device_ptr<VERTEX_PAIR> EndUnique;
    thrust::device_ptr<ULONG> Index(Graph->VertexPairsIndex);

    thrust::stable_sort(
        thrust::device,
        VertexPairs,
        VertexPairs + Graph->NumberOfKeys,
        VertexPairLessThan
    );

    UniqueCount = thrust::inner_product(
        thrust::device,
        VertexPairs,
        VertexPairs + Graph->NumberOfKeys - 1,
        VertexPairs + 1,
        ULONG(1),
        thrust::plus<ULONG>(),
        VertexPairNotEqual
    );

    printf("VertexPair NE:\n\tUniqueCount: %u\n\tKeys: %u\n\t"
           "Seed1: %u\n\tSeed2: %u\n\tSeed3: %u\n",
           UniqueCount,
           Graph->NumberOfKeys,
           Graph->Seed1,
           Graph->Seed2,
           Graph->Seed3);

    UniqueCount = thrust::inner_product(
        thrust::device,
        VertexPairs,
        VertexPairs + Graph->NumberOfKeys - 1,
        VertexPairs + 1,
        ULONG(1),
        thrust::plus<ULONG>(),
        VertexPairEqual
    );

    printf("VertexPair EQ:\n\tUniqueCount: %u\n\tKeys: %u\n\t"
           "Seed1: %u\n\tSeed2: %u\n\tSeed3: %u\n",
           UniqueCount,
           Graph->NumberOfKeys,
           Graph->Seed1,
           Graph->Seed2,
           Graph->Seed3);

    VertexPairsEnd = VertexPairs + Graph->NumberOfKeys;
    EndUnique = thrust::unique(
        thrust::device,
        VertexPairs,
        VertexPairsEnd,
        VertexPairEqual
    );

    if (VertexPairsEnd != EndUnique) {
        printf("Not unique!\n");
    } else {
        printf("All unique!\n");
    }

    printf("Start: 0x%p, End: 0x%p, EndUnique: 0x%p.\n",
           VertexPairs,
           VertexPairsEnd,
           EndUnique);

    printf("Num element: %u.\n", (VertexPairsEnd - VertexPairs));
    printf("Num unique: %u.\n", (EndUnique - VertexPairs));

}

KERNEL
VOID
GraphCuAddEdgesKernel(
    _In_ ULONG NumberOfEdges,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PVERTEX_PAIR VertexPairs,
    _Out_ PEDGE Edges,
    _Out_ PEDGE Next,
    _Out_ PVERTEX First
    )
{
    EDGE Edge1;
    EDGE Edge2;
    EDGE First1;
    EDGE First2;
    ULONG Index;
    PINT2 Input = (PINT2)VertexPairs;
    VERTEX Vertex1;
    VERTEX Vertex2;

    FOR_EACH_1D(Index, NumberOfKeys) {
        Vertex1 = Input[Index].x;
        Vertex2 = Input[Index].y;

        Edge1 = (EDGE)Index;
        Edge2 = Edge1 + NumberOfEdges;

        //
        // Insert the first edge.
        //

        First1 = First[Vertex1];
        Next[Edge1] = First1;
        First[Vertex1] = Edge1;
        Edges[Edge1] = Vertex2;

        //
        // Insert the second edge.
        //

        First2 = First[Vertex2];
        Next[Edge2] = First2;
        First[Vertex2] = Edge2;
        Edges[Edge2] = Vertex1;
    }
}

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
    return (Graph->Attempt <= 5);
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

    if (HasUserSeeds(Graph)) { // && Graph->Attempt == 1) {

        //
        // The user has supplied seeds, and this is the first attempt, so skip
        // the curand() calls.
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

EXTERN_C
DEVICE
HRESULT
GraphCuReset(
    _In_ PGRAPH Graph
    )
/*++

Routine Description:

    Resets the state of a graph instance after a solving attempt, such that it
    can be used for a subsequent attempt.

Arguments:

    Graph - Supplies a pointer to the graph instance to reset.

Return Value:

    PH_S_CONTINUE_GRAPH_SOLVING - Graph was successfully reset and graph solving
        should continue.

    PH_S_GRAPH_SOLVING_STOPPED - Graph solving has been stopped.  The graph is
        not reset and solving should not continue.

    PH_S_TABLE_RESIZE_IMMINENT - The reset was not performed as a table resize
        is imminent (and thus, attempts at solving this current graph can be
        stopped).

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

--*/
{
    HRESULT Result;
    PGRAPH_INFO Info;
    CU_STREAM Stream;
    CU_RESULT CuResult;
    ULONG TotalNumberOfPages;
    ULONG TotalNumberOfLargePages;
    ULONG TotalNumberOfCacheLines;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PASSIGNED_PAGE_COUNT NumberOfAssignedPerPage;
    PASSIGNED_LARGE_PAGE_COUNT NumberOfAssignedPerLargePage;
    PASSIGNED_CACHE_LINE_COUNT NumberOfAssignedPerCacheLine;

    //
    // Initialize aliases.
    //

    Info = Graph->CuGraphInfo;

    Result = PH_S_CONTINUE_GRAPH_SOLVING;

    Stream = Graph->CuKernelContext->Streams.Reset;

    //
    // Clear scalar values.
    //

    Graph->Collisions = 0;
    Graph->NumberOfEmptyVertices = 0;
    Graph->DeletedEdgeCount = 0;
    Graph->VisitedVerticesCount = 0;

    Graph->TraversalDepth = 0;
    Graph->TotalTraversals = 0;
    Graph->MaximumTraversalDepth = 0;

    Graph->Flags.Shrinking = FALSE;
    Graph->Flags.IsAcyclic = FALSE;

    Graph->AddKeysElapsedCycles.QuadPart = 0;
    Graph->HashKeysElapsedCycles.QuadPart = 0;
    Graph->AddHashedKeysElapsedCycles.QuadPart = 0;

    //
    // XXX: temp.
    //

    return Result;

    //
    // Clear the bitmap buffers.
    //

#define ZERO_BITMAP_BUFFER(Name)                           \
    ASSERT(0 == Info->##Name##BufferSizeInBytes -          \
           ((Info->##Name##BufferSizeInBytes >> 3) << 3)); \
    CU_ZERO(Graph->##Name##.Buffer,                        \
            Info->##Name##BufferSizeInBytes,               \
            Stream)

    ZERO_BITMAP_BUFFER(DeletedEdgesBitmap);
    ZERO_BITMAP_BUFFER(VisitedVerticesBitmap);
    ZERO_BITMAP_BUFFER(AssignedBitmap);
    ZERO_BITMAP_BUFFER(IndexBitmap);

    //
    // "Empty" all of the nodes.
    //

#define EMPTY_ARRAY(Name)                                           \
    ASSERT(0 == Info->##Name##SizeInBytes -                         \
           ((Info->##Name##SizeInBytes >> 3) << 3));                \
    CU_MEMSET(Graph->##Name, 0xffffffff, Info->##Name##SizeInBytes, Stream)

    EMPTY_ARRAY(First);
    EMPTY_ARRAY(Next);
    EMPTY_ARRAY(Edges);

    //
    // Avoid the overhead of resetting the memory coverage if we're in "first
    // graph wins" mode and have been requested to skip memory coverage.
    //

    if (!FindBestGraph(Graph)) {
        goto End;
    }

    //
    // Clear the assigned memory coverage counts and arrays.
    //

    Coverage = &Graph->AssignedMemoryCoverage;

    //
    // Capture the totals and pointers prior to zeroing the struct.
    //

    TotalNumberOfPages = Coverage->TotalNumberOfPages;
    TotalNumberOfLargePages = Coverage->TotalNumberOfLargePages;
    TotalNumberOfCacheLines = Coverage->TotalNumberOfCacheLines;

    NumberOfAssignedPerPage = Coverage->NumberOfAssignedPerPage;
    NumberOfAssignedPerLargePage = Coverage->NumberOfAssignedPerLargePage;
    NumberOfAssignedPerCacheLine = Coverage->NumberOfAssignedPerCacheLine;

    CU_ZERO(Coverage, sizeof(*Coverage), Stream);

    //
    // Restore the totals and pointers.
    //

    Coverage->TotalNumberOfPages = TotalNumberOfPages;
    Coverage->TotalNumberOfLargePages = TotalNumberOfLargePages;
    Coverage->TotalNumberOfCacheLines = TotalNumberOfCacheLines;

    Coverage->NumberOfAssignedPerPage = NumberOfAssignedPerPage;
    Coverage->NumberOfAssignedPerLargePage = NumberOfAssignedPerLargePage;
    Coverage->NumberOfAssignedPerCacheLine = NumberOfAssignedPerCacheLine;

#define ZERO_ASSIGNED_ARRAY(Name) \
    CU_ZERO(Coverage->##Name, Info->##Name##SizeInBytes, Stream)

    //ZERO_ASSIGNED_ARRAY(NumberOfAssignedPerPage);
    //ZERO_ASSIGNED_ARRAY(NumberOfAssignedPerLargePage);
    //ZERO_ASSIGNED_ARRAY(NumberOfAssignedPerCacheLine);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (SUCCEEDED(Result)) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

EXTERN_C
DEVICE
BOOLEAN
GraphCuIsAcyclic(
    _In_ PGRAPH Graph
    )
{
    return FALSE;
}

EXTERN_C
DEVICE
VOID
GraphCuAssign(
    _In_ PGRAPH Graph
    )
{

}

EXTERN_C
DEVICE
VOID
GraphCuCalculateAssignedMemoryCoverage(
    _In_ PGRAPH Graph
    )
{

}

EXTERN_C
DEVICE
VOID
GraphCuCalculateAssignedMemoryCoverageForKeysSubset(
    _In_ PGRAPH Graph
    )
{

}

EXTERN_C
DEVICE
HRESULT
GraphCuRegisterSolved(
    _In_ PGRAPH Graph,
    _Inout_ PGRAPH *NewGraphPointer
    )
{
    return PH_S_STOP_GRAPH_SOLVING;
}


EXTERN_C
DEVICE
HRESULT
GraphCuSolve(
    _In_ PGRAPH Graph,
    _Out_ PGRAPH *NewGraphPointer
    )
/*++

Routine Description:

    Add all keys to the hypergraph using the unique seeds to hash each key into
    two vertex values, connected by a "hyper-edge".  Determine if the graph is
    acyclic, if it is, we've "solved" the graph.  If not, we haven't.

Arguments:

    Graph - Supplies a pointer to the graph to be solved.

    NewGraphPointer - Supplies the address of a variable which will receive the
        address of a new graph instance to be used for solving if the routine
        returns PH_S_USE_NEW_GRAPH_FOR_SOLVING.

Return Value:

    PH_S_STOP_GRAPH_SOLVING - Stop graph solving.

    PH_S_GRAPH_SOLVING_STOPPED - Graph solving has been stopped.

    PH_S_CONTINUE_GRAPH_SOLVING - Continue graph solving.

    PH_S_USE_NEW_GRAPH_FOR_SOLVING - Continue graph solving but use the graph
        returned via the NewGraphPointer parameter.

--*/
{
    PKEY Keys;
    HRESULT Result;
    CU_RESULT CuResult;
    ULONG NumberOfKeys;
    ULONG BlocksPerGrid;
    ULONG ThreadsPerBlock;
    ULONG SharedMemoryInBytes;
    PCU_KERNEL_CONTEXT Ctx;
    PCU_KERNEL_STREAMS Streams;
    PASSIGNED_MEMORY_COVERAGE Coverage;

    //
    // Initialize aliases.
    //

    Ctx = Graph->CuKernelContext;
    Streams = &Ctx->Streams;
    Keys = (PKEY)Graph->DeviceKeys;
    NumberOfKeys = Graph->NumberOfKeys;
    BlocksPerGrid = Graph->CuBlocksPerGrid;
    ThreadsPerBlock = Graph->CuThreadsPerBlock;

    //
    // Attempt to add all the keys to the graph.
    //

    Graph->CuHashKeysResult = E_FAIL;

    SharedMemoryInBytes = (

        //
        // Account for the GRAPH_SHARED structure.
        //

        sizeof(GRAPH_SHARED) +

        //
        // Account for the array of result codes (one per block) for HashKeys.
        //

        (sizeof(HRESULT) * BlocksPerGrid)

    );

    //
    // Launch the kernel.
    //

#if 0
    HashAllMultiplyShiftRKernel<<<
        BlocksPerGrid,
        ThreadsPerBlock,
        SharedMemoryInBytes,
        Stream
    >>>(
        Keys,
        NumberOfKeys,
        Graph->VertexPairs,
        Graph->VertexMask,
        Graph->Seeds,
        &Graph->CuHashKeysResult
    );
#endif
#if 0
    HashAllMultiplyShiftRKernel2<<<
        BlocksPerGrid,
        ThreadsPerBlock,
        SharedMemoryInBytes,
        Graph->SolveStream
    >>>(
        Keys,
        NumberOfKeys,
        Graph->Vertices1,
        Graph->Vertices2,
        Graph->VerticesIndex,
        Graph->VertexMask,
        Graph->Seeds,
        &Graph->CuHashKeysResult
    );

    CuResult = cudaDeviceSynchronize();
    CU_CHECK(CuResult, cudaDeviceSynchronize);

    Result = GraphCuSortVertices(Graph);
    if (FAILED(Result)) {
        goto End;
    }
#else

    Graph->CuHashKeysResult = S_OK;

    HashAllMultiplyShiftRKernel3<<<
        BlocksPerGrid,
        ThreadsPerBlock,
        SharedMemoryInBytes,
        Streams->Solve
    >>>(
        Keys,
        NumberOfKeys,
        Graph->Vertices1,
        Graph->Vertices2,
        Graph->VertexPairs,
        Graph->Vertices1Index,
        Graph->Vertices2Index,
        Graph->VertexPairsIndex,
        Graph->VertexMask,
        Graph->Seeds,
        &Graph->CuHashKeysResult
    );

    CuResult = cudaDeviceSynchronize();
    CU_CHECK(CuResult, cudaDeviceSynchronize);

    Result = Graph->CuHashKeysResult;
    if (FAILED(Result)) {
        if (Result == PH_E_GRAPH_VERTEX_COLLISION_FAILURE) {
            printf("Collided!\n");
            Graph->CuVertexCollisionFailures++;
            goto Failed;
        }
        PH_ERROR(GraphCuSolve_AddKeys, Result);
        Result = PH_S_STOP_GRAPH_SOLVING;
        goto End;
    }

    printf("No collision.\n");
    Graph->CuNoVertexCollisionFailures++;

    GraphCuSortVertices1Kernel<<<1, 1, 0, Streams->SortVertices1>>>(Graph);
    GraphCuSortVertices2Kernel<<<1, 1, 0, Streams->SortVertices2>>>(Graph);
    GraphCuSortVertexPairsKernel<<<1, 1, 0, Streams->SortVertexPairs>>>(Graph);

    CuResult = cudaDeviceSynchronize();
    CU_CHECK(CuResult, cudaDeviceSynchronize);

    printf("Sorted kernels!\n");

#endif

#if 0
    //MAYBE_STOP_GRAPH_SOLVING(Graph);

    GraphCuAddEdgesKernel<<<BlocksPerGrid, ThreadsPerBlock, 0, Stream>>>(
        Graph->NumberOfEdges,
        Graph->NumberOfKeys,
        Graph->VertexPairs,
        Graph->Edges,
        Graph->Next,
        Graph->First
    );

    CuResult = cudaDeviceSynchronize();
    CU_CHECK(CuResult, cudaDeviceSynchronize);
#endif

    if (!GraphCuIsAcyclic(Graph)) {

        //
        // Failed to create an acyclic graph.
        //

        Graph->CuCyclicGraphFailures++;
        goto Failed;
    }

    //
    // We created an acyclic graph.
    //

    Graph->CuFinishedCount++;

    //
    // Perform the assignment step.
    //

    GraphCuAssign(Graph);

    //
    // If we're in "first graph wins" mode and we reach this point, we're the
    // winning thread, so, push the graph onto the finished list head, then
    // submit the relevant finished threadpool work item and return stop graph
    // solving.
    //

    if (FirstSolvedGraphWins(Graph)) {
        if (WantsAssignedMemoryCoverage(Graph)) {
            GraphCuCalculateAssignedMemoryCoverage(Graph);
        }
        return PH_S_STOP_GRAPH_SOLVING;
    }

    //
    // If we reach this mode, we're in "find best graph" mode, so, register the
    // solved graph then continue solving.
    //

    ASSERT(FindBestGraph(Graph));

    //
    // Calculate memory coverage information if applicable.
    //

    if (WantsAssignedMemoryCoverage(Graph)) {
        GraphCuCalculateAssignedMemoryCoverage(Graph);
    } else if (WantsAssignedMemoryCoverageForKeysSubset(Graph)) {
        GraphCuCalculateAssignedMemoryCoverageForKeysSubset(Graph);
    }

    //
    // This is a bit hacky; the graph traversal depth is proving to be more
    // interesting than initially thought, such that we've recently added a
    // best coverage type predicate aimed at maximizing it, which means we
    // need to make the value available from the coverage struct in order for
    // the X-macro to work, which means we're unnecessarily duplicating the
    // value at the table and coverage level.  Not particularly elegant.
    //

    Coverage = &Graph->AssignedMemoryCoverage;
    Coverage->MaxGraphTraversalDepth = Graph->MaximumTraversalDepth;

    //
    // Ditto for total traversals, empty vertices and collisions.
    //

    Coverage->TotalGraphTraversals = Graph->TotalTraversals;
    Coverage->NumberOfEmptyVertices = Graph->NumberOfEmptyVertices;
    Coverage->NumberOfCollisionsDuringAssignment = Graph->Collisions;

    //
    // Register the solved graph.  We can return this result directly.
    //

    Result = GraphCuRegisterSolved(Graph, NewGraphPointer);

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Intentional follow-on to Error.
    //

Error:
    return Result;

Failed:
    Graph->CuFailedAttempts++;
    return PH_S_CONTINUE_GRAPH_SOLVING;
}

EXTERN_C
DEVICE
HRESULT
GraphCuCreateKernelContext(
    _In_ PGRAPH Graph
    )
{
    HRESULT Result;
    CU_RESULT CuResult;
    PCU_KERNEL_STREAMS Streams;
    PCU_KERNEL_CONTEXT Ctx;

    if (Graph->CuKernelContext != NULL) {
        Result = S_FALSE;
        goto End;
    }

    CuResult = cudaMalloc(&Graph->CuKernelContext, sizeof(*Ctx));
    CU_CHECK(CuResult, cudaMalloc);

    //
    // Create streams.
    //

    Ctx = Graph->CuKernelContext;
    Streams = &Ctx->Streams;
    CREATE_STREAM(&Streams->Reset);
    CREATE_STREAM(&Streams->SortVertices1);
    CREATE_STREAM(&Streams->SortVertices2);
    CREATE_STREAM(&Streams->SortVertexPairs);
    CREATE_STREAM(&Streams->Solve);

    //
    // Initialize our random state.
    //

    curand_init(Graph->CuRngSeed,
                Graph->CuRngSubsequence,
                Graph->CuRngOffset,
                &Ctx->RngState.Philox4);

    Result = S_OK;
    goto End;

Error:

    if (SUCCEEDED(Result)) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}


EXTERN_C
GLOBAL
VOID
PerfectHashCudaEnterSolvingLoop(
    _In_ PGRAPH Graph
    )
/*++

Routine Description:

    This is the main entry point for the CUDA graph solving implementation.
    This kernel is intended to be called with a single thread.  It launches
    child kernels dynamically.

Arguments:

    Graph - Supplies a pointer to a GRAPH structure for which solving is to be
        performed.

Return Value:

    N.B. The return value is provided to the caller via Graph->CuKernelResult.

    S_OK - Success.

--*/
{
    HRESULT Result;
    PGRAPH NewGraph;

    //
    // Abort if the kernel is called with more than one thread.
    //

    if (GridDim.x > 1  || GridDim.y > 1  || GridDim.z > 1 ||
        BlockDim.x > 1 || BlockDim.y > 1 || BlockDim.z > 1)
    {
        Result = PH_E_CU_KERNEL_SOLVE_LOOP_INVALID_DIMENSIONS;
        goto End;
    }

    //ASSERT(Graph->SizeOfStruct == sizeof(GRAPH));
    if (Graph->SizeOfStruct != sizeof(GRAPH)) {
        printf("%u != %u!\n", (ULONG)Graph->SizeOfStruct, (ULONG)sizeof(GRAPH));
        return;
    }

    if (Graph->CuKernelContext == NULL) {
        Result = GraphCuCreateKernelContext(Graph);
        if (FAILED(Result)) {
            PH_ERROR(GraphCuCreateKernelContext, Result);
            return;
        }
        printf("Created context successfully.\n");
    }

    //
    // Begin the solving loop.
    //

    do {

        if (!GraphCuShouldWeContinueTryingToSolve(Graph, &Result)) {
            break;
        }

        Result = GraphCuLoadNewSeeds(Graph);
        if (FAILED(Result)) {
            break;
        }

        Result = GraphCuReset(Graph);
        if (FAILED(Result)) {
            break;
        } else if (Result != PH_S_CONTINUE_GRAPH_SOLVING) {
            break;
        }

        NewGraph = NULL;
        Result = GraphCuSolve(Graph, &NewGraph);
        if (FAILED(Result)) {
            break;
        }

        if (Result == PH_S_STOP_GRAPH_SOLVING ||
            Result == PH_S_GRAPH_SOLVING_STOPPED) {
            ASSERT(NewGraph == NULL);
            break;
        }

        if (Result == PH_S_USE_NEW_GRAPH_FOR_SOLVING) {
            ASSERT(NewGraph != NULL);
            Graph = NewGraph;
        } else {

            //
            // Invariant check: result should be PH_S_CONTINUE_GRAPH_SOLVING
            // at this point.
            //

            ASSERT(Result == PH_S_CONTINUE_GRAPH_SOLVING);
        }

        //
        // Continue the loop and attempt another solve.
        //

    } while (TRUE);

    //
    // We're done, finish up.
    //

    goto End;

#if 0
Error:

    if (SUCCEEDED(Result)) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

#endif
End:
    Graph->CuKernelResult = Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab filetype=cuda formatoptions=croql   :
