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

#include "Graph.cuh"

//#include "GraphImpl.cu"

//
// Define helper macros.
//

#define EMPTY ((VERTEX)-1)
#define GRAPH_NO_NEIGHBOR ((VERTEX)-1)
#define NO_THREAD_ID ((unsigned int)-1)

#define IsEmpty(Value) ((ULONG)Value == EMPTY)
#define IsNeighborEmpty(Neighbor) ((ULONG)Neighbor == EMPTY)

//
// When a solution has been found and the assignment step begins, the initial
// value assigned to a vertex is govered by the following macro.
//

#define INITIAL_ASSIGNMENT_VALUE 0

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
    _In_ ULONG NumberOfEdges,
    _In_ ULONG NumberOfKeys,
    _Out_ PVERTEX First,
    _Out_ PEDGE Next,
    _Out_ PEDGE Edges,
    _In_ ULONG Mask,
    _In_ PULONG Seeds,
    _Out_ PHRESULT GlobalResult
    )
{
    KEY Key;
    EDGE Edge1;
    EDGE Edge2;
    EDGE First1;
    EDGE First2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG Index;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    FOR_EACH_1D(Index, NumberOfKeys) {

        Key = Keys[Index];

        Vertex1 = (((Key * SEED1) >> SEED3_BYTE1) & Mask);
        Vertex2 = (((Key * SEED2) >> SEED3_BYTE2) & Mask);

        if (Vertex1 == Vertex2) {
            *GlobalResult = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        }

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

End:
    return;
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

KERNEL
VOID
GraphCuResetArraysKernel(
    _In_ PGRAPH Graph
    )
{
    ULONG Index;
    ULONG Total;

    Total = max(Graph->TotalNumberOfEdges, Graph->NumberOfVertices);

    FOR_EACH_1D(Index, Total) {

        //printf("[%d]: %d < %d\n", ThreadId, Index, Total);

        if (Index < Graph->NumberOfKeys) {
            Graph->Order[Index] = 0;
        }

        if (Index < Graph->TotalNumberOfEdges) {
            Graph->Next[Index] = EMPTY;
            Graph->Edges[Index] = EMPTY;
        }

        if (Index < Graph->NumberOfVertices) {
            Graph->First[Index] = EMPTY;
            Graph->Assigned[Index] = 0;
            Graph->Deleted[Index] = NO_THREAD_ID;
            Graph->Visited[Index] = NO_THREAD_ID;
        }
    }
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
    CU_STREAM Stream;
    CU_RESULT CuResult;
    LONG Old;
    ULONG SharedMemoryInBytes;
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

    Result = PH_S_CONTINUE_GRAPH_SOLVING;
    Stream = Graph->CuKernelContext->Streams.Reset;

    //
    // Clear scalar values.
    //

    Graph->OrderIndex = (LONG)Graph->NumberOfVertices;
    ASSERT(Graph->OrderIndex > 0);

    Old = atomicSub((LONG *)&Graph->OrderIndex, 1);
    printf("1: atomicSub(): Old = %d, Graph->OrderIndex = %d\n",
           Old,
           Graph->OrderIndex);

    Old = atomicSub((LONG *)&Graph->OrderIndex, 1);
    printf("2: atomicSub(): Old = %d, Graph->OrderIndex = %d\n",
           Old,
           Graph->OrderIndex);

    Old = atomicSub((LONG *)&Graph->OrderIndex, 1);
    printf("3: atomicSub(): Old = %d, Graph->OrderIndex = %d\n",
           Old,
           Graph->OrderIndex);

    Graph->OrderIndex = (LONG)Graph->NumberOfVertices;

    Graph->Flags.Shrinking = FALSE;
    Graph->Flags.IsAcyclic = FALSE;

    //
    // Clear the arrays via a parallel kernel.
    //

    SharedMemoryInBytes = 0;
    GraphCuResetArraysKernel<<<
        Graph->CuBlocksPerGrid,
        Graph->CuThreadsPerBlock,
        SharedMemoryInBytes,
        Stream
    >>>(Graph);

    //CuResult = cudaDeviceSynchronize();
    //CU_CHECK(CuResult, cudaDeviceSynchronize);

    //
    // Avoid the overhead of resetting the memory coverage if we're in "first
    // graph wins" mode and have been requested to skip memory coverage.
    //

    if (!FindBestGraph(Graph)) {
        goto End;
    }

    //
    // We're not calculating memory coverage yet, so skip this next bit.
    //

    goto End;

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

//
// Begin GraphImpl1.cu
//


EXTERN_C
FORCEINLINE
DEVICE
EDGE
AbsoluteEdge(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge,
    _In_ ULONG Index
    )
{
    ULONG AbsEdge;
    ULONG MaskedEdge;

    MaskedEdge = Edge & Graph->EdgeMask;

    AbsEdge = (MaskedEdge + (Index * Graph->NumberOfEdges));
    return AbsEdge;
}

EXTERN_C
FORCEINLINE
DEVICE
BOOLEAN
IsDeletedEdge(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge
    )
{
    ULONG ThreadId;

    ThreadId = Graph->Deleted[Edge];
    return (ThreadId != NO_THREAD_ID);
}

#define AtomicCompareAndSwapThreadId(Name, Index)                       \
    (NO_THREAD_ID == atomicCAS((unsigned int *)&Graph->##Name##[Index], \
                               NO_THREAD_ID,                            \
                               GlobalThreadIndex()))

EXTERN_C
FORCEINLINE
DEVICE
BOOLEAN
TryRegisterEdgeDeletion(
    _In_ PGRAPH Graph,
    _In_ EDGE Edge
    )
{
    LONG Old;
    LONG OrderIndex;

    if (!AtomicCompareAndSwapThreadId(Deleted, Edge)) {

        //
        // Some other thread has deleted this edge.
        //

        printf("[%d]: other thread %d deleted edge %u\n",
               GlobalThreadIndex(),
               Graph->Deleted[Edge],
               Edge);

        return FALSE;
    }

    printf("[%d]: we (%d) deleted edge %u\n",
           GlobalThreadIndex(),
           Graph->Deleted[Edge],
           Edge);

    //
    // We were the ones to delete this thread.  Obtain the order index for this
    // deletion, and save the edge in the order array.
    //

//#undef BREAKPOINT
//#define BREAKPOINT()

    if (Graph->OrderIndex <= 0) {
        printf("[%d]: Graph->OrderIndex <= 0: %d\n",
               GlobalThreadIndex(),
               Graph->OrderIndex);
        //BREAKPOINT();
    }
    Old = atomicSub((LONG *)&Graph->OrderIndex, 1);
    if (Old <= 0) {
        printf("[%d]: Old <= 0: %d\n",
               GlobalThreadIndex(),
               Old);
        //BREAKPOINT();
    }
    OrderIndex = Old - 1;
    if (OrderIndex < 0) {
        printf("[%d]: OrderIndex < 0: %d\n",
               GlobalThreadIndex(),
               OrderIndex);
        //BREAKPOINT();
    }
    //ASSERT((LONG)OrderIndex >= 0);
    if (OrderIndex >= 0) {
        printf("[%d]: Registering order deletion %d for edge %u.\n",
               GlobalThreadIndex(),
               OrderIndex,
               Edge);
        Graph->Order[OrderIndex] = Edge;
        return TRUE;
    }

    return FALSE;
}


EXTERN_C
DEVICE
BOOLEAN
GraphFindDegree1Edge(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex,
    _Out_ PEDGE EdgePointer
    )
/*++

Routine Description:

    This routine determines if a vertex has degree 1 within the graph, and if
    so, returns the edge associated with it.

Arguments:

    Graph - Supplies a pointer to the graph.

    Vertex - Supplies the vertex for which the degree 1 test is made.

    EdgePointer - Supplies the address of a variable that receives the EDGE
        owning this vertex if it degree 1.

Return Value:

    TRUE if the vertex has degree 1, FALSE otherwise.  EdgePointer will be
    updated if TRUE is returned.

    N.B. Actually, in the CHM implementation, they seem to update the edge
         regardless if it was a degree 1 connection.  I guess we should mirror
         that behavior now too.

--*/
{
    EDGE Edge;
    EDGE AbsEdge;
    BOOLEAN Found = FALSE;

    //
    // Get the edge for this vertex.
    //

    Edge = Graph->First[Vertex];

    //
    // If edge is empty, we're done.
    //

    if (IsEmpty(Edge)) {
        return FALSE;
    }

    AbsEdge = AbsoluteEdge(Graph, Edge, 0);

    //
    // AbsEdge should always be less than or equal to Edge here.
    //

    //ASSERT(AbsEdge <= Edge);

    //
    // If the edge has not been deleted, capture it.
    //

    if (!IsDeletedEdge(Graph, AbsEdge)) {
        Found = TRUE;
        *EdgePointer = Edge;
    }

    //
    // Determine if this is a degree 1 connection.
    //

    while (TRUE) {

        //
        // Load the next edge.
        //

        Edge = Graph->Next[Edge];

        if (IsEmpty(Edge)) {
            break;
        }

        //
        // Obtain the absolute edge for this edge.
        //

        AbsEdge = AbsoluteEdge(Graph, Edge, 0);

        //
        // If we've already deleted this edge, we can skip it and look at the
        // next edge in the graph.
        //

        if (IsDeletedEdge(Graph, AbsEdge)) {
            continue;
        }

        if (Found) {

            //
            // If we've already found an edge by this point, we're not 1 degree.
            //

            return FALSE;
        }

        //
        // We've found the first edge.
        //

        *EdgePointer = Edge;
        Found = TRUE;
    }

    return Found;
}

EXTERN_C
DEVICE
VOID
GraphCyclicDeleteEdge(
    _In_ PGRAPH Graph,
    _In_ VERTEX Vertex
    )
/*++

Routine Description:

    This routine deletes edges from a graph connected by vertices of degree 1.

Arguments:

    Graph - Supplies a pointer to the graph for which the edge is to be deleted.

    Vertex - Supplies the vertex for which the initial edge is obtained.

Return Value:

    None.

    N.B. If an edge is deleted, its corresponding bit will be set in the bitmap
         Graph->DeletedEdges.

--*/
{
    EDGE Edge = 0;
    EDGE AbsEdge;
    VERTEX Vertex1;
    VERTEX Vertex2;
    BOOLEAN IsDegree1;

    //
    // Determine if the vertex has a degree of 1, and if so, obtain the edge.
    //

//Restart:

    IsDegree1 = GraphFindDegree1Edge(Graph, Vertex, &Edge);

    //
    // If this isn't a degree 1 edge, there's nothing left to do.
    //

    if (!IsDegree1) {
        return;
    }

    //
    // We've found an edge of degree 1 to delete.
    //

    Vertex1 = Vertex;
    Vertex2 = 0;

    while (TRUE) {

        //
        // Obtain the absolute edge and register it as deleted.
        //

        AbsEdge = AbsoluteEdge(Graph, Edge, 0);

        //
        // Invariant check: Edge should always be greater than or equal to
        // AbsEdge here.
        //

        if (Edge < AbsEdge) {
            BREAKPOINT();
        }

        //ASSERT(Edge >= AbsEdge);

        //
        // Attempt to register deletion of this edge by our global thread ID.
        //

        if (!TryRegisterEdgeDeletion(Graph, Edge)) {
            NOTHING; //goto Restart;
        }

        //
        // Find the other vertex the edge is connecting.
        //

        Vertex2 = Graph->Edges[AbsEdge];

        if (Vertex2 == Vertex1) {

            //
            // We had the first vertex; get the second one.
            //

            AbsEdge = AbsoluteEdge(Graph, Edge, 1);
            Vertex2 = Graph->Edges[AbsEdge];
        }

        //
        // If the second vertex is empty, break.
        //

        if (IsEmpty(Vertex2)) {
            break;
        }

        //
        // Determine if the other vertex is degree 1.
        //

        IsDegree1 = GraphFindDegree1Edge(Graph, Vertex2, &Edge);

        if (!IsDegree1) {

            //
            // Other vertex isn't degree 1, we can stop the search.
            //

            break;
        }

        //
        // This vertex is also degree 1, so continue the deletion.
        //

        Vertex1 = Vertex2;
    }
}

//
// End GraphImpl1.cu

KERNEL
VOID
GraphCuIsAcyclicKernel(
    _In_ PGRAPH Graph
    )
{
    //ULONG Attempt = 0;
    VERTEX Vertex;

    if (GlobalThreadIndex() == 0) {
        printf("[%d]: Before Graph->OrderIndex: %d\n",
               GlobalThreadIndex(),
               Graph->OrderIndex);
    }

    FOR_EACH_1D(Vertex, Graph->NumberOfVertices) {
        GraphCyclicDeleteEdge(Graph, Vertex);
    }

    __syncthreads();

    if (GlobalThreadIndex() == 0) {
        printf("[%d]: After Graph->OrderIndex: %d\n", GlobalThreadIndex(), Graph->OrderIndex);
    }

#if 0
    __syncthreads();

    if (GlobalThreadIndex() == 0) {
        printf("[%d]: Graph->OrderIndex: %d\n", GlobalThreadIndex(), Graph->OrderIndex);
        if (Graph->OrderIndex != 0) {
            Graph->CuIsAcyclicResult = PH_E_GRAPH_CYCLIC_FAILURE;
        }
    }
#endif

}


EXTERN_C
KERNEL
VOID
GraphCuAssignKernel(
    _In_ PGRAPH Graph
    )
/*++

Routine Description:

    This routine is called after a graph has determined to be acyclic.  It is
    responsible for walking the graph and assigning values to edges in order to
    complete the perfect hash solution.

Arguments:

    Graph - Supplies a pointer to the graph to operate on.

Return Value:

    None.

--*/
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
    // Launch the hash kernel.
    //

    Graph->CuHashKeysResult = S_OK;
    Graph->CuIsAcyclicResult = S_OK;

    SharedMemoryInBytes = 0;

    HashAllMultiplyShiftRKernel<<<
        BlocksPerGrid,
        ThreadsPerBlock,
        SharedMemoryInBytes,
        Streams->Solve
    >>>(
        Keys,
        Graph->NumberOfEdges,
        NumberOfKeys,
        Graph->First,
        Graph->Next,
        Graph->Edges,
        Graph->VertexMask,
        Graph->Seeds,
        &Graph->CuHashKeysResult
    );

    CuResult = cudaDeviceSynchronize();
    CU_CHECK(CuResult, cudaDeviceSynchronize);

    Result = Graph->CuHashKeysResult;
    printf("HashKeys: %x\n", Result);
    if (FAILED(Result)) {
        if (Result == PH_E_GRAPH_VERTEX_COLLISION_FAILURE) {
            Graph->CuVertexCollisionFailures++;
            goto Failed;
        }
        PH_ERROR(GraphCuSolve_AddKeys, Result);
        Result = PH_S_STOP_GRAPH_SOLVING;
        goto End;
    }

    Graph->CuNoVertexCollisionFailures++;

    printf("Before IsAcyclicKernel(): Graph->OrderIndex: %d\n", Graph->OrderIndex);

    GraphCuIsAcyclicKernel<<<
        BlocksPerGrid,
        ThreadsPerBlock,
        SharedMemoryInBytes,
        Streams->IsAcyclic
    >>>(Graph);

    CuResult = cudaDeviceSynchronize();
    CU_CHECK(CuResult, cudaDeviceSynchronize);

    printf("After IsAcyclicKernel(): Graph->OrderIndex: %d\n", Graph->OrderIndex);
    Result = Graph->CuIsAcyclicResult;
    printf("IsAcyclic: %x\n", Result);
    if (FAILED(Result)) {
        if (Result == PH_E_GRAPH_CYCLIC_FAILURE) {
            Graph->CuCyclicGraphFailures++;
            goto Failed;
        }
        PH_ERROR(GraphCuSolve_IsAcyclic, Result);
        Result = PH_S_STOP_GRAPH_SOLVING;
        goto End;
    }

    //
    // We created an acyclic graph.
    //

    Graph->CuFinishedCount++;

    Result = PH_S_STOP_GRAPH_SOLVING;
    goto End;

    //
    // Launch the assignment kernel.
    //

    SharedMemoryInBytes = 0;

    GraphCuAssignKernel<<<
        BlocksPerGrid,
        ThreadsPerBlock,
        SharedMemoryInBytes,
        Streams->Assign
    >>>(Graph);

    //
    // If we're in "first graph wins" mode and we reach this point, optionally
    // calculate coverage and then finish up.
    //

    if (FirstSolvedGraphWins(Graph)) {
        if (WantsAssignedMemoryCoverage(Graph)) {
            GraphCuCalculateAssignedMemoryCoverage(Graph);
        }
        Result = PH_S_STOP_GRAPH_SOLVING;
        goto End;
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
    PCU_STREAM Stream;
    PCU_STREAM FirstStream;
    PCU_STREAM LastStream;
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
    FirstStream = &Streams->FirstStream;
    LastStream = &Streams->LastStream;
    for (Stream = FirstStream; Stream <= LastStream; Stream++) {
        CREATE_STREAM(Stream);
    }

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
        //printf("GraphCuReset() result: %x\n", Result);
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
