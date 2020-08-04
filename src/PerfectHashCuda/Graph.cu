/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    Graph.cu

Abstract:

    CUDA graph implementation.

--*/

#define EXTERN_C extern "C"
#define EXTERN_C_BEGIN EXTERN_C {
#define EXTERN_C_END }

EXTERN_C_BEGIN
#include <PerfectHashCuda.h>
#include "../PerfectHash/CuDeviceAttributes.h"
#include "../PerfectHash/Graph.h"

#include "Graph.cuh"

#include <cuda.h>
#include <cuda_device_runtime_api.h>
EXTERN_C_END

#include <curand_kernel.h>

#define ASSERT(Condition) \
    if (!(Condition)) {   \
        asm("trap;");     \
    }

#define CU_RESULT cudaError_t
#define CU_STREAM cudaStream_t
#define CU_EVENT cudaEvent_t

#define FindBestGraph(Graph) ((Graph)->Flags.FindBestGraph != FALSE)
#define FirstSolvedGraphWins(Graph) ((Graph)->Flags.FindBestGraph == FALSE)

#define PH_ERROR(Name, Result)           \
    PerfectHashPrintError(#Name,         \
                          __FILE__,      \
                          __LINE__,      \
                          (ULONG)Result)


#define CU_ERROR(Name, CuResult)             \
    PerfectHashPrintCuError(#Name,           \
                            __FILE__,        \
                            __LINE__,        \
                            CuResult)

#define CU_CHECK(CuResult, Name)                   \
    if (CU_FAILED(CuResult)) {                     \
        CU_ERROR(Name, CuResult);                  \
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED; \
        goto Error;                                \
    }

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

//
// Shared memory.
//

extern SHARED ULONG SharedRaw[];

EXTERN_C
GLOBAL
VOID
HashAllMultiplyShiftR(
    _In_reads_(NumberOfKeys) PKEY Keys,
    _In_ ULONG NumberOfKeys,
    _Out_writes_(NumberOfKeys) PVERTEX_PAIR VertexPairs,
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

        Vertex1 = ((Key * SEED1) >> SEED3_BYTE1);
        Vertex2 = ((Key * SEED2) >> SEED3_BYTE2);

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
    __syncthreads();
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
    ULONG ClockRate;
    ULONGLONG Delta;
    ULONGLONG ThisClock;
    PCU_DEVICE_ATTRIBUTES Attributes;

    if (Graph->CuNumberOfSolveLoops++ == 0) {
        Graph->CuStartClock = clock64();
    } else {
        ThisClock = clock64();
        Delta = ThisClock - Graph->CuStartClock;
        Attributes = (PCU_DEVICE_ATTRIBUTES)Graph->CuDeviceAttributes;
        ClockRate = Attributes->ClockRate;

        Graph->CuEndClock = ThisClock;
        Graph->CuCycles = Delta;
        Graph->CuElapsedMilliseconds = Delta / ClockRate;

        printf("Clock rate: %d.\n", ClockRate);
        printf("Elapsed: %u.\n", (ULONG)Graph->CuElapsedMilliseconds);

        if (Graph->CuElapsedMilliseconds >=
            Graph->CuKernelRuntimeTargetInMilliseconds) {
            *Result = PH_S_CU_KERNEL_RUNTIME_TARGET_REACHED;
            return FALSE;
        }
    }

    //
    // XXX: stop solving after one loop iteration.
    //

    //return TRUE;
    return (Graph->CuNumberOfSolveLoops < 2);
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

    E_POINTER - Graph was NULL.

    PH_E_SPARE_GRAPH - Graph is indicated as the spare graph.

--*/
{
    HRESULT Result;

    Graph->Seeds[0] = 2344307159;
    Graph->Seeds[1] = 2331343182;
    Graph->Seeds[2] = 2827;

    Result = S_OK;
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

    Result = PH_S_CONTINUE_GRAPH_SOLVING;
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
    PASSIGNED_MEMORY_COVERAGE Coverage;

    //
    // Initialize aliases.
    //

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

    HashAllMultiplyShiftR<<<
        BlocksPerGrid,
        ThreadsPerBlock,
        SharedMemoryInBytes
    >>>(
        Keys,
        NumberOfKeys,
        Graph->VertexPairs,
        Graph->Seeds,
        &Graph->CuHashKeysResult
    );

    CuResult = cudaDeviceSynchronize();
    CU_CHECK(CuResult, cudaDeviceSynchronize);

    if (FAILED(Result)) {
        if (Result == PH_E_GRAPH_VERTEX_COLLISION_FAILURE) {
            Graph->CuVertexCollisionFailures++;
            goto Failed;
        }
        PH_ERROR(GraphCuSolve_AddKeys, Result);
        Result = PH_S_STOP_GRAPH_SOLVING;
        goto End;
    }

    return PH_S_CONTINUE_GRAPH_SOLVING;

    MAYBE_STOP_GRAPH_SOLVING(Graph);

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
    return PH_S_USE_NEW_GRAPH_FOR_SOLVING;
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
        printf("Invalid dimensions!\n");
        Result = PH_E_CU_KERNEL_SOLVE_LOOP_INVALID_DIMENSIONS;
        goto End;
    }

    printf("Beginning.\n");

    //
    // Begin the solving loop.
    //

    do {

        printf("Loop %d\n", Graph->CuNumberOfSolveLoops);

        if (!GraphCuShouldWeContinueTryingToSolve(Graph, &Result)) {
            printf("Stop solving: %d.\n", Result);
            break;
        }

        Result = GraphCuLoadNewSeeds(Graph);
        printf("LoadNewSeeds: %d.\n", Result);
        if (FAILED(Result)) {
            break;
        }

        Result = GraphCuReset(Graph);
        printf("Reset result: %d.\n", Result);
        if (FAILED(Result)) {
            break;
        } else if (Result != PH_S_CONTINUE_GRAPH_SOLVING) {
            break;
        }

        NewGraph = NULL;
        Result = GraphCuSolve(Graph, &NewGraph);
        printf("GraphCuSolve result: %d\n", Result);
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

#if 0
    //Graph->Seeds[0] = 2344307159;
    //Graph->Seeds[1] = 2331343182;
    Graph->Seeds[0] = 83;
    Graph->Seeds[1] = 5;
    //Graph->Seeds[2] = 2827;
    Graph->Seeds[2] = 0x0101;

    BlocksPerGrid = Graph->CuBlocksPerGrid;
    ThreadsPerBlock = Graph->CuThreadsPerBlock;

    CuResult = cudaStreamCreateWithFlags(&HashKeysStream,
                                         cudaStreamNonBlocking);
    if (CU_FAILED(CuResult)) {
        printf("cuStreamCreate() failed: %x\n", CuResult);
        KernelResult = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto End;
    }

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

    HashAllMultiplyShiftR<<<
        BlocksPerGrid,
        ThreadsPerBlock,
        SharedMemoryInBytes,
        HashKeysStream
    >>>(
        Keys,
        NumberOfKeys,
        Graph->VertexPairs,
        Graph->Seeds,
        &Graph->CuHashKeysResult
    );

    CuResult = cudaDeviceSynchronize();
    if (CU_FAILED(CuResult)) {
        printf("cudaDeviceSynchronize() failed: %x\n", CuResult);
        KernelResult = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto End;
    }

    KernelResult = Graph->CuHashKeysResult;
#endif

End:
    Graph->CuKernelResult = Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
