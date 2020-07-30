/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    GraphCu.c

Abstract:

    This module implements a CUDA version of the original GRAPH module.

--*/

#include "stdafx.h"

//
// COM scaffolding routines for initialization and rundown.
//

GRAPH_INITIALIZE GraphCuInitialize;

_Use_decl_annotations_
HRESULT
GraphCuInitialize(
    PGRAPH Graph
    )
/*++

Routine Description:

    Initializes a graph structure.  This is a relatively simple method that
    just primes the COM scaffolding.

Arguments:

    Graph - Supplies a pointer to a GRAPH structure for which initialization
        is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - Graph is NULL.

    E_UNEXPECTED - All other errors.

--*/
{
    HRESULT Result = S_OK;

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    Graph->SizeOfStruct = sizeof(*Graph);

    //
    // Create Rtl and Allocator components.
    //

    Result = Graph->Vtbl->CreateInstance(Graph,
                                         NULL,
                                         &IID_PERFECT_HASH_RTL,
                                         &Graph->Rtl);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Graph->Vtbl->CreateInstance(Graph,
                                         NULL,
                                         &IID_PERFECT_HASH_ALLOCATOR,
                                         &Graph->Allocator);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // Set the IsCuGraph flag indicating we're a CUDA graph.
    //

    Graph->Flags.IsCuGraph = TRUE;

    //
    // We're done!  Indicate success and finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}


GRAPH_RUNDOWN GraphCuRundown;

_Use_decl_annotations_
VOID
GraphCuRundown(
    PGRAPH Graph
    )
/*++

Routine Description:

    Release all resources associated with a graph.

Arguments:

    Graph - Supplies a pointer to a GRAPH structure for which rundown is to
        be performed.

Return Value:

    None.

--*/
{
    //
    // Sanity check structure size.
    //

    ASSERT(Graph->SizeOfStruct == sizeof(*Graph));

    //
    // Release applicable COM references.
    //

    RELEASE(Graph->Rtl);
    RELEASE(Graph->Allocator);

    return;
}

//
// Main interface entry points.
//

GRAPH_SET_INFO GraphCuSetInfo;

_Use_decl_annotations_
HRESULT
GraphCuSetInfo(
    PGRAPH Graph,
    PGRAPH_INFO Info
    )
/*++

Routine Description:

    Registers information about a graph with an individual graph instance.
    As table resizing isn't supported with GPU graphs, this routine will only
    ever be called once per graph.

Arguments:

    Graph - Supplies a pointer to the graph instance.

    Info - Supplies a pointer to the graph info instance.

Return Value:

    S_OK - Success.

    E_POINTER - Graph or Info were NULL.

--*/
{
    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Info)) {
        return E_POINTER;
    }

    Graph->Info = Info;
    Graph->Flags.IsInfoSet = TRUE;

    return S_OK;
}

GRAPH_LOAD_INFO GraphCuLoadInfo;

_Use_decl_annotations_
HRESULT
GraphCuLoadInfo(
    PGRAPH Graph
    )
/*++

Routine Description:

    This routine is called by graph solving worker threads prior to attempting
    any solving; it is responsible for initializing the graph structure and
    allocating the necessary buffers required for graph solving, using the sizes
    indicated by the info structure previously set by the main thread via
    SetInfo().

Arguments:

    Graph - Supplies a pointer to the graph instance.

Return Value:

    S_OK - Success.

    E_POINTER - Graph was NULL.

    E_OUTOFMEMORY - Out of memory.

    PH_E_GRAPH_NO_INFO_SET - No graph information has been set for this graph.

    PH_E_GRAPH_INFO_ALREADY_LOADED - Graph information has already been loaded
        for this graph.

--*/
{
    PCU Cu;
    PRTL Rtl;
    HRESULT Result;
    CU_RESULT CuResult;
    PGRAPH DeviceGraph;
    PGRAPH_INFO Info;
    PGRAPH_INFO PrevInfo;
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_CONTEXT Context;
    PPH_CU_SOLVE_CONTEXT SolveContext;
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    CU_MEM_HOST_ALLOC_FLAGS CuMemHostAllocFlags;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Graph)) {
        return E_POINTER;
    }

    if (!IsGraphInfoSet(Graph)) {
        return PH_E_GRAPH_NO_INFO_SET;
    } else if (IsGraphInfoLoaded(Graph)) {
        return PH_E_GRAPH_INFO_ALREADY_LOADED;
    } else {
        Info = Graph->Info;
    }

    //
    // Sanity check the graph size is correct.
    //

    ASSERT(sizeof(*Graph) == Info->SizeOfGraphStruct);
    ASSERT(Graph->SizeOfStruct == sizeof(GRAPH));

    //
    // Initialize aliases.
    //

    Context = Info->Context;
    Cu = Context->Cu;
    Rtl = Context->Rtl;
    PrevInfo = Info->PrevInfo;
    Allocator = Graph->Allocator;
    Table = Context->Table;
    TableInfoOnDisk = Table->TableInfoOnDisk;
    TableCreateFlags.AsULong = Table->TableCreateFlags.AsULong;
    SolveContext = Graph->CuSolveContext;
    DeviceContext = SolveContext->DeviceContext;
    DeviceGraph = SolveContext->DeviceGraph;

    ASSERT(DeviceGraph != NULL);

    //
    // Set the relevant graph fields based on the provided info.
    //

    Graph->Context = Context;
    Graph->NumberOfSeeds = Table->TableInfoOnDisk->NumberOfSeeds;
    Graph->NumberOfKeys = Table->Keys->NumberOfElements.LowPart;

    Graph->ThreadId = GetCurrentThreadId();
    Graph->ThreadAttempt = 0;

    Graph->EdgeMask = Table->IndexMask;
    Graph->VertexMask = Table->HashMask;
    Graph->EdgeModulus = Table->IndexModulus;
    Graph->VertexModulus = Table->HashModulus;
    Graph->MaskFunctionId = Info->Context->MaskFunctionId;

    Graph->Flags.Paranoid = IsParanoid(Table);

    CopyInline(&Graph->Dimensions,
               &Info->Dimensions,
               sizeof(Graph->Dimensions));

    Result = S_OK;

    //
    // Set the CUDA context.
    //

    CuResult = Cu->CtxSetCurrent(DeviceContext->Context);
    CU_CHECK(CuResult, CtxSetCurrent);

    //
    // Allocate arrays.  The VertexPairs, Edges, Next, and First arrays are all
    // local to the device.  The Assigned array is allocated on both the device
    // and the host.
    //

    CuMemHostAllocFlags.AsULong = 0;

#define ALLOC_DEVICE_ARRAY(Name)                                     \
    ASSERT(Graph->##Name == NULL);                                   \
    CuResult = Cu->MemAlloc(                                         \
        (PCU_DEVICE_POINTER)&Graph->##Name,                          \
        (SIZE_T)Info->##Name##SizeInBytes                            \
    );                                                               \
    if (CU_FAILED(CuResult)) {                                       \
        CU_ERROR(GraphCuLoadInfo_MemAlloc_##Name##_Array, CuResult); \
        if (CuResult == CUDA_ERROR_OUT_OF_MEMORY) {                  \
            Result = PH_E_CUDA_OUT_OF_MEMORY;                        \
        } else {                                                     \
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;               \
        }                                                            \
        goto Error;                                                  \
    }

#define ALLOC_HOST_ARRAY(Name)                                   \
    ASSERT(Graph->##Name == NULL);                               \
    CuResult = Cu->MemHostAlloc(                                 \
        (PVOID *)&Graph->##Name,                                 \
        (SIZE_T)Info->##Name##SizeInBytes,                       \
        CuMemHostAllocFlags                                      \
    );                                                           \
    if (CU_FAILED(CuResult)) {                                   \
        CU_ERROR(GraphCuLoadInfo_MemHostAlloc_##Name, CuResult); \
        if (CuResult == CUDA_ERROR_OUT_OF_MEMORY) {              \
            Result = PH_E_CUDA_OUT_OF_MEMORY;                    \
        } else {                                                 \
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;           \
        }                                                        \
        goto Error;                                              \
    }

    ALLOC_DEVICE_ARRAY(VertexPairs);
    ALLOC_DEVICE_ARRAY(Edges);
    ALLOC_DEVICE_ARRAY(Next);
    ALLOC_DEVICE_ARRAY(First);
    ALLOC_DEVICE_ARRAY(AssignedDevice);

    ALLOC_HOST_ARRAY(AssignedHost);

    //
    // Set the bitmap sizes and then allocate the bitmap buffers (which all
    // live on the device).
    //

    Graph->DeletedEdgesBitmap.SizeOfBitMap = Graph->TotalNumberOfEdges;
    Graph->VisitedVerticesBitmap.SizeOfBitMap = Graph->NumberOfVertices;
    Graph->AssignedBitmap.SizeOfBitMap = Graph->NumberOfVertices;
    Graph->IndexBitmap.SizeOfBitMap = Graph->NumberOfVertices;

#define ALLOC_DEVICE_BITMAP_BUFFER(Name)                              \
    ASSERT(Graph->##Name##.Buffer == NULL);                           \
    CuResult = Cu->MemAlloc(                                          \
        (PCU_DEVICE_POINTER)&Graph->##Name##.Buffer,                  \
        (SIZE_T)Info->##Name##BufferSizeInBytes                       \
    );                                                                \
    if (CU_FAILED(CuResult)) {                                        \
        CU_ERROR(GraphCuLoadInfo_MemAlloc_##Name##_Bitmap, CuResult); \
        if (CuResult == CUDA_ERROR_OUT_OF_MEMORY) {                   \
            Result = PH_E_CUDA_OUT_OF_MEMORY;                         \
        } else {                                                      \
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;                \
        }                                                             \
        goto Error;                                                   \
    }

    ALLOC_DEVICE_BITMAP_BUFFER(DeletedEdgesBitmap);
    ALLOC_DEVICE_BITMAP_BUFFER(VisitedVerticesBitmap);
    ALLOC_DEVICE_BITMAP_BUFFER(AssignedBitmap);
    ALLOC_DEVICE_BITMAP_BUFFER(IndexBitmap);

    //
    // Check to see if we're in "first graph wins" mode, and have also been
    // asked to skip memory coverage information.  If so, we can jump straight
    // to the finalization step.
    //

    if (FirstSolvedGraphWinsAndSkipMemoryCoverage(Context)) {
        Graph->Flags.WantsAssignedMemoryCoverage = FALSE;
        goto Finalize;
    }

    if (FirstSolvedGraphWins(Context)) {

        Graph->Flags.WantsAssignedMemoryCoverage = TRUE;

    } else {

        if (DoesBestCoverageTypeRequireKeysSubset(Context->BestCoverageType)) {
            Graph->Flags.WantsAssignedMemoryCoverageForKeysSubset = TRUE;
        } else {
            Graph->Flags.WantsAssignedMemoryCoverage = TRUE;
        }

    }

    //
    // Fill out the assigned memory coverage structure and allocate buffers.
    //

    Coverage = &Graph->AssignedMemoryCoverage;

    Coverage->TotalNumberOfPages = Info->AssignedArrayNumberOfPages;
    Coverage->TotalNumberOfLargePages = Info->AssignedArrayNumberOfLargePages;
    Coverage->TotalNumberOfCacheLines = Info->AssignedArrayNumberOfCacheLines;

#define ALLOC_DEVICE_ASSIGNED_ARRAY(Name)                                    \
    ASSERT(Coverage->##Name == NULL);                                        \
    CuResult = Cu->MemAlloc(                                                 \
        (PCU_DEVICE_POINTER)&Coverage->##Name,                               \
        (SIZE_T)Info->##Name##SizeInBytes                                    \
    );                                                                       \
    if (CU_FAILED(CuResult)) {                                               \
        CU_ERROR(GraphCuLoadInfo_MemAlloc_##Name##_AssignedArray, CuResult); \
        if (CuResult == CUDA_ERROR_OUT_OF_MEMORY) {                          \
            Result = PH_E_CUDA_OUT_OF_MEMORY;                                \
        } else {                                                             \
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;                       \
        }                                                                    \
        goto Error;                                                          \
    }

    ALLOC_DEVICE_ASSIGNED_ARRAY(NumberOfAssignedPerPage);
    ALLOC_DEVICE_ASSIGNED_ARRAY(NumberOfAssignedPerCacheLine);
    ALLOC_DEVICE_ASSIGNED_ARRAY(NumberOfAssignedPerLargePage);

    //
    // If requested, have the host allocate random seed data.
    //

    //
    // Intentional follow-on to Finalize.
    //

Finalize:

    ASSERT(Result == S_OK);

    Graph->Flags.IsInfoLoaded = TRUE;
    Graph->LastLoadedNumberOfVertices = Graph->NumberOfVertices;

    //
    // At this point, we've prepared the host-backed Graph instance, and need to
    // copy the entire structure over to the GPU device.
    //

    CuResult = Cu->MemcpyHtoDAsync((CU_DEVICE_POINTER)DeviceGraph,
                                   Graph,
                                   Graph->SizeOfStruct,
                                   SolveContext->Stream);

    CU_CHECK(CuResult, MemcpyHtoDAsync);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Convert the CUDA out-of-memory error code to the corresponding HRESULT,
    // if applicable.
    //

    if (Result == PH_E_CUDA_OUT_OF_MEMORY) {
        Result = E_OUTOFMEMORY;
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (SUCCEEDED(Result)) {
    }

    return Result;

}

GRAPH_RESET GraphCuReset;

_Use_decl_annotations_
HRESULT
GraphCuReset(
    PGRAPH Graph
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
    UNREFERENCED_PARAMETER(Graph);

    //
    // Increment the attempt counter with GPU attempts, and potentially signal
    // for stop solving.
    //

    //++Graph->ThreadAttempt;
    //Graph->Attempt = InterlockedIncrement64(&Context->Attempts);

    //
    // Clear scalar values.
    //

    return PH_S_CONTINUE_GRAPH_SOLVING;
}

GRAPH_SOLVE GraphCuSolve;

_Use_decl_annotations_
HRESULT
GraphCuSolve(
    PGRAPH Graph,
    PGRAPH *NewGraphPointer
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
    PCU Cu;
    CU_DIM3 Grid = { 1, 1, 1 };
    CU_DIM3 Block = { 1, 1, 1 };
    HRESULT Result;
    HRESULT SolveResult;
    PGRAPH DeviceGraph;
    PGRAPH_INFO Info;
    CU_RESULT CuResult;
    ULONG SharedMemoryInBytes;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_CONTEXT Context;
    PPH_CU_SOLVE_CONTEXT SolveContext;
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    PVOID KernelParams[1];

    UNREFERENCED_PARAMETER(NewGraphPointer);

    //
    // Initialize aliases.
    //

    Info = Graph->Info;
    Context = Info->Context;
    Table = Context->Table;
    SolveContext = Graph->CuSolveContext;
    DeviceContext = SolveContext->DeviceContext;
    DeviceGraph = SolveContext->DeviceGraph;
    Cu = DeviceContext->Cu;

    ASSERT(SolveContext->HostGraph == Graph);

    //
    // Launch the solve kernel.
    //

    SharedMemoryInBytes = 0;
    KernelParams[0] = &DeviceGraph;

    CuResult = Cu->LaunchKernel(DeviceContext->Function,
                                Grid.X,
                                Grid.Y,
                                Grid.Z,
                                Block.X,
                                Block.Y,
                                Block.Z,
                                SharedMemoryInBytes,
                                SolveContext->Stream,
                                KernelParams,
                                NULL);
    CU_CHECK(CuResult, LaunchKernel);

#if 0

    //
    // Copy the device graph back to the host.
    //

    CuResult = Cu->MemcpyDtoHAsync(Graph,
                                   (CU_DEVICE_POINTER)DeviceGraph,
                                   sizeof(GRAPH),
                                   SolveContext->Stream);
    CU_CHECK(CuResult, MemcpyDtoHAsync);

    //
    // Wait for completion.
    //

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);
#endif

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);

    CuResult = Cu->MemcpyDtoH(Graph,
                              (CU_DEVICE_POINTER)DeviceGraph,
                              sizeof(GRAPH));
    CU_CHECK(CuResult, MemcpyDtoH);


    SolveResult = Graph->CuSolveResult;

    if (SolveResult == (HRESULT)1) {
        NOTHING;
    }

    //MAYBE_STOP_GRAPH_SOLVING(Graph);

#if 0

    //
    // We've added all of the vertices to the graph.  Determine if the graph
    // is acyclic.
    //

    if (!IsGraphAcyclic(Graph)) {

        //
        // Failed to create an acyclic graph.
        //

        InterlockedIncrement64(&Context->CyclicGraphFailures);
        goto Failed;
    }

    //
    // We created an acyclic graph.
    //

    //
    // Increment the finished count.  If the context indicates "first solved
    // graph wins", and the value is 1, we're the winning thread, so continue
    // with graph assignment.  Otherwise, just return with the stop graph
    // solving code and let the other thread finish up (i.e. perform the
    // assignment step and then persist the result).
    //

    FinishedCount = InterlockedIncrement64(&Context->FinishedCount);

    if (FirstSolvedGraphWins(Context)) {

        if (FinishedCount != 1) {

            //
            // Some other thread beat us.  Nothing left to do.
            //

            return PH_S_GRAPH_SOLVING_STOPPED;
        }
    }

    //
    // The assignment step was already done by the GPU.
    //

    //GraphAssign(Graph);

    //
    // If we're in "first graph wins" mode and we reach this point, we're the
    // winning thread, so, push the graph onto the finished list head, then
    // submit the relevant finished threadpool work item and return stop graph
    // solving.
    //

    if (FirstSolvedGraphWins(Context)) {
        CONTEXT_END_TIMERS(Solve);
        SetStopSolving(Context);
        if (WantsAssignedMemoryCoverage(Graph)) {
            Graph->Vtbl->CalculateAssignedMemoryCoverage(Graph);
            CopyCoverage(Context->Table->Coverage,
                         &Graph->AssignedMemoryCoverage);
        }
        InsertHeadFinishedWork(Context, &Graph->ListEntry);
        SubmitThreadpoolWork(Context->FinishedWork);
        return PH_S_STOP_GRAPH_SOLVING;
    }

    //
    // If we reach this mode, we're in "find best memory coverage" mode, so,
    // register the solved graph then continue solving.
    //

    ASSERT(FindBestMemoryCoverage(Context));

    //
    // Calculate memory coverage information if applicable.
    //

    if (WantsAssignedMemoryCoverage(Graph)) {
        Graph->Vtbl->CalculateAssignedMemoryCoverage(Graph);
    } else if (WantsAssignedMemoryCoverageForKeysSubset(Graph)) {
        Graph->Vtbl->CalculateAssignedMemoryCoverageForKeysSubset(Graph);
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

    Result = Graph->Vtbl->RegisterSolved(Graph, NewGraphPointer);
#endif

    Result = PH_S_STOP_GRAPH_SOLVING;
    goto End;

Error:

    //
    // Intentional follow-on to End.
    //

End:

    return Result;

#if 0
Failed:

    InterlockedIncrement64(&Context->FailedAttempts);

    return PH_S_CONTINUE_GRAPH_SOLVING;
#endif
}

GRAPH_LOAD_NEW_SEEDS GraphCuLoadNewSeeds;

_Use_decl_annotations_
HRESULT
GraphCuLoadNewSeeds(
    PGRAPH Graph
    )
{
    UNREFERENCED_PARAMETER(Graph);
    return S_OK;
}

GRAPH_REGISTER_SOLVED GraphCuRegisterSolved;

_Use_decl_annotations_
HRESULT
GraphCuRegisterSolved(
    PGRAPH Graph,
    PGRAPH *NewGraphPointer
    )
{
    UNREFERENCED_PARAMETER(Graph);
    UNREFERENCED_PARAMETER(NewGraphPointer);

    return PH_S_GRAPH_SOLVING_STOPPED;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
