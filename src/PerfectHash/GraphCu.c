/*++

Copyright (c) 2020-2024 Trent Nelson <trent@trent.me>

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
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TLS_CONTEXT TlsContext;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;

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

    Result = Graph->Vtbl->CreateInstance(Graph,
                                         NULL,
                                         &IID_PERFECT_HASH_RNG,
                                         PPV(&Graph->Rng));

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // Set the IsCuGraph flag indicating we're a CUDA graph.
    //

    Graph->Flags.IsCuGraph = TRUE;

    //
    // Create an activity GUID to track this graph in ETW events.
    //

    Result = Graph->Rtl->Vtbl->GenerateRandomBytes(Graph->Rtl,
                                                   sizeof(Graph->Activity),
                                                   (PBYTE)&Graph->Activity);
    if (FAILED(Result)) {
        goto Error;
    }

    //
    // Load the items we need from the TLS context.
    //

    TlsContext = PerfectHashTlsEnsureContext();
    Table = TlsContext->Table;
    TableCreateFlags.AsULongLong = TlsContext->TableCreateFlags.AsULongLong;

    Graph->HashFunctionId = TlsContext->Table->HashFunctionId;
    Graph->Flags.UsingAssigned16 = (Table->State.UsingAssigned16 != FALSE);

    if (TableCreateFlags.CompareGpuAndCpuGraphs) {
        Result = Graph->Vtbl->CreateInstance(Graph,
                                             NULL,
                                             &IID_PERFECT_HASH_GRAPH,
                                             PPV(&Graph->CpuGraph));
        if (FAILED(Result)) {
            goto Error;
        }
    }

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
    PCU Cu;
    CU_RESULT CuResult;
    PPERFECT_HASH_CONTEXT Context;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PASSIGNED16_MEMORY_COVERAGE Coverage16;

    //
    // Sanity check structure size.
    //

    ASSERT(Graph->SizeOfStruct == sizeof(*Graph));

    //
    // Initialize aliases.
    //

    Context = Graph->Info->Context;
    Cu = Context->Cu;

    //
    // Free applicable arrays.
    //

#define FREE_MANAGED_ARRAY(Name)                                              \
    if (Graph->Name != NULL) {                                                \
        CuResult = Cu->MemFree((CU_DEVICE_POINTER)Graph->Name);               \
        if (CU_FAILED(CuResult)) {                                            \
            CU_ERROR(GraphCuRundown_MemFree_##Name##_ManagedArray, CuResult); \
        } else {                                                              \
            Graph->Name = NULL;                                               \
        }                                                                     \
    }

    FREE_MANAGED_ARRAY(Order);
    FREE_MANAGED_ARRAY(Assigned);
    FREE_MANAGED_ARRAY(Vertices3);
    FREE_MANAGED_ARRAY(VertexPairs);

    FREE_MANAGED_ARRAY(CuVertexLocks);
    FREE_MANAGED_ARRAY(CuEdgeLocks);

    //
    // Free applicable assigned arrays.
    //

#define FREE_MANAGED_ASSIGNED_ARRAY(Coverage_, Name)                       \
    if (Coverage_->Name != NULL) {                                         \
        CuResult = Cu->MemFree((CU_DEVICE_POINTER)Coverage_->Name);        \
        if (CU_FAILED(CuResult)) {                                         \
            CU_ERROR(GraphCuRundown_MemFree_##Name##_ManagedAssignedArray, \
                     CuResult);                                            \
        } else {                                                           \
            Coverage_->Name = NULL;                                        \
        }                                                                  \
    }

    if (!IsUsingAssigned16(Graph)) {

        Coverage = &Graph->AssignedMemoryCoverage;

        FREE_MANAGED_ASSIGNED_ARRAY(Coverage, NumberOfAssignedPerPage);
        FREE_MANAGED_ASSIGNED_ARRAY(Coverage, NumberOfAssignedPerCacheLine);
        FREE_MANAGED_ASSIGNED_ARRAY(Coverage, NumberOfAssignedPerLargePage);

    } else {

        Coverage16 = &Graph->Assigned16MemoryCoverage;

        FREE_MANAGED_ASSIGNED_ARRAY(Coverage16, NumberOfAssignedPerPage);
        FREE_MANAGED_ASSIGNED_ARRAY(Coverage16, NumberOfAssignedPerCacheLine);
        FREE_MANAGED_ASSIGNED_ARRAY(Coverage16, NumberOfAssignedPerLargePage);

    }

    //
    // Release applicable COM references.
    //

    RELEASE(Graph->Rtl);
    RELEASE(Graph->Allocator);
    RELEASE(Graph->Rng);
    RELEASE(Graph->Keys);
    RELEASE(Graph->CpuGraph);

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
    HRESULT Result = S_OK;

    if (!ARGUMENT_PRESENT(Graph)) {
        Result = E_POINTER;
        goto End;
    }

    if (!ARGUMENT_PRESENT(Info)) {
        Result = E_POINTER;
        goto End;
    }

    Graph->Info = Info;
    Graph->Flags.IsInfoSet = TRUE;

    if (Graph->CpuGraph) {
        Result = Graph->CpuGraph->Vtbl->SetInfo(Graph->CpuGraph, Info);
        if (FAILED(Result)) {
            PH_ERROR(GraphCuSetInfo_CpuGraph_SetInfo, Result);
        }
    }

End:
    return Result;
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
    PGRAPH SpareGraph = NULL;
    PGRAPH DeviceGraph;
    PGRAPH_INFO Info;
    PGRAPH_INFO PrevInfo;
    PCWSTR KeysFileName;
    PALLOCATOR Allocator;
    PPERFECT_HASH_KEYS Keys;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_CONTEXT Context;
    PPH_CU_SOLVE_CONTEXT SolveContext;
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PASSIGNED16_MEMORY_COVERAGE Coverage16;
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
    TableCreateFlags.AsULongLong = Table->TableCreateFlags.AsULongLong;
    SolveContext = Graph->CuSolveContext;
    DeviceContext = SolveContext->DeviceContext;

    //
    // Set the relevant graph fields based on the provided info.
    //

    Graph->Context = Context;
    Graph->NumberOfSeeds = Table->TableInfoOnDisk->NumberOfSeeds;
    Graph->HostKeys = (PKEY)Table->Keys->KeyArrayBaseAddress;
    Graph->NumberOfKeys = Table->Keys->NumberOfKeys.LowPart;

    Graph->ThreadId = GetCurrentThreadId();
    Graph->ThreadAttempt = 0;

    Graph->EdgeMask = Table->IndexMask;
    Graph->VertexMask = Table->HashMask;
    Graph->EdgeModulus = Table->IndexModulus;
    Graph->VertexModulus = Table->HashModulus;
    Graph->MaskFunctionId = Info->Context->MaskFunctionId;

    Graph->Flags.Paranoid = IsParanoid(Table);
    Graph->Flags.FindBestGraph = FindBestGraph(Context);
    Graph->Flags.AlwaysRespectCuKernelRuntimeLimit =
        (TableCreateFlags.AlwaysRespectCuKernelRuntimeLimit != FALSE);

    CopyInline(&Graph->Dimensions,
               &Info->Dimensions,
               sizeof(Graph->Dimensions));

    //
    // Wire up the keys pointer and file name buffer pointer.
    //

    Keys = Context->Table->Keys;
    KeysFileName = Keys->File->Path->FileName.Buffer;

    if (Graph->Keys != NULL) {
        ASSERT(Graph->Keys == Keys);
        ASSERT(Graph->KeysFileName == KeysFileName);
    } else {
        Graph->Keys = Context->Table->Keys;
        Graph->Keys->Vtbl->AddRef(Keys);
        Graph->KeysFileName = KeysFileName;
    }

    Result = S_OK;

    //
    // CUDA-specific fields.
    //

    Graph->DeviceKeys = DeviceContext->KeysBaseAddress;

    Graph->CuDeviceAttributes = DeviceContext->DeviceAttributes;
    Graph->CuGraphInfo = (PGRAPH_INFO)DeviceContext->DeviceGraphInfoAddress;

    //
    // Set the CUDA context if we're in "find best graph" mode, or we're not
    // the spare graph.
    //

    if (!FindBestGraph(Context) || !IsSpareGraph(Graph)) {
        CuResult = Cu->CtxSetCurrent(DeviceContext->Context);
        CU_CHECK(CuResult, CtxSetCurrent);
    }

    if (!FindBestGraph(Context)) {
        DeviceGraph = SolveContext->DeviceGraph;
        Graph->CuHostGraph = Graph;
        Graph->CuDeviceGraph = DeviceGraph;
    } else {
        if (!IsSpareGraph(Graph)) {
            DeviceGraph = SolveContext->DeviceGraph;
            Graph->CuHostGraph = Graph;
            Graph->CuDeviceGraph = DeviceGraph;

            SpareGraph = SolveContext->HostSpareGraph;
            Graph->CuHostSpareGraph = SpareGraph;
            Graph->CuDeviceSpareGraph = SolveContext->DeviceSpareGraph;

            ASSERT(SpareGraph->CuHostGraph == NULL);
            ASSERT(SpareGraph->CuHostSpareGraph == NULL);
            ASSERT(SpareGraph->CuDeviceGraph == NULL);
            ASSERT(SpareGraph->CuDeviceSpareGraph == NULL);

            SpareGraph->CuHostGraph = Graph;
            SpareGraph->CuHostSpareGraph = SpareGraph;
            SpareGraph->CuDeviceGraph = DeviceGraph;
            SpareGraph->CuDeviceSpareGraph = SolveContext->DeviceSpareGraph;
        } else {
            DeviceGraph = SolveContext->DeviceSpareGraph;
            ASSERT(Graph->CuHostGraph != NULL);
            ASSERT(Graph->CuHostSpareGraph == Graph);
            ASSERT(Graph->CuDeviceGraph != NULL);
            ASSERT(Graph->CuDeviceSpareGraph != NULL);
        }
    }

    ASSERT(DeviceGraph != NULL);

    CuMemHostAllocFlags.AsULong = 0;

#define ALLOC_DEVICE_ARRAY(Name)                                     \
    ASSERT(Graph->Name == NULL);                                     \
    CuResult = Cu->MemAlloc(                                         \
        (PCU_DEVICE_POINTER)&Graph->Name,                            \
        (SIZE_T)Info->Name##SizeInBytes                              \
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
    ASSERT(Graph->Name == NULL);                                 \
    CuResult = Cu->MemHostAlloc(                                 \
        (PVOID *)&Graph->Name,                                   \
        (SIZE_T)Info->Name##SizeInBytes,                         \
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

#define ALLOC_MANAGED_ARRAY(Name)                                   \
    ASSERT(Graph->Name == NULL);                                    \
    CuResult = Cu->MemAllocManaged(                                 \
        (PCU_DEVICE_POINTER)&Graph->Name,                           \
        (SIZE_T)Info->Name##SizeInBytes,                            \
        CU_MEM_ATTACH_GLOBAL                                        \
    );                                                              \
    if (CU_FAILED(CuResult)) {                                      \
        CU_ERROR(GraphCuLoadInfo_MemAllocManaged_##Name, CuResult); \
        if (CuResult == CUDA_ERROR_OUT_OF_MEMORY) {                 \
            Result = PH_E_CUDA_OUT_OF_MEMORY;                       \
        } else {                                                    \
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;              \
        }                                                           \
        goto Error;                                                 \
    }

    //
    // Allocate arrays.
    //

    ALLOC_MANAGED_ARRAY(Order);
    ALLOC_MANAGED_ARRAY(Assigned);
    ALLOC_MANAGED_ARRAY(Vertices3);
    ALLOC_MANAGED_ARRAY(VertexPairs);

    ALLOC_MANAGED_ARRAY(CuVertexLocks);
    ALLOC_MANAGED_ARRAY(CuEdgeLocks);

    //
    // Set the bitmap sizes and then allocate (or reallocate) the bitmap
    // buffers.
    //

    Graph->DeletedEdgesBitmap.SizeOfBitMap = Graph->TotalNumberOfEdges;
    Graph->VisitedVerticesBitmap.SizeOfBitMap = Graph->NumberOfVertices;
    Graph->AssignedBitmap.SizeOfBitMap = Graph->NumberOfVertices;
    Graph->IndexBitmap.SizeOfBitMap = Graph->NumberOfVertices;

#define ALLOC_HOST_BITMAP_BUFFER(Name)                       \
    if (Info->Name##BufferSizeInBytes > 0) {                 \
        if (!Graph->Name.Buffer) {                           \
            Graph->Name.Buffer = (PULONG)(                   \
                Allocator->Vtbl->Malloc(                     \
                    Allocator,                               \
                    (ULONG_PTR)Info->Name##BufferSizeInBytes \
                )                                            \
            );                                               \
        } else {                                             \
            Graph->Name.Buffer = (PULONG)(                   \
                Allocator->Vtbl->ReAlloc(                    \
                    Allocator,                               \
                    Graph->Name.Buffer,                      \
                    (ULONG_PTR)Info->Name##BufferSizeInBytes \
                )                                            \
            );                                               \
        }                                                    \
        if (!Graph->Name.Buffer) {                           \
            Result = E_OUTOFMEMORY;                          \
            goto Error;                                      \
        }                                                    \
    }

    ALLOC_HOST_BITMAP_BUFFER(DeletedEdgesBitmap);
    ALLOC_HOST_BITMAP_BUFFER(VisitedVerticesBitmap);
    ALLOC_HOST_BITMAP_BUFFER(AssignedBitmap);
    ALLOC_HOST_BITMAP_BUFFER(IndexBitmap);

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

#define ALLOC_DEVICE_ASSIGNED_ARRAY(Name)                                    \
    ASSERT(Coverage->Name == NULL);                                          \
    CuResult = Cu->MemAlloc(                                                 \
        (PCU_DEVICE_POINTER)&Coverage->Name,                                 \
        (SIZE_T)Info->Name##SizeInBytes                                      \
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

#define ALLOC_MANAGED_ASSIGNED_ARRAY(Coverage_, Name)                    \
    ASSERT(Coverage_->Name == NULL);                                     \
    CuResult = Cu->MemAllocManaged(                                      \
        (PCU_DEVICE_POINTER)&Coverage_->Name,                            \
        (SIZE_T)Info->Name##SizeInBytes,                                 \
        CU_MEM_ATTACH_GLOBAL                                             \
    );                                                                   \
    if (CU_FAILED(CuResult)) {                                           \
        CU_ERROR(GraphCuLoadInfo_MemAllocManaged_##Name##_AssignedArray, \
                 CuResult);                                              \
        if (CuResult == CUDA_ERROR_OUT_OF_MEMORY) {                      \
            Result = PH_E_CUDA_OUT_OF_MEMORY;                            \
        } else {                                                         \
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;                   \
        }                                                                \
        goto Error;                                                      \
    }

    //
    // Fill out the assigned memory coverage structure and allocate buffers.
    //

    if (!IsUsingAssigned16(Graph)) {

        Coverage = &Graph->AssignedMemoryCoverage;

        Coverage->TotalNumberOfPages =
            Info->AssignedArrayNumberOfPages;

        Coverage->TotalNumberOfLargePages =
            Info->AssignedArrayNumberOfLargePages;

        Coverage->TotalNumberOfCacheLines =
            Info->AssignedArrayNumberOfCacheLines;

        ALLOC_MANAGED_ASSIGNED_ARRAY(Coverage, NumberOfAssignedPerPage);
        ALLOC_MANAGED_ASSIGNED_ARRAY(Coverage, NumberOfAssignedPerCacheLine);
        ALLOC_MANAGED_ASSIGNED_ARRAY(Coverage, NumberOfAssignedPerLargePage);

    } else {

        Coverage16 = &Graph->Assigned16MemoryCoverage;

        Coverage16->TotalNumberOfPages =
            Info->AssignedArrayNumberOfPages;

        Coverage16->TotalNumberOfLargePages =
            Info->AssignedArrayNumberOfLargePages;

        Coverage16->TotalNumberOfCacheLines =
            Info->AssignedArrayNumberOfCacheLines;

        ALLOC_MANAGED_ASSIGNED_ARRAY(Coverage16, NumberOfAssignedPerPage);
        ALLOC_MANAGED_ASSIGNED_ARRAY(Coverage16, NumberOfAssignedPerCacheLine);
        ALLOC_MANAGED_ASSIGNED_ARRAY(Coverage16, NumberOfAssignedPerLargePage);
    }

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

    if (Graph->CpuGraph) {
        Result = Graph->CpuGraph->Vtbl->LoadInfo(Graph->CpuGraph);
        if (FAILED(Result)) {
            PH_ERROR(GraphCuLoadInfo_CpuGraph_LoadInfo, Result);
            goto Error;
        }
    }

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

    if (SUCCEEDED(Result) &&
        FindBestGraph(Context) &&
        !IsSpareGraph(Graph))
    {

        //
        // Call LoadInfo() against the spare graph, too.
        //

        ASSERT(SolveContext->HostSpareGraph != Graph);
        ASSERT(SolveContext->HostSpareGraph == Graph->CuHostSpareGraph);
        Result = GraphCuLoadInfo(SolveContext->HostSpareGraph);
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
    PCU Cu;
    PRTL Rtl;
    PRNG Rng;
    PGRAPH_INFO Info;
    HRESULT Result;
    CU_RESULT CuResult;
    ULONG TotalNumberOfPages;
    ULONG TotalNumberOfLargePages;
    ULONG TotalNumberOfCacheLines;
    PPERFECT_HASH_CONTEXT Context;
    PPH_CU_SOLVE_CONTEXT SolveContext;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PASSIGNED16_MEMORY_COVERAGE Coverage16;
    PASSIGNED_PAGE_COUNT NumberOfAssignedPerPage;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PASSIGNED_LARGE_PAGE_COUNT NumberOfAssignedPerLargePage;
    PASSIGNED_CACHE_LINE_COUNT NumberOfAssignedPerCacheLine;

    //
    // Initialize aliases.
    //

    Result = S_OK;
    Context = Graph->Context;
    Cu = Context->Cu;
    Info = Graph->Info;
    Rtl = Context->Rtl;
    TableCreateFlags.AsULongLong = Context->Table->TableCreateFlags.AsULongLong;
    SolveContext = Graph->CuSolveContext;

    MAYBE_STOP_GRAPH_SOLVING(Graph);

    ++Graph->ThreadAttempt;

    Graph->Attempt = InterlockedIncrement64(&Context->Attempts);

    //
    // Check if we're capping fixed or maximum attempts; if so, and we've made
    // sufficient attempts, indicate stop solving.
    //

    if (Context->FixedAttempts > 0) {
        if (Graph->Attempt - 1 == Context->FixedAttempts) {
            Context->State.FixedAttemptsReached = TRUE;
            Result = PH_S_FIXED_ATTEMPTS_REACHED;
        }
    } else if (Context->MaxAttempts > 0) {
        if (Graph->Attempt - 1 == Context->MaxAttempts) {
            Context->State.MaxAttemptsReached = TRUE;
            Result = PH_S_MAX_ATTEMPTS_REACHED;
        }
    }

    if (Result != S_OK) {
        CONTEXT_END_TIMERS(Solve);
        SetStopSolving(Context);
        SubmitThreadpoolWork(Context->FinishedWork);
        return Result;
    }

    //
    // Clear the bitmap buffers.
    //

#define ZERO_BITMAP_BUFFER(Name)                             \
    if (Info->Name##BufferSizeInBytes > 0) {                 \
        ASSERT(0 == Info->Name##BufferSizeInBytes -          \
               ((Info->Name##BufferSizeInBytes >> 3) << 3)); \
        ZeroMemory((PDWORD64)Graph->Name.Buffer,             \
                   Info->Name##BufferSizeInBytes);           \
    }

    ZERO_BITMAP_BUFFER(DeletedEdgesBitmap);
    ZERO_BITMAP_BUFFER(VisitedVerticesBitmap);
    ZERO_BITMAP_BUFFER(AssignedBitmap);
    ZERO_BITMAP_BUFFER(IndexBitmap);

    //
    // "Empty" all of the nodes.
    //

#define EMPTY_ARRAY(Name)                              \
    if (Info->Name##SizeInBytes > 0) {                 \
        ASSERT(0 == Info->Name##SizeInBytes -          \
               ((Info->Name##SizeInBytes >> 3) << 3)); \
        FillMemory((PDWORD64)Graph->Name,              \
                   Info->Name##SizeInBytes,            \
                   (BYTE)~0);                          \
    }

    EMPTY_ARRAY(Next);
    EMPTY_ARRAY(First);
    EMPTY_ARRAY(Edges);

    //
    // The Order and Assigned arrays get zeroed.
    //

#define ZERO_MANAGED_ARRAY(Name)                                               \
    if (Info->Name##SizeInBytes > 0) {                                         \
        ASSERT(0 == Info->Name##SizeInBytes -                                  \
               ((Info->Name##SizeInBytes >> 3) << 3));                         \
        CuResult = Cu->MemsetD8Async(                                          \
            (PVOID)Graph->Name,                                                \
            0,                                                                 \
            Info->Name##SizeInBytes,                                           \
            SolveContext->Stream                                               \
        );                                                                     \
        if (CU_FAILED(CuResult)) {                                             \
            CU_ERROR(GraphCuReset_MemsetD8Async_##Name##_ZeroArray, CuResult); \
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;                         \
            goto Error;                                                        \
        }                                                                      \
    }

#define EMPTY_MANAGED_ARRAY(Name)                                               \
    if (Info->Name##SizeInBytes > 0) {                                          \
        ASSERT(0 == Info->Name##SizeInBytes -                                   \
               ((Info->Name##SizeInBytes >> 3) << 3));                          \
        CuResult = Cu->MemsetD8Async(                                           \
            (PVOID)Graph->Name,                                                 \
            ((BYTE)~0),                                                         \
            Info->Name##SizeInBytes,                                            \
            SolveContext->Stream                                                \
        );                                                                      \
        if (CU_FAILED(CuResult)) {                                              \
            CU_ERROR(GraphCuReset_MemsetD8Async_##Name##_EmptyArray, CuResult); \
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;                          \
            goto Error;                                                         \
        }                                                                       \
    }

    ZERO_MANAGED_ARRAY(Order);
    ZERO_MANAGED_ARRAY(Assigned);
    ZERO_MANAGED_ARRAY(Vertices3);
    EMPTY_MANAGED_ARRAY(VertexPairs);

    ZERO_MANAGED_ARRAY(CuVertexLocks);
    ZERO_MANAGED_ARRAY(CuEdgeLocks);

    if (!IsUsingAssigned16(Graph)) {
        Graph->OrderIndex = (LONG)Graph->NumberOfKeys;
        ASSERT(Graph->OrderIndex > 0);
    } else {
        Graph->Order16Index = (SHORT)Graph->NumberOfKeys;
        ASSERT(Graph->Order16Index > 0);
    }

    //
    // Clear any remaining values.
    //

    Graph->Collisions = 0;
    Graph->NumberOfEmptyVertices = 0;
    Graph->DeletedEdgeCount = 0;
    Graph->VisitedVerticesCount = 0;

    Graph->TraversalDepth = 0;
    Graph->TotalTraversals = 0;
    Graph->MaximumTraversalDepth = 0;

    Graph->SolvedTime.AsULongLong = 0;

    Graph->Flags.Shrinking = FALSE;
    Graph->Flags.IsAcyclic = FALSE;

    RESET_GRAPH_COUNTERS();

    //
    // Initialize the RNG.  The subsequence is derived from whatever the base
    // RNG subsequence was (via --RngSubsequence=N, or 0 default), plus the
    // current solving attempt (Graph->Attempt).  This guarantees that the
    // subsequence is a) unique for a given run, and b) always monotonically
    // increasing, which ensures we explore the same PRNG space regardless of
    // concurrency level.
    //
    // (Remember that the PRNG we support, Philox4x3210, is primarily included
    // in order to yield consistent benchmarking environments.  If actual graph
    // solving is being done in order to generate perfect hash tables, then the
    // --Rng=System should always be used, as this will yield much better random
    // numbers (at the expense of varying runtimes, so, not useful if you're
    // benchmarking).)
    //

    Rng = Graph->Rng;
    Result = Rng->Vtbl->InitializePseudo(
        Rng,
        Context->RngId,
        &Context->RngFlags,
        Context->RngSeed,
        Context->RngSubsequence + Graph->Attempt,
        Context->RngOffset
    );

    if (FAILED(Result)) {
        PH_ERROR(GraphLoadInfo_RngInitializePseudo, Result);
        goto Error;
    }

    //
    // Avoid the overhead of resetting the memory coverage if we're in "first
    // graph wins" mode and have been requested to skip memory coverage.
    //

    if (FirstSolvedGraphWinsAndSkipMemoryCoverage(Context)) {
        goto End;
    }

    //
    // Clear the assigned memory coverage counts and arrays.
    //

#define ZERO_MANAGED_ASSIGNED_ARRAY(Coverage_, Name)                           \
        CuResult = Cu->MemsetD8Async((PVOID)Coverage_->Name,                   \
                                     0,                                        \
                                     Info->Name##SizeInBytes,                  \
                                     SolveContext->Stream);                    \
        if (CU_FAILED(CuResult)) {                                             \
            CU_ERROR(GraphCuLoadInfo_MemsetD8Async_##Name##_ZeroAssignedArray, \
                     CuResult);                                                \
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;                         \
            goto Error;                                                        \
        }

    if (!IsUsingAssigned16(Graph)) {
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

        ZeroStructPointer(Coverage);

        //
        // Restore the totals and pointers.
        //

        Coverage->TotalNumberOfPages = TotalNumberOfPages;
        Coverage->TotalNumberOfLargePages = TotalNumberOfLargePages;
        Coverage->TotalNumberOfCacheLines = TotalNumberOfCacheLines;

        Coverage->NumberOfAssignedPerPage = NumberOfAssignedPerPage;
        Coverage->NumberOfAssignedPerLargePage = NumberOfAssignedPerLargePage;
        Coverage->NumberOfAssignedPerCacheLine = NumberOfAssignedPerCacheLine;

        ZERO_MANAGED_ASSIGNED_ARRAY(Coverage, NumberOfAssignedPerPage);
        ZERO_MANAGED_ASSIGNED_ARRAY(Coverage, NumberOfAssignedPerLargePage);
        ZERO_MANAGED_ASSIGNED_ARRAY(Coverage, NumberOfAssignedPerCacheLine);

    } else {

        Coverage16 = &Graph->Assigned16MemoryCoverage;

        //
        // Capture the totals and pointers prior to zeroing the struct.
        //

        TotalNumberOfPages = Coverage16->TotalNumberOfPages;
        TotalNumberOfLargePages = Coverage16->TotalNumberOfLargePages;
        TotalNumberOfCacheLines = Coverage16->TotalNumberOfCacheLines;

        NumberOfAssignedPerPage = Coverage16->NumberOfAssignedPerPage;
        NumberOfAssignedPerLargePage = Coverage16->NumberOfAssignedPerLargePage;
        NumberOfAssignedPerCacheLine = Coverage16->NumberOfAssignedPerCacheLine;

        ZeroStructPointer(Coverage16);

        //
        // Restore the totals and pointers.
        //

        Coverage16->TotalNumberOfPages = TotalNumberOfPages;
        Coverage16->TotalNumberOfLargePages = TotalNumberOfLargePages;
        Coverage16->TotalNumberOfCacheLines = TotalNumberOfCacheLines;

        Coverage16->NumberOfAssignedPerPage = NumberOfAssignedPerPage;
        Coverage16->NumberOfAssignedPerLargePage = NumberOfAssignedPerLargePage;
        Coverage16->NumberOfAssignedPerCacheLine = NumberOfAssignedPerCacheLine;

        ZERO_MANAGED_ASSIGNED_ARRAY(Coverage16, NumberOfAssignedPerPage);
        ZERO_MANAGED_ASSIGNED_ARRAY(Coverage16, NumberOfAssignedPerLargePage);
        ZERO_MANAGED_ASSIGNED_ARRAY(Coverage16, NumberOfAssignedPerCacheLine);
    }

    if (Graph->CpuGraph) {
        Result = Graph->CpuGraph->Vtbl->Reset(Graph->CpuGraph);
        if (FAILED(Result)) {
            PH_ERROR(GraphReset_CpuGraphReset, Result);
            goto End;
        }
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Normalize a successful error code to our code used to communicate that
    // graph solving should continue.
    //

    if (Result == S_OK) {
        Result = PH_S_CONTINUE_GRAPH_SOLVING;
    }

    return Result;
}

GRAPH_LOAD_NEW_SEEDS GraphCuLoadNewSeeds;

_Use_decl_annotations_
HRESULT
GraphCuLoadNewSeeds(
    PGRAPH Graph
    )
{
    PCU Cu;
    PRTL Rtl;
    HRESULT Result;
    CU_RESULT CuResult;
    PGRAPH DeviceGraph;

    Result = GraphLoadNewSeeds(Graph);
    if (FAILED(Result)) {
        PH_ERROR(GraphCuLoadNewSeeds, Result);
        goto End;
    }

    Cu = Graph->CuSolveContext->DeviceContext->Cu;
    DeviceGraph = Graph->CuSolveContext->DeviceGraph;
    CuResult = Cu->MemcpyHtoDAsync((CU_DEVICE_POINTER)&DeviceGraph->Seeds[0],
                                   Graph->Seeds,
                                   Graph->NumberOfSeeds * sizeof(ULONG),
                                   Graph->CuSolveContext->Stream);

    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuLoadNewSeeds_MemcpyHtoDAsync, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto End;
    }

    if (Graph->CpuGraph) {

        //
        // Just copy the GPU seeds explicitly to the CPU graph to ensure they're
        // identical (which they might not be if we were to call LoadNewSeeds()
        // and the user hasn't provided a command line --Seeds argument).
        //

        Rtl = Graph->Rtl;
        CopyMemory(Graph->CpuGraph->Seeds,
                   Graph->Seeds,
                   Graph->NumberOfSeeds * sizeof(ULONG));
    }

End:

    return Result;
}

HRESULT
GraphCuAddKeys(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PKEY Keys
    )
{
    PCU Cu;
    PPERFECT_HASH_CONTEXT Context;

    UNREFERENCED_PARAMETER(NumberOfKeys);
    UNREFERENCED_PARAMETER(Keys);

    Context = Graph->Context;
    Cu = Context->Cu;

    return Cu->AddKeys(Graph,
                       Graph->CuBlocksPerGrid,
                       Graph->CuThreadsPerBlock,
                       Graph->CuSharedMemory);
}

HRESULT
GraphCuIsAcyclic(
    _In_ PGRAPH Graph
    )
{
    PCU Cu;
    PPERFECT_HASH_CONTEXT Context;

    Context = Graph->Context;
    Cu = Context->Cu;

    return Cu->IsAcyclic(Graph,
                         Graph->CuBlocksPerGrid,
                         Graph->CuThreadsPerBlock,
                         Graph->CuSharedMemory);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
