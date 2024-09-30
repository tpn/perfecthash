/*++

Copyright (c) 2020-2023 Trent Nelson <trent@trent.me>

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
    //
    // Sanity check structure size.
    //

    ASSERT(Graph->SizeOfStruct == sizeof(*Graph));

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

#if 0
    ALLOC_DEVICE_ARRAY(Next);
    ALLOC_DEVICE_ARRAY(First);
    ALLOC_DEVICE_ARRAY(Edges);
    ALLOC_DEVICE_ARRAY(Order);
    ALLOC_DEVICE_ARRAY(Counts);
    ALLOC_DEVICE_ARRAY(Deleted);
    ALLOC_DEVICE_ARRAY(Visited);
    ALLOC_DEVICE_ARRAY(VertexPairs);
    ALLOC_DEVICE_ARRAY(AssignedDevice);

    ALLOC_HOST_ARRAY(AssignedHost);
#endif

#if 0

    //
    // XXX: temp experiment.
    //

#define ALLOC_SIZED_DEVICE_ARRAY(Name, ElementSize)                  \
    CuResult = Cu->MemAlloc(                                         \
        (PCU_DEVICE_POINTER)&Graph->Name,                            \
        (SIZE_T)(Graph->NumberOfKeys * ElementSize)                  \
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

    ALLOC_SIZED_DEVICE_ARRAY(Vertices1, sizeof(VERTEX));
    ALLOC_SIZED_DEVICE_ARRAY(Vertices2, sizeof(VERTEX));
    ALLOC_SIZED_DEVICE_ARRAY(Vertices1Index, sizeof(ULONG));
    ALLOC_SIZED_DEVICE_ARRAY(Vertices2Index, sizeof(ULONG));
    ALLOC_SIZED_DEVICE_ARRAY(VertexPairsIndex, sizeof(ULONG));
    ALLOC_SIZED_DEVICE_ARRAY(SortedVertexPairs, sizeof(VERTEX_PAIR));

#endif

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

EXTERN_C
VOID
IsGraphAcyclicHost(
    _In_ PGRAPH Graph
    );

VOID
GraphIsAcyclic16Phase1(
    _In_ PGRAPH Graph
    );

HRESULT
GraphIsAcyclic16Phase2(
    _In_ PGRAPH Graph
    );

VOID
GraphIsAcyclic3Phase1(
    _In_ PGRAPH Graph
    );

HRESULT
GraphIsAcyclic3Phase2(
    _In_ PGRAPH Graph
    );

GRAPH_SOLVE GraphCuSolve;

HRESULT
GraphAssign3(
    _In_ PGRAPH Graph
    );

HRESULT
GraphAssign16(
    _In_ PGRAPH Graph
    );

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
    PKEY Keys;
    HRESULT Result;
    HRESULT CpuResult;
    HRESULT CpuAddKeysResult;
    PGRAPH CpuGraph;
    //HRESULT SolveResult;
    ULONG NumberOfKeys;
    ULONG NumberOfVertices;
    PGRAPH DeviceGraph;
    PGRAPH_INFO Info;
    CU_RESULT CuResult;
    ULONG SharedMemoryInBytes;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_CONTEXT Context;
    PDEBUGGER_CONTEXT DebuggerContext;
    PPH_CU_SOLVE_CONTEXT SolveContext;
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    LONG ThreadsPerBlock = PERFECT_HASH_CU_THREADS_PER_BLOCK;
    LONG BlocksPerGridKeys;
    LONG BlocksPerGridVertices;

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
    DebuggerContext = &Graph->Rtl->DebuggerContext;
    CpuGraph = Graph->CpuGraph;

    BlocksPerGridKeys = Graph->NumberOfKeys / ThreadsPerBlock;
    BlocksPerGridVertices = Graph->NumberOfVertices / ThreadsPerBlock;

    ASSERT(SolveContext->HostGraph == Graph);

    NumberOfKeys = Table->Keys->NumberOfKeys.LowPart;
    NumberOfVertices = Graph->NumberOfVertices;
    if (CpuGraph) {
        Keys = (PKEY)Table->Keys->KeyArrayBaseAddress;

        CpuResult = CpuGraph->Vtbl->AddKeys(CpuGraph, NumberOfKeys, Keys);
        if (FAILED(CpuResult)) {
            PH_ERROR(GraphCuSolve_CpuGraphAddKeys, CpuResult);
            CpuAddKeysResult = CpuResult;
            // Let this fall through.
            //Result = CpuResult;
            //goto Error;
        }
    }

    //
    // ByThread.
    //

    CuResult = Cu->MemAllocManaged(
        (PCU_DEVICE_POINTER)&Graph->OrderByThread,
        NumberOfVertices * sizeof(Graph->OrderByThread[0]),
        CU_MEM_ATTACH_GLOBAL
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemAllocManaged_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemcpyHtoD(
        (CU_DEVICE_POINTER)&DeviceGraph->OrderByThread,
        &Graph->OrderByThread,
        sizeof(Graph->OrderByThread)
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemcpyHtoD_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemsetD32Async(
        (PVOID)Graph->OrderByThread,
        ((ULONG)-1),
        NumberOfVertices,
        SolveContext->Stream
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemsetD32Async_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // ByVertex.
    //

    CuResult = Cu->MemAllocManaged(
        (PCU_DEVICE_POINTER)&Graph->OrderByVertex,
        NumberOfVertices * sizeof(Graph->OrderByVertex[0]),
        CU_MEM_ATTACH_GLOBAL
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemAllocManaged_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemcpyHtoD(
        (CU_DEVICE_POINTER)&DeviceGraph->OrderByVertex,
        &Graph->OrderByVertex,
        sizeof(Graph->OrderByVertex)
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemcpyHtoD_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemsetD32Async(
        (PVOID)Graph->OrderByVertex,
        ((ULONG)-1),
        NumberOfVertices,
        SolveContext->Stream
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemsetD32Async_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // ByEdge.
    //

    CuResult = Cu->MemAllocManaged(
        (PCU_DEVICE_POINTER)&Graph->OrderByEdge,
        NumberOfVertices * sizeof(Graph->OrderByEdge[0]),
        CU_MEM_ATTACH_GLOBAL
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemAllocManaged_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemcpyHtoD(
        (CU_DEVICE_POINTER)&DeviceGraph->OrderByEdge,
        &Graph->OrderByEdge,
        sizeof(Graph->OrderByEdge)
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemcpyHtoD_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemsetD32Async(
        (PVOID)Graph->OrderByEdge,
        ((ULONG)-1),
        NumberOfVertices,
        SolveContext->Stream
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemsetD32Async_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // OrderByThreadOrder
    //

    CuResult = Cu->MemAllocManaged(
        (PCU_DEVICE_POINTER)&Graph->OrderByThreadOrder,
        NumberOfVertices * sizeof(Graph->OrderByThreadOrder[0]),
        CU_MEM_ATTACH_GLOBAL
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemAllocManaged_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemcpyHtoD(
        (CU_DEVICE_POINTER)&DeviceGraph->OrderByThreadOrder,
        &Graph->OrderByThreadOrder,
        sizeof(Graph->OrderByThreadOrder)
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemcpyHtoD_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }
    CuResult = Cu->MemsetD32Async(
        (PVOID)Graph->OrderByThreadOrder,
        ((ULONG)-1),
        NumberOfVertices,
        SolveContext->Stream
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemsetD32Async_OrderByThreadOrder, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // SortedVertices3Indices
    //

    CuResult = Cu->MemAllocManaged(
        (PCU_DEVICE_POINTER)&Graph->SortedVertices3Indices,
        NumberOfVertices * sizeof(Graph->SortedVertices3Indices[0]),
        CU_MEM_ATTACH_GLOBAL
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemAllocManaged_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemcpyHtoD(
        (CU_DEVICE_POINTER)&DeviceGraph->SortedVertices3Indices,
        &Graph->SortedVertices3Indices,
        sizeof(Graph->SortedVertices3Indices)
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemcpyHtoD_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }
    CuResult = Cu->MemsetD32Async(
        (PVOID)Graph->SortedVertices3Indices,
        0,
        NumberOfVertices,
        SolveContext->Stream
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemsetD32Async_SortedVertices3Indices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // SortedVertices3
    //

    CuResult = Cu->MemAllocManaged(
        (PCU_DEVICE_POINTER)&Graph->SortedVertices3,
        NumberOfVertices * sizeof(Graph->SortedVertices3[0]),
        CU_MEM_ATTACH_GLOBAL
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemAllocManaged_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemcpyHtoD(
        (CU_DEVICE_POINTER)&DeviceGraph->SortedVertices3,
        &Graph->SortedVertices3,
        sizeof(Graph->SortedVertices3)
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemcpyHtoD_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }
    CuResult = Cu->MemsetD32Async(
        (PVOID)Graph->SortedVertices3,
        0,
        NumberOfVertices,
        SolveContext->Stream
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemsetD32Async_SortedVertices3, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // OrderedVertices
    //

    CuResult = Cu->MemAllocManaged(
        (PCU_DEVICE_POINTER)&Graph->OrderedVertices,
        NumberOfVertices * sizeof(Graph->OrderedVertices[0]),
        CU_MEM_ATTACH_GLOBAL
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemAllocManaged_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemcpyHtoD(
        (CU_DEVICE_POINTER)&DeviceGraph->OrderedVertices,
        &Graph->OrderedVertices,
        sizeof(Graph->OrderedVertices)
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemcpyHtoD_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }
    CuResult = Cu->MemsetD32Async(
        (PVOID)Graph->OrderedVertices,
        0,
        NumberOfVertices,
        SolveContext->Stream
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemsetD32Async_OrderedVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // SortedOrder
    //

    CuResult = Cu->MemAllocManaged(
        (PCU_DEVICE_POINTER)&Graph->SortedOrder,
        (SIZE_T)NumberOfKeys * (IsUsingAssigned16(Graph) ? 2 : 4),
        CU_MEM_ATTACH_GLOBAL
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemAllocManaged_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemcpyHtoD(
        (CU_DEVICE_POINTER)&DeviceGraph->SortedOrder,
        &Graph->SortedOrder,
        sizeof(Graph->SortedOrder)
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemcpyHtoD_SortedOrder, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }
    if (IsUsingAssigned16(Graph)) {
        CuResult = Cu->MemsetD16Async(
            (PVOID)Graph->SortedOrder,
            0,
            NumberOfKeys,
            SolveContext->Stream
        );
    } else {
        CuResult = Cu->MemsetD32Async(
            (PVOID)Graph->SortedOrder,
            0,
            NumberOfKeys,
            SolveContext->Stream
        );
    }
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemsetD1632Async_SortedOrder, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // Indices
    //

    CuResult = Cu->MemAllocManaged(
        (PCU_DEVICE_POINTER)&Graph->Indices,
        (SIZE_T)NumberOfKeys * (IsUsingAssigned16(Graph) ? 2 : 4),
        CU_MEM_ATTACH_GLOBAL
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemAllocManaged_OrderByVertices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    CuResult = Cu->MemcpyHtoD(
        (CU_DEVICE_POINTER)&DeviceGraph->Indices,
        &Graph->Indices,
        sizeof(Graph->Indices)
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemcpyHtoD_Indices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }
    if (IsUsingAssigned16(Graph)) {
        CuResult = Cu->MemsetD16Async(
            (PVOID)Graph->Indices,
            0,
            NumberOfKeys,
            SolveContext->Stream
        );
    } else {
        CuResult = Cu->MemsetD32Async(
            (PVOID)Graph->Indices,
            0,
            NumberOfKeys,
            SolveContext->Stream
        );
    }
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemsetD1632Async_Indices, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }

    //
    // Ensure any async memsets from Reset() have completed.
    //

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);

    //
    // Maybe switch to CUDA GDB if applicable prior to kernel launch.
    //

    MaybeSwitchToCudaGdb(DebuggerContext);
    if (CtrlCPressed) {
        return PH_E_CTRL_C_PRESSED;
    }

    //
    // Launch the solve kernel.
    //

    SharedMemoryInBytes = 0;

#if 0
    //CU_DIM3 Grid = { 1, 1, 1 };
    //CU_DIM3 Block = { 1, 1, 1 };
    PCU_FUNCTION Function;
    PVOID KernelParams[1];
    KernelParams[0] = &DeviceGraph;

    //
    // Initialize the grid to a 1D grid using BlocksPerGrid
    // and ThreadsPerBlock.
    //

    Grid.X = BlocksPerGrid;
    Grid.Y = 1;
    Grid.Z = 1;

    Block.X = ThreadsPerBlock;
    Block.Y = 1;
    Block.Z = 1;

    Function = DeviceContext->HashKeysKernel.Function;
    CuResult = Cu->LaunchKernel(Function,
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
#else

#if 1
    Cu->HashKeysHost(DeviceGraph,
                     BlocksPerGridKeys,
                     ThreadsPerBlock,
                     SharedMemoryInBytes);

#endif

#endif

    //
    // Wait for completion.
    //

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);

    //Cu->IsGraphAcyclicOldHost(DeviceGraph);


    BOOL Success;
    PCHAR Output;
    PCHAR OutputBuffer;
    DWORD CharsWritten = 0;
    ULARGE_INTEGER BytesToWrite = {0,};
    HANDLE OutputHandle;
    ULONGLONG BufferSizeInBytes;

    OutputHandle = Context->OutputHandle;
    OutputBuffer = Context->ConsoleBuffer;
    BufferSizeInBytes = Context->ConsoleBufferSizeInBytes;

    //
    // Dummy write to suppress SAL warning re uninitialized memory being used.
    //

    Output = OutputBuffer;
    *Output = '\0';
    OUTPUT_CHR('\n');

#if 1
    if (CpuGraph) {

#if 0
        //
        // Copy device vertex pairs back to host graph.
        //

        CuResult = Cu->MemcpyDtoH(Graph->VertexPairs,
                                  (CU_DEVICE_POINTER)DeviceGraph->VertexPairs,
                                  Graph->Info->VertexPairsSizeInBytes);
        if (CU_FAILED(CuResult)) {
            CU_ERROR(GraphCuSolve_MemcpyDtoH_VertexPairs, CuResult);
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
            goto Error;
        }
#endif

        //
        // Loop through all of the vertex pairs and compare the GPU results with
        // the CPU results.
        //

        ULONG Index;
        ULONG Vertex1Mismatch = 0;
        ULONG Vertex2Mismatch = 0;
        ULONG Correct = 0;
        ULONG Errors = 0;
        if (IsUsingAssigned16(Graph)) {
            PVERTEX16_PAIR VertexPairsHost;
            PVERTEX16_PAIR VertexPairsDevice;
            VERTEX16_PAIR VertexPairHost;
            VERTEX16_PAIR VertexPairDevice;

            VertexPairsHost = CpuGraph->Vertex16Pairs;
            VertexPairsDevice = Graph->Vertex16Pairs;

            for (Index = 0; Index < NumberOfKeys; Index++) {
                VertexPairHost = *VertexPairsHost++;
                VertexPairDevice = *VertexPairsDevice++;

                if (VertexPairHost.Vertex1 != VertexPairDevice.Vertex1) {
                    Errors++;
                    Vertex1Mismatch++;
                    OUTPUT_CHR('[');
                    OUTPUT_HEX(Index);
                    OUTPUT_CHR(']');
                    OUTPUT_CSTR(" Vertex1 mismatch: ");
                    OUTPUT_HEX(VertexPairHost.Vertex1);
                    OUTPUT_CSTR(" != ");
                    OUTPUT_HEX(VertexPairDevice.Vertex1);
                    OUTPUT_CHR('\n');
                } else if (VertexPairHost.Vertex2 != VertexPairDevice.Vertex2) {
                    Errors++;
                    Vertex2Mismatch++;
                    OUTPUT_CHR('[');
                    OUTPUT_HEX(Index);
                    OUTPUT_CHR(']');
                    OUTPUT_CSTR(" Vertex2 mismatch: ");
                    OUTPUT_HEX(VertexPairHost.Vertex2);
                    OUTPUT_CSTR(" != ");
                    OUTPUT_HEX(VertexPairDevice.Vertex2);
                    OUTPUT_CHR('\n');
                } else {
                    Correct++;
                }

                if ((Errors % 50) == 0) {
                    OUTPUT_FLUSH_CONSOLE();
                }
            }

            OUTPUT_FLUSH_CONSOLE();
        } else {
            PVERTEX_PAIR VertexPairsHost;
            PVERTEX_PAIR VertexPairsDevice;
            VERTEX_PAIR VertexPairHost;
            VERTEX_PAIR VertexPairDevice;

            VertexPairsHost = CpuGraph->VertexPairs;
            VertexPairsDevice = Graph->VertexPairs;

            for (Index = 0; Index < NumberOfKeys; Index++) {
                VertexPairHost = *VertexPairsHost++;
                VertexPairDevice = *VertexPairsDevice++;

                if (VertexPairHost.Vertex1 != VertexPairDevice.Vertex1) {
                    Errors++;
                    Vertex1Mismatch++;
                    OUTPUT_CHR('[');
                    OUTPUT_HEX(Index);
                    OUTPUT_CHR(']');
                    OUTPUT_CSTR(" Vertex1 mismatch: ");
                    OUTPUT_HEX(VertexPairHost.Vertex1);
                    OUTPUT_CSTR(" != ");
                    OUTPUT_HEX(VertexPairDevice.Vertex1);
                    OUTPUT_CHR('\n');
                } else if (VertexPairHost.Vertex2 != VertexPairDevice.Vertex2) {
                    Errors++;
                    Vertex2Mismatch++;
                    OUTPUT_CHR('[');
                    OUTPUT_HEX(Index);
                    OUTPUT_CHR(']');
                    OUTPUT_CSTR(" Vertex2 mismatch: ");
                    OUTPUT_HEX(VertexPairHost.Vertex2);
                    OUTPUT_CSTR(" != ");
                    OUTPUT_HEX(VertexPairDevice.Vertex2);
                    OUTPUT_CHR('\n');
                } else {
                    Correct++;
                }

                if (Errors > 0 && ((Errors % 50) == 0)) {
                    OUTPUT_FLUSH_CONSOLE();
                }
            }

            OUTPUT_FLUSH_CONSOLE();

        }
    }
#endif

#if 0
    CuResult = Cu->MemsetD32Async(
        (PVOID)Graph->Vertices3,
        ((ULONG)-1),
        NumberOfVertices,
        SolveContext->Stream
    );
    if (CU_FAILED(CuResult)) {
        CU_ERROR(GraphCuSolve_MemsetD32Async_ResetVertices3, CuResult);
        Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
        goto Error;
    }
#endif

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);

    Cu->AddHashedKeysHost(DeviceGraph,
                          BlocksPerGridKeys,
                          ThreadsPerBlock,
                          SharedMemoryInBytes);

    //
    // Wait for completion.
    //

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);

    Cu->CountDegreesHost(DeviceGraph,
                         BlocksPerGridKeys,
                         ThreadsPerBlock,
                         SharedMemoryInBytes);

    //
    // Wait for completion.
    //

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);


#if 0
    Cu->GraphPostAddHashedKeysHost(DeviceGraph,
                                   BlocksPerGrid,
                                   ThreadsPerBlock,
                                   SharedMemoryInBytes);

    //
    // Wait for completion.
    //

    CuResult = Cu->StreamSynchronize(SolveContext->Stream2);
    CU_CHECK(CuResult, StreamSynchronize);
#endif

    if (CpuGraph) {

#if 0
        CuResult = Cu->MemcpyDtoH(Graph->VertexPairs,
                                  (CU_DEVICE_POINTER)DeviceGraph->VertexPairs,
                                  Graph->Info->VertexPairsSizeInBytes);
        if (CU_FAILED(CuResult)) {
            CU_ERROR(GraphCuSolve_MemcpyDtoH_VertexPairs, CuResult);
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
            goto Error;
        }
#endif

        //
        // Loop through all of the vertex pairs and compare the GPU results with
        // the CPU results.
        //

        ULONG Index;
        ULONG Ignored = 0;
        ULONG DegreeMismatch = 0;
        ULONG EdgesMismatch = 0;
        ULONG Correct = 0;
        ULONG Errors = 0;
        if (IsUsingAssigned16(Graph)) {
            PVERTEX163 Vertices3Host;
            PVERTEX163 Vertices3Device;
            VERTEX163 Vertex3Host;
            VERTEX163 Vertex3Device;

            Vertices3Host = CpuGraph->Vertices163;
            Vertices3Device = Graph->Vertices163;

            for (Index = 0; Index < NumberOfVertices; Index++) {
                Vertex3Host = *Vertices3Host++;
                Vertex3Device = *Vertices3Device++;

                if (Vertex3Host.Degree == 0 ||
                    Vertex3Device.Degree == 0) {
                    Ignored++;
                    continue;
                }

                if (Vertex3Host.Edges == 0 ||
                    Vertex3Device.Edges == 0) {
                    Ignored++;
                    continue;
                }

                if (Vertex3Host.Degree != Vertex3Device.Degree) {
                    Errors++;
                    DegreeMismatch++;
                    OUTPUT_CHR('[');
                    OUTPUT_HEX(Index);
                    OUTPUT_CHR(']');
                    OUTPUT_CSTR(" Degree mismatch: ");
                    OUTPUT_HEX(Vertex3Host.Degree);
                    OUTPUT_CSTR(" != ");
                    OUTPUT_HEX(Vertex3Device.Degree);
                    OUTPUT_CHR('\n');
                } else if (Vertex3Host.Edges != Vertex3Device.Edges) {
                    Errors++;
                    EdgesMismatch++;
                    OUTPUT_CHR('[');
                    OUTPUT_HEX(Index);
                    OUTPUT_CHR(']');
                    OUTPUT_CSTR(" Edges mismatch: ");
                    OUTPUT_HEX(Vertex3Host.Edges);
                    OUTPUT_CSTR(" != ");
                    OUTPUT_HEX(Vertex3Device.Edges);
                    OUTPUT_CHR('\n');
                } else {
                    Correct++;
                }
                if ((Errors % 50) == 0) {
                    OUTPUT_FLUSH_CONSOLE();
                }
            }

            OUTPUT_FLUSH_CONSOLE();
        }
    }

    //
    // If we were using GDB, then switched to CUDA GDB, switch back to GDB now.
    //

    MaybeSwitchBackToGdb(&DebuggerContext);
    if (CtrlCPressed) {
        return PH_E_CTRL_C_PRESSED;
    }

    //
    // Wait for completion.
    //

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);

    //
    // Copy the return code back.
    //

    CuResult = Cu->MemcpyDtoH(&Graph->CuHashKeysResult,
                              (CU_DEVICE_POINTER)&DeviceGraph->CuHashKeysResult,
                              sizeof(Graph->CuHashKeysResult));
    CU_CHECK(CuResult, MemcpyDtoH_CuHashKeysResult);

    if (Graph->CuHashKeysResult != S_OK) {
        Result = Graph->CuHashKeysResult;
        goto Error;
    }

    if (CpuGraph) {
        if (IsUsingAssigned16(Graph)) {
            GraphIsAcyclic16Phase1(CpuGraph);
        } else {
            GraphIsAcyclic3Phase1(CpuGraph);
        }
    }

    DeviceGraph->OrderIndex = Graph->NumberOfKeys;
    DeviceGraph->DeletedEdgeCount = 0;

    Cu->IsGraphAcyclicPhase1Host(DeviceGraph,
                                 BlocksPerGridVertices,
                                 ThreadsPerBlock,
                                 SharedMemoryInBytes);

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);

    Cu->CountDegreesHost(DeviceGraph,
                         BlocksPerGridKeys,
                         ThreadsPerBlock,
                         SharedMemoryInBytes);

    //
    // Wait for completion.
    //

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);


#if 0
    Cu->IsGraphAcyclicPhase2Host(DeviceGraph,
                                 BlocksPerGrid,
                                 ThreadsPerBlock,
                                 SharedMemoryInBytes);

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);
#endif

#if 0
    SIZE_T ThreadOrderCount;
    SIZE_T VertexOrderCount;
    SIZE_T EdgeOrderCount;
    SIZE_T ThreadOrderCount1;
    SIZE_T VertexOrderCount1;
    SIZE_T EdgeOrderCount1;
    SIZE_T ThreadOrderCount2;
    SIZE_T VertexOrderCount2;
    SIZE_T EdgeOrderCount2;
    SIZE_T ThreadOrderDelta;
    SIZE_T VertexOrderDelta;
    SIZE_T EdgeOrderDelta;
    SIZE_T OrderThreadOrderCount;
    SIZE_T OrderThreadOrderCount1;
    SIZE_T OrderThreadOrderCount2;
    SIZE_T OrderThreadOrderDelta;

    ThreadOrderCount1 = Cu->CountNonEmptyHost(DeviceGraph,
                                              DeviceGraph->OrderByThread,
                                              Graph->NumberOfVertices);

    VertexOrderCount1 = Cu->CountNonEmptyHost(DeviceGraph,
                                              DeviceGraph->OrderByVertex,
                                              Graph->NumberOfVertices);

    EdgeOrderCount1 = Cu->CountNonEmptyHost(DeviceGraph,
                                            DeviceGraph->OrderByEdge,
                                            Graph->NumberOfVertices);

    OrderThreadOrderCount1 = Cu->CountNonEmptyHost(DeviceGraph,
                                                   DeviceGraph->OrderByThreadOrder,
                                                   Graph->NumberOfVertices);

    ThreadOrderCount = ThreadOrderCount1;
    VertexOrderCount = VertexOrderCount1;
    EdgeOrderCount = EdgeOrderCount1;
    OrderThreadOrderCount = OrderThreadOrderCount1;

    if (CpuGraph && 0) {

        //
        // Loop through all of the vertex pairs and compare the GPU results with
        // the CPU results.
        //

        ULONG Index;
        ULONG Errors = 0;
        ULONG DegreeMismatch = 0;
        ULONG EdgesMismatch = 0;
        ULONG Correct = 0;
        if (IsUsingAssigned16(Graph)) {
            BOOLEAN DegreeError;
            BOOLEAN EdgeError;
            PVERTEX163 Vertices3Host;
            PVERTEX163 Vertices3Device;
            VERTEX163 Vertex3Host;
            VERTEX163 Vertex3Device;

            Vertices3Host = CpuGraph->Vertices163;
            Vertices3Device = Graph->Vertices163;

            for (Index = 0; Index < NumberOfVertices; Index++) {
                Vertex3Host = *Vertices3Host++;
                Vertex3Device = *Vertices3Device++;

                DegreeError = FALSE;
                EdgeError = FALSE;
                if (Vertex3Host.Degree != Vertex3Device.Degree) {
                    DegreeMismatch++;
                    DegreeError = TRUE;
                    OUTPUT_CHR('[');
                    OUTPUT_HEX(Index);
                    OUTPUT_CHR(']');
                    OUTPUT_CSTR(" Degree mismatch: ");
                    OUTPUT_INT(Vertex3Host.Degree);
                    OUTPUT_CSTR(" != ");
                    OUTPUT_INT(Vertex3Device.Degree);
                    OUTPUT_CHR('\n');
                }
                if (Vertex3Host.Edges != Vertex3Device.Edges) {
                    EdgesMismatch++;
                    EdgeError = TRUE;
                    OUTPUT_CHR('[');
                    OUTPUT_HEX(Index);
                    OUTPUT_CHR(']');
                    OUTPUT_CSTR(" Edges mismatch:  ");
                    OUTPUT_HEX(Vertex3Host.Edges);
                    OUTPUT_CSTR(" != ");
                    OUTPUT_HEX(Vertex3Device.Edges);
                    OUTPUT_CHR('\n');
                }
                if ((DegreeError == FALSE) && (EdgeError == FALSE)) {
                    Correct++;
                } else {
                    Errors++;
                }
                if ((Errors % 50) == 0) {
                    OUTPUT_FLUSH_CONSOLE();
                }
            }

            OUTPUT_FLUSH_CONSOLE();
        }
    }
#endif

    if (CpuGraph) {
        if (IsUsingAssigned16(Graph)) {
            GraphIsAcyclic16Phase2(CpuGraph);
        } else {
            GraphIsAcyclic3Phase2(CpuGraph);
        }
    }

    LONG DeviceOrderIndex;
    LONG PreviousDeviceOrderIndex = 0;
    LONG DeviceDeletedEdges;
    LONG PreviousDeviceDeletedEdges = 0;
    LONG OrderIndexDelta;
    LONG DeletedEdgeDelta = 0;
    LONG TotalDeletedEdges = 0;

    ULONG Attempts = 0;

    BOOLEAN UsePhase2 = FALSE;
    BOOLEAN IsAcyclic = FALSE;

    while (TRUE) {
        ++Attempts;

        if (UsePhase2) {
            Cu->IsGraphAcyclicPhase2Host(DeviceGraph,
                                         BlocksPerGridVertices,
                                         ThreadsPerBlock,
                                         SharedMemoryInBytes);
        } else {
            Cu->IsGraphAcyclicPhase1Host(DeviceGraph,
                                         BlocksPerGridVertices,
                                         ThreadsPerBlock,
                                         SharedMemoryInBytes);
        }

        CuResult = Cu->StreamSynchronize(SolveContext->Stream);
        CU_CHECK(CuResult, StreamSynchronize);

        Cu->CountDegreesHost(DeviceGraph,
                             BlocksPerGridKeys,
                             ThreadsPerBlock,
                             SharedMemoryInBytes);

        CuResult = Cu->StreamSynchronize(SolveContext->Stream);
        CU_CHECK(CuResult, StreamSynchronize);

#if 0
        CuResult = Cu->MemcpyDtoH(&DeviceOrderIndex,
                                  (CU_DEVICE_POINTER)&DeviceGraph->OrderIndex,
                                  sizeof(DeviceGraph->OrderIndex));
        if (CU_FAILED(CuResult)) {
            CU_ERROR(GraphCuSolve_MemcpyDtoH_OrderIndex, CuResult);
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
            goto Error;
        }
#endif

        DeviceOrderIndex = DeviceGraph->OrderIndex;
        DeviceDeletedEdges = DeviceGraph->DeletedEdgeCount;

        if (DeviceOrderIndex <= 0) {
            OUTPUT_CSTR("DeviceOrderIndex = ");
            OUTPUT_INT(DeviceOrderIndex);
            OUTPUT_CHR('\n');
            OUTPUT_CSTR("DeviceGraph->DeletedEdgeCount = ");
            OUTPUT_INT(DeviceGraph->DeletedEdgeCount);
            OUTPUT_CHR('\n');
            SIZE_T Total = DeviceGraph->DeletedEdgeCount + DeviceOrderIndex;
            OUTPUT_CSTR("Total = ");
            OUTPUT_INT(Total);
            OUTPUT_FLUSH_CONSOLE();
            TotalDeletedEdges = DeviceDeletedEdges + DeviceOrderIndex;
            if (TotalDeletedEdges == (LONG)NumberOfKeys) {
                IsAcyclic = TRUE;
                break;
            }
        }

        if (Attempts == 1) {
            PreviousDeviceOrderIndex = DeviceOrderIndex;
            PreviousDeviceDeletedEdges = DeviceDeletedEdges;
            continue;
        }

        OrderIndexDelta = PreviousDeviceOrderIndex - DeviceOrderIndex;
        DeletedEdgeDelta = DeviceDeletedEdges - PreviousDeviceDeletedEdges;
        ASSERT(OrderIndexDelta >= 0);
        ASSERT(DeletedEdgeDelta >= 0);
        if (OrderIndexDelta == 0 || DeletedEdgeDelta == 0) {
            OUTPUT_CSTR("Attempt = ");
            OUTPUT_INT(Attempts);
            OUTPUT_CSTR(", OrderIndexDelta = ");
            OUTPUT_INT(OrderIndexDelta);
            OUTPUT_CHR('\n');
            OUTPUT_CSTR("X DeviceOrderIndex = ");
            OUTPUT_INT(DeviceOrderIndex);
            OUTPUT_CHR('\n');
            OUTPUT_CSTR("DeviceGraph->DeletedEdgeCount = ");
            OUTPUT_INT(DeviceGraph->DeletedEdgeCount);
            OUTPUT_CHR('\n');
            SIZE_T Total = DeviceGraph->DeletedEdgeCount + DeviceOrderIndex;
            OUTPUT_CSTR("Total = ");
            OUTPUT_INT(Total);
            OUTPUT_FLUSH_CONSOLE();

            //TotalDeletedEdges = DeviceDeletedEdges + DeviceOrderIndex;
            //if (TotalDeletedEdges == (LONG)NumberOfKeys) {
            //    IsAcyclic = TRUE;
            //    break;
            //}

#if 0
            if (UsePhase2 != FALSE) {
                UsePhase2 = FALSE;
                PreviousDeviceOrderIndex = DeviceOrderIndex;
                continue;
            }
#endif
            break;
        } else {
            PreviousDeviceOrderIndex = DeviceOrderIndex;
            PreviousDeviceDeletedEdges = DeviceDeletedEdges;
        }

        continue;

    }

    if (CpuGraph) {

        ULONG Index;
        ULONG Errors = 0;
        ULONG Ignored = 0;
        ULONG DegreeMismatch = 0;
        ULONG EdgesMismatch = 0;
        ULONG Correct = 0;
        if (IsUsingAssigned16(Graph)) {
            BOOLEAN DegreeError;
            BOOLEAN EdgeError;
            PVERTEX163 Vertices3Host;
            PVERTEX163 Vertices3Device;
            VERTEX163 Vertex3Host;
            VERTEX163 Vertex3Device;

            Vertices3Host = CpuGraph->Vertices163;
            Vertices3Device = Graph->Vertices163;

            for (Index = 0; Index < NumberOfVertices; Index++) {
                Vertex3Host = *Vertices3Host++;
                Vertex3Device = *Vertices3Device++;

                DegreeError = FALSE;
                EdgeError = FALSE;

                if (Vertex3Host.Degree == 0 ||
                    Vertex3Device.Degree == 0) {
                    Ignored++;
                    continue;
                }

                if (Vertex3Host.Edges == 0 ||
                    Vertex3Device.Edges == 0) {
                    Ignored++;
                    continue;
                }

                if (Vertex3Host.Degree != Vertex3Device.Degree) {
                    DegreeMismatch++;
                    DegreeError = TRUE;
                    OUTPUT_CHR('[');
                    OUTPUT_HEX(Index);
                    OUTPUT_CHR(']');
                    OUTPUT_CSTR(" Degree mismatch: ");
                    OUTPUT_INT(Vertex3Host.Degree);
                    OUTPUT_CSTR(" != ");
                    OUTPUT_INT(Vertex3Device.Degree);
                    OUTPUT_CHR('\n');
                }
                if (Vertex3Host.Edges != Vertex3Device.Edges) {
                    EdgesMismatch++;
                    EdgeError = TRUE;
                    OUTPUT_CHR('[');
                    OUTPUT_HEX(Index);
                    OUTPUT_CHR(']');
                    OUTPUT_CSTR(" Edges mismatch:  ");
                    OUTPUT_HEX(Vertex3Host.Edges);
                    OUTPUT_CSTR(" != ");
                    OUTPUT_HEX(Vertex3Device.Edges);
                    OUTPUT_CHR('\n');
                }
                if ((DegreeError == FALSE) && (EdgeError == FALSE)) {
                    Correct++;
                } else {
                    Errors++;
                }
                if ((Errors % 50) == 0) {
                    OUTPUT_FLUSH_CONSOLE();
                }
            }

            OUTPUT_FLUSH_CONSOLE();
        }
    }

    if (DeviceGraph->OrderIndex <= 0) {


        DeviceGraph->CuScratch = 1;
        Cu->GraphScratchHost(DeviceGraph,
                             BlocksPerGridVertices,
                             ThreadsPerBlock,
                             SharedMemoryInBytes);

        if (CpuGraph) {
            if (IsUsingAssigned16(Graph)) {
                GraphAssign16(CpuGraph);
            } else {
                GraphAssign3(CpuGraph);
            }
        }

        CuResult = Cu->StreamSynchronize(SolveContext->Stream);
        CU_CHECK(CuResult, StreamSynchronize);

        Cu->GraphAssignHost(DeviceGraph,
                            BlocksPerGridVertices,
                            ThreadsPerBlock,
                            SharedMemoryInBytes);

        CuResult = Cu->StreamSynchronize(SolveContext->Stream);
        CU_CHECK(CuResult, StreamSynchronize);

        if (CpuGraph) {
            CpuGraph->Vtbl->CalculateAssignedMemoryCoverage(CpuGraph);
            CpuGraph->Vtbl->CalculateAssignedMemoryCoverage(DeviceGraph);
        }
    }

    CuResult = Cu->StreamSynchronize(SolveContext->Stream);
    CU_CHECK(CuResult, StreamSynchronize);

#if 0
    CuResult = Cu->MemcpyDtoH(&Graph->CuIsAcyclicResult,
                              (CU_DEVICE_POINTER)&DeviceGraph->CuIsAcyclicResult,
                              sizeof(Graph->CuIsAcyclicResult));
    CU_CHECK(CuResult, MemcpyDtoH_CuIsAcyclicResult);
#endif

    if (Graph->CuIsAcyclicResult != S_OK) {
        Result = Graph->CuIsAcyclicResult;
        goto Error;
    }

#if 0
    //Grid.X = 1;
    //Block.X = 1;
    Function = DeviceContext->IsGraphAcyclicKernel.Function;
    CuResult = Cu->LaunchKernel(Function,
                                32, //Grid.X,
                                Grid.Y,
                                Grid.Z,
                                64, //Block.X,
                                Block.Y,
                                Block.Z,
                                SharedMemoryInBytes,
                                SolveContext->Stream,
                                KernelParams,
                                NULL);
    CU_CHECK(CuResult, LaunchKernel);

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

    if (Graph->CuIsAcyclicResult != S_OK) {
        Result = Graph->CuIsAcyclicResult;
        goto Error;
    }
#endif

    //SolveResult = Graph->CuKernelResult;

    //MAYBE_STOP_GRAPH_SOLVING(Graph);

    //
    // We've added all of the vertices to the graph.  Determine if the graph
    // is acyclic.
    //

#if 0
    if (!IsGraphAcyclic(Graph)) {

        //
        // Failed to create an acyclic graph.
        //

        InterlockedIncrement64(&Context->CyclicGraphFailures);
        goto Failed;
    }
#endif

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

GRAPH_REGISTER_SOLVED GraphCuRegisterSolved;

_Use_decl_annotations_
HRESULT
GraphCuRegisterSolved(
    PGRAPH Graph,
    PGRAPH *NewGraphPointer
    )
{
    return GraphRegisterSolved(Graph, NewGraphPointer);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
