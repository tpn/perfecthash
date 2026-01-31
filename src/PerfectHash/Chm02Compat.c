/*++

Copyright (c) 2020-2026 Trent Nelson <trent@trent.me>

Module Name:

    Chm02Compat.c

Abstract:

    This module is a copy of the Chm01.c, modified to explore the viability of
    CUDA support.  It is an experimental work-in-progress.  This is the Linux
    version.

--*/

//
// For pthread_setname_np().
//

#define _GNU_SOURCE
#include <pthread.h>

#include "stdafx.h"
#include "Chm01.h"
#include "Chm02Private.h"
#include "bsthreadpool.h"

#ifdef PH_WINDOWS
#error This file is not for Windows.
#endif

#ifndef PH_COMPAT
#error PH_COMPAT not defined.
#endif

//
// Spin count for the device context best graph critical section.
//

#define BEST_CU_GRAPH_CS_SPINCOUNT 4000

//
// Forward decls.
//

VOID
GraphCallbackChm02Compat(
    PGRAPH Graph
    );


//
// Main table creation implementation routine for Chm02.
//

_Use_decl_annotations_
HRESULT
CreatePerfectHashTableImplChm02(
    PPERFECT_HASH_TABLE Table
    )
/*++

Routine Description:

    Attempts to create a perfect hash table using the CHM algorithm and a
    2-part random hypergraph.

Arguments:

    Table - Supplies a pointer to a partially-initialized PERFECT_HASH_TABLE
        structure.

Return Value:

    S_OK - Table created successfully.

    The following informational codes will be returned when no internal error
    has occurred, but the table was otherwise unable to be created.  These are
    kept separate to error codes in order to easily discern the difference
    between "this table create failed, but I can proceed with creating a new
    one for a different set of keys" (informational), versus "this table create
    failed due to an internal error and the program should terminate now" (error
    codes).

    PH_I_OUT_OF_MEMORY - The system is out of memory.

    PH_I_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT - The shutdown event
        explicitly set.

    PH_I_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE - The requested number
        of table elements exceeded limits.  If a table resize event occurrs,
        the number of requested table elements is doubled.  If this number
        exceeds MAX_ULONG, this error will be returned.

    PH_I_FAILED_TO_ALLOCATE_MEMORY_FOR_ALL_GRAPHS - Every worker thread was
        unable to allocate sufficient memory to attempt graph solving.  This
        is usually triggered by extremely large key sets and poorly performing
        hash functions that result in numerous table resize events (and thus,
        significant memory consumption).  Overall system memory pressure will
        influence this situation, too.

    PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION - No solution was found
        that met a given criteria.  Not currently used.

    N.B. Result should explicitly be tested against S_OK to verify that a table
         was created successfully.  i.e. `if (SUCCEEDED(Result)) {` won't work
         because the informational codes above are not classed as errors.

    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - Table was NULL.

    E_UNEXPECTED - Catastrophic internal error.

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

    PH_E_TABLE_VERIFICATION_FAILED - The winning perfect hash table solution
        failed internal verification.  The primary cause of this is typically
        when collisions are detected during verification.

    PH_E_INVALID_NUMBER_OF_SEEDS - The number of seeds required for the given
        hash function exceeds the number of seeds available in the on-disk
        table info structure.

--*/
{
    PCU Cu;
    PRNG Rng;
    PRTL Rtl;
    USHORT Index;
    PULONG Keys;
    PGRAPH *Graphs;
    PGRAPH Graph;
    BOOLEAN Silent;
    BOOLEAN Success;
    ULONG Attempt = 0;
    BYTE NumberOfEvents;
    SIZE_T KeysSizeInBytes;
    HRESULT Result = S_OK;
    HRESULT CloseResult = S_OK;
    CU_RESULT CuResult;
    ULONG WaitResult;
    GRAPH_INFO Info;
    PALLOCATOR Allocator;
    HANDLE OutputHandle = NULL;
    PHANDLE Event;
    ULONG Concurrency;
    ULONG CuConcurrency;
    ULONG TotalNumberOfGraphs;
    ULONG NumberOfSolveContexts;
    ULONG NumberOfDeviceContexts;
    PVOID KeysBaseAddress;
    PLIST_ENTRY ListEntry;
    ULONG CloseFileErrorCount = 0;
    ULONG NumberOfSeedsRequired;
    ULONG NumberOfSeedsAvailable;
    GRAPH_INFO_ON_DISK GraphInfo;
    PGRAPH_INFO_ON_DISK GraphInfoOnDisk;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;
    PPERFECT_HASH_CONTEXT Context;
    BOOL WaitForAllEvents = TRUE;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    PPH_CU_DEVICE_CONTEXTS DeviceContexts;
    //PPH_CU_SOLVE_CONTEXT SolveContext;
    PPH_CU_SOLVE_CONTEXTS SolveContexts;
    LARGE_INTEGER EmptyEndOfFile = { 0 };
    PLARGE_INTEGER EndOfFile;
    ULONG NumberOfFileWorkThreads;
    PTHREADPOOL FileWorkThreadpool = NULL;
    PTHREADPOOL GraphThreadpool = NULL;

    HANDLE SaveEvents[NUMBER_OF_SAVE_FILE_EVENTS];
    HANDLE PrepareEvents[NUMBER_OF_PREPARE_FILE_EVENTS];
    PHANDLE SaveEvent = SaveEvents;
    PHANDLE PrepareEvent = PrepareEvents;

#define EXPAND_AS_STACK_VAR(        \
    Verb, VUpper, Name, Upper,      \
    EofType, EofValue,              \
    Suffix, Extension, Stream, Base \
)                                   \
    FILE_WORK_ITEM Verb##Name;

    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_STACK_VAR);
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_STACK_VAR);
    CLOSE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_STACK_VAR);

#define SUCCEEDED_EVENT     WAIT_OBJECT_0+0
#define COMPLETED_EVENT     WAIT_OBJECT_0+1
#define SHUTDOWN_EVENT      WAIT_OBJECT_0+2
#define FAILED_EVENT        WAIT_OBJECT_0+3
#define LOW_MEMORY_EVENT    WAIT_OBJECT_0+4
#define LAST_EVENT          LOW_MEMORY_EVENT
#define NUMBER_OF_EVENTS    LAST_EVENT + 1

    HANDLE Events[NUMBER_OF_EVENTS];

#define EXPAND_AS_ZERO_STACK_VAR(   \
    Verb, VUpper, Name, Upper,      \
    EofType, EofValue,              \
    Suffix, Extension, Stream, Base \
)                                   \
    ZeroStructInline(Verb##Name);

    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ZERO_STACK_VAR);
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ZERO_STACK_VAR);
    CLOSE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ZERO_STACK_VAR);

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    TableCreateFlags.AsULongLong = Table->TableCreateFlags.AsULongLong;
    Silent = (TableCreateFlags.Silent != FALSE);

    //
    // Initialize variables used if we jump to Error early-on.
    //

    Context = Table->Context;
    Allocator = Table->Allocator;
    TotalNumberOfGraphs = 0;

    //
    // Initialize aliases.
    //

    Rtl = Table->Rtl;
    Keys = (PULONG)Table->Keys->KeyArrayBaseAddress;
    MaskFunctionId = Table->MaskFunctionId;
    GraphInfoOnDisk = Context->GraphInfoOnDisk = &GraphInfo;
    TableInfoOnDisk = Table->TableInfoOnDisk = &GraphInfo.TableInfoOnDisk;
    NumberOfFileWorkThreads = GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS);

    ASSERT(
        Context->FinishedWorkList->Vtbl->IsEmpty(Context->FinishedWorkList)
    );

    //
    // Initialize output handle if we're in context table/bulk create mode.
    // We print a dash every time a table resize event occurs to the output
    // handle.
    //

    if (IsContextBulkCreate(Context) || IsContextTableCreate(Context)) {
        OutputHandle = Context->OutputHandle;
        ASSERT(IsValidHandle(OutputHandle));
    }

    //
    // Verify we have sufficient seeds available in our on-disk structure
    // for the given hash function.
    //

    NumberOfSeedsRequired = HashRoutineNumberOfSeeds[Table->HashFunctionId];
    NumberOfSeedsAvailable = ((
        FIELD_OFFSET(GRAPH, LastSeed) -
        FIELD_OFFSET(GRAPH, FirstSeed)
    ) / sizeof(ULONG)) + 1;

    if (NumberOfSeedsAvailable < NumberOfSeedsRequired) {
        return PH_E_INVALID_NUMBER_OF_SEEDS;
    }

    //
    // Prepare the graph info structure.
    //

    Result = PrepareGraphInfoChm02(Table, &Info, NULL);
    if (FAILED(Result)) {
        PH_ERROR(CreatePerfectHashTableImplChm02_PrepareFirstGraphInfo, Result);
        return Result;
    }

    //
    // Perform the CUDA and graph initialization.
    //

    Result = InitializeCudaAndGraphsChm02(Context,
                                          Table->TableCreateParameters);
    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02, Result);
        return Result;
    }

    Cu = Context->Cu;
    ASSERT(Cu != NULL);

    Concurrency = Context->MaximumConcurrency;
    CuConcurrency = Context->CuConcurrency;
    ASSERT(CuConcurrency <= Concurrency);
    ASSERT(CuConcurrency > 0);

    SolveContexts = Context->CuSolveContexts;
    NumberOfSolveContexts = SolveContexts->NumberOfSolveContexts;

    //
    // Copy the keys to all devices.
    //

    Result = CopyKeysToDevices(Context, Table->Keys);
    if (FAILED(Result)) {
        PH_ERROR(CopyKeysToDevices, Result);
        goto Error;
    }

    //
    // Copy the graph info to all devices.
    //

    Result = CopyGraphInfoToDevices(Context, &Info);
    if (FAILED(Result)) {
        PH_ERROR(CopyGraphInfoToDevices, Result);
        goto Error;
    }

    //
    // Initialize event arrays.
    //

    Events[SUCCEEDED_EVENT] = Context->SucceededEvent;
    Events[COMPLETED_EVENT] = Context->CompletedEvent;
    Events[SHUTDOWN_EVENT] = Context->ShutdownEvent;
    Events[FAILED_EVENT] = Context->FailedEvent;
    Events[LOW_MEMORY_EVENT] = Context->LowMemoryEvent;

#define EXPAND_AS_ASSIGN_EVENT(                     \
    Verb, VUpper, Name, Upper,                      \
    EofType, EofValue,                              \
    Suffix, Extension, Stream, Base                 \
)                                                   \
    *Verb##Event++ = Context->Verb##d##Name##Event;

    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_EVENT);
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_EVENT);

    //
    // N.B. We don't explicitly reset all events here like in Chm01 as we're
    //      not supporting table resizes, so the events should always be reset
    //      at this point.
    //

    Context->AlgorithmContext = &Info;

    //
    // Prepare the table output directory if applicable.
    //

    if (!NoFileIo(Table)) {
        Result = PrepareTableOutputDirectory(Table);
        if (FAILED(Result)) {
            PH_ERROR(PrepareTableOutputDirectory, Result);
            goto Error;
        }
    }

    //
    // Submit all of the file preparation work items.
    //

#define EXPAND_AS_SUBMIT_FILE_WORK(                   \
    Verb, VUpper, Name, Upper,                        \
    EofType, EofValue,                                \
    Suffix, Extension, Stream, Base                   \
)                                                     \
    ASSERT(!NoFileIo(Table));                         \
    ZeroStructInline(Verb##Name);                     \
    Verb##Name.FileWorkId = FileWork##Verb##Name##Id; \
    ThreadpoolAddWork(FileWorkThreadpool,             \
                      FileWorkItemCallbackChm01,      \
                      &Verb##Name);

#define SUBMIT_PREPARE_FILE_WORK() \
    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_FILE_WORK)

#define SUBMIT_SAVE_FILE_WORK() \
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_FILE_WORK)

    ASSERT(Context->FileWorkList->Vtbl->IsEmpty(Context->FileWorkList));

    if (!NoFileIo(Table)) {
        NumberOfFileWorkThreads = 1;
        FileWorkThreadpool = ThreadpoolInit(NumberOfFileWorkThreads);
        if (!FileWorkThreadpool) {
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }
        SUBMIT_PREPARE_FILE_WORK();
    }

    //
    // Capture initial cycles as reported by __rdtsc() and the performance
    // counter.  The former is used to report a raw cycle count, the latter
    // is used to convert to microseconds reliably (i.e. unaffected by turbo
    // boosting).
    //

    QueryPerformanceFrequency(&Context->Frequency);

    CONTEXT_START_TIMERS(Solve);

    //
    // Capture the number of milliseconds since boot; this is used to derive
    // elapsed millisecond representations of when best graphs were found when
    // in FindBestGraph mode.
    //

    Context->StartMilliseconds = GetTickCount64();

    //
    // Initialize the graph memory failures counter.  If a graph encounters
    // a memory failure, it performs an interlocked decrement on this counter.
    // If the counter hits zero, FailedEvent is signaled and the context state
    // flag AllGraphsFailedMemoryAllocation is set.
    //

    Context->GraphMemoryFailures = Concurrency;

    //
    // Initialize the number of remaining solver loops and clear the active
    // graph solving loops counter and stop-solving flag.
    //

    Context->RemainingSolverLoops = Concurrency;
    Context->ActiveSolvingLoops = 0;
    ClearStopSolving(Context);

    //
    // Synchronize on each GPU device's stream that was used for the async
    // memcpy of the keys array before submitting any threadpool work.
    //

    DeviceContexts = Context->CuRuntimeContext->CuDeviceContexts;
    for (Index = 0; Index < DeviceContexts->NumberOfDeviceContexts; Index++) {
        DeviceContext = &DeviceContexts->DeviceContexts[Index];
        CuResult = Cu->CtxPushCurrent(DeviceContext->Context);
        CU_CHECK(CuResult, CtxPushCurrent);
        CuResult = Cu->StreamSynchronize(DeviceContext->Stream);
        CU_CHECK(CuResult, StreamSynchronize);
        CuResult = Cu->CtxPopCurrent(NULL);
        CU_CHECK(CuResult, CtxPopCurrent);
    }

    //
    // Initialize the graph threadpool.
    //

    GraphThreadpool = ThreadpoolInit(Concurrency);
    if (!GraphThreadpool) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // For each graph instance, set the graph info, and, if we haven't reached
    // the concurrency limit, append the graph to the context work list and
    // submit threadpool work for it (to begin graph solving).
    //

    ASSERT(Context->MainWorkList->Vtbl->IsEmpty(Context->MainWorkList));
    ASSERT(Context->FinishedWorkList->Vtbl->IsEmpty(Context->FinishedWorkList));

    //
    // The array of all graphs is based at GpuGraphs.  (The CpuGraphs are offset
    // from this array; i.e. a single allocation is performed for the total
    // graph count and then individual graphs are sliced up accordingly.)
    //

    Graphs = Context->GpuGraphs;

    for (Index = 0; Index < Context->TotalNumberOfGraphs; Index++) {

        Graph = Graphs[Index];

        AcquireGraphLockExclusive(Graph);
        Result = Graph->Vtbl->SetInfo(Graph, &Info);
        ReleaseGraphLockExclusive(Graph);

        if (FAILED(Result)) {
            PH_ERROR(GraphSetInfo, Result);
            goto Error;
        }

        Graph->Flags.IsInfoLoaded = FALSE;
        Graph->Context = Context;

        if (Graph->CpuGraph) {
            AcquireGraphLockExclusive(Graph->CpuGraph);
            Result = Graph->CpuGraph->Vtbl->SetInfo(Graph->CpuGraph, &Info);
            ReleaseGraphLockExclusive(Graph->CpuGraph);
            if (FAILED(Result)) {
                PH_ERROR(GraphSetInfo_CpuGraph, Result);
                goto Error;
            }

            Graph->CpuGraph->Flags.IsInfoLoaded = FALSE;
            Graph->CpuGraph->Context = Context;
        }

        if (!IsSpareGraph(Graph)) {
            ThreadpoolAddWork(GraphThreadpool,
                              GraphCallbackChm02Compat,
                              Graph);
        }
    }

    //
    // Wait on the graph threadpool.
    //

    ThreadpoolWait(GraphThreadpool);

    if (CtrlCPressed) {
        Result = PH_E_CTRL_C_PRESSED;
        goto Error;
    }

    //
    // Regardless of the specific event that was signalled, we want to stop
    // solving across all graphs.
    //

    SetStopSolving(Context);

    if (CtrlCPressed) {
        Result = PH_E_CTRL_C_PRESSED;
        goto Error;
    }

    //
    // Ignore the remaining results for now; determine if the graph solving was
    // successful by the finished count of the context.  We'll corroborate that
    // with whatever events have been signaled shortly.
    //

    Success = (Context->FinishedCount > 0);

    if (!Success && !CtrlCPressed) {

        BOOL CancelPending = TRUE;
        BOOLEAN FailedEventSet;
        BOOLEAN ShutdownEventSet;
        BOOLEAN LowMemoryEventSet = FALSE;

        //
        // Obtain the wait results for the failed and shutdown events.
        //

        WaitResult = WaitForSingleObject(Context->FailedEvent, 0);
        FailedEventSet = (WaitResult == WAIT_OBJECT_0);

        WaitResult = WaitForSingleObject(Context->ShutdownEvent, 0);
        ShutdownEventSet = (WaitResult == WAIT_OBJECT_0);

        //
        // If neither the failed or the shutdown event was set, assume the
        // low-memory event was signaled.  We don't explicitly test for this
        // (via WaitForSingleObject()) as its a transient state that may not
        // exist anymore.
        //

        if ((!FailedEventSet && !ShutdownEventSet) ||
            Context->LowMemoryObserved > 0) {
            LowMemoryEventSet = TRUE;
        }

        //
        // Set an appropriate error code based on which event was set.
        //

        if (LowMemoryEventSet) {

            Result = PH_I_LOW_MEMORY;

        } else if (FailedEventSet) {

            if (Context->State.AllGraphsFailedMemoryAllocation == TRUE) {
                Result = PH_I_FAILED_TO_ALLOCATE_MEMORY_FOR_ALL_GRAPHS;
            } else {
                Result = PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION;
            }

        } else if (ShutdownEventSet) {

            Result = PH_I_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT;
        }

        ASSERT(StopSolving(Context));

        if (!NoFileIo(Table)) {
            ThreadpoolWait(FileWorkThreadpool);
        }

        //
        // Sanity check we're not indicating successful table creation.
        //

        ASSERT(Result != S_OK);

        //
        // N.B. Although we're in an erroneous state, we don't jump to Error
        //      here as that sets the shutdown event and waits for the callbacks
        //      to complete; we've just done that, so jump to the End to finish
        //      up processing.
        //

        goto End;
    }

    //
    // Intentional follow-on to FinishedSolution.
    //

FinishedSolution:

    if (CtrlCPressed) {
        Result = PH_E_CTRL_C_PRESSED;
        goto Error;
    }

    ASSERT(Context->FinishedCount > 0);

#if 0
    if (!TableCreateFlags.Quiet) {
        Result = PrintCurrentContextStatsChm01(Context);
        if (FAILED(Result)) {
            PH_ERROR(PrintCurrentContextStatsChm01, Result);
            goto Error;
        }
    }
#endif

    if (FirstSolvedGraphWins(Context)) {

        //
        // Pop the winning graph off the finished list head.
        //

        ListEntry = NULL;

        if (!RemoveHeadFinishedWork(Context, &ListEntry)) {
            Result = PH_E_GUARDED_LIST_EMPTY;
            PH_ERROR(PerfectHashCreateChm01Callback_RemoveFinishedWork, Result);
            goto Error;
        }

        ASSERT(ListEntry);
        Graph = CONTAINING_RECORD(ListEntry, GRAPH, ListEntry);

    } else {

        EnterCriticalSection(&Context->BestGraphCriticalSection);
        Graph = Context->BestGraph;
        Context->BestGraph = NULL;
        LeaveCriticalSection(&Context->BestGraphCriticalSection);

        ASSERT(Graph != NULL);
    }

    //
    // Capture the maximum traversal depth, number of empty vertices, and graph
    // collisions during assignment.  (These are used in the .csv output, hence
    // the duplication between the table and graph structs.)
    //

    Table->MaximumGraphTraversalDepth = Graph->MaximumTraversalDepth;
    Table->NumberOfEmptyVertices = Graph->NumberOfEmptyVertices;
    Table->NumberOfCollisionsDuringAssignment = Graph->Collisions;

    //
    // Capture whether large pages were used for the vertex pairs array.
    //

    Table->Flags.VertexPairsArrayUsesLargePages = (
        Graph->Flags.VertexPairsArrayUsesLargePages
    );

    //
    // Capture whether optimized AVX versions of various functions were used.
    //

    Table->Flags.UsedAvx2HashFunction = Graph->Flags.UsedAvx2HashFunction;
    Table->Flags.UsedAvx512HashFunction = Graph->Flags.UsedAvx512HashFunction;
    Table->Flags.UsedAvx2MemoryCoverageFunction =
        Graph->Flags.UsedAvx2MemoryCoverageFunction;

    //
    // Copy the cycle counters and elapsed microseconds from the winning graph.
    //

    COPY_GRAPH_COUNTERS_FROM_GRAPH_TO_TABLE();

    //
    // Capture RNG details from the winning graph if the RNG used was not the
    // System one.
    //

    if (Context->RngId != PerfectHashRngSystemId) {
        Table->RngSeed = Graph->Rng->Seed;
        Table->RngSubsequence = Graph->Rng->Subsequence;
        Table->RngOffset = Graph->Rng->Offset;

        Rng = Graph->Rng;
        Result = Rng->Vtbl->GetCurrentOffset(Rng, &Table->RngCurrentOffset);
        if (FAILED(Result)) {
            PH_ERROR(CreatePerfectHashTableImplChm02_RngGetCurrentOffset,
                     Result);
            goto Error;
        }
    }

    //
    // Calculate the solve duration, solutions found ratio, and predicted
    // attempts.
    //

    Table->SolveDurationInSeconds = (DOUBLE)(
        ((DOUBLE)Context->SolveElapsedMicroseconds.QuadPart) /
        ((DOUBLE)1e6)
    );

    Table->SolutionsFoundRatio = (DOUBLE)(
        ((DOUBLE)Context->FinishedCount) /
        ((DOUBLE)Context->Attempts)
    );

    Result = CalculatePredictedAttempts(Table->SolutionsFoundRatio,
                                        &Table->PredictedAttempts);
    if (FAILED(Result)) {
        PH_ERROR(CreatePerfectHashTableImplChm02_CalculatePredictedAttempts,
                 Result);
        goto Error;
    }

    //
    // Note this graph as the one solved to the context.  This is used by the
    // save file work callback we dispatch below.
    //

    Context->SolvedContext = Graph;

    //
    // Graphs always pass verification in normal circumstances.  The only time
    // they don't is if there's an internal bug in our code.  So, knowing that
    // the graph is probably correct, we can dispatch the file work required to
    // save it to disk to the file work threadpool whilst we verify it has been
    // solved correctly.
    //

    if (!NoFileIo(Table)) {

        //
        // Dispatch save file work for the table data.
        //

        SUBMIT_SAVE_FILE_WORK();

    } else {

        PGRAPH_INFO_ON_DISK NewGraphInfoOnDisk;

        //
        // Normally, when we're performing file I/O, the save file work callback
        // for the info stream (SaveTableInfoStreamChm01()) is responsible for
        // writing the contents of Table->TableInfoOnDisk (which is currently
        // pointing to our local stack-allocated GraphInfoOnDisk structure) to
        // the info stream's base address, then allocating a new heap-backed
        // structure and copying everything over.  As we're not doing any file
        // I/O, we need to do this manually ourselves at this point.
        //
        // N.B. It might just be easier to allocate the structure from the heap
        //      up-front; something to investigate later perhaps.
        //

        NewGraphInfoOnDisk = (
            Allocator->Vtbl->Calloc(
                Allocator,
                1,
                sizeof(*NewGraphInfoOnDisk)
            )
        );

        if (!NewGraphInfoOnDisk) {
            Result = E_OUTOFMEMORY;
            goto Error;
        }

        CopyMemory(NewGraphInfoOnDisk,
                   GraphInfoOnDisk,
                   sizeof(*NewGraphInfoOnDisk));

        //
        // Sanity check the first seed is not 0.
        //

        if (Graph->FirstSeed == 0) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(CreatePerfectHashTableImplChm02_GraphFirstSeedIs0, Result);
            PH_RAISE(Result);
        }

        //
        // Copy the seed data from the winning graph.
        //

        CopyMemory(&NewGraphInfoOnDisk->TableInfoOnDisk.FirstSeed,
                   &Graph->FirstSeed,
                   Graph->NumberOfSeeds * sizeof(Graph->FirstSeed));

        //
        // Switch the pointers.
        //

        Table->TableInfoOnDisk = &NewGraphInfoOnDisk->TableInfoOnDisk;

        //
        // Update state indicating table info has been heap-allocated.
        //

        Table->State.TableInfoOnDiskWasHeapAllocated = TRUE;

        if (!IsTableCreateOnly(Table)) {

            //
            // Perform the same logic for the table data.  Note that this is
            // a copy-and-paste directly from Chm01FileWorkTableFile.c; the
            // common logic should be abstracted out somewhere.
            //

            PVOID BaseAddress;
            LONGLONG SizeInBytes;
            BOOLEAN LargePagesForTableData;
            PRTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC TryLargePageVirtualAlloc;

            //
            // Allocate and copy the table data to an in-memory copy so that the
            // table can be used after Create() completes successfully.  See the
            // comment in the SaveTableInfoStreamChm01() routine for more
            // information about why this is necessary.
            //

            LargePagesForTableData = (
                Table->TableCreateFlags.TryLargePagesForTableData == TRUE
            );

            SizeInBytes = (
                TableInfoOnDisk->NumberOfTableElements.QuadPart *
                TableInfoOnDisk->AssignedElementSizeInBytes
            );

            TryLargePageVirtualAlloc = Rtl->Vtbl->TryLargePageVirtualAlloc;
            BaseAddress = TryLargePageVirtualAlloc(Rtl,
                                                   NULL,
                                                   SizeInBytes,
                                                   MEM_RESERVE | MEM_COMMIT,
                                                   PAGE_READWRITE,
                                                   &LargePagesForTableData);

            Table->TableDataBaseAddress = BaseAddress;
            Table->TableDataSizeInBytes = SizeInBytes;

            if (!BaseAddress) {
                Result = E_OUTOFMEMORY;
                goto Error;
            }

            //
            // Update state indicating table data has been heap-allocated.
            //

            Table->State.TableDataWasHeapAllocated = TRUE;

            //
            // Update flags with large page result for values array.
            //

            Table->Flags.TableDataUsesLargePages = LargePagesForTableData;

            //
            // Copy the table data over to the newly allocated buffer.
            //

            CopyMemory(Table->TableDataBaseAddress,
                       Graph->Assigned,
                       SizeInBytes);

        }

    }

    //
    // Capture another round of cycles and performance counter values, then
    // continue with verification of the graph.
    //

    CONTEXT_START_TIMERS(Verify);

    Result = Graph->Vtbl->Verify(Graph);

    CONTEXT_END_TIMERS(Verify);

    //
    // Set the verified table event (regardless of whether or not we succeeded
    // in verification).  The save file work will be waiting upon it in order to
    // write the final timing details to the on-disk header.
    //

    if (!SetEvent(Context->VerifiedTableEvent)) {
        SYS_ERROR(SetEvent);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (FAILED(Result)) {
        Result = PH_E_TABLE_VERIFICATION_FAILED;
        goto Error;
    }

    if (NoFileIo(Table)) {
        goto End;
    }

    //
    // Wait on the saved file events before returning.
    //

    ThreadpoolWait(FileWorkThreadpool);

    if (CtrlCPressed) {
        Result = PH_E_CTRL_C_PRESSED;
        goto Error;
    }

    //
    // Check all of the save file work error indicators.
    //

    CHECK_ALL_SAVE_ERRORS();

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    SetEvent(Context->ShutdownEvent);
    ThreadpoolWait(GraphThreadpool);
    if (!NoFileIo(Table)) {
        ThreadpoolWait(FileWorkThreadpool);
    }

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Convert the out-of-memory error codes into our equivalent info codes.
    //

    if (Result == E_OUTOFMEMORY) {
        Result = PH_I_OUT_OF_MEMORY;
    } else if (Result == PH_E_CUDA_OUT_OF_MEMORY) {
        Result = PH_I_CUDA_OUT_OF_MEMORY;
    }

    //
    // If no attempts were made, no file work was submitted, which means we
    // can skip the close logic below and jump straight to releasing graphs.
    // Likewise for no file I/O.
    //

    if (Attempt == 0 || NoFileIo(Table)) {
        goto ReleaseGraphs;
    }

    //
    // Close all files.  If we weren't successful in finding a solution, pass
    // an end-of-file value of 0 to each Close() call, which will delete the
    // file.
    //

    if (Result != S_OK) {
        EndOfFile = &EmptyEndOfFile;
    } else {
        EndOfFile = NULL;
    }

#define EXPAND_AS_SUBMIT_CLOSE_FILE_WORK(               \
    Verb, VUpper, Name, Upper,                          \
    EofType, EofValue,                                  \
    Suffix, Extension, Stream, Base                     \
)                                                       \
    ASSERT(!NoFileIo(Table));                           \
    ZeroStructInline(Verb##Name);                       \
    Verb##Name.FileWorkId = FileWork##Verb##Name##Id;   \
    Verb##Name.EndOfFile = EndOfFile;                   \
    ThreadpoolAddWork(FileWorkThreadpool,               \
                      FileWorkItemCallbackChm01,        \
                      &Verb##Name);

#define SUBMIT_CLOSE_FILE_WORK() \
    CLOSE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_CLOSE_FILE_WORK)

    SUBMIT_CLOSE_FILE_WORK();

    ThreadpoolWait(FileWorkThreadpool);

#define EXPAND_AS_CHECK_CLOSE_ERRORS(                                \
    Verb, VUpper, Name, Upper,                                       \
    EofType, EofValue,                                               \
    Suffix, Extension, Stream, Base                                  \
)                                                                    \
    if (Verb##Name.NumberOfErrors > 0) {                             \
        CloseResult = Verb##Name.LastResult;                         \
        if (CloseResult == S_OK || CloseResult == E_UNEXPECTED) {    \
            CloseResult = PH_E_ERROR_DURING_##VUpper##_##Upper;      \
        }                                                            \
        PH_ERROR(                                                    \
            CreatePerfectHashTableImplChm02_ErrorDuring##Verb##Name, \
            Result                                                   \
        );                                                           \
        CloseFileErrorCount++;                                       \
    }

#define CHECK_ALL_CLOSE_ERRORS() \
    CLOSE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CHECK_CLOSE_ERRORS)

    CHECK_ALL_CLOSE_ERRORS();

    //
    // Just use whatever the last close file error was for our return value if
    // we encountered any errors.
    //

    if (CloseFileErrorCount > 0) {
        Result = CloseResult;
    }

ReleaseGraphs:

    Graphs = Context->Graphs;

    if (Graphs) {

        ULONG ReferenceCount;

        //
        // Walk the array of graph instances and release each one (assuming it
        // is not NULL), then free the array buffer.
        //

        for (Index = 0; Index < TotalNumberOfGraphs; Index++) {

            Graph = Graphs[Index];

            if (Graph != NULL) {

                if (Graph->CpuGraph != NULL) {

                    ReferenceCount = Graph->CpuGraph->Vtbl->Release(
                        Graph->CpuGraph
                    );

                    //
                    // Invariant check: reference count should always be 0 here.
                    //

                    if (ReferenceCount != 0) {
                        Result = PH_E_INVARIANT_CHECK_FAILED;
                        PH_ERROR(CpuGraphReferenceCountNotZero, Result);
                        PH_RAISE(Result);
                    }
                }

                ReferenceCount = Graph->Vtbl->Release(Graph);

                //
                // Invariant check: reference count should always be 0 here.
                //

                if (ReferenceCount != 0) {
                    Result = PH_E_INVARIANT_CHECK_FAILED;
                    PH_ERROR(GraphReferenceCountNotZero, Result);
                    PH_RAISE(Result);
                }

                Graphs[Index] = NULL;
            }
        }

        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&Graphs);
    }

    //
    // Explicitly reset all events before returning.
    //

    Event = (PHANDLE)&Context->FirstEvent;
    NumberOfEvents = GetNumberOfContextEvents(Context);

    for (Index = 0; Index < NumberOfEvents; Index++, Event++) {

        if (!ResetEvent(*Event)) {
            SYS_ERROR(ResetEvent);
            if (Result == S_OK) {
                Result = PH_E_SYSTEM_CALL_FAILED;
            }
        }
    }

    //
    // Release the table's output path if applicable.
    //

    RELEASE(Table->OutputPath);

    //
    // Destroy the threadpools if applicable.
    //

    if (GraphThreadpool) {
        ThreadpoolDestroy(GraphThreadpool);
        GraphThreadpool = NULL;
    }

    if (FileWorkThreadpool) {
        ThreadpoolDestroy(FileWorkThreadpool);
        FileWorkThreadpool = NULL;
    }

    return Result;
}

VOID
GraphSetThreadName (
    _In_ PGRAPH Graph
    )
{
    CHAR Buf[80];
    size_t Length;
    pthread_t Thread;

    Length = snprintf(Buf, ARRAYSIZE(Buf), "Graph-%d", Graph->Index);
    ASSERT(Buf[Length] == '\0');

    Thread = pthread_self();
    pthread_setname_np(Thread, Buf);
}

_Use_decl_annotations_
VOID
GraphCallbackChm02Compat(
    PGRAPH Graph
    )
/*++

Routine Description:

    This routine is the callback entry point for graph solving threads.  It
    will enter an infinite loop attempting to solve the graph; terminating
    only when the graph is solved or we detect another thread has solved it.

Arguments:

    Graph - Supplies a pointer to the graph instance for this callback.

Return Value:

    None.

--*/
{
    HRESULT Result;
    PHANDLE Event;
    ULONG WaitResult;
    PPERFECT_HASH_CONTEXT Context;

    Context = Graph->Context;

    InterlockedIncrement(&Context->ActiveSolvingLoops);

    GraphSetThreadName(Graph);

    Result = Graph->Vtbl->EnterSolvingLoop(Graph);

    if (FAILED(Result)) {

        BOOLEAN PermissibleErrorCode;

        //
        // There are only a few permissible errors at this point.  If a
        // different error is encountered, raise a runtime exception.
        // (We can't return an error code as we're running in a threadpool
        // callback with a void return signature.)
        //

        PermissibleErrorCode = (
            Result == E_OUTOFMEMORY       ||
            Result == PH_E_NO_MORE_SEEDS  ||
            Result == PH_E_CTRL_C_PRESSED
        );

        if (!PermissibleErrorCode) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(ProcessGraphCallbackChm01_InvalidErrorCode, Result);
            PH_RAISE(Result);
        }
    }

    InterlockedDecrement(&Context->ActiveSolvingLoops);

    if (InterlockedDecrement(&Context->RemainingSolverLoops) == 0) {

        //
        // We're the last graph; if the finished count indicates no solutions
        // were found, and the try larger table size event is not set, signal
        // FailedEvent.  Otherwise, signal SucceededEvent.  This ensures we
        // always unwait our parent thread's solving loop.
        //
        // N.B. There are numerous scenarios where this is a superfluous call,
        //      as a terminating event (i.e. shutdown, low-memory etc) may have
        //      already been set.  The effort required to distinguish this
        //      situation and avoid setting the event is not warranted
        //      (especially considering setting the event superfluously is
        //      harmless).
        //

        WaitResult = WaitForSingleObject(Context->TryLargerTableSizeEvent, 0);

        if (WaitResult != WAIT_OBJECT_0) {

            if (Context->FinishedCount == 0) {
                Event = &Context->FailedEvent;
            } else {
                Event = &Context->SucceededEvent;
            }

            if (!SetEvent(*Event)) {
                SYS_ERROR(SetEvent);
            }

            SetStopSolving(Context);
        }
    }
}



// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
