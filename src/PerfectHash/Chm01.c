/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    Chm01.c

Abstract:

    This module implements the CHM perfect hash table algorithm.

--*/

#include "stdafx.h"
#include "Chm01.h"

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PREPARE_GRAPH_INFO)(
    _In_ PPERFECT_HASH_TABLE Table,
    _When_(PrevInfo == NULL, _Out_)
    _When_(PrevInfo != NULL, _Inout_)
        PGRAPH_INFO Info,
    _Out_opt_ PGRAPH_INFO PrevInfo
    );
typedef PREPARE_GRAPH_INFO *PPREPARE_GRAPH_INFO;

extern PREPARE_GRAPH_INFO PrepareGraphInfoChm01;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PREPARE_TABLE_OUTPUT_DIRECTORY)(
    _In_ PPERFECT_HASH_TABLE Table
    );
typedef PREPARE_TABLE_OUTPUT_DIRECTORY *PPREPARE_TABLE_OUTPUT_DIRECTORY;

extern PREPARE_TABLE_OUTPUT_DIRECTORY PrepareTableOutputDirectory;

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
PrintCurrentContextStatsChm01(
    _In_ PPERFECT_HASH_CONTEXT Context
    );

//
// Define helper macros for checking prepare and save file work errors.
//

#define EXPAND_AS_CHECK_ERRORS(                                      \
    Verb, VUpper, Name, Upper,                                       \
    EofType, EofValue,                                               \
    Suffix, Extension, Stream, Base                                  \
)                                                                    \
    if (Verb##Name.NumberOfErrors > 0) {                             \
        Result = Verb##Name.LastResult;                              \
        if (Result == S_OK || Result == E_UNEXPECTED) {              \
            Result = PH_E_ERROR_DURING_##VUpper##_##Upper;           \
        }                                                            \
        PH_ERROR(                                                    \
            CreatePerfectHashTableImplChm01_ErrorDuring##Verb##Name, \
            Result                                                   \
        );                                                           \
        goto Error;                                                  \
    }

#define CHECK_ALL_PREPARE_ERRORS() \
    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CHECK_ERRORS)

#define CHECK_ALL_SAVE_ERRORS() \
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CHECK_ERRORS)


_Use_decl_annotations_
HRESULT
CreatePerfectHashTableImplChm01(
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

    PH_I_LOW_MEMORY - The system is indicating a low-memory state (which we
        may have caused).

    PH_I_OUT_OF_MEMORY - The system is out of memory.

    PH_I_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED - The maximum number
        of table resize events was reached before a solution could be found.

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
    PRTL Rtl;
    PRNG Rng;
    USHORT Index;
    PULONG Keys;
    PGRAPH *Graphs = NULL;
    PGRAPH Graph;
    DOUBLE Limit;
    DOUBLE Current;
    BOOLEAN Silent;
    BOOLEAN Success;
    BOOLEAN LimitConcurrency;
    ULONG Attempt = 0;
    ULONG ReferenceCount;
    BYTE NumberOfEvents;
    HRESULT Result = S_OK;
    HRESULT CloseResult = S_OK;
    ULONG WaitResult;
    ULONG BytesWritten;
    GRAPH_INFO PrevInfo;
    GRAPH_INFO Info;
    PALLOCATOR Allocator;
    HANDLE OutputHandle = NULL;
    PHANDLE Event;
    ULONG Concurrency;
    ULONG NumberOfGraphs;
    PLIST_ENTRY ListEntry;
    ULONG CloseFileErrorCount = 0;
    ULONG NumberOfSeedsRequired;
    ULONG NumberOfSeedsAvailable;
    ULONGLONG Closest;
    ULONGLONG LastClosest;
    BOOLEAN TryLargerTableSize;
    GRAPH_INFO_ON_DISK GraphInfo;
    PGRAPH_INFO_ON_DISK GraphInfoOnDisk;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;
    PPERFECT_HASH_CONTEXT Context;
    BOOL WaitForAllEvents = TRUE;
    PPERFECT_HASH_TLS_CONTEXT TlsContext;
    PERFECT_HASH_TLS_CONTEXT LocalTlsContext;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    LARGE_INTEGER EmptyEndOfFile = { 0 };
    PLARGE_INTEGER EndOfFile;

    HANDLE Events[6];
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

    Context = Table->Context;
    Concurrency = Context->MaximumConcurrency;

    //
    // If a non-zero value is supplied for predicted attempts, and the value is
    // less than the maximum concurrency, and we've been asked to limit max
    // concurrency, toggle the LimitConcurrency boolean.  This limits the number
    // of concurrent graph launches such that it won't exceed the predicted
    // number of attempts.
    //

    LimitConcurrency = (
        Table->PriorPredictedAttempts > 0 &&
        TableCreateFlags.TryUsePredictedAttemptsToLimitMaxConcurrency != FALSE
    );

    if (LimitConcurrency) {
        Concurrency = min(Concurrency, Table->PriorPredictedAttempts);
    }

    if (FirstSolvedGraphWins(Context)) {
        NumberOfGraphs = Concurrency;
    } else {

        //
        // We add 1 to the maximum concurrency in order to account for a spare
        // graph that doesn't actively participate in solving, but can be used
        // by a worker thread when it discovers a graph that is classed as the
        // "best" by the RegisterSolvedGraph() routine.
        //

        NumberOfGraphs = Concurrency + 1;
        if (NumberOfGraphs == 0) {
            return E_INVALIDARG;
        }
    }

    //
    // Initialize event arrays.
    //

    Events[0] = Context->SucceededEvent;
    Events[1] = Context->CompletedEvent;
    Events[2] = Context->ShutdownEvent;
    Events[3] = Context->FailedEvent;
    Events[4] = Context->TryLargerTableSizeEvent;
    Events[5] = Context->LowMemoryEvent;

#define EXPAND_AS_ASSIGN_EVENT(                     \
    Verb, VUpper, Name, Upper,                      \
    EofType, EofValue,                              \
    Suffix, Extension, Stream, Base                 \
)                                                   \
    *Verb##Event++ = Context->Verb##d##Name##Event;

    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_EVENT);
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_EVENT);

    //
    // Initialize aliases.
    //

    Rtl = Table->Rtl;
    Keys = (PULONG)Table->Keys->KeyArrayBaseAddress;
    Allocator = Table->Allocator;
    MaskFunctionId = Table->MaskFunctionId;
    GraphInfoOnDisk = Context->GraphInfoOnDisk = &GraphInfo;
    TableInfoOnDisk = Table->TableInfoOnDisk = &GraphInfo.TableInfoOnDisk;

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

    if (WantsAutoResizeIfKeysToEdgesRatioExceedsLimit(Table)) {
        Limit = Table->AutoResizeWhenKeysToEdgesRatioExceeds;
        Current = Table->Keys->KeysToEdgesRatio;
        if (Current > Limit) {
            Context->InitialResizes = 1;
        }
    }

    if (Context->InitialResizes > 0) {
        ULONG InitialResizes;
        ULARGE_INTEGER NumberOfEdges;
        ULARGE_INTEGER NumberOfVertices;

        //
        // We've been asked to simulate a number of table resizes prior to graph
        // solving (which is done to yield better keys-to-vertices ratios, which
        // improves solving probability).
        //

        //
        // N.B. We have to duplicate some of the sizing logic for edges and
        //      vertices from PrepareGraphInfoChm01() here.  Note that this
        //      initial resize functionality isn't supported for modulus
        //      masking.
        //

        //
        // Initialize number of edges to number of keys, then round up the
        // edges to a power of 2.
        //

        NumberOfEdges.QuadPart = Table->Keys->NumberOfKeys.QuadPart;
        ASSERT(NumberOfEdges.HighPart == 0);

        NumberOfEdges.QuadPart = (
            Rtl->RoundUpPowerOfTwo32(
                NumberOfEdges.LowPart
            )
        );

        if (NumberOfEdges.QuadPart < 8) {
            NumberOfEdges.QuadPart = 8;
        }

        //
        // Make sure we haven't overflowed.
        //

        if (NumberOfEdges.HighPart) {
            Result = PH_E_TOO_MANY_EDGES;
            goto Error;
        }

        //
        // For the number of vertices, round the number of edges up to the
        // next power of 2.
        //

        NumberOfVertices.QuadPart = (
            Rtl->RoundUpNextPowerOfTwo32(NumberOfEdges.LowPart)
        );

        Table->RequestedNumberOfTableElements.QuadPart = (
            NumberOfVertices.QuadPart
        );

        //
        // Keep doubling the number of vertices for each requested resize,
        // or until we exceed MAX_ULONG, whatever comes first.
        //

        for (InitialResizes = Context->InitialResizes;
             InitialResizes > 0;
             InitialResizes--) {

            Table->RequestedNumberOfTableElements.QuadPart <<= 1ULL;

            if (Table->RequestedNumberOfTableElements.HighPart) {
                Result = PH_I_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE;
                goto Error;
            }
        }

        Context->NumberOfTableResizeEvents = Context->InitialResizes;
    }

    Result = PrepareGraphInfoChm01(Table, &Info, NULL);
    if (FAILED(Result)) {
        PH_ERROR(CreatePerfectHashTableImplChm01_PrepareFirstGraphInfo, Result);
        goto Error;
    }

    //
    // Allocate space for an array of pointers to graph instances.
    //

    Graphs = (PGRAPH *)(
        Allocator->Vtbl->Calloc(
            Allocator,
            NumberOfGraphs,
            sizeof(Graph)
        )
    );

    if (!Graphs) {
        Result = PH_I_OUT_OF_MEMORY;
        goto Error;
    }

    //
    // We want each graph instance to have its own isolated Allocator instance
    // rather than a reference to the global (singleton) instance that is shared
    // amongst all components by default.
    //
    // There are three reasons for this.  First, it allows us to request memory
    // with the flag HEAP_NO_SERIALIZE, which avoids a small synchronization
    // penalty.  Second, it improves the usability of heap debugging tools (like
    // gflags).  Third, it allows us to release all allocations by destroying
    // the underlying heap handle (versus having to free each one individually).
    //
    // We communicate this desire to the COM component creation scaffolding by
    // way of the TLS context, which allows us to toggle a flag that disables
    // the global component override functionality for the Allocator interface,
    // as well as specify custom flags and a minimum size to HeapCreate().
    //
    // So, we now obtain the active TLS context, using our local stack-allocated
    // one if need be, toggle the disable global allocator component flag, and
    // fill out the heap create flags and minimum size (based off the total
    // allocation size calculated by the PrepareGraphInfoChm01() routine above).
    //

    TlsContext = PerfectHashTlsGetOrSetContext(&LocalTlsContext);

    ASSERT(!TlsContext->Flags.DisableGlobalAllocatorComponent);
    ASSERT(!TlsContext->Flags.CustomAllocatorDetailsPresent);
    ASSERT(!TlsContext->HeapCreateFlags);
    ASSERT(!TlsContext->HeapMinimumSize);

    TlsContextDisableGlobalAllocator(TlsContext);
    TlsContext->Flags.CustomAllocatorDetailsPresent = TRUE;
    TlsContext->HeapCreateFlags = HEAP_NO_SERIALIZE;
    TlsContext->HeapMinimumSize = (ULONG_PTR)Info.AllocSize;

    //
    // Make sure we haven't overflowed MAX_ULONG.  This should be caught
    // earlier when preparing the graph info.
    //

    if ((ULONGLONG)TlsContext->HeapMinimumSize != Info.AllocSize) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    //
    // Copy the table create flags into the TLS context as well; they are now
    // used by GraphInitialize() to tweak vtbl construction.
    //

    TlsContext->TableCreateFlags.AsULongLong = TableCreateFlags.AsULongLong;

    //
    // ....and, many years later, we find ourselves in the position of wanting
    // to access the table's hash function from within GraphInitialize() as
    // well, so, we just stash the table pointer in the TLS context now too.
    // This is a bit sloppy; we don't need TableCreateFlags if we're storing
    // a pointer to the table directly.  Future refactoring opportunity.
    //

    TlsContext->Table = Table;

    //
    // Create graph instances and capture the resulting pointer in the array
    // we just allocated above.
    //

    for (Index = 0; Index < NumberOfGraphs; Index++) {

        Result = Table->Vtbl->CreateInstance(Table,
                                             NULL,
                                             &IID_PERFECT_HASH_GRAPH,
                                             (PVOID *)&Graph);

        if (FAILED(Result)) {

            //
            // Suppress logging for out-of-memory errors (as we communicate
            // memory issues back to the caller via informational return codes).
            //

            if (Result != E_OUTOFMEMORY) {
                PH_ERROR(CreatePerfectHashTableImplChm01_CreateGraph, Result);
            }

            //
            // N.B. We 'break' instead of 'goto Error' here like we normally
            //      do in order for the TLS context cleanup logic following
            //      this routine to run.
            //

            break;
        }

        //
        // Verify the uniqueness of our graph allocator and underlying handle.
        //

        ASSERT(Graph->Allocator != Table->Allocator);
        ASSERT(Graph->Allocator->HeapHandle != Table->Allocator->HeapHandle);

        //
        // Verify the Rtl instance was global.
        //

        ASSERT(Graph->Rtl == Table->Rtl);

        //
        // Copy relevant flags over, then save the graph instance in the array.
        //

        Graph->Flags.SkipVerification = (
            TableCreateFlags.SkipGraphVerification != FALSE
        );

        Graph->Flags.WantsWriteCombiningForVertexPairsArray = (
            TableCreateFlags.EnableWriteCombineForVertexPairs != FALSE
        );

        Graph->Flags.RemoveWriteCombineAfterSuccessfulHashKeys = (
            TableCreateFlags.RemoveWriteCombineAfterSuccessfulHashKeys != FALSE
        );

        Graph->Index = Index;
        Graphs[Index] = Graph;
    }

    //
    // Restore all the values we mutated, then potentially clear the TLS context
    // if our local stack-allocated version was used.  Then, check the result
    // and jump to our error handling block if it indicates failure.
    //

    TlsContextEnableGlobalAllocator(TlsContext);
    TlsContext->Flags.CustomAllocatorDetailsPresent = FALSE;
    TlsContext->HeapCreateFlags = 0;
    TlsContext->HeapMinimumSize = 0;
    TlsContext->Table = NULL;

    PerfectHashTlsClearContextIfActive(&LocalTlsContext);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // The following label is jumped to by code later in this routine when we
    // detect that we've exceeded a plausible number of attempts at finding a
    // graph solution with the given number of vertices, and have bumped up
    // the vertex count (by adjusting Table->RequestedNumberOfElements) and
    // want to try again.
    //

RetryWithLargerTableSize:

    //
    // Explicitly reset all events.  This ensures everything is back in the
    // starting state if we happen to be attempting to solve the graph after
    // a resize event.
    //

    Event = (PHANDLE)&Context->FirstEvent;

    NumberOfEvents = GetNumberOfContextEvents(Context);

    for (Index = 0; Index < NumberOfEvents; Index++, Event++) {

        if (!ResetEvent(*Event)) {
            SYS_ERROR(ResetEvent);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }
    }

    //
    // Clear the counter of low-memory events observed.  (An interlocked
    // increment is performed against this counter in various locations
    // each time a wait is satisfied against the low-memory event.)
    //

    Context->LowMemoryObserved = 0;

    //
    // If this isn't the first attempt, prepare the graph info again.  This
    // updates the various allocation sizes based on the new table size being
    // requested.
    //

    if (++Attempt > 1) {

        ASSERT(Context->ResizeLimit > 0);
        Result = PrepareGraphInfoChm01(Table, &Info, &PrevInfo);
        if (FAILED(Result)) {
            PH_ERROR(CreatePerfectHashTableImplChm01_PrepareGraphInfo, Result);
            goto Error;
        }

    }

    //
    // Set the context's main work and console callbacks to our worker routines,
    // and the algo context to our graph info structure.
    //

    Context->MainWorkCallback = ProcessGraphCallbackChm01;
    Context->ConsoleWorkCallback = ProcessConsoleCallbackChm01;
    Context->AlgorithmContext = &Info;

    //
    // Set the context's file work callback to our worker routine.
    //

    Context->FileWorkCallback = FileWorkCallbackChm01;

    //
    // Prepare the table output directory.  If the table indicates resize events
    // require a rename, we need to call this every loop invocation.  Otherwise,
    // just call it on the first invocation (Attempt == 1).
    //

    if (!NoFileIo(Table) &&
        (Attempt == 1 || TableResizeRequiresRename(Table))) {

        Result = PrepareTableOutputDirectory(Table);
        if (FAILED(Result)) {
            PH_ERROR(PrepareTableOutputDirectory, Result);
            goto Error;
        }
    }

    //
    // If we've been asked to cap the maximum solve time, submit a threadpool
    // timer now to achieve that.
    //

    if (Table->MaxSolveTimeInSeconds > 0) {
        SetThreadpoolTimer(Context->SolveTimeout,
                           &Table->RelativeMaxSolveTimeInFiletime.AsFileTime,
                           0,
                           0);
    }

    //
    // Submit all of the file preparation work items.
    //

#define EXPAND_AS_SUBMIT_FILE_WORK(                     \
    Verb, VUpper, Name, Upper,                          \
    EofType, EofValue,                                  \
    Suffix, Extension, Stream, Base                     \
)                                                       \
    ASSERT(!NoFileIo(Table));                           \
    ZeroStructInline(Verb##Name);                       \
    Verb##Name.FileWorkId = FileWork##Verb##Name##Id;   \
    InsertTailFileWork(Context, &Verb##Name.ListEntry); \
    SubmitThreadpoolWork(Context->FileWork);

#define SUBMIT_PREPARE_FILE_WORK() \
    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_FILE_WORK)

#define SUBMIT_SAVE_FILE_WORK() \
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_FILE_WORK)

    ASSERT(Context->FileWorkList->Vtbl->IsEmpty(Context->FileWorkList));

    if (!NoFileIo(Table)) {
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
    // Submit the console thread work.
    //

    SubmitThreadpoolWork(Context->ConsoleWork);

    //
    // For each graph instance, set the graph info, and, if we haven't reached
    // the concurrency limit, append the graph to the context work list and
    // submit threadpool work for it (to begin graph solving).
    //

    ASSERT(Context->MainWorkList->Vtbl->IsEmpty(Context->MainWorkList));
    ASSERT(Context->FinishedWorkList->Vtbl->IsEmpty(Context->FinishedWorkList));

    if (FirstSolvedGraphWins(Context)) {
        ASSERT(NumberOfGraphs == Concurrency);
    } else {
        ASSERT(NumberOfGraphs - 1 == Concurrency);
    }

    for (Index = 0; Index < NumberOfGraphs; Index++) {

        Graph = Graphs[Index];

        //
        // Explicitly reset the graph's lock.  We know there is no contention
        // at this point of execution, however, in certain situations where a
        // table resize event has occurred and we're in "find best graph" mode,
        // a graph's lock may still be set, which obviously causes our attempt
        // to acquire it to hang.
        //

        ResetSRWLock(&Graph->Lock);

        AcquireGraphLockExclusive(Graph);
        Result = Graph->Vtbl->SetInfo(Graph, &Info);
        ReleaseGraphLockExclusive(Graph);

        if (FAILED(Result)) {
            PH_ERROR(GraphSetInfo, Result);
            goto Error;
        }

        Graph->Flags.IsInfoLoaded = FALSE;

        if (!FirstSolvedGraphWins(Context) && Index == 0) {

            //
            // This is our first graph, which is marked as the "spare" graph.
            // If a worker thread finds the best graph, it will swap its graph
            // for this spare one, such that it can continue looking for new
            // solutions.
            //

            Graph->Flags.IsSpare = TRUE;

            //
            // Context->SpareGraph is _Guarded_by_(BestGraphCriticalSection).
            // We know that no worker threads will be running at this point;
            // inform SAL accordingly by suppressing the concurrency warnings.
            //

            _Benign_race_begin_
            Context->SpareGraph = Graph;
            _Benign_race_end_

        } else {
            Graph->Flags.IsSpare = FALSE;
            InitializeListHead(&Graph->ListEntry);
            InsertTailMainWork(Context, &Graph->ListEntry);
            SubmitThreadpoolWork(Context->MainWork);
        }

    }

    //
    // Wait on the context's events.
    //

    WaitResult = WaitForMultipleObjects(ARRAYSIZE(Events),
                                        Events,
                                        FALSE,
                                        INFINITE);

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
    // Handle the low-memory state first.
    //

    if (WaitResult == WAIT_OBJECT_0+5) {
        InterlockedIncrement(&Context->LowMemoryObserved);
        Result = PH_I_LOW_MEMORY;
        goto Error;
    }

    //
    // If the wait result indicates the try larger table size event was set,
    // deal with that, next.
    //

    TryLargerTableSize = (
        WaitResult == WAIT_OBJECT_0+4 || (
            WaitForSingleObject(Context->TryLargerTableSizeEvent, 0) ==
            WAIT_OBJECT_0
        )
    );

    if (TryLargerTableSize) {

        //
        // The number of attempts at solving this graph have exceeded the
        // threshold.  Set the shutdown event in order to trigger all worker
        // threads to abort their current attempts and wait on the main thread
        // work, then finish work, to complete.
        //

        WaitForThreadpoolWorkCallbacks(Context->MainWork, TRUE);
        WaitForThreadpoolWorkCallbacks(Context->ConsoleWork, TRUE);
        WaitForThreadpoolWorkCallbacks(Context->FinishedWork, FALSE);
        WaitForThreadpoolTimerCallbacks(Context->SolveTimeout, TRUE);

        if (!NoFileIo(Table)) {

            //
            // Perform a blocking wait for the prepare work to complete.
            //

            WaitResult = WaitForMultipleObjects(ARRAYSIZE(PrepareEvents),
                                                PrepareEvents,
                                                WaitForAllEvents,
                                                INFINITE);

            if (WaitResult != WAIT_OBJECT_0) {
                SYS_ERROR(WaitForSingleObject);
                Result = PH_E_SYSTEM_CALL_FAILED;
                goto Error;
            }

            //
            // Verify none of the file work callbacks reported an error during
            // preparation.
            //

            CHECK_ALL_PREPARE_ERRORS();
        }

        //
        // There are no more threadpool callbacks running.  However, a thread
        // could have finished a solution between the time the try larger table
        // size event was set, and this point.  So, check the finished count
        // first.  If it indicates a solution, jump to that handler code.
        //
        // N.B. The user could have requested another resize via the console,
        //      in which case, we ignore the finished count.
        //

        if (Context->State.UserRequestedResize != FALSE) {
            Context->State.UserRequestedResize = FALSE;
        } else if (Context->FinishedCount > 0) {
            goto FinishedSolution;
        }

        //
        // Check to see if we've exceeded the maximum number of resize events.
        //

        if (Context->NumberOfTableResizeEvents >= Context->ResizeLimit) {
            Result = PH_I_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED;
            goto Error;
        }

        //
        // Increment the resize counter and update the total number of attempts
        // in the header.  Then, determine how close we came to solving the
        // graph, and store that in the header as well if it's the best so far
        // (or no previous version is present).
        //

        Context->NumberOfTableResizeEvents++;
        Context->TotalNumberOfAttemptsWithSmallerTableSizes += (
            Context->Attempts
        );

        Closest = (
            Info.Dimensions.NumberOfEdges - Context->HighestDeletedEdgesCount
        );
        LastClosest = (
            Context->ClosestWeCameToSolvingGraphWithSmallerTableSizes
        );

        if (!LastClosest || Closest < LastClosest) {
            Context->ClosestWeCameToSolvingGraphWithSmallerTableSizes = (
                Closest
            );
        }

        //
        // If this is our first resize, capture the initial size we used.
        //

        if (!Context->InitialTableSize) {
            Context->InitialTableSize = Info.Dimensions.NumberOfVertices;
        }

        //
        // Reset the remaining counters.
        //

        Context->Attempts = 0;
        Context->FailedAttempts = 0;
        Context->HighestDeletedEdgesCount = 0;

        //
        // Double the vertex count.  If we have overflowed max ULONG, abort.
        //

        Table->RequestedNumberOfTableElements.QuadPart = (
            Info.Dimensions.NumberOfVertices
        );

        Table->RequestedNumberOfTableElements.QuadPart <<= 1ULL;

        if (Table->RequestedNumberOfTableElements.HighPart) {
            Result = PH_I_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE;
            goto Error;
        }

        //
        // Reset the lists.
        //

        ResetMainWorkList(Context);
        ResetFinishedWorkList(Context);
        if (!NoFileIo(Table)) {
            ResetFileWorkList(Context);
        }

        //
        // Print a plus if we're in context table/bulk create mode to indicate
        // a table resize event has occurred.
        //

        MAYBE_PLUS();

        //
        // Jump back to the start and try again with a larger vertex count.
        //

        goto RetryWithLargerTableSize;
    }

    //
    // The wait result did not indicate a resize event.  Ignore the wait
    // result for now; determine if the graph solving was successful by the
    // finished count of the context.  We'll corroborate that with whatever
    // events have been signaled shortly.
    //

    WaitForThreadpoolWorkCallbacks(Context->MainWork, TRUE);
    WaitForThreadpoolWorkCallbacks(Context->ConsoleWork, TRUE);
    WaitForThreadpoolWorkCallbacks(Context->FinishedWork, FALSE);
    WaitForThreadpoolTimerCallbacks(Context->SolveTimeout, TRUE);

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

            if (Context->State.AllGraphsFailedMemoryAllocation != FALSE) {
                Result = PH_I_FAILED_TO_ALLOCATE_MEMORY_FOR_ALL_GRAPHS;
            } else if (Context->State.SolveTimeoutExpired != FALSE) {
                Result = PH_I_SOLVE_TIMEOUT_EXPIRED;
            } else {
                Result = PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION;
            }

        } else if (ShutdownEventSet) {

            Result = PH_I_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT;
        }

        ASSERT(StopSolving(Context));

        //
        // Wait for the main thread work group members.  This will block until
        // all the worker threads have returned.  We need to put this in place
        // prior to jumping to the End: label as that step will release all the
        // graph references and verify the resulting reference count for each
        // one is 0.  This won't be the case if any threads are still working.
        //

        WaitForThreadpoolWorkCallbacks(Context->MainWork, CancelPending);

        //
        // Wait for the console thread.
        //

        WaitForThreadpoolWorkCallbacks(Context->ConsoleWork, TRUE);

        //
        // Wait for the solve timeout, if applicable.
        //

        WaitForThreadpoolTimerCallbacks(Context->SolveTimeout, TRUE);

        //
        // Perform the same operation for the file work threadpool.  Note that
        // the only work item type we've dispatched to this pool at this point
        // is file preparation work.
        //

        if (!NoFileIo(Table)) {
            WaitForThreadpoolWorkCallbacks(Context->FileWork, FALSE);
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

    WaitForThreadpoolWorkCallbacks(Context->MainWork, TRUE);
    WaitForThreadpoolWorkCallbacks(Context->ConsoleWork, TRUE);
    WaitForThreadpoolWorkCallbacks(Context->FinishedWork, FALSE);
    WaitForThreadpoolTimerCallbacks(Context->SolveTimeout, TRUE);

    if (CtrlCPressed) {
        Result = PH_E_CTRL_C_PRESSED;
        goto Error;
    }

    ASSERT(Context->FinishedCount > 0);

    if (!TableCreateFlags.Quiet) {
        Result = PrintCurrentContextStatsChm01(Context);
        if (FAILED(Result)) {
            PH_ERROR(PrintCurrentContextStatsChm01, Result);
            goto Error;
        }
    }

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
            PH_ERROR(CreatePerfectHashTableImplChm01_RngGetCurrentOffset,
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
        PH_ERROR(CreatePerfectHashTableImplChm01_CalculatePredictedAttempts,
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
                   Table->TableInfoOnDisk,
                   sizeof(*NewGraphInfoOnDisk));

        //
        // Sanity check the first seed is not 0.
        //

        if (Graph->FirstSeed == 0) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(CreatePerfectHashTableImplChm01_GraphFirstSeedIs0, Result);
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

    WaitResult = WaitForMultipleObjects(ARRAYSIZE(SaveEvents),
                                        SaveEvents,
                                        WaitForAllEvents,
                                        INFINITE);

    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForSingleObject);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

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
    WaitForThreadpoolWorkCallbacks(Context->MainWork, TRUE);
    WaitForThreadpoolWorkCallbacks(Context->ConsoleWork, TRUE);
    WaitForThreadpoolTimerCallbacks(Context->SolveTimeout, TRUE);
    if (!NoFileIo(Table)) {
        WaitForThreadpoolWorkCallbacks(Context->FileWork, FALSE);
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (Result == E_OUTOFMEMORY) {

        //
        // Convert the out-of-memory error code into our equivalent info code.
        //

        Result = PH_I_OUT_OF_MEMORY;
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
    InsertTailFileWork(Context, &Verb##Name.ListEntry); \
    SubmitThreadpoolWork(Context->FileWork);

#define SUBMIT_CLOSE_FILE_WORK() \
    CLOSE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_CLOSE_FILE_WORK)

    SUBMIT_CLOSE_FILE_WORK();

    WaitForThreadpoolWorkCallbacks(Context->FileWork, FALSE);

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
            CreatePerfectHashTableImplChm01_ErrorDuring##Verb##Name, \
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

    if (Graphs) {

        //
        // Walk the array of graph instances and release each one (assuming it
        // is not NULL), then free the array buffer.
        //

        for (Index = 0; Index < NumberOfGraphs; Index++) {

            Graph = Graphs[Index];

            if (Graph) {

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

    return Result;
}


PREPARE_GRAPH_INFO PrepareGraphInfoChm01;

_Use_decl_annotations_
HRESULT
PrepareGraphInfoChm01(
    PPERFECT_HASH_TABLE Table,
    PGRAPH_INFO Info,
    PGRAPH_INFO PrevInfo
    )
/*++

Routine Description:

    Prepares the GRAPH_INFO structure for a given table.

    N.B. This routine was created by lifting all of the logic from the body of
         the CreatePerfectHashTableImplChm01() routine, which is where it was
         originally based.  It could do with an overhaul; it uses way too many
         local variables unnecessarily, for example.

Arguments:

    Table - Supplies a pointer to the table.

    Info - Supplies a pointer to the graph info structure to prepare.

    PrevInfo - Optionally supplies a pointer to the previous info structure
        if this is not the first time the routine is being called.

Return Value:

    S_OK - Graph info prepared successfully.

    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - Table or Info were NULL.

    E_UNEXPECTED - Catastrophic internal error.

    PH_E_TOO_MANY_KEYS - Too many keys.

    PH_E_TOO_MANY_EDGES - Too many edges.

    PH_E_TOO_MANY_TOTAL_EDGES - Too many total edges.

    PH_E_TOO_MANY_VERTICES - Too many vertices.

--*/
{
    PRTL Rtl;
    HRESULT Result = S_OK;
    BYTE AssignedShift;
    ULONG GraphImpl;
    ULONG NumberOfKeys;
    BOOLEAN UseAssigned16;
    USHORT NumberOfBitmaps;
    PGRAPH_DIMENSIONS Dim;
    SYSTEM_INFO SystemInfo;
    PCSTRING TypeNames;
    PGRAPH_INFO_ON_DISK GraphInfoOnDisk;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    ULONG NumberOfEdgeMaskBits;
    ULONG NumberOfVertexMaskBits;
    ULONGLONG NextSizeInBytes;
    ULONGLONG FirstSizeInBytes;
    ULONGLONG OrderSizeInBytes;
    ULONGLONG EdgesSizeInBytes;
    ULONGLONG Vertices3SizeInBytes;
    ULONGLONG ValuesSizeInBytes;
    ULONGLONG AssignedSizeInBytes;
    ULONGLONG VertexPairsSizeInBytes;
    PPERFECT_HASH_CONTEXT Context;
    ULARGE_INTEGER AllocSize;
    ULARGE_INTEGER NumberOfEdges;
    ULARGE_INTEGER NumberOfVertices;
    ULARGE_INTEGER TotalNumberOfEdges;
    ULARGE_INTEGER DeletedEdgesBitmapBufferSizeInBytes;
    ULARGE_INTEGER VisitedVerticesBitmapBufferSizeInBytes;
    ULARGE_INTEGER AssignedBitmapBufferSizeInBytes;
    ULARGE_INTEGER IndexBitmapBufferSizeInBytes;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;
    PTRAILING_ZEROS_32 TrailingZeros32;
    PTRAILING_ZEROS_64 TrailingZeros64;
    PPOPULATION_COUNT_32 PopulationCount32;
    PROUND_UP_POWER_OF_TWO_32 RoundUpPowerOfTwo32;
    PROUND_UP_NEXT_POWER_OF_TWO_32 RoundUpNextPowerOfTwo32;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Info)) {
        return E_POINTER;
    }

    //
    // Initialize aliases.
    //

    Rtl = Table->Rtl;
    Context = Table->Context;
    GraphImpl = Table->GraphImpl;
    MaskFunctionId = Table->MaskFunctionId;
    GraphInfoOnDisk = Context->GraphInfoOnDisk;
    TableInfoOnDisk = &GraphInfoOnDisk->TableInfoOnDisk;
    TypeNames = Table->CTypeNames;
    TrailingZeros32 = Rtl->TrailingZeros32;
    TrailingZeros64 = Rtl->TrailingZeros64;
    PopulationCount32 = Rtl->PopulationCount32;
    RoundUpPowerOfTwo32 = Rtl->RoundUpPowerOfTwo32;
    RoundUpNextPowerOfTwo32 = Rtl->RoundUpNextPowerOfTwo32;
    TableCreateFlags.AsULongLong = Table->TableCreateFlags.AsULongLong;

    //
    // If a previous Info struct pointer has been passed, copy the current
    // Info contents into it.
    //

    if (ARGUMENT_PRESENT(PrevInfo)) {
        CopyInline(PrevInfo, Info, sizeof(*PrevInfo));
    }

    //
    // Clear our Info struct and wire up the PrevInfo pointer (which may be
    // NULL).
    //

    ZeroStructPointerInline(Info);
    Info->PrevInfo = PrevInfo;

    //
    // Ensure the number of keys are under MAX_ULONG, then take a local copy.
    //

    if (Table->Keys->NumberOfKeys.HighPart) {
        return PH_E_TOO_MANY_KEYS;
    }

    NumberOfKeys = Table->Keys->NumberOfKeys.LowPart;

    //
    // The number of edges in our graph is equal to the number of keys in the
    // input data set if modulus masking is in use.  It will be rounded up to
    // a power of 2 otherwise.
    //

    NumberOfEdges.QuadPart = NumberOfKeys;

    //
    // Make sure we have at least 8 edges; this ensures the assigned array
    // will consume at least one cache line, which is required for our memory
    // coverage routine (see Graph.c) to work correctly, as it operates on
    // cache line sized strides.
    //

    if (NumberOfEdges.QuadPart < 8) {
        NumberOfEdges.QuadPart = 8;
    }

    //
    // Determine the number of vertices.  If we've reached here due to a resize
    // event, Table->RequestedNumberOfTableElements will be non-zero, and takes
    // precedence.  Otherwise, determine the vertices heuristically.
    //

    if (Table->RequestedNumberOfTableElements.QuadPart) {

        NumberOfVertices.QuadPart = (
            Table->RequestedNumberOfTableElements.QuadPart
        );

        if (IsModulusMasking(MaskFunctionId)) {

            //
            // Nothing more to do with modulus masking; we'll verify the number
            // of vertices below.
            //

            NOTHING;

        } else {

            //
            // For non-modulus masking, make sure the number of vertices are
            // rounded up to a power of 2.
            //

            NumberOfVertices.QuadPart = (
                RoundUpPowerOfTwo32(NumberOfVertices.LowPart)
            );

            //
            // If we're clamping number of edges, use number of keys rounded up
            // to a power of two.  Otherwise, use the number of vertices shifted
            // right by one (divided by two).
            //

            if (Table->TableCreateFlags.ClampNumberOfEdges) {
                NumberOfEdges.QuadPart = RoundUpPowerOfTwo32(NumberOfKeys);
            } else {
                NumberOfEdges.QuadPart = NumberOfVertices.QuadPart >> 1ULL;
            }

        }

    } else {

        //
        // No table size was requested, so we need to determine how many
        // vertices to use heuristically.  The main factor is what type of
        // masking has been requested.  The chm.c implementation, which is
        // modulus based, uses a size multiplier (c) of 2.09, and calculates
        // the final size via ceil(nedges * (double)2.09).  We can avoid the
        // need for doubles and linking with a math library (to get ceil())
        // and just use ~2.25, which we can calculate by adding the result
        // of right shifting the number of edges by 1 to the result of left
        // shifting said edge count by 2 (simulating multiplication by 0.25).
        //
        // If we're dealing with modulus masking, this will be the exact number
        // of vertices used.  For other types of masking, we need the edges size
        // to be a power of 2, and the vertices size to be the next power of 2.
        //
        // Update: attempt to use primes for modulus masking.  Still doesn't
        //         work.
        //

        if (IsModulusMasking(MaskFunctionId)) {
            SHORT PrimeIndex;
            ULONGLONG Value;
            ULONGLONG Prime;

            //
            // Find a prime greater than or equal to the number of edges.
            //

            Value = NumberOfEdges.QuadPart;

            PrimeIndex = FindIndexForFirstPrimeGreaterThanOrEqual(Value);
            if (PrimeIndex == -1) {
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
            }

            Prime = Primes[PrimeIndex];
            NumberOfEdges.QuadPart = Prime;

            //
            // Double the number of edges, then find a prime greater than or
            // equal to this new value.
            //

            Value <<= 1;

            PrimeIndex = FindIndexForFirstPrimeGreaterThanOrEqual(Value);
            if (PrimeIndex == -1) {
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
            }

            Prime = Primes[PrimeIndex];
            NumberOfVertices.QuadPart = Prime;

        } else {

            //
            // Round up the edges to a power of 2.
            //

            NumberOfEdges.QuadPart = RoundUpPowerOfTwo32(NumberOfEdges.LowPart);

            //
            // Make sure we haven't overflowed.
            //

            if (NumberOfEdges.HighPart) {
                Result = PH_E_TOO_MANY_EDGES;
                goto Error;
            }

            //
            // For the number of vertices, round the number of edges up to the
            // next power of 2.
            //

            NumberOfVertices.QuadPart = (
                RoundUpNextPowerOfTwo32(NumberOfEdges.LowPart)
            );

        }
    }

    //
    // Another sanity check we haven't exceeded MAX_ULONG.
    //

    if (NumberOfVertices.HighPart) {
        Result = PH_E_TOO_MANY_VERTICES;
        goto Error;
    }

    //
    // The r-graph (r = 2) nature of this implementation results in various
    // arrays having twice the number of elements indicated by the edge count.
    // Capture this number now, as we need it in various size calculations.
    //

    TotalNumberOfEdges.QuadPart = NumberOfEdges.QuadPart;
    TotalNumberOfEdges.QuadPart <<= 1ULL;

    //
    // Another overflow sanity check.
    //

    if (TotalNumberOfEdges.HighPart) {
        Result = PH_E_TOO_MANY_TOTAL_EDGES;
        goto Error;
    }

    //
    // Make sure vertices > edges.
    //

    if (NumberOfVertices.QuadPart <= NumberOfEdges.QuadPart) {
        Result = PH_E_NUM_VERTICES_LESS_THAN_OR_EQUAL_NUM_EDGES;
        goto Error;
    }

    //
    // Invariant check: sure vertices shifted right once == edges if applicable.
    //

    if ((!IsModulusMasking(MaskFunctionId)) &&
        (Table->TableCreateFlags.ClampNumberOfEdges == FALSE)) {

        if ((NumberOfVertices.QuadPart >> 1) != NumberOfEdges.QuadPart) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareGraphInfoChm01_NumEdgesNotNumVerticesDiv2, Result);
            goto Error;
        }
    }

    //
    // If our graph impl is v3, our vertex count - 1 is less than or equal to
    // MAX_USHORT (i.e. 65535), and the user hasn't supplied the flag to the
    // contrary, use the 16-bit hash/assigned implementation.
    //

    if ((GraphImpl == 3) &&
        ((NumberOfVertices.LowPart-1) <= 0x0000ffff) &&
        (TableCreateFlags.DoNotTryUseHash16Impl == FALSE)) {

        UseAssigned16 = TRUE;
        AssignedShift = ASSIGNED16_SHIFT;
        Table->State.UsingAssigned16 = TRUE;

        //
        // Overwrite the vtbl Index routines accordingly.
        //

        Table->Vtbl->Index = PerfectHashTableIndex16ImplChm01;
        Table->Vtbl->FastIndex = NULL;
        Table->Vtbl->SlowIndex = NULL;

    } else {
        UseAssigned16 = FALSE;
        AssignedShift = ASSIGNED_SHIFT;
        Table->State.UsingAssigned16 = FALSE;
    }

    //
    // Calculate the size required for our bitmap buffers.
    //

    DeletedEdgesBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(TotalNumberOfEdges.QuadPart, 8) >> 3
    );

    if (DeletedEdgesBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_DeletedEdgesBitmap, Result);
        goto Error;
    }

    VisitedVerticesBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(NumberOfVertices.QuadPart, 8) >> 3
    );

    if (VisitedVerticesBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_VisitedVerticesBitmap, Result);
        goto Error;
    }

    AssignedBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(NumberOfVertices.QuadPart, 8) >> 3
    );

    if (AssignedBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_AssignedBitmap, Result);
        goto Error;
    }

    IndexBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(NumberOfVertices.QuadPart, 8) >> 3
    );

    if (IndexBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_IndexBitmap, Result);
        goto Error;
    }

    //
    // Calculate the sizes required for each of the arrays.
    //

    if (GraphImpl == 1 || GraphImpl == 2) {

        EdgesSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Edges) * TotalNumberOfEdges.QuadPart
        );

        NextSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Next) * TotalNumberOfEdges.QuadPart
        );

        FirstSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, First) * NumberOfVertices.QuadPart
        );

        Vertices3SizeInBytes = 0;

        if (TableCreateFlags.HashAllKeysFirst == FALSE) {
            VertexPairsSizeInBytes = 0;
        } else {
            VertexPairsSizeInBytes = ALIGN_UP_ZMMWORD(
                RTL_ELEMENT_SIZE(GRAPH, VertexPairs) * (ULONGLONG)NumberOfKeys
            );
        }

    } else {

        ASSERT(GraphImpl == 3);

        EdgesSizeInBytes = 0;
        NextSizeInBytes = 0;
        FirstSizeInBytes = 0;

        if (!UseAssigned16) {

            VertexPairsSizeInBytes = ALIGN_UP_YMMWORD(
                RTL_ELEMENT_SIZE(GRAPH, VertexPairs) *
                (ULONGLONG)NumberOfKeys
            );

            Vertices3SizeInBytes = ALIGN_UP_YMMWORD(
                RTL_ELEMENT_SIZE(GRAPH, Vertices3) *
                NumberOfVertices.QuadPart
            );

        } else {

            VertexPairsSizeInBytes = ALIGN_UP_YMMWORD(
                RTL_ELEMENT_SIZE(GRAPH, Vertex16Pairs) *
                (ULONGLONG)NumberOfKeys
            );

            Vertices3SizeInBytes = ALIGN_UP_YMMWORD(
                RTL_ELEMENT_SIZE(GRAPH, Vertices163) *
                NumberOfVertices.QuadPart
            );

        }

        DeletedEdgesBitmapBufferSizeInBytes.QuadPart = 0;
    }

    if (!UseAssigned16) {

        OrderSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Order) * NumberOfEdges.QuadPart
        );

        AssignedSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Assigned) * NumberOfVertices.QuadPart
        );

    } else {

        OrderSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Order16) * NumberOfEdges.QuadPart
        );

        AssignedSizeInBytes = ALIGN_UP_YMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, Assigned16) * NumberOfVertices.QuadPart
        );

    }

    //
    // Calculate the size required for the values array.  This is used as part
    // of verification, where we essentially do Insert(Key, Key) in combination
    // with bitmap tracking of assigned indices, which allows us to detect if
    // there are any colliding indices, and if so, what was the previous key
    // that mapped to the same index.
    //

    ValuesSizeInBytes = ALIGN_UP_YMMWORD(
        RTL_ELEMENT_SIZE(GRAPH, Values) * NumberOfVertices.QuadPart
    );

    //
    // Calculate the number of cache lines, pages and large pages covered by
    // the assigned array, plus the respective buffer sizes for each array in
    // the ASSIGNED_MEMORY_COVERAGE structure that captures counts.
    //
    // N.B. We don't use AssignedSizeInBytes here as it is subject to being
    //      aligned up to a YMMWORD boundary, and thus, may not represent the
    //      exact number of cache lines used strictly for vertices.
    //

    //
    // Element counts.
    //

    Info->AssignedArrayNumberOfPages = (ULONG)(
        BYTES_TO_PAGES(NumberOfVertices.QuadPart << AssignedShift)
    );

    Info->AssignedArrayNumberOfLargePages = (ULONG)(
        BYTES_TO_LARGE_PAGES(NumberOfVertices.QuadPart << AssignedShift)
    );

    Info->AssignedArrayNumberOfCacheLines = (ULONG)(
        BYTES_TO_CACHE_LINES(NumberOfVertices.QuadPart << AssignedShift)
    );

    //
    // Array sizes for the number of assigned per page, large page and cache
    // line.
    //

    if (!UseAssigned16) {

        Info->NumberOfAssignedPerPageSizeInBytes = (
            Info->AssignedArrayNumberOfPages *
            RTL_ELEMENT_SIZE(ASSIGNED_MEMORY_COVERAGE,
                             NumberOfAssignedPerPage)
        );

        Info->NumberOfAssignedPerLargePageSizeInBytes = (
            Info->AssignedArrayNumberOfLargePages *
            RTL_ELEMENT_SIZE(ASSIGNED_MEMORY_COVERAGE,
                             NumberOfAssignedPerLargePage)
        );

        Info->NumberOfAssignedPerCacheLineSizeInBytes = (
            Info->AssignedArrayNumberOfCacheLines *
            RTL_ELEMENT_SIZE(ASSIGNED_MEMORY_COVERAGE,
                             NumberOfAssignedPerCacheLine)
        );

    } else {

        Info->NumberOfAssignedPerPageSizeInBytes = (
            Info->AssignedArrayNumberOfPages *
            RTL_ELEMENT_SIZE(ASSIGNED16_MEMORY_COVERAGE,
                             NumberOfAssignedPerPage)
        );

        Info->NumberOfAssignedPerLargePageSizeInBytes = (
            Info->AssignedArrayNumberOfLargePages *
            RTL_ELEMENT_SIZE(ASSIGNED16_MEMORY_COVERAGE,
                             NumberOfAssignedPerLargePage)
        );

        Info->NumberOfAssignedPerCacheLineSizeInBytes = (
            Info->AssignedArrayNumberOfCacheLines *
            RTL_ELEMENT_SIZE(ASSIGNED16_MEMORY_COVERAGE,
                             NumberOfAssignedPerCacheLine)
        );

    }

    //
    // Calculate the total size required for the underlying arrays, bitmap
    // buffers and assigned array counts, rounded up to the nearest page size.
    //

    AllocSize.QuadPart = ROUND_TO_PAGES(
        EdgesSizeInBytes +
        NextSizeInBytes +
        OrderSizeInBytes +
        FirstSizeInBytes +
        Vertices3SizeInBytes +
        AssignedSizeInBytes +
        VertexPairsSizeInBytes +
        ValuesSizeInBytes +

        Info->NumberOfAssignedPerPageSizeInBytes +
        Info->NumberOfAssignedPerLargePageSizeInBytes +
        Info->NumberOfAssignedPerCacheLineSizeInBytes +

        //
        // Begin bitmaps.
        //

        DeletedEdgesBitmapBufferSizeInBytes.QuadPart +
        VisitedVerticesBitmapBufferSizeInBytes.QuadPart +
        AssignedBitmapBufferSizeInBytes.QuadPart +
        IndexBitmapBufferSizeInBytes.QuadPart +

        //
        // End bitmaps.
        //

        //
        // Keep a dummy 0 at the end such that the last item above can use an
        // addition sign at the end of it, which minimizes the diff churn when
        // adding a new size element.
        //

        0

    );

    //
    // Capture the number of bitmaps here, where it's close to the lines above
    // that indicate how many bitmaps we're dealing with.  The number of bitmaps
    // accounted for above should match this number.  Visually confirm this any
    // time a new bitmap buffer is accounted for.
    //
    // N.B. We ASSERT() in InitializeGraph() if we detect a mismatch between
    //      Info->NumberOfBitmaps and a local counter incremented each time
    //      we initialize a bitmap.
    //

    NumberOfBitmaps = 4;

    //
    // Initialize the GRAPH_INFO structure with all the sizes captured earlier.
    // (We zero it first just to ensure any of the padding fields are cleared.)
    //

    Info->Context = Context;
    Info->AllocSize = AllocSize.QuadPart;
    Info->NumberOfBitmaps = NumberOfBitmaps;
    Info->SizeOfGraphStruct = sizeof(GRAPH);
    Info->EdgesSizeInBytes = EdgesSizeInBytes;
    Info->NextSizeInBytes = NextSizeInBytes;
    Info->OrderSizeInBytes = OrderSizeInBytes;
    Info->FirstSizeInBytes = FirstSizeInBytes;
    Info->Vertices3SizeInBytes = Vertices3SizeInBytes;
    Info->AssignedSizeInBytes = AssignedSizeInBytes;
    Info->VertexPairsSizeInBytes = VertexPairsSizeInBytes;
    Info->ValuesSizeInBytes = ValuesSizeInBytes;

    Info->DeletedEdgesBitmapBufferSizeInBytes = (
        DeletedEdgesBitmapBufferSizeInBytes.QuadPart
    );

    Info->VisitedVerticesBitmapBufferSizeInBytes = (
        VisitedVerticesBitmapBufferSizeInBytes.QuadPart
    );

    Info->AssignedBitmapBufferSizeInBytes = (
        AssignedBitmapBufferSizeInBytes.QuadPart
    );

    Info->IndexBitmapBufferSizeInBytes = (
        IndexBitmapBufferSizeInBytes.QuadPart
    );

    //
    // Capture the system allocation granularity.  This is used to align the
    // backing memory maps used for the table array.
    //

    GetSystemInfo(&SystemInfo);
    Info->AllocationGranularity = SystemInfo.dwAllocationGranularity;

    //
    // Copy the dimensions over.
    //

    Dim = &Info->Dimensions;
    Dim->NumberOfEdges = NumberOfEdges.LowPart;
    Dim->TotalNumberOfEdges = TotalNumberOfEdges.LowPart;
    Dim->NumberOfVertices = NumberOfVertices.LowPart;

    Dim->NumberOfEdgesPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpPowerOfTwo32(NumberOfEdges.LowPart))
    );

    Dim->NumberOfEdgesNextPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpNextPowerOfTwo32(NumberOfEdges.LowPart))
    );

    Dim->NumberOfVerticesPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpPowerOfTwo32(NumberOfVertices.LowPart))
    );

    Dim->NumberOfVerticesNextPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpNextPowerOfTwo32(NumberOfVertices.LowPart))
    );

    //
    // If non-modulus masking is active, initialize the edge and vertex masks
    // and underlying table data type.
    //

    if (!IsModulusMasking(MaskFunctionId)) {

        ULONG_PTR EdgeValue;

        Info->EdgeMask = NumberOfEdges.LowPart - 1;
        Info->VertexMask = NumberOfVertices.LowPart - 1;

        NumberOfEdgeMaskBits = PopulationCount32(Info->EdgeMask);
        NumberOfVertexMaskBits = PopulationCount32(Info->VertexMask);

        //
        // Sanity check our masks are correct: their popcnts should match the
        // exponent value identified above whilst filling out the dimensions
        // structure.
        //

        if (NumberOfEdgeMaskBits != Dim->NumberOfEdgesPowerOf2Exponent) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareGraphInfoChm01_EdgeMaskPopcountMismatch, Result);
            goto Error;
        }

        if (NumberOfVertexMaskBits != Dim->NumberOfVerticesPowerOf2Exponent) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareGraphInfoChm01_VertexMaskPopcountMismatch, Result);
            goto Error;
        }

        //
        // Use the number of bits set in the edge mask to derive an appropriate
        // containing data type (i.e. USHORT if there are less than 65k keys,
        // ULONG otherwise, etc).  The Table->TableDataArrayType field is used
        // as the type for the TableData/Assigned array; if we have less than
        // 65k keys, we know that the two values derived from this table will
        // never exceed 65536, so the array can be USHORT, saving 2 bytes per
        // entry vs keeping it at ULONG.  The 256->65536 USHORT sweet spot is
        // common enough (based on the typical number of keys we're targeting)
        // to warrant this logic.
        //

        EdgeValue = (ULONG_PTR)1 << NumberOfEdgeMaskBits;
        Result = GetContainingType(Rtl, EdgeValue, &Table->TableDataArrayType);
        if (FAILED(Result)) {
            PH_ERROR(PrepareGraphInfoChm01_GetContainingType, Result);
            goto Error;
        }

        Table->TableDataArrayTypeName = &TypeNames[Table->TableDataArrayType];

        //
        // Default the table values type name to ULONG for now until we add
        // more comprehensive support for varying the type name (i.e. wire up
        // a command line parameter to it at the very least).  Ditto for keys.
        //

        Table->ValueType = LongType;
        Table->TableValuesArrayTypeName = &TypeNames[LongType];
        Table->KeysArrayTypeName = &TypeNames[LongType];

    } else {

        //
        // Default names to something sensible if modulus masking is active.
        //

        Table->ValueType = LongType;
        Table->TableDataArrayTypeName = &TypeNames[LongType];
        Table->TableValuesArrayTypeName = &TypeNames[LongType];
        Table->KeysArrayTypeName = &TypeNames[LongType];
    }

    Table->SeedTypeName = &TypeNames[LongType];
    Table->IndexTypeName = &TypeNames[LongType];
    Table->ValueTypeName = &TypeNames[Table->ValueType];
    Table->KeySizeTypeName = &TypeNames[Table->Keys->KeySizeType];
    Table->OriginalKeySizeTypeName =
        &TypeNames[Table->Keys->OriginalKeySizeType];

    //
    // Set the Modulus, Size, Shift, Mask and Fold fields of the table, such
    // that the Hash and Mask vtbl functions operate correctly.
    //
    // N.B. Shift, Mask and Fold are meaningless for modulus masking.
    //
    // N.B. If you change these fields, you'll probably need to change something
    //      in LoadPerfectHashTableImplChm01() too.
    //

    Table->HashModulus = NumberOfVertices.LowPart;
    Table->IndexModulus = NumberOfEdges.LowPart;
    Table->HashSize = NumberOfVertices.LowPart;
    Table->IndexSize = NumberOfEdges.LowPart;
    Table->HashShift = 32 - Rtl->TrailingZeros32(Table->HashSize);
    Table->IndexShift = 32 - Rtl->TrailingZeros32(Table->IndexSize);
    Table->HashMask = (Table->HashSize - 1);
    Table->IndexMask = (Table->IndexSize - 1);
    Table->HashFold = Table->HashShift >> 3;
    Table->IndexFold = Table->IndexShift >> 3;

    //
    // Fill out the in-memory representation of the on-disk table/graph info.
    // This is a smaller subset of data needed in order to load a previously
    // solved graph as a perfect hash table.  The data will eventually be
    // written into the NTFS stream :Info.
    //

    ZeroStructPointerInline(GraphInfoOnDisk);
    TableInfoOnDisk->Magic.LowPart = TABLE_INFO_ON_DISK_MAGIC_LOWPART;
    TableInfoOnDisk->Magic.HighPart = TABLE_INFO_ON_DISK_MAGIC_HIGHPART;
    TableInfoOnDisk->SizeOfStruct = sizeof(*GraphInfoOnDisk);
    TableInfoOnDisk->Flags.AsULong = 0;
    TableInfoOnDisk->Flags.UsingAssigned16 = (UseAssigned16 != FALSE);
    TableInfoOnDisk->Concurrency = Context->MaximumConcurrency;
    TableInfoOnDisk->AlgorithmId = Context->AlgorithmId;
    TableInfoOnDisk->MaskFunctionId = Context->MaskFunctionId;
    TableInfoOnDisk->HashFunctionId = Context->HashFunctionId;
    TableInfoOnDisk->KeySizeInBytes = Table->Keys->KeySizeInBytes;
    TableInfoOnDisk->OriginalKeySizeInBytes = (
        Table->Keys->OriginalKeySizeInBytes
    );
    TableInfoOnDisk->HashSize = Table->HashSize;
    TableInfoOnDisk->IndexSize = Table->IndexSize;
    TableInfoOnDisk->HashShift = Table->HashShift;
    TableInfoOnDisk->IndexShift = Table->IndexShift;
    TableInfoOnDisk->HashMask = Table->HashMask;
    TableInfoOnDisk->IndexMask = Table->IndexMask;
    TableInfoOnDisk->HashFold = Table->HashFold;
    TableInfoOnDisk->IndexFold = Table->IndexFold;
    TableInfoOnDisk->HashModulus = Table->HashModulus;
    TableInfoOnDisk->IndexModulus = Table->IndexModulus;
    TableInfoOnDisk->TableDataArrayType = Table->TableDataArrayType;
    TableInfoOnDisk->NumberOfKeys.QuadPart = (
        Table->Keys->NumberOfKeys.QuadPart
    );
    TableInfoOnDisk->NumberOfSeeds = (
        HashRoutineNumberOfSeeds[Table->HashFunctionId]
    );
    TableInfoOnDisk->AssignedElementSizeInBytes = UseAssigned16 ? 2 : 4;

    //
    // This will change based on masking type and whether or not the caller
    // has provided a value for NumberOfTableElements.  For now, keep it as
    // the number of vertices.
    //

    TableInfoOnDisk->NumberOfTableElements.QuadPart = (
        NumberOfVertices.QuadPart
    );

    CopyInline(&GraphInfoOnDisk->Dimensions, Dim, sizeof(*Dim));

    //
    // Capture ratios.
    //

    Table->KeysToEdgesRatio = Table->Keys->KeysToEdgesRatio;

    Table->KeysToVerticesRatio = (DOUBLE)(
        ((DOUBLE)Table->Keys->NumberOfKeys.QuadPart) /
        ((DOUBLE)NumberOfVertices.QuadPart)
    );

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

    return Result;
}


PREPARE_TABLE_OUTPUT_DIRECTORY PrepareTableOutputDirectory;

_Use_decl_annotations_
HRESULT
PrepareTableOutputDirectory(
    PPERFECT_HASH_TABLE Table
    )
{
    HRESULT Result = S_OK;
    ULONG NumberOfResizeEvents;
    ULARGE_INTEGER NumberOfTableElements;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_PATH OutputPath = NULL;
    PPERFECT_HASH_DIRECTORY OutputDir = NULL;
    PPERFECT_HASH_DIRECTORY BaseOutputDirectory;
    PCUNICODE_STRING BaseOutputDirectoryPath;
    const UNICODE_STRING EmptyString = { 0 };

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    //
    // Invariant check: if Table->OutputDirectory is set, ensure the table
    // requires renames after table resize events.
    //

    if (Table->OutputDirectory) {
        if (!TableResizeRequiresRename(Table)) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareTableOutputDirectory_NoRenameRequired, Result);
            goto Error;
        }
    }

    //
    // Initialize aliases.
    //

    Context = Table->Context;
    BaseOutputDirectory = Context->BaseOutputDirectory;
    BaseOutputDirectoryPath = &BaseOutputDirectory->Path->FullPath;
    NumberOfResizeEvents = (ULONG)Context->NumberOfTableResizeEvents;
    NumberOfTableElements.QuadPart = (
        Table->TableInfoOnDisk->NumberOfTableElements.QuadPart
    );

    //
    // Create an output directory path name.
    //

    Result = PerfectHashTableCreatePath(Table,
                                        Table->Keys->File->Path,
                                        &NumberOfResizeEvents,
                                        &NumberOfTableElements,
                                        Table->AlgorithmId,
                                        Table->HashFunctionId,
                                        Table->MaskFunctionId,
                                        BaseOutputDirectoryPath,
                                        NULL,           // NewBaseName
                                        NULL,           // AdditionalSuffix
                                        &EmptyString,   // NewExtension
                                        NULL,           // NewStreamName
                                        &OutputPath,
                                        NULL);

    if (FAILED(Result)) {
        PH_ERROR(PrepareTableOutputDirectory_CreatePath, Result);
        goto Error;
    }

    ASSERT(IsValidUnicodeString(&OutputPath->FullPath));

    //
    // Release the existing output path, if applicable.  (This will already
    // have a value if we're being called for the second or more time due to
    // a resize event.)
    //

    RELEASE(Table->OutputPath);

    Table->OutputPath = OutputPath;

    //
    // Either create a new directory instance if this is our first pass, or
    // schedule a rename if not.
    //

    if (!Table->OutputDirectory) {

        PERFECT_HASH_DIRECTORY_CREATE_FLAGS DirectoryCreateFlags = { 0 };

        //
        // No output directory has been set; this is the first attempt at
        // trying to solve the graph.  Create a new directory instance, then
        // issue a Create() call against the output path we constructed above.
        //

        Result = Table->Vtbl->CreateInstance(Table,
                                             NULL,
                                             &IID_PERFECT_HASH_DIRECTORY,
                                             (PVOID *)&OutputDir);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashDirectoryCreateInstance, Result);
            goto Error;
        }

        Result = OutputDir->Vtbl->Create(OutputDir,
                                         OutputPath,
                                         &DirectoryCreateFlags);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashDirectoryCreate, Result);
            goto Error;
        }

        //
        // Directory creation was successful.
        //

        Table->OutputDirectory = OutputDir;

    } else {

        //
        // Directory already exists; a resize event must have occurred.
        // Schedule a rename of the directory to the output path constructed
        // above.
        //

        OutputDir = Table->OutputDirectory;
        Result = OutputDir->Vtbl->ScheduleRename(OutputDir, OutputPath);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashDirectoryScheduleRename, Result);
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
    // Intentional follow-on to End.
    //

End:

    return Result;
}

_Use_decl_annotations_
HRESULT
LoadPerfectHashTableImplChm01(
    PPERFECT_HASH_TABLE Table
    )
/*++

Routine Description:

    Loads a previously created perfect hash table.

Arguments:

    Table - Supplies a pointer to a partially-initialized PERFECT_HASH_TABLE
        structure.

Return Value:

    S_OK - Table was loaded successfully.

--*/
{
    PTABLE_INFO_ON_DISK OnDisk;

    OnDisk = Table->TableInfoOnDisk;

    Table->HashSize = OnDisk->HashSize;
    Table->IndexSize = OnDisk->IndexSize;
    Table->HashShift = OnDisk->HashShift;
    Table->IndexShift = OnDisk->IndexShift;
    Table->HashMask = OnDisk->HashMask;
    Table->IndexMask = OnDisk->IndexMask;
    Table->HashFold = OnDisk->HashFold;
    Table->IndexFold = OnDisk->IndexFold;
    Table->HashModulus = OnDisk->HashModulus;
    Table->IndexModulus = OnDisk->IndexModulus;

    return S_OK;
}

//
// The entry point into the actual per-thread solving attempts is the following
// routine.
//

_Use_decl_annotations_
VOID
ProcessGraphCallbackChm01(
    PTP_CALLBACK_INSTANCE Instance,
    PPERFECT_HASH_CONTEXT Context,
    PLIST_ENTRY ListEntry
    )
/*++

Routine Description:

    This routine is the callback entry point for graph solving threads.  It
    will enter an infinite loop attempting to solve the graph; terminating
    only when the graph is solved or we detect another thread has solved it.

Arguments:

    Instance - Supplies a pointer to the callback instance for this invocation.

    Context - Supplies a pointer to the active context for the graph solving.

    ListEntry - Supplies a pointer to the list entry that was removed from the
        context's main work list head.  The list entry will be the address of
        Graph->ListEntry, and thus, the Graph address can be obtained via the
        following CONTAINING_RECORD() construct:

            Graph = CONTAINING_RECORD(ListEntry, GRAPH, ListEntry);


Return Value:

    None.

--*/
{
    PGRAPH Graph;
    HRESULT Result;
    PHANDLE Event;
    ULONG WaitResult;

    UNREFERENCED_PARAMETER(Instance);

    InterlockedIncrement(&Context->ActiveSolvingLoops);

    //
    // Resolve the graph from the list entry then enter the solving loop.
    //

    Graph = CONTAINING_RECORD(ListEntry, GRAPH, ListEntry);

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
            Result == E_OUTOFMEMORY      ||
            Result == PH_E_NO_MORE_SEEDS
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

#ifdef PH_WINDOWS
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
PrintCurrentContextStatsChm01(
    _In_ PPERFECT_HASH_CONTEXT Context
    )
/*++

Routine Description:

    This routine constructs a textual representation of the current context's
    solving status and then prints it to stdout.

Arguments:

    Context - Supplies a pointer to the active context.

Return Value:

    S_OK on success, an appropriate error code otherwise.

--*/
{
    LONG Chars;
    BOOL Success;
    PCHAR Output;
    PGRAPH Graph;
    PCHAR Buffer;
    HRESULT Result;
    HANDLE OutputHandle;
    FILETIME64 FileTime;
    SYSTEMTIME LocalTime;
    ULONG PredictedAttempts;
    ULARGE_INTEGER Duration;
    LONGLONG CurrentAttempts;
    DOUBLE AttemptsPerSecond;
    DOUBLE DurationInSeconds;
    ULONGLONG GraphSizeInBytes;
    DOUBLE CurrentAttemptsPerSecond;
    DOUBLE CurrentDurationInSeconds;
    PFILETIME64 TableFileTime;
    DOUBLE SolutionsFoundRatio;
    ULONGLONG BufferSizeInBytes;
    ULARGE_INTEGER BytesWritten;
    DOUBLE SecondsUntilNextSolve;
    UNICODE_STRING DurationString;
    WCHAR DurationBuffer[80] = { 0, };
    PASSIGNED_MEMORY_COVERAGE Coverage;
    PASSIGNED16_MEMORY_COVERAGE Coverage16;
    UNICODE_STRING SolvedDurationString;
    ULARGE_INTEGER DurationSinceLastBest;
    WCHAR SolvedDurationBuffer[80] = { 0, };
    LARGE_INTEGER PredictedAttemptsRemaining;
    UNICODE_STRING DurationSinceLastBestString;
    WCHAR DurationSinceLastBestBuffer[80] = { 0, };
    DOUBLE VertexCollisionToCyclicGraphFailureRatio;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    CONST WCHAR FormatString[] = L"h' hours, 'm' mins, 's 'secs'";

#define CONTEXT_STATS_TABLE(ENTRY)                             \
    ENTRY(                                                     \
        "Keys File Name:                                    ", \
        &Context->Table->Keys->File->Path->FileName,           \
        OUTPUT_UNICODE_STRING_FAST                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Number of Keys:                                    ", \
        Context->Table->Keys->NumberOfKeys.QuadPart,           \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Number of Table Resize Events:                     ", \
        Context->NumberOfTableResizeEvents,                    \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Keys to Edges Ratio:                               ", \
        Context->Table->KeysToEdgesRatio,                      \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Duration:                                          ", \
        &DurationString,                                       \
        OUTPUT_UNICODE_STRING_FAST                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Duration Since Last Best Graph:                    ", \
        &DurationSinceLastBestString,                          \
        OUTPUT_UNICODE_STRING_FAST                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Attempts:                                          ", \
        Context->Attempts,                                     \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Attempts Per Second:                               ", \
        AttemptsPerSecond,                                     \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Current Attempts:                                  ", \
        CurrentAttempts,                                       \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Current Attempts Per Second:                       ", \
        CurrentAttemptsPerSecond,                              \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Successful Attempts:                               ", \
        Context->FinishedCount,                                \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Failed Attempts:                                   ", \
        Context->FailedAttempts,                               \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "First Attempt Solved:                              ", \
        Context->FirstAttemptSolved,                           \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Most Recent Attempt Solved:                        ", \
        Context->MostRecentSolvedAttempt,                      \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Predicted Attempts to Solve:                       ", \
        PredictedAttempts,                                     \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Predicted Attempts Remaining until next Solve:     ", \
        PredictedAttemptsRemaining.QuadPart,                   \
        OUTPUT_SIGNED_INT                                      \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Estimated Seconds until next Solve:                ", \
        SecondsUntilNextSolve,                                 \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "New Best Graph Count:                              ", \
        Context->NewBestGraphCount,                            \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Equal Best Graph Count:                            ", \
        Context->EqualBestGraphCount,                          \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Solutions Found Ratio:                             ", \
        SolutionsFoundRatio,                                   \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Vertex Collision Failures:                         ", \
        Context->VertexCollisionFailures,                      \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Cyclic Graph Failures:                             ", \
        Context->CyclicGraphFailures,                          \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Vertex Collision to Cyclic Graph Failure Ratio:    ", \
        VertexCollisionToCyclicGraphFailureRatio,              \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "Highest Deleted Edges Count:                       ", \
        Context->HighestDeletedEdgesCount,                     \
        OUTPUT_INT                                             \
    )                                                          \


#define CONTEXT_STATS_BEST_GRAPH_TABLE(ENTRY)                  \
    ENTRY(                                                     \
        "    Attempt:                                       ", \
        Graph->Attempt,                                        \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Solution Number:                               ", \
        Graph->SolutionNumber,                                 \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Best Graph Number:                             ", \
        Coverage->BestGraphNumber,                             \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Solved After:                                  ", \
        &SolvedDurationString,                                 \
        OUTPUT_UNICODE_STRING_FAST                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Number of Collisions During Assignment:        ", \
        Graph->Collisions,                                     \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Max Graph Traversal Depth:                     ", \
        Graph->MaximumTraversalDepth,                          \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Graph Size In Bytes:                           ", \
        GraphSizeInBytes,                                      \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Total Graph Traversals:              ", \
        Coverage->TotalGraphTraversals,                        \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Score:                               ", \
        Coverage->Score,                                       \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Rank:                                ", \
        Coverage->Rank,                                        \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Slope:                               ", \
        Coverage->Slope,                                       \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Intercept:                           ", \
        Coverage->Intercept,                                   \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Correlation Coefficient:             ", \
        Coverage->CorrelationCoefficient,                      \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Number of Empty Cache Lines          ", \
        Coverage->NumberOfEmptyCacheLines,                     \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: First Cache Line Used:               ", \
        Coverage->FirstCacheLineUsed,                          \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Last Cache Line Used:                ", \
        Coverage->LastCacheLineUsed,                           \
        OUTPUT_INT                                             \
    )                                                          \

#define CONTEXT_STATS_BEST_GRAPH16_TABLE(ENTRY)                \
    ENTRY(                                                     \
        "    Attempt:                                       ", \
        Graph->Attempt,                                        \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Solution Number:                               ", \
        Graph->SolutionNumber,                                 \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Best Graph Number:                             ", \
        Coverage16->BestGraphNumber,                           \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Solved After:                                  ", \
        &SolvedDurationString,                                 \
        OUTPUT_UNICODE_STRING_FAST                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Number of Collisions During Assignment:        ", \
        Graph->Collisions,                                     \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Max Graph Traversal Depth:                     ", \
        Graph->MaximumTraversalDepth,                          \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Graph Size In Bytes:                           ", \
        GraphSizeInBytes,                                      \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Total Graph Traversals:              ", \
        Coverage16->TotalGraphTraversals,                      \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Score:                               ", \
        Coverage16->Score,                                     \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Rank:                                ", \
        Coverage16->Rank,                                      \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Slope:                               ", \
        Coverage16->Slope,                                     \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Intercept:                           ", \
        Coverage16->Intercept,                                 \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Correlation Coefficient:             ", \
        Coverage16->CorrelationCoefficient,                    \
        OUTPUT_DOUBLE                                          \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Number of Empty Cache Lines          ", \
        Coverage16->NumberOfEmptyCacheLines,                   \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: First Cache Line Used:               ", \
        Coverage16->FirstCacheLineUsed,                        \
        OUTPUT_INT                                             \
    )                                                          \
                                                               \
    ENTRY(                                                     \
        "    Coverage: Last Cache Line Used:                ", \
        Coverage16->LastCacheLineUsed,                         \
        OUTPUT_INT                                             \
    )



#define EXPAND_STATS_ROW(Name, Value, OutputMacro) \
    OUTPUT_RAW(Name);                              \
    OutputMacro(Value);                            \
    OUTPUT_CHR('\n');

    //
    // Fast-path exit if we're not in quiet mode.
    //

    TableCreateFlags.AsULongLong = Context->Table->TableCreateFlags.AsULongLong;

    //
    // Initialize aliases.
    //

    OutputHandle = Context->OutputHandle;
    Buffer = Context->ConsoleBuffer;
    BufferSizeInBytes = Context->ConsoleBufferSizeInBytes;

    //
    // Dummy write to suppress SAL warning re uninitialized memory being used.
    //

    Output = Buffer;
    *Output = '\0';
    OUTPUT_CHR('\n');

    //
    // Capture the local system time, then convert into a friendly h/m/s
    // duration format.
    //

    GetLocalTime(&LocalTime);
    if (!SystemTimeToFileTime(&LocalTime, &FileTime.AsFileTime)) {
        SYS_ERROR(SystemTimeToFileTime);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto End;
    }

    TableFileTime = &Context->Table->FileTime;
    Duration.QuadPart = FileTime.AsULongLong - TableFileTime->AsULongLong;

    Chars = GetDurationFormatEx(NULL,
                                0,
                                NULL,
                                Duration.QuadPart,
                                FormatString,
                                &DurationBuffer[0],
                                ARRAYSIZE(DurationBuffer));

    if (Chars == 0) {
        SYS_ERROR(GetDurationFormatEx_TableSolving);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto End;
    }

    DurationString.Buffer = &DurationBuffer[0];
    DurationString.Length = (USHORT)(Chars << 1);
    DurationString.MaximumLength = (USHORT)(Chars << 1);

    //
    // Calculate total attempts per second.
    //

    DurationInSeconds = (DOUBLE)(
        ((DOUBLE)Duration.QuadPart) /
        ((DOUBLE)1e7)
    );

    AttemptsPerSecond = (((DOUBLE)Context->Attempts) / DurationInSeconds);

    //
    // Calculate "current" values.
    //

    CurrentAttempts = Context->Attempts - Context->BaselineAttempts;
    if (Context->BaselineFileTime.AsULongLong > 0) {
        Duration.QuadPart = (
            FileTime.AsULongLong -
            Context->BaselineFileTime.AsULongLong
        );

        CurrentDurationInSeconds = (DOUBLE)(
            ((DOUBLE)Duration.QuadPart) /
            ((DOUBLE)1e7)
        );
    } else {
        CurrentDurationInSeconds = DurationInSeconds;
    }

    CurrentAttemptsPerSecond = (
        ((DOUBLE)CurrentAttempts) / CurrentDurationInSeconds
    );

    //
    // If we've found at least one solution, calculate solutions found ratio
    // and perform some predictions based on the current solving rate.
    //

    if (Context->FinishedCount > 0) {
        SolutionsFoundRatio = (DOUBLE)(
            ((DOUBLE)Context->FinishedCount) /
            ((DOUBLE)Context->Attempts)
        );
        Result = CalculatePredictedAttempts(SolutionsFoundRatio,
                                            &PredictedAttempts);
        if (FAILED(Result)) {
            PH_ERROR(PrintCurrentContextStatsChm01_CalculatePredictedAttempts,
                     Result);
            goto End;
        }
        ASSERT(Context->MostRecentSolvedAttempt != 0);
        PredictedAttemptsRemaining.QuadPart = (LONGLONG)(
            ((LONGLONG)PredictedAttempts) - (
                ((LONGLONG)Context->Attempts) -
                ((LONGLONG)Context->MostRecentSolvedAttempt)
            )
        );
        if (PredictedAttemptsRemaining.QuadPart > 0) {
            SecondsUntilNextSolve = (
                ((DOUBLE)PredictedAttemptsRemaining.QuadPart) /
                AttemptsPerSecond
            );
        } else {
            SecondsUntilNextSolve = (
                ((DOUBLE)(PredictedAttemptsRemaining.QuadPart * -1LL)) /
                AttemptsPerSecond
            ) * -1;
        }

    } else {
        PredictedAttempts = 0;
        SolutionsFoundRatio = 0.0;
        SecondsUntilNextSolve = 0.0;
        PredictedAttemptsRemaining.QuadPart = 0;
    }

    if ((Context->VertexCollisionFailures > 0) &&
        (Context->CyclicGraphFailures > 0)) {

        VertexCollisionToCyclicGraphFailureRatio = (DOUBLE)(
            ((DOUBLE)Context->VertexCollisionFailures) /
            ((DOUBLE)Context->CyclicGraphFailures)
        );
    } else {
        VertexCollisionToCyclicGraphFailureRatio = 0.0;
    }

    //
    // If there's a best graph, construct a friendly h/m/s duration
    // representation for when it was solved.
    //

    _No_competing_thread_begin_
    Graph = Context->BestGraph;
    if (Graph != NULL) {

        //
        // Capture the size in bytes required for each graph instance.
        //

        GraphSizeInBytes = sizeof(GRAPH);
        GraphSizeInBytes+= Graph->Info->AllocSize;

        Duration.QuadPart = (
            Graph->SolvedTime.AsULongLong -
            TableFileTime->AsULongLong
        );

        Chars = GetDurationFormatEx(NULL,
                                    0,
                                    NULL,
                                    Duration.QuadPart,
                                    FormatString,
                                    &SolvedDurationBuffer[0],
                                    ARRAYSIZE(SolvedDurationBuffer));

        if (Chars == 0) {
            SYS_ERROR(GetDurationFormatEx_GraphSolvedDuration);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto End;
        }

        SolvedDurationString.Buffer = &SolvedDurationBuffer[0];
        SolvedDurationString.Length = (USHORT)(Chars << 1);
        SolvedDurationString.MaximumLength = (USHORT)(Chars << 1);

        //
        // Calculate the duration since we last found a best graph.
        //

        DurationSinceLastBest.QuadPart = (
            FileTime.AsULongLong - Graph->SolvedTime.AsULongLong
        );

        Chars = GetDurationFormatEx(NULL,
                                    0,
                                    NULL,
                                    DurationSinceLastBest.QuadPart,
                                    FormatString,
                                    &DurationSinceLastBestBuffer[0],
                                    ARRAYSIZE(DurationSinceLastBestBuffer));

        if (Chars == 0) {
            SYS_ERROR(GetDurationFormatEx_GraphDurationSinceLastBest);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto End;
        }

        DurationSinceLastBestString.Buffer = &DurationSinceLastBestBuffer[0];
        DurationSinceLastBestString.Length = (USHORT)(Chars << 1);
        DurationSinceLastBestString.MaximumLength = (USHORT)(Chars << 1);
    } else {
        GraphSizeInBytes = 0;
        DurationSinceLastBestString.Buffer = NULL;
        DurationSinceLastBestString.Length = 0;
        DurationSinceLastBestString.MaximumLength = 0;
    }

    _No_competing_thread_end_

    //
    // Final step: set the global _dtoa_Allocator.
    //

    _dtoa_Allocator = Context->Allocator;

    //
    // If this traps, expand the number of pages being used for the buffer.
    // (Grep for 'NumberOfPagesForConsoleBuffer'.)
    //

    _No_competing_thread_begin_
    CONTEXT_STATS_TABLE(EXPAND_STATS_ROW);
    if (Graph != NULL) {
        CHAR Char;
        ULONG Index;

        if (IsUsingAssigned16(Graph)) {
            Coverage16 = &Graph->Assigned16MemoryCoverage;
            OUTPUT_RAW("Best Graph:\n");
            CONTEXT_STATS_BEST_GRAPH16_TABLE(EXPAND_STATS_ROW);
        } else {
            Coverage = &Graph->AssignedMemoryCoverage;
            OUTPUT_RAW("Best Graph:\n");
            CONTEXT_STATS_BEST_GRAPH_TABLE(EXPAND_STATS_ROW);
        }

        for (Index = 0, Char = '1';
             Index < Graph->NumberOfSeeds;
             Index++, Char++) {

            OUTPUT_RAW("    Seed ");
            OUTPUT_CHR(Char);
            OUTPUT_RAW(":                                        0x");
            OUTPUT_HEX_RAW(Graph->Seeds[Index]);
            OUTPUT_CHR('\n');
        }
    }
    _No_competing_thread_end_

    BytesWritten.QuadPart = RtlPointerToOffset(Buffer, Output);

    ASSERT(BytesWritten.HighPart == 0);
    ASSERT(BytesWritten.LowPart <= BufferSizeInBytes);

    Success = WriteFile(OutputHandle,
                        Buffer,
                        BytesWritten.LowPart,
                        NULL,
                        NULL);

    if (!Success) {
        SYS_ERROR(WriteFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
    } else {
        Result = S_OK;
    }

    PerfectHashPrintMessage(PH_MSG_PERFECT_HASH_CONSOLE_KEYS_HELP);

    //
    // Intentional follow-on to End.
    //

End:

    _dtoa_Allocator = NULL;

    return Result;
}

_Use_decl_annotations_
VOID
ProcessConsoleCallbackChm01(
    PPERFECT_HASH_CONTEXT Context
    )
/*++

Routine Description:

    This routine is the callback for console interaction during graph solving.

Arguments:

    Context - Supplies a pointer to the active context.

Return Value:

    None.

--*/
{
    COORD Coord;
    BOOL Success;
    BOOL IsFinish;
    BOOL IsRefresh;
    BOOL IsResize;
    BOOL IsVerbose;
    BOOL IsMoreHelp;
    BOOL IsToggleCallback;
    BOOL DoFlushOnExit;
    HRESULT Result;
    ULONG WaitResult;
    DWORD LastError;
    HANDLE InputHandle;
    INPUT_RECORD Input;
    FILETIME64 FileTime;
    SYSTEMTIME LocalTime;
    ULONG NumberOfEvents;
    DWORD NumberOfEventsRead;
    HANDLE Events[3] = { 0, };
    KEY_EVENT_RECORD *KeyEvent;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PSET_FUNCTION_ENTRY_CALLBACK SetFunctionEntryCallback;
    PCLEAR_FUNCTION_ENTRY_CALLBACK ClearFunctionEntryCallback;
    PIS_FUNCTION_ENTRY_CALLBACK_ENABLED IsFunctionEntryCallbackEnabled;

    //
    // Initialize aliases.
    //

    InputHandle = Context->InputHandle;
    TableCreateFlags.AsULongLong = Context->Table->TableCreateFlags.AsULongLong;
    SetFunctionEntryCallback = Context->SetFunctionEntryCallback;
    ClearFunctionEntryCallback = Context->ClearFunctionEntryCallback;
    IsFunctionEntryCallbackEnabled = Context->IsFunctionEntryCallbackEnabled;

    //
    // Initialize the event handles upon which we will wait.
    //

    NumberOfEvents = 0;
    Events[NumberOfEvents++] = Context->ShutdownEvent;
    Events[NumberOfEvents++] = InputHandle;

    if (!TableCreateFlags.Quiet) {
        Events[NumberOfEvents++] = Context->NewBestGraphFoundEvent;
    } else {

        //
        // A user can toggle verbose on and off via the 'v' key, so we always
        // wire up the NewBestGraphFoundEvent to the third event handle.
        //

        ASSERT(NumberOfEvents == 2);
        Events[NumberOfEvents] = Context->NewBestGraphFoundEvent;
    }

    //
    // Enter our console loop.
    //

    DoFlushOnExit = TRUE;

    while (TRUE) {

        WaitResult = WaitForMultipleObjects(NumberOfEvents,
                                            Events,
                                            FALSE,
                                            INFINITE);

        if (StopSolving(Context) || (WaitResult == WAIT_OBJECT_0)) {
            goto End;
        }

        if ((WaitResult != WAIT_OBJECT_0+1) &&
            (WaitResult != WAIT_OBJECT_0+2)) {
            SYS_ERROR(WaitForSingleObject);
            goto End;
        }

        if (WaitResult == WAIT_OBJECT_0+2) {

            //
            // A new best graph has been found.
            //

            IsRefresh = TRUE;
            IsResize = FALSE;
            IsFinish = FALSE;
            IsVerbose = FALSE;
            IsMoreHelp = FALSE;
            IsToggleCallback = FALSE;

            //
            // As all the context's events are created as manual reset, we need
            // to explicitly reset the new best graph event here.
            //

            if (!ResetEvent(Context->NewBestGraphFoundEvent)) {
                SYS_ERROR(ResetEvent);
                goto End;
            }

        } else {

            ASSERT(WaitResult == WAIT_OBJECT_0+1);

            //
            // The console input handle is signaled, proceed with reading.
            //

            Success = ReadConsoleInput(InputHandle,
                                       &Input,
                                       1,
                                       &NumberOfEventsRead);

            if (!Success) {
                LastError = GetLastError();

                //
                // ERROR_INVALID_FUNCTION will be returned if the console has
                // been redirected or isn't otherwise available; this is not
                // considered fatal.  Otherwise, log the error.  In all cases,
                // disable the final console flush and exit the routine.
                //

                if (LastError != ERROR_INVALID_FUNCTION) {
                    SYS_ERROR(ReadConsoleInput);
                }

                DoFlushOnExit = FALSE;
                goto End;
            }

            ASSERT(NumberOfEventsRead == 1);

            if (Input.EventType == WINDOW_BUFFER_SIZE_EVENT) {

                //
                // This will typically be the first event.  Capture the
                // coordinates and continue.  Note that we don't actually do
                // anything with the coords currently.
                //

                Coord.X = Input.Event.WindowBufferSizeEvent.dwSize.X;
                Coord.Y = Input.Event.WindowBufferSizeEvent.dwSize.Y;
                continue;
            }

            if (Input.EventType != KEY_EVENT) {

                //
                // We're not interested in anything other than key events
                // herein.
                //

                continue;
            }

            KeyEvent = &Input.Event.KeyEvent;

            if (KeyEvent->bKeyDown) {

                //
                // We don't care about key-down events, only key-up.
                //

                continue;
            }

            IsRefresh = (
                (KeyEvent->uChar.AsciiChar == 'r') ||
                (KeyEvent->uChar.AsciiChar == 'R') ||
                (KeyEvent->wVirtualKeyCode == 0x0052)
            );

            IsResize = (
                (KeyEvent->uChar.AsciiChar == 'e') ||
                (KeyEvent->uChar.AsciiChar == 'E') ||
                (KeyEvent->wVirtualKeyCode == 0x0045)
            );

            IsFinish = (
                (KeyEvent->uChar.AsciiChar == 'f') ||
                (KeyEvent->uChar.AsciiChar == 'F') ||
                (KeyEvent->wVirtualKeyCode == 0x0045)
            );

            IsVerbose = (
                (KeyEvent->uChar.AsciiChar == 'v') ||
                (KeyEvent->uChar.AsciiChar == 'V') ||
                (KeyEvent->wVirtualKeyCode == 0x0056)
            );

            IsMoreHelp = (
                (KeyEvent->uChar.AsciiChar == '?') ||
                (KeyEvent->wVirtualKeyCode == VK_OEM_2)
            );

            IsToggleCallback = (
                (KeyEvent->uChar.AsciiChar == 'c') ||
                (KeyEvent->uChar.AsciiChar == 'C') ||
                (KeyEvent->wVirtualKeyCode == 0x0043)
            );
        }

        if (IsFinish) {

            SetStopSolving(Context);
            if (!SetEvent(Context->ShutdownEvent)) {
                SYS_ERROR(SetEvent);
            }
            break;

        } else if (IsResize) {

            Context->State.UserRequestedResize = TRUE;
            SetStopSolving(Context);
            if (!SetEvent(Context->TryLargerTableSizeEvent)) {
                SYS_ERROR(SetEvent);
            }
            break;

        } else if (IsRefresh) {

            Result = PrintCurrentContextStatsChm01(Context);
            if (FAILED(Result)) {
                PH_ERROR(PrintCurrentContextStatsChm01, Result);

                //
                // We don't break here; it's easier during development (which
                // is the only time this code path should be hit, i.e. because
                // we've broken PrintCurrentContextStatsChm01()) to continue
                // and process more key presses.
                //
            }

        } else if (IsVerbose) {

            //
            // Toggle quiet mode on or off.
            //

            if (NumberOfEvents == 3) {
                NumberOfEvents = 2;
                Context->Table->TableCreateFlags.Quiet = TRUE;
            } else {
                ASSERT(NumberOfEvents == 2);
                NumberOfEvents = 3;
                Context->Table->TableCreateFlags.Quiet = FALSE;
                ASSERT(
                    Events[NumberOfEvents-1] ==
                    Context->NewBestGraphFoundEvent
                );
            }

        } else if (IsToggleCallback) {

            if (!Context->State.HasFunctionHooking) {
                continue;
            }

            if (IsFunctionEntryCallbackEnabled()) {
                ClearFunctionEntryCallback(&Context->CallbackFunction,
                                           &Context->CallbackContext,
                                           &Context->CallbackModuleBaseAddress,
                                           &Context->CallbackModuleSizeInBytes,
                                           &Context->CallbackModuleIgnoreRip);
            } else {
                SetFunctionEntryCallback(Context->CallbackFunction,
                                         Context->CallbackContext,
                                         Context->CallbackModuleBaseAddress,
                                         Context->CallbackModuleSizeInBytes,
                                         Context->CallbackModuleIgnoreRip);
            }

            //
            // Update the baseline attempts and file time.
            //

            GetLocalTime(&LocalTime);
            if (!SystemTimeToFileTime(&LocalTime, &FileTime.AsFileTime)) {
                SYS_ERROR(SystemTimeToFileTime);
                continue;
            }

            Context->BaselineAttempts = Context->Attempts;
            Context->BaselineFileTime.AsULongLong = FileTime.AsULongLong;

            continue;

        } else if (IsMoreHelp) {

            PerfectHashPrintMessage(PH_MSG_PERFECT_HASH_CONSOLE_KEYS_MORE_HELP);
            continue;

        } else {

            //
            // If we don't recognize the key, just ignore it and continue the
            // loop.
            //

            NOTHING;
        };
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (DoFlushOnExit && !FlushConsoleInputBuffer(InputHandle)) {
        SYS_ERROR(FlushConsoleInputBuffer);
    }
}
#endif // PH_WINDOWS

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
