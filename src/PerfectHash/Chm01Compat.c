/*++

Copyright (c) 2023 Trent Nelson <trent@trent.me>

Module Name:

    Chm01Compat.c

Abstract:

    Non-Windows implementation of CreatePerfectHashTableImplChm01().

--*/

#include "stdafx.h"
#include "Chm01.h"
#include "Chm01Private.h"

#include "bsthreadpool.h"

#ifdef PH_WINDOWS
#error This file is not for Windows.
#endif

#ifndef PH_COMPAT
#error PH_COMPAT not defined.
#endif

//
// Forward decls.
//

VOID
GraphCallbackChm01Compat(
    PGRAPH Graph
    );


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

    PH_I_OUT_OF_MEMORY - The system is out of memory.

    PH_I_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED - The maximum number
        of table resize events was reached before a solution could be found.

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
    ULONG NumberOfFileWorkThreads;
    PTHREADPOOL FileWorkThreadpool = NULL;
    PTHREADPOOL GraphThreadpool = NULL;

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
    NumberOfFileWorkThreads = GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS);

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

#ifdef PH_WINDOWS
        ASSERT(Graph->Allocator != Table->Allocator);
        ASSERT(Graph->Allocator->HeapHandle != Table->Allocator->HeapHandle);
#endif

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

    Context->AlgorithmContext = &Info;

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

#if 0
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
#endif

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
    Verb##Name.Context = Context;                     \
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
        FileWorkThreadpool = Context->FileThreadpool;
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
    // TODO.
    //

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

    GraphThreadpool = Context->MainThreadpool;

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
        Graph->Context = Context;

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
            ThreadpoolAddWork(GraphThreadpool, GraphCallbackChm01Compat, Graph);
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
    // If the wait result indicates the try larger table size event was set,
    // deal with that.
    //

    TryLargerTableSize = (
        WaitForSingleObject(Context->TryLargerTableSizeEvent, 0) ==
        WAIT_OBJECT_0
    );

    if (TryLargerTableSize) {

        //
        // The number of attempts at solving this graph have exceeded the
        // threshold.  Set the shutdown event in order to trigger all worker
        // threads to abort their current attempts and wait on the main thread
        // work, then finish work, to complete.
        //

        if (!NoFileIo(Table)) {

            //
            // Perform a blocking wait for the prepare work to complete.
            //

            ThreadpoolWait(FileWorkThreadpool);

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

#define EXPAND_AS_SUBMIT_CLOSE_FILE_WORK(             \
    Verb, VUpper, Name, Upper,                        \
    EofType, EofValue,                                \
    Suffix, Extension, Stream, Base                   \
)                                                     \
    ASSERT(!NoFileIo(Table));                         \
    ZeroStructInline(Verb##Name);                     \
    Verb##Name.FileWorkId = FileWork##Verb##Name##Id; \
    Verb##Name.EndOfFile = EndOfFile;                 \
    Verb##Name.Context = Context;                     \
    ThreadpoolAddWork(FileWorkThreadpool,             \
                      FileWorkItemCallbackChm01,      \
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

_Use_decl_annotations_
VOID
GraphCallbackChm01Compat(
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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
