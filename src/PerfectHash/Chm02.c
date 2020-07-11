/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    Chm02.c

Abstract:

    This module is a copy of the Chm01.c, modified to explore the viability of
    CUDA support.  It is an experimental work-in-progress.

--*/

#include "stdafx.h"
#include "Chm01.h"

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

    PH_I_LOW_MEMORY - The system is indicating a low-memory state (which we
        may have caused).

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
    PRTL Rtl;
    USHORT Index;
    PULONG Keys;
    PGRAPH *Graphs = NULL;
    PGRAPH Graph;
    BOOLEAN Silent;
    BOOLEAN Success;
    BOOLEAN CreateCuGraph;
    LPGUID InterfaceId;
    ULONG Attempt = 0;
    ULONG ReferenceCount;
    BYTE NumberOfEvents;
    HRESULT Result = S_OK;
    HRESULT CloseResult = S_OK;
    ULONG WaitResult;
    GRAPH_INFO Info;
    PALLOCATOR Allocator;
    HANDLE OutputHandle = NULL;
    PHANDLE Event;
    ULONG Concurrency;
    ULONG CuConcurrency;
    ULONG CuGraphCount = 0;
    ULONG NumberOfGraphs;
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
    PPERFECT_HASH_TLS_CONTEXT TlsContext;
    PERFECT_HASH_TLS_CONTEXT LocalTlsContext;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    LARGE_INTEGER EmptyEndOfFile = { 0 };
    PLARGE_INTEGER EndOfFile;

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

    TableCreateFlags.AsULong = Table->TableCreateFlags.AsULong;
    Silent = (TableCreateFlags.Silent != FALSE);

    Context = Table->Context;
    Concurrency = Context->MaximumConcurrency;
    CuConcurrency = Context->CuConcurrency;
    ASSERT(CuConcurrency <= Concurrency);
    ASSERT(CuConcurrency > 0);

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

    Events[SUCCEEDED_EVENT] = Context->SucceededEvent;
    Events[COMPLETED_EVENT] = Context->CompletedEvent;
    Events[SHUTDOWN_EVENT] = Context->ShutdownEvent;
    Events[FAILED_EVENT] = Context->FailedEvent;
    Events[LOW_MEMORY_EVENT] = Context->LowMemoryEvent;

#define EXPAND_AS_ASSIGN_EVENT(                         \
    Verb, VUpper, Name, Upper,                          \
    EofType, EofValue,                                  \
    Suffix, Extension, Stream, Base                     \
)                                                       \
    *##Verb##Event++ = Context->##Verb##d##Name##Event;

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

    Result = PrepareGraphInfoChm02(Table, &Info, NULL);
    if (FAILED(Result)) {
        PH_ERROR(CreatePerfectHashTableImplChm02_PrepareFirstGraphInfo, Result);
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

    TlsContext->TableCreateFlags.AsULong = TableCreateFlags.AsULong;

    //
    // Toggle the "create CUDA graph" flag in the TLS context, which is also
    // used by GraphInitialize() to determine if a CUDA graph should be created.
    // Initialize the count of CUDA graphs to 0.  We clear the bit once we've
    // created the requested number of CUDA graphs.
    //

    CreateCuGraph = TRUE;
    TlsContext->Flags.CreateCuGraph = TRUE;
    CuGraphCount = 0;

    //
    // Create graph instances and capture the resulting pointer in the array
    // we just allocated above.
    //

    for (Index = 0; Index < NumberOfGraphs; Index++) {

        //
        // If the number of CUDA graphs created equals the desired concurrency
        // level, toggle the CUDA graph flag in the TLS context.
        //

        if (CuGraphCount == CuConcurrency) {
            CreateCuGraph = FALSE;
            TlsContext->Flags.CreateCuGraph = FALSE;
        } else if (CreateCuGraph != FALSE) {
            ASSERT(TlsContext->Flags.CreateCuGraph != FALSE);
            CuGraphCount++;
        }

        if (CreateCuGraph != FALSE) {
            InterfaceId = (LPGUID)&IID_PERFECT_HASH_GRAPH_CU;
        } else {
            InterfaceId = (LPGUID)&IID_PERFECT_HASH_GRAPH;
        }

        Result = Table->Vtbl->CreateInstance(Table,
                                             NULL,
                                             (REFIID)InterfaceId,
                                             &Graph);

        if (FAILED(Result)) {

            //
            // Suppress logging for out-of-memory errors (as we communicate
            // memory issues back to the caller via informational return codes).
            //

            if (Result != E_OUTOFMEMORY) {
                PH_ERROR(CreatePerfectHashTableImplChm02_CreateGraph, Result);
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

        if (CreateCuGraph != FALSE) {
            ASSERT(IsCuGraph(Graph));
        }

        //
        // Copy relevant flags over, then save the graph instance in the array.
        //

        Graph->Flags.SkipVerification = TableCreateFlags.SkipGraphVerification;

        Graph->Flags.WantsWriteCombiningForVertexPairsArray =
            TableCreateFlags.EnableWriteCombineForVertexPairs;

        Graph->Flags.RemoveWriteCombineAfterSuccessfulHashKeys =
            TableCreateFlags.RemoveWriteCombineAfterSuccessfulHashKeys;

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

    PerfectHashTlsClearContextIfActive(&LocalTlsContext);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // N.B. We don't explicitly reset all events here like in Chm01 as we're
    //      not supporting table resizes, so the events should always be reset
    //      at this point.
    //

    //
    // Clear the counter of low-memory events observed.  (An interlocked
    // increment is performed against this counter in various locations
    // each time a wait is satisfied against the low-memory event.)
    //

    Context->LowMemoryObserved = 0;

    //
    // Set the context's main work callback to our worker routine, and the algo
    // context to our graph info structure.
    //

    Context->MainWorkCallback = ProcessGraphCallbackChm02;
    Context->AlgorithmContext = &Info;

    //
    // Set the context's file work callback to our worker routine.
    //

    Context->FileWorkCallback = FileWorkCallbackChm01;

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

#define EXPAND_AS_SUBMIT_FILE_WORK(                       \
    Verb, VUpper, Name, Upper,                            \
    EofType, EofValue,                                    \
    Suffix, Extension, Stream, Base                       \
)                                                         \
    ASSERT(!NoFileIo(Table));                             \
    ZeroStructInline(##Verb####Name##);                   \
    Verb##Name##.FileWorkId = FileWork##Verb##Name##Id;   \
    InsertTailFileWork(Context, &Verb##Name##.ListEntry); \
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

        Graph->Lock.Ptr = NULL;

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

    if (WaitResult == LOW_MEMORY_EVENT) {
        InterlockedIncrement(&Context->LowMemoryObserved);
        Result = PH_I_LOW_MEMORY;
        goto Error;
    }

    //
    // Ignore the remaining results for now; determine if the graph solving was
    // successful by the finished count of the context.  We'll corroborate that
    // with whatever events have been signaled shortly.
    //

    WaitForThreadpoolWorkCallbacks(Context->MainWork, TRUE);
    WaitForThreadpoolWorkCallbacks(Context->FinishedWork, FALSE);

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

        //
        // Wait for the main thread work group members.  This will block until
        // all the worker threads have returned.  We need to put this in place
        // prior to jumping to the End: label as that step will release all the
        // graph references and verify the resulting reference count for each
        // one is 0.  This won't be the case if any threads are still working.
        //

        WaitForThreadpoolWorkCallbacks(Context->MainWork, CancelPending);

        //
        // Perform the same operation for the file work threadpool.  Note that
        // the only work item type we've dispatched to this pool at this point
        // is file preparation work.
        //

        if (!NoFileIo(Table)) {
            WaitForThreadpoolWorkCallbacks(Context->FileWork, CancelPending);
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

    WaitForThreadpoolWorkCallbacks(Context->MainWork, TRUE);
    WaitForThreadpoolWorkCallbacks(Context->FinishedWork, FALSE);

    if (CtrlCPressed) {
        Result = PH_E_CTRL_C_PRESSED;
        goto Error;
    }

    ASSERT(Context->FinishedCount > 0);

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

        if (!Graph) {
            goto Error;
        }
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
    // Copy the cycle counters and elapsed microseconds from the winning graph.
    //

    Table->AddKeysElapsedCycles.QuadPart =
        Graph->AddKeysElapsedCycles.QuadPart;

    Table->HashKeysElapsedCycles.QuadPart =
        Graph->HashKeysElapsedCycles.QuadPart;

    Table->AddHashedKeysElapsedCycles.QuadPart =
        Graph->AddHashedKeysElapsedCycles.QuadPart;

    Table->AddKeysElapsedMicroseconds.QuadPart =
        Graph->AddKeysElapsedMicroseconds.QuadPart;

    Table->HashKeysElapsedMicroseconds.QuadPart =
        Graph->HashKeysElapsedMicroseconds.QuadPart;

    Table->AddHashedKeysElapsedMicroseconds.QuadPart =
        Graph->AddHashedKeysElapsedMicroseconds.QuadPart;

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
                TableInfoOnDisk->KeySizeInBytes
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
    if (!NoFileIo(Table)) {
        WaitForThreadpoolWorkCallbacks(Context->FileWork, TRUE);
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

#define EXPAND_AS_SUBMIT_CLOSE_FILE_WORK(                 \
    Verb, VUpper, Name, Upper,                            \
    EofType, EofValue,                                    \
    Suffix, Extension, Stream, Base                       \
)                                                         \
    ASSERT(!NoFileIo(Table));                             \
    ZeroStructInline(##Verb####Name##);                   \
    Verb##Name##.FileWorkId = FileWork##Verb##Name##Id;   \
    Verb##Name##.EndOfFile = EndOfFile;                   \
    InsertTailFileWork(Context, &Verb##Name##.ListEntry); \
    SubmitThreadpoolWork(Context->FileWork);

#define SUBMIT_CLOSE_FILE_WORK() \
    CLOSE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_CLOSE_FILE_WORK)

    SUBMIT_CLOSE_FILE_WORK();

    WaitForThreadpoolWorkCallbacks(Context->FileWork, FALSE);

#define EXPAND_AS_CHECK_CLOSE_ERRORS(                                    \
    Verb, VUpper, Name, Upper,                                           \
    EofType, EofValue,                                                   \
    Suffix, Extension, Stream, Base                                      \
)                                                                        \
    if (Verb####Name##.NumberOfErrors > 0) {                             \
        CloseResult = Verb####Name##.LastResult;                         \
        if (CloseResult == S_OK || CloseResult == E_UNEXPECTED) {        \
            CloseResult = PH_E_ERROR_DURING_##VUpper##_##Upper##;        \
        }                                                                \
        PH_ERROR(                                                        \
            CreatePerfectHashTableImplChm02_ErrorDuring##Verb####Name##, \
            Result                                                       \
        );                                                               \
        CloseFileErrorCount++;                                           \
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
HRESULT
LoadPerfectHashTableImplChm02(
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
    return LoadPerfectHashTableImplChm01(Table);
}

PREPARE_GRAPH_INFO PrepareGraphInfoChm02;

_Use_decl_annotations_
HRESULT
PrepareGraphInfoChm02(
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
    ULONG NumberOfKeys;
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
    ULONGLONG EdgesSizeInBytes;
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
    MaskFunctionId = Table->MaskFunctionId;
    GraphInfoOnDisk = Context->GraphInfoOnDisk;
    TableInfoOnDisk = &GraphInfoOnDisk->TableInfoOnDisk;
    TypeNames = Table->CTypeNames;
    TrailingZeros32 = Rtl->TrailingZeros32;
    TrailingZeros64 = Rtl->TrailingZeros64;
    PopulationCount32 = Rtl->PopulationCount32;
    RoundUpPowerOfTwo32 = Rtl->RoundUpPowerOfTwo32;
    RoundUpNextPowerOfTwo32 = Rtl->RoundUpNextPowerOfTwo32;
    TableCreateFlags.AsULong = Table->TableCreateFlags.AsULong;

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

    if (Table->Keys->NumberOfElements.HighPart) {
        return PH_E_TOO_MANY_KEYS;
    }

    NumberOfKeys = Table->Keys->NumberOfElements.LowPart;

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
            PH_ERROR(PrepareGraphInfoChm02_NumEdgesNotNumVerticesDiv2, Result);
            goto Error;
        }
    }

    //
    // Calculate the size required for our bitmap buffers.
    //

    DeletedEdgesBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(TotalNumberOfEdges.QuadPart, 8) >> 3
    );

    if (DeletedEdgesBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm02_DeletedEdgesBitmap, Result);
        goto Error;
    }

    VisitedVerticesBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(NumberOfVertices.QuadPart, 8) >> 3
    );

    if (VisitedVerticesBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm02_VisitedVerticesBitmap, Result);
        goto Error;
    }

    AssignedBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(NumberOfVertices.QuadPart, 8) >> 3
    );

    if (AssignedBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm02_AssignedBitmap, Result);
        goto Error;
    }

    IndexBitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP(NumberOfVertices.QuadPart, 8) >> 3
    );

    if (IndexBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm02_IndexBitmap, Result);
        goto Error;
    }

    //
    // Calculate the sizes required for each of the arrays.
    //

    EdgesSizeInBytes = ALIGN_UP_YMMWORD(
        RTL_ELEMENT_SIZE(GRAPH, Edges) * TotalNumberOfEdges.QuadPart
    );

    NextSizeInBytes = ALIGN_UP_YMMWORD(
        RTL_ELEMENT_SIZE(GRAPH, Next) * TotalNumberOfEdges.QuadPart
    );

    NextSizeInBytes = ALIGN_UP_YMMWORD(
        RTL_ELEMENT_SIZE(GRAPH, Next) * TotalNumberOfEdges.QuadPart
    );

    FirstSizeInBytes = ALIGN_UP_YMMWORD(
        RTL_ELEMENT_SIZE(GRAPH, First) * NumberOfVertices.QuadPart
    );

    AssignedSizeInBytes = ALIGN_UP_YMMWORD(
        RTL_ELEMENT_SIZE(GRAPH, Assigned) * NumberOfVertices.QuadPart
    );

    if (TableCreateFlags.HashAllKeysFirst == FALSE) {
        VertexPairsSizeInBytes = 0;
    } else {
        VertexPairsSizeInBytes = ALIGN_UP_ZMMWORD(
            RTL_ELEMENT_SIZE(GRAPH, VertexPairs) * (ULONGLONG)NumberOfKeys
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
        BYTES_TO_PAGES(NumberOfVertices.QuadPart << ASSIGNED_SHIFT)
    );

    Info->AssignedArrayNumberOfLargePages = (ULONG)(
        BYTES_TO_LARGE_PAGES(NumberOfVertices.QuadPart << ASSIGNED_SHIFT)
    );

    Info->AssignedArrayNumberOfCacheLines = (ULONG)(
        BYTES_TO_CACHE_LINES(NumberOfVertices.QuadPart << ASSIGNED_SHIFT)
    );

    //
    // Array sizes for the number of assigned per page, large page and cache
    // line.
    //

    Info->NumberOfAssignedPerPageSizeInBytes = (
        Info->AssignedArrayNumberOfPages *
        RTL_ELEMENT_SIZE(ASSIGNED_MEMORY_COVERAGE, NumberOfAssignedPerPage)
    );

    Info->NumberOfAssignedPerLargePageSizeInBytes = (
        Info->AssignedArrayNumberOfLargePages *
        RTL_ELEMENT_SIZE(ASSIGNED_MEMORY_COVERAGE, NumberOfAssignedPerLargePage)
    );

    Info->NumberOfAssignedPerCacheLineSizeInBytes = (
        Info->AssignedArrayNumberOfCacheLines *
        RTL_ELEMENT_SIZE(ASSIGNED_MEMORY_COVERAGE, NumberOfAssignedPerCacheLine)
    );

    //
    // Calculate the total size required for the underlying arrays, bitmap
    // buffers and assigned array counts, rounded up to the nearest page size.
    //

    AllocSize.QuadPart = ROUND_TO_PAGES(
        EdgesSizeInBytes +
        NextSizeInBytes +
        FirstSizeInBytes +
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
    Info->FirstSizeInBytes = FirstSizeInBytes;
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
            PH_ERROR(PrepareGraphInfoChm02_EdgeMaskPopcountMismatch, Result);
            goto Error;
        }

        if (NumberOfVertexMaskBits != Dim->NumberOfVerticesPowerOf2Exponent) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareGraphInfoChm02_VertexMaskPopcountMismatch, Result);
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
            PH_ERROR(PrepareGraphInfoChm02_GetContainingType, Result);
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
    Table->HashShift = Rtl->TrailingZeros32(Table->HashSize);
    Table->IndexShift = Rtl->TrailingZeros32(Table->IndexSize);
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
        Table->Keys->NumberOfElements.QuadPart
    );
    TableInfoOnDisk->NumberOfSeeds = (
        HashRoutineNumberOfSeeds[Table->HashFunctionId]
    );

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

//
// The entry point into the actual per-thread solving attempts is the following
// routine.
//

_Use_decl_annotations_
VOID
ProcessGraphCallbackChm02(
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
            PH_ERROR(ProcessGraphCallbackChm02_InvalidErrorCode, Result);
            PH_RAISE(Result);
        }
    }

    InterlockedDecrement(&Context->ActiveSolvingLoops);

    if (InterlockedDecrement(&Context->RemainingSolverLoops) == 0) {

        //
        // We're the last graph; if the finished count indicates no solutions
        // were found, signal FailedEvent.  Otherwise, signal SucceededEvent.
        // This ensures we always unwait our parent thread's solving loop.
        //
        // N.B. There are numerous scenarios where this is a superfluous call,
        //      as a terminating event (i.e. shutdown, low-memory etc) may have
        //      already been set.  The effort required to distinguish this
        //      situation and avoid setting the event is not warranted
        //      (especially considering setting the event superfluously is
        //      harmless).
        //

        if (Context->FinishedCount == 0) {
            Event = &Context->FailedEvent;
        } else {
            Event = &Context->SucceededEvent;
        }

        SetStopSolving(Context);
    }
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
