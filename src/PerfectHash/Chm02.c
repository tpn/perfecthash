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
#include "Graph_Ptx_RawCString.h"

//
// Spin count for the device context best graph critical section.
//

#define BEST_CU_GRAPH_CS_SPINCOUNT 4000

#define CU_RNG_DEFAULT PerfectHashCuRngPhilox43210Id

//
// Forward decls.
//

HRESULT
InitializeCudaAndGraphsChm02(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
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
    PCU Cu;
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
        goto Error;
    }

    //
    // Perform the CUDA and graph initialization.
    //

    Result = InitializeCudaAndGraphsChm02(Context,
                                          Table->TableCreateParameters);
    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02, Result);
        goto Error;
    }

    Cu = Context->Cu;
    ASSERT(Cu != NULL);

    Concurrency = Context->MaximumConcurrency;
    CuConcurrency = Context->CuConcurrency;
    ASSERT(CuConcurrency <= Concurrency);
    ASSERT(CuConcurrency > 0);

    DeviceContexts = Context->CuDeviceContexts;
    NumberOfDeviceContexts = DeviceContexts->NumberOfDeviceContexts;

    SolveContexts = Context->CuSolveContexts;
    NumberOfSolveContexts = SolveContexts->NumberOfSolveContexts;

    //
    // Copy the keys over to each device participating in the solving.
    //

    KeysBaseAddress = Table->Keys->KeyArrayBaseAddress;
    KeysSizeInBytes = Table->Keys->NumberOfElements.QuadPart * sizeof(KEY);

    for (Index = 0; Index < NumberOfDeviceContexts; Index++) {

        DeviceContext = &DeviceContexts->DeviceContexts[Index];

        //
        // Active the context.
        //

        CuResult = Cu->CtxPushCurrent(DeviceContext->Context);
        if (CU_FAILED(CuResult)) {
            CU_ERROR(CtxPushCurrent, CuResult);
            Result = PH_E_CUDA_DRIVER_API_CALL_FAILED;
            goto Error;
        }

        if (DeviceContext->KeysBaseAddress == 0) {

            //
            // No device memory has been allocated for keys before, so,
            // allocate some now.
            //

            ASSERT(DeviceContext->KeysSizeInBytes == 0);

            CuResult = Cu->MemAlloc(&DeviceContext->KeysBaseAddress,
                                    KeysSizeInBytes);
            CU_CHECK(CuResult, MemAlloc);

            DeviceContext->KeysSizeInBytes = KeysSizeInBytes;

        } else {

            //
            // Device memory has already been allocated.  If it's less than what
            // we need, free what's there and allocate new memoyr.
            //

            ASSERT(DeviceContext->KeysSizeInBytes > 0);

            if (DeviceContext->KeysSizeInBytes < KeysSizeInBytes) {

                CuResult = Cu->MemFree(DeviceContext->KeysBaseAddress);
                CU_CHECK(CuResult, MemFree);

                DeviceContext->KeysBaseAddress = 0;

                CuResult = Cu->MemAlloc(&DeviceContext->KeysBaseAddress,
                                        KeysSizeInBytes);
                CU_CHECK(CuResult, MemAlloc);

                DeviceContext->KeysSizeInBytes = KeysSizeInBytes;

            } else {

                //
                // The existing device memory will fit the keys array, so
                // there's nothing more to do here.
                //

                ASSERT(DeviceContext->KeysSizeInBytes >= KeysSizeInBytes);
            }
        }

        //
        // Copy the keys over.
        //

        CuResult = Cu->MemcpyHtoDAsync(DeviceContext->KeysBaseAddress,
                                       KeysBaseAddress,
                                       KeysSizeInBytes,
                                       DeviceContext->Stream);
        CU_CHECK(CuResult, MemcpyHtoDAsync);

        if (DeviceContext->DeviceGraphInfoAddress == 0) {

            //
            // Allocate memory for the graph info.
            //

            CuResult = Cu->MemAlloc(&DeviceContext->DeviceGraphInfoAddress,
                                    sizeof(GRAPH_INFO));
            CU_CHECK(CuResult, MemAlloc);
        }

        //
        // Copy the graph info over.
        //

        CuResult = Cu->MemcpyHtoDAsync(DeviceContext->DeviceGraphInfoAddress,
                                       &Info,
                                       sizeof(GRAPH_INFO),
                                       DeviceContext->Stream);
        CU_CHECK(CuResult, MemcpyHtoDAsync);
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
    // Synchronize on each GPU device's stream that was used for the async
    // memcpy of the keys array before submitting any threadpool work.
    //

    for (Index = 0; Index < DeviceContexts->NumberOfDeviceContexts; Index++) {
        DeviceContext = &DeviceContexts->DeviceContexts[Index];
        CuResult = Cu->StreamSynchronize(DeviceContext->Stream);
        CU_CHECK(CuResult, StreamSynchronize);
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

        if (!IsSpareGraph(Graph)) {
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

    COPY_GRAPH_COUNTERS_FROM_GRAPH_TO_TABLE();

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

    //
    // Todo: free keys.
    //

    Graphs = Context->Graphs;

#if 0
    if (0 && Graphs) {
        ULONG ReferenceCount;

        //
        // Walk the array of graph instances and release each one (assuming it
        // is not NULL), then free the array buffer.
        //

        for (Index = 0; Index < TotalNumberOfGraphs; Index++) {

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
#endif

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
InitializeCudaAndGraphsChm02(
    PPERFECT_HASH_CONTEXT Context,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Attempts to initialize CUDA and all supporting graphs for the given context.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance for which
        the routine will try and initialize CUDA and all supporting graphs.

    TableCreateParameters - Supplies a pointer to the table create parameters.

Return Value:

    S_OK - Initialized successfully.

    S_FALSE - Already initialized.

    Otherwise, an appropriate error code.

--*/
{
    PCU Cu;
    PRTL Rtl;
    ULONG Index;
    ULONG Inner;
    ULONG Count;
    LONG Ordinal;
    BOOLEAN Found;
    ULONG NumberOfDevices;
    ULONG NumberOfContexts;
    PULONG BitmapBuffer = NULL;
    RTL_BITMAP Bitmap;
    HRESULT Result;
    PCHAR PtxString;
    CU_RESULT CuResult;
    CU_DEVICE DeviceId;
    CU_DEVICE MinDeviceId;
    CU_DEVICE MaxDeviceId;
    PALLOCATOR Allocator;
    PGRAPH Graph;
    PGRAPH *Graphs;
    PGRAPH *CpuGraphs;
    PGRAPH *GpuGraphs;
    PGRAPH DeviceGraph;
    PGRAPH DeviceGraphs;
    PCHAR LinkedModule;
    SIZE_T LinkedModuleSizeInBytes;
    SIZE_T PtxSizeInBytes;
    PPH_CU_DEVICE Device;
    PCU_OCCUPANCY Occupancy;
    PCU_LINK_STATE LinkState;
    BOOLEAN SawCuRngSeed;
    BOOLEAN SawCuConcurrency;
    BOOLEAN WantsRandomHostSeeds;
    BOOLEAN IsRngImplemented;
    PUNICODE_STRING CuPtxPath;
    PUNICODE_STRING CuCudaDevRuntimeLibPath;
    ULONG NumberOfGpuGraphs;
    ULONG NumberOfCpuGraphs;
    ULONG TotalNumberOfGraphs;
    ULONG NumberOfRandomHostSeeds;
    ULONG SpareGraphCount;
    ULONG MatchedGraphCount;
    ULONG BlocksPerGridValue;
    ULONG ThreadsPerBlockValue;
    ULONG KernelRuntimeTargetValue;
    ULONG NumberOfGraphsForDevice;
    ULONG NumberOfSolveContexts;
    PVALUE_ARRAY Ordinals;
    PVALUE_ARRAY BlocksPerGrid;
    PVALUE_ARRAY ThreadsPerBlock;
    PVALUE_ARRAY KernelRuntimeTarget;
    CU_STREAM_FLAGS StreamFlags;
    ULARGE_INTEGER AllocSizeInBytes;
    ULARGE_INTEGER BitmapBufferSizeInBytes;
    PPH_CU_DEVICE_CONTEXT DeviceContext;
    PPH_CU_DEVICE_CONTEXTS DeviceContexts;
    PPH_CU_SOLVE_CONTEXT SolveContext;
    PPH_CU_SOLVE_CONTEXTS SolveContexts;
    //PERFECT_HASH_CU_RNG_ID CuRngId = PerfectHashCuNullRngId;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_PATH PtxPath;
    PPERFECT_HASH_FILE PtxFile;
    PPERFECT_HASH_PATH RuntimeLibPath;
    PPERFECT_HASH_FILE RuntimeLibFile;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags = { 0 };
    LARGE_INTEGER EndOfFile = { 0 };

    STRING KernelFunctionName =
        RTL_CONSTANT_STRING("PerfectHashCudaEnterSolvingLoop");

    CU_JIT_OPTION JitOptions[] = {
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
    };
    PVOID JitOptionValues[2];
    USHORT NumberOfJitOptions = ARRAYSIZE(JitOptions);

    CHAR JitLogBuffer[PERFECT_HASH_CU_JIT_LOG_BUFFER_SIZE_IN_BYTES];

    //
    // If we've already got a CU instance, assume we're already initialized.
    //

    if (Context->Cu != NULL) {
        return S_FALSE;
    }

    PtxFile = NULL;
    PtxPath = NULL;
    CuPtxPath = NULL;
    SolveContexts = NULL;
    DeviceContexts = NULL;
    RuntimeLibPath = NULL;
    RuntimeLibFile = NULL;
    CuCudaDevRuntimeLibPath = NULL;

    Table = Context->Table;
    TableCreateFlags.AsULong = Table->TableCreateFlags.AsULong;

    //
    // Try create a CU instance.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_CU,
                                           &Context->Cu);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // The CU component is a global component, which means it can't create
    // instances of other global components like Rtl and Allocator during its
    // initialization function.  So, we manually set them now.
    //

    Cu = Context->Cu;

    Rtl = Cu->Rtl = Context->Rtl;
    Cu->Rtl->Vtbl->AddRef(Cu->Rtl);

    Cu->Allocator = Allocator = Context->Allocator;
    Cu->Allocator->Vtbl->AddRef(Cu->Allocator);

    Result = CreatePerfectHashCuDevices(Cu,
                                        Cu->Allocator,
                                        &Context->CuDevices);
    if (FAILED(Result)) {
        PH_ERROR(CreatePerfectHashCuDevices, Result);
        goto Error;
    }

    Count = TableCreateParameters->NumberOfElements;
    Param = TableCreateParameters->Params;

    //
    // Clear our local aliases.
    //

    Ordinals = NULL;
    BlocksPerGrid = NULL;
    ThreadsPerBlock = NULL;
    KernelRuntimeTarget = NULL;
    SawCuConcurrency = FALSE;
    SawCuRngSeed = FALSE;

    //
    // Disable "enum not handled in switch statement" warning.
    //
    //      warning C4061: enumerator 'TableCreateParameterNullId' in switch
    //                     of enum 'PERFECT_HASH_TABLE_CREATE_PARAMETER_ID'
    //                     is not explicitly handled by a case label
    //

#pragma warning(push)
#pragma warning(disable: 4061)

    for (Index = 0; Index < Count; Index++, Param++) {

        switch (Param->Id) {

            case TableCreateParameterCuRngId:
                Context->CuRngId = Param->AsCuRngId;
                break;

            case TableCreateParameterCuRngSeedId:
                Context->CuRngSeed = Param->AsULongLong;
                SawCuRngSeed = TRUE;
                break;

            case TableCreateParameterCuRngSubsequenceId:
                Context->CuRngSubsequence = Param->AsULongLong;
                break;

            case TableCreateParameterCuRngOffsetId:
                Context->CuRngOffset = Param->AsULongLong;
                break;

            case TableCreateParameterCuConcurrencyId:
                Context->CuConcurrency = Param->AsULong;
                SawCuConcurrency = TRUE;
                break;

            case TableCreateParameterCuDevicesId:
                Ordinals = &Param->AsValueArray;
                break;

            case TableCreateParameterCuDevicesBlocksPerGridId:
                BlocksPerGrid = &Param->AsValueArray;
                break;

            case TableCreateParameterCuDevicesThreadsPerBlockId:
                ThreadsPerBlock = &Param->AsValueArray;
                break;

            case TableCreateParameterCuDevicesKernelRuntimeTargetInMillisecondsId:
                KernelRuntimeTarget = &Param->AsValueArray;
                break;

            case TableCreateParameterCuPtxPathId:
                CuPtxPath = &Param->AsUnicodeString;
                break;

            case TableCreateParameterCuCudaDevRuntimeLibPathId:
                CuCudaDevRuntimeLibPath = &Param->AsUnicodeString;
                break;

            case TableCreateParameterCuNumberOfRandomHostSeedsId:
                NumberOfRandomHostSeeds = Param->AsULong;
                WantsRandomHostSeeds = TRUE;
                break;

            default:
                break;
        }
    }

#pragma warning(pop)

    //
    // Validate --CuRng.  We only implement a subset of algorithms.
    //

    if (!IsValidPerfectHashCuRngId(Context->CuRngId)) {
        Context->CuRngId = CU_RNG_DEFAULT;
    }

    Result = PerfectHashLookupNameForId(Rtl,
                                        PerfectHashCuRngEnumId,
                                        Context->CuRngId,
                                        &Context->CuRngName);
    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_LookupNameForId, Result);
        goto Error;
    }

    IsRngImplemented = FALSE;

#define EXPAND_AS_CU_RNG_ID_CASE(Name, Upper, Implemented) \
    case PerfectHashCuRng##Name##Id:                       \
        IsRngImplemented = Implemented;                    \
        break;

    switch (Context->CuRngId) {

        case PerfectHashNullCuRngId:
        case PerfectHashInvalidCuRngId:
            PH_RAISE(PH_E_UNREACHABLE_CODE);
            break;

        PERFECT_HASH_CU_RNG_TABLE_ENTRY(EXPAND_AS_CU_RNG_ID_CASE);

        default:
            PH_RAISE(PH_E_UNREACHABLE_CODE);
            break;
    }

    if (!IsRngImplemented) {
        Result = PH_E_UNIMPLEMENTED_CU_RNG_ID;
        goto Error;
    }

    //
    // If no seed has been supplied, generate a random one now.
    //

    if (!SawCuRngSeed) {

        Result = Rtl->Vtbl->GenerateRandomBytes(Rtl,
                                                sizeof(Context->CuRngSeed),
                                                (PBYTE)&Context->CuRngSeed);

        if (FAILED(Result)) {
            PH_ERROR(InitializeCudaAndGraphsChm02_GenerateCuRngSeed, Result);
            goto Error;
        }
    }

    //
    // Validate --CuConcurrency.  It's mandatory, it must be greater than zero,
    // and less than or equal to the maximum concurrency.  (When CuConcurrency
    // is less than max concurrency, the difference between the two will be the
    // number of CPU solving threads launched.  E.g. if --CuConcurrency=16 and
    // max concurrency is 18; there will be two CPU solving threads launched in
    // addition to the 16 GPU solver threads.)
    //

    if (!SawCuConcurrency) {
        Result = PH_E_CU_CONCURRENCY_MANDATORY_FOR_SELECTED_ALGORITHM;
        goto Error;
    }

    if (Context->CuConcurrency == 0) {
        Result = PH_E_INVALID_CU_CONCURRENCY;
        goto Error;
    }

    if (Context->CuConcurrency > Context->MaximumConcurrency) {
        Result = PH_E_CU_CONCURRENCY_EXCEEDS_MAX_CONCURRENCY;
        goto Error;
    }

    if (CuCudaDevRuntimeLibPath == NULL) {
        Result = PH_E_CU_CUDA_DEV_RUNTIME_LIB_PATH_MANDATORY;
        goto Error;
    }

    //
    // Calculate the number of CPU solving threads; this may be zero.
    //

    Context->NumberOfCpuThreads = (
        Context->MaximumConcurrency -
        Context->CuConcurrency
    );

    //
    // Initialize the number of graphs to use for CPU/GPU solving.  Initially,
    // this will match the desired respective concurrency level.
    //

    Context->NumberOfGpuGraphs = Context->CuConcurrency;
    Context->NumberOfCpuGraphs = Context->NumberOfCpuThreads;

    if (FindBestGraph(Context)) {

        //
        // Double the graph count if we're in "find best graph" mode to account
        // for the spare graphs (one per solve context).
        //

        Context->NumberOfGpuGraphs *= 2;

        //
        // Only increment the number of CPU graphs if the number of CPU threads
        // is greater than zero.  (We only need one extra spare graph for all
        // CPU solver threads; this is a side-effect of the original Chm01 CPu
        // solver implementation.)
        //

        if (Context->NumberOfCpuThreads > 0) {
            Context->NumberOfCpuGraphs += 1;
        }

    }

    //
    // Validate device ordinals optionally supplied via --CuDevices.  This
    // parameter is a bit quirky: it can be a single value or list of comma-
    // separated values.  Each value represents a device ordinal, and any
    // device ordinal can appear one or more times.  The number of *unique*
    // ordinals dictates the number of CUDA contexts we create.  (We only want
    // one context per device; multiple contexts would impede performance.)
    //
    // If only one device ordinal is supplied, then all GPU solver threads will
    // use this device.  If more than one ordinal is supplied, there must be at
    // least two unique ordinals present in the entire set.  E.g.:
    //
    //      Valid:      --CuDevices=0,1
    //      Invalid:    --CuDevices=0,0
    //
    // Additionally, if more than one ordinal is supplied, the dependent params
    // like --CuDevicesBlocksPerGrid and --CuDevicesThreadsPerBlock must have
    // the same number of values supplied.  E.g.:
    //
    //      Valid:      --CuDevices=0,1 --CuDevicesBlocksPerGrid=32,16
    //      Invalid:    --CuDevices=0,1 --CuDevicesBlocksPerGrid=32
    //
    // In this situation, the order of the device ordinal in the value list will
    // be correlated with the identically-offset value in the dependent list.
    // In the example above, the CUDA contexts for devices 0 and 1 will use 32
    // and 16 respectively as their blocks-per-grid value.
    //

    //
    // First, if --CuDevices (local variable `Ordinals`) has not been supplied,
    // verify no dependent params are present.
    //

    if (Ordinals == NULL) {

        if (BlocksPerGrid != NULL) {
            Result = PH_E_CU_BLOCKS_PER_GRID_REQUIRES_CU_DEVICES;
            goto Error;
        }

        if (ThreadsPerBlock != NULL) {
            Result = PH_E_CU_THREADS_PER_BLOCK_REQUIRES_CU_DEVICES;
            goto Error;
        }

        if (KernelRuntimeTarget != NULL) {
            Result = PH_E_CU_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS_REQUIRES_CU_DEVICES;
            goto Error;
        }

        //
        // We default the number of contexts and devices to 1 in the absence of any
        // user-supplied values.
        //

        NumberOfContexts = 1;
        NumberOfDevices = 1;
        goto FinishedOrdinalsProcessing;

    }

    //
    // Ordinals have been supplied.  Verify the number of values matches the
    // supplied value for --CuConcurrency, then verify that if any dependent
    // parameters have been supplied, they have the same number of values.
    //

    if (Context->CuConcurrency != Ordinals->NumberOfValues) {
        Result = PH_E_CU_DEVICES_COUNT_MUST_MATCH_CU_CONCONCURRENCY;
        goto Error;
    }

    if ((BlocksPerGrid != NULL) &&
        (BlocksPerGrid->NumberOfValues != Ordinals->NumberOfValues))
    {
        Result = PH_E_CU_BLOCKS_PER_GRID_COUNT_MUST_MATCH_CU_DEVICES_COUNT;
        goto Error;
    }

    if ((ThreadsPerBlock != NULL) &&
        (ThreadsPerBlock->NumberOfValues != Ordinals->NumberOfValues))
    {
        Result = PH_E_CU_THREADS_PER_BLOCK_COUNT_MUST_MATCH_CU_DEVICES_COUNT;
        goto Error;
    }

    if ((KernelRuntimeTarget != NULL) &&
        (KernelRuntimeTarget->NumberOfValues != Ordinals->NumberOfValues))
    {
        Result = PH_E_CU_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS_COUNT_MUST_MATCH_CU_DEVICES_COUNT;
        goto Error;
    }

    //
    // Initialize the min and max device IDs, then enumerate the supplied
    // ordinals, validating each one as we go and updating the min/max values
    // accordingly.
    //

    MinDeviceId = 1 << 30;
    MaxDeviceId = 0;

    for (Index = 0; Index < Ordinals->NumberOfValues; Index++) {
        Ordinal = (LONG)Ordinals->Values[Index];
        CuResult = Cu->DeviceGet(&DeviceId, Ordinal);
        if (CU_FAILED(CuResult)) {
            CU_ERROR(CuDeviceGet, CuResult);
            Result = PH_E_INVALID_CU_DEVICES;
            goto Error;
        }
        if (DeviceId > MaxDeviceId) {
            MaxDeviceId = DeviceId;
        }
        if (DeviceId < MinDeviceId) {
            MinDeviceId = DeviceId;
        }
    }

    //
    // We use a bitmap to count the number of unique devices supplied in the
    // --CuDevices parameter.  Calculate the bitmap buffer size in bytes.
    //

    BitmapBufferSizeInBytes.QuadPart = ALIGN_UP_POINTER(
        ALIGN_UP((MaxDeviceId + 1ULL), 8) >> 3
    );

    //
    // Sanity check we haven't overflowed.
    //

    if (BitmapBufferSizeInBytes.HighPart != 0) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        goto Error;
    }

    ASSERT(BitmapBufferSizeInBytes.LowPart > 0);

    //
    // Allocate sufficient bitmap buffer space.
    //

    BitmapBuffer = Allocator->Vtbl->Calloc(Allocator,
                                           1,
                                           BitmapBufferSizeInBytes.LowPart);
    if (BitmapBuffer == NULL) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // Wire-up the device bitmap.
    //

    Bitmap.Buffer = BitmapBuffer;
    Bitmap.SizeOfBitMap = (MaxDeviceId + 1);

    //
    // Enumerate the ordinals again, setting a corresponding bit for each
    // ordinal we see.
    //

    for (Index = 0; Index < Ordinals->NumberOfValues; Index++) {
        Ordinal = (LONG)Ordinals->Values[Index];
        ASSERT(Ordinal >= 0);
        _Analysis_assume_(Ordinal >= 0);
        FastSetBit(&Bitmap, Ordinal);
    }

    //
    // Count the number of bits set, this will represent the number of unique
    // devices we encountered.  Sanity check the number doesn't exceed the
    // total number of devices reported in the system.
    //

    Rtl = Context->Rtl;
    NumberOfContexts = Rtl->RtlNumberOfSetBits(&Bitmap);
    NumberOfDevices = Context->CuDevices.NumberOfDevices;

    if (NumberOfContexts > NumberOfDevices) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashContextInitializeCuda_SetBitsExceedsNumDevices,
                 Result);
        goto Error;
    }

    Context->NumberOfCuContexts = NumberOfContexts;

    //
    // Intentional follow-on to FinishedOrdinalsProcessing.
    //

FinishedOrdinalsProcessing:

    //
    // Allocate memory for the device contexts structs.
    //

    AllocSizeInBytes.QuadPart = sizeof(*Context->CuDeviceContexts);

    if (NumberOfContexts > 1) {

        //
        // Account for additional device context structures if we're creating
        // more than one.  (We get one for free via ANYSIZE_ARRAY.)
        //

        AllocSizeInBytes.QuadPart += (
            (NumberOfContexts - 1) *
            sizeof(Context->CuDeviceContexts->DeviceContexts[0])
        );

        if (FindBestGraph(Context)) {

            //
            // Sanity check our graph counts line up.
            //

            ASSERT((NumberOfContexts * 2) == Context->NumberOfGpuGraphs);
        }
    }

    //
    // Sanity check we haven't overflowed.
    //

    if (AllocSizeInBytes.HighPart > 0) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashContextInitializeCuda_DeviceContextAllocOverflow,
                 Result);
        PH_RAISE(Result);
    }

    DeviceContexts = Allocator->Vtbl->Calloc(Allocator,
                                             1,
                                             AllocSizeInBytes.LowPart);
    if (DeviceContexts == NULL) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    Context->CuDeviceContexts = DeviceContexts;
    DeviceContexts->NumberOfDeviceContexts = NumberOfContexts;

    //
    // First pass: set each device context's ordinal to the value obtained via
    // the --CuDevices parameter.  (The logic we use to do this is a little
    // different if we're dealing with one context versus more than one.)
    //

    if (NumberOfContexts == 1) {

        DeviceContext = &DeviceContexts->DeviceContexts[0];

        if (Ordinals != NULL) {
            ASSERT(Ordinals->NumberOfValues == 1);
            DeviceContext->Ordinal = (LONG)Ordinals->Values[0];
        } else {

            //
            // If no --CuDevices parameter has been supplied, default to 0 for
            // the device ordinal.
            //

            DeviceContext->Ordinal = 0;
        }

    } else {

        ULONG Bit = 0;
        const ULONG FindOneBit = 1;

        for (Index = 0; Index < NumberOfContexts; Index++) {
            DeviceContext = &DeviceContexts->DeviceContexts[Index];

            //
            // Get the device ordinal from the first set/next set bit of the
            // bitmap.
            //

            Bit = Rtl->RtlFindSetBits(&Bitmap, FindOneBit, Bit);

            if (Bit == BITS_NOT_FOUND) {
                Result = PH_E_INVARIANT_CHECK_FAILED;
                PH_ERROR(PerfectHashContextInitializeCuda_BitsNotFound,
                         Result);
                PH_RAISE(Result);
            }

            DeviceContext->Ordinal = (LONG)Bit;
            Bit += 1;
        }
    }

    if (CuPtxPath == NULL) {

        //
        // No --CuPtxPath supplied; use the embedded PTX string.
        //

        PtxString = (PCHAR)GraphPtxRawCStr;
        PtxSizeInBytes = sizeof(GraphPtxRawCStr);

    } else {

        //
        // --CuPtxPath was supplied.  Create a path instance to encapsulate the
        // argument, then a corresponding file object that can be loaded, such
        // that we can access the PTX as a raw C string.
        //

        //
        // Construct a path instance.
        //

        Result = Context->Vtbl->CreateInstance(Context,
                                               NULL,
                                               &IID_PERFECT_HASH_PATH,
                                               &PtxPath);

        if (FAILED(Result)) {
            PH_ERROR(InitializeCudaAndGraphsChm02_CreatePtxPath, Result);
            goto Error;
        }

        Result = PtxPath->Vtbl->Copy(PtxPath, CuPtxPath, NULL, NULL);
        if (FAILED(Result)) {
            PH_ERROR(InitializeCudaAndGraphsChm02_PtxPathCopy, Result);
            goto Error;
        }

        //
        // Create a file instance.
        //

        Result = Context->Vtbl->CreateInstance(Context,
                                               NULL,
                                               &IID_PERFECT_HASH_FILE,
                                               &PtxFile);

        if (FAILED(Result)) {
            PH_ERROR(InitializeCudaAndGraphsChm02_CreatePtxFile, Result);
            goto Error;
        }

        //
        // Load the PTX file (map it into memory).  We can then use the base
        // address as the PTX string.  EndOfFile will capture the PTX size in
        // bytes.
        //

        Result = PtxFile->Vtbl->Load(PtxFile,
                                     PtxPath,
                                     &EndOfFile,
                                     &FileLoadFlags);
        if (FAILED(Result)) {
            PH_ERROR(InitializeCudaAndGraphsChm02_LoadPtxFile, Result);
            goto Error;
        }

        PtxString = (PCHAR)PtxFile->BaseAddress;
        PtxSizeInBytes = EndOfFile.QuadPart;

        RELEASE(PtxPath);
    }

    //
    // Open the cudadevrt.lib path.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_PATH,
                                           &RuntimeLibPath);

    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_CreateRuntimeLibPath, Result);
        goto Error;
    }

    Result = RuntimeLibPath->Vtbl->Copy(RuntimeLibPath,
                                        CuCudaDevRuntimeLibPath,
                                        NULL,
                                        NULL);
    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_RuntimeLibPathCopy, Result);
        goto Error;
    }

    //
    // Create a file instance.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_FILE,
                                           &RuntimeLibFile);

    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_CreateRuntimeLibFile, Result);
        goto Error;
    }

    EndOfFile.QuadPart = 0;
    Result = RuntimeLibFile->Vtbl->Load(RuntimeLibFile,
                                        RuntimeLibPath,
                                        &EndOfFile,
                                        &FileLoadFlags);

    if (FAILED(Result)) {
        PH_ERROR(InitializeCudaAndGraphsChm02_LoadRuntimeLibFile, Result);
        goto Error;
    }

    //
    // Initialize the JIT options.
    //

    JitOptionValues[0] = (PVOID)sizeof(JitLogBuffer);
    JitOptionValues[1] = (PVOID)JitLogBuffer;

    //
    // Second pass: wire-up each device context (identified by ordinal, set
    // in the first pass above) to the corresponding PH_CU_DEVICE instance
    // for that device, create CUDA contexts for each device context, load
    // the module, get the solver entry function, and calculate occupancy.
    //

    Device = NULL;
    StreamFlags = CU_STREAM_NON_BLOCKING;

    for (Index = 0; Index < NumberOfContexts; Index++) {
        DeviceContext = &DeviceContexts->DeviceContexts[Index];

#if 0
        if (!InitializeCriticalSectionAndSpinCount(
                                    &DeviceContext->BestGraphCriticalSection,
                                    BEST_CU_GRAPH_CS_SPINCOUNT)) {

            //
            // This should never fail from Vista onward.
            //

            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(InitializeCudaAndGraphsChm02_InitializeCriticalSection,
                     Result);
            PH_RAISE(Result);
        }
#endif

        //
        // Find the PH_CU_DEVICE instance with the same ordinal.
        //

        Found = FALSE;
        Device = NULL;
        for (Inner = 0; Inner < NumberOfDevices; Inner++) {
            Device = &Context->CuDevices.Devices[Inner];
            if (Device->Ordinal == DeviceContext->Ordinal) {
                DeviceContext->Device = Device;
                Found = TRUE;
                break;
            }
        }

        if (!Found) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashContextInitializeCuda_OrdinalNotFound, Result);
            PH_RAISE(Result);
        }

        ASSERT(Device != NULL);

        DeviceContext->Handle = Device->Handle;
        DeviceContext->Cu = Cu;

        //
        // Create the context for the device.
        //

        CuResult = Cu->CtxCreate(&DeviceContext->Context,
                                 //CU_CTX_SCHED_YIELD,
                                 CU_CTX_SCHED_BLOCKING_SYNC,
                                 Device->Handle);
        CU_CHECK(CuResult, CtxCreate);

        //
        // Our solver kernel uses dynamic parallelism (that is, it launches
        // other kernels).  For this to work, we can't just load the PTX
        // directly; we need to perform a linking step which also adds the
        // cudadevrt.lib (CUDA device runtime static library) into the mix.
        // We do this by issuing a cuLinkCreate(), cuLinkAddData() for the
        // static .lib and our .ptx string, then cuLinkComplete() to get the
        // final module that we can pass to cuLoadModuleEx().
        //

        CuResult = Cu->LinkCreate(NumberOfJitOptions,
                                  JitOptions,
                                  JitOptionValues,
                                  &LinkState);
        CU_CHECK(CuResult, LinkCreate);

        //
        // Add cudadevrt.lib.
        //

        CuResult = Cu->LinkAddData(LinkState,
                                   CU_JIT_INPUT_LIBRARY,
                                   RuntimeLibFile->BaseAddress,
                                   EndOfFile.QuadPart,
                                   "cudadevrt.lib",
                                   0,
                                   NULL,
                                   NULL);
        CU_CHECK(CuResult, LinkAddData);

        //
        // Add the PTX file.
        //

        CuResult = Cu->LinkAddData(LinkState,
                                   CU_JIT_INPUT_PTX,
                                   PtxString,
                                   PtxSizeInBytes,
                                   "Graph.ptx",
                                   0,
                                   NULL,
                                   NULL);
        CU_CHECK(CuResult, LinkAddData);

        //
        // Complete the link.
        //

        CuResult = Cu->LinkComplete(LinkState,
                                    &LinkedModule,
                                    &LinkedModuleSizeInBytes);
        CU_CHECK(CuResult, LinkComplete);

        //
        // Load the module from the embedded PTX.
        //

        CuResult = Cu->ModuleLoadDataEx(&DeviceContext->Module,
                                        LinkedModule,
                                        NumberOfJitOptions,
                                        JitOptions,
                                        JitOptionValues);
        CU_CHECK(CuResult, ModuleLoadDataEx);

        //
        // Module loaded successfully, resolve the kernel.
        //

        CuResult = Cu->ModuleGetFunction(&DeviceContext->Function,
                                         DeviceContext->Module,
                                         (PCSZ)KernelFunctionName.Buffer);
        CU_CHECK(CuResult, ModuleGetFunction);

        //
        // Get the occupancy stats.
        //

        Occupancy = &DeviceContext->Occupancy;
        CuResult = Cu->OccupancyMaxPotentialBlockSizeWithFlags(
            &Occupancy->MinimumGridSize,
            &Occupancy->BlockSize,
            DeviceContext->Function,
            NULL,   // OccupancyBlockSizeToDynamicMemSize
            0,      // DynamicSharedMemorySize
            0,      // BlockSizeLimit
            0       // Flags
        );
        CU_CHECK(CuResult, OccupancyMaxPotentialBlockSizeWithFlags);

        CuResult = Cu->OccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &Occupancy->NumBlocks,
            DeviceContext->Function,
            Occupancy->BlockSize,
            0, // DynamicSharedMemorySize
            0  // Flags
        );
        CU_CHECK(CuResult, OccupancyMaxActiveBlocksPerMultiprocessorWithFlags);

        //
        // Create the stream to use for per-device activies (like copying keys).
        //

        CuResult = Cu->StreamCreate(&DeviceContext->Stream, StreamFlags);
        CU_CHECK(CuResult, StreamCreate);

        //
        // Pop the context off this thread (required before it can be used by
        // other threads).
        //

        CuResult = Cu->CtxPopCurrent(NULL);
        CU_CHECK(CuResult, CtxPopCurrent);

    }

    //
    // Allocate space for solver contexts; one per CUDA solving thread.
    //

    AllocSizeInBytes.QuadPart = sizeof(*Context->CuSolveContexts);

    //
    // Account for additional solve context structures if we're creating more
    // than one.  (We get one for free via ANYSIZE_ARRAY.)
    //

    if (Context->CuConcurrency > 1) {
        AllocSizeInBytes.QuadPart += (
            (Context->CuConcurrency - 1) *
            sizeof(Context->CuSolveContexts->SolveContexts[0])
        );
    }

    //
    // Sanity check we haven't overflowed.
    //

    if (AllocSizeInBytes.HighPart > 0) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashContextInitializeCuda_SolveContextAllocOverflow,
                 Result);
        PH_RAISE(Result);
    }

    SolveContexts = Allocator->Vtbl->Calloc(Allocator,
                                            1,
                                            AllocSizeInBytes.LowPart);
    if (SolveContexts == NULL) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    Context->CuSolveContexts = SolveContexts;
    SolveContexts->NumberOfSolveContexts = Context->CuConcurrency;
    NumberOfSolveContexts = Context->CuConcurrency;

    //
    // Wire up the solve contexts to their respective device context.
    //

    for (Index = 0; Index < NumberOfSolveContexts; Index++) {

        SolveContext = &SolveContexts->SolveContexts[Index];

        //
        // Resolve the ordinal and kernel launch parameters.
        //

        if (Ordinals == NULL) {
            Ordinal = 0;
        } else {
            Ordinal = (LONG)Ordinals->Values[Index];
        }

        if (BlocksPerGrid == NULL) {
            BlocksPerGridValue = 0;
        } else {
            BlocksPerGridValue = BlocksPerGrid->Values[Index];
        }
        if (BlocksPerGridValue == 0) {
            BlocksPerGridValue = PERFECT_HASH_CU_DEFAULT_BLOCKS_PER_GRID;
        }

        if (ThreadsPerBlock == NULL) {
            ThreadsPerBlockValue = 0;
        } else {
            ThreadsPerBlockValue = ThreadsPerBlock->Values[Index];
        }
        if (ThreadsPerBlockValue == 0) {
            ThreadsPerBlockValue = PERFECT_HASH_CU_DEFAULT_THREADS_PER_BLOCK;
        }

        if (KernelRuntimeTarget == NULL) {
            KernelRuntimeTargetValue = 0;
        } else {
            KernelRuntimeTargetValue = KernelRuntimeTarget->Values[Index];
        }
        if (KernelRuntimeTargetValue == 0) {
            KernelRuntimeTargetValue =
                PERFECT_HASH_CU_DEFAULT_KERNEL_RUNTIME_TARGET_IN_MILLISECONDS;
        }

        //
        // Find the device context for this device ordinal.
        //

        Found = FALSE;
        DeviceContext = NULL;
        for (Inner = 0; Inner < NumberOfContexts; Inner++) {
            DeviceContext = &DeviceContexts->DeviceContexts[Inner];
            if (DeviceContext->Ordinal == Ordinal) {
                Found = TRUE;
                break;
            }
        }

        if (!Found) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashContextInitializeCuda_ContextOrdinalNotFound,
                     Result);
            PH_RAISE(Result);
        }

        ASSERT(DeviceContext != NULL);

        //
        // Increment the count of solve contexts for this device context.
        //

        DeviceContext->NumberOfSolveContexts++;

        //
        // Link this solve context to the corresponding device context, then
        // fill in the kernel launch parameters.
        //

        SolveContext->DeviceContext = DeviceContext;

        SolveContext->BlocksPerGrid = BlocksPerGridValue;
        SolveContext->ThreadsPerBlock = ThreadsPerBlockValue;
        SolveContext->KernelRuntimeTargetInMilliseconds =
            KernelRuntimeTargetValue;

        //
        // Activate this context, create a stream, then deactivate it.
        //

        CuResult = Cu->CtxPushCurrent(DeviceContext->Context);
        CU_CHECK(CuResult, CtxPushCurrent);

        //
        // Create the stream for this solve context.
        //

        CuResult = Cu->StreamCreate(&SolveContext->Stream, StreamFlags);
        CU_CHECK(CuResult, StreamCreate);

        //
        // Pop the context off this thread.
        //

        CuResult = Cu->CtxPopCurrent(NULL);
        CU_CHECK(CuResult, CtxPopCurrent);

    }

    //
    // For each device context, allocate device memory to hold sufficient
    // graphs.  If we're in "find best graph" mode, there are two graphs per
    // solve context (to account for the spare graph); otherwise, there is one.
    //

    for (Index = 0; Index < NumberOfContexts; Index++) {

        DeviceContext = &DeviceContexts->DeviceContexts[Index];

        ASSERT(DeviceContext->NumberOfSolveContexts > 0);

        NumberOfGraphsForDevice = DeviceContext->NumberOfSolveContexts;

        if (FindBestGraph(Context)) {

            //
            // Account for the spare graphs.
            //

            NumberOfGraphsForDevice *= 2;
        }

        AllocSizeInBytes.QuadPart = NumberOfGraphsForDevice * sizeof(GRAPH);
        ASSERT(AllocSizeInBytes.HighPart == 0);

        //
        // Set the context, then allocate the array of graphs.
        //

        CuResult = Cu->CtxPushCurrent(DeviceContext->Context);
        CU_CHECK(CuResult, CtxPushCurrent);

        CuResult = Cu->MemAlloc((PCU_DEVICE_POINTER)&DeviceGraphs,
                                AllocSizeInBytes.LowPart);
        CU_CHECK(CuResult, MemAlloc);

        DeviceContext->DeviceGraphs = DeviceGraph = DeviceGraphs;

        //
        // Loop over the solve contexts pointing to this device context and
        // wire up matching ones to the device graph memory we just allocated.
        //

        MatchedGraphCount = 0;
        for (Inner = 0; Inner < NumberOfSolveContexts; Inner++) {
            SolveContext = &SolveContexts->SolveContexts[Inner];

            if (SolveContext->DeviceContext == DeviceContext) {
                SolveContext->DeviceGraph = DeviceGraph++;
                MatchedGraphCount++;
                if (FindBestGraph(Context)) {
                    SolveContext->DeviceSpareGraph = DeviceGraph++;
                    MatchedGraphCount++;
                }
            }
        }

        ASSERT(MatchedGraphCount == NumberOfGraphsForDevice);

        //
        // Allocate device memory to hold the CU_DEVICE_ATTRIBUTES structure.
        //

        CuResult = Cu->MemAlloc(&DeviceContext->DeviceAttributes,
                                sizeof(CU_DEVICE_ATTRIBUTES));
        CU_CHECK(CuResult, MemAlloc);

        //
        // Copy the host attributes to the device.
        //

        CuResult = Cu->MemcpyHtoDAsync(DeviceContext->DeviceAttributes,
                                       &DeviceContext->Device->Attributes,
                                       sizeof(CU_DEVICE_ATTRIBUTES),
                                       DeviceContext->Stream);
        CU_CHECK(CuResult, MemcpyHtoDAsync);

        //
        // Finally, pop the context.
        //

        CuResult = Cu->CtxPopCurrent(NULL);
        CU_CHECK(CuResult, CtxPopCurrent);

    }

    //
    // Time to allocate graph instances.
    //

    Graph = NULL;
    GpuGraphs = NULL;
    CpuGraphs = NULL;

    NumberOfGpuGraphs = Context->NumberOfGpuGraphs;
    NumberOfCpuGraphs = Context->NumberOfCpuGraphs;
    TotalNumberOfGraphs = NumberOfCpuGraphs + NumberOfGpuGraphs;
    Context->TotalNumberOfGraphs = TotalNumberOfGraphs;

    Graphs = Allocator->Vtbl->Calloc(Allocator,
                                     TotalNumberOfGraphs,
                                     sizeof(Graphs[0]));
    if (Graphs == NULL) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    GpuGraphs = Graphs;
    CpuGraphs = Graphs + NumberOfGpuGraphs;

    Context->GpuGraphs = GpuGraphs;
    Context->CpuGraphs = CpuGraphs;

    //
    // Create GPU graphs and assign one to each solve context.
    //

    SpareGraphCount = 0;
    DeviceContext = &DeviceContexts->DeviceContexts[0];
    SolveContext = &SolveContexts->SolveContexts[0];

    for (Index = 0; Index < NumberOfGpuGraphs; Index++) {

        Result = Context->Vtbl->CreateInstance(Context,
                                               NULL,
                                               &IID_PERFECT_HASH_GRAPH_CU,
                                               &Graph);

        if (FAILED(Result)) {

            //
            // Suppress logging for out-of-memory errors (as we communicate
            // memory issues back to the caller via informational return codes).
            //

            if (Result != E_OUTOFMEMORY) {
                PH_ERROR(InitializeCudaAndGraphsChm02_CreateGpuGraph, Result);
            }

            goto Error;
        }

        ASSERT(IsCuGraph(Graph));

        Graph->Index = Index;
        Graphs[Index] = Graph;

        Graph->CuDeviceIndex =
            InterlockedIncrement(&SolveContext->DeviceContext->NextDeviceIndex);

        Graph->CuSolveContext = SolveContext;

        ASSERT(SolveContext->DeviceGraph != NULL);

        Graph->CuRngId = Context->CuRngId;
        Graph->CuRngSeed = Context->CuRngSeed;
        Graph->CuRngSubsequence = Context->CuRngSubsequence;
        Graph->CuRngOffset = Context->CuRngOffset;

        if (!FindBestGraph(Context)) {
            SolveContext->HostGraph = Graph;
            SolveContext++;
        } else {

            ASSERT(SolveContext->DeviceSpareGraph != NULL);

            //
            // If the index is even (least significant bit is not set), this is
            // a normal graph.  If it's odd (LSB is 1), it's a spare graph.  We
            // advance the solve context pointer after every spare graph.  E.g.
            // if we have two solve contexts, we'll have four GPU graphs, which
            // will be mapped as follows:
            //
            //      Graph #0            -> SolveContext #0
            //      Graph #1 (spare)    -> SolveContext #0; SolveContext++
            //      Graph #2            -> SolveContext #1
            //      Graph #3 (spare)    -> SolveContext #1; SolveContext++
            //      etc.
            //

            if ((Index & 0x1) == 0) {

                //
                // This is a normal graph.
                //

                Graph->Flags.IsSpare = FALSE;
                SolveContext->HostGraph = Graph;
                ASSERT(SolveContext->HostSpareGraph == NULL);

            } else {

                //
                // This is a spare graph.
                //

                Graph->Flags.IsSpare = TRUE;
                SolveContext->HostSpareGraph = Graph;
                ASSERT(SolveContext->HostGraph != NULL);

                //
                // Advance the solve context.
                //

                SolveContext++;
            }
        }
    }

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // If there are CPU solver threads, create CPU graph instances.
    //

    if (Context->NumberOfCpuThreads > 0) {

        SpareGraphCount = 0;

        for (Index = NumberOfGpuGraphs;
             Index < (NumberOfGpuGraphs + NumberOfCpuGraphs);
             Index++)
        {

            Result = Context->Vtbl->CreateInstance(Context,
                                                   NULL,
                                                   &IID_PERFECT_HASH_GRAPH,
                                                   &Graph);

            if (FAILED(Result)) {

                //
                // Suppress logging for out-of-memory errors (as we communicate
                // memory issues back to the caller via informational return
                // codes).
                //

                if (Result != E_OUTOFMEMORY) {
                    PH_ERROR(InitializeCudaAndGraphsChm02_CreateCpuGraph,
                             Result);
                }

                goto Error;
            }

            ASSERT(!IsCuGraph(Graph));

            Graph->Index = Index;
            Graphs[Index] = Graph;

            if (Index == NumberOfGpuGraphs) {

                //
                // This is the first CPU graph, verify we've captured the
                // correct CPU graph starting point.
                //

                ASSERT(&Graphs[Index] == CpuGraphs);
            }

            if (FindBestGraph(Context)) {

                if ((Index - NumberOfGpuGraphs) < Context->NumberOfCpuThreads) {

                    NOTHING;

                } else {

                    //
                    // There should only ever be one spare CPU graph.
                    //

                    SpareGraphCount++;
                    ASSERT(SpareGraphCount == 1);

                    Graph->Flags.IsSpare = TRUE;

                    //
                    // Context->SpareGraph is guarded by the best graph critical
                    // section.  We know that no worker threads will be running
                    // at this point; inform SAL accordingly by suppressing the
                    // concurrency warnings.
                    //

                    _Benign_race_begin_
                    Context->SpareGraph = Graph;
                    _Benign_race_end_
                }
            }

            //
            // Copy relevant flags over.
            //

            Graph->Flags.SkipVerification =
                TableCreateFlags.SkipGraphVerification;

            Graph->Flags.WantsWriteCombiningForVertexPairsArray =
                TableCreateFlags.EnableWriteCombineForVertexPairs;

            Graph->Flags.RemoveWriteCombineAfterSuccessfulHashKeys =
                TableCreateFlags.RemoveWriteCombineAfterSuccessfulHashKeys;

        }
    }

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // We're done, finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // TODO: loop through any device contexts here and free?
    //

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Release any temporary component references.
    //

    RELEASE(PtxPath);
    RELEASE(PtxFile);
    RELEASE(RuntimeLibPath);
    RELEASE(RuntimeLibFile);

    Allocator = Context->Allocator;
    if (BitmapBuffer != NULL) {
        Allocator->Vtbl->FreePointer(Allocator, &BitmapBuffer);
    }

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

PREPARE_GRAPH_INFO PrepareGraphInfoChm01;
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
    HRESULT Result;
    ULONG NumberOfKeys;

    //
    // Call out to the Chm01 preparation version first.
    //

    Result = PrepareGraphInfoChm01(Table, Info, PrevInfo);
    if (FAILED(Result)) {
        return Result;
    }

    //
    // CUDA-specific logic.
    //

    NumberOfKeys = Table->Keys->NumberOfElements.LowPart;

    Info->VertexPairsSizeInBytes = ALIGN_UP_ZMMWORD(
        RTL_ELEMENT_SIZE(GRAPH, VertexPairs) *
        (ULONGLONG)NumberOfKeys
    );

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

        if (!SetEvent(*Event)) {
            SYS_ERROR(SetEvent);
        }

        SetStopSolving(Context);
    }
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
