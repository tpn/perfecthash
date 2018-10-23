/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm_01.c

Abstract:

    This module implements the CHM perfect hash table algorithm.

--*/

#include "stdafx.h"
#include "Chm01.h"

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(NTAPI PREPARE_GRAPH_INFO)(
    _In_ PPERFECT_HASH_TABLE Table,
    _Out_ PGRAPH_INFO Info
    );
typedef PREPARE_GRAPH_INFO *PPREPARE_GRAPH_INFO;

extern PREPARE_GRAPH_INFO PrepareGraphInfoChm01;

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(NTAPI PREPARE_TABLE_OUTPUT_DIRECTORY)(
    _In_ PPERFECT_HASH_TABLE Table
    );
typedef PREPARE_TABLE_OUTPUT_DIRECTORY *PPREPARE_TABLE_OUTPUT_DIRECTORY;

extern PREPARE_TABLE_OUTPUT_DIRECTORY PrepareTableOutputDirectory;

//
// Define the threshold for how many attempts need to be made at finding a
// perfect hash solution before we double our number of vertices and try again.
//
// With a 2-part hypergraph, solutions are found on average in sqrt(3) attempts.
// By attempt 18, there's a 99.9% chance we will have found a solution.
//

#define GRAPH_SOLVING_ATTEMPTS_THRESHOLD 18

//
// Define a limit for how many times the table resizing will be attempted before
// giving up.  For large table sizes and large concurrency values, note that we
// may hit memory limits before we hit this resize limit.
//

#define GRAPH_SOLVING_RESIZE_TABLE_LIMIT 5

//
// Define helper macros for checking prepare and save file work errors.
//

#define EXPAND_AS_CHECK_ERRORS(Verb, VUpper, Name, Upper)                \
    if (Verb####Name##.NumberOfErrors > 0) {                             \
        Result = Verb####Name##.LastResult;                              \
        if (Result == S_OK || Result == E_UNEXPECTED) {                  \
            Result = PH_E_ERROR_DURING_##VUpper##_##Upper##;             \
        }                                                                \
        PH_ERROR(                                                        \
            CreatePerfectHashTableImplChm01_ErrorDuring##Verb####Name##, \
            Result                                                       \
        );                                                               \
        goto Error;                                                      \
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


    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - Table was NULL.

    E_UNEXPECTED - Catastrophic internal error.

    E_OUTOFMEMORY - Out of memory.

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

    PH_E_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT - The shutdown event
        explicitly set.

    PH_E_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE - The requested number
        of table elements exceeded limits.  If a table resize event occurrs,
        the number of requested table elements is doubled.  If this number
        exceeds MAX_ULONG, this error will be returned.

    PH_E_TABLE_VERIFICATION_FAILED - The winning perfect hash table solution
        failed internal verification.  The primary cause of this is typically
        when collisions are detected during verification.

    PH_E_INVALID_NUMBER_OF_SEEDS - The number of seeds required for the given
        hash function exceeds the number of seeds available in the on-disk
        table info structure.

    PH_E_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED - The maximum number
        of table resize events was reached before a solution could be found.

--*/
{
    PRTL Rtl;
    USHORT Index;
    PULONG Keys;
    PGRAPH *Graphs = NULL;
    PGRAPH Graph;
    BOOLEAN Success;
    ULONG Attempt = 0;
    ULONG ReferenceCount;
    BYTE NumberOfEvents;
    HRESULT Result = S_OK;
    ULONG WaitResult;
    GRAPH_INFO Info;
    PALLOCATOR Allocator;
    PHANDLE Event;
    ULONG NumberOfGraphs;
    PLIST_ENTRY ListEntry;
    ULONG NumberOfSeedsRequired;
    ULONG NumberOfSeedsAvailable;
    ULONGLONG Closest;
    ULONGLONG LastClosest;
    GRAPH_INFO_ON_DISK GraphInfo;
    PGRAPH_INFO_ON_DISK GraphInfoOnDisk;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;
    PPERFECT_HASH_CONTEXT Context;
    BOOL WaitForAllEvents = TRUE;

    HANDLE Events[5];
    HANDLE SaveEvents[NUMBER_OF_SAVE_FILE_EVENTS];
    HANDLE PrepareEvents[NUMBER_OF_PREPARE_FILE_EVENTS];
    PHANDLE SaveEvent = SaveEvents;
    PHANDLE PrepareEvent = PrepareEvents;

#define EXPAND_AS_STACK_VAR(Verb, VUpper, Name, Upper) \
    FILE_WORK_ITEM Verb##Name;

    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_STACK_VAR);
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_STACK_VAR);

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    } else {
        Context = Table->Context;
        NumberOfGraphs = Context->MaximumConcurrency;
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

#define EXPAND_AS_ASSIGN_EVENT(Verb, VUpper, Name, Upper) \
    *##Verb##Event++ = Context->##Verb##d##Name##Event;

    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_EVENT);
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_EVENT);

    //
    // Initialize aliases.
    //

    Rtl = Table->Rtl;
    Keys = (PULONG)Table->Keys->File->BaseAddress;
    Allocator = Table->Allocator;
    MaskFunctionId = Table->MaskFunctionId;
    GraphInfoOnDisk = Context->GraphInfoOnDisk = &GraphInfo;
    TableInfoOnDisk = Table->TableInfoOnDisk = &GraphInfo.TableInfoOnDisk;

    ASSERT(
        Context->FinishedWorkList->Vtbl->IsEmpty(Context->FinishedWorkList)
    );

    //
    // If no threshold has been set, use the default.
    //

    if (!Context->ResizeTableThreshold) {
        Context->ResizeTableThreshold = GRAPH_SOLVING_ATTEMPTS_THRESHOLD;
        Context->ResizeLimit = GRAPH_SOLVING_RESIZE_TABLE_LIMIT;
    }

    //
    // Set the "first graph wins" flag.
    //

    SetFirstSolvedGraphWins(Context);

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

    Result = PrepareGraphInfoChm01(Table, &Info);
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
            sizeof(*Graph)
        )
    );

    if (!Graphs) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // Create graph instances and capture the resulting pointer in the array
    // we just allocated above.
    //

    for (Index = 0; Index < NumberOfGraphs; Index++) {

        Result = Table->Vtbl->CreateInstance(Table,
                                             NULL,
                                             &IID_PERFECT_HASH_GRAPH,
                                             &Graph);

        if (FAILED(Result)) {
            PH_ERROR(CreatePerfectHashTableImplChm01_CreateGraph, Result);
            goto Error;
        }

        Graphs[Index] = Graph;
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
    // If this isn't the first attempt, prepare the graph info again.  This
    // updates the various allocation sizes based on the new table size being
    // requested.
    //

    if (++Attempt > 1) {

        Result = PrepareGraphInfoChm01(Table, &Info);
        if (FAILED(Result)) {
            PH_ERROR(CreatePerfectHashTableImplChm01_PrepareGraphInfo, Result);
            goto Error;
        }

    }

    //
    // Set the context's main work callback to our worker routine, and the algo
    // context to our graph info structure.
    //

    Context->MainWorkCallback = ProcessGraphCallbackChm01;
    Context->AlgorithmContext = &Info;

    //
    // Set the context's file work callback to our worker routine.
    //

    Context->FileWorkCallback = FileWorkCallbackChm01;

    //
    // Prepare the table output directory.
    //

    Result = PrepareTableOutputDirectory(Table);
    if (FAILED(Result)) {
        PH_ERROR(PrepareTableOutputDirectory, Result);
        goto Error;
    }

    //
    // Submit all of the file preparation work items.
    //

#define EXPAND_AS_SUBMIT_FILE_WORK(Verb, VUpper, Name, Upper) \
    ZeroStruct(##Verb####Name##);                             \
    Verb##Name##.FileWorkId = FileWork##Verb##Name##Id;       \
    InsertTailFileWork(Context, &Verb##Name##.ListEntry);     \
    SubmitThreadpoolWork(Context->FileWork);

#define SUBMIT_PREPARE_FILE_WORK() \
    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_FILE_WORK)

#define SUBMIT_SAVE_FILE_WORK() \
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_FILE_WORK)

    ASSERT(Context->FileWorkList->Vtbl->IsEmpty(Context->FileWorkList));

    SUBMIT_PREPARE_FILE_WORK();

    //
    // Capture initial cycles as reported by __rdtsc() and the performance
    // counter.  The former is used to report a raw cycle count, the latter
    // is used to convert to microseconds reliably (i.e. unaffected by turbo
    // boosting).
    //

    QueryPerformanceFrequency(&Context->Frequency);

    CONTEXT_START_TIMERS(Solve);

    //
    // For each graph instance, set the graph info, then submit threadpool work
    // against the context's main work threadpool.
    //

    ASSERT(Context->MainWorkList->Vtbl->IsEmpty(Context->MainWorkList));

    for (Index = 0; Index < NumberOfGraphs; Index++) {

        Graph = Graphs[Index];

        Result = Graph->Vtbl->SetInfo(Graph, &Info);

        if (FAILED(Result)) {
            PH_ERROR(GraphSetInfo, Result);
            goto Error;
        }

        InitializeListHead(&Graph->ListEntry);
        InsertTailMainWork(Context, &Graph->ListEntry);
        SubmitThreadpoolWork(Context->MainWork);

        //
        // If our key set size is small and our maximum concurrency is large,
        // we may have already solved the graph, in which case, we can stop
        // submitting new solver attempts and just break out of the loop here.
        //

        if (!ShouldWeContinueTryingToSolveGraphChm01(Context)) {
            break;
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
    // If the wait result indicates the try larger table size event was set,
    // deal with that, first.
    //

    if (WaitResult == WAIT_OBJECT_0+4) {

        //
        // The number of attempts at solving this graph have exceeded the
        // threshold.  Set the shutdown event in order to trigger all worker
        // threads to abort their current attempts and wait on the main thread
        // work to complete.
        //

        SetEvent(Context->ShutdownEvent);
        WaitForThreadpoolWorkCallbacks(Context->MainWork, TRUE);

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

        //
        // There are no more threadpool callbacks running.  However, a thread
        // could have finished a solution between the time the try larger table
        // size event was set, and this point.  So, check the finished count
        // first.  If it indicates a solution, jump to that handler code.
        //

        if (Context->FinishedCount > 0) {
            goto FinishedSolution;
        }

        //
        // Check to see if we've exceeded the maximum number of resize events.
        //

        if (Context->NumberOfTableResizeEvents >= Context->ResizeLimit) {
            Result = PH_E_MAXIMUM_NUMBER_OF_TABLE_RESIZE_EVENTS_REACHED;
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
            Result = PH_E_REQUESTED_NUMBER_OF_TABLE_ELEMENTS_TOO_LARGE;
            goto Error;
        }

        //
        // Reset the lists.
        //

        ResetMainWorkList(Context);
        ResetFileWorkList(Context);
        ResetFinishedWorkList(Context);

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

    if (!Success) {

        BOOL CancelPending = TRUE;

        //
        // Invariant check: if no worker thread registered a solved graph
        // (indicated by Context->FinishedCount having a value greater than
        // 0), then verify that the shutdown event was set.
        //
        // If our WaitResult above indicates WAIT_OBJECT_2, we're done.  If
        // not, verify explicitly.
        //

        if (WaitResult != WAIT_OBJECT_0+2) {

            //
            // Manually test that the shutdown event has been signaled.
            //

            WaitResult = WaitForSingleObject(Context->ShutdownEvent, 0);

            if (WaitResult != WAIT_OBJECT_0) {
                Result = PH_E_INVARIANT_CHECK_FAILED;
                PH_ERROR(CreatePerfectHashTableImplChm01_ShutdownEvent, Result);
                goto Error;
            }
        }

        Result = PH_E_CREATE_TABLE_ROUTINE_RECEIVED_SHUTDOWN_EVENT;

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

        WaitForThreadpoolWorkCallbacks(Context->FileWork, CancelPending);

        goto End;
    }

    //
    // Intentional follow-on to FinishedSolution.
    //

FinishedSolution:

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

    //
    // Dispatch save file work for the table data.
    //

    SUBMIT_SAVE_FILE_WORK();

    //
    // Capture another round of cycles and performance counter values, then
    // continue with verification of the solution.
    //

    CONTEXT_START_TIMERS(Verify);

    Result = VerifySolvedGraph(Graph);

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
    WaitForThreadpoolWorkCallbacks(Context->FileWork, TRUE);

    //
    // Intentional follow-on to End.
    //

End:

    if (Graphs) {

        //
        // Walk the array of graph instances and release each one, then free
        // the array buffer.
        //

        for (Index = 0; Index < NumberOfGraphs; Index++) {

            Graph = Graphs[Index];
            ReferenceCount = Graph->Vtbl->Release(Graph);

            //
            // Invariant check: reference count should always be 0 here.
            //

            if (ReferenceCount != 0) {
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
            }

            Graphs[Index] = NULL;
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

    return Result;
}


PREPARE_GRAPH_INFO PrepareGraphInfoChm01;

_Use_decl_annotations_
HRESULT
PrepareGraphInfoChm01(
    PPERFECT_HASH_TABLE Table,
    PGRAPH_INFO Info
    )
/*++

Routine Description:

    Prepares the GRAPH_INFO structure for a given table.

Arguments:

    Table - Supplies a pointer to the table.

    Info - Supplies a pointer to the graph info structure to prepare.

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
    PGRAPH_INFO_ON_DISK GraphInfoOnDisk;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    ULONGLONG NextSizeInBytes;
    ULONGLONG PrevSizeInBytes;
    ULONGLONG FirstSizeInBytes;
    ULONGLONG EdgesSizeInBytes;
    ULONGLONG ValuesSizeInBytes;
    ULONGLONG AssignedSizeInBytes;
    PPERFECT_HASH_CONTEXT Context;
    ULARGE_INTEGER AllocSize;
    ULARGE_INTEGER NumberOfEdges;
    ULARGE_INTEGER NumberOfVertices;
    ULARGE_INTEGER TotalNumberOfEdges;
    ULARGE_INTEGER DeletedEdgesBitmapBufferSizeInBytes;
    ULARGE_INTEGER VisitedVerticesBitmapBufferSizeInBytes;
    ULARGE_INTEGER AssignedBitmapBufferSizeInBytes;
    ULARGE_INTEGER IndexBitmapBufferSizeInBytes;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;

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
            // rounded up to a power of 2.  The number of edges will be rounded
            // up to a power of 2 from the number of keys.
            //

            NumberOfVertices.QuadPart = (
                RoundUpPowerOf2(NumberOfVertices.LowPart)
            );

            NumberOfEdges.QuadPart = RoundUpPowerOf2(NumberOfKeys);

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

        if (IsModulusMasking(MaskFunctionId)) {

            NumberOfVertices.QuadPart = NumberOfEdges.QuadPart << 1ULL;
            NumberOfVertices.QuadPart += NumberOfEdges.QuadPart >> 2ULL;

        } else {

            //
            // Round up the edges to a power of 2.
            //

            NumberOfEdges.QuadPart = RoundUpPowerOf2(NumberOfEdges.LowPart);

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
                RoundUpNextPowerOf2(NumberOfEdges.LowPart)
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
    // Calculate the size required for the DeletedEdges bitmap buffer.  One
    // bit is used per TotalNumberOfEdges.  Convert the bits into bytes by
    // shifting right 3 (dividing by 8) then align it up to a 16 byte boundary.
    // We add 1 before shifting to account 1-based bitmaps vs 0-based indices.
    //

    DeletedEdgesBitmapBufferSizeInBytes.QuadPart = (
        ALIGN_UP(((ALIGN_UP(TotalNumberOfEdges.QuadPart + 1, 8)) >> 3), 16)
    );

    if (DeletedEdgesBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_DeletedEdgesBitmap, Result);
        goto Error;

    }

    //
    // Calculate the size required for the VisitedVertices bitmap buffer.  One
    // bit is used per NumberOfVertices.  Convert the bits into bytes by
    // shifting right 3 (dividing by 8) then align it up to a 16 byte boundary.
    // We add 1 before shifting to account 1-based bitmaps vs 0-based indices.
    //

    VisitedVerticesBitmapBufferSizeInBytes.QuadPart = (
        ALIGN_UP(((ALIGN_UP(NumberOfVertices.QuadPart + 1, 8)) >> 3), 16)
    );

    if (VisitedVerticesBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_VisitedVerticesBitmap, Result);
        goto Error;
    }

    //
    // Calculate the size required for the AssignedBitmap bitmap buffer.  One
    // bit is used per NumberOfVertices.  Convert the bits into bytes by shifting
    // right 3 (dividing by 8) then align it up to a 16 byte boundary.
    // We add 1 before shifting to account 1-based bitmaps vs 0-based indices.
    //

    AssignedBitmapBufferSizeInBytes.QuadPart = (
        ALIGN_UP(((ALIGN_UP(NumberOfVertices.QuadPart + 1, 8)) >> 3), 16)
    );

    if (AssignedBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_AssignedBitmap, Result);
        goto Error;
    }

    //
    // Calculate the size required for the IndexBitmap bitmap buffer.  One
    // bit is used per NumberOfVertices.  Convert the bits into bytes by shifting
    // right 3 (dividing by 8) then align it up to a 16 byte boundary.
    // We add 1 before shifting to account 1-based bitmaps vs 0-based indices.
    //

    IndexBitmapBufferSizeInBytes.QuadPart = (
        ALIGN_UP(((ALIGN_UP(NumberOfVertices.QuadPart + 1, 8)) >> 3), 16)
    );

    if (IndexBitmapBufferSizeInBytes.HighPart) {
        Result = PH_E_TOO_MANY_BITS_FOR_BITMAP;
        PH_ERROR(PrepareGraphInfoChm01_IndexBitmap, Result);
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

    PrevSizeInBytes = ALIGN_UP_YMMWORD(
        RTL_ELEMENT_SIZE(GRAPH, Prev) * TotalNumberOfEdges.QuadPart
    );

    AssignedSizeInBytes = ALIGN_UP_YMMWORD(
        RTL_ELEMENT_SIZE(GRAPH, Assigned) * NumberOfVertices.QuadPart
    );

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
    // Calculate the total size required for the underlying graph.
    //

    AllocSize.QuadPart = ALIGN_UP_YMMWORD(
        EdgesSizeInBytes +
        NextSizeInBytes +
        FirstSizeInBytes +
        PrevSizeInBytes +
        AssignedSizeInBytes +
        ValuesSizeInBytes +

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

    ZeroStructPointer(Info);

    Info->PageSize = PAGE_SIZE;
    Info->AllocSize = AllocSize.QuadPart;
    Info->Context = Context;
    Info->NumberOfPagesPerGraph = (ULONG)BYTES_TO_PAGES(AllocSize.QuadPart);
    Info->NumberOfBitmaps = NumberOfBitmaps;
    Info->SizeOfGraphStruct = ALIGN_UP_YMMWORD(sizeof(GRAPH));
    Info->EdgesSizeInBytes = EdgesSizeInBytes;
    Info->NextSizeInBytes = NextSizeInBytes;
    Info->FirstSizeInBytes = FirstSizeInBytes;
    Info->PrevSizeInBytes = PrevSizeInBytes;
    Info->AssignedSizeInBytes = AssignedSizeInBytes;
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
        TrailingZeros64(RoundUpPowerOf2(NumberOfEdges.LowPart))
    );

    Dim->NumberOfEdgesNextPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpNextPowerOf2(NumberOfEdges.LowPart))
    );

    Dim->NumberOfVerticesPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpPowerOf2(NumberOfVertices.LowPart))
    );

    Dim->NumberOfVerticesNextPowerOf2Exponent = (BYTE)(
        TrailingZeros64(RoundUpNextPowerOf2(NumberOfVertices.LowPart))
    );

    //
    // If non-modulus masking is active, initialize the edge and vertex masks.
    //

    if (!IsModulusMasking(MaskFunctionId)) {

        Info->EdgeMask = NumberOfEdges.LowPart - 1;
        Info->VertexMask = NumberOfVertices.LowPart - 1;

        //
        // Sanity check our masks are correct: their popcnts should match the
        // exponent value identified above whilst filling out the dimensions
        // structure.
        //

        ASSERT(PopulationCount32(Info->EdgeMask) ==
               Dim->NumberOfEdgesPowerOf2Exponent);

        ASSERT(PopulationCount32(Info->VertexMask) ==
               Dim->NumberOfVerticesPowerOf2Exponent);

    }

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
    Table->HashShift = TrailingZeros(Table->HashSize);
    Table->IndexShift = TrailingZeros(Table->IndexSize);
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

    ZeroStructPointer(GraphInfoOnDisk);
    TableInfoOnDisk->Magic.LowPart = TABLE_INFO_ON_DISK_MAGIC_LOWPART;
    TableInfoOnDisk->Magic.HighPart = TABLE_INFO_ON_DISK_MAGIC_HIGHPART;
    TableInfoOnDisk->SizeOfStruct = sizeof(*GraphInfoOnDisk);
    TableInfoOnDisk->Flags.AsULong = 0;
    TableInfoOnDisk->Concurrency = Context->MaximumConcurrency;
    TableInfoOnDisk->AlgorithmId = Context->AlgorithmId;
    TableInfoOnDisk->MaskFunctionId = Context->MaskFunctionId;
    TableInfoOnDisk->HashFunctionId = Context->HashFunctionId;
    TableInfoOnDisk->KeySizeInBytes = sizeof(Table->Keys->SizeOfKeyInBytes);
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


PREPARE_TABLE_OUTPUT_DIRECTORY PrepareTableOutputDirectory;

_Use_decl_annotations_
HRESULT
PrepareTableOutputDirectory(
    PPERFECT_HASH_TABLE Table
    )
{
    HRESULT Result = S_OK;
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
    // Initialize aliases.
    //

    Context = Table->Context;
    BaseOutputDirectory = Context->BaseOutputDirectory;
    BaseOutputDirectoryPath = &BaseOutputDirectory->Path->FullPath;
    NumberOfTableElements.QuadPart = (
        Table->TableInfoOnDisk->NumberOfTableElements.QuadPart
    );

    //
    // Release the existing output path, if applicable.  (This will already
    // have a value if we're being called for the second or more time due to
    // a resize event.)
    //

    RELEASE(Table->OutputPath);

    //
    // Create an output directory path name.
    //


    Result = PerfectHashTableCreatePath(Table,
                                        Table->Keys->File->Path,
                                        &NumberOfTableElements,
                                        Table->AlgorithmId,
                                        Table->MaskFunctionId,
                                        Table->HashFunctionId,
                                        BaseOutputDirectoryPath,
                                        NULL,           // NewBaseName
                                        NULL,           // AdditionalSuffix
                                        &EmptyString,   // NewExtension
                                        NULL,           // NewStreamName
                                        &OutputPath,
                                        NULL);

    if (FAILED(Result)) {
        goto Error;
    }

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
                                             &OutputDir);

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
        // Directory creation was successful.  AddRef() against the instance
        // to capture the table's ownership of it.  (We release the local
        // OutputDir at the end of this routine.)
        //

        Table->OutputDirectory = OutputDir;
        OutputDir->Vtbl->AddRef(OutputDir);

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

    //
    // Release applicable COM references.
    //

    RELEASE(OutputPath);
    RELEASE(OutputDir);

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
    PRTL Rtl;
    PGRAPH Graph;
    ULONG Attempt = 0;
    PGRAPH_INFO Info;
    PRTL_FILL_PAGES FillPages;

    UNREFERENCED_PARAMETER(Instance);

    //
    // Resolve the graph base address from the list entry.  Nothing will be
    // filled in initially.
    //

    Graph = CONTAINING_RECORD(ListEntry, GRAPH, ListEntry);

    //
    // Resolve aliases.
    //

    Rtl = Context->Rtl;
    FillPages = Rtl->Vtbl->FillPages;

    //
    // The graph info structure will be stashed in the algo context field.
    //

    Info = (PGRAPH_INFO)Context->AlgorithmContext;

    //
    // Begin the solving loop.  InitializeGraph() generates new seed data,
    // so each loop iteration will be attempting to solve the graph uniquely.
    //

    while (ShouldWeContinueTryingToSolveGraphChm01(Context)) {

        InitializeGraph(Info, Graph);

        Graph->ThreadAttempt = ++Attempt;

        if (SolveGraph(Graph)) {

            //
            // Hey, we were the ones to solve it, great!
            //

            break;
        }

        //
        // Our attempt at solving failed.  Zero all pages associated with the
        // graph and then try again with new seed data.
        //

        PH_RAISE(PH_E_WORK_IN_PROGRESS);
        //FillPages(Rtl, (PCHAR)Graph, 0, Info->NumberOfPagesPerGraph);

    }

    return;
}

SHOULD_WE_CONTINUE_TRYING_TO_SOLVE_GRAPH
    ShouldWeContinueTryingToSolveGraphChm01;

_Use_decl_annotations_
BOOLEAN
ShouldWeContinueTryingToSolveGraphChm01(
    PPERFECT_HASH_CONTEXT Context
    )
{
    ULONG WaitResult;
    HANDLE Events[4];
    USHORT NumberOfEvents = ARRAYSIZE(Events);

    Events[0] = Context->ShutdownEvent;
    Events[1] = Context->SucceededEvent;
    Events[2] = Context->FailedEvent;
    Events[3] = Context->CompletedEvent;

    //
    // Fast-path exit: if the finished count is not 0, then someone has already
    // solved the solution, and we don't need to wait on any of the events.
    //

    PH_RAISE(PH_E_WORK_IN_PROGRESS);
    if (Context->FinishedCount > 0) {
        return FALSE;
    }

    WaitResult = WaitForMultipleObjects(NumberOfEvents,
                                        Events,
                                        FALSE,
                                        0);

    //
    // The only situation where we continue attempting to solve the graph is
    // if the result from the wait is WAIT_TIMEOUT, which indicates none of
    // the events have been set.  We treat any other situation as an indication
    // to stop processing.  (This includes wait failures and abandonment.)
    //

    return (WaitResult == WAIT_TIMEOUT ? TRUE : FALSE);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
