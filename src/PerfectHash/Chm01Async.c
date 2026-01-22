/*++

Copyright (c) 2018-2026 Trent Nelson <trent@trent.me>

Module Name:

    Chm01Async.c

Abstract:

    This module implements the CHM01 asynchronous table creation state
    machine.  Work is decomposed into fine-grained IOCP-driven steps that
    overlap compute and file I/O.

--*/

#include "stdafx.h"
#include "Chm01.h"
#include "Chm01Private.h"
#include "Chm01FileWork.h"
#include "Chm01Async.h"

//
// Async job flags.
//

#define CHM01_ASYNC_JOB_FLAG_CLOSE_DELETES_FILES   0x00000001
#define CHM01_ASYNC_JOB_FLAG_CLOSE_SUBMITTED       0x00000002
#define CHM01_ASYNC_JOB_FLAG_SAVE_SUBMITTED        0x00000008
#define CHM01_ASYNC_IOCP_PUMP_TIMEOUT_MS           100

typedef struct _CHM01_ASYNC_GRAPH_WORK {
    PERFECT_HASH_ASYNC_WORK Work;
    PCHM01_ASYNC_JOB Job;
    PGRAPH Graph;
    BOOLEAN Started;
    BYTE Padding[7];
} CHM01_ASYNC_GRAPH_WORK;
typedef CHM01_ASYNC_GRAPH_WORK *PCHM01_ASYNC_GRAPH_WORK;

FORCEINLINE
BOOLEAN
Chm01AsyncShouldSkipContextFileWork(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ FILE_WORK_ID FileWorkId
    )
{
    FILE_ID FileId;

    if (!SkipContextFileWork(Context)) {
        return FALSE;
    }

    FileId = FileWorkIdToFileId(FileWorkId);

    return IsValidContextFileId((CONTEXT_FILE_ID)FileId);
}

static
VOID
Chm01AsyncLogJobState(
    _In_ PCHM01_ASYNC_JOB Job,
    _In_ ULONG Stage
    )
{
#ifdef PH_WINDOWS
    int Count;
    DWORD BytesWritten;
    HANDLE LogHandle;
    CHAR Buffer[512];
    LONG Outstanding = 0;
    LONG ActiveGraphs = 0;
    ULONG State = 0;
    ULONG Attempt = 0;
    ULONG Flags = 0;
    HRESULT Result = S_OK;
    ULONG NameChars = 0;
    PCWSTR NameBuffer = L"";
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_KEYS Keys;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH Path;

    if (GetEnvironmentVariableW(L"PH_LOG_CHM01_ASYNC_JOB", NULL, 0) == 0) {
        return;
    }

    if (!ARGUMENT_PRESENT(Job)) {
        return;
    }

    Outstanding = Job->Async.Outstanding;
    ActiveGraphs = Job->ActiveGraphs;
    State = (ULONG)Job->State;
    Attempt = Job->Attempt;
    Flags = Job->Flags;
    Result = Job->LastResult;

    Table = Job->Table;
    Keys = Table ? Table->Keys : NULL;
    File = Keys ? Keys->File : NULL;
    Path = File ? GetActivePath(File) : NULL;

    if (Path && Path->FileName.Buffer) {
        NameBuffer = Path->FileName.Buffer;
        NameChars = Path->FileName.Length / sizeof(WCHAR);
    }

    LogHandle = CreateFileW(L"PerfectHashChm01AsyncJob.log",
                            FILE_APPEND_DATA,
                            FILE_SHARE_READ,
                            NULL,
                            OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL,
                            NULL);
    if (!IsValidHandle(LogHandle)) {
        return;
    }

    Count = _snprintf_s(
        Buffer,
        sizeof(Buffer),
        _TRUNCATE,
        "Stage=%lu State=%lu Outstanding=%ld ActiveGraphs=%ld Attempt=%lu "
        "Flags=0x%08lX Result=0x%08lX Name=%.*S\r\n",
        Stage,
        State,
        Outstanding,
        ActiveGraphs,
        Attempt,
        Flags,
        Result,
        (int)NameChars,
        NameBuffer
    );

    if (Count > 0) {
        WriteFile(LogHandle,
                  Buffer,
                  (DWORD)strlen(Buffer),
                  &BytesWritten,
                  NULL);
    }

    CloseHandle(LogHandle);
#else
    UNREFERENCED_PARAMETER(Job);
    UNREFERENCED_PARAMETER(Stage);
#endif
}

static
HRESULT
Chm01AsyncGraphStep(
    _Inout_ PPERFECT_HASH_ASYNC_WORK Work
    );

static
VOID
Chm01AsyncGraphComplete(
    _Inout_ PPERFECT_HASH_ASYNC_WORK Work,
    _In_ HRESULT Result
    );

static
HRESULT
Chm01AsyncDispatchGraphWork(
    _In_ PCHM01_ASYNC_JOB Job
    );

static
HRESULT
Chm01AsyncInitializeJob(
    _In_ PCHM01_ASYNC_JOB Job
    );

static
HRESULT
Chm01AsyncPollSolveEvents(
    _In_ PCHM01_ASYNC_JOB Job
    );

static
HRESULT
Chm01AsyncFinalizeVerify(
    _In_ PCHM01_ASYNC_JOB Job
    );

static
HRESULT
Chm01AsyncFinalizeWaitSave(
    _In_ PCHM01_ASYNC_JOB Job
    );

static
HRESULT
Chm01AsyncFinalizeClose(
    _In_ PCHM01_ASYNC_JOB Job
    );

static
HRESULT
Chm01AsyncReleaseGraphs(
    _In_ PCHM01_ASYNC_JOB Job
    );

static
HRESULT
Chm01AsyncStep(
    _Inout_ PPERFECT_HASH_ASYNC_WORK Work
    );

static
VOID
Chm01AsyncComplete(
    _Inout_ PPERFECT_HASH_ASYNC_WORK Work,
    _In_ HRESULT Result
    );

_Use_decl_annotations_
static
HRESULT
Chm01AsyncGraphStep(
    PPERFECT_HASH_ASYNC_WORK Work
    )
{
    PGRAPH Graph;
    PGRAPH LockedGraph;
    PGRAPH NewGraph;
    ULONG OldNumberOfVertices;
    HRESULT Result;
    PPERFECT_HASH_CONTEXT Context;
    PCHM01_ASYNC_GRAPH_WORK GraphWork;

    GraphWork = CONTAINING_RECORD(Work, CHM01_ASYNC_GRAPH_WORK, Work);
    Graph = GraphWork->Graph;
    Context = GraphWork->Job->Context;

    if (!GraphWork->Started) {
        GraphWork->Started = TRUE;
        InterlockedIncrement(&Context->ActiveSolvingLoops);
    }

    LockedGraph = Graph;
    AcquireGraphLockExclusive(LockedGraph);

    if (!IsGraphInfoLoaded(LockedGraph)) {
        Result = LockedGraph->Vtbl->LoadInfo(LockedGraph);
        if (FAILED(Result)) {
            if (Result != E_OUTOFMEMORY) {
                PH_ERROR(GraphLoadInfo, Result);
                Result = PH_E_INVARIANT_CHECK_FAILED;
            } else {
                if (InterlockedDecrement(&Context->GraphMemoryFailures) == 0) {
                    Context->State.AllGraphsFailedMemoryAllocation = TRUE;
                    SetStopSolving(Context);
                    if (!SetEvent(Context->FailedEvent)) {
                        SYS_ERROR(SetEvent);
                    }
                }
                Result = S_OK;
            }
            goto End;
        }
    }

    if (!LockedGraph->Vtbl->ShouldWeContinueTryingToSolve(LockedGraph)) {
        Result = S_OK;
        goto End;
    }

    Result = LockedGraph->Vtbl->Reset(LockedGraph);
    if (FAILED(Result)) {
        if (Result == PH_E_NO_MORE_SEEDS) {
            Result = S_OK;
        }
        goto End;
    }

    if (Result != PH_S_CONTINUE_GRAPH_SOLVING) {
        Result = S_OK;
        goto End;
    }

    Result = LockedGraph->Vtbl->LoadNewSeeds(LockedGraph);
    if (FAILED(Result)) {
        if (Result == PH_E_NO_MORE_SEEDS) {
            Result = S_OK;
        }
        goto End;
    }

    NewGraph = NULL;
    Result = LockedGraph->Vtbl->Solve(LockedGraph, &NewGraph);
    if (FAILED(Result)) {
        goto End;
    }

    if (Result == PH_S_STOP_GRAPH_SOLVING ||
        Result == PH_S_GRAPH_SOLVING_STOPPED) {
        Result = S_OK;
        goto End;
    }

    if (Result == PH_S_USE_NEW_GRAPH_FOR_SOLVING) {
        if (!NewGraph) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            goto End;
        }

        OldNumberOfVertices = LockedGraph->NumberOfVertices;

        AcquireGraphLockExclusive(NewGraph);
        ReleaseGraphLockExclusive(LockedGraph);
        LockedGraph = NewGraph;
        GraphWork->Graph = NewGraph;

        if (!IsGraphInfoLoaded(LockedGraph) ||
            LockedGraph->LastLoadedNumberOfVertices < OldNumberOfVertices) {
            Result = LockedGraph->Vtbl->LoadInfo(LockedGraph);
            if (FAILED(Result)) {
                PH_ERROR(GraphLoadInfo_NewGraph, Result);
                goto End;
            }
        }

        Result = S_FALSE;
        goto End;
    }

    ASSERT(Result == PH_S_CONTINUE_GRAPH_SOLVING);

    Result = S_FALSE;

End:

    if (LockedGraph) {
        ReleaseGraphLockExclusive(LockedGraph);
    }

    return Result;
}

_Use_decl_annotations_
static
VOID
Chm01AsyncGraphComplete(
    PPERFECT_HASH_ASYNC_WORK Work,
    HRESULT Result
    )
{
    PGRAPH Graph;
    PHANDLE Event;
    PPERFECT_HASH_CONTEXT Context;
    PCHM01_ASYNC_JOB Job;
    PCHM01_ASYNC_GRAPH_WORK GraphWork;

    UNREFERENCED_PARAMETER(Result);

    GraphWork = CONTAINING_RECORD(Work, CHM01_ASYNC_GRAPH_WORK, Work);
    Job = GraphWork->Job;
    Graph = GraphWork->Graph;
    Context = Job->Context;

    if (GraphWork->Started) {
        InterlockedDecrement(&Context->ActiveSolvingLoops);
    }

    if (InterlockedDecrement(&Context->RemainingSolverLoops) == 0) {
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

    if (InterlockedDecrement(&Job->ActiveGraphs) == 0) {
        if (Job->GraphsCompleteEvent) {
            SetEvent(Job->GraphsCompleteEvent);
        }
    }

    if (Job->Allocator) {
        Job->Allocator->Vtbl->FreePointer(Job->Allocator, (PVOID *)&GraphWork);
    }
}

_Use_decl_annotations_
static
HRESULT
Chm01AsyncDispatchGraphWork(
    PCHM01_ASYNC_JOB Job
    )
{
    HRESULT Result;
    ULONG Index;
    ULONG ActiveGraphs;
    PGRAPH Graph;
    PGRAPH *Graphs;
    PPERFECT_HASH_CONTEXT Context;
    PCHM01_ASYNC_GRAPH_WORK GraphWork;

    Context = Job->Context;
    Graphs = Job->Graphs;
    ActiveGraphs = 0;

    if (Job->GraphsCompleteEvent) {
        ResetEvent(Job->GraphsCompleteEvent);
    }

    ASSERT(Context->MainWorkList->Vtbl->IsEmpty(Context->MainWorkList));
    ASSERT(Context->FinishedWorkList->Vtbl->IsEmpty(Context->FinishedWorkList));

    if (FirstSolvedGraphWins(Context)) {
        ASSERT(Job->NumberOfGraphs == Job->Concurrency);
    } else {
        ASSERT(Job->NumberOfGraphs - 1 == Job->Concurrency);
    }

    for (Index = 0; Index < Job->NumberOfGraphs; Index++) {

        Graph = Graphs[Index];

        ResetSRWLock(&Graph->Lock);
        InitializeListHead(&Graph->ListEntry);

        AcquireGraphLockExclusive(Graph);
        Result = Graph->Vtbl->SetInfo(Graph, &Job->GraphInfo);
        ReleaseGraphLockExclusive(Graph);

        if (FAILED(Result)) {
            PH_ERROR(GraphSetInfo, Result);
            return Result;
        }

        Graph->Flags.IsInfoLoaded = FALSE;

        if (!FirstSolvedGraphWins(Context) && Index == 0) {
            Graph->Flags.IsSpare = TRUE;

            _Benign_race_begin_
            Context->SpareGraph = Graph;
            _Benign_race_end_

            continue;
        }

        Graph->Flags.IsSpare = FALSE;

        GraphWork = (PCHM01_ASYNC_GRAPH_WORK)(
            Job->Allocator->Vtbl->Calloc(
                Job->Allocator,
                1,
                sizeof(*GraphWork)
            )
        );

        if (!GraphWork) {
            return E_OUTOFMEMORY;
        }

        GraphWork->Job = Job;
        GraphWork->Graph = Graph;
        GraphWork->Work.Step = Chm01AsyncGraphStep;
        GraphWork->Work.Complete = Chm01AsyncGraphComplete;
        GraphWork->Work.SliceBudget = Job->SliceBudget;

        ActiveGraphs++;

        Result = PerfectHashAsyncSubmit(&Job->Async, &GraphWork->Work);
        if (FAILED(Result)) {
            return Result;
        }
    }

    Job->ActiveGraphs = (LONG)ActiveGraphs;

    if (ActiveGraphs == 0 && Job->GraphsCompleteEvent) {
        SetEvent(Job->GraphsCompleteEvent);
    }

    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
Chm01AsyncInitializeJob(
    PCHM01_ASYNC_JOB Job
    )
{
    USHORT Index;
    HRESULT Result;
    BOOLEAN LimitConcurrency;
    ULONG Concurrency;
    ULONG NumberOfGraphs;
    ULONG NumberOfSeedsRequired;
    ULONG NumberOfSeedsAvailable;
    PGRAPH Graph;
    PGRAPH *Graphs;
    PALLOCATOR Allocator;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TLS_CONTEXT TlsContext;
    PERFECT_HASH_TLS_CONTEXT LocalTlsContext;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    GRAPH_INFO_ON_DISK *GraphInfoOnDisk;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PHANDLE Event;
    ULONG NumberOfEvents;

    Table = Job->Table;
    Context = Table->Context;
    Allocator = Table->Allocator;
    TableCreateFlags.AsULongLong = Table->TableCreateFlags.AsULongLong;

    //
    // Initialize event arrays.
    //

    Job->Events[Chm01AsyncJobEventSucceeded] = Context->SucceededEvent;
    Job->Events[Chm01AsyncJobEventCompleted] = Context->CompletedEvent;
    Job->Events[Chm01AsyncJobEventShutdown] = Context->ShutdownEvent;
    Job->Events[Chm01AsyncJobEventFailed] = Context->FailedEvent;
    Job->Events[Chm01AsyncJobEventLowMemory] = Context->LowMemoryEvent;

    {
        PHANDLE SaveEvent = Job->SaveEvents;
        PHANDLE PrepareEvent = Job->PrepareEvents;

#define EXPAND_AS_ASSIGN_EVENT(                     \
    Verb, VUpper, Name, Upper,                      \
    EofType, EofValue,                              \
    Suffix, Extension, Stream, Base                 \
)                                                   \
    *Verb##Event++ = Context->Verb##d##Name##Event;

        PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_EVENT);
        SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_EVENT);

#undef EXPAND_AS_ASSIGN_EVENT
    }

    //
    // Set up on-disk info buffers.
    //

    GraphInfoOnDisk = &Job->GraphInfoOnDisk;
    TableInfoOnDisk = &GraphInfoOnDisk->TableInfoOnDisk;

    Job->TableInfoOnDisk = TableInfoOnDisk;
    Context->GraphInfoOnDisk = GraphInfoOnDisk;
    Table->TableInfoOnDisk = TableInfoOnDisk;

    //
    // Verify we have sufficient seeds available for the hash function.
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
    // Determine concurrency.
    //

    Concurrency = Context->MaximumConcurrency;

    LimitConcurrency = (
        Table->PriorPredictedAttempts > 0 &&
        TableCreateFlags.TryUsePredictedAttemptsToLimitMaxConcurrency != FALSE
    );

    if (LimitConcurrency) {
        Concurrency = min(Concurrency, Table->PriorPredictedAttempts);
    }

    Job->Concurrency = Concurrency;

    if (FirstSolvedGraphWins(Context)) {
        NumberOfGraphs = Concurrency;
    } else {
        NumberOfGraphs = Concurrency + 1;
        if (NumberOfGraphs == 0) {
            return E_INVALIDARG;
        }
    }

    Job->NumberOfGraphs = NumberOfGraphs;

    Context->InitialResizes = 0;
    Context->NumberOfTableResizeEvents = 0;

    Result = PrepareGraphInfoChm01(Table, &Job->GraphInfo, NULL);
    if (FAILED(Result)) {
        PH_ERROR(CreatePerfectHashTableImplChm01_PrepareFirstGraphInfo, Result);
        goto Error;
    }

    //
    // Allocate graph instances.
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

    Job->Graphs = Graphs;

    //
    // Use per-graph allocator instances (TLS-toggled).
    //

    TlsContext = PerfectHashTlsGetOrSetContext(&LocalTlsContext);

    TlsContextDisableGlobalAllocator(TlsContext);
    TlsContext->Flags.CustomAllocatorDetailsPresent = TRUE;
    TlsContext->HeapCreateFlags = HEAP_NO_SERIALIZE;
    TlsContext->HeapMinimumSize = (ULONG_PTR)Job->GraphInfo.AllocSize;

#if defined(_M_IX86)
    if ((ULONGLONG)TlsContext->HeapMinimumSize != Job->GraphInfo.AllocSize) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }
#endif

    TlsContext->TableCreateFlags.AsULongLong = TableCreateFlags.AsULongLong;
    TlsContext->Table = Table;

    for (Index = 0; Index < NumberOfGraphs; Index++) {

        Result = Table->Vtbl->CreateInstance(Table,
                                             NULL,
                                             &IID_PERFECT_HASH_GRAPH,
                                             (PVOID *)&Graph);

        if (FAILED(Result)) {
            if (Result != E_OUTOFMEMORY) {
                PH_ERROR(CreatePerfectHashTableImplChm01_CreateGraph, Result);
            }
            break;
        }

#ifdef PH_WINDOWS
        ASSERT(Graph->Allocator != Table->Allocator);
        ASSERT(Graph->Allocator->HeapHandle != Table->Allocator->HeapHandle);
#endif

        ASSERT(Graph->Rtl == Table->Rtl);

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
    // Reset all context events.
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

    Context->LowMemoryObserved = 0;
    Context->State.AllGraphsFailedMemoryAllocation = FALSE;
    Context->State.SolveTimeoutExpired = FALSE;
    Context->State.FixedAttemptsReached = FALSE;
    Context->State.MaxAttemptsReached = FALSE;
    ClearStopSolving(Context);

    Job->Attempt = 1;

    //
    // Set callbacks and algorithm context.
    //

    Context->MainWorkCallback = ProcessGraphCallbackChm01;
    Context->ConsoleWorkCallback = ProcessConsoleCallbackChm01;
    Context->AlgorithmContext = &Job->GraphInfo;
    Context->FileWorkCallback = FileWorkCallbackChm01;

    //
    // Prepare output directory if required.
    //

    if (!NoFileIo(Table)) {
        Result = PrepareTableOutputDirectory(Table);
        if (FAILED(Result)) {
            PH_ERROR(PrepareTableOutputDirectory, Result);
            goto Error;
        }
    }

    //
    // Configure solve timeout if requested.
    //

    if (Table->MaxSolveTimeInSeconds > 0 && Context->SolveTimeout) {
        SetThreadpoolTimer(Context->SolveTimeout,
                           &Table->RelativeMaxSolveTimeInFiletime.AsFileTime,
                           0,
                           0);
    }

    //
    // Submit prepare file work.
    //

    ASSERT(Context->FileWorkList->Vtbl->IsEmpty(Context->FileWorkList));

    if (!NoFileIo(Table)) {

#define EXPAND_AS_SUBMIT_FILE_WORK(                     \
    Verb, VUpper, Name, Upper,                          \
    EofType, EofValue,                                  \
    Suffix, Extension, Stream, Base                     \
)                                                       \
    ZeroStructInline(Job->Verb##Name);                  \
    Job->Verb##Name.FileWorkId = FileWork##Verb##Name##Id;   \
    if (Chm01AsyncShouldSkipContextFileWork(                 \
            Context,                                        \
            Job->Verb##Name.FileWorkId)) {                  \
        if (!SetEvent(Context->Verb##d##Name##Event)) {      \
            SYS_ERROR(SetEvent);                             \
            Result = PH_E_SYSTEM_CALL_FAILED;               \
            goto Error;                                     \
        }                                                   \
    } else {                                                \
        InsertTailFileWork(Context, &Job->Verb##Name.ListEntry); \
        PerfectHashContextSubmitFileWork(Context);          \
    }

#define SUBMIT_PREPARE_FILE_WORK() \
    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_FILE_WORK)

        SUBMIT_PREPARE_FILE_WORK();

#undef SUBMIT_PREPARE_FILE_WORK
#undef EXPAND_AS_SUBMIT_FILE_WORK
    }

    //
    // Capture initial timing.
    //

    QueryPerformanceFrequency(&Context->Frequency);
    CONTEXT_START_TIMERS(Solve);
    Context->StartMilliseconds = GetTickCount64();

    //
    // Reset counters.
    //

    Context->GraphMemoryFailures = Concurrency;
    Context->RemainingSolverLoops = Concurrency;
    Context->ActiveSolvingLoops = 0;
    Context->Attempts = 0;
    Context->FailedAttempts = 0;
    Context->FinishedCount = 0;
    Context->HighestDeletedEdgesCount = 0;

    //
    // Dispatch graph work.
    //

    Result = Chm01AsyncDispatchGraphWork(Job);
    if (FAILED(Result)) {
        goto Error;
    }

    return S_OK;

Error:

    if (Result == E_OUTOFMEMORY) {
        Result = PH_I_OUT_OF_MEMORY;
    }

    if (!NoFileIo(Table)) {
        PerfectHashContextWaitForFileWorkCallbacks(Context, FALSE);
    }

    if (Result != S_OK) {
        SetEvent(Context->ShutdownEvent);
    }

    return Result;
}

_Use_decl_annotations_
static
HRESULT
Chm01AsyncPollSolveEvents(
    PCHM01_ASYNC_JOB Job
    )
{
    ULONG WaitResult;
    PPERFECT_HASH_CONTEXT Context;

    Context = Job->Context;

    WaitResult = WaitForMultipleObjects(ARRAYSIZE(Job->Events),
                                        Job->Events,
                                        FALSE,
                                        0);

    if (WaitResult == WAIT_TIMEOUT) {
        return S_FALSE;
    }

    SetStopSolving(Context);

    if (CtrlCPressed) {
        return PH_E_CTRL_C_PRESSED;
    }

    if (WaitResult == WAIT_OBJECT_0 + 4) {
        InterlockedIncrement(&Context->LowMemoryObserved);
        return PH_I_LOW_MEMORY;
    }

    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
Chm01AsyncFinalizeVerify(
    PCHM01_ASYNC_JOB Job
    )
{
    PRTL Rtl;
    PRNG Rng;
    HRESULT Result;
    PGRAPH Graph;
    ULONG WaitResult;
    PLIST_ENTRY ListEntry;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_TABLE Table;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    BOOLEAN FailedEventSet;
    BOOLEAN ShutdownEventSet;
    BOOLEAN LowMemoryEventSet = FALSE;

    Context = Job->Context;
    Table = Job->Table;
    Rtl = Table->Rtl;
    TableCreateFlags.AsULongLong = Table->TableCreateFlags.AsULongLong;

    if (Job->GraphsCompleteEvent) {
        WaitResult = WaitForSingleObject(Job->GraphsCompleteEvent, 0);
        if (WaitResult != WAIT_OBJECT_0) {
            return S_FALSE;
        }
    }

    if (CtrlCPressed) {
        return PH_E_CTRL_C_PRESSED;
    }

    if (Context->FinishedCount == 0) {

        WaitResult = WaitForSingleObject(Context->FailedEvent, 0);
        FailedEventSet = (WaitResult == WAIT_OBJECT_0);

        WaitResult = WaitForSingleObject(Context->ShutdownEvent, 0);
        ShutdownEventSet = (WaitResult == WAIT_OBJECT_0);

        if ((!FailedEventSet && !ShutdownEventSet) ||
            Context->LowMemoryObserved > 0) {
            LowMemoryEventSet = TRUE;
        }

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
        } else {
            Result = PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION;
        }

        return Result;
    }

    ASSERT(Context->FinishedCount > 0);

    if (!TableCreateFlags.Quiet) {
        Result = PrintCurrentContextStatsChm01(Context);
        if (FAILED(Result)) {
            PH_ERROR(PrintCurrentContextStatsChm01, Result);
            return Result;
        }
    }

    if (FirstSolvedGraphWins(Context)) {
        ListEntry = NULL;
        if (!RemoveHeadFinishedWork(Context, &ListEntry)) {
            Result = PH_E_GUARDED_LIST_EMPTY;
            PH_ERROR(PerfectHashCreateChm01Callback_RemoveFinishedWork, Result);
            return Result;
        }

        Graph = CONTAINING_RECORD(ListEntry, GRAPH, ListEntry);

    } else {

        EnterCriticalSection(&Context->BestGraphCriticalSection);
        Graph = Context->BestGraph;
        Context->BestGraph = NULL;
        LeaveCriticalSection(&Context->BestGraphCriticalSection);

        ASSERT(Graph != NULL);
    }

    Table->MaximumGraphTraversalDepth = Graph->MaximumTraversalDepth;
    Table->NumberOfEmptyVertices = Graph->NumberOfEmptyVertices;
    Table->NumberOfCollisionsDuringAssignment = Graph->Collisions;

    Table->Flags.VertexPairsArrayUsesLargePages = (
        Graph->Flags.VertexPairsArrayUsesLargePages
    );

    Table->Flags.UsedAvx2HashFunction = Graph->Flags.UsedAvx2HashFunction;
    Table->Flags.UsedAvx512HashFunction = Graph->Flags.UsedAvx512HashFunction;
    Table->Flags.UsedAvx2MemoryCoverageFunction =
        Graph->Flags.UsedAvx2MemoryCoverageFunction;

    COPY_GRAPH_COUNTERS_FROM_GRAPH_TO_TABLE();

    if (Context->RngId != PerfectHashRngSystemId) {
        Table->RngSeed = Graph->Rng->Seed;
        Table->RngSubsequence = Graph->Rng->Subsequence;
        Table->RngOffset = Graph->Rng->Offset;

        Rng = Graph->Rng;
        Result = Rng->Vtbl->GetCurrentOffset(Rng, &Table->RngCurrentOffset);
        if (FAILED(Result)) {
            PH_ERROR(CreatePerfectHashTableImplChm01_RngGetCurrentOffset,
                     Result);
            return Result;
        }
    }

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
        return Result;
    }

    Context->SolvedContext = Graph;

    if (!NoFileIo(Table)) {

#define EXPAND_AS_SUBMIT_FILE_WORK(                     \
    Verb, VUpper, Name, Upper,                          \
    EofType, EofValue,                                  \
    Suffix, Extension, Stream, Base                     \
)                                                       \
    ZeroStructInline(Job->Verb##Name);                  \
    Job->Verb##Name.FileWorkId = FileWork##Verb##Name##Id;   \
    if (Chm01AsyncShouldSkipContextFileWork(                 \
            Context,                                        \
            Job->Verb##Name.FileWorkId)) {                  \
        if (!SetEvent(Context->Verb##d##Name##Event)) {      \
            SYS_ERROR(SetEvent);                             \
            return PH_E_SYSTEM_CALL_FAILED;                 \
        }                                                   \
    } else {                                                \
        InsertTailFileWork(Context, &Job->Verb##Name.ListEntry); \
        PerfectHashContextSubmitFileWork(Context);          \
    }

#define SUBMIT_SAVE_FILE_WORK() \
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_FILE_WORK)

        SUBMIT_SAVE_FILE_WORK();

#undef SUBMIT_SAVE_FILE_WORK
#undef EXPAND_AS_SUBMIT_FILE_WORK
        Job->Flags |= CHM01_ASYNC_JOB_FLAG_SAVE_SUBMITTED;

    } else {

        PGRAPH_INFO_ON_DISK NewGraphInfoOnDisk;
        PGRAPH_INFO_ON_DISK ExistingGraphInfoOnDisk;

        ExistingGraphInfoOnDisk = Context->GraphInfoOnDisk;

        NewGraphInfoOnDisk = (
            Job->Allocator->Vtbl->Calloc(
                Job->Allocator,
                1,
                sizeof(*NewGraphInfoOnDisk)
            )
        );

        if (!NewGraphInfoOnDisk) {
            return E_OUTOFMEMORY;
        }

        CopyMemory(NewGraphInfoOnDisk,
                   ExistingGraphInfoOnDisk,
                   sizeof(*NewGraphInfoOnDisk));

        if (Graph->FirstSeed == 0) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(CreatePerfectHashTableImplChm01_GraphFirstSeedIs0, Result);
            PH_RAISE(Result);
        }

        CopyMemory(&NewGraphInfoOnDisk->TableInfoOnDisk.FirstSeed,
                   &Graph->FirstSeed,
                   Graph->NumberOfSeeds * sizeof(Graph->FirstSeed));

        Table->TableInfoOnDisk = &NewGraphInfoOnDisk->TableInfoOnDisk;
        Table->State.TableInfoOnDiskWasHeapAllocated = TRUE;

        if (!IsTableCreateOnly(Table)) {
            PVOID BaseAddress;
            LONGLONG SizeInBytes;
            BOOLEAN LargePagesForTableData;
            PRTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC TryLargePageVirtualAlloc;

            LargePagesForTableData = (
                Table->TableCreateFlags.TryLargePagesForTableData == TRUE
            );

            SizeInBytes = (
                Job->TableInfoOnDisk->NumberOfTableElements.QuadPart *
                Job->TableInfoOnDisk->AssignedElementSizeInBytes
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
                return E_OUTOFMEMORY;
            }

            Table->State.TableDataWasHeapAllocated = TRUE;
            Table->Flags.TableDataUsesLargePages = LargePagesForTableData;

            CopyMemory(Table->TableDataBaseAddress,
                       Graph->Assigned,
                       SizeInBytes);
        }
    }

    CONTEXT_START_TIMERS(Verify);

    Result = Graph->Vtbl->Verify(Graph);

    CONTEXT_END_TIMERS(Verify);

    if (!SetEvent(Context->VerifiedTableEvent)) {
        SYS_ERROR(SetEvent);
        return PH_E_SYSTEM_CALL_FAILED;
    }

    if (FAILED(Result)) {
        return PH_E_TABLE_VERIFICATION_FAILED;
    }

    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
Chm01AsyncFinalizeWaitSave(
    PCHM01_ASYNC_JOB Job
    )
{
    HRESULT Result;
    ULONG WaitResult;

    WaitResult = WaitForMultipleObjects(ARRAYSIZE(Job->SaveEvents),
                                        Job->SaveEvents,
                                        TRUE,
                                        0);

    if (WaitResult == WAIT_TIMEOUT) {
        return S_FALSE;
    }

    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForMultipleObjects);
        return PH_E_SYSTEM_CALL_FAILED;
    }

    Result = S_OK;

#undef EXPAND_AS_CHECK_ERRORS
#define EXPAND_AS_CHECK_ERRORS(                                      \
    Verb, VUpper, Name, Upper,                                       \
    EofType, EofValue,                                               \
    Suffix, Extension, Stream, Base                                  \
)                                                                    \
    if (Job->Verb##Name.NumberOfErrors > 0) {                        \
        Result = Job->Verb##Name.LastResult;                         \
        if (Result == S_OK || Result == E_UNEXPECTED) {              \
            Result = PH_E_ERROR_DURING_##VUpper##_##Upper;           \
        }                                                            \
        PH_ERROR(                                                    \
            CreatePerfectHashTableImplChm01_ErrorDuring##Verb##Name, \
            Result                                                   \
        );                                                           \
        goto Error;                                                  \
    }

    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CHECK_ERRORS)

#undef EXPAND_AS_CHECK_ERRORS

    return S_OK;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    return Result;
}

_Use_decl_annotations_
static
HRESULT
Chm01AsyncFinalizeClose(
    PCHM01_ASYNC_JOB Job
    )
{
    HRESULT Result;
    HRESULT CloseResult;
    ULONG CloseFileErrorCount = 0;
    ULONG WaitResult;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_TABLE Table;
    PLARGE_INTEGER EndOfFile = NULL;
    LARGE_INTEGER EmptyEndOfFile = { 0 };

    Context = Job->Context;
    Table = Job->Table;
    Result = S_OK;
    CloseResult = S_OK;

    if (Job->Attempt == 0 || NoFileIo(Table)) {
        return S_OK;
    }

    if (Job->GraphsCompleteEvent) {
        WaitResult = WaitForSingleObject(Job->GraphsCompleteEvent, 0);
        if (WaitResult != WAIT_OBJECT_0) {
            return S_FALSE;
        }
    }

    if (Job->Flags & CHM01_ASYNC_JOB_FLAG_SAVE_SUBMITTED) {
        WaitResult = WaitForMultipleObjects(ARRAYSIZE(Job->SaveEvents),
                                            Job->SaveEvents,
                                            TRUE,
                                            0);
        if (WaitResult == WAIT_TIMEOUT) {
            return S_FALSE;
        }
        if (WaitResult != WAIT_OBJECT_0) {
            SYS_ERROR(WaitForMultipleObjects);
            return PH_E_SYSTEM_CALL_FAILED;
        }
    }

    if (!NoFileIo(Table)) {
        WaitResult = WaitForMultipleObjects(ARRAYSIZE(Job->PrepareEvents),
                                            Job->PrepareEvents,
                                            TRUE,
                                            0);
        if (WaitResult == WAIT_TIMEOUT) {
            return S_FALSE;
        }
        if (WaitResult != WAIT_OBJECT_0) {
            SYS_ERROR(WaitForMultipleObjects);
            return PH_E_SYSTEM_CALL_FAILED;
        }
    }

    if (Job->Flags & CHM01_ASYNC_JOB_FLAG_CLOSE_DELETES_FILES) {
        EndOfFile = &EmptyEndOfFile;
    }

    if ((Job->Flags & CHM01_ASYNC_JOB_FLAG_CLOSE_SUBMITTED) == 0) {

#define EXPAND_AS_SUBMIT_CLOSE_FILE_WORK(               \
    Verb, VUpper, Name, Upper,                          \
    EofType, EofValue,                                  \
    Suffix, Extension, Stream, Base                     \
)                                                       \
    ZeroStructInline(Job->Verb##Name);                  \
    Job->Verb##Name.FileWorkId = FileWork##Verb##Name##Id;   \
    Job->Verb##Name.EndOfFile = EndOfFile;              \
    if (!Chm01AsyncShouldSkipContextFileWork(           \
            Context,                                    \
            Job->Verb##Name.FileWorkId)) {              \
        InsertTailFileWork(Context, &Job->Verb##Name.ListEntry); \
        PerfectHashContextSubmitFileWork(Context);      \
    }

#define SUBMIT_CLOSE_FILE_WORK() \
    CLOSE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_SUBMIT_CLOSE_FILE_WORK)

        SUBMIT_CLOSE_FILE_WORK();

#undef SUBMIT_CLOSE_FILE_WORK
#undef EXPAND_AS_SUBMIT_CLOSE_FILE_WORK

        Job->Flags |= CHM01_ASYNC_JOB_FLAG_CLOSE_SUBMITTED;
    }

    if (Context->FileWorkOutstandingEvent) {
        WaitResult = WaitForSingleObject(Context->FileWorkOutstandingEvent, 0);
        if (WaitResult != WAIT_OBJECT_0) {
            return S_FALSE;
        }
    }

#define EXPAND_AS_CHECK_CLOSE_ERRORS(                                \
    Verb, VUpper, Name, Upper,                                       \
    EofType, EofValue,                                               \
    Suffix, Extension, Stream, Base                                  \
)                                                                    \
    if (Job->Verb##Name.NumberOfErrors > 0) {                        \
        CloseResult = Job->Verb##Name.LastResult;                    \
        if (CloseResult == S_OK || CloseResult == E_UNEXPECTED) {    \
            CloseResult = PH_E_ERROR_DURING_##VUpper##_##Upper;      \
        }                                                            \
        PH_ERROR(                                                    \
            CreatePerfectHashTableImplChm01_ErrorDuring##Verb##Name, \
            Result                                                   \
        );                                                           \
        CloseFileErrorCount++;                                       \
    }

    CLOSE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CHECK_CLOSE_ERRORS)

#undef EXPAND_AS_CHECK_CLOSE_ERRORS

    if (CloseFileErrorCount > 0) {
        Result = CloseResult;
    }

    return Result;
}

_Use_decl_annotations_
static
HRESULT
Chm01AsyncReleaseGraphs(
    PCHM01_ASYNC_JOB Job
    )
{
    USHORT Index;
    ULONG ReferenceCount;
    PGRAPH Graph;
    PGRAPH *Graphs;
    PALLOCATOR Allocator;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_TABLE Table;
    PHANDLE Event;
    ULONG NumberOfEvents;
    HRESULT Result;

    Result = Job->LastResult;
    Context = Job->Context;
    Table = Job->Table;
    Allocator = Job->Allocator;
    Graphs = Job->Graphs;

    if (Result == E_OUTOFMEMORY) {
        Result = PH_I_OUT_OF_MEMORY;
    }

    if (Graphs) {
        for (Index = 0; Index < Job->NumberOfGraphs; Index++) {

            Graph = Graphs[Index];
            if (!Graph) {
                continue;
            }

            ReferenceCount = Graph->Vtbl->Release(Graph);

            if (ReferenceCount != 0) {
                Result = PH_E_INVARIANT_CHECK_FAILED;
                PH_ERROR(GraphReferenceCountNotZero, Result);
                PH_RAISE(Result);
            }

            Graphs[Index] = NULL;
        }

        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&Graphs);
        Job->Graphs = NULL;
    }

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

    RELEASE(Table->OutputPath);

    Job->LastResult = Result;

    return Result;
}

_Use_decl_annotations_
static
HRESULT
Chm01AsyncStep(
    PPERFECT_HASH_ASYNC_WORK Work
    )
{
    HRESULT Result;
    PCHM01_ASYNC_JOB Job;

    Job = CONTAINING_RECORD(Work, CHM01_ASYNC_JOB, Work);

    switch (Job->State) {

        case Chm01AsyncStateInitialize:
            Result = Chm01AsyncInitializeJob(Job);
            if (FAILED(Result)) {
                Job->LastResult = Result;
                Job->Flags |= CHM01_ASYNC_JOB_FLAG_CLOSE_DELETES_FILES;
                Job->State = Chm01AsyncStateFinalizeClose;
                return S_FALSE;
            }
            Job->State = Chm01AsyncStateSolveGraphs;
            return S_FALSE;

        case Chm01AsyncStateSolveGraphs:
            Result = Chm01AsyncPollSolveEvents(Job);
            if (Result == S_FALSE) {
                return S_FALSE;
            }
            if (FAILED(Result)) {
                Job->LastResult = Result;
                Job->Flags |= CHM01_ASYNC_JOB_FLAG_CLOSE_DELETES_FILES;
                Job->State = Chm01AsyncStateFinalizeClose;
                return S_FALSE;
            }
            Job->State = Chm01AsyncStateFinalizeVerify;
            return S_FALSE;

        case Chm01AsyncStateFinalizeVerify:
            Result = Chm01AsyncFinalizeVerify(Job);
            if (Result == S_FALSE) {
                return S_FALSE;
            }
            if (FAILED(Result)) {
                Job->LastResult = Result;
                Job->Flags |= CHM01_ASYNC_JOB_FLAG_CLOSE_DELETES_FILES;
                Job->State = Chm01AsyncStateFinalizeClose;
                return S_FALSE;
            }
            Job->LastResult = Result;
            if (NoFileIo(Job->Table)) {
                Job->State = Chm01AsyncStateReleaseGraphs;
            } else {
                Job->State = Chm01AsyncStateFinalizeWaitSave;
            }
            return S_FALSE;

        case Chm01AsyncStateFinalizeWaitSave:
            Result = Chm01AsyncFinalizeWaitSave(Job);
            if (Result == S_FALSE) {
                return S_FALSE;
            }
            if (FAILED(Result)) {
                Job->LastResult = Result;
                Job->Flags |= CHM01_ASYNC_JOB_FLAG_CLOSE_DELETES_FILES;
                Job->State = Chm01AsyncStateFinalizeClose;
                return S_FALSE;
            }
            Job->LastResult = Result;
            Job->State = Chm01AsyncStateFinalizeClose;
            return S_FALSE;

        case Chm01AsyncStateFinalizeClose:
            Result = Chm01AsyncFinalizeClose(Job);
            if (Result == S_FALSE) {
                return S_FALSE;
            }
            if (FAILED(Result) && Job->LastResult == S_OK) {
                Job->LastResult = Result;
            }
            Job->State = Chm01AsyncStateReleaseGraphs;
            return S_FALSE;

        case Chm01AsyncStateReleaseGraphs:
            Result = Chm01AsyncReleaseGraphs(Job);
            Job->State = Chm01AsyncStateComplete;
            return Result;

        case Chm01AsyncStateComplete:
            return Job->LastResult;

        case Chm01AsyncStateError:
        default:
            Job->LastResult = PH_E_INVARIANT_CHECK_FAILED;
            return Job->LastResult;
    }
}

_Use_decl_annotations_
static
VOID
Chm01AsyncComplete(
    PPERFECT_HASH_ASYNC_WORK Work,
    HRESULT Result
    )
{
    PCHM01_ASYNC_JOB Job;

    Job = CONTAINING_RECORD(Work, CHM01_ASYNC_JOB, Work);
    Job->LastResult = Result;

    Chm01AsyncLogJobState(Job, 1);

    if (Job->CompletionEvent) {
        SetEvent(Job->CompletionEvent);
    }

    if (Job->Async.IoCompletionPort) {
        PostQueuedCompletionStatus(Job->Async.IoCompletionPort,
                                   0,
                                   0,
                                   NULL);
    }
}

_Use_decl_annotations_
HRESULT
Chm01AsyncCreateJob(
    PPERFECT_HASH_TABLE Table,
    HANDLE IoCompletionPort,
    PCHM01_ASYNC_JOB *JobPointer
    )
{
    HRESULT Result;
    PCHM01_ASYNC_JOB Job;
    PPERFECT_HASH_CONTEXT Context;
    PALLOCATOR Allocator;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(JobPointer)) {
        return E_POINTER;
    }

    Context = Table->Context;
    Allocator = Table->Allocator;

    if (!Allocator) {
        return E_UNEXPECTED;
    }

    Job = (PCHM01_ASYNC_JOB)Allocator->Vtbl->Calloc(
        Allocator,
        1,
        sizeof(*Job)
    );

    if (!Job) {
        return E_OUTOFMEMORY;
    }

    Job->Table = Table;
    Job->Context = Context;
    Job->Allocator = Allocator;
    Job->Rtl = Table->Rtl;
    Job->State = Chm01AsyncStateInitialize;
    Job->SliceBudget = 1;
    Job->LastResult = S_OK;

    Job->CompletionEvent = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!Job->CompletionEvent) {
        SYS_ERROR(CreateEventW_Chm01AsyncCompletionEvent);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Job->GraphsCompleteEvent = CreateEventW(NULL, TRUE, TRUE, NULL);
    if (!Job->GraphsCompleteEvent) {
        SYS_ERROR(CreateEventW_Chm01AsyncGraphsCompleteEvent);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Result = PerfectHashAsyncInitialize(&Job->Async, Context, IoCompletionPort);
    if (FAILED(Result)) {
        goto Error;
    }

    Job->Work.Step = Chm01AsyncStep;
    Job->Work.Complete = Chm01AsyncComplete;
    Job->Work.SliceBudget = Job->SliceBudget;

    if (!Context->FileWorkIoCompletionPort) {
        Context->FileWorkIoCompletionPort = IoCompletionPort;
    }

    *JobPointer = Job;
    return S_OK;

Error:

    if (Job) {
        if (Job->CompletionEvent) {
            CloseHandle(Job->CompletionEvent);
            Job->CompletionEvent = NULL;
        }
        if (Job->GraphsCompleteEvent) {
            CloseHandle(Job->GraphsCompleteEvent);
            Job->GraphsCompleteEvent = NULL;
        }
        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&Job);
    }

    return Result;
}

_Use_decl_annotations_
HRESULT
Chm01AsyncSubmitJob(
    PCHM01_ASYNC_JOB Job
    )
{
    if (!ARGUMENT_PRESENT(Job)) {
        return E_POINTER;
    }

    return PerfectHashAsyncSubmit(&Job->Async, &Job->Work);
}

#ifdef PH_WINDOWS
static
VOID
Chm01AsyncPumpIoCompletionPort(
    _In_ PCHM01_ASYNC_JOB Job
    )
{
    BOOL Success;
    DWORD NumberOfBytes;
    DWORD LastError;
    ULONG_PTR CompletionKey;
    LPOVERLAPPED Overlapped;
    BOOLEAN LoggedCompletionSignaled = FALSE;
    BOOLEAN LoggedOutstandingZero = FALSE;
    HANDLE IoCompletionPort;
    PPERFECT_HASH_IOCP_WORK WorkItem;
    const ULONG AllowedFlags = (
        PH_IOCP_WORK_FLAG_ASYNC |
        PH_IOCP_WORK_FLAG_FILE_WORK |
        PH_IOCP_WORK_FLAG_PIPE
    );

    IoCompletionPort = Job->Async.IoCompletionPort;
    if (!IoCompletionPort || !Job->CompletionEvent) {
        return;
    }

    for (;;) {
        if (WaitForSingleObject(Job->CompletionEvent, 0) == WAIT_OBJECT_0) {
            if (Job->Async.Outstanding == 0) {
                break;
            }
            if (!LoggedCompletionSignaled) {
                Chm01AsyncLogJobState(Job, 2);
                LoggedCompletionSignaled = TRUE;
            }
        } else if (Job->Async.Outstanding == 0 && !LoggedOutstandingZero) {
            Chm01AsyncLogJobState(Job, 3);
            LoggedOutstandingZero = TRUE;
        }

        Success = GetQueuedCompletionStatus(IoCompletionPort,
                                            &NumberOfBytes,
                                            &CompletionKey,
                                            &Overlapped,
                                            CHM01_ASYNC_IOCP_PUMP_TIMEOUT_MS);

        if (!Success && !Overlapped) {
            LastError = GetLastError();
            if (LastError == WAIT_TIMEOUT) {
                continue;
            }
        }

        if (CompletionKey == PERFECT_HASH_IOCP_SHUTDOWN_KEY) {
            PostQueuedCompletionStatus(IoCompletionPort,
                                       NumberOfBytes,
                                       CompletionKey,
                                       Overlapped);
            continue;
        }

        if (!Overlapped) {
            continue;
        }

        WorkItem = (PPERFECT_HASH_IOCP_WORK)Overlapped;

        if (WorkItem->Signature != PH_IOCP_WORK_SIGNATURE ||
            !WorkItem->CompletionCallback) {
            continue;
        }

        if ((WorkItem->Flags & AllowedFlags) == 0) {
            PostQueuedCompletionStatus(IoCompletionPort,
                                       NumberOfBytes,
                                       CompletionKey,
                                       Overlapped);
            continue;
        }

        WorkItem->CompletionCallback(NULL,
                                     CompletionKey,
                                     Overlapped,
                                     NumberOfBytes,
                                     Success);
    }
}
#endif

_Use_decl_annotations_
VOID
Chm01AsyncWaitJob(
    PCHM01_ASYNC_JOB Job
    )
{
    if (!ARGUMENT_PRESENT(Job)) {
        return;
    }

    if (Job->CompletionEvent) {
#ifdef PH_WINDOWS
        if (Job->Async.IoCompletionPort) {
            Chm01AsyncPumpIoCompletionPort(Job);
            return;
        }
#endif
        WaitForSingleObject(Job->CompletionEvent, INFINITE);
    }
}

_Use_decl_annotations_
VOID
Chm01AsyncDestroyJob(
    PCHM01_ASYNC_JOB *JobPointer
    )
{
    PCHM01_ASYNC_JOB Job;
    PALLOCATOR Allocator;

    if (!ARGUMENT_PRESENT(JobPointer)) {
        return;
    }

    Job = *JobPointer;
    if (!Job) {
        return;
    }

    Allocator = Job->Allocator;

    if (Job->GraphsCompleteEvent) {
        CloseHandle(Job->GraphsCompleteEvent);
        Job->GraphsCompleteEvent = NULL;
    }

    if (Job->CompletionEvent) {
        CloseHandle(Job->CompletionEvent);
        Job->CompletionEvent = NULL;
    }

    PerfectHashAsyncRundown(&Job->Async);

    if (Job->Graphs) {
        Chm01AsyncReleaseGraphs(Job);
    }

    if (Allocator) {
        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)JobPointer);
    } else {
        *JobPointer = NULL;
    }
}

_Use_decl_annotations_
HRESULT
CreatePerfectHashTableImplChm01Async(
    PPERFECT_HASH_TABLE Table,
    HANDLE IoCompletionPort
    )
{
    HRESULT Result;
    PCHM01_ASYNC_JOB Job;

    Result = Chm01AsyncCreateJob(Table, IoCompletionPort, &Job);
    if (FAILED(Result)) {
        return Result;
    }

    Result = Chm01AsyncSubmitJob(Job);
    if (FAILED(Result)) {
        Chm01AsyncDestroyJob(&Job);
        return Result;
    }

    Chm01AsyncWaitJob(Job);
    Result = Job->LastResult;

    Chm01AsyncDestroyJob(&Job);
    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
