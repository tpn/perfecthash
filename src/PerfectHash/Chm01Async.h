/*++

Copyright (c) 2018-2026 Trent Nelson <trent@trent.me>

Module Name:

    Chm01Async.h

Abstract:

    This module defines the CHM01 asynchronous table creation state machine.
    It decomposes the CHM01 algorithm into small, IOCP-driven work steps.

--*/

#pragma once

#include "PerfectHashAsync.h"

typedef enum _CHM01_ASYNC_STATE {
    Chm01AsyncStateInitialize = 0,
    Chm01AsyncStateSolveGraphs,
    Chm01AsyncStateFinalizeVerify,
    Chm01AsyncStateFinalizeWaitSave,
    Chm01AsyncStateFinalizeClose,
    Chm01AsyncStateReleaseGraphs,
    Chm01AsyncStateComplete,
    Chm01AsyncStateError
} CHM01_ASYNC_STATE;

typedef enum _CHM01_ASYNC_JOB_EVENT_ID {
    Chm01AsyncJobEventSucceeded = 0,
    Chm01AsyncJobEventCompleted,
    Chm01AsyncJobEventShutdown,
    Chm01AsyncJobEventFailed,
    Chm01AsyncJobEventLowMemory,
    Chm01AsyncJobEventInvalid
} CHM01_ASYNC_JOB_EVENT_ID;

#define NUMBER_OF_CHM01_ASYNC_JOB_EVENTS (Chm01AsyncJobEventInvalid)

typedef struct _CHM01_ASYNC_GRAPH_WORK CHM01_ASYNC_GRAPH_WORK;
typedef CHM01_ASYNC_GRAPH_WORK *PCHM01_ASYNC_GRAPH_WORK;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4820)
#endif

typedef struct _CHM01_ASYNC_JOB {
    PERFECT_HASH_ASYNC_CONTEXT Async;
    PERFECT_HASH_ASYNC_WORK Work;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_CONTEXT Context;
    PALLOCATOR Allocator;
    PRTL Rtl;
    PGRAPH *Graphs;
    ULONG NumberOfGraphs;
    ULONG Concurrency;
    ULONG Attempt;
    ULONG Padding1;
    GRAPH_INFO GraphInfo;
    GRAPH_INFO PrevGraphInfo;
    GRAPH_INFO_ON_DISK GraphInfoOnDisk;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    HANDLE Events[NUMBER_OF_CHM01_ASYNC_JOB_EVENTS];
    HANDLE SaveEvents[NUMBER_OF_SAVE_FILE_EVENTS];
    HANDLE PrepareEvents[NUMBER_OF_PREPARE_FILE_EVENTS];
    PCHM01_ASYNC_GRAPH_WORK *GraphWorkItems;
    volatile LONG ActiveGraphs;
    ULONG Padding2;
    HANDLE GraphsCompleteEvent;

#define EXPAND_AS_ASYNC_FILE_WORK_ITEM( \
    Verb, VUpper, Name, Upper,          \
    EofType, EofValue,                  \
    Suffix, Extension, Stream, Base     \
)                                       \
    FILE_WORK_ITEM Verb##Name;

    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASYNC_FILE_WORK_ITEM)
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASYNC_FILE_WORK_ITEM)
    CLOSE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASYNC_FILE_WORK_ITEM)

#undef EXPAND_AS_ASYNC_FILE_WORK_ITEM

    CHM01_ASYNC_STATE State;
    ULONG SliceBudget;
    ULONG Flags;
    HRESULT LastResult;
    HANDLE CompletionEvent;
} CHM01_ASYNC_JOB;
typedef CHM01_ASYNC_JOB *PCHM01_ASYNC_JOB;

#ifdef _MSC_VER
#pragma warning(pop)
#endif

typedef
_Must_inspect_result_
HRESULT
(NTAPI CHM01_ASYNC_CREATE_JOB)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ HANDLE IoCompletionPort,
    _Outptr_ PCHM01_ASYNC_JOB *Job
    );
typedef CHM01_ASYNC_CREATE_JOB *PCHM01_ASYNC_CREATE_JOB;

typedef
_Must_inspect_result_
HRESULT
(NTAPI CHM01_ASYNC_SUBMIT_JOB)(
    _In_ PCHM01_ASYNC_JOB Job
    );
typedef CHM01_ASYNC_SUBMIT_JOB *PCHM01_ASYNC_SUBMIT_JOB;

typedef
VOID
(NTAPI CHM01_ASYNC_WAIT_JOB)(
    _In_ PCHM01_ASYNC_JOB Job
    );
typedef CHM01_ASYNC_WAIT_JOB *PCHM01_ASYNC_WAIT_JOB;

typedef
VOID
(NTAPI CHM01_ASYNC_DESTROY_JOB)(
    _Inout_ PCHM01_ASYNC_JOB *Job
    );
typedef CHM01_ASYNC_DESTROY_JOB *PCHM01_ASYNC_DESTROY_JOB;

extern CHM01_ASYNC_CREATE_JOB Chm01AsyncCreateJob;
extern CHM01_ASYNC_SUBMIT_JOB Chm01AsyncSubmitJob;
extern CHM01_ASYNC_WAIT_JOB Chm01AsyncWaitJob;
extern CHM01_ASYNC_DESTROY_JOB Chm01AsyncDestroyJob;

typedef
_Must_inspect_result_
HRESULT
(NTAPI CHM01_ASYNC_CREATE_TABLE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ HANDLE IoCompletionPort
    );
typedef CHM01_ASYNC_CREATE_TABLE *PCHM01_ASYNC_CREATE_TABLE;

extern CHM01_ASYNC_CREATE_TABLE CreatePerfectHashTableImplChm01Async;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
