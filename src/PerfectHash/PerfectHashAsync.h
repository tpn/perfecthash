/*++

Copyright (c) 2018-2025 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashAsync.h

Abstract:

    This module defines the asynchronous work framework used by the IOCP
    execution model.  Work items are scheduled via PostQueuedCompletionStatus
    and dispatched through per-item completion callbacks, providing a minimal
    continuation-style engine for finer-grained algorithm execution.

--*/

#pragma once

#include "stdafx.h"

typedef struct _PERFECT_HASH_ASYNC_CONTEXT PERFECT_HASH_ASYNC_CONTEXT;
typedef struct _PERFECT_HASH_ASYNC_WORK PERFECT_HASH_ASYNC_WORK;
typedef PERFECT_HASH_ASYNC_CONTEXT *PPERFECT_HASH_ASYNC_CONTEXT;
typedef PERFECT_HASH_ASYNC_WORK *PPERFECT_HASH_ASYNC_WORK;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_ASYNC_STEP_ROUTINE)(
    _Inout_ PPERFECT_HASH_ASYNC_WORK Work
    );
typedef PERFECT_HASH_ASYNC_STEP_ROUTINE *PPERFECT_HASH_ASYNC_STEP_ROUTINE;

typedef
VOID
(NTAPI PERFECT_HASH_ASYNC_COMPLETE_ROUTINE)(
    _Inout_ PPERFECT_HASH_ASYNC_WORK Work,
    _In_ HRESULT Result
    );
typedef PERFECT_HASH_ASYNC_COMPLETE_ROUTINE
      *PPERFECT_HASH_ASYNC_COMPLETE_ROUTINE;

typedef union _PERFECT_HASH_ASYNC_WORK_FLAGS {
    struct {
        ULONG InlineDispatch:1;
        ULONG Unused:31;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_ASYNC_WORK_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_ASYNC_WORK_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_ASYNC_WORK_FLAGS *PPERFECT_HASH_ASYNC_WORK_FLAGS;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4820)
#endif

typedef struct _PERFECT_HASH_ASYNC_CONTEXT {
    PPERFECT_HASH_CONTEXT Context;
    PALLOCATOR Allocator;
    HANDLE IoCompletionPort;
    HANDLE OutstandingEvent;
    volatile LONG Outstanding;
    ULONG Padding;
} PERFECT_HASH_ASYNC_CONTEXT;

typedef struct _PERFECT_HASH_ASYNC_WORK {
    PERFECT_HASH_IOCP_WORK Iocp;
    PPERFECT_HASH_ASYNC_CONTEXT AsyncContext;
    PPERFECT_HASH_ASYNC_STEP_ROUTINE Step;
    PPERFECT_HASH_ASYNC_COMPLETE_ROUTINE Complete;
    PERFECT_HASH_ASYNC_WORK_FLAGS Flags;
    ULONG SliceBudget;
    HRESULT LastResult;
    ULONG Padding;
} PERFECT_HASH_ASYNC_WORK;

#ifdef _MSC_VER
#pragma warning(pop)
#endif

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_ASYNC_INITIALIZE)(
    _In_ PPERFECT_HASH_ASYNC_CONTEXT AsyncContext,
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ HANDLE IoCompletionPort
    );
typedef PERFECT_HASH_ASYNC_INITIALIZE *PPERFECT_HASH_ASYNC_INITIALIZE;

typedef
VOID
(NTAPI PERFECT_HASH_ASYNC_RUNDOWN)(
    _In_ PPERFECT_HASH_ASYNC_CONTEXT AsyncContext
    );
typedef PERFECT_HASH_ASYNC_RUNDOWN *PPERFECT_HASH_ASYNC_RUNDOWN;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_ASYNC_SUBMIT)(
    _In_ PPERFECT_HASH_ASYNC_CONTEXT AsyncContext,
    _In_ PPERFECT_HASH_ASYNC_WORK Work
    );
typedef PERFECT_HASH_ASYNC_SUBMIT *PPERFECT_HASH_ASYNC_SUBMIT;

typedef
VOID
(NTAPI PERFECT_HASH_ASYNC_WAIT)(
    _In_ PPERFECT_HASH_ASYNC_CONTEXT AsyncContext
    );
typedef PERFECT_HASH_ASYNC_WAIT *PPERFECT_HASH_ASYNC_WAIT;

extern PERFECT_HASH_ASYNC_INITIALIZE PerfectHashAsyncInitialize;
extern PERFECT_HASH_ASYNC_RUNDOWN PerfectHashAsyncRundown;
extern PERFECT_HASH_ASYNC_SUBMIT PerfectHashAsyncSubmit;
extern PERFECT_HASH_ASYNC_WAIT PerfectHashAsyncWait;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
