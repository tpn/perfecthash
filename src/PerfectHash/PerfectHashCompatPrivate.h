/*++

Copyright (c) 2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashCompatPrivate.h

Abstract:

    Private header file for "compat" (i.e. non-Windows) components.

--*/

#include "stdafx.h"

#include "PerfectHashEventsPrivate.h"

#ifdef PH_WINDOWS
#error This file is not for Windows.
#endif

#ifndef PH_COMPAT
#error PH_COMPAT should be defined for this file.
#endif

//
// Includes.
//

#include <time.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#ifdef PH_LINUX
#include <sys/sysinfo.h>
#endif

//
// OS X doesn't have pthread_barrier_t.
//

#ifdef __APPLE__
#include "pthread_barrier.h"
#endif

#define FREE_PTR(P) \
    if ((P) != NULL && *(P) != NULL) { free(*(P)); *(P) = NULL; }

PSTR
CreateStringFromWide(
    _In_ PCWSTR WideString
    );

extern DWORD LastError;

typedef union _PH_HANDLE {
    int AsFileDescriptor;
    HANDLE AsHandle;
    LARGE_INTEGER AsLargeInteger;
} PH_HANDLE, *PPH_HANDLE;

typedef union _Struct_size_bytes_(sizeof(ULONG)) _PH_EVENT_STATE {
    struct {

        //
        // When set, indicates the event is signaled.
        //

        ULONG Signaled:1;

        //
        // When set, indicates the event is to be manually reset.
        //

        ULONG ManualReset:1;

        //
        // When set, indicates the mutex has been initialized.
        //

        ULONG MutexInitialized:1;

        //
        // When set, indicates the condition has been initialized.
        //

        ULONG ConditionInitialized:1;

        //
        // When set, indicates the name has been allocated.
        //

        ULONG NameAllocated:1;

        //
        // Unused bits.
        //

        ULONG Unused:27;

    };

    LONG AsLong;
    ULONG AsULong;
} PH_EVENT_STATE, *PPH_EVENT_STATE;

typedef struct _PH_EVENT {
    PH_EVENT_STATE State;
    ULONG Padding1;
    PSTR Name;
    pthread_mutex_t Mutex;
    pthread_cond_t Condition;
} PH_EVENT, *PPH_EVENT;

//
// Threadpool.
//

struct _TP_POOL;

typedef struct _TP_CLEANUP_GROUP {
    volatile int32_t Refcount;
    int32_t Released;
    SRWLOCK MemberLock;
    LIST_ENTRY MemberList;
    pthread_barrier_t Barrier;
    SRWLOCK CleanupLock;
    LIST_ENTRY CleanupList;
} TP_CLEANUP_GROUP, *PTP_CLEANUP_GROUP;

struct _TPP_CLEANUP_GROUP_MEMBER;

typedef
VOID
(CALLBACK TPP_CLEANUP_GROUP_MEMBER_CALLBACK)(
    _In_ struct _TPP_CLEANUP_GROUP_MEMBER *Member
    );
typedef TPP_CLEANUP_GROUP_MEMBER_CALLBACK *PTPP_CLEANUP_GROUP_MEMBER_CALLBACK;

typedef struct _TPP_CLEANUP_GROUP_MEMBER_VFUNCS
{
    PTPP_CLEANUP_GROUP_MEMBER_CALLBACK Free;
    PTPP_CLEANUP_GROUP_MEMBER_CALLBACK CallbackEpilog;
    PTPP_CLEANUP_GROUP_MEMBER_CALLBACK StopCallbackGeneration;
    PTPP_CLEANUP_GROUP_MEMBER_CALLBACK CancelPendingCallbacks;
} TPP_CLEANUP_GROUP_MEMBER_VFUNCS, *PTPP_CLEANUP_GROUP_MEMBER_VFUNCS;

typedef union _TPP_CLEANUP_GROUP_MEMBER_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // Unused bits.
        //

        ULONG Unused:32;
    };

    LONG AsLong;
    ULONG AsULong;
} TPP_CLEANUP_GROUP_MEMBER_FLAGS;
C_ASSERT(sizeof(TPP_CLEANUP_GROUP_MEMBER_FLAGS) == sizeof(ULONG));

typedef struct _TPP_CLEANUP_GROUP_MEMBER {
    volatile int32_t Refcount;
    TPP_CLEANUP_GROUP_MEMBER_FLAGS Flags;
    TPP_CLEANUP_GROUP_MEMBER_VFUNCS VFuncs;
    PTP_CLEANUP_GROUP CleanupGroup;
    PTPP_CLEANUP_GROUP_MEMBER_CALLBACK CleanupGroupCancelCallback;
    PTPP_CLEANUP_GROUP_MEMBER_CALLBACK FinalizationCallback;
    LIST_ENTRY CleanupGroupMemberLinks;
    pthread_barrier_t CallbackBarrier;
    union {
        PTP_WORK_CALLBACK WorkCallback;
        PTP_TIMER_CALLBACK TimerCallback;
    };
    PVOID Context;
    PTP_POOL Pool;
    LIST_ENTRY PoolObjectLinks;
} TPP_CLEANUP_GROUP_MEMBER, *PTPP_CLEANUP_GROUP_MEMBER;

typedef struct _TP_CALLBACK_INSTANCE {
    PTP_WORK_CALLBACK WorkCallback;
    PVOID Context;
    PTP_POOL Pool;
    PTP_CLEANUP_GROUP CleanupGroup;
    PTPP_CLEANUP_GROUP_MEMBER CleanupGroupMember;
} TP_CALLBACK_INSTANCE, *PTP_CALLBACK_INSTANCE;

typedef struct _TP_TASK_CALLBACKS
{
    void* ExecuteCallback /* function */;
    void* Unposted /* function */;
} TP_TASK_CALLBACKS, *PTP_TASK_CALLBACKS;

typedef struct _TP_WORK {
    TPP_CLEANUP_GROUP_MEMBER CleanupGroupMember;
    TP_TASK_CALLBACKS Callbacks;
    PH_EVENT Event;
    uint32_t NumaNode;
    uint8_t IdealProcessor;
    uint8_t Padding[3];
    LIST_ENTRY ListEntry;
} TP_WORK, *PTP_WORK;

typedef struct _TP_TIMER {
    TP_WORK Work;
    SRWLOCK Lock;
} TP_TIMER, *PTP_TIMER;

typedef struct _TPP_QUEUE
{
    LIST_ENTRY Queue;
    SRWLOCK Lock;
} TPP_QUEUE, *PTPP_QUEUE;

typedef struct _TP_POOL {
    volatile int32_t Refcount;
    volatile int32_t NumberOfWorkers;
    volatile int32_t ActiveWorkerCount;
    volatile int32_t PendingWorkCount;
    DWORD MaximumThreads;
    DWORD MinimumThreads;
    PTPP_QUEUE TaskQueue[3];
    SRWLOCK Lock;
    LIST_ENTRY PoolObjectList;
    LIST_ENTRY WorkerList;
    SRWLOCK ShutdownLock;
    PPH_EVENT WorkerWaitEvent;
    LIST_ENTRY PoolLinks;
} TP_POOL, *PTP_POOL;

typedef struct _TPP_WORKER {
    pthread_t ThreadId;
    ULONG Padding;
    LIST_ENTRY ListEntry;
    TP_CALLBACK_INSTANCE CallbackInstance;
} TPP_WORKER, *PTPP_WORKER;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
