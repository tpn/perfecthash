/*++

Copyright (c) 2018-2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContextIocp.h

Abstract:

    This is the private header file for the PERFECT_HASH_CONTEXT_IOCP
    component of the perfect hash library.  It defines the structure, and
    function pointer typedefs for private non-vtbl members.

--*/

#pragma once

#include "stdafx.h"

typedef union _PERFECT_HASH_CONTEXT_IOCP_STATE {
    struct {

        ULONG Initialized:1;
        ULONG Running:1;
        ULONG Stopping:1;
        ULONG Stopped:1;

        ULONG Unused:28;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_CONTEXT_IOCP_STATE;
C_ASSERT(sizeof(PERFECT_HASH_CONTEXT_IOCP_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_CONTEXT_IOCP_STATE *PPERFECT_HASH_CONTEXT_IOCP_STATE;

typedef union _PERFECT_HASH_CONTEXT_IOCP_FLAGS {
    struct {

        //
        // When set, indicates an explicit NUMA node mask is in effect.
        //

        ULONG UseNumaNodeMask:1;

        ULONG Unused:31;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_CONTEXT_IOCP_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_CONTEXT_IOCP_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_CONTEXT_IOCP_FLAGS *PPERFECT_HASH_CONTEXT_IOCP_FLAGS;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_IOCP_COMPLETION_CALLBACK)(
    _In_ struct _PERFECT_HASH_CONTEXT_IOCP *ContextIocp,
    _In_ ULONG_PTR CompletionKey,
    _In_ LPOVERLAPPED Overlapped,
    _In_ DWORD NumberOfBytesTransferred,
    _In_ BOOL Success
    );
typedef PERFECT_HASH_IOCP_COMPLETION_CALLBACK
      *PPERFECT_HASH_IOCP_COMPLETION_CALLBACK;

//
// IOCP work item header used to route completions to a per-item callback.
//

#define PERFECT_HASH_IOCP_SHUTDOWN_KEY ((ULONG_PTR)1)

#define PH_IOCP_WORK_SIGNATURE 0x50434F49u // 'IOCP'

#define PH_IOCP_WORK_FLAG_PIPE       0x00000001
#define PH_IOCP_WORK_FLAG_BULK       0x00000002
#define PH_IOCP_WORK_FLAG_ASYNC      0x00000004
#define PH_IOCP_WORK_FLAG_FILE_WORK  0x00000008

typedef struct _PERFECT_HASH_IOCP_WORK {
    OVERLAPPED Overlapped;
    ULONG Signature;
    ULONG Flags;
    PPERFECT_HASH_IOCP_COMPLETION_CALLBACK CompletionCallback;
    PVOID CompletionContext;
} PERFECT_HASH_IOCP_WORK;
typedef PERFECT_HASH_IOCP_WORK *PPERFECT_HASH_IOCP_WORK;

C_ASSERT(FIELD_OFFSET(PERFECT_HASH_IOCP_WORK, Overlapped) == 0);

typedef struct _PERFECT_HASH_IOCP_NODE {
    struct _PERFECT_HASH_CONTEXT_IOCP *ContextIocp;
    ULONG NodeId;
    ULONG ProcessorCount;
    ULONG IocpConcurrency;
    ULONG Padding1;
#ifdef PH_WINDOWS
    GROUP_AFFINITY GroupAffinity;
#else
    ULONGLONG ProcessorMask;
#endif
    HANDLE IoCompletionPort;
    HANDLE *WorkerThreads;
    ULONG WorkerThreadCount;
    ULONG Padding2;
} PERFECT_HASH_IOCP_NODE;
typedef PERFECT_HASH_IOCP_NODE *PPERFECT_HASH_IOCP_NODE;

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_CONTEXT_IOCP {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_CONTEXT_IOCP);

    //
    // Configuration.
    //

    ULONG IocpConcurrency;
    ULONG MaxWorkerThreads;
    ULONG NumaNodeCount;
    ULONG Padding1;
    PERFECT_HASH_NUMA_NODE_MASK NumaNodeMask;
    ULONG Padding2;
    ULONG Padding3;

    //
    // Pointer to the base output directory, if set.
    //

    PPERFECT_HASH_DIRECTORY BaseOutputDirectory;

    //
    // Threading resources (per NUMA node IOCPs, worker threads, etc.).
    //

    PPERFECT_HASH_IOCP_NODE Nodes;
    ULONG NodeCount;
    ULONG TotalWorkerThreadCount;
    ULONG IoCompletionPortCount;
    ULONG Padding4;

    //
    // Completion dispatch callback (owned by higher-level components).
    //

    PPERFECT_HASH_IOCP_COMPLETION_CALLBACK CompletionCallback;
    PVOID CompletionContext;

    //
    // Shutdown and coordination.
    //

    HANDLE ShutdownEvent;
    HANDLE StartedEvent;

    //
    // Backing vtbl.
    //

    PERFECT_HASH_CONTEXT_IOCP_VTBL Interface;

} PERFECT_HASH_CONTEXT_IOCP;
typedef PERFECT_HASH_CONTEXT_IOCP *PPERFECT_HASH_CONTEXT_IOCP;

#define TryAcquirePerfectHashContextIocpLockExclusive(ContextIocp) \
    TryAcquireSRWLockExclusive(&ContextIocp->Lock)

#define ReleasePerfectHashContextIocpLockExclusive(ContextIocp) \
    ReleaseSRWLockExclusive(&ContextIocp->Lock)

//
// Function pointer typedefs for non-vtbl members (if applicable).
//

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_IOCP_INITIALIZE)(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    );
typedef PERFECT_HASH_CONTEXT_IOCP_INITIALIZE
      *PPERFECT_HASH_CONTEXT_IOCP_INITIALIZE;

typedef
VOID
(NTAPI PERFECT_HASH_CONTEXT_IOCP_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    );
typedef PERFECT_HASH_CONTEXT_IOCP_RUNDOWN
      *PPERFECT_HASH_CONTEXT_IOCP_RUNDOWN;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_IOCP_START)(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    );
typedef PERFECT_HASH_CONTEXT_IOCP_START
      *PPERFECT_HASH_CONTEXT_IOCP_START;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_IOCP_STOP)(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp
    );
typedef PERFECT_HASH_CONTEXT_IOCP_STOP
      *PPERFECT_HASH_CONTEXT_IOCP_STOP;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_CONTEXT_IOCP_CREATE_TABLE_CONTEXT)(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    _Outptr_ PPERFECT_HASH_CONTEXT *Context
    );
typedef PERFECT_HASH_CONTEXT_IOCP_CREATE_TABLE_CONTEXT
      *PPERFECT_HASH_CONTEXT_IOCP_CREATE_TABLE_CONTEXT;

extern PERFECT_HASH_CONTEXT_IOCP_INITIALIZE PerfectHashContextIocpInitialize;
extern PERFECT_HASH_CONTEXT_IOCP_RUNDOWN PerfectHashContextIocpRundown;
extern PERFECT_HASH_CONTEXT_IOCP_START PerfectHashContextIocpStart;
extern PERFECT_HASH_CONTEXT_IOCP_STOP PerfectHashContextIocpStop;
extern PERFECT_HASH_CONTEXT_IOCP_CREATE_TABLE_CONTEXT
    PerfectHashContextIocpCreateTableContext;

extern PERFECT_HASH_CONTEXT_IOCP_SET_MAXIMUM_CONCURRENCY
    PerfectHashContextIocpSetMaximumConcurrency;
extern PERFECT_HASH_CONTEXT_IOCP_GET_MAXIMUM_CONCURRENCY
    PerfectHashContextIocpGetMaximumConcurrency;
extern PERFECT_HASH_CONTEXT_IOCP_SET_MAXIMUM_THREADS
    PerfectHashContextIocpSetMaximumThreads;
extern PERFECT_HASH_CONTEXT_IOCP_GET_MAXIMUM_THREADS
    PerfectHashContextIocpGetMaximumThreads;
extern PERFECT_HASH_CONTEXT_IOCP_SET_NUMA_NODE_MASK
    PerfectHashContextIocpSetNumaNodeMask;
extern PERFECT_HASH_CONTEXT_IOCP_GET_NUMA_NODE_MASK
    PerfectHashContextIocpGetNumaNodeMask;
extern PERFECT_HASH_CONTEXT_IOCP_SET_BASE_OUTPUT_DIRECTORY
    PerfectHashContextIocpSetBaseOutputDirectory;
extern PERFECT_HASH_CONTEXT_IOCP_GET_BASE_OUTPUT_DIRECTORY
    PerfectHashContextIocpGetBaseOutputDirectory;
extern PERFECT_HASH_CONTEXT_IOCP_BULK_CREATE
    PerfectHashContextIocpBulkCreate;
extern PERFECT_HASH_CONTEXT_IOCP_BULK_CREATE_ARGVW
    PerfectHashContextIocpBulkCreateArgvW;
extern PERFECT_HASH_CONTEXT_IOCP_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextIocpExtractBulkCreateArgsFromArgvW;
extern PERFECT_HASH_CONTEXT_IOCP_TABLE_CREATE
    PerfectHashContextIocpTableCreate;
extern PERFECT_HASH_CONTEXT_IOCP_TABLE_CREATE_ARGVW
    PerfectHashContextIocpTableCreateArgvW;
extern PERFECT_HASH_CONTEXT_IOCP_EXTRACT_TABLE_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextIocpExtractTableCreateArgsFromArgvW;
extern PERFECT_HASH_CONTEXT_IOCP_TABLE_CREATE_ARGVA
    PerfectHashContextIocpTableCreateArgvA;
extern PERFECT_HASH_CONTEXT_IOCP_BULK_CREATE_ARGVA
    PerfectHashContextIocpBulkCreateArgvA;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
