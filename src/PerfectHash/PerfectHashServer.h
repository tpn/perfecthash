/*++

Copyright (c) 2018-2025 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashServer.h

Abstract:

    This is the private header file for the PERFECT_HASH_SERVER component of
    the perfect hash library.  It defines the structure, and function pointer
    typedefs for private non-vtbl members.

--*/

#pragma once

#include "stdafx.h"

typedef union _PERFECT_HASH_SERVER_STATE {
    struct {

        ULONG Initialized:1;
        ULONG Running:1;
        ULONG Stopping:1;
        ULONG Stopped:1;

        ULONG Unused:28;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_SERVER_STATE;
C_ASSERT(sizeof(PERFECT_HASH_SERVER_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_SERVER_STATE *PPERFECT_HASH_SERVER_STATE;

typedef union _PERFECT_HASH_SERVER_FLAGS {
    struct {

        //
        // When set, indicates the server should only accept local clients.
        //

        ULONG LocalOnly:1;
        ULONG EndpointAllocated:1;

        ULONG Unused:30;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_SERVER_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_SERVER_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_SERVER_FLAGS *PPERFECT_HASH_SERVER_FLAGS;

struct _PERFECT_HASH_SERVER_PIPE;
typedef struct _PERFECT_HASH_SERVER_PIPE *PPERFECT_HASH_SERVER_PIPE;

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_SERVER {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_SERVER);

    ULONG MaximumConcurrency;
    ULONG NumaNodeCount;
    PERFECT_HASH_NUMA_NODE_MASK NumaNodeMask;
    ULONG Padding1;
    ULONG Padding2;

    //
    // The underlying IOCP context used by the server.
    //

    struct _PERFECT_HASH_CONTEXT_IOCP *ContextIocp;

    //
    // Named pipe endpoint (if applicable).
    //

    UNICODE_STRING Endpoint;

    //
    // Pipe instances.
    //

    PPERFECT_HASH_SERVER_PIPE Pipes;
    ULONG PipeCount;
    ULONG Padding3;

    HANDLE ShutdownEvent;
    HANDLE StartedEvent;

    //
    // Backing vtbl.
    //

    PERFECT_HASH_SERVER_VTBL Interface;

} PERFECT_HASH_SERVER;
typedef PERFECT_HASH_SERVER *PPERFECT_HASH_SERVER;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_SERVER_INITIALIZE)(
    _In_ PPERFECT_HASH_SERVER Server
    );
typedef PERFECT_HASH_SERVER_INITIALIZE *PPERFECT_HASH_SERVER_INITIALIZE;

typedef
VOID
(NTAPI PERFECT_HASH_SERVER_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PPERFECT_HASH_SERVER Server
    );
typedef PERFECT_HASH_SERVER_RUNDOWN *PPERFECT_HASH_SERVER_RUNDOWN;

extern PERFECT_HASH_SERVER_INITIALIZE PerfectHashServerInitialize;
extern PERFECT_HASH_SERVER_RUNDOWN PerfectHashServerRundown;

extern PERFECT_HASH_SERVER_SET_MAXIMUM_CONCURRENCY
    PerfectHashServerSetMaximumConcurrency;
extern PERFECT_HASH_SERVER_GET_MAXIMUM_CONCURRENCY
    PerfectHashServerGetMaximumConcurrency;
extern PERFECT_HASH_SERVER_SET_NUMA_NODE_MASK
    PerfectHashServerSetNumaNodeMask;
extern PERFECT_HASH_SERVER_GET_NUMA_NODE_MASK
    PerfectHashServerGetNumaNodeMask;
extern PERFECT_HASH_SERVER_SET_ENDPOINT PerfectHashServerSetEndpoint;
extern PERFECT_HASH_SERVER_GET_ENDPOINT PerfectHashServerGetEndpoint;
extern PERFECT_HASH_SERVER_SET_LOCAL_ONLY PerfectHashServerSetLocalOnly;
extern PERFECT_HASH_SERVER_GET_LOCAL_ONLY PerfectHashServerGetLocalOnly;
extern PERFECT_HASH_SERVER_START PerfectHashServerStart;
extern PERFECT_HASH_SERVER_STOP PerfectHashServerStop;
extern PERFECT_HASH_SERVER_WAIT PerfectHashServerWait;
extern PERFECT_HASH_SERVER_SUBMIT_REQUEST PerfectHashServerSubmitRequest;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
