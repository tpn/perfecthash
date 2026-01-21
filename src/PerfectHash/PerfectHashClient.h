/*++

Copyright (c) 2018-2025 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashClient.h

Abstract:

    This is the private header file for the PERFECT_HASH_CLIENT component of
    the perfect hash library.  It defines the structure, and function pointer
    typedefs for private non-vtbl members.

--*/

#pragma once

#include "stdafx.h"

typedef union _PERFECT_HASH_CLIENT_STATE {
    struct {

        ULONG Initialized:1;
        ULONG Connected:1;
        ULONG Disconnected:1;

        ULONG Unused:29;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_CLIENT_STATE;
C_ASSERT(sizeof(PERFECT_HASH_CLIENT_STATE) == sizeof(ULONG));
typedef PERFECT_HASH_CLIENT_STATE *PPERFECT_HASH_CLIENT_STATE;

typedef union _PERFECT_HASH_CLIENT_FLAGS {
    struct {

        ULONG Unused:32;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_CLIENT_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_CLIENT_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_CLIENT_FLAGS *PPERFECT_HASH_CLIENT_FLAGS;

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_CLIENT {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_CLIENT);

    HANDLE ConnectionHandle;
    UNICODE_STRING Endpoint;
    UNICODE_STRING ResponsePayload;
    ULONG ResponsePayloadBufferSize;
    ULONG ResponseFlags;

    //
    // Backing vtbl.
    //

    PERFECT_HASH_CLIENT_VTBL Interface;

} PERFECT_HASH_CLIENT;
typedef PERFECT_HASH_CLIENT *PPERFECT_HASH_CLIENT;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_CLIENT_INITIALIZE)(
    _In_ PPERFECT_HASH_CLIENT Client
    );
typedef PERFECT_HASH_CLIENT_INITIALIZE *PPERFECT_HASH_CLIENT_INITIALIZE;

typedef
VOID
(NTAPI PERFECT_HASH_CLIENT_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PPERFECT_HASH_CLIENT Client
    );
typedef PERFECT_HASH_CLIENT_RUNDOWN *PPERFECT_HASH_CLIENT_RUNDOWN;

extern PERFECT_HASH_CLIENT_INITIALIZE PerfectHashClientInitialize;
extern PERFECT_HASH_CLIENT_RUNDOWN PerfectHashClientRundown;

extern PERFECT_HASH_CLIENT_CONNECT PerfectHashClientConnect;
extern PERFECT_HASH_CLIENT_DISCONNECT PerfectHashClientDisconnect;
extern PERFECT_HASH_CLIENT_SUBMIT_REQUEST PerfectHashClientSubmitRequest;
extern PERFECT_HASH_CLIENT_GET_LAST_RESPONSE PerfectHashClientGetLastResponse;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
