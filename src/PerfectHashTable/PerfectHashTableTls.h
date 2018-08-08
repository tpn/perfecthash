/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableTls.h

Abstract:

    This is the private header file for TLS functionality associated with the
    perfect hash table library.

--*/

#pragma once

#include "stdafx.h"

//
// TLS-related structures and functions.
//

typedef struct _PERFECT_HASH_TABLE_TLS_CONTEXT {
    PVOID Unused;
} PERFECT_HASH_TABLE_TLS_CONTEXT;
typedef PERFECT_HASH_TABLE_TLS_CONTEXT *PPERFECT_HASH_TABLE_TLS_CONTEXT;

extern ULONG PerfectHashTableTlsIndex;

//
// The PROCESS_ATTACH and PROCESS_ATTACH functions share the same signature.
//

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(PERFECT_HASH_TABLE_TLS_FUNCTION)(
    _In_    HMODULE     Module,
    _In_    DWORD       Reason,
    _In_    LPVOID      Reserved
    );
typedef PERFECT_HASH_TABLE_TLS_FUNCTION *PPERFECT_HASH_TABLE_TLS_FUNCTION;

PERFECT_HASH_TABLE_TLS_FUNCTION PerfectHashTableTlsProcessAttach;
PERFECT_HASH_TABLE_TLS_FUNCTION PerfectHashTableTlsProcessDetach;

//
// Define TLS Get/Set context functions.
//

typedef
_Check_return_
_Success_(return != 0)
BOOL
(PERFECT_HASH_TABLE_TLS_SET_CONTEXT)(
    _In_ struct _PERFECT_HASH_TABLE_CONTEXT *Context
    );
typedef PERFECT_HASH_TABLE_TLS_SET_CONTEXT *PPERFECT_HASH_TABLE_TLS_SET_CONTEXT;

typedef
_Check_return_
_Success_(return != 0)
struct _PERFECT_HASH_TABLE_CONTEXT *
(PERFECT_HASH_TABLE_TLS_GET_CONTEXT)(
    VOID
    );
typedef PERFECT_HASH_TABLE_TLS_GET_CONTEXT *PPERFECT_HASH_TABLE_TLS_GET_CONTEXT;

extern PERFECT_HASH_TABLE_TLS_SET_CONTEXT PerfectHashTableTlsSetContext;
extern PERFECT_HASH_TABLE_TLS_GET_CONTEXT PerfectHashTableTlsGetContext;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
