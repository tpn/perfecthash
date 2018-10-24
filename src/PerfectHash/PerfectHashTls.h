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

typedef union _PERFECT_HASH_TLS_CONTEXT_FLAGS {
    struct {

        //
        // When set, prevents the global component logic from running when a
        // new component is being created.  This is used to explicitly create
        // a new component for an interface that is classed as global (i.e.
        // Rtl and Allocator), and thus, would otherwise be satisified by
        // returning a reference to a previously created global (singleton)
        // component.
        //

        ULONG DisableGlobalComponents:1;

        //
        // When set, indicates custom allocator details are available.
        //

        ULONG CustomAllocatorDetailsPresent:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TLS_CONTEXT_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TLS_CONTEXT_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TLS_CONTEXT_FLAGS *PPERFECT_HASH_TLS_CONTEXT_FLAGS;

#define TlsContextDisableGlobalComponents(TlsContext) \
    (TlsContext && TlsContext->Flags.DisableGlobalComponents)

#define TlsContextCustomAllocatorDetailsPresent(TlsContext) \
    (TlsContext && TlsContext->Flags.CustomAllocatorDetailsPresent)

typedef struct _PERFECT_HASH_TLS_CONTEXT {
    PERFECT_HASH_TLS_CONTEXT_FLAGS Flags;
    ULONG Padding;
    ULONG LastError;
    HRESULT LastResult;
    PPERFECT_HASH_KEYS Keys;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_TABLE Table;
    PRTL Rtl;
    PALLOCATOR Allocator;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_DIRECTORY Directory;
    struct _GRAPH *Graph;

    //
    // Per-component custom areas.
    //

    //
    // Allocator
    //

    struct {
        ULONG_PTR HeapMinimumSize;
        ULONG HeapCreateFlags;
        ULONG Padding2;
    };

} PERFECT_HASH_TLS_CONTEXT;
typedef PERFECT_HASH_TLS_CONTEXT *PPERFECT_HASH_TLS_CONTEXT;

extern ULONG PerfectHashTlsIndex;

//
// The PROCESS_ATTACH and PROCESS_ATTACH functions share the same signature.
//

typedef
_Check_return_
_Success_(return != 0)
BOOLEAN
(PERFECT_HASH_TLS_FUNCTION)(
    _In_    HMODULE     Module,
    _In_    DWORD       Reason,
    _In_    LPVOID      Reserved
    );
typedef PERFECT_HASH_TLS_FUNCTION *PPERFECT_HASH_TLS_FUNCTION;

PERFECT_HASH_TLS_FUNCTION PerfectHashTlsProcessAttach;
PERFECT_HASH_TLS_FUNCTION PerfectHashTlsProcessDetach;

//
// Define TLS Get/Set context functions.
//

typedef
_Check_return_
_Success_(return != 0)
BOOL
(NTAPI PERFECT_HASH_TLS_SET_CONTEXT)(
    _In_opt_ PPERFECT_HASH_TLS_CONTEXT TlsContext
    );
typedef PERFECT_HASH_TLS_SET_CONTEXT *PPERFECT_HASH_TLS_SET_CONTEXT;

typedef
_Check_return_
_Success_(return != 0)
PPERFECT_HASH_TLS_CONTEXT
(NTAPI PERFECT_HASH_TLS_GET_CONTEXT)(
    VOID
    );
typedef PERFECT_HASH_TLS_GET_CONTEXT *PPERFECT_HASH_TLS_GET_CONTEXT;

typedef
_Check_return_
_Success_(return != 0)
PPERFECT_HASH_TLS_CONTEXT
(NTAPI PERFECT_HASH_TLS_ENSURE_CONTEXT)(
    VOID
    );
typedef PERFECT_HASH_TLS_ENSURE_CONTEXT *PPERFECT_HASH_TLS_ENSURE_CONTEXT;

//
// N.B. We don't have an annotation on the TlsContext parameter (i.e. _In_)
//      for the following routine as it's valid to pass an uninitialized
//      struct pointer (as we call ZeroStructPointerInline() on it if we
//      end up using it), and I don't know how to express this in SAL.
//

typedef
_Check_return_
_Success_(return != 0)
PPERFECT_HASH_TLS_CONTEXT
(NTAPI PERFECT_HASH_TLS_GET_OR_SET_CONTEXT)(
    PPERFECT_HASH_TLS_CONTEXT TlsContext
    );
typedef PERFECT_HASH_TLS_GET_OR_SET_CONTEXT
      *PPERFECT_HASH_TLS_GET_OR_SET_CONTEXT;

typedef
VOID
(NTAPI PERFECT_HASH_TLS_CLEAR_CONTEXT_IF_ACTIVE)(
    _In_ PPERFECT_HASH_TLS_CONTEXT TlsContext
    );
typedef PERFECT_HASH_TLS_CLEAR_CONTEXT_IF_ACTIVE
      *PPERFECT_HASH_TLS_CLEAR_CONTEXT_IF_ACTIVE;

extern PERFECT_HASH_TLS_SET_CONTEXT PerfectHashTlsSetContext;
extern PERFECT_HASH_TLS_GET_CONTEXT PerfectHashTlsGetContext;
extern PERFECT_HASH_TLS_GET_OR_SET_CONTEXT PerfectHashTlsGetOrSetContext;
extern PERFECT_HASH_TLS_ENSURE_CONTEXT PerfectHashTlsEnsureContext;
extern PERFECT_HASH_TLS_CLEAR_CONTEXT_IF_ACTIVE
    PerfectHashTlsClearContextIfActive;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
