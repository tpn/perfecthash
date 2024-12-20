/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

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
        // When set, indicates custom allocator details are available.
        //

        ULONG CustomAllocatorDetailsPresent:1;

        //
        // When set, indicates new graph instances should be created as CUDA
        // graphs.  (Used by GraphInitialize() when creating instances of a
        // GRAPH structure via IID_PERFECT_HASH_GRAPH.)
        //

        ULONG CreateCuGraph:1;

        //
        // When set, indicates the global component lock has been exclusively
        // acquired.
        //

        ULONG GlobalComponentLockAcquired:1;

        //
        // Unused bits.  (Consume these before the Unused2 bits.)
        //

        ULONG Unused1:3;

        //
        // The following bits, when set, prevent the global component logic
        // from running when a new component is being created.  This is used
        // to explicitly create a new component for an interface classed as
        // global.  (Recap of current behavior: a request to create an instance
        // of a global interface ID will be satisfied by returning a reference
        // to a previously created global (i.e. singleton) instance.)
        //
        // This is currently used by CreatePerfectHashTableImplChm01(), for
        // example, to disable the global allocator component before creating
        // the graph instances that will be used for parallel solution finding.
        // This ensures each graph gets its own allocator, which means its own
        // independent heap that can be destroyed in one fell swoop with a
        // HeapDestroy() call in rundown.
        //
        // N.B. The positions of these bits must match the interface ID of the
        //      relevant component, plus 1 to accomodate bit test instructions
        //      being 0-based.  E.g. the Rtl interface ID is 6, and thus, the
        //      DisableGlobalRtlComponent is bit 7 in this structure.  Given
        //      an interface ID, this allows us to test if the appropriate bit
        //      is set via:
        //
        //      #define TlsContextIsGlobalComponentDisabled(TlsContext, Id) \
        //          TestBit32(&TlsContext->Flags.AsLong, Id)
        //
        //      If you add new global components, take special care to ensure
        //      the relevant interface bit position is used.  (This will lead
        //      to more UnusedN-type gaps in bit positions, which is fine.)
        //

        ULONG DisableGlobalRtlComponent:1;
        ULONG DisableGlobalAllocatorComponent:1;

        //
        // Remaining unused bits.  (Consume Unused1 before using these.)
        //

        ULONG Unused2:24;

    };
    LONG AsLong;
    ULONG AsULong;
} PERFECT_HASH_TLS_CONTEXT_FLAGS;
C_ASSERT(sizeof(PERFECT_HASH_TLS_CONTEXT_FLAGS) == sizeof(ULONG));
typedef PERFECT_HASH_TLS_CONTEXT_FLAGS *PPERFECT_HASH_TLS_CONTEXT_FLAGS;

#define TlsContextDisableGlobalAllocator(TlsContext)          \
    TlsContext->Flags.DisableGlobalAllocatorComponent = TRUE; \
    ASSERT(                                                   \
        TlsContextIsGlobalComponentDisabled(                  \
            TlsContext,                                       \
            PerfectHashAllocatorInterfaceId                   \
        )                                                     \
    )

#define TlsContextEnableGlobalAllocator(TlsContext)            \
    TlsContext->Flags.DisableGlobalAllocatorComponent = FALSE; \
    ASSERT(                                                    \
        !TlsContextIsGlobalComponentDisabled(                  \
            TlsContext,                                        \
            PerfectHashAllocatorInterfaceId                    \
        )                                                      \
    )

#define TlsContextTryCreateGlobalComponent(TlsContext, Id) \
    IsGlobalComponentInterfaceId(Id) &&                    \
    !TlsContextIsGlobalComponentDisabled(TlsContext, Id)

#define TlsContextCustomAllocatorDetailsPresent(TlsContext) \
    (TlsContext && TlsContext->Flags.CustomAllocatorDetailsPresent)

typedef struct _PERFECT_HASH_TLS_CONTEXT {
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PERFECT_HASH_TLS_CONTEXT_FLAGS Flags;
    ULONG LastError;
    HRESULT LastResult;
    ULONG Padding1;
    PPERFECT_HASH_KEYS Keys;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_TABLE Table;
    PRTL Rtl;
    PALLOCATOR Allocator;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_DIRECTORY Directory;
    struct _GRAPH *Graph;
    struct _CU *Cu;

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

#define TlsContextIsGlobalComponentDisabled2(TlsContext, Id) \
    TestBit32(&(TlsContext)->Flags.AsLong, (LONG)Id)

static
BOOL
TlsContextIsGlobalComponentDisabled(
    _In_ PPERFECT_HASH_TLS_CONTEXT TlsContext,
    _In_ PERFECT_HASH_INTERFACE_ID Id
    )
{
    BOOL Result;
    BOOL Result2;
    LONG Flags;

    Flags = TlsContext->Flags.AsLong;

    Result = TestBit32(&Flags, (LONG)Id);
    Result2 = TlsContextIsGlobalComponentDisabled2(TlsContext, Id);
    ASSERT(Result == Result2);
    return Result;
}

extern ULONG PerfectHashTlsIndex;

//
// The PROCESS_ATTACH and PROCESS_ATTACH functions share the same signature.
//

typedef
_Must_inspect_result_
_Success_(return != 0)
BOOLEAN
(PERFECT_HASH_TLS_FUNCTION)(
    _In_    HMODULE     Module,
    _In_    DWORD       Reason,
    _In_    LPVOID      Reserved
    );
typedef PERFECT_HASH_TLS_FUNCTION *PPERFECT_HASH_TLS_FUNCTION;

#ifdef PH_WINDOWS
PERFECT_HASH_TLS_FUNCTION PerfectHashTlsProcessAttach;
PERFECT_HASH_TLS_FUNCTION PerfectHashTlsProcessDetach;
#endif

//
// Define TLS Get/Set context functions.
//

typedef
_Must_inspect_result_
_Success_(return != 0)
BOOL
(NTAPI PERFECT_HASH_TLS_SET_CONTEXT)(
    _In_opt_ PPERFECT_HASH_TLS_CONTEXT TlsContext
    );
typedef PERFECT_HASH_TLS_SET_CONTEXT *PPERFECT_HASH_TLS_SET_CONTEXT;

typedef
_Must_inspect_result_
_Success_(return != 0)
PPERFECT_HASH_TLS_CONTEXT
(NTAPI PERFECT_HASH_TLS_GET_CONTEXT)(
    VOID
    );
typedef PERFECT_HASH_TLS_GET_CONTEXT *PPERFECT_HASH_TLS_GET_CONTEXT;

typedef
_Ret_notnull_
PPERFECT_HASH_TLS_CONTEXT
(NTAPI PERFECT_HASH_TLS_ENSURE_CONTEXT)(
    VOID
    );
typedef PERFECT_HASH_TLS_ENSURE_CONTEXT *PPERFECT_HASH_TLS_ENSURE_CONTEXT;

//
// N.B. We don't have an annotation on the TlsContext parameter (i.e. _In_)
//      for the following routine as it's valid to pass an uninitialized
//      struct pointer (as we call ZeroStructPointerInline() on it if we
//      end up using it), and I don't know how to express this in SAL as it
//      is dependent upon the value returned by TlsGetValue().
//

typedef
_Ret_notnull_
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

#ifndef __INTELLISENSE__
extern PERFECT_HASH_TLS_SET_CONTEXT PerfectHashTlsSetContext;
extern PERFECT_HASH_TLS_GET_CONTEXT PerfectHashTlsGetContext;
extern PERFECT_HASH_TLS_GET_OR_SET_CONTEXT PerfectHashTlsGetOrSetContext;
extern PERFECT_HASH_TLS_ENSURE_CONTEXT PerfectHashTlsEnsureContext;
extern PERFECT_HASH_TLS_CLEAR_CONTEXT_IF_ACTIVE
    PerfectHashTlsClearContextIfActive;
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
