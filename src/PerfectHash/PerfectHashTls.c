/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTls.c

Abstract:

    This module provides TLS glue to the perfect hash library.

--*/

#include "stdafx.h"

//
// Our TLS index.  Assigned at PROCESS_ATTACH, free'd at PROCESS_DETACH.
//

ULONG PerfectHashTlsIndex;

PERFECT_HASH_TLS_FUNCTION PerfectHashTlsProcessAttach;

_Use_decl_annotations_
BOOLEAN
PerfectHashTlsProcessAttach(
    HMODULE Module,
    ULONG   Reason,
    LPVOID  Reserved
    )
{
    UNREFERENCED_PARAMETER(Module);
    UNREFERENCED_PARAMETER(Reason);
    UNREFERENCED_PARAMETER(Reserved);

    PerfectHashTlsIndex = TlsAlloc();

    if (PerfectHashTlsIndex == TLS_OUT_OF_INDEXES) {
        return FALSE;
    }

    return TRUE;
}

PERFECT_HASH_TLS_FUNCTION PerfectHashTlsProcessDetach;

_Use_decl_annotations_
BOOLEAN
PerfectHashTlsProcessDetach(
    HMODULE Module,
    ULONG   Reason,
    LPVOID  Reserved
    )
{
    UNREFERENCED_PARAMETER(Module);
    UNREFERENCED_PARAMETER(Reason);

    BOOL IsProcessTerminating;

    IsProcessTerminating = (Reserved != NULL);

    if (IsProcessTerminating) {
        goto End;
    }

    if (PerfectHashTlsIndex == TLS_OUT_OF_INDEXES) {
        goto End;
    }

    if (!TlsFree(PerfectHashTlsIndex)) {

        //
        // Can't do anything here.
        //

        NOTHING;
    }

    //
    // Note that we always return TRUE here, even if we had a failure.  We're
    // only called at DLL_PROCESS_DETACH, so there's not much we can do when
    // there is actually an error anyway.
    //

End:

    return TRUE;
}

//
// TLS Set/Get Context functions.
//

PERFECT_HASH_TLS_SET_CONTEXT PerfectHashTlsSetContext;

_Use_decl_annotations_
BOOL
PerfectHashTlsSetContext(
    PPERFECT_HASH_TLS_CONTEXT Context
    )
{
    return TlsSetValue(PerfectHashTlsIndex, Context);
}

PERFECT_HASH_TLS_GET_CONTEXT PerfectHashTlsGetContext;

_Use_decl_annotations_
PPERFECT_HASH_TLS_CONTEXT
PerfectHashTlsGetContext(
    VOID
    )
{
    PVOID Value;

    Value = TlsGetValue(PerfectHashTlsIndex);

    return (PPERFECT_HASH_TLS_CONTEXT)Value;
}

_Use_decl_annotations_
PPERFECT_HASH_TLS_CONTEXT
PerfectHashTlsEnsureContext(
    VOID
    )
{
    PVOID Value;

    Value = TlsGetValue(PerfectHashTlsIndex);

    if (!Value) {
        PH_RAISE(PH_E_NO_TLS_CONTEXT_SET);
    }

    return (PPERFECT_HASH_TLS_CONTEXT)Value;
}

PERFECT_HASH_TLS_GET_OR_SET_CONTEXT PerfectHashTlsGetOrSetContext;

_Use_decl_annotations_
PPERFECT_HASH_TLS_CONTEXT
PerfectHashTlsGetOrSetContext(
    PPERFECT_HASH_TLS_CONTEXT Context
    )
{
    PVOID Value;

    Value = TlsGetValue(PerfectHashTlsIndex);

    if (!Value) {
        TlsSetValue(PerfectHashTlsIndex, Context);
        Value = Context;
    }

    return (PPERFECT_HASH_TLS_CONTEXT)Value;
}

PERFECT_HASH_TLS_CLEAR_CONTEXT_IF_ACTIVE PerfectHashTlsClearContextIfActive;

_Use_decl_annotations_
VOID
PerfectHashTlsClearContextIfActive(
    PPERFECT_HASH_TLS_CONTEXT Context
    )
{
    PPERFECT_HASH_TLS_CONTEXT Active;

    Active = PerfectHashTlsEnsureContext();

    if (Active == Context) {
        TlsSetValue(PerfectHashTlsIndex, NULL);
    }
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
