/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableTls.c

Abstract:

    This module provides TLS glue to the PerfectHashTable component.

--*/

#include "stdafx.h"

//
// Our TLS index.  Assigned at PROCESS_ATTACH, free'd at PROCESS_DETACH.
//

ULONG PerfectHashTableTlsIndex;

PERFECT_HASH_TABLE_TLS_FUNCTION PerfectHashTableTlsProcessAttach;

_Use_decl_annotations_
PerfectHashTableTlsProcessAttach(
    HMODULE Module,
    ULONG   Reason,
    LPVOID  Reserved
    )
{
    PerfectHashTableTlsIndex = TlsAlloc();

    if (PerfectHashTableTlsIndex == TLS_OUT_OF_INDEXES) {
        return FALSE;
    }

    return TRUE;
}

PERFECT_HASH_TABLE_TLS_FUNCTION PerfectHashTableTlsProcessDetach;

_Use_decl_annotations_
PerfectHashTableTlsProcessDetach(
    HMODULE Module,
    ULONG   Reason,
    LPVOID  Reserved
    )
{
    BOOL IsProcessTerminating;

    IsProcessTerminating = (Reserved != NULL);

    if (IsProcessTerminating) {
        goto End;
    }

    if (PerfectHashTableTlsIndex == TLS_OUT_OF_INDEXES) {
        goto End;
    }

    if (!TlsFree(PerfectHashTableTlsIndex)) {

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

PERFECT_HASH_TABLE_TLS_SET_CONTEXT PerfectHashTableTlsSetContext;

_Use_decl_annotations_
BOOLEAN
PerfectHashTableTlsSetContext(
    PPERFECT_HASH_TABLE_CONTEXT Context
    )
{
    return TlsSetValue(PerfectHashTableTlsIndex, Context);
}

PERFECT_HASH_TABLE_TLS_GET_CONTEXT PerfectHashTableTlsGetContext;

_Use_decl_annotations_
PPERFECT_HASH_TABLE_CONTEXT
PerfectHashTableTlsGetContext(
    VOID
    )
{
    PVOID Value;

    Value = TlsGetValue(PerfectHashTableTlsIndex);

    return (PPERFECT_HASH_TABLE_CONTEXT)Value;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
