/*++

Copyright (c) 2025 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashOnline.c

Abstract:

    This module implements initialization and rundown for the PERFECT_HASH_ONLINE
    component.

--*/

#include "stdafx.h"

PERFECT_HASH_ONLINE_INITIALIZE PerfectHashOnlineInitialize;

_Use_decl_annotations_
HRESULT
PerfectHashOnlineInitialize(
    PPERFECT_HASH_ONLINE Online
    )
/*++

Routine Description:

    Initializes a PERFECT_HASH_ONLINE instance.

Arguments:

    Online - Supplies a pointer to a PERFECT_HASH_ONLINE structure for which
        initialization is to be performed.

Return Value:

    S_OK on success.  E_POINTER if Online is NULL.

--*/
{
    HRESULT Result = S_OK;

    if (!ARGUMENT_PRESENT(Online)) {
        return E_POINTER;
    }

    Online->SizeOfStruct = sizeof(*Online);

    //
    // Create Rtl and Allocator components.
    //

    Result = Online->Vtbl->CreateInstance(Online,
                                          NULL,
                                          &IID_PERFECT_HASH_RTL,
                                          PPV(&Online->Rtl));

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Online->Vtbl->CreateInstance(Online,
                                          NULL,
                                          &IID_PERFECT_HASH_ALLOCATOR,
                                          PPV(&Online->Allocator));

    if (FAILED(Result)) {
        goto Error;
    }

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

End:

    return Result;
}

PERFECT_HASH_ONLINE_RUNDOWN PerfectHashOnlineRundown;

_Use_decl_annotations_
VOID
PerfectHashOnlineRundown(
    PPERFECT_HASH_ONLINE Online
    )
/*++

Routine Description:

    Releases all resources associated with a PERFECT_HASH_ONLINE instance.

Arguments:

    Online - Supplies a pointer to a PERFECT_HASH_ONLINE structure for which
        rundown is to be performed.

Return Value:

    None.

--*/
{
    if (!ARGUMENT_PRESENT(Online)) {
        return;
    }

    ASSERT(Online->SizeOfStruct == sizeof(*Online));

    RELEASE(Online->Allocator);
    RELEASE(Online->Rtl);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
