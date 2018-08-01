/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableInsert.c

Abstract:

    This module implements the Insert() routine for the PerfectHashTable
    component.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PerfectHashTableInsert(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG Value,
    PULONG PreviousValue
    )
/*++

Routine Description:

    Looks up given key in a perfect hash table and returns the value set by
    the Insert() routine.  If no insertion has taken place for this key, this
    routine guarantees to return 0 as the value.

    N.B. If Key did not appear in the original set the hash table was created
         from, the behavior of this routine is undefined.  (In practice, the
         key will hash to either an existing key's location or an empty slot,
         so there is potential to corrupt the table in the sense that previously
         inserted values will be trampled over.)

Arguments:

    Table - Supplies a pointer to the table to insert the key/value into.

    Key - Supplies the key to insert.

    Value - Supplies the value to insert.

    PreviousValue - Optionally supplies a pointer that will receive the previous
        value at the relevant table location prior to this insertion.  If no
        prior insertion, the previous value is guaranteed to be 0.

Return Value:

    S_OK in all normal operating conditions.  E_FAIL may be returned in some
    cases when passed a key not in the original input set.  The PreviousValue
    parameter, if non-NULL, will be cleared in this case.

--*/
{
    ULONG Index;
    ULONG Existing;
    HRESULT Result;

    //
    // Obtain the index for this key.
    //

    Result = Table->Vtbl->Index(Table, Key, &Index);

    if (FAILED(Result)) {

        //
        // Clear the caller's pointer if applicable and return error.
        //

        if (ARGUMENT_PRESENT(PreviousValue)) {
            *PreviousValue = 0;
        }

        return E_FAIL;
    }

    //
    // Get the existing value.
    //

    Existing = Table->Values[Index];

    //
    // Write the new value.
    //

    Table->Values[Index] = Value;

    //
    // Update the caller's pointer if applicable.
    //

    if (ARGUMENT_PRESENT(PreviousValue)) {
        *PreviousValue = Existing;
    }

    return S_OK;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
