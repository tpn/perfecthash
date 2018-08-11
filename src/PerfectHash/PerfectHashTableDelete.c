/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableDelete.c

Abstract:

    This module implements the Delete() routine for the PerfectHashTable
    component.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PerfectHashTableDelete(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG PreviousValue
    )
/*++

Routine Description:

    Deletes a key from a perfect hash table, optionally returning the value
    prior to deletion back to the caller.  Deletion simply clears the value
    associated with the key, and thus, is a simple O(1) operation.  Deleting
    a key that has not yet been inserted has no effect other than potentially
    returning 0 as the previous value.  That is, a caller can safely issue
    deletes of keys regardless of whether or not said keys were inserted first.

    N.B. If Key did not appear in the original set the hash table was created
         from, the behavior of this routine is undefined.  (In practice, the
         key will hash to either an existing key's location or an empty slot,
         so there is potential to corrupt the table in the sense that a
         previously inserted value for an unrelated, valid key will be cleared.)

Arguments:

    Table - Supplies a pointer to the table for which the key is to be deleted.

    Key - Supplies the key to delete.

    PreviousValue - Optionally supplies a pointer that will receive the previous
        value at the relevant table location prior to this deletion.  If no
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
    // Clear the value.
    //

    Table->Values[Index] = 0;

    //
    // Update the caller's pointer if applicable.
    //

    if (ARGUMENT_PRESENT(PreviousValue)) {
        *PreviousValue = Existing;
    }

    return S_OK;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
