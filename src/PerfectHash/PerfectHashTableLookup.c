/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableLookup.c

Abstract:

    This module implements the Lookup() routine for the PerfectHashTable
    component.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PerfectHashTableLookup(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Value
    )
/*++

Routine Description:

    Looks up given key in a perfect hash table and returns the value set by
    the Insert() routine.  If no insertion has taken place for this key, this
    routine guarantees to return 0 as the value.

    N.B. If Key did not appear in the original set the hash table was created
         from, the behavior of this routine is undefined.  (In practice, the
         value returned will be the value for some other key in the table that
         hashes to the same location -- or potentially an empty slot in the
         table.)

Arguments:

    Table - Supplies a pointer to the table for which the key lookup is to be
        performed.

    Key - Supplies the key to look up.

    Value - Receives the vaue for the given key.

Return Value:

    S_OK in all normal operating conditions.  E_FAIL may be returned in some
    cases when passed a key not in the original input set.  The Value parameter
    will be set to NULL in this case.

--*/
{
    ULONG Index;
    HRESULT Result;

    //
    // Obtain the index for this key.
    //

    Result = Table->Vtbl->Index(Table, Key, &Index);

    if (FAILED(Result)) {

        //
        // Clear the caller's pointers and return the error code.
        //

        *Value = 0;
        return E_FAIL;
    }

    *Value = Table->Values[Index];

    return S_OK;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
