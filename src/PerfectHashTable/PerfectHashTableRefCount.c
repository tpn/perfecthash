/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableRefCount.c

Abstract:

    This module implements the AddRef and Release routines for the
    PerfectHashTable component.

--*/

#include "stdafx.h"

_Use_decl_annotations_
ULONG
PerfectHashTableAddRef(
    PPERFECT_HASH_TABLE Table
    )
/*++

Routine Description:

    Increments the reference count for a perfect hash table.

Arguments:

    Table - Supplies a pointer to the table for which the reference count
        is to be incremented.

Return Value:

    The new reference count.

--*/
{
    return InterlockedIncrement(&Table->ReferenceCount);
}

_Use_decl_annotations_
ULONG
PerfectHashTableRelease(
    PPERFECT_HASH_TABLE Table
    )
/*++

Routine Description:

    Decrements the reference count associated with a perfect hash table.  If
    this is the last reference, the table is destroyed.

Arguments:

    Table - Supplies a pointer to the table for which the reference count
        is to be decremented.

Return Value:

    The new reference count.

--*/
{
    ULONG Count = InterlockedDecrement(&Table->ReferenceCount);
    PPERFECT_HASH_TABLE TablePointer = Table;

    if (Count > 0) {
        return Count;
    }

    DestroyPerfectHashTable(&TablePointer, NULL);

    return Count;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
