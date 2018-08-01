/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    DestroyPerfectHashTableKeys.c

Abstract:

    This module implements the destroy routine for the PerfectHashTable
    component's keys structure.

--*/

#include "stdafx.h"

_Use_decl_annotations_
BOOLEAN
DestroyPerfectHashTableKeys(
    PPERFECT_HASH_TABLE_KEYS *KeysPointer
    )
/*++

Routine Description:

    Destroys a previously loaded PERFECT_HASH_TABLE_KEYS structure.

Arguments:

    KeysPointers - Supplies the address of a variable that contains the address
        of the previously created PERFECT_HASH_TABLE_KEYS structure, obtained
        via LoadPerfectHashTableKeys().  The underlying pointer will be cleared
        by this routine.

Return Value:

    TRUE on success, FALSE on failure.

--*/
{
    PPERFECT_HASH_TABLE_KEYS Keys;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(KeysPointer) || !ARGUMENT_PRESENT(*KeysPointer)) {
        return FALSE;
    }

    Keys = *KeysPointer;

    //
    // Clean up any resources we may have allocated.
    //

    if (Keys->BaseAddress) {
        UnmapViewOfFile(Keys->BaseAddress);
        Keys->BaseAddress = NULL;
    }

    if (Keys->MappingHandle) {
        CloseHandle(Keys->MappingHandle);
        Keys->MappingHandle = NULL;
    }

    if (Keys->FileHandle) {
        CloseHandle(Keys->FileHandle);
        Keys->FileHandle = NULL;
    }

    if (Keys->Allocator) {
        Keys->Allocator->FreePointer(Keys->Allocator->Context,
                                     KeysPointer);
    } else {
        *KeysPointer = NULL;
    }

    return TRUE;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
