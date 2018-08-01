/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    DestroyPerfectHashTable.c

Abstract:

    This module implements the destroy routine for the PerfectHashTable
    component.

--*/

#include "stdafx.h"

_Use_decl_annotations_
BOOLEAN
DestroyPerfectHashTable(
    PPERFECT_HASH_TABLE *PerfectHashTablePointer,
    PBOOLEAN IsProcessTerminating
    )
/*++

Routine Description:

    Destroys a previously created PERFECT_HASH_TABLE structure, freeing all
    memory unless the IsProcessTerminating flag is TRUE.

Arguments:

    PerfectHashTablePointer - Supplies the address of a variable that contains
        the address of the PERFECT_HASH_TABLE structure to destroy.  This
        variable will be cleared (i.e. the pointer will be set to NULL) if the
        routine destroys the structure successfully (returns TRUE).

    IsProcessTerminating - Optionally supplies a pointer to a boolean flag
        indicating whether or not the process is terminating.  If the pointer
        is non-NULL and the underlying value is TRUE, the method returns success
        immediately.  (If the process is terminating, there is no need to walk
        any internal data structures and individually free elements.)

Return Value:

    TRUE on success, FALSE on failure.  If successful, a NULL pointer will be
    written to the PerfectHashTablePointer parameter.

--*/
{
    PRTL Rtl;
    BOOLEAN Success;
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE Table;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(PerfectHashTablePointer)) {
        goto Error;
    }

    if (!ARGUMENT_PRESENT(*PerfectHashTablePointer)) {
        goto Error;
    }

    if (ARGUMENT_PRESENT(IsProcessTerminating)) {
        if (*IsProcessTerminating) {

            //
            // Fast-path exit.  Clear the caller's pointer and return success.
            //

            *PerfectHashTablePointer = NULL;
            return TRUE;
        }
    }

    //
    // A valid pointer has been provided, and the process isn't terminating.
    // Initialize aliases and continue with destroy logic.
    //

    Table = *PerfectHashTablePointer;
    Rtl = Table->Rtl;
    Allocator = Table->Allocator;

    //
    // Sanity check the perfect hash structure size matches what we expect.
    //

    ASSERT(Table->SizeOfStruct == sizeof(*Table));

    //
    // Close resources associated with the :Info stream.
    //

    if (Table->InfoStreamBaseAddress) {
        UnmapViewOfFile(Table->InfoStreamBaseAddress);
        Table->InfoStreamBaseAddress = NULL;
    }

    if (Table->InfoStreamMappingHandle) {
        CloseHandle(Table->InfoStreamMappingHandle);
        Table->InfoStreamMappingHandle = NULL;
    }

    if (Table->InfoStreamFileHandle) {
        CloseHandle(Table->InfoStreamFileHandle);
        Table->InfoStreamFileHandle = NULL;
    }

    //
    // Clean up any resources that are still active.
    //

    if (Table->BaseAddress) {
        UnmapViewOfFile(Table->BaseAddress);
        Table->BaseAddress = NULL;
    }

    if (Table->MappingHandle) {
        CloseHandle(Table->MappingHandle);
        Table->MappingHandle = NULL;
    }

    if (Table->FileHandle) {
        CloseHandle(Table->FileHandle);
        Table->FileHandle = NULL;
    }

    //
    // Free the buffer created as part of key validation, if applicable.
    //

    if (Table->KeysBitmap.Buffer) {
        VirtualFree(Table->KeysBitmap.Buffer, 0, MEM_RELEASE);
        Table->KeysBitmap.Buffer = NULL;
        Table->KeysBitmap.SizeOfBitMap = 0;
    }

    //
    // Free the memory used for the values array, if applicable.
    //

    if (Table->ValuesBaseAddress) {
        VirtualFree(Table->ValuesBaseAddress, 0, MEM_RELEASE);
        Table->ValuesBaseAddress = NULL;
    }

    //
    // Free the underlying memory and clear the caller's pointer.
    //

    Allocator->FreePointer(Allocator->Context, PerfectHashTablePointer);

    Success = TRUE;

    goto End;

Error:

    Success = FALSE;

    //
    // Intentional follow-on to End.
    //

End:

    return Success;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
