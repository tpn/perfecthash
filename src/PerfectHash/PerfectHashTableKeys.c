/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableKeys.c

Abstract:

    This is the module for the PERFECT_HASH_TABLE_KEYS component of the perfect
    hash table library.  Keys refer to the set of ULONGs for which a perfect
    hash table is to be generated.  Routines are provided for initialization,
    rundown, and verification of unique keys.

--*/

#include "stdafx.h"

PERFECT_HASH_TABLE_KEYS_INITIALIZE PerfectHashTableKeysInitialize;

_Use_decl_annotations_
HRESULT
PerfectHashTableKeysInitialize(
    PPERFECT_HASH_TABLE_KEYS Keys
    )
/*++

Routine Description:

    Initializes a perfect hash table keys structure.  This is a relatively
    simple method that just primes the COM scaffolding; the bulk of the work
    is done when loading the keys file (via PerfectHashTableKeys->Vtbl->Load).

Arguments:

    Keys - Supplies a pointer to a PERFECT_HASH_TABLE_KEYS structure for which
        initialization is to be performed.

Return Value:

    S_OK on success.  E_POINTER if Keys is NULL.

--*/
{
    HRESULT Result = S_OK;

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    Keys->SizeOfStruct = sizeof(*Keys);

    //
    // Create Rtl and Allocator components.
    //

    Result = Keys->Vtbl->CreateInstance(Keys,
                                        NULL,
                                        &IID_PERFECT_HASH_TABLE_RTL,
                                        &Keys->Rtl);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Keys->Vtbl->CreateInstance(Keys,
                                        NULL,
                                        &IID_PERFECT_HASH_TABLE_ALLOCATOR,
                                        &Keys->Allocator);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // We're done!  Indicate success and finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

PERFECT_HASH_TABLE_KEYS_RUNDOWN PerfectHashTableKeysRundown;

_Use_decl_annotations_
VOID
PerfectHashTableKeysRundown(
    PPERFECT_HASH_TABLE_KEYS Keys
    )
/*++

Routine Description:

    Release all resources associated with a perfect hash table.

Arguments:

    Keys - Supplies a pointer to a PERFECT_HASH_TABLE_KEYS structure for which
        rundown is to be performed.

Return Value:

    None.

--*/
{
    PRTL Rtl;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Keys)) {
        return;
    }

    //
    // Sanity check structure size.
    //

    ASSERT(Keys->SizeOfStruct == sizeof(*Keys));

    Rtl = Keys->Rtl;
    if (!Rtl) {
        return;
    }

    //
    // Clean up any resources that are still active.
    //

    if (Keys->BaseAddress) {
        if (!UnmapViewOfFile(Keys->BaseAddress)) {
            SYS_ERROR(UnmapViewOfFile);
        }
        Keys->BaseAddress = NULL;
    }

    if (Keys->MappingHandle) {
        if (!CloseHandle(Keys->MappingHandle)) {
            SYS_ERROR(CloseHandle);
        }
        Keys->MappingHandle = NULL;
    }

    if (Keys->FileHandle) {
        if (!CloseHandle(Keys->FileHandle)) {
            SYS_ERROR(CloseHandle);
        }
        Keys->FileHandle = NULL;
    }

    //
    // Release COM references, if applicable.
    //

    if (Keys->Allocator) {
        if (Keys->Path.Buffer) {
            Keys->Allocator->Vtbl->FreePointer(Keys->Allocator,
                                               &Keys->Path.Buffer);
        }
        Keys->Allocator->Vtbl->Release(Keys->Allocator);
        Keys->Allocator = NULL;
    }

    if (Keys->Rtl) {
        Keys->Rtl->Vtbl->Release(Keys->Rtl);
        Keys->Rtl = NULL;
    }
}

PERFECT_HASH_TABLE_KEYS_VERIFY_UNIQUE PerfectHashTableKeysVerifyUnique;

_Use_decl_annotations_
HRESULT
PerfectHashTableKeysVerifyUnique(
    PPERFECT_HASH_TABLE_KEYS Keys
    )
/*++

Routine Description:

    This routine verifies that there are no duplicate values in a given keys
    file.

    N.B. The current implementation simply creates a 512MB bitmap buffer (1
         bit for every possible value) and uses that to detect duplicates.
         This adds a non-negligible amount of memory pressure, which could
         be alleviated with a more efficient data structure.

         Additionally, no information is currently provided as to which keys
         are duplicate.  This could be improved by at least adding an error
         message reporting which value is at fault.

Arguments:

    Keys - Supplies a pointer to a PERFECT_HASH_TABLE_KEYS instance for which
        the duplicate key check will be performed.

Return Value:

    S_OK on success, PH_E_DUPLICATE_KEYS_DETECTED if duplicate keys are
    detected, PH_E_TOO_MANY_KEYS if the keys instance has too many keys in
    it, and E_UNEXPECTED for all other errors.

--*/
{
    PRTL Rtl;
    ULONG Key;
    ULONG Index;
    ULONG NumberOfKeys;
    ULONG NumberOfSetBits;
    PVOID Buffer = NULL;
    ULONGLONG Bit;
    HRESULT Result = S_OK;
    RTL_BITMAP Bitmap;
    PLONGLONG BitmapBuffer;
    ULONGLONG BitmapBufferSize;
    BOOLEAN LargePagesForBitmapBuffer;

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (Keys->NumberOfElements.HighPart) {
        return PH_E_TOO_MANY_KEYS;
    }

    NumberOfKeys = Keys->NumberOfElements.LowPart;

    //
    // Allocate a 512MB buffer for the keys bitmap.
    //

    BitmapBufferSize = ((1ULL << 32ULL) >> 3ULL);

    //
    // Try a large page allocation for the bitmap buffer.
    //

    LargePagesForBitmapBuffer = TRUE;

    Rtl = Keys->Rtl;

    Buffer = Rtl->Vtbl->TryLargePageVirtualAlloc(Rtl,
                                                 NULL,
                                                 BitmapBufferSize,
                                                 MEM_RESERVE | MEM_COMMIT,
                                                 PAGE_READWRITE,
                                                 &LargePagesForBitmapBuffer);

    if (!Buffer) {

        //
        // Failed to create a bitmap buffer, abort.
        //

        SYS_ERROR(VirtualAlloc);
        goto Error;
    }

    BitmapBuffer = (PLONGLONG)Buffer;

    //
    // Loop through all the keys, obtain the bitmap bit representation, verify
    // that the bit hasn't been set yet, and set it.
    //

    for (Index = 0; Index < NumberOfKeys; Index++) {

        Key = Keys->Keys[Index];
        Bit = Key + 1;

        if (BitTestAndSet64(BitmapBuffer, Bit)) {
            Result = PH_E_DUPLICATE_KEYS_DETECTED;
            goto Error;
        }
    }

    //
    // Count all bits set.  It should match the number of keys.
    //

    Bitmap.SizeOfBitMap = (ULONG)-1;
    Bitmap.Buffer = (PULONG)Buffer;

    NumberOfSetBits = Rtl->RtlNumberOfSetBits(&Bitmap);

    if (NumberOfSetBits != NumberOfKeys) {
        Result = PH_E_KEYS_NUM_SET_BITS_NUM_KEYS_MISMATCH;
        goto Error;
    }

    //
    // We're done!  Indicate success and finish up.
    //

    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (Buffer) {
        if (!VirtualFree(Buffer, 0, MEM_RELEASE)) {
            SYS_ERROR(VirtualFree);
        }
        Buffer = NULL;
    }

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
