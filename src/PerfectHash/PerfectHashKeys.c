/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashKeys.c

Abstract:

    This is the module for the PERFECT_HASH_KEYS component of the perfect
    hash table library.  Keys refer to the set of ULONGs for which a perfect
    hash table is to be generated.  Routines are provided for initialization,
    rundown, and getting the bitmap of all key values.

--*/

#include "stdafx.h"

PERFECT_HASH_KEYS_INITIALIZE PerfectHashKeysInitialize;

_Use_decl_annotations_
HRESULT
PerfectHashKeysInitialize(
    PPERFECT_HASH_KEYS Keys
    )
/*++

Routine Description:

    Initializes a perfect hash keys structure.  This is a relatively simple
    method that just primes the COM scaffolding; the bulk of the work is done
    when loading the keys file (via PerfectHashKeys->Vtbl->Load).

Arguments:

    Keys - Supplies a pointer to a PERFECT_HASH_KEYS structure for which
        initialization is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - Keys is NULL.

    E_UNEXPECTED - All other errors.

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
                                        &IID_PERFECT_HASH_RTL,
                                        &Keys->Rtl);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Keys->Vtbl->CreateInstance(Keys,
                                        NULL,
                                        &IID_PERFECT_HASH_ALLOCATOR,
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

PERFECT_HASH_KEYS_RUNDOWN PerfectHashKeysRundown;

_Use_decl_annotations_
VOID
PerfectHashKeysRundown(
    PPERFECT_HASH_KEYS Keys
    )
/*++

Routine Description:

    Release all resources associated with a perfect hash table.

Arguments:

    Keys - Supplies a pointer to a PERFECT_HASH_KEYS structure for which
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

    if (Keys->MappedAddress) {

        //
        // If MappedAddress is non-NULL, BaseAddress is actually our
        // large page address which needs to be freed with VirtualFree().
        //

        ASSERT(Keys->Flags.MappedWithLargePages);
        if (!VirtualFree(Keys->BaseAddress, 0, MEM_RELEASE)) {
            SYS_ERROR(VirtualFree);
        }

        //
        // Switch the base address back so it's unmapped correctly below.
        //

        Keys->BaseAddress = Keys->MappedAddress;
        Keys->MappedAddress = NULL;
    }

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

PERFECT_HASH_KEYS_GET_BITMAP PerfectHashKeysGetBitmap;

_Use_decl_annotations_
HRESULT
PerfectHashKeysGetBitmap(
    PPERFECT_HASH_KEYS Keys,
    ULONG SizeOfBitmap,
    PPERFECT_HASH_KEYS_BITMAP Bitmap
    )
/*++

Routine Description:

    Returns a bitmap that reflects all seen bits within the key file.

Arguments:

    Keys - Supplies a pointer to a PERFECT_HASH_KEYS structure for which the
        bitmap is to be obtained.

    SizeOfBitmap - Supplies the size of the structure pointed to by the
        Bitmap parameter, in bytes.

    Bitmap - Supplies the address of a variable that receives the bitmap.

Return Value:

    S_OK - Success.

    E_POINTER - Keys or Bitmap is NULL.

    E_INVALIDARG - SizeOfBitmap does not match the size of the bitmap
        structure.

    PH_E_KEYS_NOT_LOADED - No file has been loaded yet.

    PH_E_KEYS_LOAD_ALREADY_IN_PROGRESS - A keys file load is in progress.

--*/
{
    PRTL Rtl;

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Bitmap)) {
        return E_POINTER;
    }

    if (SizeOfBitmap != sizeof(*Bitmap)) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashKeysLockExclusive(Keys)) {
        return PH_E_KEYS_LOAD_ALREADY_IN_PROGRESS;
    }

    if (!Keys->State.Loaded) {
        ReleasePerfectHashKeysLockExclusive(Keys);
        return PH_E_KEYS_NOT_LOADED;
    }

    Rtl = Keys->Rtl;

    CopyMemory(Bitmap,
               &Keys->Stats.KeysBitmap,
               sizeof(*Bitmap));

    ReleasePerfectHashKeysLockExclusive(Keys);

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
