/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashKeys.c

Abstract:

    This is the module for the PERFECT_HASH_KEYS component of the perfect
    hash table library.  Keys refer to the set of ULONGs for which a perfect
    hash table is to be generated.  Routines are provided for initialization,
    rundown, getting flags, getting the base address, and getting the bitmap
    of all key values.

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
    // Create Rtl, Allocator and File components.
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

    Result = Keys->Vtbl->CreateInstance(Keys,
                                        NULL,
                                        &IID_PERFECT_HASH_FILE,
                                        &Keys->File);

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

    Release all resources associated with a perfect hash keys instance.

Arguments:

    Keys - Supplies a pointer to a PERFECT_HASH_KEYS structure for which
        rundown is to be performed.

Return Value:

    None.

--*/
{
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

    //
    // Release COM references, if applicable.
    //

    if (Keys->File) {
        Keys->File->Vtbl->Release(Keys->File);
        Keys->File = NULL;
    }

    if (Keys->Allocator) {
        Keys->Allocator->Vtbl->Release(Keys->Allocator);
        Keys->Allocator = NULL;
    }

    if (Keys->Rtl) {
        Keys->Rtl->Vtbl->Release(Keys->Rtl);
        Keys->Rtl = NULL;
    }

}

PERFECT_HASH_KEYS_GET_FLAGS PerfectHashKeysGetFlags;

_Use_decl_annotations_
HRESULT
PerfectHashKeysGetFlags(
    PPERFECT_HASH_KEYS Keys,
    ULONG SizeOfFlags,
    PPERFECT_HASH_KEYS_FLAGS Flags
    )
/*++

Routine Description:

    Returns the flags associated with a loaded keys instance.

Arguments:

    Keys - Supplies a pointer to a PERFECT_HASH_KEYS structure for which the
        flags are to be obtained.

    SizeOfFlags - Supplies the size of the structure pointed to by the Flags
        parameter, in bytes.

    Flags - Supplies the address of a variable that receives the flags.

Return Value:

    S_OK - Success.

    E_POINTER - Keys or Flags is NULL.

    E_INVALIDARG - SizeOfFlags does not match the size of the flags structure.

    PH_E_KEYS_NOT_LOADED - No file has been loaded yet.

    PH_E_KEYS_LOAD_ALREADY_IN_PROGRESS - A keys file load is in progress.

--*/
{
    PRTL Rtl;

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Flags)) {
        return E_POINTER;
    }

    if (SizeOfFlags != sizeof(*Flags)) {
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

    Flags->AsULong = Keys->Flags.AsULong;

    ReleasePerfectHashKeysLockExclusive(Keys);

    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashKeysGetAddress(
    PPERFECT_HASH_KEYS Keys,
    PVOID *BaseAddress,
    PULARGE_INTEGER NumberOfElements
    )
/*++

Routine Description:

    Obtains the base address of the keys array and the number of elements
    present in the array.

Arguments:

    Keys - Supplies a pointer to a PERFECT_HASH_KEYS structure for which the
        base address and number of elements are to be obtained.

    BaseAddress - Receives the base address of the keys array.

    NumberOfElements - Receives the number of elements present in the array.

Return Value:

    S_OK - Success.

    E_POINTER - Keys, BaseAddress, or NumberOfElements is NULL.

    PH_E_KEYS_NOT_LOADED - No file has been loaded yet.

    PH_E_KEYS_LOCKED - The keys are locked.

--*/
{
    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(BaseAddress)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(NumberOfElements)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashKeysLockExclusive(Keys)) {
        return PH_E_KEYS_LOAD_ALREADY_IN_PROGRESS;
    }

    if (!Keys->State.Loaded) {
        ReleasePerfectHashKeysLockExclusive(Keys);
        return PH_E_KEYS_NOT_LOADED;
    }

    *BaseAddress = Keys->File->BaseAddress;
    NumberOfElements->QuadPart = Keys->NumberOfElements.QuadPart;

    ReleasePerfectHashKeysLockExclusive(Keys);

    return S_OK;
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

    PH_E_KEYS_LOCKED - The keys are locked.

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
        return PH_E_KEYS_LOCKED;
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
