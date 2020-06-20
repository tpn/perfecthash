/*++

Copyright (c) 2018-2022 Trent Nelson <trent@trent.me>

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
    // Create Rtl and Allocator components.
    //

    Result = Keys->Vtbl->CreateInstance(Keys,
                                        NULL,
                                        &IID_PERFECT_HASH_RTL,
                                        PPV(&Keys->Rtl));

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Keys->Vtbl->CreateInstance(Keys,
                                        NULL,
                                        &IID_PERFECT_HASH_ALLOCATOR,
                                        PPV(&Keys->Allocator));

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // Default to 4 byte/32-bit key size.  This can be overridden via the table
    // create parameter --KeySizeInBytes=N.
    //

    Keys->OriginalKeySizeInBytes = Keys->KeySizeInBytes =
        DEFAULT_KEY_SIZE_IN_BYTES;
    Keys->OriginalKeySizeType = Keys->KeySizeType = DEFAULT_KEY_TYPE;

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
    HRESULT Result;
    PALLOCATOR Allocator;

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

    if (Keys->File && Keys->File->BaseAddress) {

        Allocator = Keys->Allocator;

        //
        // Invariant check: if downsizing has occurred, the file's base address
        // should not match Keys->KeyArrayBaseAddress.  If no downsizing has
        // occurred, the addresses should match.
        //

        if (KeysWereDownsized(Keys)) {

            //
            // Downsizing occurred; addresses should differ.
            //

            if (Keys->File->BaseAddress == Keys->KeyArrayBaseAddress) {
                Result = PH_E_INVARIANT_CHECK_FAILED;
                PH_ERROR(KeysRelease_EqualBaseAddressesAfterDownsizing, Result);
                PH_RAISE(Result);
            }

            //
            // Release the heap-allocated key array.
            //

            Allocator->Vtbl->FreePointer(Allocator,
                                         PPV(&Keys->KeyArrayBaseAddress));

        } else {

            //
            // No downsizing occurred; addresses should be the same.
            //

            if (Keys->File->BaseAddress != Keys->KeyArrayBaseAddress) {
                Result = PH_E_INVARIANT_CHECK_FAILED;
                PH_ERROR(KeysRelease_UnequalBaseAddressesNoDownsizing, Result);
                PH_RAISE(Result);
            }

            //
            // Clear the key array pointer; we don't have to explicitly free
            // the memory as in the downsizing case above.
            //

            Keys->KeyArrayBaseAddress = NULL;
        }
    }

    //
    // Invariant check: Keys->KeyArrayBaseAddress should be NULL here.
    //

    if (Keys->KeyArrayBaseAddress != NULL) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(KeysRelease_KeyArrayBaseAddressNotNull, Result);
        PH_RAISE(Result);
    }

    //
    // Release COM references, if applicable.
    //

    RELEASE(Keys->File);
    RELEASE(Keys->Allocator);
    RELEASE(Keys->Rtl);

    return;
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

    *BaseAddress = Keys->KeyArrayBaseAddress;
    NumberOfElements->QuadPart = Keys->NumberOfKeys.QuadPart;

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

    PH_E_KEYS_VERIFICATION_SKIPPED - Keys verification was skipped.

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

    if (SkipKeysVerification(Keys)) {
        ReleasePerfectHashKeysLockExclusive(Keys);
        return PH_E_KEYS_VERIFICATION_SKIPPED;
    }

    Rtl = Keys->Rtl;

    CopyMemory(Bitmap,
               &Keys->Stats.KeysBitmap,
               sizeof(*Bitmap));

    ReleasePerfectHashKeysLockExclusive(Keys);

    return S_OK;
}


PERFECT_HASH_KEYS_GET_FILE PerfectHashKeysGetFile;

_Use_decl_annotations_
HRESULT
PerfectHashKeysGetFile(
    PPERFECT_HASH_KEYS Keys,
    PPERFECT_HASH_FILE *File
    )
/*++

Routine Description:

    Obtains the file instance for a given keys instance.

Arguments:

    Keys - Supplies a pointer to a PERFECT_HASH_KEYS structure for which the
        file is to be obtained.

    File - Supplies the address of a variable that receives a pointer to the
        file instance.  The caller must release this reference when finished
        with it via File->Vtbl->Release(File).

Return Value:

    S_OK - Success.

    E_POINTER - Keys or File parameters were NULL.

    PH_E_KEYS_LOCKED - Keys locked.

    PH_E_KEYS_NOT_LOADED - Keys not loaded.

--*/
{

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    //
    // Clear the caller's pointer up-front.
    //

    *File = NULL;

    if (!TryAcquirePerfectHashKeysLockShared(Keys)) {
        return PH_E_KEYS_LOCKED;
    }

    if (!IsLoadedKeys(Keys)) {
        ReleasePerfectHashKeysLockShared(Keys);
        return PH_E_KEYS_NOT_LOADED;
    }

    //
    // Argument validation complete.  Add a reference to the path and update
    // the caller's pointer, then return success.
    //

    Keys->File->Vtbl->AddRef(Keys->File);
    *File = Keys->File;

    ReleasePerfectHashKeysLockShared(Keys);

    return S_OK;
}


TRY_EXTRACT_KEY_SIZE_FROM_FILENAME TryExtractKeySizeFromFilename;

_Use_decl_annotations_
HRESULT
TryExtractKeySizeFromFilename(
    PPERFECT_HASH_PATH Path,
    PULONG KeySizeInBytesPointer
    )
/*++

Routine Description:

    Attempts to extract a key size, in bits, from the given path.  Key sizes
    currently detected: 8, 16, 32 and 64.

    N.B. The key size is specified in bits in the file name as an aesthetic
         preference, however, internally, the key size is always represented
         as bytes, which is why the second (output) parameter receives the
         value in bytes, not bits.

Arguments:

    Path - Supplies the path for which an attempt will be made to extract the
        key size in bits.

    KeySizeInBytesPointer - Receives the extracted key size, in *bytes*, if
        the routine was successful.  If no key size could be extracted, or an
        error occurred, 0 will be used.

Return Value:

    S_OK - Key size in bits successfully extracted.

    PH_S_NO_KEY_SIZE_EXTRACTED_FROM_FILENAME - No key size was extracted from
        the file name (note that this is not an error condition).

    E_POINTER - Path was NULL.

    PH_E_NO_PATH_SET - Path has not been set.

    PH_E_PATH_LOCKED - Path is locked.

--*/
{
    PWCHAR Wide;
    ULONG KeySizeInBytes = 0;
    HRESULT Result = S_OK;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashPathLockShared(Path)) {
        return PH_E_PATH_LOCKED;
    }

    if (!IsPathSet(Path)) {
        Result = PH_E_NO_PATH_SET;
        goto Error;
    }

    if (!IsValidUnicodeString(&Path->Extension)) {
        Result = PH_E_NO_PATH_EXTENSION_PRESENT;
        goto Error;
    }

    //
    // Argument validation complete.  Wire up our wide char pointer such that
    // it's pointing at the character preceding the file extension period (i.e.
    // if the file name is "foo64.keys", point at the '4').
    //

    Wide = Path->Extension.Buffer - 2;

    ASSERT(*(Wide + 1) == L'.');
    ASSERT((ULONG_PTR)Wide > (ULONG_PTR)Path->FileName.Buffer);

    //
    // Attempt to look for either 8, 16, 32 or 64 as the bit size.
    //

    if (Path->BaseName.Length >= sizeof(WCHAR)) {

        //
        // There is at least one character we can dereference; determine if it
        // is 8.
        //

        if (*Wide == L'8') {
            KeySizeInBytes = 1;
            goto End;
        }
    }

    if (Path->BaseName.Length >= (sizeof(WCHAR) * 2)) {

        //
        // There are at least two characters we can dereference; determine if
        // they are 16, 32 or 64.
        //

        if (*Wide == L'6' && *(Wide - 1) == L'1') {
            KeySizeInBytes = 2;
            goto End;
        }

        if (*Wide == L'2' && *(Wide - 1) == L'3') {
            KeySizeInBytes = 4;
            goto End;
        }

        if (*Wide == L'4' && *(Wide - 1) == L'6') {
            KeySizeInBytes = 8;
            goto End;
        }
    }

    //
    // If we get here, we weren't able to extract a key size.
    //

    Result = PH_S_NO_KEY_SIZE_EXTRACTED_FROM_FILENAME;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Update the caller's pointer, release the path lock, and return.
    //

    *KeySizeInBytesPointer = KeySizeInBytes;
    ReleasePerfectHashPathLockShared(Path);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
