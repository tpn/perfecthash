/*++

Copyright (c) 2018-2022 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashKeysLoad.c

Abstract:

    This module implements the key load and stats load routines for the perfect
    hash library's PERFECT_HASH_KEYS component.

--*/

#include "stdafx.h"

PERFECT_HASH_KEYS_LOAD PerfectHashKeysLoad;

_Use_decl_annotations_
HRESULT
PerfectHashKeysLoad(
    PPERFECT_HASH_KEYS Keys,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlagsPointer,
    PCUNICODE_STRING KeysPath,
    ULONG KeySizeInBytes
    )
/*++

Routine Description:

    Loads a keys file from the file system.

Arguments:

    Keys - Supplies a pointer to the PERFECT_HASH_KEYS structure for
        which the keys are to be loaded.

    KeysLoadFlags - Optionally supplies a pointer to a keys load flags structure
        that can be used to customize key loading behavior.

    KeysPath - Supplies a pointer to a UNICODE_STRING structure that represents
        a fully-qualified path of the keys to use for the perfect hash table.

        N.B. Path must be NULL-terminated, which is not normally required for
             UNICODE_STRING structures.  Howver, the underlying buffer is passed
             to CreateFileW(), which requires a NULL-terminated wstr.

    KeySizeInBytes - Supplies the size of each key element, in bytes.  If the
        TryInferKeySizeFromKeysFilename flag is set in the KeysLoadFlags param,
        this argument is ignored.

Return Value:

    S_OK - Success.

    E_POINTER - Keys or KeysPath was NULL.

    E_INVALIDARG - KeysPath was invalid.

    PH_E_INVALID_KEY_SIZE - The key size provided was not valid.

    PH_E_INVALID_KEYS_LOAD_FLAGS - The provided load flags were invalid.

    PH_E_KEYS_NOT_SORTED - Keys were not sorted.

    PH_E_DUPLICATE_KEYS_DETECTED - Duplicate keys were detected.

    PH_E_KEYS_LOCKED - The keys are locked.

    PH_E_KEYS_ALREADY_LOADED - A keys file has already been loaded.

    PH_E_KEYS_FILE_BASENAME_NOT_VALID_C_IDENTIFIER - The basename of the file
        is not a valid C identifier.

    E_UNEXPECTED - All other errors.

--*/
{
    BOOLEAN Is32Bit;
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH Path = NULL;
    LARGE_INTEGER EndOfFile = { 0 };
    PPERFECT_HASH_PATH_PARTS Parts = NULL;
    PERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags;
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(KeysPath)) {
        return E_POINTER;
    }

    if (!IsValidMinimumDirectoryNullTerminatedUnicodeString(KeysPath)) {
        return E_INVALIDARG;
    }

    VALIDATE_FLAGS(KeysLoad, KEYS_LOAD);

    if (!TryAcquirePerfectHashKeysLockExclusive(Keys)) {
        return PH_E_KEYS_LOCKED;
    }

    if (IsLoadedKeys(Keys)) {
        ReleasePerfectHashKeysLockExclusive(Keys);
        return PH_E_KEYS_ALREADY_LOADED;
    }

    //
    // Argument validation complete.  Continue with loading.
    //

    //
    // Create a path instance.
    //

    Result = Keys->Vtbl->CreateInstance(Keys,
                                        NULL,
                                        &IID_PERFECT_HASH_PATH,
                                        &Path);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreateInstance, Result);
        goto Error;
    }

    Result = Path->Vtbl->Copy(Path, KeysPath, &Parts, NULL);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCopy, Result);
        goto Error;
    }

    if (!IsBaseNameValidCIdentifier(Path)) {
        Result = PH_E_KEYS_FILE_BASENAME_NOT_VALID_C_IDENTIFIER;
        PH_ERROR(PerfectHashKeysLoad, Result);
        goto Error;
    }

    //
    // If we've been asked to try infer the key size from the .keys file name,
    // attempt that now.
    //

    if (!KeysLoadFlags.TryInferKeySizeFromKeysFilename) {

        //
        // Don't try and infer key size; just use the size provided by the
        // caller, it will be verified below.
        //

        NOTHING;

    } else {

        Result = TryExtractKeySizeFromFilename(Path, &KeySizeInBytes);

        if (Result == S_OK) {

            //
            // Key size was successfully extracted.
            //

            NOTHING;

        } else if (Result == PH_S_NO_KEY_SIZE_EXTRACTED_FROM_FILENAME) {

            //
            // No key size was extracted; use the default.
            //

            KeySizeInBytes = DEFAULT_KEY_SIZE_IN_BYTES;

        } else {

            //
            // Invariant check: Result should be indicating an error code here.
            //

            ASSERT(FAILED(Result));
            PH_ERROR(PerfectHashKeysLoad_TryExtractKeySizeFromFilename, Result);
            goto Error;

        }
    }

    //
    // Validate the key size; we only support ULONG and ULONGLONG at the moment.
    //

    if (KeySizeInBytes == sizeof(ULONG)) {
        Is32Bit = TRUE;
    } else if (KeySizeInBytes == sizeof(ULONGLONG)) {
        Is32Bit = FALSE;
    } else {
        Result = PH_E_INVALID_KEY_SIZE;
        goto Error;
    }

    //
    // Create a file instance.
    //

    Result = Keys->Vtbl->CreateInstance(Keys,
                                        NULL,
                                        &IID_PERFECT_HASH_FILE,
                                        &Keys->File);

    if (FAILED(Result)) {
        PH_ERROR(CreateInstancePerfectHashFile, Result);
        goto Error;
    }

    File = Keys->File;

    //
    // Initialize load flags, then load the path.
    //

    FileLoadFlags.AsULong = 0;

    FileLoadFlags.TryLargePagesForFileData = (
        KeysLoadFlags.TryLargePagesForKeysData
    );

    Result = File->Vtbl->Load(File, Path, &EndOfFile, &FileLoadFlags);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileLoad, Result);
        goto Error;
    }

    //
    // Capture the large page status from the file flags.
    //

    Keys->Flags.KeysDataUsesLargePages = File->Flags.UsesLargePages;

    //
    // Update the key size and initialize the key array base address to point
    // at the memory-mapped base address of the file.  Initialize the number
    // of elements (key count).
    //

    Keys->OriginalKeySizeInBytes = Keys->KeySizeInBytes = KeySizeInBytes;
    Keys->OriginalKeySizeType = Keys->KeySizeType = (
        Is32Bit ? LongType : LongLongType
    );
    Keys->KeyArrayBaseAddress = Keys->File->BaseAddress;

    Keys->NumberOfElements.QuadPart = (
        File->FileInfo.EndOfFile.QuadPart /
        KeySizeInBytes
    );

    //
    // Dispatch to the relevant LoadStats() routine.
    //

    if (!SkipKeysVerification(Keys)) {
        if (Is32Bit) {
            Result = PerfectHashKeysLoadStats32(Keys);
        } else {
            Result = PerfectHashKeysLoadStats64(Keys);
        }
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashKeysLoadStats, Result);
            goto Error;
        }
    }

    //
    // We've completed initialization, indicate success and jump to the end.
    //

    Keys->State.Loaded = TRUE;
    Keys->LoadFlags.AsULong = KeysLoadFlags.AsULong;
    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // N.B. We don't clean up any resources here; the rundown routine will
    //      take care of that.
    //

    //
    // Intentional follow-on to End.
    //

End:

    RELEASE(Path);

    ReleasePerfectHashKeysLockExclusive(Keys);

    return Result;
}


PERFECT_HASH_KEYS_LOAD_STATS PerfectHashKeysLoadStats32;

_Use_decl_annotations_
HRESULT
PerfectHashKeysLoadStats32(
    PPERFECT_HASH_KEYS Keys
    )
/*++

Routine Description:

    Loads statistics about a set of 32-bit keys during initialization.

Arguments:

    Keys - Supplies a pointer to the PERFECT_HASH_KEYS structure for
        which the stats are to be gathered.

Return Value:

    S_OK - Success.

    E_POINTER - Keys was NULL.

    PH_E_TOO_MANY_KEYS - Too many keys were present.

    PH_E_KEYS_NOT_SORTED - Keys were not sorted.

    PH_E_DUPLICATE_KEYS_DETECTED - Duplicate keys were detected.

--*/
{
    PRTL Rtl;
    ULONG Key;
    ULONG Prev;
    ULONG Bitmap;
    BYTE PopCount;
    ULONG Longest;
    ULONG Start;
    PULONG Values;
    PULONG KeyArray;
    PCHAR String;
    const ULONG_PTR One = 1;
    ULONG_PTR Bit;
    ULONG_PTR Mask;
    ULONG_PTR Index;
    ULONG_PTR Shifted;
    ULONG_PTR Leading;
    ULONG_PTR Trailing;
    ULONG_PTR NumberOfKeys;
    ULONG InvertedBitmap;
    RTL_BITMAP RtlBitmap;
    PERFECT_HASH_KEYS_STATS Stats;
    PPERFECT_HASH_KEYS_BITMAP KeysBitmap;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (Keys->NumberOfElements.HighPart) {
        return PH_E_TOO_MANY_KEYS;
    }

    ASSERT(!SkipKeysVerification(Keys));

    //
    // Zero the stats struct, initialize local variables, then loop through
    // the key array, verify the keys are sorted and unique, and update the
    // bit position histogram and population count histogram.
    //

    Rtl = Keys->Rtl;
    ZeroStruct(Stats);
    Bitmap = 0;
    Prev = (ULONG)-1;
    KeyArray = (PULONG)Keys->KeyArrayBaseAddress;
    NumberOfKeys = Keys->NumberOfElements.LowPart;
    KeysBitmap = &Stats.KeysBitmap;

    Stats.MinValue = (ULONG)-1;
    Stats.MaxValue = 0;

    Values = KeyArray;

    //
    // If the first value is zero, toggle the relevant bitmap flag.
    //

    if (*Values == 0) {
        KeysBitmap->Flags.HasZero = TRUE;
    }

    for (Index = 0; Index < NumberOfKeys; Index++) {
        Key = *Values++;

        if (Index > 0) {
            if (Prev > Key) {
                return PH_E_KEYS_NOT_SORTED;
            } else if (Prev == Key) {
                return PH_E_DUPLICATE_KEYS_DETECTED;
            }
        }

        Prev = Key;

        PopCount = (BYTE)Rtl->PopulationCount32(Key);
        Stats.PopCount[PopCount] += 1;

        while (Key) {
            Bit = Rtl->TrailingZeros32(Key);
            Key &= Key - 1;
            Bitmap |= (1 << Bit);
            Stats.BitCount[Bit] += 1;
        }
    }

    //
    // We've verified the keys are sorted and unique, so we can obtain the
    // min/max values from the start and end of the array.
    //

    Stats.MinValue = KeyArray[0];
    Stats.MaxValue = KeyArray[NumberOfKeys - 1];

    //
    // Set the linear flag if the array of keys represents a linear sequence of
    // values with no gaps.
    //

    Keys->Flags.Linear = (
        (Stats.MaxValue - Stats.MinValue) == NumberOfKeys
    );

    //
    // Determine if the set bits in the bitmap are all contiguous.
    //

    KeysBitmap->Bitmap = Bitmap;

    Leading = Rtl->LeadingZeros32(Bitmap);
    Trailing = Rtl->TrailingZeros32(Bitmap);
    PopCount = (BYTE)Rtl->PopulationCount32(Bitmap);
    Mask = (One << (32 - Leading - Trailing)) - One;

    KeysBitmap->LeadingZeros = (BYTE)Leading;
    KeysBitmap->TrailingZeros = (BYTE)Trailing;

    if (PopCount == 32) {
        KeysBitmap->Flags.Contiguous = TRUE;
    } else if (Leading == 0) {
        KeysBitmap->Flags.Contiguous = FALSE;
    } else {
        Shifted = Bitmap;
        Shifted >>= Trailing;
        KeysBitmap->Flags.Contiguous = (Mask == Shifted);
    }

    if (KeysBitmap->Flags.Contiguous) {
        KeysBitmap->ShiftedMask = Mask;
    }

    //
    // Wire up the RTL_BITMAP structure and obtain the longest run of set bits.
    // We invert the bitmap first as we rely on RtlFindLongestRunClear().
    //

    InvertedBitmap = ~Bitmap;
    RtlBitmap.Buffer = &InvertedBitmap;
    RtlBitmap.SizeOfBitMap = 32;

    Start = 0;
    Longest = Rtl->RtlFindLongestRunClear(&RtlBitmap, &Start);
    KeysBitmap->LongestRunLength = (BYTE)Longest;
    KeysBitmap->LongestRunStart = (BYTE)Start;

    //
    // Construct a string representation of the bitmap.
    //

    Key = Bitmap;
    String = (PCHAR)&KeysBitmap->String;

    for (Bit = 0; Bit < 32; Bit++) {
        String[Bit] = ((Bitmap & (1 << Bit)) != 0) ? '1' : '0';
    }

    //
    // Clear the high 32-bits.
    //

    for (Bit = 32; Bit < 64; Bit++) {
        String[Bit] = '0';
    }

    //
    // Copy the local stack structure back to the keys instance and return
    // success.
    //

    CopyMemory(&Keys->Stats, &Stats, sizeof(Keys->Stats));

    return S_OK;
}


PERFECT_HASH_KEYS_LOAD_STATS PerfectHashKeysLoadStats64;

_Use_decl_annotations_
HRESULT
PerfectHashKeysLoadStats64(
    PPERFECT_HASH_KEYS Keys
    )
/*++

Routine Description:

    Loads statistics about a set of 64-bit keys during initialization.

    N.B. This routine has been copied-and-pasted from the 32-bit routine above
         with the necessary types and intrinsic function names changed.  If you
         need to make additional changes to it, make sure the 32-bit routine
         above is also updated accordingly.

Arguments:

    Keys - Supplies a pointer to the PERFECT_HASH_KEYS structure for
        which the stats are to be gathered.

Return Value:

    S_OK - Success.

    E_POINTER - Keys was NULL.

    PH_E_TOO_MANY_KEYS - Too many keys were present.

    PH_E_KEYS_NOT_SORTED - Keys were not sorted.

    PH_E_DUPLICATE_KEYS_DETECTED - Duplicate keys were detected.

--*/
{
    PRTL Rtl;
    ULONGLONG Key;
    ULONGLONG Prev;
    ULONGLONG Bitmap;
    BYTE PopCount;
    BYTE NumberOfClearBits;
    ULONG Longest;
    ULONG Start;
    PULONGLONG Values;
    PULONGLONG KeyArray;
    PCHAR String;
    HRESULT Result = S_OK;
    const ULONG_PTR One = 1;
    ULONG_PTR Bit;
    ULONG_PTR Mask;
    ULONG_PTR Index;
    ULONG_PTR Shifted;
    ULONG_PTR Leading;
    ULONG_PTR Trailing;
    ULONGLONG NumberOfKeys;
    ULONGLONG InvertedBitmap;
    PALLOCATOR Allocator;
    PULONG DownsizedKeyArray = NULL;
    RTL_BITMAP RtlBitmap;
    PERFECT_HASH_KEYS_STATS Stats;
    PPERFECT_HASH_KEYS_BITMAP KeysBitmap;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (Keys->NumberOfElements.HighPart) {
        return PH_E_TOO_MANY_KEYS;
    }

    ASSERT(!SkipKeysVerification(Keys));

    //
    // Zero the stats struct, initialize local variables, then loop through
    // the key array, verify the keys are sorted and unique, and update the
    // bit position histogram and population count histogram.
    //

    Rtl = Keys->Rtl;
    Allocator = Keys->Allocator;
    ZeroStruct(Stats);
    Bitmap = 0;
    Prev = (ULONGLONG)-1;
    KeyArray = (PULONGLONG)Keys->File->BaseAddress;
    NumberOfKeys = Keys->NumberOfElements.QuadPart;
    KeysBitmap = &Stats.KeysBitmap;

    Stats.MinValue = (ULONGLONG)-1;
    Stats.MaxValue = 0;

    Values = KeyArray;

    //
    // If the first value is zero, toggle the relevant bitmap flag.
    //

    if (*Values == 0) {
        KeysBitmap->Flags.HasZero = TRUE;
    }

    for (Index = 0; Index < NumberOfKeys; Index++) {
        Key = *Values++;

        if (Index > 0) {
            if (Prev > Key) {
                return PH_E_KEYS_NOT_SORTED;
            } else if (Prev == Key) {
                return PH_E_DUPLICATE_KEYS_DETECTED;
            }
        }

        Prev = Key;

        PopCount = (BYTE)Rtl->PopulationCount64(Key);
        Stats.PopCount[PopCount] += 1;

        while (Key) {
            Bit = Rtl->TrailingZeros64(Key);
            Key &= Key - 1ULL;
            Bitmap |= (1ULL << Bit);
            Stats.BitCount[Bit] += 1;
        }
    }

    //
    // We've verified the keys are sorted and unique, so we can obtain the
    // min/max values from the start and end of the array.
    //

    Stats.MinValue = KeyArray[0];
    Stats.MaxValue = KeyArray[NumberOfKeys - 1];

    //
    // Set the linear flag if the array of keys represents a linear sequence of
    // values with no gaps.
    //

    Keys->Flags.Linear = (
        (Stats.MaxValue - Stats.MinValue) == NumberOfKeys
    );

    //
    // Determine if the set bits in the bitmap are all contiguous.
    //

    KeysBitmap->Bitmap = Bitmap;

    Leading = Rtl->LeadingZeros64(Bitmap);
    Trailing = Rtl->TrailingZeros64(Bitmap);
    PopCount = (BYTE)Rtl->PopulationCount64(Bitmap);
    Mask = (One << (64 - Leading - Trailing)) - One;

    KeysBitmap->LeadingZeros = (BYTE)Leading;
    KeysBitmap->TrailingZeros = (BYTE)Trailing;

    if (PopCount == 64) {
        KeysBitmap->Flags.Contiguous = TRUE;
    } else if (Leading == 0) {
        KeysBitmap->Flags.Contiguous = FALSE;
    } else {
        Shifted = Bitmap;
        Shifted >>= Trailing;
        KeysBitmap->Flags.Contiguous = (Mask == Shifted);
    }

    if (KeysBitmap->Flags.Contiguous) {
        KeysBitmap->ShiftedMask = Mask;
    }

    //
    // Wire up the RTL_BITMAP structure and obtain the longest run of set bits.
    // We invert the bitmap first as we rely on RtlFindLongestRunClear().
    //

    InvertedBitmap = ~Bitmap;
    NumberOfClearBits = (BYTE)Rtl->PopulationCount64(InvertedBitmap);

    RtlBitmap.Buffer = (PULONG)&InvertedBitmap;
    RtlBitmap.SizeOfBitMap = 64;

    Start = 0;
    Longest = Rtl->RtlFindLongestRunClear(&RtlBitmap, &Start);
    KeysBitmap->LongestRunLength = (BYTE)Longest;
    KeysBitmap->LongestRunStart = (BYTE)Start;

    //
    // If there are 32 or more clear bits, we have an opportunity to shrink
    // the keys from ULONGLONG to ULONG using the parallel bit extraction
    // intrinsic _pext_u64().
    //

    if (NumberOfClearBits >= 32 && !DisableImplicitKeyDownsizing(Keys)) {

        const ULONG DownsizedKeySizeInBytes = sizeof(ULONG);
        ULONGLONG DownsizedKey;

        //
        // Allocate a new array for the downsized keys.
        //

        DownsizedKeyArray = Allocator->Vtbl->Calloc(Allocator,
                                                    NumberOfKeys,
                                                    DownsizedKeySizeInBytes);

        if (!DownsizedKeyArray) {
            Result = E_OUTOFMEMORY;
            goto Error;
        }

        //
        // Loop through all of the keys again and downsize each one into its
        // corresponding location in the newly-allocated array.
        //

        Values = KeyArray;

        for (Index = 0; Index < NumberOfKeys; Index++) {
            Key = *Values++;

            DownsizedKey = _pext_u64(Key, Bitmap);
            ASSERT(Rtl->LeadingZeros64(DownsizedKey) >= 32);

            DownsizedKeyArray[Index] = (ULONG)DownsizedKey;
        }

        //
        // Keys have been downsized successfully.  Update the key size, capture
        // the downsize bitmap we used, and set the "downsizing occurred" flag.
        //

        Keys->KeySizeInBytes = DownsizedKeySizeInBytes;
        Keys->KeySizeType = LongType;
        Keys->DownsizeBitmap = Bitmap;
        Keys->Flags.DownsizingOccurred = TRUE;

        //
        // Invariant check: our key array base address should always match the
        // file's base address at this point.
        //

        ASSERT(Keys->KeyArrayBaseAddress == Keys->File->BaseAddress);

        //
        // Update the array address to point at our new downsized array and
        // clear the local pointer.
        //

        Keys->KeyArrayBaseAddress = DownsizedKeyArray;
        DownsizedKeyArray = NULL;

        //
        // Now that we've got 32-bit keys active, run the equivalent load stats
        // routine to ensure the main invariants are still being complied with.
        //

        Result = PerfectHashKeysLoadStats32(Keys);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashKeysLoadStats32_Downsized, Result);
            goto Error;
        }

        //
        // Skip the final bitmap representation and copying of stats; the
        // 32-bit load stats routine above takes precedence now that we've
        // downsized our keys.
        //

        goto End;
    }

    //
    // If we get to this point, no key downsizing has been performed.  That is,
    // the bitmap of all key bit values has less than 32 zeros present in it.
    // Continue with the normal stats finalization.
    //

    //
    // Construct a string representation of the bitmap.
    //

    Key = Bitmap;
    String = (PCHAR)&KeysBitmap->String;

    for (Bit = 0; Bit < 64; Bit++) {
        String[Bit] = ((Bitmap & (1ULL << Bit)) != 0) ? '1' : '0';
    }

    //
    // Copy the local stack structure back to the keys instance then finish
    // up.
    //

    CopyMemory(&Keys->Stats, &Stats, sizeof(Keys->Stats));

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
    // Free the downsized array if non-null.
    //

    if (DownsizedKeyArray) {
        Allocator->Vtbl->FreePointer(Allocator, &DownsizedKeyArray);
    }

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
