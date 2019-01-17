/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

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

    KeySizeInBytes - Supplies the size of each key element, in bytes.

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

    if (KeySizeInBytes == sizeof(ULONG)) {
        Is32Bit = TRUE;
    } else if (KeySizeInBytes == sizeof(ULONGLONG)) {
        Is32Bit = FALSE;
    } else {
        return PH_E_INVALID_KEY_SIZE;
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

    Keys->NumberOfElements.QuadPart = (
        File->FileInfo.EndOfFile.QuadPart /
        KeySizeInBytes
    );

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
    ULONG_PTR Offset;
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
    KeyArray = (PULONG)Keys->File->BaseAddress;
    NumberOfKeys = Keys->NumberOfElements.LowPart;
    KeysBitmap = &Stats.KeysBitmap;

    Stats.MinValue = (ULONG)-1;
    Stats.MaxValue = 0;

    Values = KeyArray;

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

        PopCount = (BYTE)PopulationCount32(Key);
        Stats.PopCount[PopCount] += 1;

        while (Key) {
            Bit = TrailingZeros32(Key);
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

    Leading = LeadingZeros32(Bitmap);
    Trailing = TrailingZeros32(Bitmap);
    PopCount = (BYTE)PopulationCount32(Bitmap);
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
    // Construct a string representation of the bitmap.  All bit positions are
    // initialized with the '0' character, the bitmap is enumerated, and a set
    // bit has its corresponding character set to '1'.
    //

    Key = Bitmap;
    String = (PCHAR)&KeysBitmap->String;
    FillMemory(String, sizeof(KeysBitmap->String), '0');

    while (Key) {
        Bit = TrailingZeros32(Key);
        Offset = (31 - Bit);
        Key &= Key - 1;
        String[Offset] = '1';
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
    ULONG NumberOfClearRuns;
    ULONG NumberOfClearRunsUnsorted;
    ULONG NumberOfBitmapRuns;
    PULONGLONG Values;
    PULONGLONG KeyArray;
    PCHAR String;
    const ULONG_PTR One = 1;
    ULONG_PTR Bit;
    ULONG_PTR Mask;
    ULONG_PTR Index;
    ULONG_PTR Offset;
    ULONG_PTR Shifted;
    ULONG_PTR Leading;
    ULONG_PTR Trailing;
    ULONGLONG NumberOfKeys;
    ULONGLONG InvertedBitmap;
    RTL_BITMAP RtlBitmap;
    PERFECT_HASH_KEYS_STATS Stats;
    PPERFECT_HASH_KEYS_BITMAP KeysBitmap;
    RTL_BITMAP_RUN BitmapRunsSorted[32];
    RTL_BITMAP_RUN BitmapRunsUnsorted[32];
    PRTL_BITMAP_RUN BitmapRuns;

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
    Prev = (ULONGLONG)-1;
    KeyArray = (PULONGLONG)Keys->File->BaseAddress;
    NumberOfKeys = Keys->NumberOfElements.QuadPart;
    KeysBitmap = &Stats.KeysBitmap;

    Stats.MinValue = (ULONGLONG)-1;
    Stats.MaxValue = 0;

    Values = KeyArray;

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

        PopCount = (BYTE)PopulationCount64(Key);
        Stats.PopCount[PopCount] += 1;

        while (Key) {
            Bit = TrailingZeros64(Key);
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

    Leading = LeadingZeros64(Bitmap);
    Trailing = TrailingZeros64(Bitmap);
    PopCount = (BYTE)PopulationCount64(Bitmap);
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
    NumberOfClearBits = (BYTE)PopulationCount64(InvertedBitmap);

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

    if (NumberOfClearBits >= 32) {

        //
        // N.B. The following is a work-in-progress.  The bitmap runs are not
        //      currently used anywhere.
        //

        RtlBitmap.Buffer = (PULONG)&Bitmap;
        RtlBitmap.SizeOfBitMap = 64;

        ZeroArray(BitmapRunsSorted);
        BitmapRuns = (PRTL_BITMAP_RUN)&BitmapRunsSorted;
        NumberOfBitmapRuns = ARRAYSIZE(BitmapRunsSorted);

        NumberOfClearRuns = Rtl->RtlFindClearRuns(&RtlBitmap,
                                                  BitmapRuns,
                                                  NumberOfBitmapRuns,
                                                  TRUE);

        ZeroArray(BitmapRunsUnsorted);
        BitmapRuns = (PRTL_BITMAP_RUN)&BitmapRunsUnsorted;

        NumberOfClearRunsUnsorted = Rtl->RtlFindClearRuns(&RtlBitmap,
                                                          BitmapRuns,
                                                          NumberOfClearRuns,
                                                          FALSE);
    }

    //
    // Construct a string representation of the bitmap.  All bit positions are
    // initialized with the '0' character, the bitmap is enumerated, and a set
    // bit has its corresponding character set to '1'.
    //

    Key = Bitmap;
    String = (PCHAR)&KeysBitmap->String;
    FillMemory(String, sizeof(KeysBitmap->String), '0');

    while (Key) {
        Bit = TrailingZeros64(Key);
        Offset = (63 - Bit);
        Key &= Key - 1;
        String[Offset] = '1';
    }

    //
    // Copy the local stack structure back to the keys instance and return
    // success.
    //

    CopyMemory(&Keys->Stats, &Stats, sizeof(Keys->Stats));

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
