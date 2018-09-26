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
    PCUNICODE_STRING Path,
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

    Path - Supplies a pointer to a UNICODE_STRING structure that represents
        a fully-qualified path of the keys to use for the perfect hash table.

        N.B. Path must be NULL-terminated, which is not normally required for
             UNICODE_STRING structures.  Howver, the underlying buffer is passed
             to CreateFileW(), which requires a NULL-terminated wstr.

    KeySizeInBytes - Supplies the size of each key element, in bytes.

Return Value:

    S_OK - Success.

    E_POINTER - Keys or Path was NULL.

    E_INVALIDARG - Path was invalid.

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
    PRTL Rtl;
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;
    LARGE_INTEGER EndOfFile = { 0 };
    LARGE_INTEGER NumberOfElements;
    PERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags;
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    }

    if (!IsValidMinimumDirectoryNullTerminatedUnicodeString(Path)) {
        return E_INVALIDARG;
    }

    if (KeySizeInBytes != sizeof(ULONG)) {
        return PH_E_INVALID_KEY_SIZE;
    }

    VALIDATE_FLAGS(KeysLoad, KEYS_LOAD);

    if (!TryAcquirePerfectHashKeysLockExclusive(Keys)) {
        return PH_E_KEYS_LOCKED;
    }

    if (Keys->State.Loaded) {
        ReleasePerfectHashKeysLockExclusive(Keys);
        return PH_E_KEYS_ALREADY_LOADED;
    }

    //
    // Argument validation complete.  Continue with loading.
    //

    //
    // Initialize aliases.
    //

    Rtl = Keys->Rtl;

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

    if (KeysLoadFlags.DisableTryLargePagesForKeysData) {
        FileLoadFlags.DisableTryLargePagesForFileData = TRUE;
    }

    Result = File->Vtbl->Load(File, Path, &FileLoadFlags, &EndOfFile);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileLoad, Result);
        goto Error;
    }


    //
    // Calculate the size required for a copy of the Path's underlying
    // unicode string buffer.
    //

    AllocSize.QuadPart = Path->Length + sizeof(Path->Buffer[0]);

    //
    // Sanity check our size.
    //

    ASSERT(!AllocSize.HighPart);

    //
    // Proceed with allocation.
    //

    Buffer = (PCHAR)(
        Allocator->Vtbl->Calloc(
            Allocator,
            1,
            AllocSize.LowPart
        )
    );

    if (!Buffer) {
        SYS_ERROR(HeapAlloc);
        goto Error;
    }

    //
    // Open the file, create a file mapping, then map it into memory.
    //

    FileHandle = CreateFileW(
        Path->Buffer,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED,
        NULL
    );

    if (!FileHandle || FileHandle == INVALID_HANDLE_VALUE) {
        SYS_ERROR(CreateFileW);
        ReleasePerfectHashKeysLockExclusive(Keys);
        return E_UNEXPECTED;
    }

    //
    // N.B. Error handling should 'goto Error' from this point onward now that
    //      we have resources that may need to be cleaned up.
    //

    Success = GetFileInformationByHandleEx(
        FileHandle,
        (FILE_INFO_BY_HANDLE_CLASS)FileStandardInfo,
        &FileInfo,
        sizeof(FileInfo)
    );

    if (!Success) {
        SYS_ERROR(GetFileInformationByHandleEx);
        ReleasePerfectHashKeysLockExclusive(Keys);
        return E_UNEXPECTED;
    }

    //
    // The number of elements in the key file can be ascertained by right
    // shifting by 2.
    //

    NumberOfElements.QuadPart = FileInfo.EndOfFile.QuadPart >> 2;

    //
    // Sanity check the number of elements.  There shouldn't be more than
    // MAX_ULONG.
    //

    if (NumberOfElements.HighPart) {
        Result = PH_E_TOO_MANY_KEYS;
        goto Error;
    }

    //
    // Create the file mapping.
    //

    MappingHandle = CreateFileMappingW(FileHandle,
                                       NULL,
                                       PAGE_READONLY,
                                       FileInfo.EndOfFile.HighPart,
                                       FileInfo.EndOfFile.LowPart,
                                       NULL);

    if (!MappingHandle || MappingHandle == INVALID_HANDLE_VALUE) {
        SYS_ERROR(CreateFileMappingW);
        goto Error;
    }

    //
    // Successfully created a file mapping.  Now map it into memory.
    //

    BaseAddress = MapViewOfFile(MappingHandle,
                                FILE_MAP_READ,
                                0,
                                0,
                                FileInfo.EndOfFile.LowPart);

    if (!BaseAddress) {
        SYS_ERROR(MapViewOfFile);
        goto Error;
    }

    //
    // The file has been mapped successfully.  Fill out the main structure
    // details.
    //

    Keys->FileHandle = FileHandle;
    Keys->MappingHandle = MappingHandle;
    Keys->BaseAddress = BaseAddress;
    Keys->NumberOfElements.QuadPart = NumberOfElements.QuadPart;

    if (!KeysLoadFlags.DisableTryLargePagesForKeysData) {

        //
        // Attempt a large page allocation to contain the keys buffer.
        //

        ULONG LargePageAllocFlags;

        LargePageAllocFlags = MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
        LargePageAllocSize = ALIGN_UP_LARGE_PAGE(FileInfo.EndOfFile.QuadPart);
        LargePageAddress = VirtualAlloc(NULL,
                                        LargePageAllocSize,
                                        LargePageAllocFlags,
                                        PAGE_READWRITE);

        if (LargePageAddress) {

            //
            // The large page allocation was successful.
            //

            ULONG_PTR NumberOfPages;

            Keys->Flags.KeysDataUsesLargePages = TRUE;
            Keys->MappedAddress = Keys->BaseAddress;
            Keys->BaseAddress = LargePageAddress;

            //
            // We can use the allocation size here to capture the appropriate
            // number of pages that are mapped and accessible.
            //

            NumberOfPages = BYTES_TO_PAGES(FileInfo.AllocationSize.QuadPart);

            Rtl->Vtbl->CopyPages(Rtl,
                                 LargePageAddress,
                                 BaseAddress,
                                 (ULONG)NumberOfPages);
        }

    }

    //
    // Initialize the path length variables, point the buffer at the space after
    // our structure, then copy the string over and NULL-terminate it.
    //

    Keys->Path.Length = Path->Length;
    Keys->Path.MaximumLength = Path->Length + sizeof(Path->Buffer[0]);
    Keys->Path.Buffer = (PWSTR)Buffer;
    CopyMemory(Keys->Path.Buffer, Path->Buffer, Path->Length);
    Keys->Path.Buffer[Path->Length >> 1] = L'\0';

    Result = PerfectHashKeysLoadStats(Keys);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashKeysLoadStats, Result);
        goto Error;
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

    ReleasePerfectHashKeysLockExclusive(Keys);

    return Result;
}

PERFECT_HASH_KEYS_LOAD_STATS PerfectHashKeysLoadStats;

_Use_decl_annotations_
HRESULT
PerfectHashKeysLoadStats(
    PPERFECT_HASH_KEYS Keys
    )
/*++

Routine Description:

    Loads statistics about a set of keys during initialization.

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
    ULONG PopCount;
    ULONG Longest;
    ULONG Leading;
    ULONG Trailing;
    ULONG Start;
    PULONG Values;
    PCHAR String;
    ULONG_PTR Bit;
    ULONG_PTR Mask;
    ULONG_PTR Index;
    ULONG_PTR Offset;
    ULONG_PTR Shifted;
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

    //
    // Zero the stats struct, initialize local variables, then loop through
    // the key array, verify the keys are sorted and unique, and update the
    // bit position histogram and population count histogram.
    //

    Rtl = Keys->Rtl;
    ZeroStruct(Stats);
    Bitmap = 0;
    Prev = (ULONG)-1;
    Values = Keys->Keys;
    NumberOfKeys = Keys->NumberOfElements.LowPart;
    KeysBitmap = &Stats.KeysBitmap;

    Stats.MinValue = (ULONG)-1;
    Stats.MaxValue = 0;

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

        PopCount = PopulationCount32(Key);
        Stats.PopCount[PopCount] += 1;

        while (Key) {
            Bit = TrailingZeros(Key);
            Key &= Key - 1;
            Bitmap |= (1 << Bit);
            Stats.BitCount[Bit] += 1;
        }
    }

    //
    // We've verified the keys are sorted and unique, so we can obtain the
    // min/max values from the start and end of the array.
    //

    Stats.MinValue = Keys->Keys[0];
    Stats.MaxValue = Keys->Keys[NumberOfKeys - 1];

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

    Leading = LeadingZeros(Bitmap);
    Trailing = TrailingZeros(Bitmap);
    PopCount = PopulationCount32(Bitmap);
    Mask = (1 << (32 - Leading - Trailing)) - 1;

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
        KeysBitmap->ShiftedMask = (ULONG)Mask;
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
        Bit = TrailingZeros(Key);
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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
