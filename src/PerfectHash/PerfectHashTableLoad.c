/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableLoad.c

Abstract:

    This module implements functionality for loading on-disk representations
    of previously created perfect hash tables.

--*/

#include "stdafx.h"

PERFECT_HASH_TABLE_LOAD PerfectHashTableLoad;

_Use_decl_annotations_
HRESULT
PerfectHashTableLoad(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_LOAD_FLAGS LoadFlagsPointer,
    PCUNICODE_STRING Path,
    PPERFECT_HASH_KEYS Keys
    )
/*++

Routine Description:

    Loads an on-disk representation of a perfect hash table.

Arguments:

    Table - Supplies a pointer to the PERFECT_HASH_TABLE interface for which
        the on-disk table is to be loaded.

    LoadFlags - Optionally supplies a pointer to a PERFECT_HASH_TABLE_LOAD_FLAGS
        structure that can be used to customize the loading behavior.

    Path - Supplies a pointer to a UNICODE_STRING structure representing the
        fully-qualified, NULL-terminated path of the file to be used to load
        the table.

    Keys - Optionally supplies a pointer to the keys for the hash table.

Return Value:

    S_OK - Table loaded successfully.

    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - Table or Path was NULL.

    E_INVALIDARG - Path was not valid.

    E_UNEXPECTED - General error.

    E_OUTOFMEMORY - Out of memory.

    PH_E_INVALID_TABLE_LOAD_FLAGS - Invalid table load flags provided.

    PH_E_TABLE_LOAD_ALREADY_IN_PROGRESS - A table load is already in progress.

    PH_E_TABLE_ALREADY_CREATED - A table has already been created.

    PH_E_TABLE_ALREADY_LOADED - A table file has already been loaded.

    PH_E_INFO_FILE_SMALLER_THAN_HEADER - The table's :Info file was smaller
        than our smallest-known on-disk header structure.

    PH_E_INVALID_MAGIC_VALUES - The table's magic numbers were invalid.

    PH_E_INVALID_INFO_HEADER_SIZE - Invalid header size reported by :Info.

    PH_E_NUM_KEYS_MISMATCH_BETWEEN_HEADER_AND_KEYS - The number of keys
        reported by the header does not match the deduced number of keys
        (obtained by dividing the file size by the key size reported by
        the header).

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID in header.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID in header.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID in header.

    PH_E_HEADER_KEY_SIZE_TOO_LARGE - Key size reported by the header is
        too large.

    PH_E_NUM_KEYS_IS_ZERO - The number of keys reported by the header is 0.

    PH_E_NUM_TABLE_ELEMENTS_IS_ZERO - The number of table elements reported by
        the header is 0.

    PH_E_NUM_KEYS_EXCEEDS_NUM_TABLE_ELEMENTS - The number of keys indicated in
        the header exceeds the number of table elements indicated by the header.

    PH_E_EXPECTED_EOF_ACTUAL_EOF_MISMATCH - The expected end-of-file, which
        is calculated by dividing the file size by number of table elements,
        did not match the actual on-disk file size.

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

--*/
{
    BOOL Success;
    PRTL Rtl = NULL;
    PWSTR Dest;
    PWSTR Source;
    PCHAR Buffer;
    PCHAR BaseBuffer = NULL;
    PCHAR ExpectedBuffer;
    ULONG ShareMode;
    HRESULT Result = S_OK;
    PVOID BaseAddress;
    PVOID LargePageAddress;
    ULONG_PTR LargePageAllocSize;
    HANDLE FileHandle;
    HANDLE MappingHandle;
    PALLOCATOR Allocator = NULL;
    ULONG FlagsAndAttributes;
    ULARGE_INTEGER AllocSize;
    FILE_STANDARD_INFO FileInfo;
    BOOLEAN LargePagesForValues;
    ULONG_INTEGER PathBufferSize;
    ULONG_INTEGER InfoPathBufferSize;
    LARGE_INTEGER ExpectedEndOfFile;
    ULONG_INTEGER AlignedPathBufferSize;
    ULONG_INTEGER AlignedInfoPathBufferSize;
    ULONGLONG NumberOfKeys;
    ULONGLONG NumberOfTableElements;
    ULONGLONG ValuesSizeInBytes;
    PTABLE_INFO_ON_DISK_HEADER Header;
    PERFECT_HASH_ALGORITHM_ID AlgorithmId;
    UNICODE_STRING InfoSuffix = RTL_CONSTANT_STRING(L":Info");
    PERFECT_HASH_TABLE_LOAD_FLAGS LoadFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    }

    if (!IsValidMinimumDirectoryNullTerminatedUnicodeString(Path)) {
        return E_INVALIDARG;
    }

    if (ARGUMENT_PRESENT(LoadFlagsPointer)) {
        if (FAILED(IsValidTableLoadFlags(LoadFlagsPointer))) {
            return PH_E_INVALID_TABLE_LOAD_FLAGS;
        } else {
            LoadFlags.AsULong = LoadFlagsPointer->AsULong;
        }
    } else {
        LoadFlags.AsULong = 0;
    }

    if (!TryAcquirePerfectHashTableLockExclusive(Table)) {
        return PH_E_TABLE_LOAD_ALREADY_IN_PROGRESS;
    }

    if (Table->Flags.Loaded) {
        ReleasePerfectHashTableLockExclusive(Table);
        return PH_E_TABLE_ALREADY_LOADED;
    } else if (Table->Flags.Created) {
        ReleasePerfectHashTableLockExclusive(Table);
        return PH_E_TABLE_ALREADY_CREATED;
    }

    //
    // Argument validation complete.
    //

    //
    // Calculate the size required to store a copy of the Path's unicode string
    // buffer, plus a trailing NULL.
    //

    PathBufferSize.LongPart = Path->Length + sizeof(Path->Buffer[0]);

    //
    // Ensure we haven't overflowed MAX_USHORT.
    //

    ASSERT(!PathBufferSize.HighPart);

    //
    // Align up to a 16 byte boundary.
    //

    AlignedPathBufferSize.LongPart = (
        ALIGN_UP(
            PathBufferSize.LongPart,
            16
        )
    );

    //
    // Repeat overflow check.
    //

    ASSERT(!AlignedPathBufferSize.HighPart);

    //
    // Calculate the size required for the :Info stream's fully-qualified path
    // name, which we obtain by adding the main path's length to the length
    // of the suffix, then plus the trailing NULL.
    //

    InfoPathBufferSize.LongPart = (

        //
        // Account for the length in bytes of the incoming path name.
        //

        Path->Length +

        //
        // Account for the length of the suffix.
        //

        InfoSuffix.Length +

        //
        // Account for a trailing NULL.
        //

        sizeof(InfoSuffix.Buffer[0])
    );

    //
    // Overflow check.
    //

    ASSERT(!InfoPathBufferSize.HighPart);

    //
    // Align up to a 16 byte boundary.
    //

    AlignedInfoPathBufferSize.LongPart = (
        ALIGN_UP(
            InfoPathBufferSize.LongPart,
            16
        )
    );

    ASSERT(!AlignedInfoPathBufferSize.HighPart);

    //
    // Calculate the allocation size required for supporting unicode
    // string buffers.
    //

    AllocSize.QuadPart = (

        //
        // Account for the unicode string buffer backing the path.
        //

        AlignedPathBufferSize.LongPart +

        //
        // Account for the unicode string buffer backing the :Info stream.
        //

        AlignedInfoPathBufferSize.LongPart

    );

    //
    // Sanity check we haven't overflowed MAX_ULONG.
    //

    ASSERT(!AllocSize.HighPart);

    //
    // Initialize aliases.
    //

    Rtl = Table->Rtl;
    Allocator = Table->Allocator;

    //
    // Proceed with allocation.
    //

    BaseBuffer = Buffer = (PCHAR)(
        Allocator->Vtbl->Calloc(
            Allocator,
            1,
            AllocSize.LowPart
        )
    );

    if (!BaseBuffer) {
        SYS_ERROR(HeapAlloc);
        ReleasePerfectHashTableLockExclusive(Table);
        return E_OUTOFMEMORY;
    }

    //
    // Allocation was successful, continue with initialization.  Carve out the
    // backing memory structures for the path's unicode buffer.
    //

    Table->Path.Buffer = (PWSTR)Buffer;
    Table->Path.Length = Path->Length;
    Table->Path.MaximumLength = Path->MaximumLength;

    //
    // Copy the path provided as a parameter to our local table's copy.
    //

    CopyMemory(Table->Path.Buffer, Path->Buffer, Path->Length);

    //
    // Verify the buffer is NULL-terminated.
    //

    Dest = &Table->Path.Buffer[Table->Path.Length >> 1];
    ASSERT(*Dest == L'\0');

    //
    // Advance past the aligned path buffer size such that we're positioned at
    // the start of the info stream buffer.
    //

    Buffer += AlignedPathBufferSize.LongPart;
    Table->InfoStreamPath.Buffer = (PWSTR)Buffer;
    Table->InfoStreamPath.MaximumLength = InfoPathBufferSize.LowPart;
    Table->InfoStreamPath.Length = (
        Table->InfoStreamPath.MaximumLength -
        sizeof(Table->InfoStreamPath.Buffer[0])
    );

    //
    // Advance the buffer and verify it points to the expected location, then
    // clear it, as it isn't required for the remainder of the routine.  (The
    // original address was captured in BaseBuffer, which we use at the end
    // to free the memory.)
    //

    Buffer += AlignedInfoPathBufferSize.LongPart;
    ExpectedBuffer = RtlOffsetToPointer(BaseBuffer, AllocSize.LowPart);
    ASSERT(Buffer == ExpectedBuffer);
    Buffer = NULL;

    //
    // Copy the full path into the info stream buffer.
    //

    CopyMemory(Table->InfoStreamPath.Buffer,
               Table->Path.Buffer,
               Table->Path.Length);

    //
    // Advance the Dest pointer to the end of the path buffer (e.g. after the
    // ".pht1" suffix).  Assert we're looking at a NULL character.
    //

    Dest = Table->InfoStreamPath.Buffer;
    Dest += (Table->Path.Length >> 1);
    ASSERT(*Dest == L'\0');

    //
    // Copy the :Info suffix over, then NULL-terminate.
    //

    Source = InfoSuffix.Buffer;

    while (*Source) {
        *Dest++ = *Source++;
    }

    *Dest = L'\0';

    //
    // We've finished initializing our two unicode string buffers for the
    // backing file and its :Info counterpart.  Initialize some aliases for
    // the CreateFile() calls, then attempt to open the :Info stream.
    //

    ShareMode = (
        FILE_SHARE_READ  |
        FILE_SHARE_WRITE |
        FILE_SHARE_DELETE
    );

    FlagsAndAttributes = FILE_FLAG_OVERLAPPED;

    FileHandle = CreateFileW(Table->InfoStreamPath.Buffer,
                             GENERIC_READ,
                             ShareMode,
                             NULL,
                             OPEN_EXISTING,
                             FlagsAndAttributes,
                             NULL);

    Table->InfoStreamFileHandle = FileHandle;

    if (!FileHandle || FileHandle == INVALID_HANDLE_VALUE) {

        //
        // Failed to open the file.  Without the :Info file, we can't proceed,
        // as it contains metadata information about the underlying perfect
        // hash table (such as algorithm used, hash function used, etc).  So,
        // error out.
        //

        SYS_ERROR(CreateFileW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Successfully opened the :Info stream.  Obtain the current size of the
    // file and make sure it has a file size that meets our minimum :Info
    // on-disk size.
    //

    Success = GetFileInformationByHandleEx(
        FileHandle,
        (FILE_INFO_BY_HANDLE_CLASS)FileStandardInfo,
        &FileInfo,
        sizeof(FileInfo)
    );

    if (!Success) {
        SYS_ERROR(GetFileInformationByHandleEx);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (FileInfo.EndOfFile.QuadPart < sizeof(*Header)) {

        //
        // File is too small, it can't be an :Info we know about.
        //

        Result = PH_E_INFO_FILE_SMALLER_THAN_HEADER;
        goto Error;
    }

    //
    // The file is a sensible non-zero size.  Proceed with creating a mapping.
    // We use 0 as the mapping size such that it defaults to whatever the file
    // size is.
    //

    MappingHandle = CreateFileMappingW(FileHandle,
                                       NULL,
                                       PAGE_READONLY,
                                       0,
                                       0,
                                       NULL);

    Table->InfoStreamMappingHandle = MappingHandle;

    if (!MappingHandle || MappingHandle == INVALID_HANDLE_VALUE) {
        SYS_ERROR(CreateFileMappingW);
        goto Error;
    }

    //
    // Successfully created the mapping handle.  Now, map it.
    //

    BaseAddress = MapViewOfFile(MappingHandle, FILE_MAP_READ, 0, 0, 0);

    Table->InfoStreamBaseAddress = BaseAddress;

    if (!BaseAddress) {
        SYS_ERROR(MapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // We've obtained the TABLE_INFO_ON_DISK_HEADER structure.  Ensure the
    // magic values are what we expect.
    //

    Header = (PTABLE_INFO_ON_DISK_HEADER)BaseAddress;

    if (Header->Magic.LowPart  != TABLE_INFO_ON_DISK_MAGIC_LOWPART ||
        Header->Magic.HighPart != TABLE_INFO_ON_DISK_MAGIC_HIGHPART) {

        //
        // Magic values don't match what we expect.  Abort the loading efforts.
        //

        Result = PH_E_INVALID_MAGIC_VALUES;
        goto Error;
    }

    //
    // Verify the size of the struct reported by the header is at least as
    // large as our header structure size.
    //

    if (Header->SizeOfStruct < sizeof(*Header)) {
        Result = PH_E_INVALID_INFO_HEADER_SIZE;
        goto Error;
    }

    //
    // If a Keys parameter has been provided, compare the number of keys it
    // reports to the number of keys registered in the :Info header.  Error
    // out if they differ.
    //

    if (ARGUMENT_PRESENT(Keys)) {

        if (Keys->NumberOfElements.QuadPart != Header->NumberOfKeys.QuadPart) {
            Result = PH_E_NUM_KEYS_MISMATCH_BETWEEN_HEADER_AND_KEYS;
            goto Error;
        }

    }

    //
    // Validate the algorithm ID.  We use this as a lookup directly into the
    // loader routines array, so validation is especially important.
    //

    AlgorithmId = Header->AlgorithmId;
    if (!IsValidPerfectHashAlgorithmId(AlgorithmId)) {
        Result = PH_E_INVALID_ALGORITHM_ID;
        goto Error;
    }

    //
    // Validate the hash function ID.
    //

    if (!IsValidPerfectHashHashFunctionId(Header->HashFunctionId)) {
        Result = PH_E_INVALID_HASH_FUNCTION_ID;
        goto Error;
    }

    //
    // Validate the masking type.
    //

    if (!IsValidPerfectHashMaskFunctionId(Header->MaskFunctionId)) {
        Result = PH_E_INVALID_MASK_FUNCTION_ID;
        goto Error;
    }

    //
    // We only support 32-bit (4 byte) keys at the moment.  Enforce this
    // restriction now.
    //

    if (Header->KeySizeInBytes != sizeof(ULONG)) {
        Result = PH_E_HEADER_KEY_SIZE_TOO_LARGE;
        goto Error;
    }

    //
    // Ensure both the number of keys and number of table elements are non-zero,
    // and that the number of keys is less than or equal to the number of table
    // elements.
    //

    NumberOfKeys = Header->NumberOfKeys.QuadPart;
    NumberOfTableElements = Header->NumberOfTableElements.QuadPart;

    if (NumberOfKeys == 0) {
        Result = PH_E_NUM_KEYS_IS_ZERO;
        goto Error;
    }

    if (NumberOfTableElements == 0) {
        Result = PH_E_NUM_TABLE_ELEMENTS_IS_ZERO;
        goto Error;
    }

    if (NumberOfKeys > NumberOfTableElements) {
        Result = PH_E_NUM_KEYS_EXCEEDS_NUM_TABLE_ELEMENTS;
        goto Error;
    }

    //
    // Initial validation checks have passed.  Proceed with opening up the
    // actual table data file.
    //

    FileHandle = CreateFileW(Table->Path.Buffer,
                             GENERIC_READ,
                             ShareMode,
                             NULL,
                             OPEN_EXISTING,
                             FILE_FLAG_OVERLAPPED,
                             NULL);

    Table->FileHandle = FileHandle;

    if (!FileHandle || FileHandle == INVALID_HANDLE_VALUE) {

        //
        // Failed to open the underlying data file.  Abort.
        //

        SYS_ERROR(CreateFileW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Successfully opened the data file.  Get the file information.
    //

    Success = GetFileInformationByHandleEx(
        FileHandle,
        (FILE_INFO_BY_HANDLE_CLASS)FileStandardInfo,
        &FileInfo,
        sizeof(FileInfo)
    );

    if (!Success) {
        SYS_ERROR(GetFileInformationByHandleEx);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // We can determine the expected file size by multipling the number of
    // table elements by the key size; both of which are available in the
    // :Info header.
    //

    ExpectedEndOfFile.QuadPart = NumberOfTableElements * Header->KeySizeInBytes;

    //
    // Sanity check that the expected end of file is not 0.
    //

    ASSERT(ExpectedEndOfFile.QuadPart > 0);

    //
    // Compare the expected value to the actual on-disk file size.  They should
    // be identical.  If they're not, abort.
    //

    if (FileInfo.EndOfFile.QuadPart != ExpectedEndOfFile.QuadPart) {

        //
        // Sizes don't match, abort.
        //

        Result = PH_E_EXPECTED_EOF_ACTUAL_EOF_MISMATCH;
        goto Error;
    }

    //
    // File size is valid.  Proceed with creating a mapping.  As with :Info,
    // we don't specify a size, allowing instead to just default to the
    // underlying file size.
    //

    MappingHandle = CreateFileMappingW(FileHandle,
                                       NULL,
                                       PAGE_READONLY,
                                       0,
                                       0,
                                       NULL);

    Table->MappingHandle = MappingHandle;

    if (!MappingHandle || MappingHandle == INVALID_HANDLE_VALUE) {

        //
        // Couldn't obtain a mapping, abort.
        //

        SYS_ERROR(CreateFileMappingW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Created the mapping successfully.  Now, map it.
    //

    BaseAddress = MapViewOfFile(MappingHandle, FILE_MAP_READ, 0, 0, 0);

    Table->BaseAddress = BaseAddress;

    if (!BaseAddress) {

        //
        // Failed to map the contents into memory.  Abort.
        //

        SYS_ERROR(MapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!LoadFlags.DisableTryLargePagesForTableData) {

        //
        // Attempt a large page allocation to contain the table data buffer.
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

            Table->Flags.TableDataUsesLargePages = TRUE;
            Table->MappedAddress = Table->BaseAddress;
            Table->BaseAddress = LargePageAddress;

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
    // Allocate an array for the table values (i.e. the things stored when the
    // Insert(Key, Value) routine is called).  The dimensions will be the same
    // as the number of table elements * key size, and can be indexed directly
    // by the result of the Index() routine.
    //

    LargePagesForValues = (BOOLEAN)(
        !LoadFlags.DisableTryLargePagesForValuesArray
    );

    ValuesSizeInBytes = (
        Header->NumberOfTableElements.QuadPart *
        (ULONGLONG)Header->KeySizeInBytes
    );

    BaseAddress = Rtl->Vtbl->TryLargePageVirtualAlloc(Rtl,
                                                      NULL,
                                                      ValuesSizeInBytes,
                                                      MEM_RESERVE | MEM_COMMIT,
                                                      PAGE_READWRITE,
                                                      &LargePagesForValues);

    Table->ValuesBaseAddress = BaseAddress;

    if (!BaseAddress) {
        SYS_ERROR(VirtualAlloc);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Update flags with large page result for values array.
    //

    Table->Flags.ValuesArrayUsesLargePages = LargePagesForValues;

    //
    // Copy the enumeration IDs back into the table structure.
    //

    Table->AlgorithmId = AlgorithmId;
    Table->MaskFunctionId = Header->MaskFunctionId;
    Table->HashFunctionId = Header->HashFunctionId;

    //
    // Complete initialization of the table's vtbl now that the hash/mask IDs
    // have been set.
    //

    CompletePerfectHashTableVtblInitialization(Table);

    //
    // We've completed loading the :Info structure and corresponding data array.
    // Call the algorithm's loader routine to give it a chance to continue
    // initialization.
    //

    Result = LoaderRoutines[AlgorithmId](Table);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // We're done!  Indicate success and finish up.
    //

    Table->State.Valid = TRUE;
    Table->Flags.Loaded = TRUE;
    Table->LoadFlags.AsULong = LoadFlags.AsULong;
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
    // Release the buffer we allocated for paths if applicable.
    //

    if (BaseBuffer && Allocator != NULL) {
        Allocator->Vtbl->FreePointer(Allocator, &BaseBuffer);
    }

    ReleasePerfectHashTableLockExclusive(Table);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
