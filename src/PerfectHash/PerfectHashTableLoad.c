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
    PPERFECT_HASH_TABLE_LOAD_FLAGS TableLoadFlagsPointer,
    PCUNICODE_STRING Path,
    PPERFECT_HASH_KEYS Keys
    )
/*++

Routine Description:

    Loads an on-disk representation of a perfect hash table.

Arguments:

    Table - Supplies a pointer to the PERFECT_HASH_TABLE interface for which
        the on-disk table is to be loaded.

    TableLoadFlags - Optionally supplies a pointer to a table load flags
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
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_FILE InfoStream;
    LARGE_INTEGER EndOfFile;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PERFECT_HASH_ALGORITHM_ID AlgorithmId;
    PERFECT_HASH_TABLE_LOAD_FLAGS TableLoadFlags;

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

    VALIDATE_FLAGS(TableLoad, TABLE_LOAD);

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
    // Initialize aliases.
    //

    Rtl = Table->Rtl;
    Allocator = Table->Allocator;
    File = Table->TableFile;
    InfoStream = Table->InfoStream;

    Result = InfoStream->Vtbl->Load(InfoStream, Path, NULL, &EndOfFile);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashTableLoad, Result);
        goto Error;
    }

    if (EndOfFile.QuadPart < sizeof(*TableInfoOnDisk)) {

        //
        // File is too small, it can't be an :Info we know about.
        //

        Result = PH_E_INFO_FILE_SMALLER_THAN_HEADER;
        goto Error;
    }

    //
    // We've obtained the TABLE_INFO_ON_DISK structure.  Ensure the
    // magic values are what we expect.
    //

    TableInfoOnDisk = (PTABLE_INFO_ON_DISK)File->BaseAddress;

    if (TableInfoOnDisk->Magic.LowPart  != TABLE_INFO_ON_DISK_MAGIC_LOWPART ||
        TableInfoOnDisk->Magic.HighPart != TABLE_INFO_ON_DISK_MAGIC_HIGHPART) {

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

    if (TableInfoOnDisk->SizeOfStruct < sizeof(*TableInfoOnDisk)) {
        Result = PH_E_INVALID_INFO_HEADER_SIZE;
        goto Error;
    }

    //
    // If a Keys parameter has been provided, compare the number of keys it
    // reports to the number of keys registered in the :Info header.  Error
    // out if they differ.
    //

    if (ARGUMENT_PRESENT(Keys)) {

        if (Keys->NumberOfElements.QuadPart !=
            TableInfoOnDisk->NumberOfKeys.QuadPart) {

            Result = PH_E_NUM_KEYS_MISMATCH_BETWEEN_HEADER_AND_KEYS;
            goto Error;
        }

    }

    //
    // Validate the algorithm ID.  We use this as a lookup directly into the
    // loader routines array, so validation is especially important.
    //

    AlgorithmId = TableInfoOnDisk->AlgorithmId;
    if (!IsValidPerfectHashAlgorithmId(AlgorithmId)) {
        Result = PH_E_INVALID_ALGORITHM_ID;
        goto Error;
    }

    //
    // Validate the hash function ID.
    //

    if (!IsValidPerfectHashHashFunctionId(TableInfoOnDisk->HashFunctionId)) {
        Result = PH_E_INVALID_HASH_FUNCTION_ID;
        goto Error;
    }

    //
    // Validate the masking type.
    //

    if (!IsValidPerfectHashMaskFunctionId(TableInfoOnDisk->MaskFunctionId)) {
        Result = PH_E_INVALID_MASK_FUNCTION_ID;
        goto Error;
    }

    //
    // We only support 32-bit (4 byte) keys at the moment.  Enforce this
    // restriction now.
    //

    if (TableInfoOnDisk->KeySizeInBytes != sizeof(ULONG)) {
        Result = PH_E_HEADER_KEY_SIZE_TOO_LARGE;
        goto Error;
    }

    //
    // Ensure both the number of keys and number of table elements are non-zero,
    // and that the number of keys is less than or equal to the number of table
    // elements.
    //

    NumberOfKeys = TableInfoOnDisk->NumberOfKeys.QuadPart;
    NumberOfTableElements = TableInfoOnDisk->NumberOfTableElements.QuadPart;

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

    ExpectedEndOfFile.QuadPart = (
        NumberOfTableElements *
        TableInfoOnDisk->KeySizeInBytes
    );

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

    if (!TableLoadFlags.DisableTryLargePagesForTableData) {

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
        !TableLoadFlags.DisableTryLargePagesForValuesArray
    );

    ValuesSizeInBytes = (
        TableInfoOnDisk->NumberOfTableElements.QuadPart *
        (ULONGLONG)TableInfoOnDisk->KeySizeInBytes
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
    Table->MaskFunctionId = TableInfoOnDisk->MaskFunctionId;
    Table->HashFunctionId = TableInfoOnDisk->HashFunctionId;

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
    Table->TableLoadFlags.AsULong = TableLoadFlags.AsULong;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    ReleasePerfectHashTableLockExclusive(Table);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
