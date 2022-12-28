/*++

Copyright (c) 2018-2022. Trent Nelson <trent@trent.me>

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
    PCUNICODE_STRING TablePath,
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

    TablePath - Supplies a pointer to a UNICODE_STRING structure representing
        the fully-qualified, NULL-terminated path of the file to be used to
        load the table.

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

    PH_E_TABLE_LOCKED - The table is locked.

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

    PH_E_INVARIANT_CHECK_FAILED - An internal invariant check failed.

--*/
{
    HRESULT Result = S_OK;
    LARGE_INTEGER ExpectedEndOfFile;
    ULONGLONG NumberOfKeys;
    ULONGLONG NumberOfTableElements;
    LARGE_INTEGER EndOfFile;
    PPERFECT_HASH_FILE File = NULL;
    PPERFECT_HASH_PATH Path = NULL;
    PPERFECT_HASH_FILE InfoStream = NULL;
    PPERFECT_HASH_PATH InfoStreamPath = NULL;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PERFECT_HASH_ALGORITHM_ID AlgorithmId;
    PERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags;
    PERFECT_HASH_FILE_LOAD_FLAGS InfoStreamLoadFlags;
    PERFECT_HASH_TABLE_LOAD_FLAGS TableLoadFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TablePath)) {
        return E_POINTER;
    }

    if (!IsValidUnicodeString(TablePath)) {
        return E_INVALIDARG;
    }

    VALIDATE_FLAGS(TableLoad, TABLE_LOAD);

    if (!TryAcquirePerfectHashTableLockExclusive(Table)) {
        return PH_E_TABLE_LOCKED;
    }

    if (Table->Flags.Loaded) {
        ReleasePerfectHashTableLockExclusive(Table);
        return PH_E_TABLE_ALREADY_LOADED;
    } else if (Table->Flags.Created) {
        ReleasePerfectHashTableLockExclusive(Table);
        return PH_E_TABLE_ALREADY_CREATED;
    }

    //
    // Verify Table->File and Table->InfoStream are NULL.  (If the loaded flag
    // is not set, they should be.)
    //

    if (Table->TableFile) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashTableLoad, Result);
        ReleasePerfectHashTableLockExclusive(Table);
        return Result;
    }

    if (Table->TableInfoStream) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashTableLoad, Result);
        ReleasePerfectHashTableLockExclusive(Table);
        return Result;
    }

    //
    // Argument validation complete.
    //

    //
    // We need to create two path instances.  One for the table path, and one
    // for the :Info stream.
    //

    Result = Table->Vtbl->CreateInstance(Table,
                                         NULL,
                                         &IID_PERFECT_HASH_PATH,
                                         &Path);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreateInstance, Result);
        goto Error;
    }

    Result = Path->Vtbl->Copy(Path, TablePath, NULL, NULL);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCopy, Result);
        goto Error;
    }

    //
    // Table path created successfully.  Now create one for the :Info stream.
    //

    Result = Table->Vtbl->CreateInstance(Table,
                                         NULL,
                                         &IID_PERFECT_HASH_PATH,
                                         &InfoStreamPath);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreateInstance, Result);
        goto Error;
    }

    Result = InfoStreamPath->Vtbl->Create(
        InfoStreamPath,
        Path,                   // ExistingPath
        NULL,                   // NewDirectory
        NULL,                   // DirectorySuffix
        NULL,                   // NewBaseName
        NULL,                   // BaseNameSuffix
        NULL,                   // NewExtension
        &TableInfoStreamName,   // NewStreamName
        NULL,                   // Parts
        NULL                    // Reserved
    );

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreate, Result);
        goto Error;
    }

    //
    // :Info stream path created successfully.  Create a file instance for it,
    // then Load() it.
    //

    Result = Table->Vtbl->CreateInstance(Table,
                                         NULL,
                                         &IID_PERFECT_HASH_FILE,
                                         &InfoStream);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileCreateInstance, Result);
        goto Error;
    }

    //
    // We don't need large pages for the :Info stream.
    //

    InfoStreamLoadFlags.AsULong = 0;
    InfoStreamLoadFlags.TryLargePagesForFileData = FALSE;

    Result = InfoStream->Vtbl->Load(InfoStream,
                                    InfoStreamPath,
                                    &EndOfFile,
                                    &InfoStreamLoadFlags);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashTableLoad, Result);
        goto Error;
    }

    Table->TableInfoStream = InfoStream;

    if (EndOfFile.QuadPart < sizeof(*TableInfoOnDisk)) {

        //
        // File is too small, it can't be an :Info we know about.
        //

        Result = PH_E_INFO_FILE_SMALLER_THAN_HEADER;
        goto Error;
    }

    //
    // Cast the :Info stream's base address to the TABLE_INFO_ON_DISK structure
    // and verify the magic values are what we expect.
    //

    TableInfoOnDisk = (PTABLE_INFO_ON_DISK)InfoStream->BaseAddress;

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

        if (Keys->NumberOfKeys.QuadPart !=
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

    Table->TableInfoOnDisk = TableInfoOnDisk;

    //
    // We've completed our validation of the :Info stream.  Proceed with the
    // table data file; create a new file instance, then Load() the path we
    // prepared earlier.
    //

    Result = Table->Vtbl->CreateInstance(Table,
                                         NULL,
                                         &IID_PERFECT_HASH_FILE,
                                         &File);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileCreateInstance, Result);
        goto Error;
    }

    //
    // Reset the end of file and initialize load flags.
    //

    EndOfFile.QuadPart = 0;
    FileLoadFlags.AsULong = 0;

    if (TableLoadFlags.TryLargePagesForTableData) {
        FileLoadFlags.TryLargePagesForFileData = TRUE;
    }

    Result = File->Vtbl->Load(File, Path, &EndOfFile, &FileLoadFlags);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashTableLoad, Result);
        goto Error;
    }

    Table->TableFile = File;

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

    if (ExpectedEndOfFile.QuadPart <= 0) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashTableLoad, Result);
        goto Error;
    }

    //
    // Compare the expected value to the actual on-disk file size.  They should
    // be identical.  If they're not, abort.
    //

    if (EndOfFile.QuadPart != ExpectedEndOfFile.QuadPart) {

        //
        // Sizes don't match, abort.
        //

        Result = PH_E_EXPECTED_EOF_ACTUAL_EOF_MISMATCH;
        goto Error;
    }

    //
    // Copy load flags.
    //

    Table->TableLoadFlags.AsULong = TableLoadFlags.AsULong;

    //
    // Create the values array.
    //

    Result = PerfectHashTableCreateValuesArray(Table, 0);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashTableCreateValuesArray, Result);
        goto Error;
    }

    //
    // Copy the enumeration IDs back into the table structure.
    //

    Table->AlgorithmId = AlgorithmId;
    Table->MaskFunctionId = TableInfoOnDisk->MaskFunctionId;
    Table->HashFunctionId = TableInfoOnDisk->HashFunctionId;

    //
    // Complete initialization of the table now that the algo/hash/mask IDs
    // have been set.
    //

    CompletePerfectHashTableInitialization(Table);

    //
    // We've completed loading the :Info structure and corresponding data array.
    // Call the algorithm's loader routine to give it a chance to continue
    // initialization.
    //

    Result = LoaderRoutines[AlgorithmId](Table);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashTableLoad, Result);
        goto Error;
    }

    //
    // We're done!  Indicate success and finish up.
    //

    Table->State.Valid = TRUE;
    Table->Flags.Loaded = TRUE;
    Table->TableDataBaseAddress = Table->TableFile->BaseAddress;

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Release file and :Info stream references if applicable.
    //

    RELEASE(Table->TableFile);
    RELEASE(Table->TableInfoStream);

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Release path instances if applicable.
    //

    RELEASE(Path);
    RELEASE(InfoStreamPath);

    ReleasePerfectHashTableLockExclusive(Table);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
