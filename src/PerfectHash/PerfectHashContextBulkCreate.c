/*++

Copyright (c) 2018-2020 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContextBulkCreate.c

Abstract:

    This module implements the bulk-create routine for the perfect hash library.

    N.B. This component is a work in progress.  It is based off the self-test
         component.

--*/

#include "stdafx.h"
#include "BulkCreateCsv.h"


#define PH_ERROR_EX(Name, Result, ...) \
    PH_ERROR(Name, Result)

#define PH_KEYS_ERROR(Name, Result) \
    PH_ERROR(Name, Result)

#define PH_TABLE_ERROR(Name, Result) \
    PH_ERROR(Name, Result)

#if 0
#define PH_ERROR_EX(Name, Result, ...)     \
    PerfectHashPrintErrorEx(#Name,         \
                            __FILE__,      \
                            __FUNCTION__,  \
                            __LINE__,      \
                            (ULONG)Result, \
                            __VA_ARGS__)
#endif

//
// Forward decls.
//

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(STDAPICALLTYPE PREPARE_BULK_CREATE_CSV_FILE)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfKeysFiles
    );
typedef PREPARE_BULK_CREATE_CSV_FILE *PPREPARE_BULK_CREATE_CSV_FILE;
extern PREPARE_BULK_CREATE_CSV_FILE PrepareBulkCreateCsvFile;

//
// Method implementations.
//

PERFECT_HASH_CONTEXT_BULK_CREATE PerfectHashContextBulkCreate;

_Use_decl_annotations_
HRESULT
PerfectHashContextBulkCreate(
    PPERFECT_HASH_CONTEXT Context,
    PCUNICODE_STRING KeysDirectory,
    PCUNICODE_STRING BaseOutputDirectory,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlagsPointer,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlagsPointer,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Attempts to create perfect hash tables for all key files in a directory.

Arguments:

    Context - Supplies an instance of PERFECT_HASH_CONTEXT.

    KeysDirectory - Supplies a pointer to a UNICODE_STRING structure that
        represents a fully-qualified path of the keys directory.

    BaseOutputDirectory - Supplies a pointer to a UNICODE_STRING structure that
        represents a fully-qualified path of the directory where the perfect
        hash table files generated as part of this routine will be saved.

    AlgorithmId - Supplies the algorithm to use.

    HashFunctionId - Supplies the hash function to use.

    MaskFunctionId - Supplies the type of masking to use.

    ContextBulkCreateFlags - Optionally supplies a pointer to a bulk-create flags
        structure that can be used to customize bulk-create behavior.

    KeysLoadFlags - Optionally supplies a pointer to a key loading flags
        structure that can be used to customize key loading behavior.

    TableCreateFlags - Optionally supplies a pointer to a table create flags
        structure that can be used to customize table creation behavior.

    TableCompileFlags - Optionally supplies a pointer to a compile table flags
        structure that can be used to customize table compilation behavior.

    TableCreateParameters - Optionally supplies a pointer to a table create
        parameters struct that can be used to further customize table creation
        behavior.

Return Value:

    S_OK - Success.

    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - One or more mandatory parameters were NULL pointers.

    E_INVALIDARG - KeysDirectory or BaseOutputDirectory were invalid.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_MAXIMUM_CONCURRENCY - Invalid maximum concurrency.

    PH_E_INVALID_KEYS_LOAD_FLAGS - Invalid keys load flags.

    PH_E_INVALID_TABLE_CREATE_FLAGS - Invalid table create flags.

    PH_E_INVALID_TABLE_LOAD_FLAGS - Invalid table load flags.

    PH_E_INVALID_TABLE_COMPILE_FLAGS - Invalid table compile flags.

    PH_E_INVALID_CONTEXT_BULK_CREATE_FLAGS - Invalid context bulk create flags.

    PH_E_NO_KEYS_FOUND_IN_DIRECTORY - No keys found in directory.

--*/
{
    PRTL Rtl;
    PWSTR Dest;
    PWSTR Source;
    ULONG LastError;
    USHORT Length;
    USHORT BaseLength;
    USHORT NumberOfPages;
    ULONG Count = 0;
    ULONG ReferenceCount;
    ULONG NumberOfKeysFiles = 0;
    BOOLEAN Silent;
    BOOLEAN Failed;
    BOOLEAN Terminate;
    HRESULT Result;
    HRESULT TableCreateResult;
    PCHAR Buffer;
    PCHAR BaseBuffer = NULL;
    PCHAR RowBuffer = NULL;
    PALLOCATOR Allocator;
    PVOID KeysBaseAddress;
    ULARGE_INTEGER NumberOfKeys;
    HANDLE FindHandle = NULL;
    HANDLE OutputHandle = NULL;
    HANDLE ProcessHandle = NULL;
    ULONG Failures;
    ULONG KeySizeInBytes;
    ULONGLONG BufferSize;
    ULONGLONG RowBufferSize;
    LONG_INTEGER AllocSize;
    ULONG BytesWritten = 0;
    WIN32_FIND_DATAW FindData;
    UNICODE_STRING WildcardPath;
    UNICODE_STRING KeysPathString;
    LARGE_INTEGER EmptyEndOfFile = { 0 };
    PLARGE_INTEGER EndOfFile;
    PPERFECT_HASH_KEYS Keys = NULL;
    PPERFECT_HASH_TABLE Table = NULL;
    PPERFECT_HASH_FILE CsvFile = NULL;
    PPERFECT_HASH_FILE TableFile = NULL;
    PPERFECT_HASH_PATH TablePath = NULL;
    PPERFECT_HASH_DIRECTORY BaseOutputDir = NULL;
    PERFECT_HASH_KEYS_FLAGS KeysFlags;
    PERFECT_HASH_KEYS_BITMAP KeysBitmap;
    PCUNICODE_STRING Suffix = &KeysWildcardSuffix;
    PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags;
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags;
    PERFECT_HASH_CPU_ARCH_ID CpuArchId;
    ASSIGNED_MEMORY_COVERAGE EmptyCoverage;
    PASSIGNED_MEMORY_COVERAGE Coverage;
    BOOLEAN UnknownTableCreateResult = FALSE;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    } else {
        Rtl = Context->Rtl;
        Allocator = Context->Allocator;
    }

    VALIDATE_FLAGS(ContextBulkCreate, CONTEXT_BULK_CREATE);
    VALIDATE_FLAGS(KeysLoad, KEYS_LOAD);
    VALIDATE_FLAGS(TableCreate, TABLE_CREATE);
    VALIDATE_FLAGS(TableCompile, TABLE_COMPILE);

    if (!ARGUMENT_PRESENT(KeysDirectory)) {
        return E_POINTER;
    } else if (!IsValidMinimumDirectoryUnicodeString(KeysDirectory)) {
        return E_INVALIDARG;
    }

    if (!ARGUMENT_PRESENT(BaseOutputDirectory)) {
        return E_POINTER;
    } else if (!IsValidMinimumDirectoryUnicodeString(BaseOutputDirectory)) {
        return E_INVALIDARG;
    }
    else {
        Result = Context->Vtbl->SetBaseOutputDirectory(Context,
                                                       BaseOutputDirectory);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashContextSetBaseOutputDirectory, Result);
            return Result;
        }

        Result = Context->Vtbl->GetBaseOutputDirectory(Context, &BaseOutputDir);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashContextGetBaseOutputDirectory, Result);
            return Result;
        }
        RELEASE(BaseOutputDir);
    }

    if (!IsValidPerfectHashAlgorithmId(AlgorithmId)) {
        return E_INVALIDARG;
    }

    if (!IsValidPerfectHashHashFunctionId(HashFunctionId)) {
        return E_INVALIDARG;
    }

    if (!IsValidPerfectHashMaskFunctionId(MaskFunctionId)) {
        return E_INVALIDARG;
    }

    //
    // Arguments have been validated, proceed.
    //

    Silent = (TableCreateFlags.Silent == TRUE);
    ZeroStruct(EmptyCoverage);

    //
    // Create a buffer we can use for temporary path construction.  We want it
    // to be MAX_USHORT in size, so (1 << 16) >> PAGE_SHIFT converts this into
    // the number of pages we need.
    //

    NumberOfPages = (1 << 16) >> PAGE_SHIFT;

    Result = Rtl->Vtbl->CreateBuffer(Rtl,
                                     &ProcessHandle,
                                     NumberOfPages,
                                     NULL,
                                     &BufferSize,
                                     &BaseBuffer);

    if (FAILED(Result)) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    Buffer = BaseBuffer;

    //
    // Create a "row buffer" we can use for the CSV file.
    //

    NumberOfPages = 2;

    Result = Rtl->Vtbl->CreateBuffer(Rtl,
                                     &ProcessHandle,
                                     NumberOfPages,
                                     NULL,
                                     &RowBufferSize,
                                     &RowBuffer);

    if (FAILED(Result)) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    SetContextBulkCreate(Context);

    Context->RowBuffer = Context->BaseRowBuffer = RowBuffer;
    Context->RowBufferSize = RowBufferSize;

    //
    // Get a reference to the stdout handle.
    //

    if (!Context->OutputHandle) {
        Context->OutputHandle = GetStdHandle(STD_OUTPUT_HANDLE);
        if (!Context->OutputHandle) {
            SYS_ERROR(GetStdHandle);
            goto Error;
        }
    }

    OutputHandle = Context->OutputHandle;

    //
    // Calculate the size required for a new concatenated wide string buffer
    // that combines the test data directory with the "*.keys" suffix.  The
    // 2 * sizeof(*Dest) accounts for the joining slash and trailing NULL.
    //

    AllocSize.LongPart = KeysDirectory->Length;
    AllocSize.LongPart += Suffix->Length + (2 * sizeof(*Dest));

    //
    // Sanity check we haven't overflowed.
    //

    if (AllocSize.HighPart) {
        Result = PH_E_STRING_BUFFER_OVERFLOW;
        PH_ERROR(PerfectHashContextBulkCreate_AllocSize, Result);
        goto Error;
    }

    WildcardPath.Buffer = (PWSTR)Buffer;

    if (!WildcardPath.Buffer) {
        goto Error;
    }

    //
    // Copy incoming keys directory name.
    //

    Length = KeysDirectory->Length;
    CopyInline(WildcardPath.Buffer,
               KeysDirectory->Buffer,
               Length);

    //
    // Advance our Dest pointer to the end of the directory name, write a
    // slash, then copy the suffix over.
    //

    Dest = (PWSTR)RtlOffsetToPointer(WildcardPath.Buffer, Length);
    *Dest++ = L'\\';
    CopyInline(Dest, Suffix->Buffer, Suffix->Length);

    //
    // Wire up the search path length and maximum length variables.  The max
    // length will be our AllocSize, length will be this value minus 2 to
    // account for the trailing NULL.
    //

    WildcardPath.MaximumLength = AllocSize.LowPart;
    WildcardPath.Length = AllocSize.LowPart - sizeof(*Dest);
    ASSERT(WildcardPath.Buffer[WildcardPath.Length] == L'\0');

    //
    // Advance the buffer past this string allocation, up to the next 16-byte
    // boundary.
    //

    Buffer = (PSTR)(
        RtlOffsetToPointer(
            Buffer,
            ALIGN_UP(WildcardPath.MaximumLength, 16)
        )
    );

    //
    // Create a find handle for the <keys dir>\*.keys search pattern we
    // created.  Note that we actually do this twice; the first time, below,
    // is used to count the number of *.keys file in the target directory,
    // which allows us to size our .csv file relative to the number rows we'll
    // be appending to it.
    //

    FindHandle = FindFirstFileW(WildcardPath.Buffer, &FindData);

    if (!IsValidHandle(FindHandle)) {

        //
        // Check to see if we failed because there were no files matching the
        // wildcard *.keys in the test directory.  In this case, GetLastError()
        // will report ERROR_FILE_NOT_FOUND.
        //

        FindHandle = NULL;
        LastError = GetLastError();

        if (LastError == ERROR_FILE_NOT_FOUND) {
            Result = PH_E_NO_KEYS_FOUND_IN_DIRECTORY;
            PH_MESSAGE(Result);
        } else {
            SYS_ERROR(FindFirstFileW);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }

        goto Error;
    }

    //
    // Count the number of keys files in the directory.
    //

    NumberOfKeysFiles = 0;

    do {
        NumberOfKeysFiles++;
    } while (FindNextFile(FindHandle, &FindData));

    //
    // Sanity check we saw at least one *.keys file.
    //

    if (NumberOfKeysFiles == 0) {
        Result = PH_E_NO_KEYS_FOUND_IN_DIRECTORY;
        PH_MESSAGE(Result);
        goto Error;
    }

    //
    // Close the find handle, then create a new one.
    //

    if (!FindClose(FindHandle)) {
        SYS_ERROR(FindClose);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Now that we know how many *.keys files to expect in the target directory,
    // ignoring the unlikely case where keys are being added/deleted from the
    // directory whilst this routine is running, we can prepare our .csv file,
    // if applicable.
    //

    if (TableCreateFlags.DisableCsvOutputFile == FALSE) {
        Result = PrepareBulkCreateCsvFile(Context, NumberOfKeysFiles);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashContextBulkCreate_PrepareCsvFile, Result);
            goto Error;
        }
        CsvFile = Context->BulkCreateCsvFile;
    }

    //
    // Create a new file handle.  This is the one we'll use to drive the
    // table create operations.
    //

    FindHandle = FindFirstFileW(WildcardPath.Buffer, &FindData);

    if (!IsValidHandle(FindHandle)) {

        //
        // Duplicate the error handling logic from above.
        //

        FindHandle = NULL;
        LastError = GetLastError();

        if (LastError == ERROR_FILE_NOT_FOUND) {
            Result = PH_E_NO_KEYS_FOUND_IN_DIRECTORY;
            PH_MESSAGE(Result);
        } else {
            SYS_ERROR(FindFirstFileW);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }

        goto Error;
    }

    //
    // Initialize the fully-qualified keys path.
    //

    KeysPathString.Buffer = (PWSTR)Buffer;
    CopyInline(KeysPathString.Buffer, KeysDirectory->Buffer, Length);

    //
    // Advance our Dest pointer to the end of the directory name, then write
    // a slash.
    //

    Dest = (PWSTR)RtlOffsetToPointer(KeysPathString.Buffer, Length);
    *Dest++ = L'\\';

    //
    // Update the length to account for the slash we just wrote, then make a
    // copy of it in the variable BaseLength.
    //

    Length += sizeof(*Dest);
    BaseLength = Length;

    //
    // Initialize the key size based on keys load flags or table create params.
    // We pass this value to Keys->Vtbl->Load().
    //

    Result = PerfectHashContextInitializeKeySize(&KeysLoadFlags,
                                                 TableCreateParameters,
                                                 &KeySizeInBytes);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashContextInitializeKeySize, Result);
        goto Error;
    }

    //
    // Initialize local variables and then begin the main loop.
    //

    Failures = 0;
    Terminate = FALSE;
    ZeroStruct(KeysBitmap);

    Table = NULL;
    KeysBaseAddress = NULL;
    NumberOfKeys.QuadPart = 0;
    CpuArchId = PerfectHashGetCurrentCpuArch();

    //
    // If we've been asked to skip testing after creation, we can toggle the
    // CreateOnly flag in order to avoid some additional memory allocation
    // and copying overhead.
    //

    if (ContextBulkCreateFlags.SkipTestAfterCreate) {
        TableCreateFlags.CreateOnly = TRUE;
    }

    do {

        //
        // Helper macro for testing if Ctrl-C has been pressed.
        //

#define CHECK_CTRL_C()                \
    if (CtrlCPressed) {               \
        Failures++;                   \
        Terminate = TRUE;             \
        Result = PH_E_CTRL_C_PRESSED; \
        goto ReleaseTable;            \
    }

        Count++;

        //
        // Clear the failure flag at the top of every loop invocation.
        //

        Failed = FALSE;

        //
        // Copy the filename over to the fully-qualified keys path.
        //

        Dest = (PWSTR)RtlOffsetToPointer(KeysPathString.Buffer, BaseLength);
        Source = (PWSTR)FindData.cFileName;

        while (*Source) {
            *Dest++ = *Source++;
        }
        *Dest = L'\0';

        Length = (USHORT)RtlPointerToOffset(KeysPathString.Buffer, Dest);
        KeysPathString.Length = Length;
        KeysPathString.MaximumLength = Length + sizeof(*Dest);
        ASSERT(KeysPathString.Buffer[KeysPathString.Length >> 1] == L'\0');
        ASSERT(&KeysPathString.Buffer[KeysPathString.Length >> 1] == Dest);

        Result = Context->Vtbl->CreateInstance(Context,
                                               NULL,
                                               &IID_PERFECT_HASH_KEYS,
                                               &Keys);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashKeysCreateInstance, Result);
            Failures++;
            break;
        }

        Result = Keys->Vtbl->Load(Keys,
                                  &KeysLoadFlags,
                                  &KeysPathString,
                                  KeySizeInBytes);

        if (FAILED(Result)) {
            PH_KEYS_ERROR(PerfectHashKeysLoad, Result);
            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        Result = Keys->Vtbl->GetFlags(Keys,
                                      sizeof(KeysFlags),
                                      &KeysFlags);

        if (FAILED(Result)) {
            PH_KEYS_ERROR(PerfectHashKeysGetFlags, Result);
            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        Result = Keys->Vtbl->GetAddress(Keys,
                                        &KeysBaseAddress,
                                        &NumberOfKeys);

        if (FAILED(Result)) {
            PH_KEYS_ERROR(PerfectHashKeysGetAddress, Result);
            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        //
        // Keys were loaded successfully.  Proceed with table creation.
        //

        ASSERT(Table == NULL);

        Result = Context->Vtbl->CreateInstance(Context,
                                               NULL,
                                               &IID_PERFECT_HASH_TABLE,
                                               &Table);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashTableCreateInstance, Result);
            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        Result = Table->Vtbl->Create(Table,
                                     Context,
                                     AlgorithmId,
                                     HashFunctionId,
                                     MaskFunctionId,
                                     Keys,
                                     &TableCreateFlags,
                                     TableCreateParameters);

        TableCreateResult = Result;

        CHECK_CTRL_C();

        if (FAILED(Result)) {
            PH_KEYS_ERROR(PerfectHashTableCreate, Result);
            Failed = TRUE;
            Failures++;
            goto ReleaseTable;
        }

        PRINT_CHAR_FOR_TABLE_CREATE_RESULT(Result);

        if (Result != S_OK) {

            Coverage = &EmptyCoverage;

        } else {

            Coverage = Table->Coverage;

            //
            // Test the table, if applicable.
            //

            if (!ContextBulkCreateFlags.SkipTestAfterCreate) {

                Result = Table->Vtbl->Test(Table, Keys, FALSE);

                if (FAILED(Result)) {
                    PH_TABLE_ERROR(PerfectHashTableTest, Result);
                    Failed = TRUE;
                    Failures++;
                    goto ReleaseTable;
                }
            }

            CHECK_CTRL_C();

            //
            // Compile the table.
            //

            if (ContextBulkCreateFlags.Compile) {

                Result = Table->Vtbl->Compile(Table,
                                              &TableCompileFlags,
                                              CpuArchId);

                if (FAILED(Result)) {
                    PH_TABLE_ERROR(PerfectHashTableCompile, Result);
                    Failures++;
                    Failed = TRUE;
                    goto ReleaseTable;
                }
            }

            CHECK_CTRL_C();

        }

        //
        // Write the .csv row if applicable.
        //

        if (TableCreateFlags.DisableCsvOutputFile != FALSE) {
            goto ReleaseTable;
        }

        if (SkipWritingCsvRow(TableCreateFlags, TableCreateResult)) {
            goto ReleaseTable;
        }

        _Analysis_assume_(CsvFile != NULL);

        //
        // N.B. The SAL annotations are required to suppress the concurrency
        //      warnings for accessing the Context->NewBestGraphCount and
        //      Context->EqualBestGraphCount members outside of the best graph
        //      critical section.
        //

        _No_competing_thread_begin_
        WRITE_BULK_CREATE_CSV_ROW();
        _No_competing_thread_end_

    ReleaseTable:

        //
        // Release the table.
        //

        ReferenceCount = Table->Vtbl->Release(Table);
        Table = NULL;

        if (ReferenceCount != 0) {
            PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        }

        //
        // Release the table path and file.
        //

        RELEASE(TablePath);
        RELEASE(TableFile);

        //
        // Intentional follow-on to ReleaseKeys.
        //

    ReleaseKeys:

        RELEASE(Keys);

        if (Terminate || CtrlCPressed) {
            break;
        }

    } while (FindNextFile(FindHandle, &FindData));

    //
    // Bulk create complete!
    //

    NEWLINE();

    if ((!Failures && !Terminate) || CtrlCPressed) {
        Result = S_OK;
        goto End;
    }

    //
    // Intentional follow-on to Error.
    //

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Release any references we still hold.
    //

    RELEASE(Keys);
    RELEASE(Table);
    RELEASE(TablePath);
    RELEASE(TableFile);


    if (RowBuffer) {
        ASSERT(Context->RowBuffer);
        Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                          ProcessHandle,
                                          &RowBuffer);
        if (FAILED(Result)) {
            SYS_ERROR(VirtualFree);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
        Context->RowBuffer = RowBuffer = NULL;
    }

    if (BaseBuffer) {
        Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                          ProcessHandle,
                                          &BaseBuffer);
        if (FAILED(Result)) {
            SYS_ERROR(VirtualFree);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
        BaseBuffer = NULL;
    }

    if (FindHandle) {
        if (!FindClose(FindHandle)) {
            SYS_ERROR(FindClose);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
        FindHandle = NULL;
    }

    ClearContextBulkCreate(Context);

    //
    // Close the .csv file.  If we encountered an error, use 0 as end-of-file,
    // which will cause the Close() call to restore it to its original state
    // (including potentially deleting it if it didn't exist when we opened it).
    //

    if (Result != S_OK && !CtrlCPressed) {
        EndOfFile = &EmptyEndOfFile;
    } else {
        EndOfFile = NULL;
    }

    if (CsvFile) {
        Result = CsvFile->Vtbl->Close(CsvFile, EndOfFile);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashBulkCreate_CloseCsvFile, Result);
        }
    }

    RELEASE(Context->BulkCreateCsvFile);

    return Result;
}


PREPARE_BULK_CREATE_CSV_FILE PrepareBulkCreateCsvFile;

_Use_decl_annotations_
HRESULT
PrepareBulkCreateCsvFile(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfKeysFiles
    )
/*++

Routine Description:

    Prepares the <BaseOutputDir>\PerfectHashBulkCreate_<HeaderHash>.csv file.
    This involves determining the header hash, constructing a path instance,
    creating a file instance, and opening it for append.

Arguments:

    Context - Supplies the context for which the .csv file is to be prepared.

    NumberOfKeysFiles - Supplies the number of keys files anticipated to be
        processed during the bulk create operation.  This is used to derive
        an appropriate file size to use for the .csv file.

Return Value:

    S_OK on success, an appropriate error code otherwise.

--*/
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    PWCHAR WideOutput;
    STRING Header;
    STRING HexHash;
    UNICODE_STRING Suffix;
    HRESULT Result = S_OK;
    ULONG_PTR BufferBytesConsumed;
    PPERFECT_HASH_PATH ExistingPath;
    PPERFECT_HASH_PATH Path = NULL;
    PPERFECT_HASH_FILE File = NULL;
    LARGE_INTEGER EndOfFile;
    PCUNICODE_STRING BaseName;
    PCUNICODE_STRING NewDirectory;
    PERFECT_HASH_FILE_CREATE_FLAGS FileCreateFlags = { 0 };

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Header.Buffer = Base = Output = Context->BaseRowBuffer;

    //
    // Construct an ASCII representation of all column names concatenated
    // together, wire up a STRING structure, then generate a hash.
    //

#define EXPAND_AS_COLUMN_NAME_THEN_COMMA(Name, Value, OutputMacro) \
    OUTPUT_RAW(#Name);                                             \
    OUTPUT_CHR(',');

#define EXPAND_AS_COLUMN_NAME_THEN_NEWLINE(Name, Value, OutputMacro) \
    OUTPUT_RAW(#Name);                                               \
    OUTPUT_CHR('\n');

    BULK_CREATE_CSV_ROW_TABLE(EXPAND_AS_COLUMN_NAME_THEN_COMMA,
                              EXPAND_AS_COLUMN_NAME_THEN_COMMA,
                              EXPAND_AS_COLUMN_NAME_THEN_NEWLINE);

    Header.Length = (USHORT)RtlPointerToOffset(Base, Output);
    Header.MaximumLength = Header.Length;

    HashString(&Header);

    //
    // Convert the hash into an ASCII hex representation, then wire it up into
    // a STRING structure.
    //

    Output = (PSTR)ALIGN_UP(Output, 8);
    HexHash.Buffer = Base = Output;

    *Output++ = '_';

    AppendIntegerToCharBufferAsHexRaw(&Output, Header.Hash);

    HexHash.Length = (USHORT)RtlPointerToOffset(Base, Output);
    HexHash.MaximumLength = HexHash.Length;

    //
    // Convert the ASCII hex representation into a wide character version.
    //

    Base = Output = (PSTR)ALIGN_UP(Output, 8);
    Suffix.Buffer = WideOutput = (PWSTR)Output;

    AppendStringToWideCharBufferFast(&WideOutput, &HexHash);

    Output = (PSTR)WideOutput;

    Suffix.Length = (USHORT)RtlPointerToOffset(Base, Output);
    Suffix.MaximumLength = Suffix.Length;

    //
    // Capture the total number of bytes of the row buffer we consumed at this
    // point such that we can zero the memory at the end of this routine.
    //

    BufferBytesConsumed = RtlPointerToOffset(Context->BaseRowBuffer, Output);

    //
    // Create a path instance.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_PATH,
                                           &Path);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCreateInstance, Result);
        goto Error;
    }

    //
    // Create the .csv file's path name.
    //

    BaseName = &PerfectHashBulkCreateCsvBaseName;
    ExistingPath = Context->BaseOutputDirectory->Path;
    NewDirectory = &Context->BaseOutputDirectory->Path->FullPath;

    Result = Path->Vtbl->Create(Path,
                                ExistingPath,
                                NewDirectory,   // NewDirectory
                                NULL,           // DirectorySuffix
                                BaseName,       // NewBaseName
                                &Suffix,        // BaseNameSuffix
                                &CsvExtension,  // NewExtension
                                NULL,           // NewStreamName
                                NULL,           // Parts
                                NULL);          // Reserved

    if (FAILED(Result)) {
        PH_ERROR(PrepareBulkCreateCsvFile_PathCreate, Result);
        goto Error;
    }

    //
    // Create a file instance.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_FILE,
                                           &File);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileCreateInstance, Result);
        goto Error;
    }

    //
    // Heuristically derive an appropriate file size (EndOfFile) to use based
    // on the length of the header, plus 32 bytes, aligned up to a 32-byte
    // boundary, then multiplied by the number of keys files.  We then add the
    // system allocation size, then align the final amount up to this boundary.
    //
    // N.B. The resulting file size is very generous; we shouldn't ever come
    //      close to hitting it in normal operating conditions.  If a large
    //      number of keys were added to the directory after we did our initial
    //      count, worst case scenario is that we'll eventually trap when
    //      writing rows to the .csv file.  The solution to this is: don't
    //      add large numbers of keys files to the directory when running a
    //      bulk create operation.
    //

    EndOfFile.QuadPart = (
        ALIGN_UP((ULONG_PTR)Header.Length + 32, 32) *
        (ULONG_PTR)NumberOfKeysFiles
    );
    EndOfFile.QuadPart += Context->SystemAllocationGranularity;
    EndOfFile.QuadPart = ALIGN_UP(EndOfFile.QuadPart,
                                  Context->SystemAllocationGranularity);

    FileCreateFlags.NoTruncate = TRUE;
    FileCreateFlags.EndOfFileIsExtensionSizeIfFileExists = TRUE;

    Result = File->Vtbl->Create(File,
                                Path,
                                &EndOfFile,
                                NULL,
                                &FileCreateFlags);

    if (FAILED(Result)) {
        PH_ERROR(PrepareBulkCreateCsvFile_FileCreate, Result);
        goto Error;
    }

    if (File->NumberOfBytesWritten.QuadPart > 0) {
        SIZE_T BytesMatched;
        BOOLEAN HeaderMatches;

        //
        // Compare the on-disk CSV header to the header we just constructed.
        // This should always match unless there's a hash collision between
        // two header strings, which will hopefully be very unlikely.
        //

        BytesMatched = Rtl->RtlCompareMemory(Header.Buffer,
                                             File->BaseAddress,
                                             Header.Length);

        HeaderMatches = (BytesMatched == Header.Length);

        if (!HeaderMatches) {
            Result = PH_E_BULK_CREATE_CSV_HEADER_MISMATCH;
            goto Error;
        }

    } else {

        //
        // Write the header to the newly created file and update the number of
        // bytes written.
        //

        CopyMemory(File->BaseAddress,
                   Header.Buffer,
                   Header.Length);

        File->NumberOfBytesWritten.QuadPart += Header.Length;

    }

    //
    // Capture the file pointer in the context and add a reference to offset
    // the RELEASE(File) at the end of the routine.
    //

    Context->BulkCreateCsvFile = File;
    File->Vtbl->AddRef(File);

    //
    // Capture the header hash in the context.
    //

    Context->HexHeaderHash.Length = sizeof(Context->HexHeaderHashBuffer);
    Context->HexHeaderHash.MaximumLength = Context->HexHeaderHash.Length;
    ASSERT(Context->HexHeaderHash.Length >= HexHash.Length - 1);

    ZeroMemory(&Context->HexHeaderHashBuffer, Context->HexHeaderHash.Length);
    CopyMemory(Context->HexHeaderHashBuffer,
               HexHash.Buffer + 1,
               HexHash.Length - 1);

    Context->HexHeaderHash.Buffer = (PCHAR)&Context->HexHeaderHashBuffer;

    //
    // We're done, finish up.
    //

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
    // Clear the bytes we wrote, release the path if applicable, then return.
    //

    ZeroMemory(Context->BaseRowBuffer, BufferBytesConsumed);

    RELEASE(Path);
    RELEASE(File);

    return Result;
}

//
// Helper macros for argument extraction.
//

#define GET_LENGTH(Name) (USHORT)wcslen(Name->Buffer) << (USHORT)1
#define GET_MAX_LENGTH(Name) Name->Length + 2

#define VALIDATE_ID(Name, Upper)                                       \
    if (FAILED(Rtl->RtlUnicodeStringToInteger(String,                  \
                                              10,                      \
                                              (PULONG)##Name##Id))) {  \
        return PH_E_INVALID_##Upper##_ID;                              \
    } else if (*##Name##Id == 0) {                                     \
        Result = PerfectHashLookupIdForName(Rtl,                       \
                                            PerfectHash##Name##EnumId, \
                                            String,                    \
                                            (PULONG)##Name##Id);       \
        if (FAILED(Result)) {                                          \
            return PH_E_INVALID_##Upper##_ID;                          \
        }                                                              \
    }                                                                  \
    if (!IsValidPerfectHash##Name##Id(*##Name##Id)) {                  \
        return PH_E_INVALID_##Upper##_ID;                              \
    }

#define EXTRACT_ID(Name, Upper)                     \
    CurrentArg++;                                   \
    String->Buffer = *ArgW++;                       \
    String->Length = GET_LENGTH(String);            \
    String->MaximumLength = GET_MAX_LENGTH(String); \
    VALIDATE_ID(Name, Upper)

PERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextExtractBulkCreateArgsFromArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextExtractBulkCreateArgsFromArgvW(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW,
    PUNICODE_STRING KeysDirectory,
    PUNICODE_STRING BaseOutputDirectory,
    PPERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PPERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PULONG MaximumConcurrency,
    PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Extracts arguments for the bulk-create functionality from an argument vector
    array, typically obtained from a commandline invocation.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance
        for which the arguments are to be extracted.

    NumberOfArguments - Supplies the number of elements in the ArgvW array.

    ArgvW - Supplies a pointer to an array of wide C string arguments.

    KeysDirectory - Supplies a pointer to a UNICODE_STRING structure that
        will be filled out with the keys directory.

    BaseOutputDirectory - Supplies a pointer to a UNICODE_STRING structure that
        will be filled out with the output directory.

    AlgorithmId - Supplies the address of a variable that will receive the
        algorithm ID.

    HashFunctionId - Supplies the address of a variable that will receive the
        hash function ID.

    MaskFunctionId - Supplies the address of a variable that will receive the
        mask function ID.

    MaximumConcurrency - Supplies the address of a variable that will receive
        the maximum concurrency.

    ContextBulkCreateFlags - Supplies the address of a variable that will
        receive the bulk-create flags.

    KeysLoadFlags - Supplies the address of a variable that will receive
        the keys load flags.

    TableCreateFlags - Supplies the address of a variable that will receive
        the table create flags.

    TableLoadFlags - Supplies the address of a variable that will receive the
        the load table flags.

    TableCompileFlags - Supplies the address of a variable that will receive
        the table compile flags.

    TableCreateParameters - Supplies the address of a variable that will receive
        a pointer to a table create parameters structure.

Return Value:

    S_OK - Arguments extracted successfully.

    E_POINTER - One or more mandatory parameters were NULL pointers.

    PH_E_CONTEXT_BULK_CREATE_INVALID_NUM_ARGS - Invalid number of arguments.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_MAXIMUM_CONCURRENCY - Invalid maximum concurrency.

--*/
{
    PRTL Rtl;
    LPWSTR *ArgW;
    LPWSTR Arg;
    HRESULT Result = S_OK;
    HRESULT CleanupResult;
    ULONG CurrentArg = 1;
    PALLOCATOR Allocator;
    UNICODE_STRING Temp;
    PUNICODE_STRING String;
    BOOLEAN InvalidPrefix;
    BOOLEAN ValidNumberOfArguments;

    String = &Temp;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(ArgvW)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(KeysDirectory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(BaseOutputDirectory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(AlgorithmId)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(HashFunctionId)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(MaskFunctionId)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(MaximumConcurrency)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(ContextBulkCreateFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(KeysLoadFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCreateFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCompileFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCreateParameters)) {
        return E_POINTER;
    }

    ValidNumberOfArguments = (
        NumberOfArguments >= 7
    );

    if (!ValidNumberOfArguments) {
        return PH_E_CONTEXT_BULK_CREATE_INVALID_NUM_ARGS;
    }

    //
    // Argument validation complete, continue.
    //

    ArgW = &ArgvW[1];
    Rtl = Context->Rtl;
    Allocator = Context->Allocator;

    //
    // The first six arguments (keys directory, base output directory, algo ID,
    // hash function ID, mask function ID and maximum concurrency) are special
    // in that they're mandatory and expected to appear sequentially, prior to
    // any additional arguments (i.e. table create parameters) appearing.
    //

    //
    // Extract keys directory.
    //

    CurrentArg++;
    KeysDirectory->Buffer = *ArgW++;
    KeysDirectory->Length = GET_LENGTH(KeysDirectory);
    KeysDirectory->MaximumLength = GET_MAX_LENGTH(KeysDirectory);

    //
    // Extract base output directory.
    //

    CurrentArg++;
    BaseOutputDirectory->Buffer = *ArgW++;
    BaseOutputDirectory->Length = GET_LENGTH(BaseOutputDirectory);
    BaseOutputDirectory->MaximumLength = GET_MAX_LENGTH(BaseOutputDirectory);

    //
    // Originally, the numerical value for algo, hash function and masking type
    // was required.  We've since relaxed that to support both the numerical
    // value (i.e. for hash function, you can supply either 1 or Crc32Rotate15).
    // Extract these values now.
    //

    EXTRACT_ID(Algorithm, ALGORITHM);
    EXTRACT_ID(HashFunction, HASH_FUNCTION);
    EXTRACT_ID(MaskFunction, MASK_FUNCTION);

    //
    // Extract maximum concurrency.
    //

    CurrentArg++;
    String->Buffer = *ArgW++;
    String->Length = GET_LENGTH(String);
    String->MaximumLength = GET_MAX_LENGTH(String);

    if (FAILED(Rtl->RtlUnicodeStringToInteger(String,
                                              10,
                                              MaximumConcurrency))) {
        return PH_E_INVALID_MAXIMUM_CONCURRENCY;
    }

    //
    // Zero all flags and table create parameters.
    //

    ContextBulkCreateFlags->AsULong = 0;
    KeysLoadFlags->AsULong = 0;
    TableCreateFlags->AsULong = 0;
    TableCompileFlags->AsULong = 0;

    for (; CurrentArg < NumberOfArguments; CurrentArg++, ArgW++) {

        String->Buffer = Arg = *ArgW;
        String->Length = GET_LENGTH(String);
        String->MaximumLength = GET_MAX_LENGTH(String);

        //
        // If the argument doesn't start with two dashes, report it.
        //

        InvalidPrefix = (
            (String->Length <= (sizeof(L'-') + sizeof(L'-'))) ||
            (!(*Arg++ == L'-' && *Arg++ == L'-'))
        );

        if (InvalidPrefix) {
            goto InvalidArg;
        }

        //
        // Advance the buffer past the two dashes and update lengths
        // accordingly.
        //

        String->Buffer += 2;
        String->Length -= 4;
        String->MaximumLength -= 4;

        //
        // Try each argument extraction routine for this argument; if it
        // indicates an error, report it and break out of the loop.  If it
        // indicates it successfully extracted the argument (Result == S_OK),
        // continue onto the next argument.  Otherwise, verify it indicates
        // that no argument was extracted (S_FALSE), then try the next routine.
        //

#define TRY_EXTRACT_ARG(Name)                                                 \
    Result = TryExtractArg##Name##Flags(Rtl, Allocator, String, Name##Flags); \
    if (FAILED(Result)) {                                                     \
        PH_ERROR(ExtractBulkCreateArgs_TryExtractArg##Name##Flags, Result);   \
        break;                                                                \
    } else if (Result == S_OK) {                                              \
        continue;                                                             \
    } else {                                                                  \
        ASSERT(Result == S_FALSE);                                            \
    }

        TRY_EXTRACT_ARG(ContextBulkCreate);
        TRY_EXTRACT_ARG(KeysLoad);
        TRY_EXTRACT_ARG(TableCreate);
        TRY_EXTRACT_ARG(TableCompile);

        //
        // If we get here, none of the previous extraction routines claimed the
        // argument, so, provide the table create parameters extraction routine
        // an opportunity to run.
        //

        Result = TryExtractArgTableCreateParameters(Rtl,
                                                    String,
                                                    TableCreateParameters);

        if (FAILED(Result)) {
            PH_ERROR(ExtractBulkCreateArgs_TryExtractTableCreateParams, Result);
            break;
        }

        if (Result == S_OK) {
            continue;
        }

        if (Result == PH_E_COMMANDLINE_ARG_MISSING_VALUE) {
            PH_MESSAGE(Result, String);
            break;
        }

        if (FAILED(Result)) {
            PH_ERROR(ExtractBulkCreateArgs_TryExtractTableCreateParams, Result);
            break;
        }

        ASSERT(Result == S_FALSE);

InvalidArg:

        //
        // If we get here, we don't recognize the argument.
        //


        Result = PH_E_INVALID_COMMANDLINE_ARG;
        PH_MESSAGE(Result, String);
        break;
    }

    //
    // If we failed, clean up the table create parameters.  If that fails,
    // report the error, then replace our return value error code with that
    // error code.
    //

    if (FAILED(Result)) {
        CleanupResult = CleanupTableCreateParameters(TableCreateParameters);
        if (FAILED(CleanupResult)) {
            PH_ERROR(CleanupTableCreateParameters, CleanupResult);
            Result = CleanupResult;
        }
    }

    return Result;
}

PERFECT_HASH_CONTEXT_BULK_CREATE_ARGVW PerfectHashContextBulkCreateArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextBulkCreateArgvW(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW
    )
/*++

Routine Description:

    Extracts arguments for the bulk-create functionality from an argument vector
    array and then invokes the context bulk-create functionality.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance
        for which the arguments are to be extracted.

    NumberOfArguments - Supplies the number of elements in the ArgvW array.

    ArgvW - Supplies a pointer to an array of wide C string arguments.

Return Value:

    S_OK - Arguments extracted successfully.

    E_POINTER - One or more mandatory parameters were NULL pointers.

    PH_E_CONTEXT_BULK_CREATE_INVALID_NUM_ARGS - Invalid number of arguments.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_MAXIMUM_CONCURRENCY - Invalid maximum concurrency.

    PH_E_INVALID_CONTEXT_BULK_CREATE_FLAGS - Invalid bulk create flags.

    PH_E_INVALID_KEYS_LOAD_FLAGS - Invalid keys load flags.

    PH_E_INVALID_TABLE_LOAD_FLAGS - Invalid table load flags.

    PH_E_INVALID_TABLE_COMPILE_FLAGS - Invalid table compile flags.

--*/
{
    HRESULT Result;
    HRESULT CleanupResult;
    UNICODE_STRING KeysDirectory = { 0 };
    UNICODE_STRING BaseOutputDirectory = { 0 };
    PERFECT_HASH_ALGORITHM_ID AlgorithmId = 0;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId = 0;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId = 0;
    ULONG MaximumConcurrency = 0;
    PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags = { 0 };
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags = { 0 };
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags = { 0 };
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags = { 0 };
    PPERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW
        ExtractBulkCreateArgs;
    PERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters;

    TableCreateParameters.SizeOfStruct = sizeof(TableCreateParameters);
    TableCreateParameters.NumberOfElements = 0;
    TableCreateParameters.Allocator = Context->Allocator;
    TableCreateParameters.Params = NULL;

    ExtractBulkCreateArgs = Context->Vtbl->ExtractBulkCreateArgsFromArgvW;
    Result = ExtractBulkCreateArgs(Context,
                                   NumberOfArguments,
                                   ArgvW,
                                   &KeysDirectory,
                                   &BaseOutputDirectory,
                                   &AlgorithmId,
                                   &HashFunctionId,
                                   &MaskFunctionId,
                                   &MaximumConcurrency,
                                   &ContextBulkCreateFlags,
                                   &KeysLoadFlags,
                                   &TableCreateFlags,
                                   &TableCompileFlags,
                                   &TableCreateParameters);

    if (FAILED(Result)) {
        return Result;
    }

    if (MaximumConcurrency > 0) {
        Result = Context->Vtbl->SetMaximumConcurrency(Context,
                                                      MaximumConcurrency);
        if (FAILED(Result)) {
            Result = PH_E_SET_MAXIMUM_CONCURRENCY_FAILED;
            PH_ERROR(PerfectHashContextContextBulkCreateArgvW, Result);
            return Result;
        }
    }

    PerfectHashContextApplyThreadpoolPriorities(Context,
                                                &TableCreateParameters);

    Result = Context->Vtbl->BulkCreate(Context,
                                       &KeysDirectory,
                                       &BaseOutputDirectory,
                                       AlgorithmId,
                                       HashFunctionId,
                                       MaskFunctionId,
                                       &ContextBulkCreateFlags,
                                       &KeysLoadFlags,
                                       &TableCreateFlags,
                                       &TableCompileFlags,
                                       &TableCreateParameters);

    if (FAILED(Result)) {

        //
        // There's is nothing we can do here.  We don't PH_ERROR() the return
        // code as BulkCreate() will have done that multiple times each time
        // the error bubbled back up the stack.
        //

        NOTHING;
    }

    CleanupResult = CleanupTableCreateParameters(&TableCreateParameters);
    if (FAILED(CleanupResult)) {
        PH_ERROR(BulkCreateArgvW_CleanupTableCreateParams, CleanupResult);
        Result = CleanupResult;
    }

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
