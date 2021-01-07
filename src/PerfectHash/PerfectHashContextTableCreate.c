/*++

Copyright (c) 2018-2019 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContextTableCreate.c

Abstract:

    This module implements the context's table create routine for the perfect
    hash library.  It is analogous to the bulk create functionality, except it
    operates on a single keys file instead of a directory of keys.

--*/

#include "stdafx.h"
#include "TableCreateCsv.h"


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
(STDAPICALLTYPE PREPARE_TABLE_CREATE_CSV_FILE)(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG NumberOfRows
    );
typedef PREPARE_TABLE_CREATE_CSV_FILE *PPREPARE_TABLE_CREATE_CSV_FILE;
extern PREPARE_TABLE_CREATE_CSV_FILE PrepareTableCreateCsvFile;

//
// Method implementations.
//

PERFECT_HASH_CONTEXT_TABLE_CREATE PerfectHashContextTableCreate;

_Use_decl_annotations_
HRESULT
PerfectHashContextTableCreate(
    PPERFECT_HASH_CONTEXT Context,
    PCUNICODE_STRING KeysPath,
    PCUNICODE_STRING BaseOutputDirectory,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PPERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlagsPointer,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlagsPointer,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Attempts to create perfect hash table for the given keys file.

Arguments:

    Context - Supplies an instance of PERFECT_HASH_CONTEXT.

    KeysPath - Supplies a pointer to a UNICODE_STRING structure that represents
        a fully-qualified path of the keys directory.

    BaseOutputDirectory - Supplies a pointer to a UNICODE_STRING structure that
        represents a fully-qualified path of the directory where the perfect
        hash table files generated as part of this routine will be saved.

    AlgorithmId - Supplies the algorithm to use.

    HashFunctionId - Supplies the hash function to use.

    MaskFunctionId - Supplies the type of masking to use.

    ContextTableCreateFlags - Optionally supplies a pointer to a context
        table create flags structure that can be used to customize behavior.

    KeysLoadFlags - Optionally supplies a pointer to a key loading flags
        structure that can be used to customize key loading behavior.

    TableCreateFlags - Optionally supplies a pointer to a table create flags
        structure that can be used to customize table creation behavior.

    TableCompileFlags - Optionally supplies a pointer to a compile table flags
        structure that can be used to customize table compilation behavior.

    TableCreateParameters - Optionally supplies a pointer to table create
        parameters that can be used to further customize table creation
        behavior.

Return Value:

    S_OK - Success.

    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - One or more mandatory parameters were NULL pointers.

    E_INVALIDARG - KeysPath or BaseOutputDirectory were invalid.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_MAXIMUM_CONCURRENCY - Invalid maximum concurrency.

    PH_E_INVALID_KEYS_LOAD_FLAGS - Invalid keys load flags.

    PH_E_INVALID_TABLE_CREATE_FLAGS - Invalid table create flags.

    PH_E_INVALID_TABLE_LOAD_FLAGS - Invalid table load flags.

    PH_E_INVALID_TABLE_COMPILE_FLAGS - Invalid table compile flags.

--*/
{
    PRTL Rtl;
    BOOLEAN Silent;
    USHORT NumberOfPages;
    ULONG NumberOfRows = 0;
    HRESULT Result = S_OK;
    HRESULT TableCreateResult;
    PCHAR RowBuffer = NULL;
    PALLOCATOR Allocator;
    PVOID KeysBaseAddress;
    ULARGE_INTEGER NumberOfKeys;
    HANDLE OutputHandle = NULL;
    HANDLE ProcessHandle = NULL;
    ULONGLONG RowBufferSize;
    ULONG BytesWritten = 0;
    ULONG KeySizeInBytes = 0;
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
    PERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags;
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

    VALIDATE_FLAGS(ContextTableCreate, CONTEXT_TABLE_CREATE);
    VALIDATE_FLAGS(KeysLoad, KEYS_LOAD);
    VALIDATE_FLAGS(TableCompile, TABLE_COMPILE);

    //
    // IsValidTableCreateFlags() returns a more specific error code than the
    // other validation routines invoked above (which would be converted into
    // PH_E_INVALID_TABLE_CREATE_FLAGS if we used the VALIDATE_FLAGS macro).
    //

    if (ARGUMENT_PRESENT(TableCreateFlagsPointer)) {
        Result = IsValidTableCreateFlags(TableCreateFlagsPointer);
        if (FAILED(Result)) {
            return Result;
        } else {
            TableCreateFlags.AsULong = TableCreateFlagsPointer->AsULong;
        }
    } else {
        TableCreateFlags.AsULong = 0;
    }

    if (!ARGUMENT_PRESENT(KeysPath)) {
        return E_POINTER;
    } else if (!IsValidMinimumDirectoryUnicodeString(KeysPath)) {
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
        return Result;
    }

    SetContextTableCreate(Context);

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
    // Prepare the .csv file if applicable.
    //

    if (TableCreateFlags.DisableCsvOutputFile == FALSE) {
        NumberOfRows = 1;
        Result = PrepareTableCreateCsvFile(Context, NumberOfRows);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashContextTableCreate_PrepareCsvFile, Result);
            goto Error;
        }
        CsvFile = Context->TableCreateCsvFile;
    }

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

    if (ContextTableCreateFlags.SkipTestAfterCreate) {
        TableCreateFlags.CreateOnly = TRUE;
    }

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
    // Create a keys instance.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_KEYS,
                                           &Keys);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashKeysCreateInstance, Result);
        goto Error;
    }

    //
    // Load the keys.
    //

    Result = Keys->Vtbl->Load(Keys,
                              &KeysLoadFlags,
                              KeysPath,
                              KeySizeInBytes);

    if (FAILED(Result)) {
        PH_KEYS_ERROR(PerfectHashKeysLoad, Result);
        goto Error;
    }

    Result = Keys->Vtbl->GetFlags(Keys,
                                  sizeof(KeysFlags),
                                  &KeysFlags);

    if (FAILED(Result)) {
        PH_KEYS_ERROR(PerfectHashKeysGetFlags, Result);
        goto Error;
    }

    Result = Keys->Vtbl->GetAddress(Keys,
                                    &KeysBaseAddress,
                                    &NumberOfKeys);

    if (FAILED(Result)) {
        PH_KEYS_ERROR(PerfectHashKeysGetAddress, Result);
        goto Error;
    }

    //
    // Keys were loaded successfully.  If CUDA is available, register the base
    // address of the array.
    //

    if (Context->Cu) {
        Result = PerfectHashKeysCopyToCuDevice(Keys,
                                               Context->Cu,
                                               ActiveCuDevice(Context));
        if (FAILED(Result)) {
            PH_KEYS_ERROR(PerfectHashKeysCopyToCuDevice, Result);
            goto Error;
        }
    }

    //
    // Proceed with table creation.
    //

    ASSERT(Table == NULL);

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_TABLE,
                                           &Table);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashTableCreateInstance, Result);
        goto Error;
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

    if (FAILED(Result)) {
        PH_KEYS_ERROR(PerfectHashTableCreate, Result);
        goto Error;
    }

    if (CtrlCPressed) {
        Result = PH_E_CTRL_C_PRESSED;
        goto Error;
    }

    PRINT_CHAR_FOR_TABLE_CREATE_RESULT(Result);

    if (Result != S_OK) {

        Coverage = &EmptyCoverage;

    } else {

        Coverage = Table->Coverage;

        //
        // Test the table, if applicable.
        //

        if (!ContextTableCreateFlags.SkipTestAfterCreate) {

            Result = Table->Vtbl->Test(Table, Keys, FALSE);

            if (FAILED(Result)) {
                PH_TABLE_ERROR(PerfectHashTableTest, Result);
                goto Error;
            }
        }

        //
        // Compile the table.
        //

        if (ContextTableCreateFlags.Compile) {

            Result = Table->Vtbl->Compile(Table,
                                          &TableCompileFlags,
                                          CpuArchId);

            if (FAILED(Result)) {
                PH_TABLE_ERROR(PerfectHashTableCompile, Result);
                goto Error;
            }
        }
    }

    //
    // Write the .csv row if applicable.
    //

    if (TableCreateFlags.DisableCsvOutputFile != FALSE) {
        goto End;
    }

    if (SkipWritingCsvRow(TableCreateFlags, TableCreateResult)) {
        goto End;
    }

    _Analysis_assume_(CsvFile != NULL);

    //
    // N.B. The SAL annotations are required to suppress the concurrency
    //      warnings for accessing the Context->NewBestGraphCount and
    //      Context->EqualBestGraphCount members outside of the best graph
    //      critical section.
    //

    _No_competing_thread_begin_
    WRITE_TABLE_CREATE_CSV_ROW();
    _No_competing_thread_end_

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
    // Release all references.
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

    ClearContextTableCreate(Context);

    if (CtrlCPressed) {
        Result = PH_E_CTRL_C_PRESSED;
    }

    //
    // Close the .csv file.  If we encountered an error, use 0 as end-of-file,
    // which will cause the Close() call to reset it to its original state.
    //

    if (Result != S_OK) {
        EndOfFile = &EmptyEndOfFile;
    } else {
        EndOfFile = NULL;
    }

    if (CsvFile) {
        Result = CsvFile->Vtbl->Close(CsvFile, EndOfFile);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashTableCreate_CloseCsvFile, Result);
        }
    }

    RELEASE(Context->TableCreateCsvFile);

    return Result;
}


PREPARE_TABLE_CREATE_CSV_FILE PrepareTableCreateCsvFile;

_Use_decl_annotations_
HRESULT
PrepareTableCreateCsvFile(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfRows
    )
/*++

Routine Description:

    Prepares the <BaseOutputDir>\PerfectHashTableCreate_<HeaderHash>.csv file.
    This involves determining the header hash, constructing a path instance,
    creating a file instance, and opening it for append.

Arguments:

    Context - Supplies the context for which the .csv file is to be prepared.

    NumberOfRows - Supplies the number of rows anticipated to be written during
        processing.  This is used to derive an appropriate file and memory map
        size to use for the .csv file.

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

    TABLE_CREATE_CSV_ROW_TABLE(EXPAND_AS_COLUMN_NAME_THEN_COMMA,
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

    BaseName = &PerfectHashTableCreateCsvBaseName;
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
        PH_ERROR(PrepareTableCreateCsvFile_PathCreate, Result);
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
    // boundary, then multiplied by the number of rows.  We then add the system
    // allocation size, then align the final amount up to this boundary.
    //
    // N.B. The resulting file size is very generous; we shouldn't ever come
    //      close to hitting it in normal operating conditions.
    //

    EndOfFile.QuadPart = (
        ALIGN_UP((ULONG_PTR)Header.Length + 32, 32) *
        (ULONG_PTR)NumberOfRows
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
        PH_ERROR(PrepareTableCreateCsvFile_FileCreate, Result);
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
            Result = PH_E_TABLE_CREATE_CSV_HEADER_MISMATCH;
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

    Context->TableCreateCsvFile = File;
    File->Vtbl->AddRef(File);

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
// Helper macros for argument extraction.  Copy-and-pasted from the bulk
// create implementation; we should look at cleaning this up down the track.
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

PERFECT_HASH_CONTEXT_EXTRACT_TABLE_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextExtractTableCreateArgsFromArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextExtractTableCreateArgsFromArgvW(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW,
    PUNICODE_STRING KeysPath,
    PUNICODE_STRING BaseOutputDirectory,
    PPERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PPERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PULONG MaximumConcurrency,
    PPERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Extracts arguments for the table create functionality from an argument
    vector array, typically obtained from a commandline invocation.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance
        for which the arguments are to be extracted.

    NumberOfArguments - Supplies the number of elements in the ArgvW array.

    ArgvW - Supplies a pointer to an array of wide C string arguments.

    KeysPath - Supplies a pointer to a UNICODE_STRING structure that will be
        filled out with the keys path.

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

    ContextTableCreateFlags - Supplies the address of a variable that will
        receive the context table-create flags.

    TableCreateFlags - Supplies the address of a variable that will receive the
        bulk-create flags.

    KeysLoadFlags - Supplies the address of a variable that will receive
        the keys load flags.

    TableCreateFlags - Supplies the address of a variable that will receive
        the table create flags.

    TableLoadFlags - Supplies the address of a variable that will receive the
        the load table flags.

    TableCompileFlags - Supplies the address of a variable that will receive
        the table compile flags.

    TableCreateParameters - Supplies the address of a table create params
        structure that will receive any extracted params.

Return Value:

    S_OK - Arguments extracted successfully.

    E_POINTER - One or more mandatory parameters were NULL pointers.

    PH_E_CONTEXT_TABLE_CREATE_INVALID_NUM_ARGS - Invalid number of arguments.

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

    if (!ARGUMENT_PRESENT(KeysPath)) {
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

    if (!ARGUMENT_PRESENT(ContextTableCreateFlags)) {
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

    ValidNumberOfArguments = (NumberOfArguments >= 7);

    if (!ValidNumberOfArguments) {
        return PH_E_CONTEXT_TABLE_CREATE_INVALID_NUM_ARGS;
    }

    //
    // Argument validation complete, continue.
    //

    ArgW = &ArgvW[1];
    Rtl = Context->Rtl;
    Allocator = Context->Allocator;

    //
    // Extract keys path.
    //

    CurrentArg++;
    KeysPath->Buffer = *ArgW++;
    KeysPath->Length = GET_LENGTH(KeysPath);
    KeysPath->MaximumLength = GET_MAX_LENGTH(KeysPath);

    //
    // Extract base output directory.
    //

    CurrentArg++;
    BaseOutputDirectory->Buffer = *ArgW++;
    BaseOutputDirectory->Length = GET_LENGTH(BaseOutputDirectory);
    BaseOutputDirectory->MaximumLength = GET_MAX_LENGTH(BaseOutputDirectory);

    //
    // Extract algorithm ID, hash function and mask function.
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
        PH_ERROR(ExtractTableCreateArgs_TryExtractArg##Name##Flags, Result);  \
        break;                                                                \
    } else if (Result == S_OK) {                                              \
        continue;                                                             \
    } else {                                                                  \
        ASSERT(Result == S_FALSE);                                            \
    }

        TRY_EXTRACT_ARG(ContextTableCreate);
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


PERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVW PerfectHashContextTableCreateArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextTableCreateArgvW(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW
    )
/*++

Routine Description:

    Extracts arguments for the table create functionality from an argument
    vector array and then invokes the context table create functionality.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance
        for which the arguments are to be extracted.

    NumberOfArguments - Supplies the number of elements in the ArgvW array.

    ArgvW - Supplies a pointer to an array of wide C string arguments.

Return Value:

    S_OK - Arguments extracted successfully.

    E_POINTER - One or more mandatory parameters were NULL pointers.

    PH_E_CONTEXT_TABLE_CREATE_INVALID_NUM_ARGS - Invalid number of arguments.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_MAXIMUM_CONCURRENCY - Invalid maximum concurrency.

    PH_E_INVALID_CONTEXT_TABLE_CREATE_FLAGS - Invalid table create flags.

    PH_E_INVALID_KEYS_LOAD_FLAGS - Invalid keys load flags.

    PH_E_INVALID_TABLE_LOAD_FLAGS - Invalid table load flags.

    PH_E_INVALID_TABLE_COMPILE_FLAGS - Invalid table compile flags.

--*/
{
    HRESULT Result;
    HRESULT CleanupResult;
    ULONG MaximumConcurrency = 0;
    UNICODE_STRING KeysPath = { 0 };
    UNICODE_STRING BaseOutputDirectory = { 0 };
    PERFECT_HASH_ALGORITHM_ID AlgorithmId = 0;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId = 0;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId = 0;
    PERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags = { 0 };
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags = { 0 };
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags = { 0 };
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags = { 0 };
    PPERFECT_HASH_CONTEXT_EXTRACT_TABLE_CREATE_ARGS_FROM_ARGVW
        ExtractTableCreateArgs;
    PERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters = { 0 };

    TableCreateParameters.SizeOfStruct = sizeof(TableCreateParameters);
    TableCreateParameters.Allocator = Context->Allocator;

    ExtractTableCreateArgs = Context->Vtbl->ExtractTableCreateArgsFromArgvW;
    Result = ExtractTableCreateArgs(Context,
                                    NumberOfArguments,
                                    ArgvW,
                                    &KeysPath,
                                    &BaseOutputDirectory,
                                    &AlgorithmId,
                                    &HashFunctionId,
                                    &MaskFunctionId,
                                    &MaximumConcurrency,
                                    &ContextTableCreateFlags,
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
            PH_ERROR(PerfectHashContextTableCreateArgvW, Result);
            return Result;
        }
    }

    PerfectHashContextApplyThreadpoolPriorities(Context,
                                                &TableCreateParameters);

    Result = PerfectHashContextInitializeRng(Context,
                                             &TableCreateFlags,
                                             &TableCreateParameters);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashContextTableCreateArgvW_InitRng, Result);
        return Result;
    }

    if (ContextTableCreateFlags.TryCuda != FALSE) {
        Result = PerfectHashContextInitializeCuda(Context,
                                                  &TableCreateParameters);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashContextTableCreateArgvW_InitializeCuda, Result);
            return Result;
        }
    }

    Result = Context->Vtbl->TableCreate(Context,
                                        &KeysPath,
                                        &BaseOutputDirectory,
                                        AlgorithmId,
                                        HashFunctionId,
                                        MaskFunctionId,
                                        &ContextTableCreateFlags,
                                        &KeysLoadFlags,
                                        &TableCreateFlags,
                                        &TableCompileFlags,
                                        &TableCreateParameters);

    if (FAILED(Result)) {

        //
        // There's is nothing we can do here.  We don't PH_ERROR() the return
        // code as TableCreate() will have done that multiple times each time
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
