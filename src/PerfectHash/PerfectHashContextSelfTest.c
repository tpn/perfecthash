/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContextSelfTest.c

Abstract:

    This module implements the self-test routine for the PerfectHash
    component.  It is responsible for end-to-end testing of the entire
    component with all known test data from a single function entry point.

--*/

#include "stdafx.h"

PERFECT_HASH_CONTEXT_SELF_TEST PerfectHashContextSelfTest;

_Use_decl_annotations_
HRESULT
PerfectHashContextSelfTest(
    PPERFECT_HASH_CONTEXT Context,
    PCUNICODE_STRING TestDataDirectory,
    PCUNICODE_STRING BaseOutputDirectory,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PPERFECT_HASH_CONTEXT_SELF_TEST_FLAGS ContextSelfTestFlagsPointer,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlagsPointer,
    PPERFECT_HASH_TABLE_LOAD_FLAGS TableLoadFlagsPointer,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Performs a self-test of the entire PerfectHash component.

Arguments:

    Context - Supplies an instance of PERFECT_HASH_CONTEXT.

    TestDataDirectory - Supplies a pointer to a UNICODE_STRING structure that
        represents a fully-qualified path of the test data directory.

    BaseOutputDirectory - Supplies a pointer to a UNICODE_STRING structure that
        represents a fully-qualified path of the directory where the perfect
        hash table files generated as part of this routine will be saved.

    AlgorithmId - Supplies the algorithm to use.

    HashFunctionId - Supplies the hash function to use.

    MaskFunctionId - Supplies the type of masking to use.

    ContextSelfTestFlags - Optionally supplies a pointer to a self-test flags
        structure that can be used to customize self-test behavior.

    KeysLoadFlags - Optionally supplies a pointer to a key loading flags
        structure that can be used to customize key loading behavior.

    TableCreateFlags - Optionally supplies a pointer to a table create flags
        structure that can be used to customize table creation behavior.

    TableLoadFlags - Optionally supplies a pointer to a load table flags
        structure that can be used to customize table loading behavior.

    TableCompileFlags - Optionally supplies a pointer to a compile table flags
        structure that can be used to customize table compilation behavior.

    TableCreateParameters - Supplies an array of additional
        parameters that can be used to further customize table creation
        behavior.

Return Value:

    S_OK - Self test performed successfully.

    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - One or more mandatory parameters were NULL pointers.

    E_INVALIDARG - TestDataDirectory or BaseOutputDirectory were invalid.

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
    PWSTR Dest;
    PWSTR Source;
    ULONG Index;
    ULONG LastError;
    USHORT Length;
    USHORT BaseLength;
    USHORT NumberOfPages;
    ULONG ReferenceCount;
    BOOL Success;
    BOOLEAN Failed;
    BOOLEAN Terminate;
    HRESULT Result;
    PWCHAR Buffer;
    PWCHAR BaseBuffer;
    PWCHAR WideOutput;
    PWCHAR WideOutputBuffer;
    PALLOCATOR Allocator;
    PVOID KeysBaseAddress;
    ULARGE_INTEGER NumberOfKeys;
    WCHAR WideBitmapString[65];
    UNICODE_STRING UnicodeBitmapString;
    HANDLE FindHandle = NULL;
    HANDLE WideOutputHandle;
    HANDLE ProcessHandle = NULL;
    ULONG Failures;
    ULONG BytesWritten;
    ULONG WideCharsWritten;
    ULONGLONG BufferSize;
    ULONGLONG WideOutputBufferSize;
    LONG_INTEGER AllocSize;
    LARGE_INTEGER BytesToWrite;
    LARGE_INTEGER WideCharsToWrite;
    WIN32_FIND_DATAW FindData;
    UNICODE_STRING WildcardPath;
    UNICODE_STRING KeysPathString;
    PUNICODE_STRING TableFullPath;
    PPERFECT_HASH_KEYS Keys;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_FILE TableFile = NULL;
    PPERFECT_HASH_PATH TablePath = NULL;
    PPERFECT_HASH_DIRECTORY BaseOutputDir = NULL;
    PPERFECT_HASH_PATH_PARTS TablePathParts;
    PERFECT_HASH_KEYS_FLAGS KeysFlags;
    PERFECT_HASH_KEYS_BITMAP KeysBitmap;
    PERFECT_HASH_TABLE_FLAGS TableFlags;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PCUNICODE_STRING Suffix = &KeysWildcardSuffix;
    PUNICODE_STRING AlgorithmName;
    PUNICODE_STRING HashFunctionName;
    PUNICODE_STRING MaskFunctionName;
    PERFECT_HASH_CPU_ARCH_ID CpuArchId;
    PERFECT_HASH_CONTEXT_SELF_TEST_FLAGS ContextSelfTestFlags;
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags;
    PERFECT_HASH_TABLE_LOAD_FLAGS TableLoadFlags;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    } else {
        Rtl = Context->Rtl;
        Allocator = Context->Allocator;
    }

    VALIDATE_FLAGS(ContextSelfTest, CONTEXT_SELF_TEST);
    VALIDATE_FLAGS(KeysLoad, KEYS_LOAD);
    VALIDATE_FLAGS(TableCreate, TABLE_CREATE);
    VALIDATE_FLAGS(TableLoad, TABLE_LOAD);
    VALIDATE_FLAGS(TableCompile, TABLE_COMPILE);

    if (!ARGUMENT_PRESENT(TestDataDirectory)) {
        return E_POINTER;
    } else if (!IsValidMinimumDirectoryUnicodeString(TestDataDirectory)) {
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

    //
    // Create a buffer we can use for stdout, using a very generous buffer size.
    //

    NumberOfPages = 10;
    ProcessHandle = GetCurrentProcess();

    Result = Rtl->Vtbl->CreateBuffer(Rtl,
                                     &ProcessHandle,
                                     NumberOfPages,
                                     NULL,
                                     &WideOutputBufferSize,
                                     &WideOutputBuffer);

    if (FAILED(Result)) {
        return Result;
    }

    WideOutput = WideOutputBuffer;

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
        Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                          ProcessHandle,
                                          &WideOutputBuffer);
        if (FAILED(Result)) {
            PH_ERROR(RtlDestroyBuffer, Result);
        }
        return Result;
    }

    Buffer = BaseBuffer;

    //
    // Get a reference to the stdout handle.
    //

    WideOutputHandle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (!WideOutputHandle) {
        SYS_ERROR(GetStdHandle);
        return E_UNEXPECTED;
    }

    //
    // Calculate the size required for a new concatenated wide string buffer
    // that combines the test data directory with the "*.keys" suffix.  The
    // 2 * sizeof(*Dest) accounts for the joining slash and trailing NULL.
    //

    AllocSize.LongPart = TestDataDirectory->Length;
    AllocSize.LongPart += Suffix->Length + (2 * sizeof(*Dest));

    ASSERT(!AllocSize.HighPart);

    WildcardPath.Buffer = (PWSTR)Buffer;

    if (!WildcardPath.Buffer) {
        goto Error;
    }

    //
    // Copy incoming test data directory name.
    //

    Length = TestDataDirectory->Length;
    CopyMemory(WildcardPath.Buffer,
               TestDataDirectory->Buffer,
               Length);

    //
    // Advance our Dest pointer to the end of the directory name, write a
    // slash, then copy the suffix over.
    //

    Dest = (PWSTR)RtlOffsetToPointer(WildcardPath.Buffer, Length);
    *Dest++ = L'\\';
    CopyMemory(Dest, Suffix->Buffer, Suffix->Length);

    //
    // Wire up the search path length and maximum length variables.  The max
    // length will be our AllocSize, length will be this value minus 2 to
    // account for the trailing NULL.
    //

    WildcardPath.MaximumLength = AllocSize.LowPart;
    WildcardPath.Length = AllocSize.LowPart - sizeof(*Dest);
    ASSERT(WildcardPath.Length < WildcardPath.MaximumLength);
    ASSERT(WildcardPath.Buffer[WildcardPath.Length >> 1] == L'\0');

    //
    // Advance the buffer past this string allocation, up to the next 16-byte
    // boundary.
    //

    Buffer = (PWSTR)(
        RtlOffsetToPointer(
            Buffer,
            ALIGN_UP(WildcardPath.MaximumLength, 16)
        )
    );

    WIDE_OUTPUT_RAW(WideOutput,
                    L"Starting perfect hash self-test for directory: ");
    WIDE_OUTPUT_UNICODE_STRING(WideOutput, TestDataDirectory);
    WIDE_OUTPUT_RAW(WideOutput, L".\n");
    WIDE_OUTPUT_FLUSH();

    //
    // Create a find handle for the <test data>\*.keys search pattern we
    // created.
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

            WIDE_OUTPUT_RAW(WideOutput,
                            L"No files matching pattern '*.keys' found in "
                            L"test data directory.\n");
            WIDE_OUTPUT_FLUSH();

            goto End;

        } else {

            //
            // We failed for some other reason.
            //

            WIDE_OUTPUT_RAW(WideOutput,
                            L"FindFirstFileW() failed with error code: ");
            WIDE_OUTPUT_INT(WideOutput, LastError);
            WIDE_OUTPUT_LF(WideOutput);
            WIDE_OUTPUT_FLUSH();

            goto Error;
        }
    }

    //
    // Initialize the fully-qualified keys path.
    //

    KeysPathString.Buffer = Buffer;
    CopyMemory(KeysPathString.Buffer, TestDataDirectory->Buffer, Length);

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
    // Zero the failure count and terminate flag, zero the bitmap structure,
    // wire up the unicode string representation of the bitmap, initialize
    // various flags, obtain the current CPU architecture ID, and begin the main
    // loop.
    //

    Failures = 0;
    Terminate = FALSE;
    ZeroStruct(KeysBitmap);
    UnicodeBitmapString.Buffer = (PWCHAR)WideBitmapString;
    UnicodeBitmapString.Length = sizeof(WideBitmapString)-2;
    UnicodeBitmapString.MaximumLength = sizeof(WideBitmapString);
    UnicodeBitmapString.Buffer[UnicodeBitmapString.Length >> 1] = L'\0';
    Table = NULL;
    KeysBaseAddress = NULL;
    NumberOfKeys.QuadPart = 0;
    KeysFlags.AsULong = 0;
    TableCreateFlags.AsULong = 0;
    CpuArchId = PerfectHashGetCurrentCpuArch();

    ASSERT(IsValidPerfectHashCpuArchId(CpuArchId));

    do {

        //
        // Clear the failure flag at the top of every loop invocation.
        //

        Failed = FALSE;

        WIDE_OUTPUT_RAW(WideOutput, L"Processing key file: ");
        WIDE_OUTPUT_WCSTR(WideOutput, (PCWSZ)FindData.cFileName);
        WIDE_OUTPUT_LF(WideOutput);
        WIDE_OUTPUT_FLUSH();

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
            WIDE_OUTPUT_RAW(WideOutput, L"Failed to create keys instance.\n");
            WIDE_OUTPUT_FLUSH();
            Failures++;
            break;
        }

        Result = Keys->Vtbl->Load(Keys,
                                  &KeysLoadFlags,
                                  &KeysPathString,
                                  sizeof(ULONG));

        if (FAILED(Result)) {

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to load keys for ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPathString);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Successfully loaded keys: ");
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPathString);
        WIDE_OUTPUT_RAW(WideOutput, L".\n");
        WIDE_OUTPUT_FLUSH();

        //
        // Verify GetFlags().
        //

        Result = Keys->Vtbl->GetFlags(Keys,
                                      sizeof(KeysFlags),
                                      &KeysFlags);

        if (FAILED(Result)) {
            WIDE_OUTPUT_RAW(WideOutput, L"Failed to obtain flags for keys: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPathString);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        //
        // Verify base address and number of keys.
        //

        Result = Keys->Vtbl->GetAddress(Keys,
                                        &KeysBaseAddress,
                                        &NumberOfKeys);

        if (FAILED(Result)) {
            WIDE_OUTPUT_RAW(WideOutput, L"Failed to obtain base "
                                        L"address for keys: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPathString);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        //
        // Verify the bitmap function returns success.
        //

        Result = Keys->Vtbl->GetBitmap(Keys, sizeof(KeysBitmap), &KeysBitmap);

        if (FAILED(Result)) {
            WIDE_OUTPUT_RAW(WideOutput, L"Failed to get keys bitmap for ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPathString);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Keys bitmap: ");
        WIDE_OUTPUT_INT(WideOutput, KeysBitmap.Bitmap);
        WIDE_OUTPUT_RAW(WideOutput, L".\n");

        //
        // The bitmap buffer is a normal 8-bit character string, but our output
        // uses 16-bit wide character strings.  Do a simple conversion now.  We
        // don't need to worry about utf-8 multi-byte characters as the only
        // possible bitmap character values are '0' and '1'.
        //

        for (Index = 0; Index < sizeof(KeysBitmap.String); Index++) {
            WideBitmapString[Index] = (WCHAR)KeysBitmap.String[Index];
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Keys bitmap string: ");
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, &UnicodeBitmapString);
        WIDE_OUTPUT_RAW(WideOutput, L".\n");

        WIDE_OUTPUT_RAW(WideOutput, L"Keys bitmap contiguous? ");
        if (KeysBitmap.Flags.Contiguous) {
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &Yes);
            WIDE_OUTPUT_RAW(WideOutput, L"Keys bitmap shifted mask: ");
            WIDE_OUTPUT_INT(WideOutput, KeysBitmap.ShiftedMask);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
        } else {
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &No);
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Keys bitmap longest run length: ");
        WIDE_OUTPUT_INT(WideOutput, KeysBitmap.LongestRunLength);
        WIDE_OUTPUT_RAW(WideOutput, L".\nKeys bitmap longest run start: ");
        WIDE_OUTPUT_INT(WideOutput, KeysBitmap.LongestRunStart);
        WIDE_OUTPUT_RAW(WideOutput, L".\nKeys bitmap trailing zeros: ");
        WIDE_OUTPUT_INT(WideOutput, KeysBitmap.TrailingZeros);
        WIDE_OUTPUT_RAW(WideOutput, L".\nKeys bitmap leading zeros: ");
        WIDE_OUTPUT_INT(WideOutput, KeysBitmap.LeadingZeros);
        WIDE_OUTPUT_RAW(WideOutput, L".\n");

        WIDE_OUTPUT_FLUSH();

        //
        // Verify a subsequent load attempt indicates keys have already
        // been loaded.
        //

        Result = Keys->Vtbl->Load(Keys,
                                  &KeysLoadFlags,
                                  &KeysPathString,
                                  sizeof(ULONG));

        if (Result != PH_E_KEYS_ALREADY_LOADED) {
            WIDE_OUTPUT_RAW(WideOutput, L"Invariant failed; multiple "
                                        L"key loads did not raise an "
                                        L"error for ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPathString);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

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
            WIDE_OUTPUT_RAW(WideOutput, L"Failed to create table instance.\n");
            WIDE_OUTPUT_FLUSH();
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

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashTableCreate, Result);

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to create perfect hash "
                                        L"table for keys: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPathString);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failed = TRUE;
            Failures++;
            goto ReleaseTable;
        }

        //
        // Get the underlying file for the table.
        //

        Result = Table->Vtbl->GetFile(Table, &TableFile);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashTableGetFile, Result);
            Failed = TRUE;
            Failures++;
            goto ReleaseTable;
        }

        //
        // Get the underlying path for the table's file.
        //

        Result = TableFile->Vtbl->GetPath(TableFile, &TablePath);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileGetPath, Result);
            Failed = TRUE;
            Failures++;
            goto ReleaseTable;
        }

        //
        // Obtain the path parts.
        //

        Result = TablePath->Vtbl->GetParts(TablePath, &TablePathParts);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashPathGetParts, Result);
            Failed = TRUE;
            Failures++;
            goto ReleaseTable;
        }

        TableFullPath = &TablePathParts->FullPath;
        WIDE_OUTPUT_RAW(WideOutput, L"Successfully created perfect "
                                    L"hash table: ");
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, TableFullPath);
        WIDE_OUTPUT_RAW(WideOutput, L".\n");
        WIDE_OUTPUT_FLUSH();

        //
        // Verify a subsequent load attempt indicates that the table is already
        // loaded.
        //

        Result = Table->Vtbl->Load(Table,
                                   &TableLoadFlags,
                                   TableFullPath,
                                   Keys);

        if (Result != PH_E_TABLE_ALREADY_CREATED) {
            WIDE_OUTPUT_RAW(WideOutput, L"Invariant failed; multiple "
                                        L"table loads did not raise an "
                                        L"error for ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, TableFullPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseTable;
        }

        //
        // Test the newly-created table.
        //

        Result = Table->Vtbl->Test(Table, Keys, TRUE);

        if (FAILED(Result)) {

            WIDE_OUTPUT_RAW(WideOutput, L"Test failed for perfect hash table "
                                        L"created from context: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, TableFullPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Failed = TRUE;
            goto ReleaseTable;
        }

        //
        // Disable compilation at the moment as it adds an extra ~6-10 seconds
        // per iteration.
        //

#if 0

        //
        // Compile the table.
        //

        Result = Table->Vtbl->Compile(Table,
                                      &TableCompileFlags,
                                      CpuArchId);

        if (FAILED(Result)) {

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to compile table: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, TableFullPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Failed = TRUE;
            goto ReleaseTable;
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Compiled table successfully.\n");
        WIDE_OUTPUT_FLUSH();

#endif

        //
        // Release the table.
        //

        ReferenceCount = Table->Vtbl->Release(Table);
        Table = NULL;

        if (ReferenceCount != 0) {
            WIDE_OUTPUT_RAW(WideOutput, L"Invariant failed; releasing table "
                                        L"did not indicate a refcount of 0.\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        //
        // Create a new table instance.
        //

        Result = Context->Vtbl->CreateInstance(Context,
                                               NULL,
                                               &IID_PERFECT_HASH_TABLE,
                                               &Table);

        if (FAILED(Result)) {
            WIDE_OUTPUT_RAW(WideOutput, L"Failed to create table instance.\n");
            WIDE_OUTPUT_FLUSH();
            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        //
        // Load the perfect hash table we just created.
        //

        Result = Table->Vtbl->Load(Table,
                                   &TableLoadFlags,
                                   TableFullPath,
                                   Keys);

        if (FAILED(Result)) {

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to load perfect hash table: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, TableFullPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        //
        // Verify a subsequent load attempt indicates that the table is already
        // loaded.
        //

        Result = Table->Vtbl->Load(Table,
                                   &TableLoadFlags,
                                   TableFullPath,
                                   Keys);

        if (Result != PH_E_TABLE_ALREADY_LOADED) {
            WIDE_OUTPUT_RAW(WideOutput, L"Invariant failed; multiple "
                                        L"table loads did not raise an "
                                        L"error for ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, TableFullPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseTable;
        }

        //
        // Table was loaded successfully from disk.  Obtain the names of all
        // the enumeration IDs.  Currently these should always match the same
        // enums provided as input parameters to this routine.
        //

#define GET_NAME(Desc)                                               \
        Result = Table->Vtbl->Get##Desc##Name(Table, &##Desc##Name); \
        if (FAILED(Result)) {                                        \
            WIDE_OUTPUT_RAW(WideOutput,                              \
                            L"Get" L#Desc "Name() failed.\n");       \
            Terminate = TRUE;                                        \
            goto ReleaseTable;                                       \
        }

        GET_NAME(Algorithm);
        GET_NAME(HashFunction);
        GET_NAME(MaskFunction);

        WIDE_OUTPUT_RAW(WideOutput, L"Successfully loaded perfect "
                                    L"hash table: ");
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, TableFullPath);
        WIDE_OUTPUT_RAW(WideOutput, L".\n");

        WIDE_OUTPUT_RAW(WideOutput, L"Algorithm: ");
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, AlgorithmName);
        WIDE_OUTPUT_RAW(WideOutput, L" (");
        WIDE_OUTPUT_INT(WideOutput, Table->AlgorithmId);
        WIDE_OUTPUT_RAW(WideOutput, L").\n");

        WIDE_OUTPUT_RAW(WideOutput, L"Hash Function: ");
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, HashFunctionName);
        WIDE_OUTPUT_RAW(WideOutput, L" (");
        WIDE_OUTPUT_INT(WideOutput, Table->HashFunctionId);
        WIDE_OUTPUT_RAW(WideOutput, L").\n");

        WIDE_OUTPUT_RAW(WideOutput, L"Mask Function: ");
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, MaskFunctionName);
        WIDE_OUTPUT_RAW(WideOutput, L" (");
        WIDE_OUTPUT_INT(WideOutput, Table->MaskFunctionId);
        WIDE_OUTPUT_RAW(WideOutput, L").\n");

        WIDE_OUTPUT_RAW(WideOutput, L"Keys backed by large pages: ");
        if (KeysFlags.KeysDataUsesLargePages) {
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &Yes);
        } else {
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &No);
        }

        //
        // Verify GetFlags().
        //

        Result = Table->Vtbl->GetFlags(Table,
                                       sizeof(TableFlags),
                                       &TableFlags);

        if (FAILED(Result)) {
            WIDE_OUTPUT_RAW(WideOutput, L"Failed to obtain flags for table: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, TableFullPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseTable;
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Table data backed by large pages: ");
        if (TableFlags.TableDataUsesLargePages) {
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &Yes);
        } else {
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &No);
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Values array allocated with large "
                                    L"pages: ");
        if (TableFlags.ValuesArrayUsesLargePages) {
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &Yes);
        } else {
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &No);
        }

        WIDE_OUTPUT_FLUSH();

        TableInfoOnDisk = Table->TableInfoOnDisk;

        //
        // Define some helper macros here for dumping stats.
        //

#define STATS_INT(String, Name)                                        \
        WIDE_OUTPUT_RAW(WideOutput, String);                           \
        WIDE_OUTPUT_INT(WideOutput, Table->TableInfoOnDisk->##Name##); \
        WIDE_OUTPUT_RAW(WideOutput, L".\n")

#define STATS_QUAD(String, Name)                                                \
        WIDE_OUTPUT_RAW(WideOutput, String);                                    \
        WIDE_OUTPUT_INT(WideOutput, Table->TableInfoOnDisk->##Name##.QuadPart); \
        WIDE_OUTPUT_RAW(WideOutput, L".\n")

        if (TableInfoOnDisk->NumberOfTableResizeEvents > 0) {

            STATS_INT(L"Number of table resize events: ",
                      NumberOfTableResizeEvents);

            STATS_INT(L"Total number of attempts with smaller table sizes: ",
                      TotalNumberOfAttemptsWithSmallerTableSizes);

            STATS_INT(L"First table size attempted: ",
                      InitialTableSize);

            STATS_INT(L"Closest we came to solving the graph in previous "
                      L"attempts by number of deleted edges away: ",
                      ClosestWeCameToSolvingGraphWithSmallerTableSizes);

        } else {

            WIDE_OUTPUT_RAW(WideOutput,
                            L"Number of table resize events: 0.\n");
        }

        STATS_INT(L"Concurrency: ", Concurrency);
        STATS_INT(L"Number of attempts: ", NumberOfAttempts);
        STATS_INT(L"Number of failed attempts: ", NumberOfFailedAttempts);
        STATS_INT(L"Number of solutions found: ", NumberOfSolutionsFound);

        STATS_QUAD(L"Number of keys: ", NumberOfKeys);
        STATS_QUAD(L"Number of table elements (vertices): ",
                   NumberOfTableElements);

        STATS_INT(L"Seed 1: ", Seed1);
        STATS_INT(L"Seed 2: ", Seed2);
        STATS_INT(L"Seed 3: ", Seed3);
        STATS_INT(L"Seed 4: ", Seed4);


        STATS_QUAD(L"Cycles to solve: ", SolveCycles);
        STATS_QUAD(L"Cycles to verify: ", VerifyCycles);

        //STATS_QUAD(L"Cycles to prepare table file: ", PrepareTableFileCycles);
        //STATS_QUAD(L"Cycles to save table file: ", SaveTableFileCycles);
        //STATS_QUAD(L"Cycles to prepare header file: ", PrepareHeaderFileCycles);
        //STATS_QUAD(L"Cycles to save header file: ", SaveHeaderFileCycles);

        STATS_QUAD(L"Microseconds to solve: ", SolveMicroseconds);
        STATS_QUAD(L"Microseconds to verify: ", VerifyMicroseconds);

        //STATS_QUAD(L"Microseconds to prepare table file: ",
        //           PrepareTableFileMicroseconds);
        //STATS_QUAD(L"Microseconds to save table file: ",
        //           SaveTableFileMicroseconds);
        //STATS_QUAD(L"Microseconds to prepare header file: ",
        //           PrepareHeaderFileMicroseconds);
        //STATS_QUAD(L"Microseconds to save header file: ",
        //           SaveHeaderFileMicroseconds);

        WIDE_OUTPUT_FLUSH();

        //
        // Test the table loaded from disk.
        //

        Result = Table->Vtbl->Test(Table, Keys, TRUE);

        if (FAILED(Result)) {

            WIDE_OUTPUT_RAW(WideOutput, L"Test failed for perfect hash table "
                                        L"loaded from disk: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, TableFullPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Failed = TRUE;
            goto ReleaseTable;
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Successfully tested perfect hash "
                                    L"table.\n\n");
        WIDE_OUTPUT_FLUSH();

        //
        // Verify compiling a loaded table fails.
        //

        Result = Table->Vtbl->Compile(Table,
                                      &TableCompileFlags,
                                      CpuArchId);

        if (Result != PH_E_TABLE_NOT_CREATED) {
            WIDE_OUTPUT_RAW(WideOutput, L"Invariant failed; attempting to "
                                        L"compile a loaded table did not "
                                        L"return PH_E_TABLE_NOT_CREATED.\n");
            WIDE_OUTPUT_FLUSH();
            Failures++;
            Terminate = TRUE;

            //
            // Intentional follow-on to ReleaseTable.
            //

        }

        //
        // Release the table and keys.
        //

ReleaseTable:

        if (TablePath) {
            TablePath->Vtbl->Release(TablePath);
            TablePath = NULL;
        }

        if (TableFile) {
            TableFile->Vtbl->Release(TableFile);
            TableFile = NULL;
        }

        Table->Vtbl->Release(Table);
        Table = NULL;

ReleaseKeys:

        Keys->Vtbl->Release(Keys);
        Keys = NULL;

        if (Terminate) {
            break;
        }

    } while (FindNextFile(FindHandle, &FindData));

    //
    // Self test complete!
    //

    if (!Failures && !Terminate) {
        Result = S_OK;
        goto End;
    }

    //
    // Intentional follow-on to Error.
    //

Error:

    if (Result == S_OK) {
        Result = E_FAIL;
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (WideOutputBuffer) {
        Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                          ProcessHandle,
                                          &WideOutputBuffer);
        if (FAILED(Result)) {
            SYS_ERROR(VirtualFree);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
        WideOutput = NULL;
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

    return Result;
}

//
// Commandline support.
//

static const STRING Usage = RTL_CONSTANT_STRING(
    "Usage: PerfectHashSelfTest.exe "
    "<TestDataDirectory (must be fully-qualified)> "
    "<BaseOutputDirectory (must be fully-qualified)> "
    "<AlgorithmId> "
    "<HashFunctionId> "
    "<MaskFunctionId> "
    "<MaximumConcurrency (0-ncpu)> "
    "E.g.: PerfectHashSelfTest.exe "
    "C:\\Users\\Trent\\Home\\src\\perfecthash\\data "
    "C:\\Temp\\output "
    "1 1 2 0\n"
);

//
// Helper macros for argument extraction.
//

#define GET_LENGTH(Name) (USHORT)wcslen(Name->Buffer) << (USHORT)1
#define GET_MAX_LENGTH(Name) Name->Length + 2

#define VALIDATE_ID(Name, Upper)                                      \
    if (FAILED(Rtl->RtlUnicodeStringToInteger(String,                 \
                                              10,                     \
                                              (PULONG)##Name##Id))) { \
        return PH_E_INVALID_##Upper##_ID;                             \
    } else if (!IsValidPerfectHash##Name##Id(*##Name##Id)) {          \
        return PH_E_INVALID_##Upper##_ID;                             \
    }



PERFECT_HASH_CONTEXT_EXTRACT_SELF_TEST_ARGS_FROM_ARGVW
    PerfectHashContextExtractSelfTestArgsFromArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextExtractSelfTestArgsFromArgvW(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW,
    PUNICODE_STRING TestDataDirectory,
    PUNICODE_STRING BaseOutputDirectory,
    PPERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PPERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PULONG MaximumConcurrency,
    PPERFECT_HASH_CONTEXT_SELF_TEST_FLAGS ContextSelfTestFlags,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    PPERFECT_HASH_TABLE_LOAD_FLAGS TableLoadFlags,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Extracts arguments for the self-test functionality from an argument vector
    array, typically obtained from a commandline invocation.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance
        for which the arguments are to be extracted.

    NumberOfArguments - Supplies the number of elements in the ArgvW array.

    ArgvW - Supplies a pointer to an array of wide C string arguments.

    TestDataDirectory - Supplies a pointer to a UNICODE_STRING structure that
        will be filled out with the test data directory.

    TestDataDirectory - Supplies a pointer to a UNICODE_STRING structure that
        will be filled out with the test data directory.

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

    SelfTestFlags - Supplies the address of a variable that will receive the
        self-test flags.

    KeysLoadFlags - Supplies the address of a variable that will receive
        the keys load flags.

    TableCreateFlags - Supplies the address of a variable that will receive
        the table create flags.

    TableLoadFlags - Supplies the address of a variable that will receive the
        the load table flags.

    TableCompileFlags - Supplies the address of a variable that will receive
        the table compile flags.

    TableCreateParameters - Supplies a pointer to a table create params struct
        that will be used to capture parameters.

Return Value:

    S_OK - Arguments extracted successfully.

    E_POINTER - One or more mandatory parameters were NULL pointers.

    PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS - Invalid number of arguments.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_MAXIMUM_CONCURRENCY - Invalid maximum concurrency.

--*/
{
    PRTL Rtl;
    LPWSTR *ArgW;
    UNICODE_STRING Temp;
    PUNICODE_STRING String;
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

    if (!ARGUMENT_PRESENT(TestDataDirectory)) {
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

    if (!ARGUMENT_PRESENT(ContextSelfTestFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(KeysLoadFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCreateFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableLoadFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCompileFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCreateParameters)) {
        return E_POINTER;
    }

    ValidNumberOfArguments = (
        NumberOfArguments == 7 ||
        NumberOfArguments == 8
    );

    if (!ValidNumberOfArguments) {
        return PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS;
    }

    //
    // Argument validation complete, continue.
    //

    ArgW = &ArgvW[1];
    Rtl = Context->Rtl;

    //
    // Extract test data directory.
    //

    TestDataDirectory->Buffer = *ArgW++;
    TestDataDirectory->Length = GET_LENGTH(TestDataDirectory);
    TestDataDirectory->MaximumLength = GET_MAX_LENGTH(TestDataDirectory);

    //
    // Extract test data directory.
    //

    BaseOutputDirectory->Buffer = *ArgW++;
    BaseOutputDirectory->Length = GET_LENGTH(BaseOutputDirectory);

    ValidNumberOfArguments = (
        NumberOfArguments == 7
    );

    if (!ValidNumberOfArguments) {
        return PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS;
    }

    //
    // Argument validation complete, continue.
    //

    ArgW = &ArgvW[1];
    Rtl = Context->Rtl;

    //
    // Extract test data directory.
    //

    TestDataDirectory->Buffer = *ArgW++;
    TestDataDirectory->Length = GET_LENGTH(TestDataDirectory);
    TestDataDirectory->MaximumLength = GET_MAX_LENGTH(TestDataDirectory);

    //
    // Extract base output directory.
    //

    BaseOutputDirectory->Buffer = *ArgW++;
    BaseOutputDirectory->Length = GET_LENGTH(BaseOutputDirectory);
    BaseOutputDirectory->MaximumLength = GET_MAX_LENGTH(BaseOutputDirectory);

    //
    // Extract algorithm ID.
    //

    String->Buffer = *ArgW++;
    String->Length = GET_LENGTH(String);
    String->MaximumLength = GET_MAX_LENGTH(String);
    VALIDATE_ID(Algorithm, ALGORITHM);

    //
    // Extract hash function ID.
    //

    String->Buffer = *ArgW++;
    String->Length = GET_LENGTH(String);
    String->MaximumLength = GET_MAX_LENGTH(String);
    VALIDATE_ID(HashFunction, HASH_FUNCTION);

    //
    // Extract mask function ID.
    //

    String->Buffer = *ArgW++;
    String->Length = GET_LENGTH(String);
    String->MaximumLength = GET_MAX_LENGTH(String);
    VALIDATE_ID(MaskFunction, MASK_FUNCTION);

    //
    // Extract maximum concurrency.
    //

    String->Buffer = *ArgW++;
    String->Length = GET_LENGTH(String);
    String->MaximumLength = GET_MAX_LENGTH(String);

    if (FAILED(Rtl->RtlUnicodeStringToInteger(String,
                                              10,
                                              MaximumConcurrency))) {
        return PH_E_INVALID_MAXIMUM_CONCURRENCY;
    }

    ContextSelfTestFlags->AsULong = 0;

    //
    // We haven't implemented support for parsing flags from the command line,
    // so zero everything for now.
    //

    KeysLoadFlags->AsULong = 0;
    TableCreateFlags->AsULong = 0;
    TableLoadFlags->AsULong = 0;
    TableCompileFlags->AsULong = 0;

    return S_OK;
}

PERFECT_HASH_CONTEXT_SELF_TEST_ARGVW PerfectHashContextSelfTestArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextSelfTestArgvW(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW
    )
/*++

Routine Description:

    Extracts arguments for the self-test functionality from an argument vector
    array and then invokes the context self-test functionality.

Arguments:

    Context - Supplies a pointer to the PERFECT_HASH_CONTEXT instance
        for which the arguments are to be extracted.

    NumberOfArguments - Supplies the number of elements in the ArgvW array.

    ArgvW - Supplies a pointer to an array of wide C string arguments.

Return Value:

    S_OK - Arguments extracted successfully.

    E_POINTER - One or more mandatory parameters were NULL pointers.

    PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS - Invalid number of arguments.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_MAXIMUM_CONCURRENCY - Invalid maximum concurrency.

    PH_E_INVALID_KEYS_LOAD_FLAGS - Invalid keys load flags.

    PH_E_INVALID_CONTEXT_CREATE_TABLE_FLAGS - Invalid context create table
        flags.

    PH_E_INVALID_TABLE_LOAD_FLAGS - Invalid table load flags.

    PH_E_INVALID_TABLE_COMPILE_FLAGS - Invalid table compile flags.

--*/
{
    HRESULT Result;
    HRESULT CleanupResult;
    UNICODE_STRING TestDataDirectory = { 0 };
    UNICODE_STRING BaseOutputDirectory = { 0 };
    PERFECT_HASH_ALGORITHM_ID AlgorithmId = 0;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId = 0;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId = 0;
    ULONG MaximumConcurrency = 0;
    PERFECT_HASH_CONTEXT_SELF_TEST_FLAGS ContextSelfTestFlags = { 0 };
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags = { 0 };
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags = { 0 };
    PERFECT_HASH_TABLE_LOAD_FLAGS TableLoadFlags = { 0 };
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags = { 0 };
    PPERFECT_HASH_CONTEXT_EXTRACT_SELF_TEST_ARGS_FROM_ARGVW ExtractSelfTestArgs;
    PERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters;

    TableCreateParameters.SizeOfStruct = sizeof(TableCreateParameters);
    TableCreateParameters.NumberOfElements = 0;
    TableCreateParameters.Allocator = Context->Allocator;
    TableCreateParameters.Params = NULL;

    ExtractSelfTestArgs = Context->Vtbl->ExtractSelfTestArgsFromArgvW;
    Result = ExtractSelfTestArgs(Context,
                                 NumberOfArguments,
                                 ArgvW,
                                 &TestDataDirectory,
                                 &BaseOutputDirectory,
                                 &AlgorithmId,
                                 &HashFunctionId,
                                 &MaskFunctionId,
                                 &MaximumConcurrency,
                                 &ContextSelfTestFlags,
                                 &KeysLoadFlags,
                                 &TableCreateFlags,
                                 &TableLoadFlags,
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
            PH_ERROR(PerfectHashContextContextSelfTestArgvW, Result);
            return Result;
        }
    }

    PerfectHashContextApplyThreadpoolPriorities(Context,
                                                &TableCreateParameters);

    Result = Context->Vtbl->SelfTest(Context,
                                     &TestDataDirectory,
                                     &BaseOutputDirectory,
                                     AlgorithmId,
                                     HashFunctionId,
                                     MaskFunctionId,
                                     &ContextSelfTestFlags,
                                     &KeysLoadFlags,
                                     &TableCreateFlags,
                                     &TableLoadFlags,
                                     &TableCompileFlags,
                                     &TableCreateParameters);

    if (FAILED(Result)) {

        //
        // We don't take any action here, there's not much we can do.
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
