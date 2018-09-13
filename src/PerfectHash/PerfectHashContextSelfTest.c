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
    PPERFECT_HASH_CONTEXT_SELF_TEST_FLAGS SelfTestFlagsPointer,
    PPERFECT_HASH_KEYS_LOAD_FLAGS LoadKeysFlagsPointer,
    PPERFECT_HASH_CONTEXT_CREATE_TABLE_FLAGS CreateTableFlagsPointer,
    PPERFECT_HASH_TABLE_LOAD_FLAGS LoadTableFlagsPointer,
    PCUNICODE_STRING TestDataDirectory,
    PCUNICODE_STRING OutputDirectory,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId
    )
/*++

Routine Description:

    Performs a self-test of the entire PerfectHash component.

Arguments:

    Context - Supplies an instance of PERFECT_HASH_CONTEXT.

    SelfTestFlags - Optionally supplies a pointer to a self-test flags
        structure that can be used to customize self-test behavior.

    LoadKeysFlags - Optionally supplies a pointer to a key loading flags
        structure that can be used to customize key loading behavior.

    CreateTableFlags - Optionally supplies a pointer to a create table
        flags structure that can be used to customize table creation
        behavior.

    LoadTableFlags - Optionally supplies a pointer to a load table
        flags structure that can be used to customize table loading
        behavior.

    TestDataDirectory - Supplies a pointer to a UNICODE_STRING structure that
        represents a fully-qualified path of the test data directory.

    OutputDirectory - Supplies a pointer to a UNICODE_STRING structure that
        represents a fully-qualified path of the directory where the perfect
        hash table files generated as part of this routine will be saved.

    AlgorithmId - Supplies the algorithm to use.

    MaskFunctionId - Supplies the type of masking to use.

    HashFunctionId - Supplies the hash function to use.

Return Value:

    S_OK - Self test performed successfully.

    An apppropriate error code otherwise.

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
    USHORT FileNameLengthInBytes;
    BOOL Success;
    BOOLEAN Failed;
    BOOLEAN Terminate;
    HRESULT Result;
    PWCHAR Buffer;
    PWCHAR BaseBuffer;
    PWCHAR WideOutput;
    PWCHAR WideOutputBuffer;
    PWCHAR FileName;
    PVOID KeysBaseAddress;
    ULARGE_INTEGER NumberOfKeys;
    WCHAR WideBitmapString[33];
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
    UNICODE_STRING KeysPath;
    UNICODE_STRING TablePath;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_KEYS Keys;
    PERFECT_HASH_KEYS_FLAGS KeysFlags;
    PERFECT_HASH_KEYS_BITMAP KeysBitmap;
    PERFECT_HASH_TABLE_FLAGS TableFlags;
    PTABLE_INFO_ON_DISK_HEADER Header;
    PCUNICODE_STRING Suffix = &KeysWildcardSuffix;
    PERFECT_HASH_TLS_CONTEXT TlsContext;
    PUNICODE_STRING AlgorithmName;
    PUNICODE_STRING HashFunctionName;
    PUNICODE_STRING MaskFunctionName;
    PERFECT_HASH_CONTEXT_SELF_TEST_FLAGS SelfTestFlags;
    PERFECT_HASH_CONTEXT_CREATE_TABLE_FLAGS CreateTableFlags;
    PERFECT_HASH_KEYS_LOAD_FLAGS LoadKeysFlags;
    PERFECT_HASH_TABLE_LOAD_FLAGS LoadTableFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    } else {
        Rtl = Context->Rtl;
    }

    if (ARGUMENT_PRESENT(SelfTestFlagsPointer)) {
        if (FAILED(IsValidContextSelfTestFlags(SelfTestFlagsPointer))) {
            return PH_E_INVALID_CONTEXT_SELF_TEST_FLAGS;
        } else {
            SelfTestFlags.AsULong = SelfTestFlagsPointer->AsULong;
        }
    } else {
        SelfTestFlags.AsULong = 0;
    }

    if (ARGUMENT_PRESENT(LoadKeysFlagsPointer)) {
        if (FAILED(IsValidKeysLoadFlags(LoadKeysFlagsPointer))) {
            return PH_E_INVALID_KEYS_LOAD_FLAGS;
        } else {
            LoadKeysFlags.AsULong = LoadKeysFlagsPointer->AsULong;
        }
    } else {
        LoadKeysFlags.AsULong = 0;
    }

    if (ARGUMENT_PRESENT(CreateTableFlagsPointer)) {
        if (FAILED(IsValidContextCreateTableFlags(CreateTableFlagsPointer))) {
            return PH_E_INVALID_CONTEXT_CREATE_TABLE_FLAGS;
        } else {
            CreateTableFlags.AsULong = CreateTableFlagsPointer->AsULong;
        }
    } else {
        CreateTableFlags.AsULong = 0;
    }

    if (ARGUMENT_PRESENT(LoadTableFlagsPointer)) {
        if (FAILED(IsValidTableLoadFlags(LoadTableFlagsPointer))) {
            return PH_E_INVALID_TABLE_LOAD_FLAGS;
        } else {
            LoadTableFlags.AsULong = LoadTableFlagsPointer->AsULong;
        }
    } else {
        LoadTableFlags.AsULong = 0;
    }

    if (!ARGUMENT_PRESENT(TestDataDirectory)) {
        return E_POINTER;
    } else if (!IsValidMinimumDirectoryUnicodeString(TestDataDirectory)) {
        return E_INVALIDARG;
    }

    if (!ARGUMENT_PRESENT(OutputDirectory)) {
        return E_POINTER;
    } else if (!IsValidMinimumDirectoryUnicodeString(OutputDirectory)) {
        return E_INVALIDARG;
    } else {
        if (!CreateDirectoryW(OutputDirectory->Buffer, NULL)) {
            LastError = GetLastError();
            if (LastError != ERROR_ALREADY_EXISTS) {
                SYS_ERROR(CreateDirectoryW);
                return E_INVALIDARG;
            }
        }
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
        SYS_ERROR(VirtualAlloc);
        Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                          ProcessHandle,
                                          &WideOutputBuffer);
        if (FAILED(Result)) {
            SYS_ERROR(VirtualFree);
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
    ASSERT(WildcardPath.Buffer[WildcardPath.Length] == L'\0');

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

    if (!FindHandle || FindHandle == INVALID_HANDLE_VALUE) {

        //
        // Check to see if we failed because there were no files matching the
        // wildcard *.keys in the test directory.  In this case, GetLastError()
        // will report ERROR_FILE_NOT_FOUND.
        //

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

    KeysPath.Buffer = Buffer;
    CopyMemory(KeysPath.Buffer, TestDataDirectory->Buffer, Length);

    //
    // Advance our Dest pointer to the end of the directory name, then write
    // a slash.
    //

    Dest = (PWSTR)RtlOffsetToPointer(KeysPath.Buffer, Length);
    *Dest++ = L'\\';

    //
    // Update the length to account for the slash we just wrote, then make a
    // copy of it in the variable BaseLength.
    //

    Length += sizeof(*Dest);
    BaseLength = Length;

    //
    // Zero the local TLS context structure, fill out the relevant fields,
    // then set it.  This will allow other components to re-use our Allocator
    // and Rtl components (this is handled in our COM creation logic).
    //

    ZeroStruct(TlsContext);
    TlsContext.Rtl = Context->Rtl;
    TlsContext.Allocator = Context->Allocator;
    if (!PerfectHashTlsSetContext(&TlsContext)) {
        SYS_ERROR(TlsSetValue);
        goto Error;
    }

    //
    // Zero the failure count and terminate flag, zero the bitmap structure,
    // wire up the unicode string representation of the bitmap, initialize
    // various flags, and begin the main loop.
    //

    Failures = 0;
    Terminate = FALSE;
    ZeroStruct(KeysBitmap);
    UnicodeBitmapString.Buffer = (PWCHAR)WideBitmapString;
    UnicodeBitmapString.Length = sizeof(WideBitmapString)-2;
    UnicodeBitmapString.MaximumLength = sizeof(WideBitmapString);
    UnicodeBitmapString.Buffer[UnicodeBitmapString.Length >> 1] = L'\0';
    KeysBaseAddress = NULL;
    NumberOfKeys.QuadPart = 0;
    KeysFlags.AsULong = 0;

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

        Dest = (PWSTR)RtlOffsetToPointer(KeysPath.Buffer, BaseLength);
        Source = (PWSTR)FindData.cFileName;

        while (*Source) {
            *Dest++ = *Source++;
        }
        *Dest = L'\0';

        Length = (USHORT)RtlPointerToOffset(KeysPath.Buffer, Dest);
        KeysPath.Length = Length;
        KeysPath.MaximumLength = Length + sizeof(*Dest);
        ASSERT(KeysPath.Buffer[KeysPath.Length >> 1] == L'\0');
        ASSERT(&KeysPath.Buffer[KeysPath.Length >> 1] == Dest);

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
                                  &LoadKeysFlags,
                                  &KeysPath,
                                  sizeof(ULONG));

        if (FAILED(Result)) {

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to load keys for ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Successfully loaded keys: ");
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPath);
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
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPath);
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
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPath);
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
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPath);
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
                                  &LoadKeysFlags,
                                  &KeysPath,
                                  sizeof(ULONG));

        if (Result != PH_E_KEYS_ALREADY_LOADED) {
            WIDE_OUTPUT_RAW(WideOutput, L"Invariant failed; multiple "
                                        L"key loads did not raise an "
                                        L"error for ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        //
        // Keys were loaded successfully.  Construct the equivalent path name
        // for the backing perfect hash table when persisted to disk.
        //

        //
        // Align the Dest pointer up to a 16-byte boundary.  (We add 1 to its
        // current value to advance it past the terminating NULL of the keys
        // path.)
        //

        Dest = (PWSTR)ALIGN_UP(Dest + 1, 16);

        //
        // Calculate the length in bytes of the final table path name, which
        // consists of the output directory length, plus a joining slash, plus
        // the file name part of the keys file, plus the trailing table suffix.
        //

        FileNameLengthInBytes = (
            KeysPath.Length -
            TestDataDirectory->Length -
            DotKeysSuffix.Length
        );

        FileName = (PWCHAR)(
            RtlOffsetToPointer(
                KeysPath.Buffer,
                TestDataDirectory->Length + sizeof(WCHAR)
            )
        );

        ASSERT(*FileName != L'\\');
        ASSERT(*(FileName - 1) == L'\\');

        TablePath.Length = (

            OutputDirectory->Length +

            //
            // Account for the joining slash.
            //

            sizeof(WCHAR) +

            //
            // Account for the file name, including the period.
            //

            FileNameLengthInBytes +

            //
            // Account for the table suffix.
            //

            TableSuffix.Length
        );

        TablePath.MaximumLength = TablePath.Length + sizeof(WCHAR);
        TablePath.Buffer = (PWSTR)Dest;

        //
        // Copy the output directory and trailing slash.
        //

        CopyMemory(Dest, OutputDirectory->Buffer, OutputDirectory->Length);
        Dest += ((ULONG_PTR)OutputDirectory->Length >> 1);
        *Dest++ = L'\\';

        //
        // Copy the filename.
        //

        CopyMemory(Dest, FileName, FileNameLengthInBytes);
        Dest += ((ULONG_PTR)FileNameLengthInBytes >> 1);
        ASSERT(*(Dest - 1) == L'.');

        //
        // Copy the suffix then null terminate the path.
        //

        Source = TableSuffix.Buffer;
        while (*Source) {
            *Dest++ = *Source++;
        }
        *Dest = L'\0';

        //
        // Sanity check invariants.
        //

        ASSERT(TablePath.Buffer[TablePath.Length >> 1] == L'\0');
        ASSERT(&TablePath.Buffer[TablePath.Length >> 1] == Dest);

        //
        // We now have the fully-qualified path name of the backing perfect
        // hash table file living in TablePath.  Continue with creation of the
        // perfect hash table, using this path we've just created and the keys
        // that were loaded.
        //

        Result = Context->Vtbl->CreateTable(Context,
                                            &CreateTableFlags,
                                            AlgorithmId,
                                            MaskFunctionId,
                                            HashFunctionId,
                                            Keys,
                                            &TablePath);

        if (FAILED(Result)) {

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to create perfect hash "
                                        L"table for keys: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failed = TRUE;
            Failures++;
            goto ReleaseKeys;
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Successfully created perfect "
                                    L"hash table: ");
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, &TablePath);
        WIDE_OUTPUT_RAW(WideOutput, L".\n");

        //
        // Load the perfect hash table we just created.
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

        Result = Table->Vtbl->Load(Table,
                                   &LoadTableFlags,
                                   &TablePath,
                                   Keys);

        if (FAILED(Result)) {

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to load perfect hash table: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &TablePath);
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
                                   &LoadTableFlags,
                                   &TablePath,
                                   Keys);

        if (Result != PH_E_TABLE_ALREADY_LOADED) {
            WIDE_OUTPUT_RAW(WideOutput, L"Invariant failed; multiple "
                                        L"table loads did not raise an "
                                        L"error for ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &TablePath);
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
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, &TablePath);
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
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &TablePath);
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

        //
        // Test the table.
        //

        Result = Table->Vtbl->Test(Table, Keys, TRUE);

        if (FAILED(Result)) {

            WIDE_OUTPUT_RAW(WideOutput, L"Test failed for perfect hash table "
                                        L"loaded from disk: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &TablePath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Failed = TRUE;
            goto ReleaseTable;
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Successfully tested perfect hash "
                                    L"table.\n");

        //
        // Initialize header alias.
        //

        Header = Table->Header;

        //
        // Define some helper macros here for dumping stats.
        //

#define STATS_INT(String, Name)                               \
        WIDE_OUTPUT_RAW(WideOutput, String);                  \
        WIDE_OUTPUT_INT(WideOutput, Table->Header->##Name##); \
        WIDE_OUTPUT_RAW(WideOutput, L".\n")

#define STATS_QUAD(String, Name)                                       \
        WIDE_OUTPUT_RAW(WideOutput, String);                           \
        WIDE_OUTPUT_INT(WideOutput, Table->Header->##Name##.QuadPart); \
        WIDE_OUTPUT_RAW(WideOutput, L".\n")

        if (Header->NumberOfTableResizeEvents > 0) {

            STATS_INT(L"Number of table resize events: ",
                      NumberOfTableResizeEvents);

            STATS_INT(L"Total number of attempts with smaller table sizes: ",
                      TotalNumberOfAttemptsWithSmallerTableSizes);

            STATS_INT(L"First table size attempted: ",
                      InitialTableSize);

            STATS_INT(L"Closest we came to solving the graph in previous "
                      L"attempts by number of deleted edges away: ",
                      ClosestWeCameToSolvingGraphWithSmallerTableSizes);

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
        STATS_QUAD(L"Cycles to prepare file: ", PrepareFileCycles);
        STATS_QUAD(L"Cycles to save file: ", SaveFileCycles);

        STATS_QUAD(L"Microseconds to solve: ", SolveMicroseconds);
        STATS_QUAD(L"Microseconds to verify: ", VerifyMicroseconds);
        STATS_QUAD(L"Microseconds to prepare file: ", PrepareFileMicroseconds);
        STATS_QUAD(L"Microseconds to save file: ", SaveFileMicroseconds);


        WIDE_OUTPUT_RAW(WideOutput, L"\n\n");
        WIDE_OUTPUT_FLUSH();

        //
        // Release the table and keys.
        //

ReleaseTable:

        Table->Vtbl->Release(Table);

ReleaseKeys:

        Keys->Vtbl->Release(Keys);

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

    if (!PerfectHashTlsSetContext(NULL)) {
        SYS_ERROR(TlsSetValue);
    }

    if (WideOutputBuffer) {
        Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                          ProcessHandle,
                                          &WideOutputBuffer);
        if (FAILED(Result)) {
            SYS_ERROR(VirtualFree);
        }
        WideOutput = NULL;
    }

    if (BaseBuffer) {
        Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                          ProcessHandle,
                                          &BaseBuffer);
        if (FAILED(Result)) {
            SYS_ERROR(VirtualFree);
        }
        BaseBuffer = NULL;
    }

    if (FindHandle) {
        if (!FindClose(FindHandle)) {
            NOTHING;
        }
        FindHandle = NULL;
    }

    return Result;
}

//
// Commandline support.
//

const STRING Usage = RTL_CONSTANT_STRING(
    "Usage: PerfectHashSelfTest.exe "
    "<TestDataDirectory (must be fully-qualified)> "
    "<OutputDirectory (must be fully-qualified)> "
    "<AlgorithmId> "
    "<HashFunctionId> "
    "<MaskFunctionId> "
    "<MaximumConcurrency (0-ncpu)> "
    "[PauseBeforeExit (can be any character)]\n"
    "E.g.: PerfectHashSelfTest.exe "
    "C:\\Users\\Trent\\Home\\src\\perfecthash\\data "
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
    } else if (!IsValidPerfectHash##Name##Id(*##Name##Id)) {     \
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
    PUNICODE_STRING OutputDirectory,
    PPERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PPERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PULONG MaximumConcurrency,
    PPERFECT_HASH_CONTEXT_SELF_TEST_FLAGS SelfTestFlags,
    PPERFECT_HASH_KEYS_LOAD_FLAGS LoadKeysFlags,
    PPERFECT_HASH_CONTEXT_CREATE_TABLE_FLAGS CreateTableFlags,
    PPERFECT_HASH_TABLE_LOAD_FLAGS LoadTableFlags
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

    OutputDirectory - Supplies a pointer to a UNICODE_STRING structure that
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

    LoadKeysFlags - Supplies the address of a variable that will receive
        the key loading flags.

    CreateTableFlags - Supplies the address of a variable that will receive
        the create table flags.

    LoadTableFlags - Supplies the address of a variable that will receive the
        the load table flags.

Return Value:

    S_OK on success.

    E_POINTER if any pointer arguments are NULL.

    PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS if NumberOfArguments is not a valid
        value.

    PH_E_INVALID_ALGORITHM_ID if algorithm ID is invalid.

    PH_E_INVALID_HASH_FUNCTION_ID if hash function ID is invalid.

    PH_E_INVALID_MASK_FUNCTION_ID if mask function ID is invalid.

    PH_E_INVALID_MAXIMUM_CONCURRENCY is maximum concurrency value is invalid.

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

    if (!ARGUMENT_PRESENT(OutputDirectory)) {
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

    if (!ARGUMENT_PRESENT(SelfTestFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(LoadKeysFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(CreateTableFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(LoadTableFlags)) {
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

    OutputDirectory->Buffer = *ArgW++;
    OutputDirectory->Length = GET_LENGTH(OutputDirectory);

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

    OutputDirectory->Buffer = *ArgW++;
    OutputDirectory->Length = GET_LENGTH(OutputDirectory);
    OutputDirectory->MaximumLength = GET_MAX_LENGTH(OutputDirectory);

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

    if (NumberOfArguments == 8) {
        SelfTestFlags->PauseBeforeExit = TRUE;
    }

    //
    // Remaining flags not yet supported.
    //

    LoadKeysFlags->AsULong = 0;
    CreateTableFlags->AsULong = 0;
    LoadTableFlags->AsULong = 0;

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

    S_OK on success.

    E_POINTER if any pointer arguments are NULL.

    PH_E_CONTEXT_SELF_TEST_INVALID_NUM_ARGS if NumberOfArguments is not a valid
        value.

    PH_E_INVALID_ALGORITHM_ID if algorithm ID is invalid.

    PH_E_INVALID_HASH_FUNCTION_ID if hash function ID is invalid.

    PH_E_INVALID_MASK_FUNCTION_ID if mask function ID is invalid.

    PH_E_INVALID_MAXIMUM_CONCURRENCY is maximum concurrency value is invalid.

    PH_E_CREATE_TABLE_ALREADY_IN_PROGRESS if a create table is in progress.

    PH_E_SET_MAXIMUM_CONCURRENCY_FAILED if setting the context's maximum
        concurrency failed.

--*/
{
    PRTL Rtl;
    HRESULT Result;
    UNICODE_STRING TestDataDirectory = { 0 };
    UNICODE_STRING OutputDirectory = { 0 };
    PERFECT_HASH_ALGORITHM_ID AlgorithmId = 0;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId = 0;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId = 0;
    ULONG MaximumConcurrency = 0;
    PERFECT_HASH_CONTEXT_SELF_TEST_FLAGS SelfTestFlags = { 0 };
    PERFECT_HASH_KEYS_LOAD_FLAGS LoadKeysFlags = { 0 };
    PERFECT_HASH_CONTEXT_CREATE_TABLE_FLAGS CreateTableFlags = { 0 };
    PERFECT_HASH_TABLE_LOAD_FLAGS LoadTableFlags = { 0 };

    Rtl = Context->Rtl;

    Result = Context->Vtbl->ExtractSelfTestArgsFromArgvW(Context,
                                                         NumberOfArguments,
                                                         ArgvW,
                                                         &TestDataDirectory,
                                                         &OutputDirectory,
                                                         &AlgorithmId,
                                                         &HashFunctionId,
                                                         &MaskFunctionId,
                                                         &MaximumConcurrency,
                                                         &SelfTestFlags,
                                                         &LoadKeysFlags,
                                                         &CreateTableFlags,
                                                         &LoadTableFlags);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashContextSelfTestArgvW, Result);
        return Result;
    }

    if (MaximumConcurrency > 0) {
        Result = Context->Vtbl->SetMaximumConcurrency(Context,
                                                      MaximumConcurrency);
        if (FAILED(Result)) {
            Result = PH_E_SET_MAXIMUM_CONCURRENCY_FAILED;
            PH_ERROR(PerfectHashContextSelfTestArgvW, Result);
            return Result;
        }
    }

    Result = Context->Vtbl->SelfTest(Context,
                                     &SelfTestFlags,
                                     &LoadKeysFlags,
                                     &CreateTableFlags,
                                     &LoadTableFlags,
                                     &TestDataDirectory,
                                     &OutputDirectory,
                                     AlgorithmId,
                                     HashFunctionId,
                                     MaskFunctionId);

    if (FAILED(Result)) {
        NOTHING;
    }

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
