/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableContextSelfTest.c

Abstract:

    This module implements the self-test routine for the PerfectHashTable
    component.  It is responsible for end-to-end testing of the entire
    component with all known test data from a single function entry point.

--*/

#include "stdafx.h"

PERFECT_HASH_TABLE_CONTEXT_SELF_TEST PerfectHashTableContextSelfTest;

_Use_decl_annotations_
HRESULT
PerfectHashTableContextSelfTest(
    PPERFECT_HASH_TABLE_CONTEXT Context,
    PCUNICODE_STRING TestDataDirectory,
    PCUNICODE_STRING OutputDirectory,
    PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId
    )
/*++

Routine Description:

    Performs a self-test of the entire PerfectHashTable component.

Arguments:

    Context - Supplies an instance of PERFECT_HASH_TABLE_CONTEXT.

    TestDataDirectory - Supplies a pointer to a UNICODE_STRING structure that
        represents a fully-qualified path of the test data directory.

    OutputDirectory - Supplies a pointer to a UNICODE_STRING structure that
        represents a fully-qualified path of the directory where the perfect
        hash table files generated as part of this routine will be saved.

    AlgorithmId - Supplies the algorithm to use.

    MaskFunctionId - Supplies the type of masking to use.

    HashFunctionId - Supplies the hash function to use.

Return Value:

    S_OK on success, an appropriate error code on failure.

--*/
{
    PRTL Rtl;
    PWSTR Dest;
    PWSTR Source;
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
    PPERFECT_HASH_TABLE_KEYS Keys;
    PTABLE_INFO_ON_DISK_HEADER Header;
    PCUNICODE_STRING Suffix = &KeysWildcardSuffix;
    //PUNICODE_STRING AlgorithmName;
    //PUNICODE_STRING HashFunctionName;
    //PUNICODE_STRING MaskFunctionName;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    } else {
        Rtl = Context->Rtl;
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
            SYS_ERROR(CreateDirectoryW);
            return E_INVALIDARG;
        }
    }

    if (!IsValidPerfectHashTableAlgorithmId(AlgorithmId)) {
        return E_INVALIDARG;
    }

    if (!IsValidPerfectHashTableHashFunctionId(HashFunctionId)) {
        return E_INVALIDARG;
    }

    if (!IsValidPerfectHashTableMaskFunctionId(MaskFunctionId)) {
        return E_INVALIDARG;
    }

    //
    // Arguments have been validated, proceed.
    //

    //
    // Create a buffer we can use for stdout, using a very generous buffer size.
    //

    NumberOfPages = 10;

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
        if (!Rtl->Vtbl->DestroyBuffer(Rtl, ProcessHandle, &WideOutputBuffer)) {
            NOTHING;
        }
        return Result;
    }

    Buffer = BaseBuffer;

    //
    // Get a reference to the stdout handle.
    //

    WideOutputHandle = GetStdHandle(STD_OUTPUT_HANDLE);
    ASSERT(WideOutputHandle);

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

        ULONG LastError;

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
    // Zero the failure count and terminate flag and begin the main loop.
    //

    Failures = 0;
    Terminate = FALSE;

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

        Length = (USHORT)RtlPointerToOffset(Dest, KeysPath.Buffer);
        KeysPath.Length = Length;
        KeysPath.MaximumLength = Length + sizeof(*Dest);
        ASSERT(KeysPath.Buffer[KeysPath.Length >> 1] == L'\0');
        ASSERT(&KeysPath.Buffer[KeysPath.Length >> 1] == Dest);

        Result = Context->Vtbl->CreateInstance(Context,
                                               NULL,
                                               &IID_PERFECT_HASH_TABLE_KEYS,
                                               &Keys);

        if (FAILED(Result)) {
            WIDE_OUTPUT_RAW(WideOutput, L"Failed to create keys instance.\n");
            WIDE_OUTPUT_FLUSH();
            Failures++;
            break;
        }

        Result = Keys->Vtbl->Load(Keys, &KeysPath);

        if (FAILED(Result)) {

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to load keys for ");
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
            ((USHORT)wcslen(L"keys") << 1) -
            TestDataDirectory->Length
        );

        FileName = (PWCHAR)(
            RtlOffsetToPointer(
                TestDataDirectory->Buffer,
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
        *Dest++ = L'\\';

        //
        // Copy the filename.
        //

        CopyMemory(Dest, FileName, FileNameLengthInBytes);
        Dest += (FileNameLengthInBytes << 1);
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

        Result = Table->Vtbl->Load(Table, &TablePath, Keys);

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
        // Table was loaded successfully from disk.  Obtain the names of all
        // the enumeration IDs.  Currently these should always match the same
        // enums provided as input parameters to this routine.
        //

#define GET_NAME(Desc)                                                   \
        Result = Table->Vtbl->Get##Desc##Name(Table, &##Desc##Name);     \
        if (FAILED(Result)) {                                            \
            WIDE_OUTPUT_RAW(WideOutput,                                  \
                            (PCWCHAR)L"Get" #Desc "Name() failed.\n");   \
            Terminate = TRUE;                                            \
            goto ReleaseTable;                                           \
        }

        //GET_NAME(Algorithm);
        //GET_NAME(HashFunction);
        //GET_NAME(MaskFunction);

        WIDE_OUTPUT_RAW(WideOutput, L"Successfully loaded perfect "
                                    L"hash table: ");
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, &TablePath);
        WIDE_OUTPUT_RAW(WideOutput, L".\n");

#if 0
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
#endif

        WIDE_OUTPUT_RAW(WideOutput, L"Table data backed by large pages: ");
        if (Table->Flags.TableDataUsesLargePages) {
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &Yes);
        } else {
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &No);
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Values array allocated with large "
                                    L"pages: ");
        if (Table->Flags.ValuesArrayUsesLargePages) {
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

    //
    // We can't do much if any of these routines error out, hence the NOTHINGs.
    //

    if (WideOutputBuffer) {
        if (!Rtl->Vtbl->DestroyBuffer(Rtl, ProcessHandle, &WideOutputBuffer)) {
            NOTHING;
        }
        WideOutput = NULL;
    }

    if (BaseBuffer) {
        if (!Rtl->Vtbl->DestroyBuffer(Rtl, ProcessHandle, &BaseBuffer)) {
            NOTHING;
        }
        Buffer = NULL;
    }

    if (FindHandle) {
        if (!FindClose(FindHandle)) {
            NOTHING;
        }
        FindHandle = NULL;
    }

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
