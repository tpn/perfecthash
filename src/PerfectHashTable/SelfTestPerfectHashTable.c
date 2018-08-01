/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    SelfTestPerfectHashTable.c

Abstract:

    This module implements the self-test routine for the PerfectHashTable
    component.  It is responsible for end-to-end testing of the entire
    component with all known test data from a single function entry point
    (SelfTestPerfectHashTable()).

--*/

#include "stdafx.h"

_Use_decl_annotations_
BOOLEAN
SelfTestPerfectHashTable(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PPERFECT_HASH_TABLE_ANY_API AnyApi,
    PCUNICODE_STRING TestDataDirectory,
    PULONG MaximumConcurrency,
    PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId
    )
/*++

Routine Description:

    Performs a self-test of the entire PerfectHashTable component.

Arguments:

    Rtl - Supplies a pointer to an initialized RTL structure.

    Allocator - Supplies a pointer to an initialized ALLOCATOR structure that
        will be used for all memory allocations.

    AnyApi - Supplies a pointer to an initialized PERFECT_HASH_TABLE_ANY_API
        structure.  Note that this must be an instance of the extended API;
        this is verified by looking at the Api->SizeOfStruct field and ensuring
        it matches our expected size of the extended API structure.

    TestDataDirectory - Supplies a pointer to a UNICODE_STRING structure that
        represents a fully-qualified path of the test data directory.

    MaximumConcurrency - Optionally supplies a pointer to a variable that
        contains the desired maximum concurrency to be used for the underlying
        threadpool.  If NULL, or non-NULL but points to a value of 0, then the
        number of system processors will be used as a default value.

        N.B. This value is passed directly to SetThreadpoolThreadMinimum() and
             SetThreadpoolThreadMaximum().

    AlgorithmId - Supplies the algorithm to use.

    MaskFunctionId - Supplies the type of masking to use.

    HashFunctionId - Supplies the hash function to use.

Return Value:

    TRUE on success, FALSE on failure.

--*/
{
    PWSTR Dest;
    PWSTR Source;
    USHORT Length;
    USHORT BaseLength;
    USHORT NumberOfPages;
    BOOLEAN Success;
    BOOLEAN Failed;
    BOOLEAN IsProcessTerminating = FALSE;
    PWCHAR Buffer;
    PWCHAR BaseBuffer;
    PWCHAR WideOutput;
    PWCHAR WideOutputBuffer;
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
    UNICODE_STRING SearchPath;
    UNICODE_STRING KeysPath;
    UNICODE_STRING TablePath;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_KEYS Keys;
    PTABLE_INFO_ON_DISK_HEADER Header;
    PPERFECT_HASH_TABLE_CONTEXT Context;
    PPERFECT_HASH_TABLE_API Api;
    PPERFECT_HASH_TABLE_API_EX ApiEx;
    PCUNICODE_STRING Suffix = &KeysWildcardSuffix;
    PUNICODE_STRING AlgorithmName;
    PUNICODE_STRING HashFunctionName;
    PUNICODE_STRING MaskFunctionName;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(Allocator)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(AnyApi)) {

        return FALSE;

    } else {

        Api = &AnyApi->Api;

        if (Api->SizeOfStruct == sizeof(*ApiEx)) {

            ApiEx = &AnyApi->ApiEx;

        } else {

            //
            // The API should be the extended version.  (Otherwise, how did
            // the caller even find this function?)
            //

            return FALSE;
        }

    }

    if (!ARGUMENT_PRESENT(TestDataDirectory)) {
        return FALSE;
    }

    if (!IsValidMinimumDirectoryUnicodeString(TestDataDirectory)) {
        return FALSE;
    }

    if (!IsValidPerfectHashTableAlgorithmId(AlgorithmId)) {

        return FALSE;

    } else {


    }

    //
    // Arguments have been validated, proceed.
    //

    //
    // Create a buffer we can use for stdout, using a very generous buffer size.
    //

    NumberOfPages = 10;
    Success = Rtl->CreateBuffer(Rtl,
                                &ProcessHandle,
                                NumberOfPages,
                                NULL,
                                &WideOutputBufferSize,
                                &WideOutputBuffer);

    if (!Success) {
        return FALSE;
    }

    WideOutput = WideOutputBuffer;

    //
    // Create a buffer we can use for temporary path construction.  We want it
    // to be MAX_USHORT in size, so (1 << 16) >> PAGE_SHIFT converts this into
    // the number of pages we need.
    //

    NumberOfPages = (1 << 16) >> PAGE_SHIFT;
    Success = Rtl->CreateBuffer(Rtl,
                                &ProcessHandle,
                                NumberOfPages,
                                NULL,
                                &BufferSize,
                                &BaseBuffer);

    if (!Success) {
        return FALSE;
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

    SearchPath.Buffer = (PWSTR)Buffer;

    if (!SearchPath.Buffer) {
        goto Error;
    }

    //
    // Copy incoming test data directory name.
    //

    Length = TestDataDirectory->Length;
    CopyMemory(SearchPath.Buffer,
               TestDataDirectory->Buffer,
               Length);

    //
    // Advance our Dest pointer to the end of the directory name, write a
    // slash, then copy the suffix over.
    //

    Dest = (PWSTR)RtlOffsetToPointer(SearchPath.Buffer, Length);
    *Dest++ = L'\\';
    CopyMemory(Dest, Suffix->Buffer, Suffix->Length);

    //
    // Wire up the search path length and maximum length variables.  The max
    // length will be our AllocSize, length will be this value minus 2 to
    // account for the trailing NULL.
    //

    SearchPath.MaximumLength = AllocSize.LowPart;
    SearchPath.Length = AllocSize.LowPart - sizeof(*Dest);
    ASSERT(SearchPath.Buffer[SearchPath.Length] == L'\0');

    //
    // Advance the buffer past this string allocation, up to the next 16-byte
    // boundary.
    //

    Buffer = (PWSTR)(
        RtlOffsetToPointer(
            Buffer,
            ALIGN_UP(SearchPath.MaximumLength, 16)
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

    FindHandle = FindFirstFileW(SearchPath.Buffer, &FindData);

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
    // Zero the failure count and begin the main loop.
    //

    Failures = 0;

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
        // Create a new perfect hash table context.
        //

        Success = Api->CreatePerfectHashTableContext(Rtl,
                                                     Allocator,
                                                     MaximumConcurrency,
                                                     &Context);

        if (!Success) {

            //
            // We can't do anything without a context.
            //

            WIDE_OUTPUT_RAW(WideOutput, L"Fatal: failed to create context.\n");
            WIDE_OUTPUT_FLUSH();
            Failures++;
            break;
        }

        //
        // Copy the filename over to the fully-qualified keys path.
        //

        Dest = (PWSTR)RtlOffsetToPointer(KeysPath.Buffer, BaseLength);
        Source = (PWSTR)FindData.cFileName;

        while (*Source) {
            *Dest++ = *Source++;
        }
        *Dest = L'\0';

        Length = (USHORT)RtlOffsetFromPointer(Dest, KeysPath.Buffer);
        KeysPath.Length = Length;
        KeysPath.MaximumLength = Length + sizeof(*Dest);
        ASSERT(KeysPath.Buffer[KeysPath.Length >> 1] == L'\0');
        ASSERT(&KeysPath.Buffer[KeysPath.Length >> 1] == Dest);

        Success = Api->LoadPerfectHashTableKeys(Rtl,
                                                Allocator,
                                                &KeysPath,
                                                &Keys);

        if (!Success) {

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to load keys for ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            goto DestroyContext;
        }

        //
        // Keys were loaded successfully.  Construct the equivalent path name
        // for the backing perfect hash table when persisted to disk.  Although
        // this can be automated for us as part of CreatePerfectHashTable(),
        // having it backed by our memory simplifies things a little further
        // down the track when we want to load the table via the path but have
        // destroyed the original table that came from CreatePerfectHashTable().
        //

        //
        // Align the Dest pointer up to a 16-byte boundary.  (We add 1 to its
        // current value to advance it past the terminating NULL of the keys
        // path.)
        //

        Dest = (PWSTR)ALIGN_UP(Dest + 1, 16);

        //
        // We know the keys file ended with ".keys".  We're going to use the
        // identical name for the backing hash table file, except replace the
        // ".keys" extension at the end with ".pht1".  So, we can just copy the
        // lengths used for KeysPath, plus the entire buffer, then just copy the
        // new extension over the old one.  The 1 doesn't have any significance
        // other than it padding out the extension length such that it matches
        // the length of ".keys".  (Although it may act as a nice versioning
        // tool down the track.)
        //

        TablePath.Length = KeysPath.Length;
        TablePath.MaximumLength = KeysPath.MaximumLength;
        TablePath.Buffer = Dest;

        //
        // Copy the keys path over.
        //

        CopyMemory(Dest, KeysPath.Buffer, KeysPath.MaximumLength);

        //
        // Advance the Dest pointer to the end of the buffer, then retreat it
        // five characters, such that it's positioned on the 'k' of keys.
        //

        Dest += (KeysPath.MaximumLength >> 1) - 5;
        ASSERT(*Dest == L'k');
        ASSERT(*(Dest - 1) == L'.');

        //
        // Copy the "pht1" extension over "keys".
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

        Success = Api->CreatePerfectHashTable(Rtl,
                                              Allocator,
                                              Context,
                                              AlgorithmId,
                                              MaskFunctionId,
                                              HashFunctionId,
                                              NULL,
                                              Keys,
                                              &TablePath);

        if (!Success) {

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to create perfect hash "
                                        L"table for keys: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failed = TRUE;
            Failures++;
            goto DestroyKeys;
        }

        WIDE_OUTPUT_RAW(WideOutput, L"Successfully created perfect "
                                    L"hash table: ");
        WIDE_OUTPUT_UNICODE_STRING(WideOutput, &TablePath);
        WIDE_OUTPUT_RAW(WideOutput, L".\n");

        //
        // Load the perfect hash table we just created.
        //

        Success = Api->LoadPerfectHashTable(Rtl,
                                            Allocator,
                                            Keys,
                                            &TablePath,
                                            &Table);

        if (!Success) {

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to load perfect hash table: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &TablePath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            goto DestroyKeys;

        }

        //
        // Table was loaded successfully from disk.  Obtain the names of all
        // the enumeration IDs.  Currently these should always match the same
        // enums provided as input parameters to this routine.
        //
        // N.B. I'm being lazy with the ASSERT()s here instead of reporting the
        //      error properly like we do with other failures.
        //

        ASSERT(Api->GetAlgorithmName(Table->AlgorithmId, &AlgorithmName));

        ASSERT(Api->GetHashFunctionName(Table->HashFunctionId,
                                        &HashFunctionName));

        ASSERT(Api->GetMaskFunctionName(Table->MaskFunctionId,
                                        &MaskFunctionName));


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

        Success = Api->TestPerfectHashTable(Table, TRUE);

        if (!Success) {

            WIDE_OUTPUT_RAW(WideOutput, L"Test failed for perfect hash table "
                                        L"loaded from disk: ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &TablePath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Failed = TRUE;
            goto DestroyTable;
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
        // Destroy the table.
        //

DestroyTable:

        Table->Vtbl->Release(Table);

DestroyKeys:

        Success = Api->DestroyPerfectHashTableKeys(&Keys);
        if (!Success) {

            WIDE_OUTPUT_RAW(WideOutput, L"Failed to destroy keys for ");
            WIDE_OUTPUT_UNICODE_STRING(WideOutput, &KeysPath);
            WIDE_OUTPUT_RAW(WideOutput, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
        }

DestroyContext:

        Success = Api->DestroyPerfectHashTableContext(&Context, NULL);
        if (!Success) {

            //
            // Failure to destroy a context is a fatal error that we can't
            // recover from.  Bomb out now.
            //

            WIDE_OUTPUT_RAW(WideOutput, L"Fatal: failed to destroy context.\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            break;
        }

    } while (FindNextFile(FindHandle, &FindData));

    //
    // Self test complete!
    //

    if (!Failures) {
        Success = TRUE;
        goto End;
    }

    //
    // Intentional follow-on to Error.
    //

Error:

    Success = FALSE;

    //
    // Intentional follow-on to End.
    //

End:

    //
    // We can't do much if any of these routines error out, hence the NOTHINGs.
    //

    if (WideOutputBuffer) {
        if (!Rtl->DestroyBuffer(Rtl, ProcessHandle, &WideOutputBuffer)) {
            NOTHING;
        }
        WideOutput = NULL;
    }

    if (BaseBuffer) {
        if (!Rtl->DestroyBuffer(Rtl, ProcessHandle, &BaseBuffer)) {
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

    return Success;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
