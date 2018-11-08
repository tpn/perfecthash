/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContextBulkCreate.c

Abstract:

    This module implements the bulk-create routine for the perfect hash library.

    N.B. This component is a work in progress.  It is based off the self-test
         component.

--*/

#include "stdafx.h"

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
    ULONG NumberOfTableCreateParameters,
    PPERFECT_HASH_TABLE_CREATE_PARAMETER TableCreateParameters
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

    MaskFunctionId - Supplies the type of masking to use.

    HashFunctionId - Supplies the hash function to use.

    ContextBulkCreateFlags - Optionally supplies a pointer to a bulk-create flags
        structure that can be used to customize bulk-create behavior.

    KeysLoadFlags - Optionally supplies a pointer to a key loading flags
        structure that can be used to customize key loading behavior.

    TableCreateFlags - Optionally supplies a pointer to a table create flags
        structure that can be used to customize table creation behavior.

    TableCompileFlags - Optionally supplies a pointer to a compile table flags
        structure that can be used to customize table compilation behavior.

    NumberOfTableCreateParameters - Optionally supplies the number of elements
        in the TableCreateParameters array.

    TableCreateParameters - Optionally supplies an array of additional
        parameters that can be used to further customize table creation
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

--*/
{
    PRTL Rtl;
    PWSTR Dest;
    PWSTR Source;
    //ULONG Index;
    ULONG LastError;
    USHORT Length;
    USHORT BaseLength;
    USHORT NumberOfPages;
    ULONG ReferenceCount;
    //BOOL Success;
    BOOLEAN Failed;
    BOOLEAN Terminate;
    HRESULT Result;
    PCHAR Buffer;
    PCHAR BaseBuffer;
    PCHAR Output;
    PCHAR OutputBuffer;
    PALLOCATOR Allocator;
    PVOID KeysBaseAddress;
    ULARGE_INTEGER NumberOfKeys;
    HANDLE FindHandle = NULL;
    HANDLE OutputHandle;
    HANDLE ProcessHandle = NULL;
    ULONG Failures;
    //ULONG BytesWritten;
    ULONGLONG BufferSize;
    ULONGLONG OutputBufferSize;
    LONG_INTEGER AllocSize;
    //LARGE_INTEGER BytesToWrite;
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
    //PERFECT_HASH_TABLE_FLAGS TableFlags;
    //PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PCUNICODE_STRING Suffix = &KeysWildcardSuffix;
    //PUNICODE_STRING AlgorithmName;
    //PUNICODE_STRING HashFunctionName;
    //PUNICODE_STRING MaskFunctionName;
    PERFECT_HASH_CPU_ARCH_ID CpuArchId;
    PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags;
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags;
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

    //
    // Create a buffer we can use for stdout, using a very generous buffer size.
    //

    NumberOfPages = 10;
    ProcessHandle = GetCurrentProcess();

    Result = Rtl->Vtbl->CreateBuffer(Rtl,
                                     &ProcessHandle,
                                     NumberOfPages,
                                     NULL,
                                     &OutputBufferSize,
                                     &OutputBuffer);

    if (FAILED(Result)) {
        return Result;
    }

    Output = OutputBuffer;

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
                                          &OutputBuffer);
        if (FAILED(Result)) {
            SYS_ERROR(VirtualFree);
        }
        return Result;
    }

    Buffer = BaseBuffer;

    //
    // Get a reference to the stdout handle.
    //

    OutputHandle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (!OutputHandle) {
        SYS_ERROR(GetStdHandle);
        return E_UNEXPECTED;
    }

    //
    // Calculate the size required for a new concatenated wide string buffer
    // that combines the test data directory with the "*.keys" suffix.  The
    // 2 * sizeof(*Dest) accounts for the joining slash and trailing NULL.
    //

    AllocSize.LongPart = KeysDirectory->Length;
    AllocSize.LongPart += Suffix->Length + (2 * sizeof(*Dest));

    ASSERT(!AllocSize.HighPart);

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
    // Write the output header.
    //

    //OUTPUT_HEADER();

    //
    // Create a find handle for the <keys dir>\*.keys search pattern we
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

#if 0
            WIDE_OUTPUT_RAW(Output,
                            L"No files matching pattern '*.keys' found in "
                            L"test data directory.\n");
            WIDE_OUTPUT_FLUSH();
#endif

            goto End;

        } else {

            //
            // We failed for some other reason.
            //

#if 0
            WIDE_OUTPUT_RAW(Output,
                            L"FindFirstFileW() failed with error code: ");
            WIDE_OUTPUT_INT(Output, LastError);
            WIDE_OUTPUT_LF(Output);
            WIDE_OUTPUT_FLUSH();
#endif

            goto Error;
        }
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
    // Zero the failure count and terminate flag, zero the bitmap structure,
    // wire up the unicode string representation of the bitmap, initialize
    // various flags, obtain the current CPU architecture ID, and begin the main
    // loop.
    //

    Failures = 0;
    Terminate = FALSE;
    ZeroStruct(KeysBitmap);

#if 0
    UnicodeBitmapString.Buffer = (PWCHAR)WideBitmapString;
    UnicodeBitmapString.Length = sizeof(WideBitmapString)-2;
    UnicodeBitmapString.MaximumLength = sizeof(WideBitmapString);
    UnicodeBitmapString.Buffer[UnicodeBitmapString.Length >> 1] = L'\0';
#endif

    Table = NULL;
    KeysBaseAddress = NULL;
    NumberOfKeys.QuadPart = 0;
    CpuArchId = PerfectHashGetCurrentCpuArch();

    //
    // We're not using the tables after we create them, so toggle the relevant
    // table create flag explicitly.
    //

    TableCreateFlags.CreateOnly = TRUE;

    ASSERT(IsValidPerfectHashCpuArchId(CpuArchId));

    do {

        //
        // Clear the failure flag at the top of every loop invocation.
        //

        Failed = FALSE;

#if 0
        WIDE_OUTPUT_RAW(Output, L"Processing key file: ");
        WIDE_OUTPUT_WCSTR(Output, (PCWSZ)FindData.cFileName);
        WIDE_OUTPUT_LF(Output);
        WIDE_OUTPUT_FLUSH();
#endif

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
#if 0
            WIDE_OUTPUT_RAW(Output, L"Failed to create keys instance.\n");
            WIDE_OUTPUT_FLUSH();
#endif
            Failures++;
            break;
        }

        Result = Keys->Vtbl->Load(Keys,
                                  &KeysLoadFlags,
                                  &KeysPathString,
                                  sizeof(ULONG));

        if (FAILED(Result)) {

#if 0
            WIDE_OUTPUT_RAW(Output, L"Failed to load keys for ");
            WIDE_OUTPUT_UNICODE_STRING(Output, &KeysPathString);
            WIDE_OUTPUT_RAW(Output, L".\n");
            WIDE_OUTPUT_FLUSH();
#endif

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

#if 0
        WIDE_OUTPUT_RAW(Output, L"Successfully loaded keys: ");
        WIDE_OUTPUT_UNICODE_STRING(Output, &KeysPathString);
        WIDE_OUTPUT_RAW(Output, L".\n");
        WIDE_OUTPUT_FLUSH();
#endif

        //
        // Verify GetFlags().
        //

        Result = Keys->Vtbl->GetFlags(Keys,
                                      sizeof(KeysFlags),
                                      &KeysFlags);

        if (FAILED(Result)) {
#if 0
            WIDE_OUTPUT_RAW(Output, L"Failed to obtain flags for keys: ");
            WIDE_OUTPUT_UNICODE_STRING(Output, &KeysPathString);
            WIDE_OUTPUT_RAW(Output, L".\n");
            WIDE_OUTPUT_FLUSH();
#endif

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

#if 0

        //
        // Verify base address and number of keys.
        //

        Result = Keys->Vtbl->GetAddress(Keys,
                                        &KeysBaseAddress,
                                        &NumberOfKeys);

        if (FAILED(Result)) {
            WIDE_OUTPUT_RAW(Output, L"Failed to obtain base "
                                        L"address for keys: ");
            WIDE_OUTPUT_UNICODE_STRING(Output, &KeysPathString);
            WIDE_OUTPUT_RAW(Output, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }
#endif

#if 0
        //
        // Verify the bitmap function returns success.
        //

        Result = Keys->Vtbl->GetBitmap(Keys, sizeof(KeysBitmap), &KeysBitmap);

        if (FAILED(Result)) {
            WIDE_OUTPUT_RAW(Output, L"Failed to get keys bitmap for ");
            WIDE_OUTPUT_UNICODE_STRING(Output, &KeysPathString);
            WIDE_OUTPUT_RAW(Output, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }
#endif

#if 0
        WIDE_OUTPUT_RAW(Output, L"Keys bitmap: ");
        WIDE_OUTPUT_INT(Output, KeysBitmap.Bitmap);
        WIDE_OUTPUT_RAW(Output, L".\n");
#endif

#if 0
        //
        // The bitmap buffer is a normal 8-bit character string, but our output
        // uses 16-bit wide character strings.  Do a simple conversion now.  We
        // don't need to worry about utf-8 multi-byte characters as the only
        // possible bitmap character values are '0' and '1'.
        //

        for (Index = 0; Index < sizeof(KeysBitmap.String); Index++) {
            WideBitmapString[Index] = (WCHAR)KeysBitmap.String[Index];
        }

        WIDE_OUTPUT_RAW(Output, L"Keys bitmap string: ");
        WIDE_OUTPUT_UNICODE_STRING(Output, &UnicodeBitmapString);
        WIDE_OUTPUT_RAW(Output, L".\n");

        WIDE_OUTPUT_RAW(Output, L"Keys bitmap contiguous? ");
        if (KeysBitmap.Flags.Contiguous) {
            WIDE_OUTPUT_UNICODE_STRING(Output, &Yes);
            WIDE_OUTPUT_RAW(Output, L"Keys bitmap shifted mask: ");
            WIDE_OUTPUT_INT(Output, KeysBitmap.ShiftedMask);
            WIDE_OUTPUT_RAW(Output, L".\n");
        } else {
            WIDE_OUTPUT_UNICODE_STRING(Output, &No);
        }

        WIDE_OUTPUT_RAW(Output, L"Keys bitmap longest run length: ");
        WIDE_OUTPUT_INT(Output, KeysBitmap.LongestRunLength);
        WIDE_OUTPUT_RAW(Output, L".\nKeys bitmap longest run start: ");
        WIDE_OUTPUT_INT(Output, KeysBitmap.LongestRunStart);
        WIDE_OUTPUT_RAW(Output, L".\nKeys bitmap trailing zeros: ");
        WIDE_OUTPUT_INT(Output, KeysBitmap.TrailingZeros);
        WIDE_OUTPUT_RAW(Output, L".\nKeys bitmap leading zeros: ");
        WIDE_OUTPUT_INT(Output, KeysBitmap.LeadingZeros);
        WIDE_OUTPUT_RAW(Output, L".\n");

        WIDE_OUTPUT_FLUSH();
#endif

        //
        // Keys were loaded successfully.  Proceed with table creation.
        //

        ASSERT(Table == NULL);

        Result = Context->Vtbl->CreateInstance(Context,
                                               NULL,
                                               &IID_PERFECT_HASH_TABLE,
                                               &Table);

        if (FAILED(Result)) {
#if 0
            WIDE_OUTPUT_RAW(Output, L"Failed to create table instance.\n");
            WIDE_OUTPUT_FLUSH();
#endif
            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

        Result = Table->Vtbl->Create(Table,
                                     Context,
                                     AlgorithmId,
                                     MaskFunctionId,
                                     HashFunctionId,
                                     Keys,
                                     &TableCreateFlags,
                                     NumberOfTableCreateParameters,
                                     TableCreateParameters);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashTableCreate, Result);

#if 0
            WIDE_OUTPUT_RAW(Output, L"Failed to create perfect hash "
                                        L"table for keys: ");
            WIDE_OUTPUT_UNICODE_STRING(Output, &KeysPathString);
            WIDE_OUTPUT_RAW(Output, L".\n");
            WIDE_OUTPUT_FLUSH();
#endif

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
#if 0
        WIDE_OUTPUT_RAW(Output, L"Successfully created perfect "
                                    L"hash table: ");
        WIDE_OUTPUT_UNICODE_STRING(Output, TableFullPath);
        WIDE_OUTPUT_RAW(Output, L".\n");
        WIDE_OUTPUT_FLUSH();
#endif

#if 0
        //
        // Test the newly-created table.
        //

        Result = Table->Vtbl->Test(Table, Keys, TRUE);

        if (FAILED(Result)) {

#if 0
            WIDE_OUTPUT_RAW(Output, L"Test failed for perfect hash table "
                                        L"created from context: ");
            WIDE_OUTPUT_UNICODE_STRING(Output, TableFullPath);
            WIDE_OUTPUT_RAW(Output, L".\n");
            WIDE_OUTPUT_FLUSH();
#endif

            Failures++;
            Failed = TRUE;
            goto ReleaseTable;
        }

#endif

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

            WIDE_OUTPUT_RAW(Output, L"Failed to compile table: ");
            WIDE_OUTPUT_UNICODE_STRING(Output, TableFullPath);
            WIDE_OUTPUT_RAW(Output, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Failed = TRUE;
            goto ReleaseTable;
        }

        WIDE_OUTPUT_RAW(Output, L"Compiled table successfully.\n");
        WIDE_OUTPUT_FLUSH();

#endif

        //
        // Release the table.
        //

        ReferenceCount = Table->Vtbl->Release(Table);
        Table = NULL;

        if (ReferenceCount != 0) {
#if 0
            WIDE_OUTPUT_RAW(Output, L"Invariant failed; releasing table "
                                    L"did not indicate a refcount of 0.\n");
            WIDE_OUTPUT_FLUSH();
#endif

            Failures++;
            Terminate = TRUE;
            goto ReleaseKeys;
        }

#if 0
#define GET_NAME(Desc)                                               \
        Result = Table->Vtbl->Get##Desc##Name(Table, &##Desc##Name); \
        if (FAILED(Result)) {                                        \
            WIDE_OUTPUT_RAW(Output,                                  \
                            L"Get" L#Desc "Name() failed.\n");       \
            Terminate = TRUE;                                        \
            goto ReleaseTable;                                       \
        }

        GET_NAME(Algorithm);
        GET_NAME(HashFunction);
        GET_NAME(MaskFunction);

        WIDE_OUTPUT_RAW(Output, L"Successfully loaded perfect "
                                    L"hash table: ");
        WIDE_OUTPUT_UNICODE_STRING(Output, TableFullPath);
        WIDE_OUTPUT_RAW(Output, L".\n");

        WIDE_OUTPUT_RAW(Output, L"Algorithm: ");
        WIDE_OUTPUT_UNICODE_STRING(Output, AlgorithmName);
        WIDE_OUTPUT_RAW(Output, L" (");
        WIDE_OUTPUT_INT(Output, Table->AlgorithmId);
        WIDE_OUTPUT_RAW(Output, L").\n");

        WIDE_OUTPUT_RAW(Output, L"Hash Function: ");
        WIDE_OUTPUT_UNICODE_STRING(Output, HashFunctionName);
        WIDE_OUTPUT_RAW(Output, L" (");
        WIDE_OUTPUT_INT(Output, Table->HashFunctionId);
        WIDE_OUTPUT_RAW(Output, L").\n");

        WIDE_OUTPUT_RAW(Output, L"Mask Function: ");
        WIDE_OUTPUT_UNICODE_STRING(Output, MaskFunctionName);
        WIDE_OUTPUT_RAW(Output, L" (");
        WIDE_OUTPUT_INT(Output, Table->MaskFunctionId);
        WIDE_OUTPUT_RAW(Output, L").\n");

        WIDE_OUTPUT_RAW(Output, L"Keys backed by large pages: ");
        if (KeysFlags.KeysDataUsesLargePages) {
            WIDE_OUTPUT_UNICODE_STRING(Output, &Yes);
        } else {
            WIDE_OUTPUT_UNICODE_STRING(Output, &No);
        }
#endif

#if 0
        //
        // Verify GetFlags().
        //

        Result = Table->Vtbl->GetFlags(Table,
                                       sizeof(TableFlags),
                                       &TableFlags);

        if (FAILED(Result)) {
            WIDE_OUTPUT_RAW(Output, L"Failed to obtain flags for table: ");
            WIDE_OUTPUT_UNICODE_STRING(Output, TableFullPath);
            WIDE_OUTPUT_RAW(Output, L".\n");
            WIDE_OUTPUT_FLUSH();

            Failures++;
            Terminate = TRUE;
            goto ReleaseTable;
        }

        WIDE_OUTPUT_RAW(Output, L"Table data backed by large pages: ");
        if (TableFlags.TableDataUsesLargePages) {
            WIDE_OUTPUT_UNICODE_STRING(Output, &Yes);
        } else {
            WIDE_OUTPUT_UNICODE_STRING(Output, &No);
        }

        WIDE_OUTPUT_RAW(Output, L"Values array allocated with large "
                                    L"pages: ");
        if (TableFlags.ValuesArrayUsesLargePages) {
            WIDE_OUTPUT_UNICODE_STRING(Output, &Yes);
        } else {
            WIDE_OUTPUT_UNICODE_STRING(Output, &No);
        }

        WIDE_OUTPUT_FLUSH();

        TableInfoOnDisk = Table->TableInfoOnDisk;

        //
        // Define some helper macros here for dumping stats.
        //

#define STATS_INT(String, Name)                                    \
        WIDE_OUTPUT_RAW(Output, String);                           \
        WIDE_OUTPUT_INT(Output, Table->TableInfoOnDisk->##Name##); \
        WIDE_OUTPUT_RAW(Output, L".\n")

#define STATS_QUAD(String, Name)                                            \
        WIDE_OUTPUT_RAW(Output, String);                                    \
        WIDE_OUTPUT_INT(Output, Table->TableInfoOnDisk->##Name##.QuadPart); \
        WIDE_OUTPUT_RAW(Output, L".\n")

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

            WIDE_OUTPUT_RAW(Output,
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

#endif

        //
        // Release the table and keys.
        //

ReleaseTable:

        RELEASE(TablePath);
        RELEASE(TableFile);
        RELEASE(Table);

ReleaseKeys:

        RELEASE(Keys);

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

    if (OutputBuffer) {
        Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                          ProcessHandle,
                                          &OutputBuffer);
        if (FAILED(Result)) {
            SYS_ERROR(VirtualFree);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
        Output = NULL;
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
    PULONG NumberOfTableCreateParameters,
    PPERFECT_HASH_TABLE_CREATE_PARAMETER *TableCreateParameters
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

    BulkCreateFlags - Supplies the address of a variable that will receive the
        bulk-create flags.

    KeysLoadFlags - Supplies the address of a variable that will receive
        the keys load flags.

    TableCreateFlags - Supplies the address of a variable that will receive
        the table create flags.

    TableLoadFlags - Supplies the address of a variable that will receive the
        the load table flags.

    TableCompileFlags - Supplies the address of a variable that will receive
        the table compile flags.

    NumberOfTableCreateParameters - Supplies the address of a variable that will
        receive the number of elements in the TableCreateParameters array.

    TableCreateParameters - Supplies the address of a variable that will receive
        a pointer to an array of table create parameters.  If this is not NULL,
        the memory will be allocated via the context's allocator and the caller
        is responsible for freeing it.

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
    ULONG CurrentArg = 1;
    PALLOCATOR Allocator;
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

    if (!ARGUMENT_PRESENT(NumberOfTableCreateParameters)) {
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
    // Extract algorithm ID.
    //

    CurrentArg++;
    String->Buffer = *ArgW++;
    String->Length = GET_LENGTH(String);
    String->MaximumLength = GET_MAX_LENGTH(String);
    VALIDATE_ID(Algorithm, ALGORITHM);

    //
    // Extract hash function ID.
    //

    CurrentArg++;
    String->Buffer = *ArgW++;
    String->Length = GET_LENGTH(String);
    String->MaximumLength = GET_MAX_LENGTH(String);
    VALIDATE_ID(HashFunction, HASH_FUNCTION);

    //
    // Extract mask function ID.
    //

    CurrentArg++;
    String->Buffer = *ArgW++;
    String->Length = GET_LENGTH(String);
    String->MaximumLength = GET_MAX_LENGTH(String);
    VALIDATE_ID(MaskFunction, MASK_FUNCTION);

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
    *NumberOfTableCreateParameters = 0;
    *TableCreateParameters = NULL;

    for (; CurrentArg < NumberOfArguments; CurrentArg++, ArgW++) {

        String->Buffer = Arg = *ArgW;
        String->Length = GET_LENGTH(String);
        String->MaximumLength = GET_MAX_LENGTH(String);

        //
        // If the argument doesn't start with two dashes, ignore it.
        //

        if (String->Length <= (sizeof(L'-') + sizeof(L'-'))) {
            continue;
        }

        if (!(*Arg++ == L'-' && *Arg++ == L'-')) {
            continue;
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

        Result =
            TryExtractArgTableCreateParameters(Rtl,
                                               Allocator,
                                               String,
                                               NumberOfTableCreateParameters,
                                               TableCreateParameters);

        if (FAILED(Result)) {

            PH_ERROR(ExtractBulkCreateArgs_TryExtractTableCreateParams, Result);
            break;

        } else {

            //
            // Ignore anything not recognized for now.
            //

            continue;
        }
    }

    //
    // If we failed, free the table create parameters if they are non-null and
    // clear the corresponding number-of variable.
    //

    if (FAILED(Result) && TableCreateParameters) {
        Context->Allocator->Vtbl->FreePointer(Context->Allocator,
                                              (PVOID *)&TableCreateParameters);
        *NumberOfTableCreateParameters = 0;
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

    PH_E_INVALID_KEYS_LOAD_FLAGS - Invalid keys load flags.

    PH_E_INVALID_CONTEXT_CREATE_TABLE_FLAGS - Invalid context create table
        flags.

    PH_E_INVALID_TABLE_LOAD_FLAGS - Invalid table load flags.

    PH_E_INVALID_TABLE_COMPILE_FLAGS - Invalid table compile flags.

--*/
{
    PRTL Rtl;
    HRESULT Result;
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
    ULONG NumberOfTableCreateParameters = 0;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER TableCreateParameters = 0;

    Rtl = Context->Rtl;

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
                                   &NumberOfTableCreateParameters,
                                   &TableCreateParameters);

    if (FAILED(Result)) {

        //
        // Todo: write the usage string.
        //

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
                                       NumberOfTableCreateParameters,
                                       TableCreateParameters);

    if (FAILED(Result)) {

        //
        // There's is nothing we can do here.  We don't PH_ERROR() the return
        // code as BulkCreate() will have done that multiple times each time
        // the error bubbled back up the stack.
        //

        NOTHING;
    }

    if (TableCreateParameters) {
        Context->Allocator->Vtbl->FreePointer(Context->Allocator,
                                              (PVOID *)&TableCreateParameters);
    }

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
