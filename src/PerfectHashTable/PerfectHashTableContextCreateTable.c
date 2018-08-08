/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableContextCreateTable.c

Abstract:

    This module implements the creation routine for the PerfectHashTable
    component.

--*/

#include "stdafx.h"

PERFECT_HASH_TABLE_CONTEXT_CREATE_TABLE PerfectHashTableContextCreateTable;

_Use_decl_annotations_
HRESULT
PerfectHashTableContextCreateTable(
    PPERFECT_HASH_TABLE_CONTEXT Context,
    PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId,
    PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId,
    PPERFECT_HASH_TABLE_KEYS Keys,
    PCUNICODE_STRING HashTablePath
    )
/*++

Routine Description:

    Creates and initializes a PERFECT_HASH_TABLE structure from a given set
    of keys, using the requested algorithm.

Arguments:

    Context - Supplies a pointer to an initialized PERFECT_HASH_TABLE_CONTEXT
        structure that can be used by the underlying algorithm in order to
        search for perfect hash solutions in parallel.

    AlgorithmId - Supplies the algorithm to use.

    MaskFunctionId - Supplies the type of masking to use.  The algorithm and hash
        function must both support the requested masking type.

    HashFunctionId - Supplies the hash function to use.

    Keys - Supplies a pointer to a PERFECT_HASH_TABLE_KEYS interface.

    HashTablePath - Optionally supplies a pointer to a UNICODE_STRING structure
        that represents the fully-qualified, NULL-terminated path of the backing
        file used to save the hash table.  If NULL, the file name of the keys
        file will be used, with ".pht1" appended to it.

Return Value:

    TRUE on success, FALSE on failure.

    If TRUE, the table will be persisted at the path described by for the
    HashTablePath parameter above.  This can be subsequently interacted with
    once loaded via LoadPerfectHashTable().

--*/
{
    PRTL Rtl;
    BOOL Success;
    PWSTR Dest;
    PWSTR Source;
    PCHAR Buffer;
    PCHAR BaseBuffer;
    PCHAR ExpectedBuffer;
    PWSTR PathBuffer;
    ULONG Status;
    ULONG ShareMode;
    ULONG LastError;
    ULONG DesiredAccess;
    ULONG InfoMappingSize;
    ULONG FlagsAndAttributes;
    BOOLEAN UsingKeysPath;
    PVOID BaseAddress;
    HANDLE FileHandle;
    HANDLE MappingHandle;
    HRESULT Result;
    PALLOCATOR Allocator;
    SYSTEM_INFO SystemInfo;
    ULARGE_INTEGER AllocSize;
    ULONG_INTEGER PathBufferSize;
    ULONG IncomingPathBufferSizeInBytes;
    ULONG_INTEGER AlignedPathBufferSize;
    ULONG_INTEGER InfoStreamPathBufferSize;
    ULONG_INTEGER AlignedInfoStreamPathBufferSize;
    PPERFECT_HASH_TABLE Table = NULL;

    //
    // XXX TODO: relocate these to constants.
    //

    UNICODE_STRING Suffix = RTL_CONSTANT_STRING(L".pht1");
    UNICODE_STRING InfoStreamSuffix = RTL_CONSTANT_STRING(L":Info");

    //
    // XXX TODO.
    //

#define PerfectHashTableContextReset(Context) (E_UNEXPECTED)

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Context)) {

        return E_POINTER;

    } else {

        //
        // Initialize aliases.
        //

        Allocator = Context->Allocator;
        Rtl = Context->Rtl;

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

    if (!ARGUMENT_PRESENT(Keys)) {

        return E_POINTER;

    } else {

        //
        // Ensure the number of keys is within our maximum tested limit.
        //

        if (Keys->NumberOfElements.QuadPart > MAXIMUM_NUMBER_OF_KEYS) {
            return PH_E_TOO_MANY_KEYS;
        }
    }

    if (!TryAcquirePerfectHashTableContextLockExclusive(Context)) {
        return PH_E_CREATE_TABLE_ALREADY_IN_PROGRESS;
    }

    if (Context->State.NeedsReset) {
        Result = PerfectHashTableContextReset(Context);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashTableContextReset, Result);
            return E_FAIL;
        }
    }

    //
    // No table should be associated with the context at this point.
    //

    ASSERT(!Context->Table);

    if (ARGUMENT_PRESENT(HashTablePath) &&
        !IsValidMinimumDirectoryNullTerminatedUnicodeString(HashTablePath)) {
        return E_INVALIDARG;
    }

    //
    // Calculate the allocation size required for the backing file name path.
    //

    if (ARGUMENT_PRESENT(HashTablePath)) {

        //
        // Use the length of the caller-provided path, plus a trailing NULL.
        //

        PathBufferSize.LongPart = (
            HashTablePath->Length +
            sizeof(HashTablePath->Buffer[0])
        );

        UsingKeysPath = FALSE;
        IncomingPathBufferSizeInBytes = HashTablePath->Length;
        PathBuffer = HashTablePath->Buffer;

    } else {

        //
        // No path has been provided by the caller, so we'll use the path of
        // the keys file with ".pht1" appended.  Perform a quick invariant check
        // first: maximum length should be 1 character (2 bytes) larger than
        // length.  (This is handled in PerfectHashTableKeysLoad().)
        //

        ASSERT(Keys->Path.Length + sizeof(Keys->Path.Buffer[0]) ==
               Keys->Path.MaximumLength);

        PathBufferSize.LongPart = (
            Keys->Path.MaximumLength +
            Suffix.Length
        );

        UsingKeysPath = TRUE;
        IncomingPathBufferSizeInBytes = Keys->Path.Length;
        PathBuffer = Keys->Path.Buffer;
    }

    //
    // Align the path buffer up to a 16 byte boundary.
    //

    AlignedPathBufferSize.LongPart = ALIGN_UP(PathBufferSize.LongPart, 16);

    //
    // Sanity check we haven't overflowed MAX_USHORT for the path buffer size.
    //

    ASSERT(!AlignedPathBufferSize.HighPart);

    //
    // Calculate the size required by the :Info stream that will be created
    // for the on-disk metadata.  We derive this by adding the length of the
    // path to the length of the :Info suffix, plus an additional trailing NULL.
    //

    InfoStreamPathBufferSize.LongPart = (
        PathBufferSize.LowPart +
        InfoStreamSuffix.Length +
        sizeof(InfoStreamSuffix.Buffer[0])
    );

    //
    // Align the size up to a 16 byte boundary.
    //

    AlignedInfoStreamPathBufferSize.LongPart = (
        ALIGN_UP(
            InfoStreamPathBufferSize.LowPart,
            16
        )
    );

    //
    // Sanity check we haved overflowed MAX_USHORT.
    //

    ASSERT(!AlignedInfoStreamPathBufferSize.HighPart);

    //
    // Compute the total size required, then check we haven't overflowed
    // MAX_ULONG.
    //

    AllocSize.QuadPart = (
        AlignedPathBufferSize.LowPart +
        AlignedInfoStreamPathBufferSize.LowPart
    );

    ASSERT(!AllocSize.HighPart);

    //
    // Allocate space for the buffer.
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
        return E_OUTOFMEMORY;
    }

    //
    // Allocation was successful.  Create a new table instance then continue
    // initializing its internal state.
    //

    Result = Context->Vtbl->CreateInstance(Context,
                                           NULL,
                                           &IID_PERFECT_HASH_TABLE,
                                           &Context->Table);

    if (FAILED(Result)) {
        goto Error;
    }

    Table = Context->Table;

    Keys->Vtbl->AddRef(Keys);
    Table->Keys = Keys;

    Context->Vtbl->AddRef(Context);
    Table->Context = Context;

    //
    // Our main enumeration IDs get replicated in both structures.
    //

    Table->AlgorithmId = Context->AlgorithmId = AlgorithmId;
    Table->MaskFunctionId = Context->MaskFunctionId = MaskFunctionId;
    Table->HashFunctionId = Context->HashFunctionId = HashFunctionId;

    //
    // Carve out the backing memory structures for the unicode string buffers
    // for the path names.
    //

    Table->Path.Buffer = (PWSTR)Buffer;
    CopyMemory(Table->Path.Buffer, PathBuffer, IncomingPathBufferSizeInBytes);

    if (!UsingKeysPath) {

        //
        // Inherit the lengths provided by the input parameter string.
        //

        Table->Path.Length = HashTablePath->Length;
        Table->Path.MaximumLength = HashTablePath->MaximumLength;

    } else {

        //
        // Replace the keys file's file name extension (".keys") with our table
        // file name extension.
        //

        Dest = Table->Path.Buffer;
        Dest += (Keys->Path.MaximumLength >> 1) - 5ULL;

        ASSERT(*Dest == L'k');
        ASSERT(*(Dest - 1) == L'.');

        Source = Suffix.Buffer;

        while (*Source) {
            *Dest++ = *Source++;
        }

        *Dest = L'\0';

        //
        // We can use the Keys->Path lengths directly.
        //

        Table->Path.Length = Keys->Path.Length;
        Table->Path.MaximumLength = Keys->Path.MaximumLength;
    }

    //
    // Advance past the aligned path buffer size such that we're positioned at
    // the start of the info stream buffer.
    //

    Buffer += AlignedPathBufferSize.LongPart;
    Table->InfoStreamPath.Buffer = (PWSTR)Buffer;
    Table->InfoStreamPath.MaximumLength = InfoStreamPathBufferSize.LowPart;
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

    Buffer += AlignedInfoStreamPathBufferSize.LongPart;
    ExpectedBuffer = RtlOffsetToPointer(BaseBuffer, AllocSize.LowPart);
    ASSERT(Buffer == ExpectedBuffer);
    Buffer = NULL;

    //
    // Continue with path processing.  Copy the full path into the info stream
    // buffer.
    //

    CopyMemory(Table->InfoStreamPath.Buffer,
               Table->Path.Buffer,
               Table->Path.Length);

    Dest = Table->InfoStreamPath.Buffer;
    Dest += (Table->Path.Length >> 1);
    ASSERT(*Dest == L'\0');

    //
    // Copy the :Info suffix over.
    //

    Source = InfoStreamSuffix.Buffer;

    while (*Source) {
        *Dest++ = *Source++;
    }

    *Dest = L'\0';

    //
    // We've finished initializing our two unicode string buffers for the
    // backing file and it's :Info counterpart.  Now, let's open file handles
    // to them.
    //

    //
    // Open the file handle for the backing hash table store.
    //

    ShareMode = (
        FILE_SHARE_READ  |
        FILE_SHARE_WRITE |
        FILE_SHARE_DELETE
    );

    DesiredAccess = (
        GENERIC_READ |
        GENERIC_WRITE
    );

    FlagsAndAttributes = FILE_FLAG_OVERLAPPED;

    FileHandle = CreateFileW(Table->Path.Buffer,
                             DesiredAccess,
                             ShareMode,
                             NULL,
                             OPEN_ALWAYS,
                             FlagsAndAttributes,
                             NULL);

    LastError = GetLastError();

    Table->FileHandle = FileHandle;

    if (!FileHandle || FileHandle == INVALID_HANDLE_VALUE) {

        //
        // Failed to open the file successfully.
        //

        SYS_ERROR(CreateFileW);
        goto Error;

    } else if (LastError == ERROR_ALREADY_EXISTS) {

        //
        // The file was opened successfully, but it already existed.  Clear the
        // local last error variable then truncate the file.
        //

        LastError = ERROR_SUCCESS;

        Status = SetFilePointer(FileHandle, 0, NULL, FILE_BEGIN);
        if (Status == INVALID_SET_FILE_POINTER) {
            SYS_ERROR(SetFilePointer);
            goto Error;
        }

        Success = SetEndOfFile(FileHandle);
        if (!Success) {
            SYS_ERROR(SetEndOfFile);
            goto Error;
        }

        //
        // We've successfully truncated the file.  The creation routine
        // implementation can now allocate the space required for it as part
        // of successful graph solving.
        //

    }

    //
    // The :Info stream is slightly different.  As it's a fixed size metadata
    // record, we can memory map an entire section up-front prior to calling
    // the algorithm implementation.  So, do that now.
    //

    FileHandle = CreateFileW(Table->InfoStreamPath.Buffer,
                             DesiredAccess,
                             ShareMode,
                             NULL,
                             OPEN_ALWAYS,
                             FlagsAndAttributes,
                             NULL);

    Table->InfoStreamFileHandle = FileHandle;

    LastError = GetLastError();

    if (!FileHandle || FileHandle == INVALID_HANDLE_VALUE) {

        //
        // Failed to open the file successfully.
        //

        SYS_ERROR(CreateFileW);
        goto Error;

    } else if (LastError == ERROR_ALREADY_EXISTS) {

        //
        // The file was opened successfully, but it already existed.  Clear the
        // local last error variable then truncate the file.
        //

        LastError = ERROR_SUCCESS;

        Status = SetFilePointer(FileHandle, 0, NULL, FILE_BEGIN);
        if (Status == INVALID_SET_FILE_POINTER) {
            SYS_ERROR(SetFilePointer);
            goto Error;
        }

        Success = SetEndOfFile(FileHandle);
        if (!Success) {
            SYS_ERROR(SetEndOfFile);
            goto Error;
        }

        //
        // We've successfully truncated the :Info file.
        //

    }

    //
    // Get the system allocation granularity, as we use this to govern the size
    // we request of the underlying file mapping.
    //

    GetSystemInfo(&SystemInfo);

    InfoMappingSize = SystemInfo.dwAllocationGranularity;
    ASSERT(InfoMappingSize >= PAGE_SIZE);

    //
    // Create a file mapping for the :Info stream.
    //

    MappingHandle = CreateFileMappingW(FileHandle,
                                       NULL,
                                       PAGE_READWRITE,
                                       0,
                                       InfoMappingSize,
                                       NULL);

    Table->InfoStreamMappingHandle = MappingHandle;
    Table->InfoMappingSizeInBytes.QuadPart = InfoMappingSize;

    if (!MappingHandle || MappingHandle == INVALID_HANDLE_VALUE) {
        SYS_ERROR(CreateFileMappingW);
        goto Error;
    }

    //
    // We successfully created a file mapping.  Proceed with mapping it into
    // memory.
    //

    BaseAddress = MapViewOfFile(MappingHandle,
                                FILE_MAP_READ | FILE_MAP_WRITE,
                                0,
                                0,
                                InfoMappingSize);

    Table->InfoStreamBaseAddress = BaseAddress;

    if (!BaseAddress) {
        SYS_ERROR(MapViewOfFile);
        goto Error;
    }

    //
    // Common initialization is complete, dispatch remaining work to the
    // algorithm's creation routine.
    //

    Success = CreationRoutines[AlgorithmId](Table);
    if (!Success) {
        goto Error;
    }

    //
    // We're done!  Indicate success and finish up.
    //

    Result = S_OK;
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
    // Release the table if one is present.
    //
    // N.B. We currently always delete the table if it is created successfully
    //      so as to ensure the only way to use a table is by loading one from
    //      disk via PerfectHashTableLoad().
    //

    if (Table) {
        Table->Vtbl->Release(Table);
        Table = NULL;
    }

    Context->Table = NULL;
    Context->State.NeedsReset = TRUE;

    //
    // Release the buffer we allocated for paths if applicable.
    //

    if (BaseBuffer) {
        Allocator->Vtbl->FreePointer(Allocator, &BaseBuffer);
    }

    ReleasePerfectHashTableContextLockExclusive(Context);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
