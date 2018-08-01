/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    CreatePerfectHashTable.c

Abstract:

    This module implements the creation routine for the PerfectHashTable
    component.

--*/

#include "stdafx.h"

CREATE_PERFECT_HASH_TABLE CreatePerfectHashTable;

_Use_decl_annotations_
BOOLEAN
CreatePerfectHashTable(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PPERFECT_HASH_TABLE_CONTEXT Context,
    PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId,
    PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId,
    PULARGE_INTEGER NumberOfTableElementsPointer,
    PPERFECT_HASH_TABLE_KEYS Keys,
    PCUNICODE_STRING HashTablePath
    )
/*++

Routine Description:

    Creates and initializes a PERFECT_HASH_TABLE structure from a given set
    of keys, using the requested algorithm.

Arguments:

    Rtl - Supplies a pointer to an initialized RTL structure.

    Allocator - Supplies a pointer to an initialized ALLOCATOR structure that
        will be used for all memory allocations.

    Context - Supplies a pointer to an initialized PERFECT_HASH_TABLE_CONTEXT
        structure that can be used by the underlying algorithm in order to
        search for perfect hash solutions in parallel.

    AlgorithmId - Supplies the algorithm to use.

    MaskFunctionId - Supplies the type of masking to use.  The algorithm and hash
        function must both support the requested masking type.

    HashFunctionId - Supplies the hash function to use.

    NumberOfTableElementsPointer - Optionally supplies a pointer to a
        ULARGE_INTEGER structure that, if non-zero, indicates the number of
        elements to assume when sizing the hash table.  If a non-NULL pointer
        is supplied, it will receive the final number of elements in the table
        if a solution could be found.

        N.B. If this value is non-zero, it must be equal to or greater than
             the number of keys indicated by the Keys parameter.  (It should be
             at least 2.09 times the number of keys; the higher the value, the
             larger the hash table, the faster a perfect hash solution will be
             found.)

    Keys - Supplies a pointer to a PERFECT_HASH_TABLE_KEYS structure obtained
        from LoadPerfectHashTableKeys().

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
    BOOLEAN Success;
    BOOLEAN UsingKeysPath;
    PWSTR Dest;
    PWSTR Source;
    PBYTE Buffer;
    PWSTR PathBuffer;
    ULONG Key;
    ULONG Index;
    ULONG Result;
    ULONGLONG Bit;
    ULONG ShareMode;
    ULONG LastError;
    ULONG NumberOfKeys;
    ULONG DesiredAccess;
    ULONG NumberOfSetBits;
    ULONG InfoMappingSize;
    ULONG FlagsAndAttributes;
    PLONGLONG BitmapBuffer;
    USHORT VtblExSize;
    PVOID BaseAddress;
    HANDLE FileHandle;
    HANDLE MappingHandle;
    SYSTEM_INFO SystemInfo;
    ULARGE_INTEGER AllocSize;
    ULONG_INTEGER PathBufferSize;
    ULONGLONG KeysBitmapBufferSize;
    BOOLEAN LargePagesForBitmapBuffer;
    ULONG IncomingPathBufferSizeInBytes;
    ULONG_INTEGER AlignedPathBufferSize;
    ULONG_INTEGER InfoStreamPathBufferSize;
    ULONG_INTEGER AlignedInfoStreamPathBufferSize;
    ULARGE_INTEGER NumberOfTableElements;
    PPERFECT_HASH_TABLE_VTBL_EX Vtbl;
    PPERFECT_HASH_TABLE Table = NULL;
    UNICODE_STRING Suffix = RTL_CONSTANT_STRING(L".pht1");
    UNICODE_STRING InfoStreamSuffix = RTL_CONSTANT_STRING(L":Info");

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(Allocator)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(Context)) {

        return FALSE;

    } else if (Context->Table) {

        //
        // We don't support a context being used more than once at the moment.
        // If it has been used before, Context->Table will have a value, and
        // thus, we need to error out.
        //

        return FALSE;
    }

    if (!ARGUMENT_PRESENT(Keys)) {

        return FALSE;

    } else {

        //
        // Ensure the number of keys is within our maximum tested limit.
        //

        if (Keys->NumberOfElements.QuadPart > MAXIMUM_NUMBER_OF_KEYS) {

            return FALSE;

        }
    }

    if (!IsValidPerfectHashTableMaskFunctionId(MaskFunctionId)) {
        return FALSE;
    }

    if (ARGUMENT_PRESENT(NumberOfTableElementsPointer)) {

        //
        // If the number of table elements is non-zero, verify it is greater
        // than or equal to the number of keys present.
        //

        NumberOfTableElements.QuadPart = (
            NumberOfTableElementsPointer->QuadPart
        );

        if (NumberOfTableElements.QuadPart > 0) {

            if (NumberOfTableElements.QuadPart <
                Keys->NumberOfElements.QuadPart) {

                //
                // Requested table size is too small, abort.
                //

                return FALSE;
            }
        }
    } else {

        NumberOfTableElements.QuadPart = 0;

    }

    if (ARGUMENT_PRESENT(HashTablePath) &&
        !IsValidMinimumDirectoryNullTerminatedUnicodeString(HashTablePath)) {

        return FALSE;
    }

    if (!IsValidPerfectHashTableAlgorithmId(AlgorithmId)) {
        return FALSE;
    }

    if (!IsValidPerfectHashTableHashFunctionId(HashFunctionId)) {
        return FALSE;
    }

    //
    // Calculate the allocation size required for the structure, including the
    // memory required to take a copy of the backing file name path.
    //

    AllocSize.QuadPart = sizeof(*Table);

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
        // length.  (This is handled in LoadPerfectHashTableKeys().)
        //

        ASSERT(Keys->Path.Length + sizeof(Keys->Path.Buffer[0]) ==
               Keys->Path.MaximumLength);

        PathBufferSize.LongPart = (Keys->Path.MaximumLength + Suffix.Length);

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
    // Add the path buffer size to the structure allocation size, then check
    // we haven't overflowed MAX_ULONG.
    //

    AllocSize.QuadPart += AlignedPathBufferSize.LowPart;

    ASSERT(!AllocSize.HighPart);

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
    // Add the stream path size to the total size and perform a final overflow
    // check.
    //

    AllocSize.QuadPart += AlignedInfoStreamPathBufferSize.LowPart;

    ASSERT(!AllocSize.HighPart);

    //
    // Account for the vtbl interface size.
    //

    VtblExSize = GetVtblExSizeRoutines[AlgorithmId]();
    AllocSize.QuadPart += VtblExSize;

    ASSERT(!AllocSize.HighPart);

    //
    // Allocate space for the structure and backing paths.
    //

    Table = (PPERFECT_HASH_TABLE)(
        Allocator->Calloc(Allocator->Context,
                          1,
                          AllocSize.LowPart)
    );

    if (!Table) {
        return FALSE;
    }

    //
    // Allocation was successful, continue with initialization.
    //

    Table->SizeOfStruct = sizeof(*Table);
    Table->Rtl = Rtl;
    Table->Allocator = Allocator;
    Table->Flags.AsULong = 0;
    Table->Keys = Keys;
    Table->Context = Context;
    Context->Table = Table;

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

    Buffer = RtlOffsetToPointer(Table, sizeof(*Table));
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
        // Replace the ".keys" suffix with ".pht1".
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
    // Advance the buffer to the vtbl interface area and initialize it.
    //

    Buffer += AlignedInfoStreamPathBufferSize.LongPart;
    Vtbl = (PPERFECT_HASH_TABLE_VTBL_EX)Buffer;
    InitializeExtendedVtbl(Table, Vtbl);

    //
    // Copy the full path into the info stream buffer.
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

    FileHandle = Rtl->CreateFileW(Table->Path.Buffer,
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

        goto Error;

    } else if (LastError == ERROR_ALREADY_EXISTS) {

        //
        // The file was opened successfully, but it already existed.  Clear the
        // local last error variable then truncate the file.
        //

        LastError = ERROR_SUCCESS;

        Result = SetFilePointer(FileHandle, 0, NULL, FILE_BEGIN);
        if (Result == INVALID_SET_FILE_POINTER) {
            LastError = GetLastError();
            goto Error;
        }

        Success = SetEndOfFile(FileHandle);
        if (!Success) {
            LastError = GetLastError();
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

    FileHandle = Rtl->CreateFileW(Table->InfoStreamPath.Buffer,
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

        goto Error;

    } else if (LastError == ERROR_ALREADY_EXISTS) {

        //
        // The file was opened successfully, but it already existed.  Clear the
        // local last error variable then truncate the file.
        //

        LastError = ERROR_SUCCESS;

        Result = SetFilePointer(FileHandle, 0, NULL, FILE_BEGIN);
        if (Result == INVALID_SET_FILE_POINTER) {
            LastError = GetLastError();
            goto Error;
        }

        Success = SetEndOfFile(FileHandle);
        if (!Success) {
            LastError = GetLastError();
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
        goto Error;
    }

    //
    // Set the number of table elements requested by the user (0 is a valid
    // value here).
    //

    Table->RequestedNumberOfTableElements.QuadPart = (
        NumberOfTableElements.QuadPart
    );

    //
    // Allocate a 512MB buffer for the keys bitmap.
    //

    KeysBitmapBufferSize = ((1ULL << 32ULL) >> 3ULL);

    //
    // Try a large page allocation for the bitmap buffer.
    //

    LargePagesForBitmapBuffer = TRUE;
    BaseAddress = Rtl->TryLargePageVirtualAlloc(NULL,
                                                KeysBitmapBufferSize,
                                                MEM_RESERVE | MEM_COMMIT,
                                                PAGE_READWRITE,
                                                &LargePagesForBitmapBuffer);

    Table->KeysBitmap.Buffer = (PULONG)BaseAddress;

    if (!BaseAddress) {

        //
        // Failed to create a bitmap buffer, abort.
        //

        LastError = GetLastError();
        goto Error;
    }

    //
    // Initialize the keys bitmap.
    //

    Table->KeysBitmap.SizeOfBitMap = (ULONG)-1;
    BitmapBuffer = (PLONGLONG)Table->KeysBitmap.Buffer;

    ASSERT(!Keys->NumberOfElements.HighPart);

    NumberOfKeys = Keys->NumberOfElements.LowPart;

    //
    // Loop through all the keys, obtain the bitmap bit representation, verify
    // that the bit hasn't been set yet, and set it.
    //

    for (Index = 0; Index < NumberOfKeys; Index++) {

        Key = Keys->Keys[Index];
        Bit = Key + 1;

        ASSERT(!BitTestAndSet64(BitmapBuffer, Bit));

    }

    //
    // Count all bits set.  It should match the number of keys.
    //

    NumberOfSetBits = Rtl->RtlNumberOfSetBits(&Table->KeysBitmap);
    ASSERT(NumberOfSetBits == NumberOfKeys);

    //
    // Common initialization is complete, dispatch remaining work to the
    // algorithm's creation routine.
    //

    Success = CreationRoutines[AlgorithmId](Table);
    if (!Success) {
        goto Error;
    }

    //
    // Update the caller's number of table elements pointer, if applicable.
    //

    if (ARGUMENT_PRESENT(NumberOfTableElementsPointer)) {
        NumberOfTableElementsPointer->QuadPart = Table->HashSize;
    }

    //
    // We're done!  Set the reference count to 1 and finish up.
    //

    Table->ReferenceCount = 1;
    goto End;

Error:

    Success = FALSE;

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Call the destroy routine on the table if one is present.
    //
    // N.B. We currently always delete the table if it is created successfully
    //      so as to ensure the only way to use a table is by loading one from
    //      disk via LoadPerfectHashTable().
    //

    if (Table) {

        if (!DestroyPerfectHashTable(&Table, NULL)) {

            //
            // There's nothing we can do here.
            //

            NOTHING;
        }

        //
        // N.B. DestroyPerfectHashTable() should clear the Table pointer.
        //      Assert that invariant now.
        //

        ASSERT(Table == NULL);
    }

    return Success;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
