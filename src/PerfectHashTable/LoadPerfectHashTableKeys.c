/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    LoadPerfectHashTableKeys.c

Abstract:

    This module implements the key load routine for PerfectHashTable component.

--*/

#include "stdafx.h"

LOAD_PERFECT_HASH_TABLE_KEYS LoadPerfectHashTableKeys;

_Use_decl_annotations_
BOOLEAN
LoadPerfectHashTableKeys(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PCUNICODE_STRING Path,
    PPERFECT_HASH_TABLE_KEYS *KeysPointer
    )
/*++

Routine Description:

    Loads a keys file from the file system and returns a PERFECT_HASH_TABLE_KEYS
    structure.

Arguments:

    Rtl - Supplies a pointer to an initialized RTL structure.

    Allocator - Supplies a pointer to an initialized ALLOCATOR structure that
        will be used for al memory allocations.

    Path - Supplies a pointer to a UNICODE_STRING structure that represents
        a fully-qualified path of the keys to use for the perfect hash table.

        N.B. Path must be NULL-terminated, which is not normally required for
             UNICODE_STRING structures.  Howver, the underlying buffer is passed
             to CreateFileW(), which requires a NULL-terminated wstr.

    KeysPointers - Supplies the address of a variable that will receive the
        address of the newly created PERFECT_HASH_TABLE_KEYS structure if the
        routine is successful, or NULL if the routine fails.

Return Value:

    TRUE on success, FALSE on failure.

--*/
{
    BOOLEAN Success;
    PVOID BaseAddress = NULL;
    HANDLE FileHandle = NULL;
    HANDLE MappingHandle = NULL;
    LARGE_INTEGER AllocSize;
    LARGE_INTEGER NumberOfElements;
    FILE_STANDARD_INFO FileInfo;
    PPERFECT_HASH_TABLE_KEYS Keys = NULL;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(Allocator)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(KeysPointer)) {
        return FALSE;
    }

    if (!ARGUMENT_PRESENT(Path)) {
        return FALSE;
    }

    if (!IsValidMinimumDirectoryNullTerminatedUnicodeString(Path)) {
        return FALSE;
    }

    //
    // Clear the caller's pointer up-front.
    //

    *KeysPointer = NULL;

    //
    // Open the file, create a file mapping, then map it into memory.
    //

    FileHandle = Rtl->CreateFileW(
        Path->Buffer,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED,
        NULL
    );

    if (!FileHandle || FileHandle == INVALID_HANDLE_VALUE) {
        return FALSE;
    }

    //
    // N.B. Error handling should 'goto Error' from this point onward now that
    //      we have resources that may need to be cleaned up.
    //

    Success = GetFileInformationByHandleEx(
        FileHandle,
        (FILE_INFO_BY_HANDLE_CLASS)FileStandardInfo,
        &FileInfo,
        sizeof(FileInfo)
    );

    if (!Success) {
        goto Error;
    }

    //
    // Make sure the file is a multiple of our key size.
    //

    if (FileInfo.EndOfFile.QuadPart % 4ULL) {
        goto Error;
    }

    //
    // The number of elements in the key file can be ascertained by right
    // shifting by 2.
    //

    NumberOfElements.QuadPart = FileInfo.EndOfFile.QuadPart >> 2;

    //
    // Sanity check the number of elements.  There shouldn't be more than
    // MAX_ULONG.
    //

    ASSERT(!NumberOfElements.HighPart);

    //
    // Create the file mapping.
    //

    MappingHandle = CreateFileMappingW(FileHandle,
                                       NULL,
                                       PAGE_READONLY,
                                       FileInfo.EndOfFile.HighPart,
                                       FileInfo.EndOfFile.LowPart,
                                       NULL);

    if (!MappingHandle || MappingHandle == INVALID_HANDLE_VALUE) {
        goto Error;
    }

    //
    // Successfully created a file mapping.  Now map it into memory.
    //

    BaseAddress = MapViewOfFile(MappingHandle,
                                FILE_MAP_READ,
                                0,
                                0,
                                FileInfo.EndOfFile.LowPart);

    if (!BaseAddress) {
        goto Error;
    }

    //
    // The file has been mapped successfully.  Calculate the size required for
    // the keys structure and a copy of the Path's underlying unicode string
    // buffer.
    //

    AllocSize.QuadPart = (

        //
        // Account for the keys structure size.
        //

        sizeof(*Keys) +

        //
        // Account for the length of the UNICODE_STRING buffer and terminating
        // NULL.
        //

        Path->Length + sizeof(Path->Buffer[0])

    );

    //
    // Sanity check our size.
    //

    ASSERT(!AllocSize.HighPart);

    //
    // Proceed with allocation.
    //

    Keys = (PPERFECT_HASH_TABLE_KEYS)(
        Allocator->Calloc(Allocator->Context,
                          1,
                          AllocSize.LowPart)
    );

    if (!Keys) {
        goto Error;
    }

    //
    // Fill out the main structure details.
    //

    Keys->SizeOfStruct = sizeof(*Keys);
    Keys->Rtl = Rtl;
    Keys->Allocator = Allocator;
    Keys->FileHandle = FileHandle;
    Keys->MappingHandle = MappingHandle;
    Keys->BaseAddress = BaseAddress;
    Keys->NumberOfElements.QuadPart = NumberOfElements.QuadPart;

    //
    // Initialize the path length variables, point the buffer at the space after
    // our structure, then copy the string over and NULL-terminate it.
    //

    Keys->Path.Length = Path->Length;
    Keys->Path.MaximumLength = Path->Length + sizeof(Path->Buffer[0]);
    Keys->Path.Buffer = (PWSTR)RtlOffsetToPointer(Keys, sizeof(*Keys));
    CopyMemory(Keys->Path.Buffer, Path->Buffer, Path->Length);
    Keys->Path.Buffer[Path->Length >> 1] = L'\0';

    //
    // We've completed initialization, indicate success and jump to the end.
    //

    Success = TRUE;

    goto End;

Error:

    Success = FALSE;

    //
    // Clean up any resources we may have allocated.
    //

    if (BaseAddress) {
        UnmapViewOfFile(BaseAddress);
        BaseAddress = NULL;
    }

    if (MappingHandle) {
        CloseHandle(MappingHandle);
        MappingHandle = NULL;
    }

    if (FileHandle) {
        CloseHandle(FileHandle);
        FileHandle = NULL;
    }

    if (Keys) {
        Allocator->FreePointer(Allocator->Context, &Keys);
    }

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Update the caller's pointer and return.
    //
    // N.B. Keys could be NULL here, which is fine.
    //

    *KeysPointer = Keys;

    return Success;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
