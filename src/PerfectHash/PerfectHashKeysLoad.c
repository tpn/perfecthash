/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashKeysLoad.c

Abstract:

    This module implements the key load routine for PerfectHash component.

--*/

#include "stdafx.h"

PERFECT_HASH_KEYS_LOAD PerfectHashKeysLoad;

_Use_decl_annotations_
HRESULT
PerfectHashKeysLoad(
    PPERFECT_HASH_KEYS Keys,
    PCUNICODE_STRING Path
    )
/*++

Routine Description:

    Loads a keys file from the file system.

Arguments:

    Keys - Supplies a pointer to the PERFECT_HASH_KEYS structure for
        which the keys are to be loaded.

    Path - Supplies a pointer to a UNICODE_STRING structure that represents
        a fully-qualified path of the keys to use for the perfect hash table.

        N.B. Path must be NULL-terminated, which is not normally required for
             UNICODE_STRING structures.  Howver, the underlying buffer is passed
             to CreateFileW(), which requires a NULL-terminated wstr.

Return Value:

    S_OK on success, an appropriate error code on failure.

--*/
{
    PRTL Rtl;
    BOOL Success;
    PCHAR Buffer = NULL;
    HRESULT Result = S_OK;
    PALLOCATOR Allocator;
    PVOID BaseAddress = NULL;
    HANDLE FileHandle = NULL;
    HANDLE MappingHandle = NULL;
    LARGE_INTEGER AllocSize;
    LARGE_INTEGER NumberOfElements;
    FILE_STANDARD_INFO FileInfo;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Keys)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    }

    if (!IsValidMinimumDirectoryNullTerminatedUnicodeString(Path)) {
        return E_INVALIDARG;
    }

    //
    // Initialize aliases.
    //

    Rtl = Keys->Rtl;
    Allocator = Keys->Allocator;

    //
    // Open the file, create a file mapping, then map it into memory.
    //

    FileHandle = CreateFileW(
        Path->Buffer,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED,
        NULL
    );

    if (!FileHandle || FileHandle == INVALID_HANDLE_VALUE) {
        SYS_ERROR(CreateFileW);
        return E_UNEXPECTED;
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
        SYS_ERROR(GetFileInformationByHandleEx);
        goto Error;
    }

    //
    // Make sure the file is a multiple of our key size.
    //

    if (FileInfo.EndOfFile.QuadPart % PERFECT_HASH_KEY_SIZE_IN_BYTES) {
        Result = PH_E_KEYS_FILE_SIZE_NOT_MULTIPLE_OF_KEY_SIZE;
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

    if (NumberOfElements.HighPart) {
        Result = PH_E_TOO_MANY_KEYS;
        goto Error;
    }

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
        SYS_ERROR(CreateFileMappingW);
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
        SYS_ERROR(MapViewOfFile);
        goto Error;
    }

    //
    // The file has been mapped successfully.  Calculate the size required for
    // a copy of the Path's underlying unicode string buffer.
    //

    AllocSize.QuadPart = Path->Length + sizeof(Path->Buffer[0]);

    //
    // Sanity check our size.
    //

    ASSERT(!AllocSize.HighPart);

    //
    // Proceed with allocation.
    //

    Buffer = (PCHAR)(
        Allocator->Vtbl->Calloc(
            Allocator,
            1,
            AllocSize.LowPart
        )
    );

    if (!Buffer) {
        SYS_ERROR(HeapAlloc);
        goto Error;
    }

    //
    // Fill out the main structure details.
    //

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
    Keys->Path.Buffer = (PWSTR)Buffer;
    CopyMemory(Keys->Path.Buffer, Path->Buffer, Path->Length);
    Keys->Path.Buffer[Path->Length >> 1] = L'\0';

    //
    // We've completed initialization, indicate success and jump to the end.
    //

    Keys->State.Loaded = TRUE;
    Result = S_OK;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Clean up any resources we may have allocated.
    //

    if (BaseAddress) {
        if (!UnmapViewOfFile(BaseAddress)) {
            SYS_ERROR(UnmapViewOfFile);
        }
        BaseAddress = NULL;
    }

    if (MappingHandle) {
        if (!CloseHandle(MappingHandle)) {
            SYS_ERROR(CloseHandle);
        }
        MappingHandle = NULL;
    }

    if (FileHandle) {
        if (!CloseHandle(FileHandle)) {
            SYS_ERROR(CloseHandle);
        }
        FileHandle = NULL;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
