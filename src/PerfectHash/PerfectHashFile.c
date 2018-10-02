/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashFile.c

Abstract:

    This is the module for the PERFECT_HASH_FILE component of the perfect
    hash table library.  Routines are provided for initialization, rundown,
    getting flags, getting names, and getting handles and addresses.

--*/

#include "stdafx.h"

//
// Helper inline method for updating the file's FILE_INFO structure.
//

FORCEINLINE
HRESULT
PerfectHashFileUpdateFileInfo(
    _In_ PPERFECT_HASH_FILE File
    )
{
    BOOL Success;

    Success = GetFileInformationByHandleEx(File->FileHandle,
                                           FileStandardInfo,
                                           &File->FileInfo,
                                           sizeof(File->FileInfo));

    if (!Success) {
        SYS_ERROR(GetFileInformationByHandleEx);
        return PH_E_SYSTEM_CALL_FAILED;
    }

    return S_OK;
}


PERFECT_HASH_FILE_INITIALIZE PerfectHashFileInitialize;

_Use_decl_annotations_
HRESULT
PerfectHashFileInitialize(
    PPERFECT_HASH_FILE File
    )
/*++

Routine Description:

    Initializes a perfect hash file structure.  This is a relatively simple
    method that just primes the COM scaffolding; the bulk of the work is done
    when loading or creating the file.

Arguments:

    File - Supplies a pointer to a PERFECT_HASH_FILE structure for which
        initialization is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - File is NULL.

    E_UNEXPECTED - All other errors.

--*/
{
    HRESULT Result = S_OK;
    SYSTEM_INFO SystemInfo;

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    File->SizeOfStruct = sizeof(*File);

    //
    // Create Rtl and Allocator components.
    //

    Result = File->Vtbl->CreateInstance(File,
                                        NULL,
                                        &IID_PERFECT_HASH_RTL,
                                        &File->Rtl);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = File->Vtbl->CreateInstance(File,
                                        NULL,
                                        &IID_PERFECT_HASH_ALLOCATOR,
                                        &File->Allocator);

    if (FAILED(Result)) {
        goto Error;
    }

    //
    // Capture the system's file allocation granularity.  This is used in
    // various places to align a given memory mapping size.
    //

    GetSystemInfo(&SystemInfo);
    File->AllocationGranularity = SystemInfo.dwAllocationGranularity;

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

    return Result;
}

PERFECT_HASH_FILE_RUNDOWN PerfectHashFileRundown;

_Use_decl_annotations_
VOID
PerfectHashFileRundown(
    PPERFECT_HASH_FILE File
    )
/*++

Routine Description:

    Release all resources associated with a perfect hash file.

Arguments:

    File - Supplies a pointer to a PERFECT_HASH_FILE structure for which
        rundown is to be performed.

Return Value:

    None.

--*/
{
    HRESULT Result;

    //
    // Sanity check structure size.
    //

    ASSERT(File->SizeOfStruct == sizeof(*File));

    //
    // Close the file if necessary.
    //

    if (!IsFileClosed(File)) {

        //
        // Issue a close with 0 for end-of-file, indicating the file should
        // be deleted.  (Callers should explicitly Close() a file prior to
        // rundown; otherwise, if we get to this point, we assume the absense
        // of a Close() call is indicative of an error, and the file should be
        // discarded.)
        //

        LARGE_INTEGER EndOfFile = { 0 };
        Result = File->Vtbl->Close(File, &EndOfFile);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileClose, Result);
        }
    }

    //
    // Release COM references.
    //

    RELEASE(File->Path);
    RELEASE(File->RenamePath);
    RELEASE(File->Rtl);
    RELEASE(File->Allocator);

    return;
}

PERFECT_HASH_FILE_LOAD PerfectHashFileLoad;

_Use_decl_annotations_
HRESULT
PerfectHashFileLoad(
    PPERFECT_HASH_FILE File,
    PPERFECT_HASH_PATH SourcePath,
    PLARGE_INTEGER EndOfFilePointer,
    PPERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlagsPointer
    )
/*++

Routine Description:

    Loads a a file.

Arguments:

    File - Supplies a pointer to the file to load.

    SourcePath - Supplies a pointer to the path instance to load.

    EndOfFile - Optionally supplies a pointer to a variable that will receive
        the current end of file (i.e. size in bytes) if the file was loaded
        successfully.

    FileLoadFlags - Optionally supplies a pointer to file load flags that can
        be used to customize load behavior.

Return Value:

    S_OK - File was closed successfully.

    E_POINTER - File or Path parameters were NULL.

    PH_E_INVALID_FILE_LOAD_FLAGS - Invalid file load flags.

    PH_E_SOURCE_PATH_LOCKED - Source path is locked.

    PH_E_SOURCE_PATH_NO_PATH_SET - Source path has not been set.

    PH_E_FILE_LOCKED - The file is locked.

    PH_E_FILE_ALREADY_OPEN - An existing file has already been loaded/created.

    PH_E_FILE_ALREADY_CLOSED - File has been opened and then subsequently
        closed already.

    PH_E_FILE_EMPTY - The file was empty.

    PH_E_SYSTEM_CALL_FAILED - A system call failed; the file may be in an
        inconsistent state.

--*/
{
    PRTL Rtl;
    ULONG ShareMode;
    ULONG DesiredAccess;
    ULONG FlagsAndAttributes;
    PALLOCATOR Allocator;
    HRESULT Result = S_OK;
    BOOLEAN Opened = FALSE;
    PERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags = { 0 };

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(SourcePath)) {
        return E_POINTER;
    }

    VALIDATE_FLAGS(FileLoad, FILE_LOAD);

    if (!TryAcquirePerfectHashPathLockShared(SourcePath)) {
        return PH_E_SOURCE_PATH_LOCKED;
    }

    if (!IsPathSet(SourcePath)) {
        ReleasePerfectHashPathLockShared(SourcePath);
        return PH_E_SOURCE_PATH_NO_PATH_SET;
    }

    if (!TryAcquirePerfectHashFileLockExclusive(File)) {
        ReleasePerfectHashPathLockShared(SourcePath);
        return PH_E_FILE_LOCKED;
    }

    if (IsFileOpen(File)) {
        Result = PH_E_FILE_ALREADY_OPEN;
        goto Error;
    }

    if (IsFileClosed(File)) {
        Result = PH_E_FILE_ALREADY_CLOSED;
        goto Error;
    }

    //
    // Invariant check: File->Path and File->RenamePath should both be NULL
    // at this point.
    //

    if (File->Path) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashFileCreate, Result);
        goto Error;
    }

    if (File->RenamePath) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashFileCreate, Result);
        goto Error;
    }

    //
    // Argument validation complete, continue with loading.
    //

    //
    // Initialize aliases, flags and state.
    //

    Rtl = File->Rtl;
    Allocator = File->Allocator;

    if (FileLoadFlags.DisableTryLargePagesForFileData) {
        File->Flags.DoesNotWantLargePages = TRUE;
    }

    File->State.IsReadOnly = TRUE;

    //
    // Add a reference to the source path.
    //

    SourcePath->Vtbl->AddRef(SourcePath);
    File->Path = SourcePath;

    //
    // Open the file using the newly created path.
    //

    DesiredAccess = GENERIC_READ;
    ShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    FlagsAndAttributes = FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED;

    File->FileHandle = CreateFileW(File->Path->FullPath.Buffer,
                                   DesiredAccess,
                                   ShareMode,
                                   NULL,
                                   OPEN_EXISTING,
                                   FlagsAndAttributes,
                                   NULL);

    if (!IsValidHandle(File->FileHandle)) {
        File->FileHandle = NULL;
        SYS_ERROR(CreateFileW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Opened = TRUE;
    SetFileOpened(File);
    SetFileLoaded(File);

    //
    // Update the file info in order to obtain the current file size.
    //

    Result = PerfectHashFileUpdateFileInfo(File);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileLoad, Result);
        goto Error;
    }

    //
    // Error out if the file is empty; Load()'ing implies memory mapping an
    // existing file, which won't work on an empty file.
    //

    if (File->FileInfo.EndOfFile.QuadPart == 0) {
        Result = PH_E_FILE_EMPTY;
        PH_ERROR(PerfectHashFileLoad, Result);
        goto Error;
    }

    //
    // Update the caller's EndOfFile pointer if applicable.
    //

    if (ARGUMENT_PRESENT(EndOfFilePointer)) {
        EndOfFilePointer->QuadPart = File->FileInfo.EndOfFile.QuadPart;
    }

    //
    // Map the file into memory.
    //

    Result = File->Vtbl->Map(File);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileMap, Result);
        goto Error;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    RELEASE(File->Path);

    //
    // Intentional follow-on to End.
    //

End:

    ReleasePerfectHashPathLockShared(SourcePath);
    ReleasePerfectHashFileLockExclusive(File);

    if (Opened && FAILED(Result)) {
        HRESULT CloseResult;
        LARGE_INTEGER EndOfFile = { 0 };

        CloseResult = File->Vtbl->Close(File, &EndOfFile);
        if (FAILED(CloseResult)) {
            PH_ERROR(PerfectHashFileClose, CloseResult);
        }
    }

    return Result;
}

PERFECT_HASH_FILE_CREATE PerfectHashFileCreate;

_Use_decl_annotations_
HRESULT
PerfectHashFileCreate(
    PPERFECT_HASH_FILE File,
    PPERFECT_HASH_PATH SourcePath,
    PLARGE_INTEGER EndOfFilePointer,
    PPERFECT_HASH_FILE_CREATE_FLAGS FileCreateFlagsPointer
    )
/*++

Routine Description:

    Creates a file.

Arguments:

    File - Supplies a pointer to the file to load.

    SourcePath - Supplies a pointer to the path instance to load.

    EndOfFile - Supplies a pointer to a LARGE_INTEGER structure that contains
        the desired size of the file being created.  Once created (and truncated
        if necessary), a memory map will be created for the entire file size.

    FileCreateFlags - Optionally supplies a pointer to file load flags that can
        be used to customize load behavior.

Return Value:

    S_OK - File was closed successfully.

    E_POINTER - File, SourcePath or EndOfFile parameters were NULL.

    PH_E_INVALID_FILE_CREATE_FLAGS - Invalid file create flags.

    PH_E_INVALID_END_OF_FILE - EndOfFile parameter was invalid (<= 0).

    PH_E_SOURCE_PATH_LOCKED - Source path is locked.

    PH_E_SOURCE_PATH_NO_PATH_SET - Source path has not been set.

    PH_E_FILE_LOCKED - The file is locked.

    PH_E_FILE_ALREADY_OPEN - An existing file has already been loaded/created.

    PH_E_FILE_ALREADY_CLOSED - File has been opened and then subsequently
        closed already.

    PH_E_SYSTEM_CALL_FAILED - A system call failed; the file may be in an
        inconsistent state.

--*/
{
    PRTL Rtl;
    ULONG LastError;
    ULONG ShareMode;
    ULONG DesiredAccess;
    ULONG FlagsAndAttributes;
    HRESULT Result = S_OK;
    BOOLEAN Opened = FALSE;
    LARGE_INTEGER EndOfFile;
    LARGE_INTEGER EmptyEndOfFile = { 0 };
    PERFECT_HASH_FILE_CREATE_FLAGS FileCreateFlags = { 0 };

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(SourcePath)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(EndOfFilePointer)) {
        return E_POINTER;
    } else {
        EndOfFile.QuadPart = EndOfFilePointer->QuadPart;
        if (EndOfFile.QuadPart <= 0) {
            return PH_E_INVALID_END_OF_FILE;
        }
    }

    VALIDATE_FLAGS(FileCreate, FILE_CREATE);

    if (!TryAcquirePerfectHashPathLockShared(SourcePath)) {
        return PH_E_SOURCE_PATH_LOCKED;
    }

    if (!IsPathSet(SourcePath)) {
        ReleasePerfectHashPathLockShared(SourcePath);
        return PH_E_SOURCE_PATH_NO_PATH_SET;
    }

    if (!TryAcquirePerfectHashFileLockExclusive(File)) {
        ReleasePerfectHashPathLockShared(SourcePath);
        return PH_E_FILE_LOCKED;
    }

    if (IsFileOpen(File)) {
        Result = PH_E_FILE_ALREADY_OPEN;
        goto Error;
    }

    if (IsFileClosed(File)) {
        Result = PH_E_FILE_ALREADY_CLOSED;
        goto Error;
    }

    //
    // Invariant check: File->Path and File->RenamePath should both be NULL
    // at this point.
    //

    if (File->Path) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashFileCreate, Result);
        goto Error;
    }

    if (File->RenamePath) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashFileCreate, Result);
        goto Error;
    }

    //
    // Argument validation complete, continue with creation.
    //

    //
    // Initialize aliases and create flags.
    //

    Rtl = File->Rtl;

    if (FileCreateFlags.DisableTryLargePagesForFileData) {
        File->Flags.DoesNotWantLargePages = TRUE;
    }

    File->State.IsReadOnly = FALSE;

    //
    // Add a reference to the source path.
    //

    SourcePath->Vtbl->AddRef(SourcePath);
    File->Path = SourcePath;

    //
    // Open the file using the newly created path.
    //

    DesiredAccess = GENERIC_READ | GENERIC_WRITE;
    ShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    FlagsAndAttributes = FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED;

    File->FileHandle = CreateFileW(File->Path->FullPath.Buffer,
                                   DesiredAccess,
                                   ShareMode,
                                   NULL,
                                   OPEN_ALWAYS,
                                   FlagsAndAttributes,
                                   NULL);

    LastError = GetLastError();

    if (!IsValidHandle(File->FileHandle)) {
        File->FileHandle = NULL;
        SYS_ERROR(CreateFileW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Opened = TRUE;
    SetFileOpened(File);
    SetFileCreated(File);

    //
    // If the file already existed, truncate it to 0.  This ensures we're not
    // picking up any existing file contents.
    //

    if (LastError == ERROR_ALREADY_EXISTS) {
        Result = File->Vtbl->Truncate(File, &EmptyEndOfFile);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileTruncate, Result);
            goto Error;
        }
    }

    //
    // Extend the file to the desired size.
    //

    Result = File->Vtbl->Truncate(File, &EndOfFile);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileTruncate, Result);
        goto Error;
    }

    //
    // Map the file into memory.
    //

    Result = File->Vtbl->Map(File);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileMap, Result);
        goto Error;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    RELEASE(File->Path);

    //
    // Intentional follow-on to End.
    //

End:

    ReleasePerfectHashPathLockShared(SourcePath);
    ReleasePerfectHashFileLockExclusive(File);

    if (Opened && FAILED(Result)) {
        HRESULT CloseResult;

        CloseResult = File->Vtbl->Close(File, &EmptyEndOfFile);
        if (FAILED(CloseResult)) {
            PH_ERROR(PerfectHashFileClose, CloseResult);
        }
    }

    return Result;
}

PERFECT_HASH_FILE_CLOSE PerfectHashFileClose;

_Use_decl_annotations_
HRESULT
PerfectHashFileClose(
    PPERFECT_HASH_FILE File,
    PLARGE_INTEGER EndOfFilePointer
    )
/*++

Routine Description:

    Closes a previously loaded or created file.  This involves unmapping the
    file view, closing the mapping handle, and closing the file handle.  If
    a large page buffer was allocated for the file data, this will also be
    freed.

    If this routine returns success, the instance enters a 'closed' state, and
    no further operations may be performed on it.

Arguments:

    File - Supplies a pointer to the file to close.

    EndOfFile - Optionally supplies a pointer to a LARGE_INTEGER structure that
        contains the desired end-of-file offset.

Return Value:

    S_OK - File was closed successfully.

    E_POINTER - File parameter was NULL.

    E_INVALIDARG - EndOfFile was invalid (e.g. negative).

    PH_E_FILE_LOCKED - The file is locked.

    PH_E_FILE_NOT_OPEN - The file is not open.

    PH_E_INVARIANT_CHECK_FAILED - The file indicates that bytes were written
        to the base address, however, the file is marked as readonly.  The file
        may be left in an inconsistent state.

    PH_E_SYSTEM_CALL_FAILED - A system call failed; the file may be in an
        inconsistent state.

--*/
{
    HRESULT Result;
    LARGE_INTEGER EndOfFile = { 0 };

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (ARGUMENT_PRESENT(EndOfFilePointer)) {
        EndOfFile.QuadPart = EndOfFilePointer->QuadPart;
        if (EndOfFile.QuadPart < 0) {
            return E_INVALIDARG;
        }
    } else {
        EndOfFile.QuadPart = File->NumberOfBytesWritten.QuadPart;
    }

    if (!TryAcquirePerfectHashFileLockExclusive(File)) {
        return PH_E_FILE_LOCKED;
    }

    if (!IsFileOpen(File)) {
        ReleasePerfectHashFileLockExclusive(File);
        return PH_E_FILE_NOT_OPEN;
    }

    if (EndOfFile.QuadPart < 0) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashFileClose, Result);
        ReleasePerfectHashFileLockExclusive(File);
        return Result;
    }

    File->PendingEndOfFile.QuadPart = EndOfFile.QuadPart;

    //
    // Unmap the file.
    //

    Result = File->Vtbl->Unmap(File);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileUnmap, Result);
    }

    //
    // If the end of file is non-zero, truncate the file.  Otherwise, if it's
    // zero, treat this as an indication that the file should be deleted.
    //

    if (EndOfFile.QuadPart > 0) {

        if (IsFileReadOnly(File)) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashFileClose, Result);
        } else {
            Result = File->Vtbl->Truncate(File, &EndOfFile);
            if (FAILED(Result)) {
                PH_ERROR(PerfectHashFileTruncate, Result);
            }
        }

    } else if (!IsFileReadOnly(File)) {

        ASSERT(EndOfFile.QuadPart == 0);

        //
        // End of file is 0, indicating that we should delete the file.  If a
        // rename is scheduled, release the rename path's lock and ref count.
        //

        if (IsRenameScheduled(File)) {
            File->RenamePath->Vtbl->Release(File->RenamePath);
            File->RenamePath = NULL;
        }

        if (!DeleteFileW(File->Path->FullPath.Buffer)) {
            SYS_ERROR(DeleteFileW);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }

    }

    if (File->FileHandle) {
        if (!CloseHandle(File->FileHandle)) {
            SYS_ERROR(CloseHandle);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
        File->FileHandle = NULL;
    }

    SetFileClosed(File);

    //
    // If a rename has been scheduled, do it now.
    //

    if (IsRenameScheduled(File)) {
        ASSERT(!IsFileReadOnly(File));
        Result = File->Vtbl->DoRename(File);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileDoRename, Result);
        }
    }

    ReleasePerfectHashFileLockExclusive(File);

    return Result;
}

PERFECT_HASH_FILE_MAP PerfectHashFileMap;

_Use_decl_annotations_
HRESULT
PerfectHashFileMap(
    PPERFECT_HASH_FILE File
    )
/*++

Routine Description:

    Maps a file into memory.

Arguments:

    File - Supplies a pointer to the file to map.

Return Value:

    S_OK - File was mapped successfully.

    E_POINTER - File parameter was NULL.

    PH_E_INVALID_END_OF_FILE - Invalid end of file.

    PH_E_FILE_NOT_OPEN - File is not open.

    PH_E_FILE_VIEW_CREATED - A view has already been created.

    PH_E_FILE_VIEW_MAPPED - A view has already been mapped.

    PH_E_SYSTEM_CALL_FAILED - A system call failed; the file may be in an
        inconsistent state.

--*/
{
    ULONG Access;
    ULONG Protection;
    HRESULT Result = S_OK;
    LARGE_INTEGER EndOfFile;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!IsFileOpen(File)) {
        return PH_E_FILE_NOT_OPEN;
    }

    if (IsViewCreated(File)) {
        return PH_E_FILE_VIEW_CREATED;
    }

    if (IsViewMapped(File)) {
        return PH_E_FILE_VIEW_MAPPED;
    }

    EndOfFile.QuadPart = File->FileInfo.EndOfFile.QuadPart;

    if (EndOfFile.QuadPart <= 0) {
        return PH_E_INVALID_END_OF_FILE;
    }

    //
    // Argument validation complete.  Create the file mapping.
    //

    if (IsFileReadOnly(File)) {
        Protection = PAGE_READONLY;
        Access = FILE_MAP_READ;
    } else {
        Protection = PAGE_READWRITE;
        Access = FILE_MAP_READ | FILE_MAP_WRITE;
    }

    File->MappingHandle = CreateFileMappingW(File->FileHandle,
                                             NULL,
                                             Protection,
                                             0,
                                             0,
                                             NULL);

    if (!IsValidHandle(File->MappingHandle)) {
        SYS_ERROR(CreateFileMappingW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Successfully created the file mapping.  Now, map it into memory.
    //

    File->BaseAddress = MapViewOfFile(File->MappingHandle, Access, 0, 0, 0);

    if (!File->BaseAddress) {
        SYS_ERROR(MapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // File has been mapped successfully.  Attempt a large page allocation if
    // applicable.
    //

    if (!WantsLargePages(File)) {

        File->Flags.UsesLargePages = FALSE;

    } else {

        ULONG LargePageAllocFlags;
        PVOID LargePageAddress = NULL;
        ULONG_PTR LargePageAllocSize;

        LargePageAllocFlags = MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
        LargePageAllocSize = (
            ALIGN_UP_LARGE_PAGE(LARGE_INTEGER_TO_SIZE_T(EndOfFile))
        );

        LargePageAddress = VirtualAlloc(NULL,
                                        LargePageAllocSize,
                                        LargePageAllocFlags,
                                        PAGE_READWRITE);

        if (!LargePageAddress) {

            File->Flags.UsesLargePages = FALSE;

        } else {

            //
            // The large page allocation was successful.
            //

            File->Flags.UsesLargePages = TRUE;
            File->MappedAddress = File->BaseAddress;
            File->BaseAddress = LargePageAddress;

            //
            // If the file is readonly, copy the mapped contents into the large
            // page buffer.
            //

            if (IsFileReadOnly(File)) {

                PRTL Rtl = File->Rtl;
                ULONG NumberOfPages = NumberOfPagesForFile(File);

                Rtl->Vtbl->CopyPages(Rtl,
                                     LargePageAddress,
                                     File->MappedAddress,
                                     NumberOfPages);

            }

        }

    }

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

    return Result;
}

PERFECT_HASH_FILE_UNMAP PerfectHashFileUnmap;

_Use_decl_annotations_
HRESULT
PerfectHashFileUnmap(
    PPERFECT_HASH_FILE File
    )
/*++

Routine Description:

    Unmaps a previously mapped file.  Does not close the underlying file handle.

Arguments:

    File - Supplies a pointer to the file to unmap.

Return Value:

    S_OK - File was unmapped successfully, if applicable.

    E_POINTER - File parameter was NULL.

    PH_E_FILE_NOT_OPEN - File is not open.

    PH_E_SYSTEM_CALL_FAILED - A system call failed; the file may be in an
        inconsistent state.

--*/
{
    PRTL Rtl;
    HRESULT Result = S_OK;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!IsFileOpen(File)) {
        return PH_E_FILE_NOT_OPEN;
    }

    //
    // Argument validation complete.
    //

    Rtl = File->Rtl;

    if (!File->MappedAddress) {

        ASSERT(!File->Flags.UsesLargePages);

    } else if (File->BaseAddress) {

        //
        // If MappedAddress is non-NULL, BaseAddress is actually our
        // large page address which needs to be freed with VirtualFree().
        //

        ASSERT(File->Flags.UsesLargePages);

        if (!IsFileReadOnly(File) && !IsFileBeingExtended(File)) {

            //
            // The file is not read only and not being extended, so, copy the
            // data back to the mapped address first.
            //

            ULONG NumberOfPagesForAllocation;
            ULONG NumberOfPagesForPending;

            NumberOfPagesForAllocation = NumberOfPagesForFile(File);
            NumberOfPagesForPending = NumberOfPagesForPendingEndOfFile(File);

            ASSERT(NumberOfPagesForPending <= NumberOfPagesForAllocation);

            if (NumberOfPagesForPending > 0) {

                Rtl->Vtbl->CopyPages(Rtl,
                                     File->MappedAddress,
                                     File->BaseAddress,
                                     NumberOfPagesForPending);
            }
        }

        if (!VirtualFree(File->BaseAddress, 0, MEM_RELEASE)) {
            SYS_ERROR(VirtualFree);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }

        //
        // Switch the base address back so it's unmapped correctly below.
        //

        File->Flags.UsesLargePages = FALSE;
        File->BaseAddress = File->MappedAddress;
        File->MappedAddress = NULL;
    }

    if (File->FileHandle && !IsFileReadOnly(File)) {
        if (!FlushFileBuffers(File->FileHandle)) {
            SYS_ERROR(FlushFileBuffers);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
    }

    if (File->BaseAddress) {
        if (!UnmapViewOfFile(File->BaseAddress)) {
            SYS_ERROR(UnmapViewOfFile);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
        File->BaseAddress = NULL;
    }

    if (File->MappingHandle) {
        if (!CloseHandle(File->MappingHandle)) {
            SYS_ERROR(CloseHandle);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
        File->MappingHandle = NULL;
    }

    return Result;
}

PERFECT_HASH_FILE_GET_FLAGS PerfectHashFileGetFlags;

_Use_decl_annotations_
HRESULT
PerfectHashFileGetFlags(
    PPERFECT_HASH_FILE File,
    ULONG SizeOfFlags,
    PPERFECT_HASH_FILE_FLAGS Flags
    )
/*++

Routine Description:

    Returns the flags associated with a loaded file instance.

Arguments:

    File - Supplies a pointer to a PERFECT_HASH_FILE structure for which the
        flags are to be obtained.

    SizeOfFlags - Supplies the size of the structure pointed to by the Flags
        parameter, in bytes.

    Flags - Supplies the address of a variable that receives the flags.

Return Value:

    S_OK - Success.

    E_POINTER - File or Flags is NULL.

    E_INVALIDARG - SizeOfFlags does not match the size of the flags structure.

    PH_E_FILE_LOCKED - The file is locked.

    PH_E_FILE_NOT_OPEN - The file is not open.

--*/
{
    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Flags)) {
        return E_POINTER;
    }

    if (SizeOfFlags != sizeof(*Flags)) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashFileLockShared(File)) {
        return PH_E_FILE_LOCKED;
    }

    if (!IsFileOpen(File)) {
        ReleasePerfectHashFileLockShared(File);
        return PH_E_FILE_NOT_OPEN;
    }

    Flags->AsULong = File->Flags.AsULong;

    ReleasePerfectHashFileLockShared(File);

    return S_OK;
}

PERFECT_HASH_FILE_GET_PATH PerfectHashFileGetPath;

_Use_decl_annotations_
HRESULT
PerfectHashFileGetPath(
    PPERFECT_HASH_FILE File,
    PPERFECT_HASH_PATH *Path
    )
/*++

Routine Description:

    Obtains the path instance for a given file.

Arguments:

    File - Supplies a pointer to a PERFECT_HASH_FILE structure for which the
        path is to be obtained.

    Path - Supplies the address of a variable that receives a pointer to the
        path instance.  The caller must release this reference when finished
        with it via Path->Vtbl->Release(Path).

Return Value:

    S_OK - Success.

    E_POINTER - File or Path parameters were NULL.

    PH_E_FILE_LOCKED - The file is locked exclusively.

    PH_E_FILE_NEVER_OPENED - No file has ever been loaded or created.

--*/
{

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    } else {
        *Path = NULL;
    }

    if (!TryAcquirePerfectHashFileLockShared(File)) {
        return PH_E_FILE_LOCKED;
    }

    if (FileNeverOpened(File)) {
        ReleasePerfectHashFileLockShared(File);
        return PH_E_FILE_NEVER_OPENED;
    }

    //
    // Argument validation complete.  Add a reference to the path and update
    // the caller's pointer, then return success.
    //

    File->Path->Vtbl->AddRef(File->Path);
    *Path = File->Path;

    ReleasePerfectHashFileLockShared(File);

    return S_OK;
}

//
// Helper macro used by GetNames() and GetResources().
//

#define SAVE_POINTER(Name)        \
    if (ARGUMENT_PRESENT(Name)) { \
        *##Name = &File->##Name;  \
        Count++;                  \
    }


PERFECT_HASH_FILE_GET_RESOURCES PerfectHashFileGetResources;

_Use_decl_annotations_
HRESULT
PerfectHashFileGetResources(
    PPERFECT_HASH_FILE File,
    PHANDLE FileHandle,
    PHANDLE MappingHandle,
    PVOID *BaseAddress,
    PVOID *MappedAddress,
    PLARGE_INTEGER EndOfFile
    )
/*++

Routine Description:

    Obtains resources for a given file.

Arguments:

    File - Supplies a pointer to a PERFECT_HASH_FILE structure for which the
        handles and addresses are to be obtained.

    FileHandle - Optionally receives the file handle.

    MappingHandle - Optionally receives the file handle.

    BaseAddress - Optionally receives the base address for file data.

    MappedAddress - Optionally receives the mapped address.

    EndOfFile - Optionally receives current end of file.

Return Value:

    S_OK - Success.

    E_POINTER - File parameter was NULL.

    E_INVALIDARG - All out parameters were NULL.

    PH_E_FILE_LOCKED - The file is locked exclusively.

    PH_E_FILE_NOT_OPEN - No file has been opened, or has been closed.


--*/
{
    BYTE Count = 0;

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!IsFileOpen(File)) {
        return PH_E_FILE_NOT_OPEN;
    }

    if (!TryAcquirePerfectHashFileLockShared(File)) {
        return PH_E_FILE_LOCKED;
    }

    SAVE_POINTER(FileHandle);
    SAVE_POINTER(MappingHandle);
    SAVE_POINTER(BaseAddress);
    SAVE_POINTER(MappedAddress);

    if (ARGUMENT_PRESENT(EndOfFile)) {
        Count++;
        EndOfFile->QuadPart = File->FileInfo.EndOfFile.QuadPart;
    }

    ReleasePerfectHashFileLockShared(File);

    return (Count > 0 ? S_OK : E_INVALIDARG);
}

PERFECT_HASH_FILE_EXTEND PerfectHashFileExtend;

_Use_decl_annotations_
HRESULT
PerfectHashFileExtend(
    PPERFECT_HASH_FILE File,
    PLARGE_INTEGER NewEndOfFile
    )
/*++

Routine Description:

    Unmaps the current file mapping, extends the underlying file, and re-maps
    the file.

Arguments:

    File - Supplies a pointer to the file to extend.

    NewEndOfFile - Supplies a pointer to the new end-of-file size to be used.

Return Value:

    S_OK - File extended successfully.

    E_POINTER - File or NewEndOfFile parameter were NULL.

    E_UNEXPECTED - Internal error.

    PH_E_INVALID_END_OF_FILE - Invalid end of file value.

    PH_E_FILE_NOT_OPEN - The file is not open.

    PH_E_FILE_READONLY - The file is read-only.

    PH_E_SYSTEM_CALL_FAILED - A system call failed and the file was not
        extended.

    PH_E_NEW_EOF_LESS_THAN_OR_EQUAL_TO_CURRENT_EOF - The new end-of-file is
        less than or equal to the current end-of-file.  (To reduce the size
        of an existing file, Truncate() should be used.)

    PH_E_FILE_ALREADY_BEING_EXTENDED - A file extension is already in
        progress.

--*/
{
    HRESULT Result = S_OK;
    LARGE_INTEGER EndOfFile;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(NewEndOfFile)) {
        return E_POINTER;
    }

    EndOfFile.QuadPart = NewEndOfFile->QuadPart;

    if (EndOfFile.QuadPart <= 0) {
        return PH_E_INVALID_END_OF_FILE;
    }

    if (!IsFileOpen(File)) {
        Result = PH_E_FILE_NOT_OPEN;
        goto Error;
    }

    if (IsFileReadOnly(File)) {
        Result = PH_E_FILE_READONLY;
        goto Error;
    }

    if (IsFileBeingExtended(File)) {
        Result = PH_E_FILE_ALREADY_BEING_EXTENDED;
        goto Error;
    }

    File->State.IsBeingExtended = TRUE;

    if (EndOfFile.QuadPart <= File->FileInfo.EndOfFile.QuadPart) {
        Result = PH_E_NEW_EOF_LESS_THAN_OR_EQUAL_TO_CURRENT_EOF;
        goto Error;
    }

    File->PendingEndOfFile.QuadPart = EndOfFile.QuadPart;

    //
    // Validation complete, continue with extension.  The first step is to
    // unmap any existing mapping.
    //

    Result = File->Vtbl->Unmap(File);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileExtend, Result);
        goto Error;
    }

    //
    // Extend the file to the new size via Truncate().
    //

    Result = File->Vtbl->Truncate(File, &EndOfFile);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileTruncate, Result);
        goto Error;
    }

    //
    // Clear the number of bytes written and map the file.
    //

    File->NumberOfBytesWritten.QuadPart = 0;

    Result = File->Vtbl->Map(File);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileExtend, Result);
        goto Error;
    }

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

    File->State.IsBeingExtended = FALSE;

    return Result;
}

PERFECT_HASH_FILE_TRUNCATE PerfectHashFileTruncate;

_Use_decl_annotations_
HRESULT
PerfectHashFileTruncate(
    PPERFECT_HASH_FILE File,
    PLARGE_INTEGER NewEndOfFile
    )
/*++

Routine Description:

    Truncates the file to the given size.  File must be unmapped first via the
    Unmap() call if previously mapped.

Arguments:

    File - Supplies a pointer to the file to truncate.

    NewEndOfFile - Supplies the new file size, in bytes.

Return Value:

    S_OK - File truncated successfully.

    E_POINTER - File or NewEndOfFile parameters were NULL.

    E_UNEXPECTED - Internal error.

    PH_E_FILE_NOT_OPEN - The file is not open.

    PH_E_FILE_READONLY - The file is read-only.

    PH_E_SYSTEM_CALL_FAILED - A system call failed and the file was not
        extended.

--*/
{
    BOOL Success;
    HRESULT Result = S_OK;
    LARGE_INTEGER EndOfFile;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(NewEndOfFile)) {
        return E_POINTER;
    }

    EndOfFile.QuadPart = NewEndOfFile->QuadPart;

    if (EndOfFile.QuadPart < 0) {
        return PH_E_INVALID_END_OF_FILE;
    }

    if (!IsFileOpen(File)) {
        return PH_E_FILE_NOT_OPEN;
    }

    if (IsFileReadOnly(File)) {
        return PH_E_FILE_READONLY;
    }

    //
    // Argument validation complete, continue truncation.
    //

    //
    // Set the file pointer to the desired size.
    //

    Success = SetFilePointerEx(File->FileHandle,
                               EndOfFile,
                               NULL,
                               FILE_BEGIN);

    if (!Success) {
        SYS_ERROR(SetFilePointerEx);
        return PH_E_SYSTEM_CALL_FAILED;
    }

    //
    // Set the end of file.
    //

    Success = SetEndOfFile(File->FileHandle);
    if (!Success) {
        SYS_ERROR(SetEndOfFile);
        return PH_E_SYSTEM_CALL_FAILED;
    }

    //
    // Update the file info to reflect the new size.
    //

    Result = PerfectHashFileUpdateFileInfo(File);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileUpdateFileInfo, Result);
    }

    return Result;
}

PERFECT_HASH_FILE_SCHEDULE_RENAME PerfectHashFileScheduleRename;

_Use_decl_annotations_
HRESULT
PerfectHashFileScheduleRename(
    PPERFECT_HASH_FILE File,
    PPERFECT_HASH_PATH NewPath
    )
/*++

Routine Description:

    Schedules the renaming of the underlying file once it has been unmapped and
    closed.  If this routine is called after a previous rename request has been
    scheduled but not issued, the previous path is released.  That is, it is
    safe to call this routine multiple times prior to the underlying file being
    unmapped and closed.

Arguments:

    File - Supplies a pointer to the file to truncate.

    NewEndOfFile - Supplies the new file size, in bytes.

Return Value:

    S_OK - File extended successfully.

    E_POINTER - File or NewPath parameters were NULL.

    E_UNEXPECTED - Internal error.

    E_INVALIDARG - NewPath was invalid.

    PH_E_FILE_LOCKED - File was locked.

    PH_E_PATH_LOCKED - NewPath was locked.

    PH_E_FILE_NEVER_OPENED - The file has never been opened.

    PH_E_FILE_READONLY - The file is read-only.

    PH_E_RENAME_PATH_IS_SAME_AS_CURRENT_PATH - The rename path requested is the
        same as the current path.

--*/
{
    PRTL Rtl;
    BOOLEAN Equal;
    HRESULT Result = S_OK;
    PVOID OldAddress;
    PVOID OriginalAddress;
    PPERFECT_HASH_PATH OldPath;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(NewPath)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashFileLockExclusive(File)) {
        return PH_E_FILE_LOCKED;
    }

    if (!TryAcquirePerfectHashPathLockExclusive(NewPath)) {
        ReleasePerfectHashFileLockExclusive(File);
        return PH_E_PATH_LOCKED;
    }

    if (FileNeverOpened(File)) {
        Result = PH_E_FILE_NEVER_OPENED;
        goto Error;
    }

    if (IsFileReadOnly(File)) {
        Result = PH_E_FILE_READONLY;
        goto Error;
    }


    //
    // Verify the requested new path and current path differ.
    //

    Rtl = File->Rtl;
    Equal = Rtl->RtlEqualUnicodeString(&File->Path->FullPath,
                                       &NewPath->FullPath,
                                       TRUE);

    if (Equal) {
        Result = PH_E_RENAME_PATH_IS_SAME_AS_CURRENT_PATH;
        goto Error;
    }

    //
    // Argument validation complete.  Atomically exchange the underlying rename
    // path pointer with the new path.
    //

    do {
        OriginalAddress = File->RenamePath;
        OldAddress = InterlockedCompareExchangePointer(&File->RenamePath,
                                                       NewPath,
                                                       OriginalAddress);
    } while (OriginalAddress != OldAddress);

    OldPath = (PPERFECT_HASH_PATH)OriginalAddress;

    //
    // Release the old path if one was present and add ref on the new path.
    //

    if (OldPath) {
        OldPath->Vtbl->Release(OldPath);
        OldPath = NULL;
    }

    NewPath->Vtbl->AddRef(NewPath);

    //
    // If the file has already been closed, do the rename now.
    //

    if (IsFileClosed(File)) {
        Result = File->Vtbl->DoRename(File);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileDoRename, Result);
        }
    }

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

    ReleasePerfectHashPathLockExclusive(NewPath);
    ReleasePerfectHashFileLockExclusive(File);

    return Result;
}

PERFECT_HASH_FILE_DO_RENAME PerfectHashFileDoRename;

_Use_decl_annotations_
HRESULT
PerfectHashFileDoRename(
    PPERFECT_HASH_FILE File
    )
/*++

Routine Description:

    Renames a closed file based on a previous rename request via ScheduleRename.

Arguments:

    File - Supplies a pointer to the file to be renamed.

Return Value:

    S_OK - File renamed successfully.

    E_POINTER - File was NULL.

    PH_E_FILE_NEVER_OPENED - The file was never opened.

    PH_E_FILE_NOT_CLOSED - The file is not closed.

    PH_E_FILE_NO_RENAME_SCHEDULED - No rename is scheduled.

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

--*/
{
    BOOL Success;
    ULONG MoveFileFlags;
    HRESULT Result = S_OK;
    PPERFECT_HASH_PATH OldPath;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (FileNeverOpened(File)) {
        return PH_E_FILE_NEVER_OPENED;
    }

    if (IsFileOpen(File)) {
        return PH_E_FILE_NOT_CLOSED;
    }

    if (!IsRenameScheduled(File)) {
        return PH_E_FILE_NO_RENAME_SCHEDULED;
    }

    //
    // File->Path should be non-NULL, and point to a 'set' path.
    //

    if (!File->Path || !IsPathSet(File->Path)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    //
    // Argument validation complete.  Continue with rename.
    //

    MoveFileFlags = MOVEFILE_REPLACE_EXISTING;

    Success = MoveFileExW(File->Path->FullPath.Buffer,
                          File->RenamePath->FullPath.Buffer,
                          MoveFileFlags);

    if (!Success) {
        SYS_ERROR(MoveFileEx);
        return PH_E_SYSTEM_CALL_FAILED;
    }

    //
    // File was renamed successfully.  Update the path variables accordingly.
    //

    OldPath = File->Path;
    File->Path = File->RenamePath;
    File->RenamePath = NULL;

    //
    // Release the old path's lock and COM reference count.
    //

    OldPath->Vtbl->Release(OldPath);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
