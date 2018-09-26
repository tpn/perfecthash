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
    PRTL Rtl;
    HRESULT Result;
    PALLOCATOR Allocator;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return;
    }

    //
    // Sanity check structure size.
    //

    ASSERT(File->SizeOfStruct == sizeof(*File));

    //
    // Initialize aliases.
    //

    Rtl = File->Rtl;
    Allocator = File->Allocator;

    if (!Rtl) {
        return;
    }

    ASSERT(Allocator);

    //
    // Close the file if necessary.
    //

    if (!File->State.IsClosed) {
        Result = File->Vtbl->Close(File);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileRundown, Result);
        }
        ASSERT(IsFileClosed(File));
    }

    //
    // Release COM references.
    //

    File->Path->Vtbl->Release(File->Path);
    File->Path = NULL;

    Allocator->Vtbl->Release(Allocator);
    File->Allocator = NULL;

    Rtl->Vtbl->Release(Rtl);
    File->Rtl = NULL;

    return;
}

PERFECT_HASH_FILE_LOAD PerfectHashFileLoad;

_Use_decl_annotations_
HRESULT
PerfectHashFileLoad(
    PPERFECT_HASH_FILE File,
    PPERFECT_HASH_PATH SourcePath,
    PPERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlagsPointer,
    PLARGE_INTEGER EndOfFile
    )
/*++

Routine Description:

    Loads a a file.

Arguments:

    File - Supplies a pointer to the file to load.

    SourcePath - Supplies a pointer to the path instance to load.

    FileLoadFlags - Optionally supplies a pointer to file load flags that can
        be used to customize load behavior.

    EndOfFile - Optionally supplies a pointer to a variable that will receive
        the current end of file (i.e. size in bytes) if the file was loaded
        successfully.

Return Value:

    S_OK - File was closed successfully.

    E_POINTER - File or Path parameters were NULL.

    PH_E_INVALID_FILE_LOAD_FLAGS - Invalid file load flags.

    PH_E_SOURCE_PATH_LOCKED - Source path is locked.

    PH_E_SOURCE_PATH_NO_PATH_SET - Source path has not been set.

    PH_E_FILE_LOCKED - The file is locked.

    PH_E_FILE_ALREADY_OPEN - An existing file has already been loaded/created.

    PH_E_SYSTEM_CALL_FAILED - A system call failed; the file may be in an
        inconsistent state.

--*/
{
    PRTL Rtl;
    BOOL Success;
    ULONG ShareMode;
    ULONG DesiredAccess;
    ULONG FlagsAndAttributes;
    PALLOCATOR Allocator;
    HRESULT Result = S_OK;
    BOOLEAN Opened = FALSE;
    PPERFECT_HASH_PATH Path;
    PERFECT_HASH_FILE_LOAD_FLAGS FileLoadFlags = { 0 };
    PCPERFECT_HASH_PATH_PARTS Parts = NULL;

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
        ReleasePerfectHashPathLockShared(SourcePath);
        ReleasePerfectHashFileLockExclusive(File);
        return PH_E_FILE_ALREADY_OPEN;
    }

    if (IsFileClosed(File)) {
        ReleasePerfectHashPathLockShared(SourcePath);
        ReleasePerfectHashFileLockExclusive(File);
        return PH_E_FILE_ALREADY_CLOSED;
    }

    //
    // Argument validation complete, continue with loading.
    //

    //
    // Initialize aliases and load flags.
    //

    Rtl = File->Rtl;
    Path = File->Path;
    Allocator = File->Allocator;

    if (FileLoadFlags.DisableTryLargePagesForFileData) {
        File->Flags.DoesNotWantLargePages = TRUE;
    }

    //
    // Deep copy the incoming path and extract the various components.
    //

    Result = Path->Vtbl->Copy(Path, &SourcePath->FullPath, &Parts, NULL);

    ReleasePerfectHashPathLockShared(Path);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashPathCopy, Result);
        goto Error;
    }

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

    SetFileOpened(File);
    SetFileLoaded(File);

    //
    // Get the current file size.
    //

    Success = GetFileInformationByHandleEx(File->FileHandle,
                                           FileStandardInfo,
                                           &File->FileInfo,
                                           sizeof(File->FileInfo));

    if (!Success) {
        SYS_ERROR(GetFileInformationByHandleEx);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Error out if the file is empty; Load()'ing implies memory mapping an
    // existing file, which won't work on an empty file.
    //

    if (File->FileInfo.EndOfFile.QuadPart > 0) {
        Result = PH_E_FILE_EMPTY;
        PH_ERROR(PerfectHashFileLoad, Result);
        goto Error;
    }

    //
    // Update the caller's EndOfFile pointer if applicable.
    //

    if (ARGUMENT_PRESENT(EndOfFile)) {
        EndOfFile->QuadPart = File->FileInfo.EndOfFile.QuadPart;
    }

    //
    // Set the mapping size to the total size of the file.
    //

    File->MappingSize.QuadPart = File->FileInfo.EndOfFile.QuadPart;

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

End:

    ReleasePerfectHashFileLockExclusive(File);

    if (Opened && FAILED(Result)) {
        HRESULT CloseResult;

        CloseResult = File->Vtbl->Close(File);
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
    PPERFECT_HASH_FILE File
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

Return Value:

    S_OK - File was closed successfully.

    E_POINTER - File parameter was NULL.

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

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashFileLockExclusive(File)) {
        return PH_E_FILE_LOCKED;
    }

    if (!IsFileOpen(File)) {
        ReleasePerfectHashFileLockExclusive(File);
        return PH_E_FILE_NOT_OPEN;
    }

    //
    // Unmap the file.
    //

    Result = File->Vtbl->Unmap(File);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileUnmap, Result);
    }

    //
    // If bytes have been written to the mapping, truncate the file.
    //

    if (File->NumberOfBytesWritten.QuadPart > 0) {
        if (IsFileReadOnly(File)) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashFileClose, Result);
        } else {
            Result = File->Vtbl->Truncate(File, File->NumberOfBytesWritten);
            if (FAILED(Result)) {
                PH_ERROR(PerfectHashFileTruncate, Result);
            }
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

    PH_E_FILE_MAPPING_SIZE_IS_ZERO - The mapping size is 0.

    PH_E_FILE_MAPPING_SIZE_NOT_SYSTEM_ALIGNED - The mapping size for the file is
        not aligned to the system allocation granularity.

    PH_E_FILE_MAPPING_SIZE_NOT_LARGE_PAGE_ALIGNED - The mapping size for the
        file is not aligned to the large page granularity, and large pages have
        been requested.

    PH_E_FILE_NOT_OPEN - File is not open.

    PH_E_FILE_VIEW_CREATED - A view has already been created.

    PH_E_FILE_VIEW_MAPPED - A view has already been mapped.

    PH_E_SYSTEM_CALL_FAILED - A system call failed; the file may be in an
        inconsistent state.

--*/
{
    PRTL Rtl;
    ULONG Access = FILE_MAP_READ;
    ULONG Protection;
    SIZE_T BytesToMap;
    HRESULT Result = S_OK;
    ULARGE_INTEGER Aligned;

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

    if (!IsFileReadOnly(File)) {

        Access |= FILE_MAP_WRITE;

        if (File->MappingSize.QuadPart == 0) {

            return PH_E_FILE_MAPPING_SIZE_IS_ZERO;

        } else if (WantsLargePages(File)) {

            Aligned.QuadPart = ALIGN_UP_LARGE_PAGE(File->MappingSize.QuadPart);

            if (File->MappingSize.QuadPart != Aligned.QuadPart) {
                return PH_E_FILE_MAPPING_SIZE_NOT_LARGE_PAGE_ALIGNED;
            }

        } else {

            SYSTEM_INFO SystemInfo;

            GetSystemInfo(&SystemInfo);

            Aligned.QuadPart = ALIGN_UP(File->MappingSize.QuadPart,
                                        SystemInfo.dwAllocationGranularity);

            if (File->MappingSize.QuadPart != Aligned.QuadPart) {
                return PH_E_FILE_MAPPING_SIZE_NOT_SYSTEM_ALIGNED;
            }

        }

    }

    //
    // Argument validation complete.
    //

    Rtl = File->Rtl;

    //
    // Create the file mapping.
    //

    Protection = (IsFileReadOnly(File) ? PAGE_READONLY : PAGE_READWRITE);

    File->MappingHandle = CreateFileMappingW(File->FileHandle,
                                             NULL,
                                             Protection,
                                             File->MappingSize.HighPart,
                                             File->MappingSize.LowPart,
                                             NULL);

    if (!IsValidHandle(File->MappingHandle)) {
        SYS_ERROR(CreateFileMappingW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Successfully created the file mapping.  Now, map it into memory.
    //

    BytesToMap = LARGE_INTEGER_TO_SIZE_T(File->MappingSize);

    File->BaseAddress = MapViewOfFile(File->MappingHandle,
                                      Access,
                                      0,
                                      0,
                                      BytesToMap);

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
            ALIGN_UP_LARGE_PAGE(LARGE_INTEGER_TO_SIZE_T(File->MappingSize))
        );

        LargePageAddress = VirtualAlloc(NULL,
                                        LargePageAllocSize,
                                        LargePageAllocFlags,
                                        Protection);

        if (LargePageAddress) {

            //
            // The large page allocation was successful.
            //

            ULONG_PTR NumberOfPages;

            File->Flags.UsesLargePages = TRUE;
            File->MappedAddress = File->BaseAddress;
            File->BaseAddress = LargePageAddress;

            NumberOfPages = BYTES_TO_PAGES(F);

            Rtl->Vtbl->CopyPages(Rtl,
                                 LargePageAddress,
                                 File->BaseAddress,
                                 (ULONG)NumberOfPages);
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

    //
    // Flush file buffers, potentially free a large page allocation, unmap the
    // file view, and close the mapping handle.
    //

    if (File->FileHandle) {
        if (!FlushFileBuffers(File->FileHandle)) {
            SYS_ERROR(FlushFileBuffers);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
    }

    if (!File->MappedAddress) {

        ASSERT(!File->Flags.UsesLargePages);

    } else if (File->BaseAddress) {

        //
        // If MappedAddress is non-NULL, BaseAddress is actually our
        // large page address which needs to be freed with VirtualFree().
        //

        ASSERT(File->Flags.UsesLargePages);
        if (!VirtualFree(File->BaseAddress, 0, MEM_RELEASE)) {
            SYS_ERROR(VirtualFree);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }

        //
        // Switch the base address back so it's unmapped correctly below.
        //

        File->BaseAddress = File->MappedAddress;
        File->MappedAddress = NULL;
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
    PCPERFECT_HASH_PATH *Path
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

    PH_E_FILE_NOT_OPEN - No file has been opened, or has been closed.

--*/
{
    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    }

    //
    // Clear the caller's pointer up-front.
    //

    *Path = NULL;

    if (!TryAcquirePerfectHashFileLockShared(File)) {
        return PH_E_FILE_LOCKED;
    }

    if (!IsFileOpen(File)) {
        ReleasePerfectHashFileLockShared(File);
        return PH_E_FILE_NOT_OPEN;
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
    PULARGE_INTEGER *MappingSize
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

    MappingSize - Optionally receives the mapping size.

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

    if (!TryAcquirePerfectHashFileLockShared(File)) {
        return PH_E_FILE_LOCKED;
    }

    if (!IsFileOpen(File)) {
        return PH_E_FILE_NOT_OPEN;
    }

    SAVE_POINTER(FileHandle);
    SAVE_POINTER(MappingHandle);
    SAVE_POINTER(BaseAddress);
    SAVE_POINTER(MappedAddress);
    SAVE_POINTER(MappingSize);

    ReleasePerfectHashFileLockShared(File);

    return (Count > 0 ? S_OK : E_INVALIDARG);
}

PERFECT_HASH_FILE_EXTEND PerfectHashFileExtend;

_Use_decl_annotations_
HRESULT
PerfectHashFileExtend(
    PPERFECT_HASH_FILE File,
    ULARGE_INTEGER NewMappingSize
    )
/*++

Routine Description:

    Unmaps the current file mapping, extends the underlying file, and re-maps
    the file.

Arguments:

    File - Supplies a pointer to the file to extend.

    NewMappingSize - Supplies the new mapping size to be used.  This value
        will be automatically aligned up to a system allocation granularity
        if necessary.

Return Value:

    S_OK - File extended successfully.

    E_POINTER - File parameter was NULL.

    E_INVALIDARG - NewMappingSize was 0.

    E_UNEXPECTED - Internal error.

    PH_E_FILE_LOCKED - The file is locked.

    PH_E_FILE_NOT_OPEN - The file is not open.

    PH_E_FILE_READONLY - The file is read-only.

    PH_E_SYSTEM_CALL_FAILED - A system call failed and the file was not
        extended.

    PH_E_MAPPING_SIZE_LESS_THAN_OR_EQUAL_TO_CURRENT_SIZE - The supplied size
        by NewMappingSize is less than or equal to the current file size.

--*/
{
    PRTL Rtl;
    BOOL Success;
    HRESULT Result = S_OK;
    SYSTEM_INFO SystemInfo;
    FILE_STANDARD_INFO FileInfo;
    ULONG AllocationGranularity;
    LARGE_INTEGER EndOfFile;
    ULARGE_INTEGER MappingSize;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (NewMappingSize.QuadPart == 0) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashFileLockExclusive(File)) {
        return PH_E_FILE_LOCKED;
    }

    if (!IsFileOpen(File)) {
        Result = PH_E_FILE_NOT_OPEN;
        goto Error;
    }

    if (IsFileReadOnly(File)) {
        Result = PH_E_FILE_READONLY;
        goto Error;
    }

    Rtl = File->Rtl;

    Success = GetFileInformationByHandleEx(File->FileHandle,
                                           FileStandardInfo,
                                           &FileInfo,
                                           sizeof(FileInfo));

    if (!Success) {
        SYS_ERROR(GetFileInformationByHandleEx);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    GetSystemInfo(&SystemInfo);
    AllocationGranularity = SystemInfo.dwAllocationGranularity;

    //
    // Align the provided size up to the system allocation granularity.
    //

    MappingSize.QuadPart = ALIGN_UP(NewMappingSize.QuadPart,
                                    AllocationGranularity);

    if (FileInfo.EndOfFile.QuadPart >= (LONGLONG)MappingSize.QuadPart) {
        Result = PH_E_MAPPING_SIZE_LESS_THAN_OR_EQUAL_TO_CURRENT_SIZE;
        goto Error;
    }

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

    EndOfFile.QuadPart = (LONGLONG)MappingSize.QuadPart;

    Result = File->Vtbl->Truncate(File, EndOfFile);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileTruncate, Result);
        goto Error;
    }

    //
    // Update the mapping size, clear the number of bytes written, and re-map
    // the file.
    //

    File->MappingSize.QuadPart = MappingSize.QuadPart;
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

    ReleasePerfectHashFileLockExclusive(File);

    return Result;
}

PERFECT_HASH_FILE_TRUNCATE PerfectHashFileTruncate;

_Use_decl_annotations_
HRESULT
PerfectHashFileTruncate(
    PPERFECT_HASH_FILE File,
    LARGE_INTEGER NewEndOfFile
    )
/*++

Routine Description:

    Truncates the file to the given size.  File must be unmapped first via the
    Unmap() call if previously mapped.

Arguments:

    File - Supplies a pointer to the file to truncate.

    NewEndOfFile - Supplies the new file size, in bytes.

Return Value:

    S_OK - File extended successfully.

    E_POINTER - File parameter was NULL.

    E_UNEXPECTED - Internal error.

    PH_E_FILE_NOT_OPEN - The file is not open.

    PH_E_FILE_READONLY - The file is read-only.

    PH_E_SYSTEM_CALL_FAILED - A system call failed and the file was not
        extended.

--*/
{
    PRTL Rtl;
    BOOL Success;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
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

    Rtl = File->Rtl;

    //
    // Set the file pointer to the desired size.
    //

    Success = SetFilePointerEx(File->FileHandle,
                               NewEndOfFile,
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
    // Indicate success and return.
    //

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
