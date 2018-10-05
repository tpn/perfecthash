/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashDirectory.c

Abstract:

    This is the module for the PERFECT_HASH_DIRECTORY component of the perfect
    hash library.  Routines are provided for initialization, rundown, getting
    flags, getting the path, scheduling and doing renames.

--*/

#include "stdafx.h"

PERFECT_HASH_DIRECTORY_INITIALIZE PerfectHashDirectoryInitialize;

_Use_decl_annotations_
HRESULT
PerfectHashDirectoryInitialize(
    PPERFECT_HASH_DIRECTORY Directory
    )
/*++

Routine Description:

    Initializes a perfect hash directory structure.  This is a relatively simple
    method that just primes the COM scaffolding; the bulk of the work is done
    when opening or creating the directory.

Arguments:

    Directory - Supplies a pointer to a PERFECT_HASH_DIRECTORY structure for
        which initialization is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - Directory is NULL.

    E_UNEXPECTED - All other errors.

--*/
{
    HRESULT Result = S_OK;

    if (!ARGUMENT_PRESENT(Directory)) {
        return E_POINTER;
    }

    Directory->SizeOfStruct = sizeof(*Directory);

    //
    // Create Rtl and Allocator components.
    //

    Result = Directory->Vtbl->CreateInstance(Directory,
                                             NULL,
                                             &IID_PERFECT_HASH_RTL,
                                             &Directory->Rtl);

    if (FAILED(Result)) {
        goto Error;
    }

    Result = Directory->Vtbl->CreateInstance(Directory,
                                             NULL,
                                             &IID_PERFECT_HASH_ALLOCATOR,
                                             &Directory->Allocator);

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

PERFECT_HASH_DIRECTORY_RUNDOWN PerfectHashDirectoryRundown;

_Use_decl_annotations_
VOID
PerfectHashDirectoryRundown(
    PPERFECT_HASH_DIRECTORY Directory
    )
/*++

Routine Description:

    Release all resources associated with a perfect hash directory.

Arguments:

    Directory - Supplies a pointer to a PERFECT_HASH_DIRECTORY structure for
        which rundown is to be performed.

Return Value:

    None.

--*/
{
    HRESULT Result;

    //
    // Sanity check structure size.
    //

    ASSERT(Directory->SizeOfStruct == sizeof(*Directory));

    //
    // Close the directory if necessary.
    //

    if (!IsDirectoryClosed(Directory)) {
        Result = Directory->Vtbl->Close(Directory);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashDirectoryClose, Result);
        }
    }

    //
    // Release COM references.
    //

    RELEASE(Directory->Path);
    RELEASE(Directory->RenamePath);
    RELEASE(Directory->Rtl);
    RELEASE(Directory->Allocator);

    return;
}

PERFECT_HASH_DIRECTORY_OPEN PerfectHashDirectoryOpen;

_Use_decl_annotations_
HRESULT
PerfectHashDirectoryOpen(
    PPERFECT_HASH_DIRECTORY Directory,
    PPERFECT_HASH_PATH SourcePath,
    PPERFECT_HASH_DIRECTORY_OPEN_FLAGS DirectoryOpenFlagsPointer
    )
/*++

Routine Description:

    Opens a a directory.

Arguments:

    Directory - Supplies a pointer to the directory to open.

    SourcePath - Supplies a pointer to the path instance to open.

    DirectoryOpenFlags - Optionally supplies a pointer to directory open flags that can
        be used to customize open behavior.

Return Value:

    S_OK - Directory was closed successfully.

    E_POINTER - Directory or Path parameters were NULL.

    PH_E_INVALID_DIRECTORY_OPEN_FLAGS - Invalid directory open flags.

    PH_E_SOURCE_PATH_LOCKED - Source path is locked.

    PH_E_SOURCE_PATH_NO_PATH_SET - Source path has not been set.

    PH_E_DIRECTORY_LOCKED - The directory is locked.

    PH_E_DIRECTORY_ALREADY_SET - An existing directory has already been
        opened or created.

    PH_E_DIRECTORY_ALREADY_CLOSED - Directory has been opened and then
        subsequently closed.

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

--*/
{
    ULONG LastError;
    ULONG ShareMode;
    ULONG DesiredAccess;
    ULONG FlagsAndAttributes;
    HRESULT Result = S_OK;
    BOOLEAN Opened = FALSE;
    PERFECT_HASH_DIRECTORY_OPEN_FLAGS DirectoryOpenFlags = { 0 };

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Directory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(SourcePath)) {
        return E_POINTER;
    }

    VALIDATE_FLAGS(DirectoryOpen, DIRECTORY_OPEN);

    if (!TryAcquirePerfectHashPathLockShared(SourcePath)) {
        return PH_E_SOURCE_PATH_LOCKED;
    }

    if (!IsPathSet(SourcePath)) {
        ReleasePerfectHashPathLockShared(SourcePath);
        return PH_E_SOURCE_PATH_NO_PATH_SET;
    }

    if (!TryAcquirePerfectHashDirectoryLockExclusive(Directory)) {
        ReleasePerfectHashPathLockShared(SourcePath);
        return PH_E_DIRECTORY_LOCKED;
    }

    if (IsDirectorySet(Directory)) {
        Result = PH_E_DIRECTORY_ALREADY_SET;
        goto Error;
    }

    if (IsDirectoryClosed(Directory)) {
        Result = PH_E_DIRECTORY_ALREADY_CLOSED;
        goto Error;
    }

    //
    // Invariant check: Directory->Path and Directory->RenamePath should both be
    // NULL at this point.
    //

    if (Directory->Path) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashDirectoryOpen, Result);
        goto Error;
    }

    if (Directory->RenamePath) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashDirectoryOpen, Result);
        goto Error;
    }

    //
    // Argument validation complete, continue with opening.
    //

    Directory->State.IsReadOnly = TRUE;

    //
    // Add a reference to the source path.
    //

    SourcePath->Vtbl->AddRef(SourcePath);
    Directory->Path = SourcePath;

    //
    // Open a handle.
    //

    DesiredAccess = GENERIC_READ;
    ShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    FlagsAndAttributes = FILE_FLAG_BACKUP_SEMANTICS;

    Directory->DirectoryHandle = CreateFileW(Directory->Path->FullPath.Buffer,
                                             DesiredAccess,
                                             ShareMode,
                                             NULL,
                                             OPEN_EXISTING,
                                             FlagsAndAttributes,
                                             NULL);

    LastError = GetLastError();

    if (!IsValidHandle(Directory->DirectoryHandle)) {
        Directory->DirectoryHandle = NULL;
        SYS_ERROR(CreateDirectoryW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (LastError == ERROR_SUCCESS) {

        Directory->State.WasCreatedByUs = TRUE;

    } else if (LastError != ERROR_ALREADY_EXISTS) {

        //
        // If we get here, the only permissible value for last error, other
        // than success, is ERROR_ALREADY_EXISTS.  So, as we've encountered
        // something else, treat that as an invariant failure.
        //

        PH_ERROR(PerfectHashDirectoryOpen_InvalidLastError, LastError);
        Result = PH_E_INVARIANT_CHECK_FAILED;
        goto Error;

    } else {
        Directory->State.WasCreatedByUs = FALSE;
    }

    Opened = TRUE;
    SetDirectoryOpened(Directory);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    RELEASE(Directory->Path);

    //
    // Intentional follow-on to End.
    //

End:

    ReleasePerfectHashPathLockShared(SourcePath);
    ReleasePerfectHashDirectoryLockExclusive(Directory);

    if (Opened && FAILED(Result)) {
        HRESULT CloseResult;

        CloseResult = Directory->Vtbl->Close(Directory);
        if (FAILED(CloseResult)) {
            PH_ERROR(PerfectHashDirectoryClose, CloseResult);
        }
    }

    return Result;
}

PERFECT_HASH_DIRECTORY_CREATE PerfectHashDirectoryCreate;

_Use_decl_annotations_
HRESULT
PerfectHashDirectoryCreate(
    PPERFECT_HASH_DIRECTORY Directory,
    PPERFECT_HASH_PATH SourcePath,
    PPERFECT_HASH_DIRECTORY_CREATE_FLAGS DirectoryCreateFlagsPointer
    )
/*++

Routine Description:

    Creates a directory.

Arguments:

    Directory - Supplies a pointer to the directory to create.

    SourcePath - Supplies a pointer to the path instance.

    DirectoryCreateFlags - Optionally supplies a pointer to directory create
        flags that can be used to customize create behavior.

Return Value:

    S_OK - Directory was created successfully.

    E_POINTER - Directory or SourcePath parameters were NULL.

    PH_E_INVALID_DIRECTORY_CREATE_FLAGS - Invalid directory create flags.

    PH_E_SOURCE_PATH_LOCKED - Source path is locked.

    PH_E_SOURCE_PATH_NO_PATH_SET - Source path has not been set.

    PH_E_DIRECTORY_LOCKED - The directory is locked.

    PH_E_DIRECTORY_ALREADY_SET - An existing directory has already been opened
        or created.

    PH_E_DIRECTORY_ALREADY_CLOSED - Directory has been opened and then
        subsequently closed already.

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

--*/
{
    ULONG LastError;
    ULONG ShareMode;
    ULONG DesiredAccess;
    ULONG FlagsAndAttributes;
    HRESULT Result = S_OK;
    BOOLEAN Opened = FALSE;
    PERFECT_HASH_DIRECTORY_CREATE_FLAGS DirectoryCreateFlags = { 0 };

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Directory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(SourcePath)) {
        return E_POINTER;
    }

    VALIDATE_FLAGS(DirectoryCreate, DIRECTORY_CREATE);

    if (!TryAcquirePerfectHashPathLockShared(SourcePath)) {
        return PH_E_SOURCE_PATH_LOCKED;
    }

    if (!IsPathSet(SourcePath)) {
        ReleasePerfectHashPathLockShared(SourcePath);
        return PH_E_SOURCE_PATH_NO_PATH_SET;
    }

    if (!TryAcquirePerfectHashDirectoryLockExclusive(Directory)) {
        ReleasePerfectHashPathLockShared(SourcePath);
        return PH_E_DIRECTORY_LOCKED;
    }

    if (IsDirectorySet(Directory)) {
        Result = PH_E_DIRECTORY_ALREADY_SET;
        goto Error;
    }

    if (IsDirectoryClosed(Directory)) {
        Result = PH_E_DIRECTORY_ALREADY_CLOSED;
        goto Error;
    }

    //
    // Invariant check: Directory->Path and Directory->RenamePath should both be
    // NULL at this point.
    //

    if (Directory->Path) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashDirectoryCreate, Result);
        goto Error;
    }

    if (Directory->RenamePath) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashDirectoryCreate, Result);
        goto Error;
    }

    //
    // Argument validation complete, continue with creation.
    //

    Directory->State.IsReadOnly = FALSE;

    //
    // Add a reference to the source path.
    //

    SourcePath->Vtbl->AddRef(SourcePath);
    Directory->Path = SourcePath;

    //
    // Attempt to create the directory.
    //

    if (!CreateDirectoryW(Directory->Path->FullPath.Buffer, NULL)) {
        LastError = GetLastError();
        if (LastError != ERROR_ALREADY_EXISTS) {
            SYS_ERROR(CreateDirectoryW);
            goto Error;
        }
    } else {
        Directory->State.WasCreatedByUs = TRUE;
    }

    //
    // Open the directory using the newly created path.
    //

    DesiredAccess = GENERIC_READ | GENERIC_WRITE;
    ShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    FlagsAndAttributes = FILE_FLAG_BACKUP_SEMANTICS;

    Directory->DirectoryHandle = CreateFileW(Directory->Path->FullPath.Buffer,
                                             DesiredAccess,
                                             ShareMode,
                                             NULL,
                                             OPEN_ALWAYS,
                                             FlagsAndAttributes,
                                             NULL);

    LastError = GetLastError();

    if (!IsValidHandle(Directory->DirectoryHandle)) {
        Directory->DirectoryHandle = NULL;
        SYS_ERROR(CreateDirectoryW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Opened = TRUE;
    SetDirectoryCreated(Directory);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    RELEASE(Directory->Path);

    //
    // Intentional follow-on to End.
    //

End:

    ReleasePerfectHashPathLockShared(SourcePath);
    ReleasePerfectHashDirectoryLockExclusive(Directory);

    if (Opened && FAILED(Result)) {
        HRESULT CloseResult;

        CloseResult = Directory->Vtbl->Close(Directory);
        if (FAILED(CloseResult)) {
            PH_ERROR(PerfectHashDirectoryClose, CloseResult);
        }
    }

    return Result;
}

PERFECT_HASH_DIRECTORY_CLOSE PerfectHashDirectoryClose;

_Use_decl_annotations_
HRESULT
PerfectHashDirectoryClose(
    PPERFECT_HASH_DIRECTORY Directory
    )
/*++

Routine Description:

    Closes a previously opened or created directory.

    If this routine returns success, the instance enters a 'closed' state, and
    no further operations may be performed on it.

Arguments:

    Directory - Supplies a pointer to the directory to close.

Return Value:

    S_OK - Directory was closed successfully.

    E_POINTER - Directory parameter was NULL.

    PH_E_DIRECTORY_LOCKED - The directory is locked.

    PH_E_DIRECTORY_NOT_SET - The directory is not set.

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

--*/
{
    HRESULT Result = S_OK;
    ULONG LastError;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Directory)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashDirectoryLockExclusive(Directory)) {
        return PH_E_DIRECTORY_LOCKED;
    }

    if (!IsDirectorySet(Directory)) {
        ReleasePerfectHashDirectoryLockExclusive(Directory);
        return PH_E_DIRECTORY_NOT_SET;
    }

    //
    // Close the directory handle, if applicable.
    //

    if (Directory->DirectoryHandle) {
        if (!CloseHandle(Directory->DirectoryHandle)) {
            SYS_ERROR(CloseHandle);
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
        Directory->DirectoryHandle = NULL;
    }

    SetDirectoryClosed(Directory);

    if (DirectoryWasCreatedByUs(Directory)) {

        //
        // Attempt to delete the directory, which will fail if there are any
        // files under it.  If it succeeds, and a rename was scheduled, release
        // the rename path's lock and ref count.
        //

        if (!RemoveDirectoryW(Directory->Path->FullPath.Buffer)) {

            //
            // Ignore ERROR_DIR_NOT_EMPTY.
            //

            LastError = GetLastError();
            if (LastError == ERROR_DIR_NOT_EMPTY) {
                SetLastError(ERROR_SUCCESS);
            } else {
                SYS_ERROR(RemoveDirectoryW);
                Result = PH_E_SYSTEM_CALL_FAILED;
            }
        } else if (IsDirectoryRenameScheduled(Directory)) {
            Directory->RenamePath->Vtbl->Release(Directory->RenamePath);
            Directory->RenamePath = NULL;
        }
    }

    //
    // If a rename has been scheduled, do it now.
    //

    if (IsDirectoryRenameScheduled(Directory)) {
        ASSERT(!IsDirectoryReadOnly(Directory));
        Result = Directory->Vtbl->DoRename(Directory);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashDirectoryDoRename, Result);
        }
    }

    ReleasePerfectHashDirectoryLockExclusive(Directory);

    return Result;
}

PERFECT_HASH_DIRECTORY_GET_FLAGS PerfectHashDirectoryGetFlags;

_Use_decl_annotations_
HRESULT
PerfectHashDirectoryGetFlags(
    PPERFECT_HASH_DIRECTORY Directory,
    ULONG SizeOfFlags,
    PPERFECT_HASH_DIRECTORY_FLAGS Flags
    )
/*++

Routine Description:

    Returns the flags associated with a loaded directory instance.

Arguments:

    Directory - Supplies a pointer to a PERFECT_HASH_DIRECTORY structure for
        which the flags are to be obtained.

    SizeOfFlags - Supplies the size of the structure pointed to by the Flags
        parameter, in bytes.

    Flags - Supplies the address of a variable that receives the flags.

Return Value:

    S_OK - Success.

    E_POINTER - Directory or Flags is NULL.

    E_INVALIDARG - SizeOfFlags does not match the size of the flags structure.

    PH_E_DIRECTORY_LOCKED - The directory is locked.

    PH_E_DIRECTORY_NOT_SET - The directory is not set.

--*/
{
    if (!ARGUMENT_PRESENT(Directory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Flags)) {
        return E_POINTER;
    }

    if (SizeOfFlags != sizeof(*Flags)) {
        return E_INVALIDARG;
    }

    if (!TryAcquirePerfectHashDirectoryLockShared(Directory)) {
        return PH_E_DIRECTORY_LOCKED;
    }

    if (!IsDirectorySet(Directory)) {
        ReleasePerfectHashDirectoryLockShared(Directory);
        return PH_E_DIRECTORY_NOT_SET;
    }

    Flags->AsULong = Directory->Flags.AsULong;

    ReleasePerfectHashDirectoryLockShared(Directory);

    return S_OK;
}

PERFECT_HASH_DIRECTORY_GET_PATH PerfectHashDirectoryGetPath;

_Use_decl_annotations_
HRESULT
PerfectHashDirectoryGetPath(
    PPERFECT_HASH_DIRECTORY Directory,
    PPERFECT_HASH_PATH *Path
    )
/*++

Routine Description:

    Obtains the path instance for a given directory.

Arguments:

    Directory - Supplies a pointer to a PERFECT_HASH_DIRECTORY structure for
        which the path is to be obtained.

    Path - Supplies the address of a variable that receives a pointer to the
        path instance.  The caller must release this reference when finished
        with it via Path->Vtbl->Release(Path).

Return Value:

    S_OK - Success.

    E_POINTER - Directory or Path parameters were NULL.

    PH_E_DIRECTORY_LOCKED - The directory is locked exclusively.

    PH_E_DIRECTORY_NEVER_SET - No directory has ever been opened or created.

--*/
{

    if (!ARGUMENT_PRESENT(Directory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Path)) {
        return E_POINTER;
    } else {
        *Path = NULL;
    }

    if (!TryAcquirePerfectHashDirectoryLockShared(Directory)) {
        return PH_E_DIRECTORY_LOCKED;
    }

    if (DirectoryNeverSet(Directory)) {
        ReleasePerfectHashDirectoryLockShared(Directory);
        return PH_E_DIRECTORY_NEVER_SET;
    }

    //
    // Argument validation complete.  Add a reference to the path and update
    // the caller's pointer, then return success.
    //

    Directory->Path->Vtbl->AddRef(Directory->Path);
    *Path = Directory->Path;

    ReleasePerfectHashDirectoryLockShared(Directory);

    return S_OK;
}

PERFECT_HASH_DIRECTORY_SCHEDULE_RENAME PerfectHashDirectoryScheduleRename;

_Use_decl_annotations_
HRESULT
PerfectHashDirectoryScheduleRename(
    PPERFECT_HASH_DIRECTORY Directory,
    PPERFECT_HASH_PATH NewPath
    )
/*++

Routine Description:

    Schedules the renaming of the underlying directory once it has been closed.
    If this routine is called after a previous rename request has been scheduled
    but not issued, the previous path is released.  That is, it is safe to call
    this routine multiple times prior to the underlying directory being unmapped
    and closed.

Arguments:

    Directory - Supplies a pointer to the directory for which a rename will
        be scheduled.

    NewPath - Supplies the new path to rename.

Return Value:

    S_OK - Directory rename scheduled successfully.

    E_POINTER - Directory or NewPath parameters were NULL.

    E_UNEXPECTED - Internal error.

    E_INVALIDARG - NewPath was invalid.

    PH_E_DIRECTORY_LOCKED - Directory was locked.

    PH_E_PATH_LOCKED - NewPath was locked.

    PH_E_DIRECTORY_NEVER_SET - The directory has never been set.

    PH_E_DIRECTORY_READONLY - The directory is read-only.

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

    if (!ARGUMENT_PRESENT(Directory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(NewPath)) {
        return E_POINTER;
    }

    if (!TryAcquirePerfectHashDirectoryLockExclusive(Directory)) {
        return PH_E_DIRECTORY_LOCKED;
    }

    if (!TryAcquirePerfectHashPathLockExclusive(NewPath)) {
        ReleasePerfectHashDirectoryLockExclusive(Directory);
        return PH_E_PATH_LOCKED;
    }

    if (DirectoryNeverSet(Directory)) {
        Result = PH_E_DIRECTORY_NEVER_SET;
        goto Error;
    }

    if (IsDirectoryReadOnly(Directory)) {
        Result = PH_E_DIRECTORY_READONLY;
        goto Error;
    }


    //
    // Verify the requested new path and current path differ.
    //

    Rtl = Directory->Rtl;
    Equal = Rtl->RtlEqualUnicodeString(&Directory->Path->FullPath,
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
        OriginalAddress = Directory->RenamePath;
        OldAddress = InterlockedCompareExchangePointer(&Directory->RenamePath,
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
    // If the directory has already been closed, do the rename now.
    //

    if (IsDirectoryClosed(Directory)) {
        Result = Directory->Vtbl->DoRename(Directory);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashDirectoryDoRename, Result);
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
    ReleasePerfectHashDirectoryLockExclusive(Directory);

    return Result;
}

PERFECT_HASH_DIRECTORY_DO_RENAME PerfectHashDirectoryDoRename;

_Use_decl_annotations_
HRESULT
PerfectHashDirectoryDoRename(
    PPERFECT_HASH_DIRECTORY Directory
    )
/*++

Routine Description:

    Renames a closed directory based on a previous rename request via
    ScheduleRename().

Arguments:

    Directory - Supplies a pointer to the directory to be renamed.

Return Value:

    S_OK - Directory renamed successfully.

    E_POINTER - Directory was NULL.

    PH_E_DIRECTORY_NEVER_SET - The directory was never set.

    PH_E_DIRECTORY_NOT_CLOSED - The directory is not closed.

    PH_E_DIRECTORY_NO_RENAME_SCHEDULED - No rename is scheduled.

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

    if (!ARGUMENT_PRESENT(Directory)) {
        return E_POINTER;
    }

    if (DirectoryNeverSet(Directory)) {
        return PH_E_DIRECTORY_NEVER_SET;
    }

    if (!IsDirectoryClosed(Directory)) {
        return PH_E_DIRECTORY_NOT_CLOSED;
    }

    if (!IsDirectoryRenameScheduled(Directory)) {
        return PH_E_DIRECTORY_NO_RENAME_SCHEDULED;
    }

    //
    // Directory->Path should be non-NULL, and point to a 'set' path.
    //

    if (!Directory->Path || !IsPathSet(Directory->Path)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    //
    // Argument validation complete.  Continue with rename.
    //

    MoveFileFlags = MOVEFILE_REPLACE_EXISTING;

    Success = MoveFileExW(Directory->Path->FullPath.Buffer,
                          Directory->RenamePath->FullPath.Buffer,
                          MoveFileFlags);

    if (!Success) {
        SYS_ERROR(MoveDirectoryEx);
        return PH_E_SYSTEM_CALL_FAILED;
    }

    //
    // Directory was renamed successfully.  Update the path variables
    // accordingly.
    //

    OldPath = Directory->Path;
    Directory->Path = Directory->RenamePath;
    Directory->RenamePath = NULL;

    //
    // Release the old path's lock and COM reference count.
    //

    OldPath->Vtbl->Release(OldPath);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
