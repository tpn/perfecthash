/*++

Copyright (c) 2018-2023. Trent Nelson <trent@trent.me>

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
    // Create Rtl, Allocator and GuardedList components.
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


    Result = Directory->Vtbl->CreateInstance(Directory,
                                             NULL,
                                             &IID_PERFECT_HASH_GUARDED_LIST,
                                             &Directory->FilesList);

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

    RELEASE(Directory->FilesList);
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

    DirectoryOpenFlags - Optionally supplies a pointer to directory open flags
        that can be used to customize open behavior.

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

    VALIDATE_FLAGS(DirectoryOpen, DIRECTORY_OPEN, ULong);

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

    if (!IsValidHandle(Directory->DirectoryHandle)) {
        Directory->DirectoryHandle = NULL;
        SYS_ERROR(CreateFileW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
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

    VALIDATE_FLAGS(DirectoryCreate, DIRECTORY_CREATE, ULong);

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
                                             OPEN_EXISTING,
                                             FlagsAndAttributes,
                                             NULL);

    if (!IsValidHandle(Directory->DirectoryHandle)) {
        Directory->DirectoryHandle = NULL;
        SYS_ERROR(CreateFileW);
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
        if (IsDirectoryReadOnly(Directory)) {
            PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        }
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
    this routine multiple times prior to the underlying directory being closed.

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

    RELEASE(OldPath);

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
    PRTL Rtl;
    BOOL Success;
    BOOLEAN Equal;
    ULONG LastError;
    ULONG MoveFileFlags;
    HRESULT Result = S_OK;
    PGUARDED_LIST List;
    PLIST_ENTRY Entry;
    PPERFECT_HASH_PATH OldPath;
    PPERFECT_HASH_FILE File = NULL;
    PPERFECT_HASH_PATH Path;
    PCUNICODE_STRING NewDirectory;
    BOOLEAN FileInvariantCheckFailed = FALSE;

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
    // Argument validation complete.  Continue with rename.  Unlike files,
    // we can't just specify "replace existing" when moving directories.  As
    // it's highly likely there will already be a directory in place with our
    // desired final name (i.e. the path currently captured in RenamePath), we
    // will move the existing one into an "old" subdirectory (that we may need
    // to create first), appending the directory's creation time whilst we're
    // at it.
    //

    MoveFileFlags = 0;

    Success = MoveFileExW(Directory->Path->FullPath.Buffer,
                          Directory->RenamePath->FullPath.Buffer,
                          MoveFileFlags);

    LastError = GetLastError();

    if (Success) {

        NOTHING;

    } else if (LastError != ERROR_ALREADY_EXISTS) {

        SYS_ERROR(MoveDirectoryEx);
        return PH_E_SYSTEM_CALL_FAILED;

    } else {

        //
        // N.B. This should probably be moved into its own subroutine.
        //

        HANDLE Handle;
        ULONG Value;
        ULONG ShareMode;
        ULONG DesiredAccess;
        FILETIME CreationFileTime = { 0 };
        SYSTEMTIME CreationSysTime = { 0 };
        const UNICODE_STRING Old = RTL_CONSTANT_STRING(L"old");
        WCHAR Buffer[] = L"_YYYY-MM-DD_hhmmss.SSS";
        UNICODE_STRING Suffix;
        PUNICODE_STRING String = &Suffix;
        PUNICODE_STRING Dir;
        PPERFECT_HASH_PATH ExistingPath;
        PCUNICODE_STRING BaseNameSuffix;

        DesiredAccess = GENERIC_READ | GENERIC_WRITE;
        ShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

        Handle = CreateFileW(Directory->RenamePath->FullPath.Buffer,
                             DesiredAccess,
                             ShareMode,
                             NULL,
                             OPEN_EXISTING,
                             FILE_FLAG_BACKUP_SEMANTICS,
                             NULL);

        if (!IsValidHandle(Handle)) {
            SYS_ERROR(CreateFileW);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }

        //
        // Get the creation file time, then convert into a system time.
        //

        Success = GetFileTime(Handle, &CreationFileTime, NULL, NULL);
        if (!Success) {
            SYS_ERROR(GetFileTime);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }

        if (!CloseHandle(Handle)) {
            SYS_ERROR(CloseHandle);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }

        Handle = NULL;

        Success = FileTimeToSystemTime(&CreationFileTime, &CreationSysTime);
        if (!Success) {
            SYS_ERROR(FileTimeToSystemTime);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }

#define APPEND_TIME_FIELD(Field, Digits, Trailer)                        \
    Value = CreationSysTime.Field;                                       \
    if (!AppendIntegerToUnicodeString(String, Value, Digits, Trailer)) { \
        Result = PH_E_STRING_BUFFER_OVERFLOW;                            \
        PH_ERROR(PerfectHashDirectoryDoRename_AppendTimeField, Result);  \
        goto Error;                                                      \
    }

        //
        // Wire up the length such that we skip the leading underscore.
        //

        String->Length = sizeof(L'_');
        String->MaximumLength = sizeof(Buffer);
        String->Buffer = (PWSTR)Buffer;

        APPEND_TIME_FIELD(wYear,          4, L'-');
        APPEND_TIME_FIELD(wMonth,         2, L'-');
        APPEND_TIME_FIELD(wDay,           2, L'-');
        APPEND_TIME_FIELD(wHour,          2,    0);
        APPEND_TIME_FIELD(wMinute,        2,    0);
        APPEND_TIME_FIELD(wSecond,        2, L'-');
        APPEND_TIME_FIELD(wMilliseconds,  3,    0);

        //
        // Create a new path instance.
        //

        Result = Directory->Vtbl->CreateInstance(Directory,
                                                 NULL,
                                                 &IID_PERFECT_HASH_PATH,
                                                 &Path);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashPathCreateInstance, Result);
            return Result;
        }

        //
        // Construct a new path name.
        //

        ExistingPath = Directory->RenamePath;
        NewDirectory = &Directory->RenamePath->Directory;
        BaseNameSuffix = (PCUNICODE_STRING)&Suffix;

        Result = Path->Vtbl->Create(Path,
                                    ExistingPath,
                                    NewDirectory,
                                    &Old,           // DirectorySuffix
                                    NULL,           // NewBaseName
                                    BaseNameSuffix,
                                    NULL,           // NewExtension
                                    NULL,           // NewStreamName
                                    NULL,           // Parts
                                    NULL);          // Reserved

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashDirectoryDoRename_PathCreate, Result);
            RELEASE(Path);
            goto Error;
        }

        //
        // Attempt to create the underlying directory (i.e. the "old" part).
        // Temporarily NULL-terminate the Path->Directory buffer so that we
        // can pass it directly to CreateDirectoryW().
        //

        Dir = &Path->Directory;

        ASSERT(Dir->Buffer[Dir->Length >> 1] == L'\\');
        Dir->Buffer[Dir->Length >> 1] = L'\0';

        Success = CreateDirectoryW(Dir->Buffer, NULL);

        //
        // Restore the slash.
        //

        Dir->Buffer[Dir->Length >> 1] = L'\\';

        if (!Success) {
            LastError = GetLastError();
            if (LastError != ERROR_ALREADY_EXISTS) {
                SYS_ERROR(CreateDirectoryW);
                RELEASE(Path);
                goto Error;
            }
        }

        //
        // We can now try moving the path at our desired location to the new
        // location in the old subdirectory.
        //

        Success = MoveFileExW(Directory->RenamePath->FullPath.Buffer,
                              Path->FullPath.Buffer,
                              MoveFileFlags);

        if (!Success) {
            SYS_ERROR(MoveFileEx);
            RELEASE(Path);
            goto Error;
        }

        //
        // Move was successful.  Attempt the original move again.
        //

        Success = MoveFileExW(Directory->Path->FullPath.Buffer,
                              Directory->RenamePath->FullPath.Buffer,
                              MoveFileFlags);

        if (!Success) {
            SYS_ERROR(MoveFileEx);
            RELEASE(Path);
            goto Error;
        }

        RELEASE(Path);
    }

    //
    // Directory was renamed successfully.  Update the path variables
    // accordingly.
    //

    OldPath = Directory->Path;
    Directory->Path = Directory->RenamePath;
    Directory->RenamePath = NULL;

    //
    // Release the old path, then proceed with processing files that have been
    // added to us, if applicable.
    //

    OldPath->Vtbl->Release(OldPath);

    Rtl = Directory->Rtl;
    List = Directory->FilesList;
    NewDirectory = &Directory->Path->FullPath;

    ASSERT(Directory->ReferenceCount >= 1);

    while (List->Vtbl->RemoveHeadEx(List, &Entry)) {

        //
        // One or more files are associated with this directory.  We need to
        // update their File->Path instances to point to a new path instance
        // that features our new directory name.
        //

        File = CONTAINING_RECORD(Entry, PERFECT_HASH_FILE, ListEntry);

        AcquirePerfectHashFileLockExclusive(File);

        //
        // Invariant checks: file should not be readonly, should be closed,
        // should not have any renames scheduled, and the file's directory
        // path should not match ours.
        //

        if (IsFileReadOnly(File)) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashDirectoryDoRename_FileReadonly, Result);
            goto InvariantFailed;
        }

        if (!IsFileClosed(File)) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashDirectoryDoRename_FileNotClosed, Result);
            goto InvariantFailed;
        }

        if (IsFileRenameScheduled(File)) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashDirectoryDoRename_FileRenameScheduled, Result);
            goto InvariantFailed;
        }

        Equal = Rtl->RtlEqualUnicodeString(&File->Path->Directory,
                                           NewDirectory,
                                           TRUE);

        if (Equal) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PerfectHashDirectoryDoRename_DirsEqual, Result);
            goto InvariantFailed;
        }

        //
        // Invariant checks complete, proceed with creating a new path with
        // an updated directory.
        //

        Result = File->Vtbl->CreateInstance(File,
                                            NULL,
                                            &IID_PERFECT_HASH_PATH,
                                            &Path);

        if (FAILED(Result)) {

            //
            // If we can't create instances anymore, things are pretty dire.
            // Log the error then break out of the loop.
            //

            PH_ERROR(PerfectHashPathCreateInstance, Result);
            ReleasePerfectHashFileLockExclusive(File);
            goto Error;
        }


        Result = Path->Vtbl->Create(Path,
                                    File->Path,     // ExistingPath
                                    NewDirectory,
                                    NULL,           // DirectorySuffix
                                    NULL,           // NewBaseName
                                    NULL,           // BaseNameSuffix
                                    NULL,           // NewExtension
                                    NULL,           // NewStreamName
                                    NULL,           // Parts
                                    NULL);          // Reserved

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashPathCreate, Result);
            goto FinishFile;
        }

        //
        // Path creation was successful, swap the paths over and release the
        // old path.
        //

        OldPath = File->Path;
        File->Path = Path;
        OldPath->Vtbl->Release(OldPath);

        goto FinishFile;

InvariantFailed:

        FileInvariantCheckFailed = TRUE;

        //
        // Intentional follow-on to FinishFile.
        //

FinishFile:

        //
        // Release references.  Note that we need to effectively do two
        // releases against the directory; one to account for the AddRef()
        // performed against Dictionary in PerfectHashDictionaryAddFile(),
        // and one performed against File->ParentDictionary after AddFile()
        // is called in PerfectHashFileCreate().
        //

        Directory->Vtbl->Release(Directory);
        RELEASE(File->ParentDirectory);
        File->Vtbl->Release(File);

        ReleasePerfectHashFileLockExclusive(File);
    }

    if (FileInvariantCheckFailed) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
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

    //
    // Invariant check: the directory's reference count should be 1 here
    // if we did not encounter any failures.
    //

    if (SUCCEEDED(Result) && Directory->ReferenceCount != 1) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    return Result;
}

PERFECT_HASH_DIRECTORY_ADD_FILE PerfectHashDirectoryAddFile;

_Use_decl_annotations_
HRESULT
PerfectHashDirectoryAddFile(
    PPERFECT_HASH_DIRECTORY Directory,
    PPERFECT_HASH_FILE File
    )
/*++

Routine Description:

    Associates a given file instance with the directory.

Arguments:

    Directory - Supplies a pointer to the directory instance for which the
        file will be added.

    File - Supplies a pointer to the file instance to add.

Return Value:

    S_OK - File added to directory successfully.

    E_POINTER - Directory or File parameters were NULL.

    PH_E_DIRECTORY_NOT_SET - The directory is not set.

    PH_E_DIRECTORY_READONLY - The directory is readonly.

    PH_E_DIRECTORY_CLOSED - The directory is closed.

    PH_E_FILE_NOT_OPEN - The file has not yet been opened, or has been closed.

    PH_E_FILE_ALREADY_ADDED_TO_A_DIRECTORY - The file has already been added
        to a directory.

    PH_E_DIRECTORY_RENAME_ALREADY_SCHEDULED - A directory rename has already
        been scheduled.  Files must be added to a directory prior to this.

--*/
{
    HRESULT Result = S_OK;
    PGUARDED_LIST List;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Directory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (File->ParentDirectory) {
        return PH_E_FILE_ALREADY_ADDED_TO_A_DIRECTORY;
    }

    //
    // If the parent directory isn't set; the list entry should indicate empty.
    //

    if (!IsListEmpty(&File->ListEntry)) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashDirectoryAddFile_FileListEntry, Result);
        return Result;
    }

    AcquirePerfectHashDirectoryLockShared(Directory);

    if (!IsDirectorySet(Directory)) {
        Result = PH_E_DIRECTORY_NOT_SET;
        goto Error;
    }

    if (IsDirectoryReadOnly(Directory)) {
        Result = PH_E_DIRECTORY_READONLY;
        goto Error;
    }

    if (IsDirectoryRenameScheduled(Directory)) {
        Result = PH_E_DIRECTORY_RENAME_ALREADY_SCHEDULED;
        goto Error;
    }

    if (IsDirectoryClosed(Directory)) {
        Result = PH_E_DIRECTORY_CLOSED;
        goto Error;
    }

    //
    // Argument validation complete.  Add the file to the files list and
    // increment the reference counts for both the file and the directory.
    //

    List = Directory->FilesList;
    List->Vtbl->InsertTail(List, &File->ListEntry);

    Directory->Vtbl->AddRef(Directory);
    File->Vtbl->AddRef(File);

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    ReleasePerfectHashDirectoryLockShared(Directory);

    return Result;
}

PERFECT_HASH_DIRECTORY_REMOVE_FILE PerfectHashDirectoryRemoveFile;

_Use_decl_annotations_
HRESULT
PerfectHashDirectoryRemoveFile(
    PPERFECT_HASH_DIRECTORY Directory,
    PPERFECT_HASH_FILE File
    )
/*++

Routine Description:

    Removes a file previously added to a directory.

Arguments:

    Directory - Supplies a pointer to the directory instance for which the
        file will be removed.

    File - Supplies a pointer to the file instance to remove.

Return Value:

    S_OK - File added to directory successfully.

    E_POINTER - Directory or File parameters were NULL.

    PH_E_DIRECTORY_NOT_SET - The directory is not set.

    PH_E_DIRECTORY_READONLY - The directory is readonly.

    PH_E_DIRECTORY_ALREADY_CLOSED - The directory has already been closed.

    PH_E_FILE_NOT_OPEN - The file has not yet been opened, or has been closed.

    PH_E_FILE_ADDED_TO_DIFFERENT_DIRECTORY - The file has already been added
        to a different directory.

    PH_E_FILE_NOT_ADDED_TO_DIRECTORY - The file has not been added to a
        directory.

--*/
{
    HRESULT Result = S_OK;
    PGUARDED_LIST List;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Directory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (!File->ParentDirectory) {
        return PH_E_FILE_NOT_ADDED_TO_DIRECTORY;
    }

    if (File->ParentDirectory != Directory) {
        return PH_E_FILE_ADDED_TO_DIFFERENT_DIRECTORY;
    }

    //
    // If the parent directory is set; the list entry shouldn't be empty.
    //

    if (IsListEmpty(&File->ListEntry)) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(PerfectHashDirectoryRemoveFile_FileListEntry, Result);
        return Result;
    }

    AcquirePerfectHashDirectoryLockShared(Directory);

    if (!IsDirectorySet(Directory)) {
        Result = PH_E_DIRECTORY_NOT_SET;
        goto Error;
    }

    if (IsDirectoryReadOnly(Directory)) {
        Result = PH_E_DIRECTORY_READONLY;
        goto Error;
    }

    if (IsDirectoryClosed(Directory)) {
        Result = PH_E_DIRECTORY_CLOSED;
        goto Error;
    }

    //
    // Argument validation complete.  Remove the file from the directory.
    //

    List = Directory->FilesList;
    List->Vtbl->RemoveEntry(List, &File->ListEntry);

    File->Vtbl->Release(File);
    RELEASE(File->ParentDirectory);
    Directory->Vtbl->Release(Directory);

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    ReleasePerfectHashDirectoryLockShared(Directory);

    return Result;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
