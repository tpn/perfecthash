/*++

Copyright (c) 2016 Trent Nelson <trent@trent.me>

Module Name:

    Privilege.c

Abstract:

    This module implements functionality related to system security components.
    Routines are provided for setting or revoking generic token privileges,
    and enabling and disabling the volume management privilege.

--*/

#include "stdafx.h"

SET_PRIVILEGE SetPrivilege;

_Use_decl_annotations_
HRESULT
SetPrivilege(
    PRTL Rtl,
    PWSTR PrivilegeName,
    BOOLEAN Enable
    )
/*++

Routine Description:

    This routine enables or disables a given privilege name for the current
    process token.

Arguments:

    Rtl - Supplies a pointer to an RTL instance.

    PrivilegeName - Supplies a pointer to a NULL-terminated wide string that
        represents the privilege name.

    Enable - Supplies a boolean value indicating that the privilege should be
        enabled when set to TRUE, disabled when set to FALSE.

Return Value:

    TRUE on success, FALSE on failure.

--*/
{
    BOOL Success;
    HRESULT Result;
    DWORD LastError;
    DWORD DesiredAccess;
    DWORD TokenAttributes;
    HANDLE ProcessHandle;
    HANDLE TokenHandle;
    TOKEN_PRIVILEGES TokenPrivileges;

    UNREFERENCED_PARAMETER(Rtl);

    //
    // Initialize local variables.
    //

    ProcessHandle = GetCurrentProcess();
    DesiredAccess = TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY;

    if (Enable) {
        TokenAttributes = SE_PRIVILEGE_ENABLED;
    } else {
        TokenAttributes = 0;
    }

    //
    // Obtain a token handle for the current process.
    //

    Success = OpenProcessToken(ProcessHandle, DesiredAccess, &TokenHandle);

    if (!Success) {
        SYS_ERROR(OpenProcessToken);
        return E_FAIL;
    }

    //
    // Lookup the privilege value for the name passed in by the caller.
    //

    Success = LookupPrivilegeValueW(NULL,
                                    PrivilegeName,
                                    &TokenPrivileges.Privileges[0].Luid);

    if (!Success) {
        SYS_ERROR(LookupPrivilegeValueW);
        goto Error;
    }

    //
    // Fill in the remaining token privilege fields.
    //

    TokenPrivileges.PrivilegeCount = 1;
    TokenPrivileges.Privileges[0].Attributes = TokenAttributes;

    //
    // Attempt to adjust the token privileges.
    //

    Success = AdjustTokenPrivileges(TokenHandle,
                                    FALSE,
                                    &TokenPrivileges,
                                    0,
                                    NULL,
                                    0);

    LastError = GetLastError();

    if (Success && LastError == ERROR_SUCCESS) {
        Result = S_OK;
        goto End;
    }

    //
    // If the error indicates anything other than the call failing because the
    // user does not have the required privileges (ERROR_NOT_ALL_ASSIGNED), then
    // report it.
    //

    if (LastError != ERROR_NOT_ALL_ASSIGNED) {
        SYS_ERROR(AdjustTokenPrivileges);
    }

    //
    // Intentional follow-on to Error.
    //

Error:

    Result = E_FAIL;

    //
    // Intentional follow-on to End.
    //

End:

    CloseHandle(TokenHandle);

    return Result;
}

ENABLE_PRIVILEGE EnablePrivilege;

_Use_decl_annotations_
HRESULT
EnablePrivilege(
    PRTL Rtl,
    PWSTR PrivilegeName
    )
{
    return SetPrivilege(Rtl, PrivilegeName, TRUE);
}

DISABLE_PRIVILEGE DisablePrivilege;

_Use_decl_annotations_
HRESULT
DisablePrivilege(
    PRTL Rtl,
    PWSTR PrivilegeName
    )
{
    return SetPrivilege(Rtl, PrivilegeName, FALSE);
}

////////////////////////////////////////////////////////////////////////////////
// SE_MANAGE_VOLUME/SeManageVolume
////////////////////////////////////////////////////////////////////////////////

ENABLE_MANAGE_VOLUME_PRIVILEGE EnableManageVolumePrivilege;

_Use_decl_annotations_
HRESULT
EnableManageVolumePrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_MANAGE_VOLUME_NAME);
}

DISABLE_MANAGE_VOLUME_PRIVILEGE DisableManageVolumePrivilege;

_Use_decl_annotations_
HRESULT
DisableManageVolumePrivilege(
    PRTL Rtl
    )
{
    return DisablePrivilege(Rtl, SE_MANAGE_VOLUME_NAME);
}

////////////////////////////////////////////////////////////////////////////////
// SE_LOCK_MEMORY/SeLockMemory
////////////////////////////////////////////////////////////////////////////////

ENABLE_LOCK_MEMORY_PRIVILEGE EnableLockMemoryPrivilege;

_Use_decl_annotations_
HRESULT
EnableLockMemoryPrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_LOCK_MEMORY_NAME);
}

DISABLE_LOCK_MEMORY_PRIVILEGE DisableLockMemoryPrivilege;

_Use_decl_annotations_
HRESULT
DisableLockMemoryPrivilege(
    PRTL Rtl
    )
{
    return DisablePrivilege(Rtl, SE_LOCK_MEMORY_NAME);
}

////////////////////////////////////////////////////////////////////////////////
// SE_DEBUG/SeDebug
////////////////////////////////////////////////////////////////////////////////

ENABLE_DEBUG_PRIVILEGE EnableDebugPrivilege;

_Use_decl_annotations_
HRESULT
EnableDebugPrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_DEBUG_NAME);
}

DISABLE_DEBUG_PRIVILEGE DisableDebugPrivilege;

_Use_decl_annotations_
HRESULT
DisableDebugPrivilege(
    PRTL Rtl
    )
{
    return DisablePrivilege(Rtl, SE_DEBUG_NAME);
}

////////////////////////////////////////////////////////////////////////////////
// SE_SYSTEM_PROFILE/SeSystemProfile
////////////////////////////////////////////////////////////////////////////////


ENABLE_SYSTEM_PROFILE_PRIVILEGE EnableSystemProfilePrivilege;

_Use_decl_annotations_
HRESULT
EnableSystemProfilePrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_SYSTEM_PROFILE_NAME);
}

DISABLE_SYSTEM_PROFILE_PRIVILEGE DisableSystemProfilePrivilege;

_Use_decl_annotations_
HRESULT
DisableSystemProfilePrivilege(
    PRTL Rtl
    )
{
    return DisablePrivilege(Rtl, SE_SYSTEM_PROFILE_NAME);
}

////////////////////////////////////////////////////////////////////////////////
// PROF_SINGLE_PROCESS/SeProfileSingleProcess
////////////////////////////////////////////////////////////////////////////////

ENABLE_PROFILE_SINGLE_PROCESS_PRIVILEGE EnableProfileSingleProcessPrivilege;

_Use_decl_annotations_
HRESULT
EnableProfileSingleProcessPrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_PROF_SINGLE_PROCESS_NAME);
}

DISABLE_PROFILE_SINGLE_PROCESS_PRIVILEGE DisableProfileSingleProcessPrivilege;

_Use_decl_annotations_
HRESULT
DisableProfileSingleProcessPrivilege(
    PRTL Rtl
    )
{
    return DisablePrivilege(Rtl, SE_PROF_SINGLE_PROCESS_NAME);
}

////////////////////////////////////////////////////////////////////////////////
// SE_INC_WORKING_SET/SeIncreaseWorkingSet
////////////////////////////////////////////////////////////////////////////////

ENABLE_INCREASE_WORKING_SET_PRIVILEGE EnableIncreaseWorkingSetPrivilege;

_Use_decl_annotations_
HRESULT
EnableIncreaseWorkingSetPrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_INC_WORKING_SET_NAME);
}

DISABLE_INCREASE_WORKING_SET_PRIVILEGE DisableIncreaseWorkingSetPrivilege;

_Use_decl_annotations_
HRESULT
DisableIncreaseWorkingSetPrivilege(
    PRTL Rtl
    )
{
    return DisablePrivilege(Rtl, SE_INC_WORKING_SET_NAME);
}

////////////////////////////////////////////////////////////////////////////////
// SE_CREATE_SYMBOLIC_LINK/SeCreateSymbolicLink
////////////////////////////////////////////////////////////////////////////////

ENABLE_CREATE_SYMBOLIC_LINK_PRIVILEGE EnableCreateSymbolicLinkPrivilege;

_Use_decl_annotations_
HRESULT
EnableCreateSymbolicLinkPrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_CREATE_SYMBOLIC_LINK_NAME);
}

DISABLE_CREATE_SYMBOLIC_LINK_PRIVILEGE DisableCreateSymbolicLinkPrivilege;

_Use_decl_annotations_
HRESULT
DisableCreateSymbolicLinkPrivilege(
    PRTL Rtl
    )
{
    return DisablePrivilege(Rtl, SE_CREATE_SYMBOLIC_LINK_NAME);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
