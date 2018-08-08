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

    SYS_ERROR(AdjustTokenPrivileges);

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

_Use_decl_annotations_
HRESULT
EnablePrivilege(
    PRTL Rtl,
    PWSTR PrivilegeName
    )
{
    return SetPrivilege(Rtl, PrivilegeName, TRUE);
}

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

_Use_decl_annotations_
HRESULT
EnableManageVolumePrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_MANAGE_VOLUME_NAME);
}

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

_Use_decl_annotations_
HRESULT
EnableLockMemoryPrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_LOCK_MEMORY_NAME);
}

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

_Use_decl_annotations_
HRESULT
EnableDebugPrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_DEBUG_NAME);
}

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

_Use_decl_annotations_
HRESULT
EnableSystemProfilePrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_SYSTEM_PROFILE_NAME);
}

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

_Use_decl_annotations_
HRESULT
EnableProfileSingleProcessPrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_PROF_SINGLE_PROCESS_NAME);
}

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

_Use_decl_annotations_
HRESULT
EnableIncreaseWorkingSetPrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_INC_WORKING_SET_NAME);
}

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

_Use_decl_annotations_
HRESULT
EnableCreateSymbolicLinkPrivilege(
    PRTL Rtl
    )
{
    return EnablePrivilege(Rtl, SE_CREATE_SYMBOLIC_LINK_NAME);
}

_Use_decl_annotations_
HRESULT
DisableCreateSymbolicLinkPrivilege(
    PRTL Rtl
    )
{
    return DisablePrivilege(Rtl, SE_CREATE_SYMBOLIC_LINK_NAME);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
