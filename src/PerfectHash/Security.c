/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Security.c

Abstract:

    This module implements security-related routines for the perfect hash
    library.  Currently, a routine is provided to construct SECURITY_ATTRIBUTES
    for passing to functions like CreateEventW().

--*/

#include "stdafx.h"

CREATE_EXCLUSIVE_DACL_FOR_CURRENT_USER CreateExclusiveDaclForCurrentUser;

_Use_decl_annotations_
HRESULT
CreateExclusiveDaclForCurrentUser(
    PRTL Rtl,
    PSECURITY_ATTRIBUTES SecurityAttributes,
    PSECURITY_DESCRIPTOR SecurityDescriptor,
    PEXPLICIT_ACCESS_W ExplicitAccess,
    PACL *AclPointer
    )
{
    BOOL Success;
    PACL Acl = NULL;
    ULONG LastError;
    HRESULT Result = S_OK;
    PTOKEN_USER Token;
    HANDLE TokenHandle = NULL;
    PCHAR Buffer = NULL;
    ULONG BufferSize = 0;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(SecurityAttributes)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(SecurityDescriptor)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(ExplicitAccess)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(AclPointer)) {
        return E_POINTER;
    }

    //
    // Open a handle to the current process.
    //

    Success = OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &TokenHandle);
    if (!Success) {
        SYS_ERROR(OpenProcessToken);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Determine how big the buffer size needs to be to capture a user token.
    //

    Success = GetTokenInformation(TokenHandle,
                                  TokenUser,
                                  NULL,
                                  0,
                                  &BufferSize);

    LastError = GetLastError();
    if (!(!Success && LastError == ERROR_INSUFFICIENT_BUFFER)) {
        SYS_ERROR(GetTokenInformation);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Allocation sufficient space.
    //

    Buffer = HeapAlloc(GetProcessHeap(),
                       HEAP_ZERO_MEMORY,
                       BufferSize);
    if (!Buffer) {
        SYS_ERROR(HeapAlloc);
        Result = E_OUTOFMEMORY;
        goto End;
    }

    //
    // Call GetTokenInformation() again with the newly-allocated buffer.
    //

    Token = (PTOKEN_USER)Buffer;

    Success = GetTokenInformation(TokenHandle,
                                  TokenUser,
                                  Token,
                                  BufferSize,
                                  &BufferSize);

    if (!Success) {
        SYS_ERROR(GetTokenInformation);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Success = IsValidSid(Token->User.Sid);
    if (!Success) {
        SYS_ERROR(IsValidSid);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Initialize the explicit access descriptor with the user SID.
    //

    ZeroStructPointer(ExplicitAccess);

    ExplicitAccess->grfAccessPermissions = (
        GENERIC_READ    |
        GENERIC_WRITE   |
        SYNCHRONIZE
    );
    ExplicitAccess->grfAccessMode = SET_ACCESS;
    ExplicitAccess->grfInheritance = NO_INHERITANCE;
    ExplicitAccess->Trustee.TrusteeForm = TRUSTEE_IS_SID;
    ExplicitAccess->Trustee.TrusteeType = TRUSTEE_IS_USER;
    ExplicitAccess->Trustee.ptstrName = (PWSTR)Token->User.Sid;

    //
    // Create an ACL from the explicit access descriptor.
    //

    Result = SetEntriesInAclW(1, ExplicitAccess, NULL, &Acl);

    if (FAILED(Result)) {
        SYS_ERROR(SetEntriesInAclW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto End;
    }

    //
    // Prepare the security attributes structure.
    //

    ZeroStructPointer(SecurityAttributes);
    SecurityAttributes->nLength = sizeof(*SecurityAttributes);
    SecurityAttributes->lpSecurityDescriptor = NULL;
    SecurityAttributes->bInheritHandle = FALSE;

    Success = InitializeSecurityDescriptor(SecurityDescriptor,
                                           SECURITY_DESCRIPTOR_REVISION);
    if (!Success) {
        SYS_ERROR(InitializeSecurityDescriptor);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Set the DACL.
    //

    Success = SetSecurityDescriptorDacl(SecurityDescriptor,
                                        TRUE,
                                        Acl,
                                        FALSE);

    if (!Success) {
        SYS_ERROR(SetSecurityDescriptorDacl);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    SecurityAttributes->lpSecurityDescriptor = SecurityDescriptor;

    //
    // We're done, update the caller's pointer and jump to end.
    //

    *AclPointer = Acl;
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (TokenHandle) {
        if (!CloseHandle(TokenHandle)) {
            SYS_ERROR(CloseHandle);
        }
        TokenHandle = NULL;
    }

    if (Buffer) {
        if (!HeapFree(GetProcessHeap(), 0, Buffer)) {
            SYS_ERROR(HeapFree);
        }
        Buffer = NULL;
    }

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
