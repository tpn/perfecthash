/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Security.h

Abstract:

    This is the private header for the security module of the perfect hash
    library.  It contains function and typedefs for functionality related to
    this module.

--*/

#include "stdafx.h"

typedef
_Success_(return >= 0)
HRESULT
(NTAPI CREATE_EXCLUSIVE_DACL_FOR_CURRENT_USER)(
    _In_ PRTL Rtl,
    _Out_ PSECURITY_ATTRIBUTES SecurityAttributes,
    _Out_ PSECURITY_DESCRIPTOR SecurityDescriptor,
    _Out_ PEXPLICIT_ACCESS_W ExplicitAccess,
    _Out_ PACL *AclPointer
    );
typedef CREATE_EXCLUSIVE_DACL_FOR_CURRENT_USER
      *PCREATE_EXCLUSIVE_DACL_FOR_CURRENT_USER;

extern CREATE_EXCLUSIVE_DACL_FOR_CURRENT_USER CreateExclusiveDaclForCurrentUser;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
