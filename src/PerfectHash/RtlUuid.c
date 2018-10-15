/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    RtlUuid.c

Abstract:

    This file implements the create and free UUID string routines.

--*/

#include "stdafx.h"
#include <rpc.h>

RTL_CREATE_UUID_STRING RtlCreateUuidString;

_Use_decl_annotations_
HRESULT
RtlCreateUuidString(
    PRTL Rtl,
    PSTRING String
    )
{
    USHORT Index;
    USHORT Count;
    PCHAR Buffer;
    CHAR Char;
    CHAR Upper;
    GUID Guid;
    HRESULT Result;
    RPC_CSTR GuidCStr = NULL;

    UNREFERENCED_PARAMETER(Rtl);

    Result = UuidCreate(&Guid);
    if (FAILED(Result)) {
        SYS_ERROR(UuidCreate);
        goto End;
    }

    Result = UuidToStringA(&Guid, &GuidCStr);
    if (FAILED(Result)) {
        SYS_ERROR(UuidToStringA);
        goto End;
    }

    String->Buffer = (PCHAR)GuidCStr;
    String->Length = (USHORT)strlen(String->Buffer);
    String->MaximumLength = String->Length + 1;
    ASSERT(String->Length == UUID_STRING_LENGTH);
    ASSERT(String->Buffer[String->Length] == '\0');

    //
    // Convert the UUID into uppercase.
    //

    Buffer = (PCHAR)GuidCStr;
    Count = UUID_STRING_LENGTH;

    for (Index = 0; Index < Count; Index++, Buffer++) {
        Upper = Char = *Buffer;

        if (Char >= 'a' && Char <= 'f') {
            Upper -= 0x20;
            *Buffer = Upper;
        }

    }

End:
    return Result;
}


RTL_FREE_UUID_STRING RtlFreeUuidString;

_Use_decl_annotations_
HRESULT
RtlFreeUuidString(
    PRTL Rtl,
    PSTRING String
    )
{
    HRESULT Result = S_OK;

    if (!IsValidUuidString(String)) {
        Result = PH_E_INVALID_UUID_STRING;
        goto End;
    }

    Result = RpcStringFreeA((RPC_CSTR *)&String->Buffer);

    if (FAILED(Result)) {

        //
        // N.B. We use PH_ERROR() here instead of SYS_ERROR() as the error
        //      code information is in Result and not GetLastError() as this
        //      is an RPC/COM-type call (not a Win32 call).
        //

        PH_ERROR(RpcStringFreeA, Result);
    }

    ZeroStructPointer(String);

End:

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
