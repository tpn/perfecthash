/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashNames.h

Abstract:

    This is the private header file for generic functionality that converts
    enum IDs to string representations and vice versa.

--*/

#pragma once

#include "stdafx.h"

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PERFECT_HASH_LOOKUP_NAME_FOR_ID)(
    _In_ PRTL Rtl,
    _In_ PERFECT_HASH_ENUM_ID EnumId,
    _In_ ULONG Id,
    _Out_ PCUNICODE_STRING *NamePointer
    );
typedef PERFECT_HASH_LOOKUP_NAME_FOR_ID *PPERFECT_HASH_LOOKUP_NAME_FOR_ID;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PERFECT_HASH_LOOKUP_ID_FOR_NAME)(
    _In_ PRTL Rtl,
    _In_ PERFECT_HASH_ENUM_ID EnumId,
    _In_ PCUNICODE_STRING Name,
    _Out_ PULONG Id
    );
typedef PERFECT_HASH_LOOKUP_ID_FOR_NAME *PPERFECT_HASH_LOOKUP_ID_FOR_NAME;

#ifndef __INTELLISENSE__
extern PERFECT_HASH_LOOKUP_NAME_FOR_ID PerfectHashLookupNameForId;
extern PERFECT_HASH_LOOKUP_ID_FOR_NAME PerfectHashLookupIdForName;
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
