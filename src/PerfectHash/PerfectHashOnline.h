/*++

Copyright (c) 2025 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashOnline.h

Abstract:

    This is the private header file for the PERFECT_HASH_ONLINE component of
    the perfect hash table library.

--*/

#pragma once

#include "stdafx.h"

DEFINE_UNUSED_STATE(PERFECT_HASH_ONLINE);
DEFINE_UNUSED_FLAGS(PERFECT_HASH_ONLINE);

typedef struct _Struct_size_bytes_(SizeOfStruct) _PERFECT_HASH_ONLINE {

    COMMON_COMPONENT_HEADER(PERFECT_HASH_ONLINE);

    //
    // Backing interface.
    //

    PERFECT_HASH_ONLINE_VTBL Interface;

} PERFECT_HASH_ONLINE;
typedef PERFECT_HASH_ONLINE *PPERFECT_HASH_ONLINE;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_ONLINE_INITIALIZE)(
    _In_ PPERFECT_HASH_ONLINE Online
    );
typedef PERFECT_HASH_ONLINE_INITIALIZE *PPERFECT_HASH_ONLINE_INITIALIZE;

typedef
VOID
(NTAPI PERFECT_HASH_ONLINE_RUNDOWN)(
    _In_ _Post_invalid_ PPERFECT_HASH_ONLINE Online
    );
typedef PERFECT_HASH_ONLINE_RUNDOWN *PPERFECT_HASH_ONLINE_RUNDOWN;

#ifndef __INTELLISENSE__
extern PERFECT_HASH_ONLINE_INITIALIZE PerfectHashOnlineInitialize;
extern PERFECT_HASH_ONLINE_RUNDOWN PerfectHashOnlineRundown;
extern PERFECT_HASH_ONLINE_CREATE_TABLE_FROM_KEYS
    PerfectHashOnlineCreateTableFromKeys;
extern PERFECT_HASH_ONLINE_COMPILE_TABLE PerfectHashOnlineCompileTable;
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
