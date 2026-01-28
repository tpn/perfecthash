/*++

Copyright (c) 2025 Trent Nelson <trent@trent.me>

Module Name:

    ChmOnline01.h

Abstract:

    This is the private header file for the CHM online (CPU) JIT routines.

--*/

#pragma once

#include "stdafx.h"

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PERFECT_HASH_TABLE_COMPILE_JIT)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_opt_ PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    );
typedef PERFECT_HASH_TABLE_COMPILE_JIT *PPERFECT_HASH_TABLE_COMPILE_JIT;

typedef
VOID
(NTAPI PERFECT_HASH_TABLE_JIT_RUNDOWN)(
    _Inout_ PPERFECT_HASH_TABLE Table
    );
typedef PERFECT_HASH_TABLE_JIT_RUNDOWN *PPERFECT_HASH_TABLE_JIT_RUNDOWN;

#ifndef __INTELLISENSE__
extern PERFECT_HASH_TABLE_COMPILE_JIT PerfectHashTableCompileJit;
extern PERFECT_HASH_TABLE_JIT_RUNDOWN PerfectHashTableJitRundown;
#if defined(PH_HAS_RAWDOG_JIT)
extern PERFECT_HASH_TABLE_COMPILE_JIT PerfectHashTableCompileJitRawDog;
extern PERFECT_HASH_TABLE_JIT_RUNDOWN PerfectHashTableJitRundownRawDog;
#endif
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
