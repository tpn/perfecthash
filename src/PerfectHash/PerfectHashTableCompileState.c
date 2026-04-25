/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableCompileState.c

Abstract:

    This module validates table state before compile dispatch.

--*/

#include "stdafx.h"

/*++

Routine Description:

    Validates that a table is in a state that can enter compile dispatch.  A
    created table may compile through the existing create-time paths.  A loaded
    GraphImpl4 table is admitted only when both Jit and JitBackendLlvm are set.
    Loaded GraphImpl4 tables using RawDog, an unspecified backend, or
    JitBackendLlvm without Jit return PH_E_NOT_IMPLEMENTED.  Loaded
    non-GraphImpl4 tables also return PH_E_NOT_IMPLEMENTED.  Tables that are
    neither created nor loaded return PH_E_TABLE_NOT_CREATED.

--*/

_Use_decl_annotations_
HRESULT
PerfectHashTableValidateCompileState(
    PPERFECT_HASH_TABLE Table,
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags
    )
{
    if (Table->Flags.Created) {
        return S_OK;
    }

    //
    // Loaded GraphImpl4 tables are admitted only when callers explicitly set
    // both Jit and JitBackendLlvm.  Backend-specific flags do not imply Jit.
    // The IgnoreAssembly flag describes the non-JIT generated-source path, so
    // it is ignored for loaded-table JIT.  LLVM is currently the only backend
    // admitted for loaded GraphImpl4 tables; RawDog and unspecified backends
    // are rejected here before reaching backend-specific dispatch.
    //

    if (Table->Flags.Loaded && Table->GraphImpl == 4) {
        if (TableCompileFlags.Jit && TableCompileFlags.JitBackendLlvm) {
            return S_OK;
        }

        return PH_E_NOT_IMPLEMENTED;
    }

    if (Table->Flags.Loaded) {
        return PH_E_NOT_IMPLEMENTED;
    }

    return PH_E_TABLE_NOT_CREATED;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
