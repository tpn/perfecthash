/*++

Copyright (c) 2025 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableCompileCompat.c

Abstract:

    This module provides the non-Windows implementation of the table compile
    routine, which currently supports JIT-only compilation.

--*/

#include "stdafx.h"

#ifndef PH_WINDOWS

PERFECT_HASH_TABLE_COMPILE PerfectHashTableCompile;

_Use_decl_annotations_
HRESULT
PerfectHashTableCompile(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlagsPointer,
    PERFECT_HASH_CPU_ARCH_ID CpuArchId
    )
/*++

Routine Description:

    Compiles a loaded perfect hash table into an optimized format.

    Non-Windows platforms currently support JIT compilation only.

Arguments:

    Table - Supplies a pointer to the PERFECT_HASH_TABLE interface for which
        the compilation is to be performed.

    TableCompileFlags - Optionally supplies a pointer to a table compile flags
        structure that can be used to customize the compilation behavior.

    CpuArchId - Supplies the CPU architecture for which the perfect hash table
        compilation is to target.

Return Value:

    S_OK - Table compiled successfully.

    E_POINTER - Table was NULL.

    PH_E_INVALID_CPU_ARCH_ID - Invalid CPU architecture ID.

    PH_E_INVALID_TABLE_COMPILE_FLAGS - Invalid table compile flags provided.

    PH_E_TABLE_LOCKED - The table is locked.

    PH_E_TABLE_NOT_CREATED - The table has not been created.

    PH_E_NOT_IMPLEMENTED - Non-JIT compilation is not available.

--*/
{
    HRESULT Result = S_OK;
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags;

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!IsValidPerfectHashCpuArchId(CpuArchId)) {
        return PH_E_INVALID_CPU_ARCH_ID;
    }

    VALIDATE_FLAGS(TableCompile, TABLE_COMPILE, ULong);

    if (!TryAcquirePerfectHashTableLockExclusive(Table)) {
        return PH_E_TABLE_LOCKED;
    }

    if (!Table->Flags.Created) {
        ReleasePerfectHashTableLockExclusive(Table);
        return PH_E_TABLE_NOT_CREATED;
    }

    if (TableCompileFlags.Jit) {
        if (!TableCompileFlags.JitBackendLlvm) {
            TableCompileFlags.JitBackendLlvm = TRUE;
        }
        Result = PerfectHashTableCompileJit(Table, &TableCompileFlags);
        ReleasePerfectHashTableLockExclusive(Table);
        return Result;
    }

    ReleasePerfectHashTableLockExclusive(Table);
    return PH_E_NOT_IMPLEMENTED;
}

#endif // PH_WINDOWS

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
