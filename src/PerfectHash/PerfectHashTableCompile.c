/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableCompile.c

Abstract:

    This module implements functionality for compiling perfect hash tables into
    a more optimal format.

--*/

#include "stdafx.h"

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

Arguments:

    Table - Supplies a pointer to the PERFECT_HASH_TABLE interface for which
        the compilation is to be performed.

    TableCompileFlags - Optionally supplies a pointer to a table compile flags
        structure that can be used to customize the compilation behavior.

    CpuArchId - Supplies the CPU architecture for which the perfect hash table
        compilation is to target.  If this differs from the current CPU arch,
        cross-compilation must be supported by the underlying algorith, hash
        function and masking type.

Return Value:

    S_OK - Table compiled successfully.

    The following error codes may also be returned.  Note that this list is not
    guaranteed to be exhaustive; that is, error codes other than the ones listed
    below may also be returned.

    E_POINTER - Table was NULL.

    E_UNEXPECTED - General error.

    E_OUTOFMEMORY - Out of memory.

    PH_E_INVALID_TABLE_COMPILE_FLAGS - Invalid table compile flags provided.

    PH_E_TABLE_LOCKED - The table is locked.

    PH_E_TABLE_NOT_LOADED - The table has not been loaded.

    PH_E_INVALID_CPU_ARCH_ID - Invalid CPU architecture ID.

    PH_E_TABLE_COMPILATION_NOT_AVAILABLE - Table compilation is not available
        for the current configuration, which is composed of: architecture (e.g.
        x64), algorithm ID, hash function and masking type.

    PH_E_TABLE_CROSS_COMPILATION_NOT_AVAILABLE - Table cross-compilation is not
        available for this combination of CPU architecture or

--*/
{
    HRESULT Result = S_OK;
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!IsValidPerfectHashCpuArchId(CpuArchId)) {
        return PH_E_INVALID_CPU_ARCH_ID;
    }

    VALIDATE_FLAGS(TableCompile, TABLE_COMPILE);

    if (!TryAcquirePerfectHashTableLockExclusive(Table)) {
        return PH_E_TABLE_LOCKED;
    }

    if (!Table->Flags.Loaded) {
        ReleasePerfectHashTableLockExclusive(Table);
        return PH_E_TABLE_NOT_LOADED;
    }

    //
    // Argument validation complete.
    //

    //
    // WIP.
    //

    Result = PH_E_WORK_IN_PROGRESS;
    goto End;

#if 0
Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

#endif

End:

    ReleasePerfectHashTableLockExclusive(Table);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
