/*++

Copyright (c) 2025 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashOnlineCompile.c

Abstract:

    This module implements the online compilation routine for producing a
    JIT-compiled table representation.

--*/

#include "stdafx.h"

PERFECT_HASH_ONLINE_COMPILE_TABLE PerfectHashOnlineCompileTable;

_Use_decl_annotations_
HRESULT
PerfectHashOnlineCompileTable(
    PPERFECT_HASH_ONLINE Online,
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlagsPointer
    )
/*++

Routine Description:

    Compiles a table into an online, in-memory JIT representation.

Arguments:

    Online - Supplies a pointer to the PERFECT_HASH_ONLINE interface.

    Table - Supplies a pointer to the table to compile.

    CompileFlags - Optionally supplies a pointer to a table compile flags
        structure that can be used to customize the compilation behavior.

Return Value:

    S_OK on success, an appropriate error code on failure.

--*/
{
#ifdef PH_ONLINE_CORE_ONLY
    UNREFERENCED_PARAMETER(Online);
    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(CompileFlagsPointer);
    return PH_E_NOT_IMPLEMENTED;
#else
    HRESULT Result;
    PERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Online)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (ARGUMENT_PRESENT(CompileFlagsPointer)) {
        Result = IsValidTableCompileFlags(CompileFlagsPointer);
        if (FAILED(Result)) {
            return PH_E_INVALID_TABLE_COMPILE_FLAGS;
        }
        CompileFlags.AsULong = CompileFlagsPointer->AsULong;
    } else {
        CompileFlags.AsULong = 0;
    }

    //
    // Ensure the JIT path is active.
    //

    CompileFlags.Jit = TRUE;

    //
    // Dispatch to the table's Compile() routine.
    //

    return Table->Vtbl->Compile(Table,
                                &CompileFlags,
                                PerfectHashGetCurrentCpuArch());
#endif
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
