/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    ChmOnline01HashJit.c

Abstract:

    Stub implementations for hash JIT dispatch and LLVM JIT plumbing for
    core builds.

--*/

#include "stdafx.h"
#include "ChmOnline01.h"

#if defined(PH_ONLINE_CORE_ONLY) && !defined(PH_USE_LLVMLIB)

PERFECT_HASH_TABLE_COMPILE_HASH_JIT PerfectHashTableCompileHashJit;
PERFECT_HASH_TABLE_HASH_JIT_RUNDOWN PerfectHashTableHashJitRundown;

_Use_decl_annotations_
HRESULT
PerfectHashTableCompileHashJit(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(CompileFlags);

    return PH_E_NOT_IMPLEMENTED;
}

_Use_decl_annotations_
VOID
PerfectHashTableHashJitRundown(
    PPERFECT_HASH_TABLE Table
    )
{
    UNREFERENCED_PARAMETER(Table);
}

#elif defined(PH_USE_LLVMLIB)

_Use_decl_annotations_
HRESULT
PerfectHashTableCompileHashJit(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
    const PERFECT_HASH_LLVM_FUNCTIONS *Functions;
    PERFECT_HASH_TABLE_COMPILE_FLAGS LocalFlags;

    if (!ARGUMENT_PRESENT(CompileFlags)) {
        UNREFERENCED_PARAMETER(Table);
        return PH_E_NOT_IMPLEMENTED;
    }

    if (CompileFlags->JitBackendRawDog) {
        UNREFERENCED_PARAMETER(Table);
        return PH_E_NOT_IMPLEMENTED;
    }

    if (!CompileFlags->JitBackendLlvm) {
        LocalFlags.AsULong = CompileFlags->AsULong;
        LocalFlags.JitBackendLlvm = TRUE;
        CompileFlags = &LocalFlags;
    }

    Functions = GetPerfectHashLLVMFunctions();
    if (!Functions) {
        return PH_E_LLVM_BACKEND_NOT_FOUND;
    }

#if defined(PH_WINDOWS)
    __try {
        return Functions->TableCompileHashJit(Table, CompileFlags);
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        return PH_E_TABLE_COMPILATION_FAILED;
    }
#else
    return Functions->TableCompileHashJit(Table, CompileFlags);
#endif
}

_Use_decl_annotations_
VOID
PerfectHashTableHashJitRundown(
    PPERFECT_HASH_TABLE Table
    )
{
    const PERFECT_HASH_LLVM_FUNCTIONS *Functions;

    if (!ARGUMENT_PRESENT(Table) || !ARGUMENT_PRESENT(Table->HashJit)) {
        return;
    }

    Functions = GetPerfectHashLLVMFunctions();
    if (!Functions) {
        return;
    }

    Functions->TableHashJitRundown(Table);
}

#endif // PH_USE_LLVMLIB

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
