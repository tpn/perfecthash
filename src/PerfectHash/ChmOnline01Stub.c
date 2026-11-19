/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    ChmOnline01Stub.c

Abstract:

    Stub implementations for core-only online builds and LLVM JIT dispatch
    plumbing for core builds.

--*/

#include "stdafx.h"

#if defined(PH_ONLINE_CORE_ONLY) && !defined(PH_USE_LLVMLIB)

PERFECT_HASH_TABLE_COMPILE_JIT PerfectHashTableCompileJit;
PERFECT_HASH_TABLE_JIT_RUNDOWN PerfectHashTableJitRundown;

_Use_decl_annotations_
HRESULT
PerfectHashTableCompileJit(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
#if defined(PH_HAS_RAWDOG_JIT)
    if (ARGUMENT_PRESENT(CompileFlags) &&
        CompileFlags->JitBackendRawDog) {
        return PerfectHashTableCompileJitRawDog(Table, CompileFlags);
    }
#endif

    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(CompileFlags);

    return PH_E_NOT_IMPLEMENTED;
}

_Use_decl_annotations_
VOID
PerfectHashTableJitRundown(
    PPERFECT_HASH_TABLE Table
    )
{
#if defined(PH_HAS_RAWDOG_JIT)
    if (ARGUMENT_PRESENT(Table) &&
        ARGUMENT_PRESENT(Table->Jit) &&
        Table->Jit->Flags.BackendRawDog) {
        PerfectHashTableJitRundownRawDog(Table);
        return;
    }
#endif

    UNREFERENCED_PARAMETER(Table);
}

#elif defined(PH_USE_LLVMLIB)

_Use_decl_annotations_
HRESULT
PerfectHashTableCompileJit(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
    const PERFECT_HASH_LLVM_FUNCTIONS *Functions;

    if (!ARGUMENT_PRESENT(CompileFlags)) {
        UNREFERENCED_PARAMETER(Table);
        return PH_E_NOT_IMPLEMENTED;
    }

#if defined(PH_HAS_RAWDOG_JIT)
    if (CompileFlags->JitBackendRawDog) {
        return PerfectHashTableCompileJitRawDog(Table, CompileFlags);
    }
#endif

    if (!CompileFlags->JitBackendLlvm) {
        UNREFERENCED_PARAMETER(Table);
        return PH_E_NOT_IMPLEMENTED;
    }

    Functions = GetPerfectHashLLVMFunctions();
    if (!Functions) {
        return PH_E_LLVM_BACKEND_NOT_FOUND;
    }

#if defined(PH_WINDOWS)
    __try {
        return Functions->TableCompileJit(Table, CompileFlags);
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        return PH_E_TABLE_COMPILATION_FAILED;
    }
#else
    return Functions->TableCompileJit(Table, CompileFlags);
#endif
}

_Use_decl_annotations_
VOID
PerfectHashTableJitRundown(
    PPERFECT_HASH_TABLE Table
    )
{
    const PERFECT_HASH_LLVM_FUNCTIONS *Functions;

    if (!ARGUMENT_PRESENT(Table) || !ARGUMENT_PRESENT(Table->Jit)) {
        return;
    }

#if defined(PH_HAS_RAWDOG_JIT)
    if (Table->Jit->Flags.BackendRawDog) {
        PerfectHashTableJitRundownRawDog(Table);
        return;
    }
#endif

    Functions = GetPerfectHashLLVMFunctions();
    if (!Functions) {
        return;
    }

    Functions->TableJitRundown(Table);
}

#endif // PH_USE_LLVMLIB

typedef struct _INDEX_JIT_BINDING {
    PPERFECT_HASH_TABLE_JIT Jit;
    PPERFECT_HASH_TABLE_INDEX Index;
    PPERFECT_HASH_TABLE_QUERY_INTERFACE QueryInterface;
} INDEX_JIT_BINDING;

static
VOID
ExchangeIndexJitBinding(
    PPERFECT_HASH_TABLE Table,
    INDEX_JIT_BINDING *Binding
    )
{
    INDEX_JIT_BINDING Active = {
        Table->Jit, Table->Vtbl->Index, Table->Vtbl->QueryInterface
    };
    Table->Jit = Binding->Jit;
    Table->Vtbl->Index = Binding->Index;
    Table->Vtbl->QueryInterface = Binding->QueryInterface;
    Table->Flags.JitEnabled = (Binding->Jit && Binding->Jit->Flags.Valid);
    *Binding = Active;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableCompileJitLocked(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
    HRESULT Result;
    INDEX_JIT_BINDING Saved = { 0 };
    PPERFECT_HASH_TABLE_JIT Jit;

    // The caller holds the exclusive table lock. Keep the old backend and
    // its executable pages alive while compiling against the original vtable.
    Saved.Index = Table->Jit ? Table->Jit->OriginalIndex : Table->Vtbl->Index;
    Saved.QueryInterface = Table->Jit ? Table->Jit->OriginalQueryInterface :
                                       Table->Vtbl->QueryInterface;
    ExchangeIndexJitBinding(Table, &Saved);
    Result = PerfectHashTableCompileJit(Table, CompileFlags);
    Jit = Table->Jit;
    if (SUCCEEDED(Result) && CompileFlags &&
        CompileFlags->JitRequireRequestedWidth &&
        (!Jit ||
         ((CompileFlags->JitIndex32x2 || CompileFlags->JitVectorIndex32x2) &&
          !Jit->Flags.Index32x2Compiled) ||
         ((CompileFlags->JitIndex32x4 || CompileFlags->JitVectorIndex32x4) &&
          !Jit->Flags.Index32x4Compiled) ||
         ((CompileFlags->JitIndex32x8 || CompileFlags->JitVectorIndex32x8) &&
          !Jit->Flags.Index32x8Compiled) ||
         (CompileFlags->JitIndex32x16 && !Jit->Flags.Index32x16Compiled))) {
        Result = PH_E_NOT_IMPLEMENTED;
    }
    if (FAILED(Result)) {
        PerfectHashTableJitRundown(Table);
        ExchangeIndexJitBinding(Table, &Saved);
    } else if (Saved.Jit) {
        // Rundown dispatches through the owner of the replaced JIT.
        ExchangeIndexJitBinding(Table, &Saved);
        PerfectHashTableJitRundown(Table);
        ExchangeIndexJitBinding(Table, &Saved);
    }
    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
