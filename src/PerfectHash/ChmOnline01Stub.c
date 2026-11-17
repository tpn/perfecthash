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

extern LOAD_SYMBOLS LoadSymbols;

#if defined(PH_WINDOWS)
#define PERFECT_HASH_LLVM_DLL_NAME "PerfectHashLLVM.dll"
#elif defined(PH_MAC)
#define PERFECT_HASH_LLVM_DLL_NAME "@rpath/libPerfectHashLLVM.dylib"
#else
#define PERFECT_HASH_LLVM_DLL_NAME "libPerfectHashLLVM.so"
#endif

typedef struct _PERFECT_HASH_LLVMLIB_FUNCTIONS {
    PPERFECT_HASH_TABLE_COMPILE_JIT TableCompileJit;
    PPERFECT_HASH_TABLE_JIT_RUNDOWN TableJitRundown;
} PERFECT_HASH_LLVMLIB_FUNCTIONS;

PERFECT_HASH_TABLE_COMPILE_JIT PerfectHashTableCompileJit;
PERFECT_HASH_TABLE_JIT_RUNDOWN PerfectHashTableJitRundown;

static INIT_ONCE PerfectHashLLVMInitOnce = INIT_ONCE_STATIC_INIT;
static PERFECT_HASH_LLVMLIB_FUNCTIONS PerfectHashLLVMFunctions;
static HMODULE PerfectHashLLVMModule;
static volatile BOOLEAN PerfectHashLLVMLoaded;
static volatile HRESULT PerfectHashLLVMLoadResult = PH_E_LLVM_BACKEND_NOT_FOUND;

static
BOOLEAN
LoadPerfectHashLLVMFunctions(
    VOID
    )
{
    BOOL Success;
    HMODULE Module;
    ULONG NumberOfResolvedSymbols = 0;
    RTL_BITMAP FailedBitmap;
    ULONG FailedBitmapBuffer[1] = { 0 };

    CONST PCSZ Names[] = {
        "PerfectHashTableCompileJit",
        "PerfectHashTableJitRundown",
    };

    FailedBitmap.SizeOfBitMap = ARRAYSIZE(Names) + 1;
    FailedBitmap.Buffer = (PULONG)&FailedBitmapBuffer;

    Module = LoadLibraryA(PERFECT_HASH_LLVM_DLL_NAME);
    if (!IsValidHandle(Module)) {
        fprintf(stderr,
                "Failed to load %s for LLVM JIT support.\n",
                PERFECT_HASH_LLVM_DLL_NAME);
        PerfectHashLLVMLoadResult = PH_E_LLVM_BACKEND_NOT_FOUND;
        return FALSE;
    }

    Success = LoadSymbols(
        Names,
        ARRAYSIZE(Names),
        (PULONG_PTR)&PerfectHashLLVMFunctions,
        sizeof(PerfectHashLLVMFunctions) / sizeof(ULONG_PTR),
        Module,
        &FailedBitmap,
        &NumberOfResolvedSymbols
    );

    if (!Success || NumberOfResolvedSymbols != ARRAYSIZE(Names)) {
        ZeroStructInline(PerfectHashLLVMFunctions);
        FreeLibrary(Module);
        PerfectHashLLVMLoadResult = PH_E_LLVM_BACKEND_NOT_FOUND;
        return FALSE;
    }

    PerfectHashLLVMModule = Module;
    PerfectHashLLVMLoadResult = S_OK;
    return TRUE;
}

static
BOOL
CALLBACK
InitPerfectHashLLVM(
    _Inout_ PINIT_ONCE InitOnce,
    _Inout_opt_ PVOID Parameter,
    _Outptr_opt_result_maybenull_ PVOID *Context
    )
{
    UNREFERENCED_PARAMETER(InitOnce);
    UNREFERENCED_PARAMETER(Parameter);

    if (ARGUMENT_PRESENT(Context)) {
        *Context = NULL;
    }

    PerfectHashLLVMLoaded = LoadPerfectHashLLVMFunctions();

    return TRUE;
}

static
HRESULT
EnsurePerfectHashLLVMLoaded(
    VOID
    )
{
    InitOnceExecuteOnce(&PerfectHashLLVMInitOnce,
                        InitPerfectHashLLVM,
                        NULL,
                        NULL);

    if (PerfectHashLLVMLoaded) {
        return S_OK;
    }

    return PerfectHashLLVMLoadResult;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableCompileJit(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
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

    {
        HRESULT Result = EnsurePerfectHashLLVMLoaded();
        if (FAILED(Result)) {
            return Result;
        }
    }

    if (!ARGUMENT_PRESENT(PerfectHashLLVMFunctions.TableCompileJit)) {
        return PH_E_LLVM_BACKEND_NOT_FOUND;
    }

#if defined(PH_HAS_RAWDOG_JIT)
    // Only the host module can release a RawDog code allocation. The LLVM
    // module treats an existing execution engine as an LLVM MCJIT instance.
    if (Table && Table->Jit && Table->Jit->Flags.BackendRawDog) {
        PerfectHashTableJitRundownRawDog(Table);
    }
#endif

#if defined(PH_WINDOWS)
    __try {
        return PerfectHashLLVMFunctions.TableCompileJit(Table, CompileFlags);
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        return PH_E_TABLE_COMPILATION_FAILED;
    }
#else
    return PerfectHashLLVMFunctions.TableCompileJit(Table, CompileFlags);
#endif
}

_Use_decl_annotations_
VOID
PerfectHashTableJitRundown(
    PPERFECT_HASH_TABLE Table
    )
{
    if (!ARGUMENT_PRESENT(Table) || !ARGUMENT_PRESENT(Table->Jit)) {
        return;
    }

#if defined(PH_HAS_RAWDOG_JIT)
    if (Table->Jit->Flags.BackendRawDog) {
        PerfectHashTableJitRundownRawDog(Table);
        return;
    }
#endif

    if (FAILED(EnsurePerfectHashLLVMLoaded())) {
        return;
    }

    if (!ARGUMENT_PRESENT(PerfectHashLLVMFunctions.TableJitRundown)) {
        return;
    }

    PerfectHashLLVMFunctions.TableJitRundown(Table);
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
    PERFECT_HASH_TABLE_COMPILE_FLAGS Flags;
    INDEX_JIT_BINDING Saved = { 0 };
    PPERFECT_HASH_TABLE_JIT Jit;

    if (!CompileFlags || !CompileFlags->JitRequireRequestedWidth) {
        return PerfectHashTableCompileJit(Table, CompileFlags);
    }

    // The caller holds the exclusive table lock. Keep the old backend and
    // its executable pages alive while compiling against the original vtable.
    // The DLL sees only its established backend flags.
    Flags = *CompileFlags;
    Flags.JitRequireRequestedWidth = FALSE;
    Saved.Index = Table->Jit ? Table->Jit->OriginalIndex : Table->Vtbl->Index;
    Saved.QueryInterface = Table->Jit ? Table->Jit->OriginalQueryInterface :
                                       Table->Vtbl->QueryInterface;
    ExchangeIndexJitBinding(Table, &Saved);
    Result = PerfectHashTableCompileJit(Table, &Flags);
    Jit = Table->Jit;
    if (SUCCEEDED(Result) &&
        (!Jit ||
         ((Flags.JitIndex32x2 || Flags.JitVectorIndex32x2) &&
          !Jit->Flags.Index32x2Compiled) ||
         ((Flags.JitIndex32x4 || Flags.JitVectorIndex32x4) &&
          !Jit->Flags.Index32x4Compiled) ||
         ((Flags.JitIndex32x8 || Flags.JitVectorIndex32x8) &&
          !Jit->Flags.Index32x8Compiled) ||
         (Flags.JitIndex32x16 && !Jit->Flags.Index32x16Compiled))) {
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
