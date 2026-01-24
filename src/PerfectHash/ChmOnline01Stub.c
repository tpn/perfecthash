/*++

Copyright (c) 2025 Trent Nelson <trent@trent.me>

Module Name:

    ChmOnline01Stub.c

Abstract:

    Stub implementations for core-only online builds and LLVM JIT dispatch
    plumbing for core builds.

--*/

#include "stdafx.h"

#if defined(PH_ONLINE_CORE_ONLY)

PERFECT_HASH_TABLE_COMPILE_JIT PerfectHashTableCompileJit;
PERFECT_HASH_TABLE_JIT_RUNDOWN PerfectHashTableJitRundown;

_Use_decl_annotations_
HRESULT
PerfectHashTableCompileJit(
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
PerfectHashTableJitRundown(
    PPERFECT_HASH_TABLE Table
    )
{
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
        ZeroStruct(PerfectHashLLVMFunctions);
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

    PerfectHashLLVMLoaded = LoadPerfectHashLLVMFunctions();

    if (ARGUMENT_PRESENT(Context)) {
        *Context = (PVOID)(ULONG_PTR)(PerfectHashLLVMLoaded != FALSE);
    }

    return TRUE;
}

static
HRESULT
EnsurePerfectHashLLVMLoaded(
    VOID
    )
{
    PVOID Context = NULL;

    InitOnceExecuteOnce(&PerfectHashLLVMInitOnce,
                        InitPerfectHashLLVM,
                        NULL,
                        &Context);

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

    return PerfectHashLLVMFunctions.TableCompileJit(Table, CompileFlags);
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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
