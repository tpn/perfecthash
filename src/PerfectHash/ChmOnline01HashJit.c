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

#if defined(PH_ONLINE_CORE_ONLY)

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

extern LOAD_SYMBOLS LoadSymbols;

#if defined(PH_WINDOWS)
#define PERFECT_HASH_LLVM_DLL_NAME "PerfectHashLLVM.dll"
#elif defined(PH_MAC)
#define PERFECT_HASH_LLVM_DLL_NAME "@rpath/libPerfectHashLLVM.dylib"
#else
#define PERFECT_HASH_LLVM_DLL_NAME "libPerfectHashLLVM.so"
#endif

typedef struct _PERFECT_HASH_LLVMLIB_HASH_FUNCTIONS {
    PPERFECT_HASH_TABLE_COMPILE_HASH_JIT TableCompileHashJit;
    PPERFECT_HASH_TABLE_HASH_JIT_RUNDOWN TableHashJitRundown;
} PERFECT_HASH_LLVMLIB_HASH_FUNCTIONS;

PERFECT_HASH_TABLE_COMPILE_HASH_JIT PerfectHashTableCompileHashJit;
PERFECT_HASH_TABLE_HASH_JIT_RUNDOWN PerfectHashTableHashJitRundown;

static INIT_ONCE PerfectHashLLVMInitOnce = INIT_ONCE_STATIC_INIT;
static PERFECT_HASH_LLVMLIB_HASH_FUNCTIONS PerfectHashLLVMFunctions;
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
        "PerfectHashTableCompileHashJit",
        "PerfectHashTableHashJitRundown",
    };

    FailedBitmap.SizeOfBitMap = ARRAYSIZE(Names) + 1;
    FailedBitmap.Buffer = (PULONG)&FailedBitmapBuffer;

    Module = LoadLibraryA(PERFECT_HASH_LLVM_DLL_NAME);
    if (!IsValidHandle(Module)) {
        fprintf(stderr,
                "Failed to load %s for LLVM hash JIT support.\n",
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
PerfectHashTableCompileHashJit(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
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

    {
        HRESULT Result = EnsurePerfectHashLLVMLoaded();
        if (FAILED(Result)) {
            return Result;
        }
    }

    if (!ARGUMENT_PRESENT(PerfectHashLLVMFunctions.TableCompileHashJit)) {
        return PH_E_LLVM_BACKEND_NOT_FOUND;
    }

#if defined(PH_WINDOWS)
    __try {
        return PerfectHashLLVMFunctions.TableCompileHashJit(Table,
                                                            CompileFlags);
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        return PH_E_TABLE_COMPILATION_FAILED;
    }
#else
    return PerfectHashLLVMFunctions.TableCompileHashJit(Table, CompileFlags);
#endif
}

_Use_decl_annotations_
VOID
PerfectHashTableHashJitRundown(
    PPERFECT_HASH_TABLE Table
    )
{
    if (!ARGUMENT_PRESENT(Table) || !ARGUMENT_PRESENT(Table->HashJit)) {
        return;
    }

    if (FAILED(EnsurePerfectHashLLVMLoaded())) {
        return;
    }

    if (!ARGUMENT_PRESENT(PerfectHashLLVMFunctions.TableHashJitRundown)) {
        return;
    }

    PerfectHashLLVMFunctions.TableHashJitRundown(Table);
}

#endif // PH_USE_LLVMLIB

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
