/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashOnlineJit.c

Abstract:

    Minimal C API wrapper for creating and querying 32-bit perfect hash
    tables with online mode and either RawDog JIT or LLVM JIT backends.

--*/

#include "stdafx.h"
#include <PerfectHash/PerfectHashOnlineJit.h>

#include <stdint.h>
#include <stdlib.h>

extern DLL_GET_CLASS_OBJECT PerfectHashDllGetClassObject;

struct PH_ONLINE_JIT_CONTEXT {
    PICLASSFACTORY ClassFactory;
    PPERFECT_HASH_ONLINE Online;
};

struct PH_ONLINE_JIT_TABLE {
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT_INTERFACE JitInterface;
};

static
HRESULT
PhReleaseOnlineJitInterface(
    _Inout_ PH_ONLINE_JIT_TABLE *Table
    )
{
    if (!ARGUMENT_PRESENT(Table)) {
        return E_INVALIDARG;
    }

    if (Table->JitInterface) {
        Table->JitInterface->Vtbl->Release(Table->JitInterface);
        Table->JitInterface = NULL;
    }

    return S_OK;
}

static
HRESULT
PhEnsureOnlineJitInterface(
    _Inout_ PH_ONLINE_JIT_TABLE *Table
    )
{
    HRESULT Result;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table)) {
        return E_INVALIDARG;
    }

    if (ARGUMENT_PRESENT(Table->JitInterface)) {
        return S_OK;
    }

    Result = Table->Table->Vtbl->QueryInterface(
        Table->Table,
#ifdef __cplusplus
        IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#else
        &IID_PERFECT_HASH_TABLE_JIT_INTERFACE,
#endif
        (void **)&Table->JitInterface
    );

    return Result;
}

static
HRESULT
PhMapOnlineJitHashFunction(
    _In_ PH_ONLINE_JIT_HASH_FUNCTION HashFunction,
    _Out_ PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId
    )
{
    if (!ARGUMENT_PRESENT(HashFunctionId)) {
        return E_POINTER;
    }

    switch (HashFunction) {
        case PhOnlineJitHashMultiplyShiftR:
            *HashFunctionId = PerfectHashHashMultiplyShiftRFunctionId;
            break;
        case PhOnlineJitHashMultiplyShiftLR:
            *HashFunctionId = PerfectHashHashMultiplyShiftLRFunctionId;
            break;
        case PhOnlineJitHashMultiplyShiftRMultiply:
            *HashFunctionId = PerfectHashHashMultiplyShiftRMultiplyFunctionId;
            break;
        case PhOnlineJitHashMultiplyShiftR2:
            *HashFunctionId = PerfectHashHashMultiplyShiftR2FunctionId;
            break;
        case PhOnlineJitHashMultiplyShiftRX:
            *HashFunctionId = PerfectHashHashMultiplyShiftRXFunctionId;
            break;
        case PhOnlineJitHashMulshrolate1RX:
            *HashFunctionId = PerfectHashHashMulshrolate1RXFunctionId;
            break;
        case PhOnlineJitHashMulshrolate2RX:
            *HashFunctionId = PerfectHashHashMulshrolate2RXFunctionId;
            break;
        case PhOnlineJitHashMulshrolate3RX:
            *HashFunctionId = PerfectHashHashMulshrolate3RXFunctionId;
            break;
        case PhOnlineJitHashMulshrolate4RX:
            *HashFunctionId = PerfectHashHashMulshrolate4RXFunctionId;
            break;
        default:
            return E_INVALIDARG;
    }

    return S_OK;
}

static
HRESULT
PhApplyOnlineJitVectorWidth(
    _In_ ULONG VectorWidth,
    _Inout_ PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
    if (!ARGUMENT_PRESENT(CompileFlags)) {
        return E_POINTER;
    }

    switch (VectorWidth) {
        case 0:
        case 1:
            break;
        case 2:
            CompileFlags->JitIndex32x2 = TRUE;
            break;
        case 4:
            CompileFlags->JitIndex32x4 = TRUE;
            break;
        case 8:
            CompileFlags->JitIndex32x8 = TRUE;
            break;
        case 16:
            CompileFlags->JitIndex32x16 = TRUE;
            break;
        default:
            return E_INVALIDARG;
    }

    return S_OK;
}

static
HRESULT
PhCompileOnlineJitBackend(
    _In_ PH_ONLINE_JIT_CONTEXT *Context,
    _In_ PH_ONLINE_JIT_TABLE *Table,
    _In_ PH_ONLINE_JIT_BACKEND Backend,
    _In_ ULONG VectorWidth,
    _In_ PH_ONLINE_JIT_MAX_ISA JitMaxIsa
    )
{
    HRESULT Result;
    PERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags = {0};
    PTABLE_INFO_ON_DISK TableInfo;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table) ||
        !ARGUMENT_PRESENT(Table->Table->TableInfoOnDisk)) {
        return E_INVALIDARG;
    }

    TableInfo = Table->Table->TableInfoOnDisk;

    CompileFlags.Jit = TRUE;

    switch (Backend) {
        case PhOnlineJitBackendRawDogJit:
            CompileFlags.JitBackendRawDog = TRUE;
            Result = PhApplyOnlineJitVectorWidth(VectorWidth, &CompileFlags);
            if (FAILED(Result)) {
                return Result;
            }
            break;

        case PhOnlineJitBackendLlvmJit:
            CompileFlags.JitBackendLlvm = TRUE;
            Result = PhApplyOnlineJitVectorWidth(VectorWidth,
                                                 &CompileFlags);
            if (FAILED(Result)) {
                return Result;
            }
            break;

        case PhOnlineJitBackendAuto:
            return E_INVALIDARG;

        default:
            return E_INVALIDARG;
    }

    CompileFlags.JitMaxIsa = (ULONG)JitMaxIsa;

    return Context->Online->Vtbl->CompileTable(Context->Online,
                                               Table->Table,
                                               &CompileFlags);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitOpen(
    PH_ONLINE_JIT_CONTEXT **ContextPointer
    )
{
    HRESULT Result;
    PH_ONLINE_JIT_CONTEXT *Context = NULL;
    PICLASSFACTORY ClassFactory = NULL;
    PPERFECT_HASH_ONLINE Online = NULL;

    if (!ARGUMENT_PRESENT(ContextPointer)) {
        return E_POINTER;
    }

    *ContextPointer = NULL;

    Context = (PH_ONLINE_JIT_CONTEXT *)calloc(1, sizeof(*Context));
    if (!Context) {
        return E_OUTOFMEMORY;
    }

    Result = PerfectHashDllGetClassObject(&CLSID_PERFECT_HASH,
                                          &IID_PERFECT_HASH_ICLASSFACTORY,
                                          (PVOID *)&ClassFactory);
    if (FAILED(Result)) {
        goto Error;
    }

    Result = ClassFactory->Vtbl->CreateInstance(
        ClassFactory,
        NULL,
        &IID_PERFECT_HASH_ONLINE,
        (void **)&Online
    );
    if (FAILED(Result)) {
        goto Error;
    }

    Context->ClassFactory = ClassFactory;
    Context->Online = Online;

    *ContextPointer = Context;
    return S_OK;

Error:
    if (Online) {
        Online->Vtbl->Release(Online);
        Online = NULL;
    }

    if (ClassFactory) {
        ClassFactory->Vtbl->Release(ClassFactory);
        ClassFactory = NULL;
    }

    free(Context);
    Context = NULL;

    return Result;
}

PH_ONLINE_JIT_API
void
PhOnlineJitClose(
    PH_ONLINE_JIT_CONTEXT *Context
    )
{
    if (!Context) {
        return;
    }

    if (Context->Online) {
        Context->Online->Vtbl->Release(Context->Online);
        Context->Online = NULL;
    }

    if (Context->ClassFactory) {
        Context->ClassFactory->Vtbl->Release(Context->ClassFactory);
        Context->ClassFactory = NULL;
    }

    free(Context);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCreateTable32(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_HASH_FUNCTION HashFunction,
    const uint32_t *Keys,
    uint64_t NumberOfKeys,
    PH_ONLINE_JIT_TABLE **TablePointer
    )
{
    HRESULT Result;
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags = {0};
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags = {0};
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;
    PPERFECT_HASH_TABLE Table = NULL;
    PH_ONLINE_JIT_TABLE *Wrapper = NULL;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(Context->Online) ||
        !ARGUMENT_PRESENT(Keys) ||
        NumberOfKeys == 0 ||
        !ARGUMENT_PRESENT(TablePointer)) {
        return E_INVALIDARG;
    }

    *TablePointer = NULL;

    Result = PhMapOnlineJitHashFunction(HashFunction, &HashFunctionId);
    if (FAILED(Result)) {
        return Result;
    }

    KeysLoadFlags.SortKeys = TRUE;
    KeysLoadFlags.KeysAreSorted = FALSE;

    TableCreateFlags.NoFileIo = TRUE;
    TableCreateFlags.Quiet = TRUE;
    TableCreateFlags.DoNotTryUseHash16Impl = TRUE;

    Result = Context->Online->Vtbl->CreateTableFromKeys(
        Context->Online,
        PerfectHashChm01AlgorithmId,
        HashFunctionId,
        PerfectHashAndMaskFunctionId,
        sizeof(ULONG),
        (ULONGLONG)NumberOfKeys,
        (PVOID)Keys,
        &KeysLoadFlags,
        &TableCreateFlags,
        NULL,
        &Table
    );
    if (FAILED(Result)) {
        return Result;
    }

    Wrapper = (PH_ONLINE_JIT_TABLE *)calloc(1, sizeof(*Wrapper));
    if (!Wrapper) {
        Table->Vtbl->Release(Table);
        return E_OUTOFMEMORY;
    }

    Wrapper->Table = Table;
    *TablePointer = Wrapper;

    return S_OK;
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCreateTable64(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_HASH_FUNCTION HashFunction,
    const uint64_t *Keys,
    uint64_t NumberOfKeys,
    PH_ONLINE_JIT_TABLE **TablePointer
    )
{
    HRESULT Result;
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags = {0};
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags = {0};
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;
    PPERFECT_HASH_TABLE Table = NULL;
    PH_ONLINE_JIT_TABLE *Wrapper = NULL;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(Context->Online) ||
        !ARGUMENT_PRESENT(Keys) ||
        NumberOfKeys == 0 ||
        !ARGUMENT_PRESENT(TablePointer)) {
        return E_INVALIDARG;
    }

    *TablePointer = NULL;

    Result = PhMapOnlineJitHashFunction(HashFunction, &HashFunctionId);
    if (FAILED(Result)) {
        return Result;
    }

    KeysLoadFlags.SortKeys = TRUE;
    KeysLoadFlags.KeysAreSorted = FALSE;

    TableCreateFlags.NoFileIo = TRUE;
    TableCreateFlags.Quiet = TRUE;
    TableCreateFlags.DoNotTryUseHash16Impl = TRUE;

    Result = Context->Online->Vtbl->CreateTableFromKeys(
        Context->Online,
        PerfectHashChm01AlgorithmId,
        HashFunctionId,
        PerfectHashAndMaskFunctionId,
        sizeof(ULONGLONG),
        (ULONGLONG)NumberOfKeys,
        (PVOID)Keys,
        &KeysLoadFlags,
        &TableCreateFlags,
        NULL,
        &Table
    );
    if (FAILED(Result)) {
        return Result;
    }
    if (!ARGUMENT_PRESENT(Table)) {
        return E_UNEXPECTED;
    }

    if (!ARGUMENT_PRESENT(Table->TableInfoOnDisk) ||
        Table->TableInfoOnDisk->OriginalKeySizeInBytes <= sizeof(uint32_t) ||
        Table->TableInfoOnDisk->KeySizeInBytes > sizeof(uint32_t)) {
        Table->Vtbl->Release(Table);
        return PH_E_NOT_IMPLEMENTED;
    }

    Wrapper = (PH_ONLINE_JIT_TABLE *)calloc(1, sizeof(*Wrapper));
    if (!Wrapper) {
        Table->Vtbl->Release(Table);
        return E_OUTOFMEMORY;
    }

    Wrapper->Table = Table;
    *TablePointer = Wrapper;

    return S_OK;
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCompileTable(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_TABLE *Table,
    PH_ONLINE_JIT_BACKEND Backend,
    uint32_t VectorWidth,
    PH_ONLINE_JIT_MAX_ISA JitMaxIsa
    )
{
    return PhOnlineJitCompileTableEx(Context,
                                     Table,
                                     Backend,
                                     VectorWidth,
                                     JitMaxIsa,
                                     0,
                                     NULL,
                                     NULL);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitCompileTableEx(
    PH_ONLINE_JIT_CONTEXT *Context,
    PH_ONLINE_JIT_TABLE *Table,
    PH_ONLINE_JIT_BACKEND Backend,
    uint32_t VectorWidth,
    PH_ONLINE_JIT_MAX_ISA JitMaxIsa,
    uint32_t Flags,
    PH_ONLINE_JIT_BACKEND *EffectiveBackend,
    uint32_t *EffectiveVectorWidth
    )
{
    HRESULT Result;
    HRESULT LastResult;
    ULONG CandidateWidths[4] = {0};
    ULONG CandidateCount = 0;
    ULONG Index;
    BOOLEAN StrictVectorWidth;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(Context->Online) ||
        !ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table)) {
        return E_INVALIDARG;
    }

    //
    // Force lazy re-acquisition on subsequent vector index requests after
    // every compile path.
    //

    PhReleaseOnlineJitInterface(Table);

    if (!IsValidPerfectHashJitMaxIsaId((PERFECT_HASH_JIT_MAX_ISA_ID)JitMaxIsa)) {
        return E_INVALIDARG;
    }

    if (Backend != PhOnlineJitBackendAuto &&
        Backend != PhOnlineJitBackendRawDogJit &&
        Backend != PhOnlineJitBackendLlvmJit) {
        return E_INVALIDARG;
    }

    StrictVectorWidth = ((Flags & PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH) != 0);

    if (ARGUMENT_PRESENT(EffectiveBackend)) {
        *EffectiveBackend = Backend;
    }
    if (ARGUMENT_PRESENT(EffectiveVectorWidth)) {
        *EffectiveVectorWidth = VectorWidth;
    }

    if (Backend == PhOnlineJitBackendLlvmJit) {
        Result = PhCompileOnlineJitBackend(Context,
                                           Table,
                                           Backend,
                                           VectorWidth,
                                           JitMaxIsa);
        if (SUCCEEDED(Result)) {
            if (ARGUMENT_PRESENT(EffectiveBackend)) {
                *EffectiveBackend = PhOnlineJitBackendLlvmJit;
            }
            if (ARGUMENT_PRESENT(EffectiveVectorWidth)) {
                *EffectiveVectorWidth = VectorWidth;
            }
        }
        return Result;
    }

    switch (VectorWidth) {
        case 0:
        case 1:
            CandidateWidths[CandidateCount++] = VectorWidth;
            break;
        case 2:
            CandidateWidths[CandidateCount++] = 2;
            if (!StrictVectorWidth) {
                CandidateWidths[CandidateCount++] = 1;
            }
            break;
        case 4:
            CandidateWidths[CandidateCount++] = 4;
            if (!StrictVectorWidth) {
                CandidateWidths[CandidateCount++] = 1;
            }
            break;
        case 8:
            CandidateWidths[CandidateCount++] = 8;
            if (!StrictVectorWidth) {
                CandidateWidths[CandidateCount++] = 4;
                CandidateWidths[CandidateCount++] = 1;
            }
            break;
        case 16:
            CandidateWidths[CandidateCount++] = 16;
            if (!StrictVectorWidth) {
                CandidateWidths[CandidateCount++] = 8;
                CandidateWidths[CandidateCount++] = 4;
                CandidateWidths[CandidateCount++] = 1;
            }
            break;
        default:
            return E_INVALIDARG;
    }

    LastResult = PH_E_NOT_IMPLEMENTED;

    for (Index = 0; Index < CandidateCount; Index++) {
        Result = PhCompileOnlineJitBackend(Context,
                                           Table,
                                           PhOnlineJitBackendRawDogJit,
                                           CandidateWidths[Index],
                                           JitMaxIsa);
        if (SUCCEEDED(Result)) {
            if (ARGUMENT_PRESENT(EffectiveBackend)) {
                *EffectiveBackend = PhOnlineJitBackendRawDogJit;
            }
            if (ARGUMENT_PRESENT(EffectiveVectorWidth)) {
                *EffectiveVectorWidth = CandidateWidths[Index];
            }
            return S_OK;
        }

        if (Result != PH_E_NOT_IMPLEMENTED &&
            Result != PH_E_INVARIANT_CHECK_FAILED) {
            if (Backend == PhOnlineJitBackendRawDogJit) {
                return Result;
            }
            LastResult = Result;
            break;
        }

        LastResult = Result;
    }

    if (Backend == PhOnlineJitBackendRawDogJit) {
        return LastResult;
    }

    Result = PhCompileOnlineJitBackend(Context,
                                       Table,
                                       PhOnlineJitBackendLlvmJit,
                                       VectorWidth,
                                       JitMaxIsa);
    if (SUCCEEDED(Result)) {
        if (ARGUMENT_PRESENT(EffectiveBackend)) {
            *EffectiveBackend = PhOnlineJitBackendLlvmJit;
        }
        if (ARGUMENT_PRESENT(EffectiveVectorWidth)) {
            *EffectiveVectorWidth = VectorWidth;
        }
        return S_OK;
    }

    if (Result != PH_E_LLVM_BACKEND_NOT_FOUND &&
        Result != PH_E_NOT_IMPLEMENTED) {
        return Result;
    }

    return LastResult;
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32(
    PH_ONLINE_JIT_TABLE *Table,
    uint32_t Key,
    uint32_t *Index
    )
{
    PTABLE_INFO_ON_DISK TableInfo;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table) ||
        !ARGUMENT_PRESENT(Index)) {
        return E_INVALIDARG;
    }

    TableInfo = Table->Table->TableInfoOnDisk;
    if (ARGUMENT_PRESENT(TableInfo) &&
        TableInfo->OriginalKeySizeInBytes > sizeof(uint32_t)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    return Table->Table->Vtbl->Index(Table->Table, (ULONG)Key, (PULONG)Index);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex64(
    PH_ONLINE_JIT_TABLE *Table,
    uint64_t Key,
    uint32_t *Index
    )
{
    PTABLE_INFO_ON_DISK TableInfo;
    ULONG DownsizedKey;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Index)) {
        return E_INVALIDARG;
    }

    if (!ARGUMENT_PRESENT(Table->Table) ||
        !ARGUMENT_PRESENT(Table->Table->TableInfoOnDisk)) {
        return E_INVALIDARG;
    }

    TableInfo = Table->Table->TableInfoOnDisk;
    if (TableInfo->OriginalKeySizeInBytes <= sizeof(uint32_t) ||
        TableInfo->KeySizeInBytes > sizeof(uint32_t)) {
        return PH_E_NOT_IMPLEMENTED;
    }
    DownsizedKey = (ULONG)ExtractBits64((ULONGLONG)Key, Table->Table->DownsizeBitmap);
    if (Table->Table->GraphImpl == 4 &&
        Table->Table->GraphImpl4EffectiveKeySizeInBytes < sizeof(ULONG) &&
        Table->Table->GraphImpl4KeyDownsizeBitmap != 0) {
        if (Table->Table->GraphImpl4KeyDownsizeContiguous) {
            DownsizedKey = (ULONG)(
                (DownsizedKey >> Table->Table->GraphImpl4KeyDownsizeTrailingZeros) &
                Table->Table->GraphImpl4KeyDownsizeShiftedMask
            );
        } else {
            DownsizedKey = (ULONG)ExtractBits64(
                (ULONGLONG)DownsizedKey,
                Table->Table->GraphImpl4KeyDownsizeBitmap
            );
        }
    }
    return Table->Table->Vtbl->Index(Table->Table, DownsizedKey, (PULONG)Index);
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32x4(
    PH_ONLINE_JIT_TABLE *Table,
    const uint32_t *Keys,
    uint32_t *Indexes
    )
{
    HRESULT Result;
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Keys) ||
        !ARGUMENT_PRESENT(Indexes)) {
        return E_INVALIDARG;
    }

    Result = PhEnsureOnlineJitInterface(Table);
    if (FAILED(Result)) {
        return Result;
    }

    Jit = Table->JitInterface;
    return Jit->Vtbl->Index32x4(
        Jit,
        (ULONG)Keys[0],
        (ULONG)Keys[1],
        (ULONG)Keys[2],
        (ULONG)Keys[3],
        (PULONG)&Indexes[0],
        (PULONG)&Indexes[1],
        (PULONG)&Indexes[2],
        (PULONG)&Indexes[3]
    );
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32x8(
    PH_ONLINE_JIT_TABLE *Table,
    const uint32_t *Keys,
    uint32_t *Indexes
    )
{
    HRESULT Result;
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Keys) ||
        !ARGUMENT_PRESENT(Indexes)) {
        return E_INVALIDARG;
    }

    Result = PhEnsureOnlineJitInterface(Table);
    if (FAILED(Result)) {
        return Result;
    }

    Jit = Table->JitInterface;
    return Jit->Vtbl->Index32x8(
        Jit,
        (ULONG)Keys[0],
        (ULONG)Keys[1],
        (ULONG)Keys[2],
        (ULONG)Keys[3],
        (ULONG)Keys[4],
        (ULONG)Keys[5],
        (ULONG)Keys[6],
        (ULONG)Keys[7],
        (PULONG)&Indexes[0],
        (PULONG)&Indexes[1],
        (PULONG)&Indexes[2],
        (PULONG)&Indexes[3],
        (PULONG)&Indexes[4],
        (PULONG)&Indexes[5],
        (PULONG)&Indexes[6],
        (PULONG)&Indexes[7]
    );
}

PH_ONLINE_JIT_API
int32_t
PhOnlineJitIndex32x16(
    PH_ONLINE_JIT_TABLE *Table,
    const uint32_t *Keys,
    uint32_t *Indexes
    )
{
    HRESULT Result;
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Keys) ||
        !ARGUMENT_PRESENT(Indexes)) {
        return E_INVALIDARG;
    }

    Result = PhEnsureOnlineJitInterface(Table);
    if (FAILED(Result)) {
        return Result;
    }

    Jit = Table->JitInterface;
    return Jit->Vtbl->Index32x16(
        Jit,
        (ULONG)Keys[0],
        (ULONG)Keys[1],
        (ULONG)Keys[2],
        (ULONG)Keys[3],
        (ULONG)Keys[4],
        (ULONG)Keys[5],
        (ULONG)Keys[6],
        (ULONG)Keys[7],
        (ULONG)Keys[8],
        (ULONG)Keys[9],
        (ULONG)Keys[10],
        (ULONG)Keys[11],
        (ULONG)Keys[12],
        (ULONG)Keys[13],
        (ULONG)Keys[14],
        (ULONG)Keys[15],
        (PULONG)&Indexes[0],
        (PULONG)&Indexes[1],
        (PULONG)&Indexes[2],
        (PULONG)&Indexes[3],
        (PULONG)&Indexes[4],
        (PULONG)&Indexes[5],
        (PULONG)&Indexes[6],
        (PULONG)&Indexes[7],
        (PULONG)&Indexes[8],
        (PULONG)&Indexes[9],
        (PULONG)&Indexes[10],
        (PULONG)&Indexes[11],
        (PULONG)&Indexes[12],
        (PULONG)&Indexes[13],
        (PULONG)&Indexes[14],
        (PULONG)&Indexes[15]
    );
}

PH_ONLINE_JIT_API
void
PhOnlineJitReleaseTable(
    PH_ONLINE_JIT_TABLE *Table
    )
{
    if (!Table) {
        return;
    }

    PhReleaseOnlineJitInterface(Table);

    if (Table->Table) {
        Table->Table->Vtbl->Release(Table->Table);
        Table->Table = NULL;
    }

    free(Table);
}
