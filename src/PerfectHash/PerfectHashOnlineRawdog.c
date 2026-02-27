/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashOnlineRawdog.c

Abstract:

    Minimal C API wrapper for creating and querying 32-bit perfect hash
    tables with online mode + RawDog JIT.

--*/

#include "stdafx.h"
#include <PerfectHashOnlineRawdog.h>

#include <stdlib.h>

extern DLL_GET_CLASS_OBJECT PerfectHashDllGetClassObject;

struct PH_ONLINE_RAWDOG_CONTEXT {
    PICLASSFACTORY ClassFactory;
    PPERFECT_HASH_ONLINE Online;
};

struct PH_ONLINE_RAWDOG_TABLE {
    PPERFECT_HASH_TABLE Table;
};

static
HRESULT
PhMapRawdogHashFunction(
    _In_ PH_ONLINE_RAWDOG_HASH_FUNCTION HashFunction,
    _Out_ PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId
    )
{
    if (!ARGUMENT_PRESENT(HashFunctionId)) {
        return E_POINTER;
    }

    switch (HashFunction) {
        case PhOnlineRawdogHashMultiplyShiftR:
            *HashFunctionId = PerfectHashHashMultiplyShiftRFunctionId;
            break;
        case PhOnlineRawdogHashMultiplyShiftLR:
            *HashFunctionId = PerfectHashHashMultiplyShiftLRFunctionId;
            break;
        case PhOnlineRawdogHashMultiplyShiftRMultiply:
            *HashFunctionId = PerfectHashHashMultiplyShiftRMultiplyFunctionId;
            break;
        case PhOnlineRawdogHashMultiplyShiftR2:
            *HashFunctionId = PerfectHashHashMultiplyShiftR2FunctionId;
            break;
        case PhOnlineRawdogHashMultiplyShiftRX:
            *HashFunctionId = PerfectHashHashMultiplyShiftRXFunctionId;
            break;
        case PhOnlineRawdogHashMulshrolate1RX:
            *HashFunctionId = PerfectHashHashMulshrolate1RXFunctionId;
            break;
        case PhOnlineRawdogHashMulshrolate2RX:
            *HashFunctionId = PerfectHashHashMulshrolate2RXFunctionId;
            break;
        case PhOnlineRawdogHashMulshrolate3RX:
            *HashFunctionId = PerfectHashHashMulshrolate3RXFunctionId;
            break;
        case PhOnlineRawdogHashMulshrolate4RX:
            *HashFunctionId = PerfectHashHashMulshrolate4RXFunctionId;
            break;
        default:
            return E_INVALIDARG;
    }

    return S_OK;
}

static
HRESULT
PhApplyRawdogVectorWidth(
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
PhCompileRawdogTableWithVectorWidth(
    _In_ PH_ONLINE_RAWDOG_CONTEXT *Context,
    _In_ PH_ONLINE_RAWDOG_TABLE *Table,
    _In_ ULONG VectorWidth,
    _In_ PH_ONLINE_RAWDOG_JIT_MAX_ISA JitMaxIsa
    )
{
    HRESULT Result;
    PERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags = {0};

    CompileFlags.Jit = TRUE;
    CompileFlags.JitBackendRawDog = TRUE;

    Result = PhApplyRawdogVectorWidth(VectorWidth, &CompileFlags);
    if (FAILED(Result)) {
        return Result;
    }

    CompileFlags.JitMaxIsa = (ULONG)JitMaxIsa;

    return Context->Online->Vtbl->CompileTable(Context->Online,
                                               Table->Table,
                                               &CompileFlags);
}

PH_ONLINE_RAWDOG_API
int32_t
PhOnlineRawdogOpen(
    PH_ONLINE_RAWDOG_CONTEXT **ContextPointer
    )
{
    HRESULT Result;
    PH_ONLINE_RAWDOG_CONTEXT *Context = NULL;
    PICLASSFACTORY ClassFactory = NULL;
    PPERFECT_HASH_ONLINE Online = NULL;

    if (!ARGUMENT_PRESENT(ContextPointer)) {
        return E_POINTER;
    }

    *ContextPointer = NULL;

    Context = (PH_ONLINE_RAWDOG_CONTEXT *)calloc(1, sizeof(*Context));
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

PH_ONLINE_RAWDOG_API
void
PhOnlineRawdogClose(
    PH_ONLINE_RAWDOG_CONTEXT *Context
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

PH_ONLINE_RAWDOG_API
int32_t
PhOnlineRawdogCreateTable32(
    PH_ONLINE_RAWDOG_CONTEXT *Context,
    PH_ONLINE_RAWDOG_HASH_FUNCTION HashFunction,
    const uint32_t *Keys,
    uint64_t NumberOfKeys,
    PH_ONLINE_RAWDOG_TABLE **TablePointer
    )
{
    HRESULT Result;
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags = {0};
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags = {0};
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;
    PPERFECT_HASH_TABLE Table = NULL;
    PH_ONLINE_RAWDOG_TABLE *Wrapper = NULL;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(Context->Online) ||
        !ARGUMENT_PRESENT(Keys) ||
        NumberOfKeys == 0 ||
        !ARGUMENT_PRESENT(TablePointer)) {
        return E_INVALIDARG;
    }

    *TablePointer = NULL;

    Result = PhMapRawdogHashFunction(HashFunction, &HashFunctionId);
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

    Wrapper = (PH_ONLINE_RAWDOG_TABLE *)calloc(1, sizeof(*Wrapper));
    if (!Wrapper) {
        Table->Vtbl->Release(Table);
        return E_OUTOFMEMORY;
    }

    Wrapper->Table = Table;
    *TablePointer = Wrapper;

    return S_OK;
}

PH_ONLINE_RAWDOG_API
int32_t
PhOnlineRawdogCompileTable(
    PH_ONLINE_RAWDOG_CONTEXT *Context,
    PH_ONLINE_RAWDOG_TABLE *Table,
    uint32_t VectorWidth,
    PH_ONLINE_RAWDOG_JIT_MAX_ISA JitMaxIsa
    )
{
    HRESULT Result;
    HRESULT LastResult;
    ULONG CandidateWidths[4] = {0};
    ULONG CandidateCount = 0;
    ULONG Index;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(Context->Online) ||
        !ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table)) {
        return E_INVALIDARG;
    }

    if (!IsValidPerfectHashJitMaxIsaId((PERFECT_HASH_JIT_MAX_ISA_ID)JitMaxIsa)) {
        return E_INVALIDARG;
    }

    //
    // RawDog JIT supports a platform/hash-specific subset of vectorized
    // kernels. If a requested width is unavailable, retry progressively
    // smaller widths before surfacing not-implemented.
    //

    switch (VectorWidth) {
        case 0:
        case 1:
            CandidateWidths[CandidateCount++] = VectorWidth;
            break;
        case 2:
            CandidateWidths[CandidateCount++] = 2;
            CandidateWidths[CandidateCount++] = 1;
            break;
        case 4:
            CandidateWidths[CandidateCount++] = 4;
            CandidateWidths[CandidateCount++] = 1;
            break;
        case 8:
            CandidateWidths[CandidateCount++] = 8;
            CandidateWidths[CandidateCount++] = 4;
            CandidateWidths[CandidateCount++] = 1;
            break;
        case 16:
            CandidateWidths[CandidateCount++] = 16;
            CandidateWidths[CandidateCount++] = 8;
            CandidateWidths[CandidateCount++] = 4;
            CandidateWidths[CandidateCount++] = 1;
            break;
        default:
            return E_INVALIDARG;
    }

    LastResult = PH_E_NOT_IMPLEMENTED;

    for (Index = 0; Index < CandidateCount; Index++) {
        Result = PhCompileRawdogTableWithVectorWidth(Context,
                                                     Table,
                                                     CandidateWidths[Index],
                                                     JitMaxIsa);
        if (SUCCEEDED(Result)) {
            return S_OK;
        }

        if (Result != PH_E_NOT_IMPLEMENTED &&
            Result != PH_E_INVARIANT_CHECK_FAILED) {
            return Result;
        }

        LastResult = Result;
    }

    return LastResult;
}

PH_ONLINE_RAWDOG_API
int32_t
PhOnlineRawdogIndex32(
    PH_ONLINE_RAWDOG_TABLE *Table,
    uint32_t Key,
    uint32_t *Index
    )
{
    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Table->Table) ||
        !ARGUMENT_PRESENT(Index)) {
        return E_INVALIDARG;
    }

    return Table->Table->Vtbl->Index(Table->Table, (ULONG)Key, (PULONG)Index);
}

PH_ONLINE_RAWDOG_API
void
PhOnlineRawdogReleaseTable(
    PH_ONLINE_RAWDOG_TABLE *Table
    )
{
    if (!Table) {
        return;
    }

    if (Table->Table) {
        Table->Table->Vtbl->Release(Table->Table);
        Table->Table = NULL;
    }

    free(Table);
}
