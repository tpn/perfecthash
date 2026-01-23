/*++

Copyright (c) 2025 Trent Nelson <trent@trent.me>

Module Name:

    ChmOnline01RawDog.c

Abstract:

    This module implements the RawDog JIT backend for CHM01 online tables.

--*/

#include "stdafx.h"

#if defined(PH_HAS_RAWDOG_JIT)

#include "ChmOnline01.h"
#include "PerfectHashJitRawDogMultiplyShiftR_x64.h"

#define RAWDOG_SENTINEL_ASSIGNED    0xA1A1A1A1A1A1A1A1ULL
#define RAWDOG_SENTINEL_SEED1       0xB1B1B1B1B1B1B1B1ULL
#define RAWDOG_SENTINEL_SEED2       0xC1C1C1C1C1C1C1C1ULL
#define RAWDOG_SENTINEL_SEED3_BYTE1 0xD1D1D1D1D1D1D1D1ULL
#define RAWDOG_SENTINEL_SEED3_BYTE2 0xE1E1E1E1E1E1E1E1ULL
#define RAWDOG_SENTINEL_HASH_MASK   0xF1F1F1F1F1F1F1F1ULL
#define RAWDOG_SENTINEL_INDEX_MASK  0x2121212121212121ULL

typedef struct _RAW_DOG_PATCH_ENTRY {
    ULONGLONG Sentinel;
    ULONGLONG Value;
    PCSZ Name;
} RAW_DOG_PATCH_ENTRY;

static const CHAR RawDogTargetCpu[] = "rawdog-x64";

_Use_decl_annotations_
VOID
PerfectHashTableJitRundownRawDog(
    PPERFECT_HASH_TABLE Table
    );

static
HRESULT
PatchRawDogSentinel(
    _Inout_ PBYTE Code,
    _In_ SIZE_T CodeSize,
    _In_ ULONGLONG Sentinel,
    _In_ ULONGLONG Value,
    _In_ PCSZ Name
    )
{
    SIZE_T Offset;
    ULONG Matches = 0;

    for (Offset = 0; Offset + sizeof(ULONGLONG) <= CodeSize; Offset++) {
        ULONGLONG Candidate;

        memcpy(&Candidate, Code + Offset, sizeof(Candidate));
        if (Candidate == Sentinel) {
            memcpy(Code + Offset, &Value, sizeof(Value));
            Matches++;
        }
    }

    if (Matches != 1) {
        UNREFERENCED_PARAMETER(Name);
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    return S_OK;
}

static
HRESULT
CompileChm01IndexJitRawDog(
    _In_ PPERFECT_HASH_TABLE Table,
    _Inout_ PPERFECT_HASH_TABLE_JIT Jit,
    _In_ PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
    BOOL Success;
    DWORD OldProtect = 0;
    PTABLE_INFO_ON_DISK TableInfo;
    ULONG_BYTES Seed3Bytes;
    ULONGLONG Assigned;
    SIZE_T CodeSize;
    SIZE_T AllocSize;
    PBYTE Code;
    RAW_DOG_PATCH_ENTRY Entries[7];
    ULONG EntryIndex = 0;
    HRESULT Result;

    if (Table->HashFunctionId != PerfectHashHashMultiplyShiftRFunctionId ||
        Table->MaskFunctionId != PerfectHashAndMaskFunctionId) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (Table->State.UsingAssigned16) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (!ARGUMENT_PRESENT(Table->TableData)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    TableInfo = Table->TableInfoOnDisk;
    if (!ARGUMENT_PRESENT(TableInfo)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    Seed3Bytes.AsULong = TableInfo->Seed3;
    Assigned = (ULONGLONG)(ULONG_PTR)Table->TableData;

    CodeSize = sizeof(PerfectHashJitRawDogMultiplyShiftR_x64);
    AllocSize = ALIGN_UP(CodeSize, PAGE_SIZE);

    Code = (PBYTE)VirtualAlloc(NULL,
                               AllocSize,
                               MEM_RESERVE | MEM_COMMIT,
                               PAGE_READWRITE);
    if (!Code) {
        SYS_ERROR(VirtualAlloc);
        return E_OUTOFMEMORY;
    }

    memcpy(Code, PerfectHashJitRawDogMultiplyShiftR_x64, CodeSize);

    Entries[EntryIndex++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_ASSIGNED,
        Assigned,
        "Assigned"
    };
    Entries[EntryIndex++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_SEED1,
        (ULONGLONG)TableInfo->Seed1,
        "Seed1"
    };
    Entries[EntryIndex++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_SEED2,
        (ULONGLONG)TableInfo->Seed2,
        "Seed2"
    };
    Entries[EntryIndex++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_SEED3_BYTE1,
        (ULONGLONG)Seed3Bytes.Byte1,
        "Seed3Byte1"
    };
    Entries[EntryIndex++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_SEED3_BYTE2,
        (ULONGLONG)Seed3Bytes.Byte2,
        "Seed3Byte2"
    };
    Entries[EntryIndex++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_HASH_MASK,
        (ULONGLONG)Table->HashMask,
        "HashMask"
    };
    Entries[EntryIndex++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_INDEX_MASK,
        (ULONGLONG)Table->IndexMask,
        "IndexMask"
    };

    for (EntryIndex = 0; EntryIndex < ARRAYSIZE(Entries); EntryIndex++) {
        Result = PatchRawDogSentinel(Code,
                                     CodeSize,
                                     Entries[EntryIndex].Sentinel,
                                     Entries[EntryIndex].Value,
                                     Entries[EntryIndex].Name);
        if (FAILED(Result)) {
            VirtualFree(Code, 0, MEM_RELEASE);
            return Result;
        }
    }

#if defined(__GNUC__)
    __builtin___clear_cache((char *)Code, (char *)Code + CodeSize);
#endif

    Success = VirtualProtect(Code,
                             AllocSize,
                             PAGE_EXECUTE_READ,
                             &OldProtect);
    if (!Success) {
        SYS_ERROR(VirtualProtect);
        VirtualFree(Code, 0, MEM_RELEASE);
        return PH_E_SYSTEM_CALL_FAILED;
    }

    Jit->ExecutionEngine = Code;
    Jit->Index32Function = Code;
    Jit->Flags.Index32Compiled = TRUE;
    Jit->Flags.BackendRawDog = TRUE;
    Jit->JitMaxIsa = CompileFlags->JitMaxIsa;

    if (RawDogTargetCpu[0] != '\0') {
        strncpy(Jit->TargetCpu,
                RawDogTargetCpu,
                sizeof(Jit->TargetCpu) - 1);
        Jit->TargetCpu[sizeof(Jit->TargetCpu) - 1] = '\0';
    }

    return S_OK;
}

static
HRESULT
RawDogTableQueryInterfaceJit(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ REFIID InterfaceId,
    _Out_ PVOID *Interface
    )
{
    BOOLEAN Match;
    PPERFECT_HASH_TABLE_JIT Jit;

    if (!ARGUMENT_PRESENT(Interface)) {
        return E_POINTER;
    }

    *Interface = NULL;

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

#ifdef __cplusplus
    Match = InlineIsEqualGUID(InterfaceId,
                              IID_PERFECT_HASH_TABLE_JIT_INTERFACE);
#else
    Match = InlineIsEqualGUID(InterfaceId,
                              &IID_PERFECT_HASH_TABLE_JIT_INTERFACE);
#endif

    if (Match) {
        Jit = Table->Jit;
        if (!ARGUMENT_PRESENT(Jit) || !Jit->Flags.Valid) {
            return E_NOINTERFACE;
        }

        *Interface = &Jit->Interface;
        Table->Vtbl->AddRef(Table);
        return S_OK;
    }

    Jit = Table->Jit;
    if (ARGUMENT_PRESENT(Jit) &&
        ARGUMENT_PRESENT(Jit->OriginalQueryInterface)) {
        return Jit->OriginalQueryInterface(Table, InterfaceId, Interface);
    }

    return E_NOINTERFACE;
}

static
HRESULT
RawDogTableIndexJit(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ ULONG Key,
    _Out_ PULONG Index
    )
{
    PPERFECT_HASH_TABLE_JIT Jit;
    ULONG (*IndexFunction)(_In_ ULONG Key);

    if (!ARGUMENT_PRESENT(Table) || !ARGUMENT_PRESENT(Index)) {
        return E_POINTER;
    }

    Jit = Table->Jit;
    if (!ARGUMENT_PRESENT(Jit) ||
        !Jit->Flags.Valid ||
        !Jit->Flags.Index32Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (ULONG (*)(_In_ ULONG Key))Jit->Index32Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    *Index = IndexFunction(Key);
    return S_OK;
}

static
HRESULT
RawDogJitInterfaceQueryInterface(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ REFIID InterfaceId,
    _Out_ PVOID *Interface
    )
{
    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Jit->Table) ||
        !ARGUMENT_PRESENT(Interface)) {
        return E_POINTER;
    }

    return Jit->Table->Vtbl->QueryInterface(Jit->Table,
                                            InterfaceId,
                                            Interface);
}

static
ULONG
RawDogJitInterfaceAddRef(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit
    )
{
    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Jit->Table)) {
        return 0;
    }

    return Jit->Table->Vtbl->AddRef(Jit->Table);
}

static
ULONG
RawDogJitInterfaceRelease(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit
    )
{
    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Jit->Table)) {
        return 0;
    }

    return Jit->Table->Vtbl->Release(Jit->Table);
}

static
HRESULT
RawDogJitInterfaceCreateInstance(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ PIUNKNOWN UnknownOuter,
    _In_ REFIID InterfaceId,
    _Out_ PVOID *Interface
    )
{
    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Jit->Table)) {
        return E_POINTER;
    }

    return Jit->Table->Vtbl->CreateInstance(Jit->Table,
                                            UnknownOuter,
                                            InterfaceId,
                                            Interface);
}

static
HRESULT
RawDogJitInterfaceLockServer(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ BOOL Lock
    )
{
    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Jit->Table)) {
        return E_POINTER;
    }

    return Jit->Table->Vtbl->LockServer(Jit->Table, Lock);
}

static
HRESULT
RawDogJitInterfaceIndex32(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ ULONG Key,
    _Out_ PULONG Index
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    ULONG (*IndexFunction)(_In_ ULONG Key);

    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Index)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index32Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (ULONG (*)(_In_ ULONG Key))JitState->Index32Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    *Index = IndexFunction(Key);
    return S_OK;
}

static
HRESULT
RawDogJitInterfaceIndex64(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ ULONGLONG Key,
    _Out_ PULONG Index
    )
{
    UNREFERENCED_PARAMETER(Jit);
    UNREFERENCED_PARAMETER(Key);
    UNREFERENCED_PARAMETER(Index);
    return PH_E_NOT_IMPLEMENTED;
}

static
HRESULT
RawDogJitInterfaceIndex32x2(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ ULONG Key1,
    _In_ ULONG Key2,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2
    )
{
    UNREFERENCED_PARAMETER(Jit);
    UNREFERENCED_PARAMETER(Key1);
    UNREFERENCED_PARAMETER(Key2);
    UNREFERENCED_PARAMETER(Index1);
    UNREFERENCED_PARAMETER(Index2);
    return PH_E_NOT_IMPLEMENTED;
}

static
HRESULT
RawDogJitInterfaceIndex32x4(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ ULONG Key1,
    _In_ ULONG Key2,
    _In_ ULONG Key3,
    _In_ ULONG Key4,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4
    )
{
    UNREFERENCED_PARAMETER(Jit);
    UNREFERENCED_PARAMETER(Key1);
    UNREFERENCED_PARAMETER(Key2);
    UNREFERENCED_PARAMETER(Key3);
    UNREFERENCED_PARAMETER(Key4);
    UNREFERENCED_PARAMETER(Index1);
    UNREFERENCED_PARAMETER(Index2);
    UNREFERENCED_PARAMETER(Index3);
    UNREFERENCED_PARAMETER(Index4);
    return PH_E_NOT_IMPLEMENTED;
}

static
HRESULT
RawDogJitInterfaceIndex32x8(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ ULONG Key1,
    _In_ ULONG Key2,
    _In_ ULONG Key3,
    _In_ ULONG Key4,
    _In_ ULONG Key5,
    _In_ ULONG Key6,
    _In_ ULONG Key7,
    _In_ ULONG Key8,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4,
    _Out_ PULONG Index5,
    _Out_ PULONG Index6,
    _Out_ PULONG Index7,
    _Out_ PULONG Index8
    )
{
    UNREFERENCED_PARAMETER(Jit);
    UNREFERENCED_PARAMETER(Key1);
    UNREFERENCED_PARAMETER(Key2);
    UNREFERENCED_PARAMETER(Key3);
    UNREFERENCED_PARAMETER(Key4);
    UNREFERENCED_PARAMETER(Key5);
    UNREFERENCED_PARAMETER(Key6);
    UNREFERENCED_PARAMETER(Key7);
    UNREFERENCED_PARAMETER(Key8);
    UNREFERENCED_PARAMETER(Index1);
    UNREFERENCED_PARAMETER(Index2);
    UNREFERENCED_PARAMETER(Index3);
    UNREFERENCED_PARAMETER(Index4);
    UNREFERENCED_PARAMETER(Index5);
    UNREFERENCED_PARAMETER(Index6);
    UNREFERENCED_PARAMETER(Index7);
    UNREFERENCED_PARAMETER(Index8);
    return PH_E_NOT_IMPLEMENTED;
}

static
HRESULT
RawDogJitInterfaceIndex32x16(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ ULONG Key1,
    _In_ ULONG Key2,
    _In_ ULONG Key3,
    _In_ ULONG Key4,
    _In_ ULONG Key5,
    _In_ ULONG Key6,
    _In_ ULONG Key7,
    _In_ ULONG Key8,
    _In_ ULONG Key9,
    _In_ ULONG Key10,
    _In_ ULONG Key11,
    _In_ ULONG Key12,
    _In_ ULONG Key13,
    _In_ ULONG Key14,
    _In_ ULONG Key15,
    _In_ ULONG Key16,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4,
    _Out_ PULONG Index5,
    _Out_ PULONG Index6,
    _Out_ PULONG Index7,
    _Out_ PULONG Index8,
    _Out_ PULONG Index9,
    _Out_ PULONG Index10,
    _Out_ PULONG Index11,
    _Out_ PULONG Index12,
    _Out_ PULONG Index13,
    _Out_ PULONG Index14,
    _Out_ PULONG Index15,
    _Out_ PULONG Index16
    )
{
    UNREFERENCED_PARAMETER(Jit);
    UNREFERENCED_PARAMETER(Key1);
    UNREFERENCED_PARAMETER(Key2);
    UNREFERENCED_PARAMETER(Key3);
    UNREFERENCED_PARAMETER(Key4);
    UNREFERENCED_PARAMETER(Key5);
    UNREFERENCED_PARAMETER(Key6);
    UNREFERENCED_PARAMETER(Key7);
    UNREFERENCED_PARAMETER(Key8);
    UNREFERENCED_PARAMETER(Key9);
    UNREFERENCED_PARAMETER(Key10);
    UNREFERENCED_PARAMETER(Key11);
    UNREFERENCED_PARAMETER(Key12);
    UNREFERENCED_PARAMETER(Key13);
    UNREFERENCED_PARAMETER(Key14);
    UNREFERENCED_PARAMETER(Key15);
    UNREFERENCED_PARAMETER(Key16);
    UNREFERENCED_PARAMETER(Index1);
    UNREFERENCED_PARAMETER(Index2);
    UNREFERENCED_PARAMETER(Index3);
    UNREFERENCED_PARAMETER(Index4);
    UNREFERENCED_PARAMETER(Index5);
    UNREFERENCED_PARAMETER(Index6);
    UNREFERENCED_PARAMETER(Index7);
    UNREFERENCED_PARAMETER(Index8);
    UNREFERENCED_PARAMETER(Index9);
    UNREFERENCED_PARAMETER(Index10);
    UNREFERENCED_PARAMETER(Index11);
    UNREFERENCED_PARAMETER(Index12);
    UNREFERENCED_PARAMETER(Index13);
    UNREFERENCED_PARAMETER(Index14);
    UNREFERENCED_PARAMETER(Index15);
    UNREFERENCED_PARAMETER(Index16);
    return PH_E_NOT_IMPLEMENTED;
}

static
HRESULT
RawDogJitInterfaceIndex64x2(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ ULONGLONG Key1,
    _In_ ULONGLONG Key2,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2
    )
{
    UNREFERENCED_PARAMETER(Jit);
    UNREFERENCED_PARAMETER(Key1);
    UNREFERENCED_PARAMETER(Key2);
    UNREFERENCED_PARAMETER(Index1);
    UNREFERENCED_PARAMETER(Index2);
    return PH_E_NOT_IMPLEMENTED;
}

static
HRESULT
RawDogJitInterfaceIndex64x4(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ ULONGLONG Key1,
    _In_ ULONGLONG Key2,
    _In_ ULONGLONG Key3,
    _In_ ULONGLONG Key4,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4
    )
{
    UNREFERENCED_PARAMETER(Jit);
    UNREFERENCED_PARAMETER(Key1);
    UNREFERENCED_PARAMETER(Key2);
    UNREFERENCED_PARAMETER(Key3);
    UNREFERENCED_PARAMETER(Key4);
    UNREFERENCED_PARAMETER(Index1);
    UNREFERENCED_PARAMETER(Index2);
    UNREFERENCED_PARAMETER(Index3);
    UNREFERENCED_PARAMETER(Index4);
    return PH_E_NOT_IMPLEMENTED;
}

static
HRESULT
RawDogJitInterfaceIndex64x8(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ ULONGLONG Key1,
    _In_ ULONGLONG Key2,
    _In_ ULONGLONG Key3,
    _In_ ULONGLONG Key4,
    _In_ ULONGLONG Key5,
    _In_ ULONGLONG Key6,
    _In_ ULONGLONG Key7,
    _In_ ULONGLONG Key8,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4,
    _Out_ PULONG Index5,
    _Out_ PULONG Index6,
    _Out_ PULONG Index7,
    _Out_ PULONG Index8
    )
{
    UNREFERENCED_PARAMETER(Jit);
    UNREFERENCED_PARAMETER(Key1);
    UNREFERENCED_PARAMETER(Key2);
    UNREFERENCED_PARAMETER(Key3);
    UNREFERENCED_PARAMETER(Key4);
    UNREFERENCED_PARAMETER(Key5);
    UNREFERENCED_PARAMETER(Key6);
    UNREFERENCED_PARAMETER(Key7);
    UNREFERENCED_PARAMETER(Key8);
    UNREFERENCED_PARAMETER(Index1);
    UNREFERENCED_PARAMETER(Index2);
    UNREFERENCED_PARAMETER(Index3);
    UNREFERENCED_PARAMETER(Index4);
    UNREFERENCED_PARAMETER(Index5);
    UNREFERENCED_PARAMETER(Index6);
    UNREFERENCED_PARAMETER(Index7);
    UNREFERENCED_PARAMETER(Index8);
    return PH_E_NOT_IMPLEMENTED;
}

static
HRESULT
RawDogJitInterfaceIndex64x16(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _In_ ULONGLONG Key1,
    _In_ ULONGLONG Key2,
    _In_ ULONGLONG Key3,
    _In_ ULONGLONG Key4,
    _In_ ULONGLONG Key5,
    _In_ ULONGLONG Key6,
    _In_ ULONGLONG Key7,
    _In_ ULONGLONG Key8,
    _In_ ULONGLONG Key9,
    _In_ ULONGLONG Key10,
    _In_ ULONGLONG Key11,
    _In_ ULONGLONG Key12,
    _In_ ULONGLONG Key13,
    _In_ ULONGLONG Key14,
    _In_ ULONGLONG Key15,
    _In_ ULONGLONG Key16,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4,
    _Out_ PULONG Index5,
    _Out_ PULONG Index6,
    _Out_ PULONG Index7,
    _Out_ PULONG Index8,
    _Out_ PULONG Index9,
    _Out_ PULONG Index10,
    _Out_ PULONG Index11,
    _Out_ PULONG Index12,
    _Out_ PULONG Index13,
    _Out_ PULONG Index14,
    _Out_ PULONG Index15,
    _Out_ PULONG Index16
    )
{
    UNREFERENCED_PARAMETER(Jit);
    UNREFERENCED_PARAMETER(Key1);
    UNREFERENCED_PARAMETER(Key2);
    UNREFERENCED_PARAMETER(Key3);
    UNREFERENCED_PARAMETER(Key4);
    UNREFERENCED_PARAMETER(Key5);
    UNREFERENCED_PARAMETER(Key6);
    UNREFERENCED_PARAMETER(Key7);
    UNREFERENCED_PARAMETER(Key8);
    UNREFERENCED_PARAMETER(Key9);
    UNREFERENCED_PARAMETER(Key10);
    UNREFERENCED_PARAMETER(Key11);
    UNREFERENCED_PARAMETER(Key12);
    UNREFERENCED_PARAMETER(Key13);
    UNREFERENCED_PARAMETER(Key14);
    UNREFERENCED_PARAMETER(Key15);
    UNREFERENCED_PARAMETER(Key16);
    UNREFERENCED_PARAMETER(Index1);
    UNREFERENCED_PARAMETER(Index2);
    UNREFERENCED_PARAMETER(Index3);
    UNREFERENCED_PARAMETER(Index4);
    UNREFERENCED_PARAMETER(Index5);
    UNREFERENCED_PARAMETER(Index6);
    UNREFERENCED_PARAMETER(Index7);
    UNREFERENCED_PARAMETER(Index8);
    UNREFERENCED_PARAMETER(Index9);
    UNREFERENCED_PARAMETER(Index10);
    UNREFERENCED_PARAMETER(Index11);
    UNREFERENCED_PARAMETER(Index12);
    UNREFERENCED_PARAMETER(Index13);
    UNREFERENCED_PARAMETER(Index14);
    UNREFERENCED_PARAMETER(Index15);
    UNREFERENCED_PARAMETER(Index16);
    return PH_E_NOT_IMPLEMENTED;
}

static
HRESULT
RawDogJitInterfaceGetInfo(
    _In_ PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    _Out_ PPERFECT_HASH_TABLE_JIT_INFO Info
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;

    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Info)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) || !JitState->Flags.Valid) {
        return PH_E_NOT_IMPLEMENTED;
    }

    ZeroMemory(Info, sizeof(*Info));
    Info->SizeOfStruct = sizeof(*Info);
    Info->Flags.AsULong = JitState->Flags.AsULong;
    Info->JitMaxIsa = JitState->JitMaxIsa;
    if (JitState->TargetCpu[0] != '\0') {
        strncpy(Info->TargetCpu,
                JitState->TargetCpu,
                sizeof(Info->TargetCpu) - 1);
        Info->TargetCpu[sizeof(Info->TargetCpu) - 1] = '\0';
    }
    if (JitState->TargetFeatures[0] != '\0') {
        strncpy(Info->TargetFeatures,
                JitState->TargetFeatures,
                sizeof(Info->TargetFeatures) - 1);
        Info->TargetFeatures[sizeof(Info->TargetFeatures) - 1] = '\0';
    }

    return S_OK;
}

static const PERFECT_HASH_TABLE_JIT_INTERFACE_VTBL RawDogJitInterfaceVtbl = {
    RawDogJitInterfaceQueryInterface,
    RawDogJitInterfaceAddRef,
    RawDogJitInterfaceRelease,
    RawDogJitInterfaceCreateInstance,
    RawDogJitInterfaceLockServer,
    RawDogJitInterfaceIndex32,
    RawDogJitInterfaceIndex64,
    RawDogJitInterfaceIndex32x2,
    RawDogJitInterfaceIndex32x4,
    RawDogJitInterfaceIndex32x8,
    RawDogJitInterfaceIndex32x16,
    RawDogJitInterfaceIndex64x2,
    RawDogJitInterfaceIndex64x4,
    RawDogJitInterfaceIndex64x8,
    RawDogJitInterfaceIndex64x16,
    RawDogJitInterfaceGetInfo,
};

_Use_decl_annotations_
HRESULT
PerfectHashTableCompileJitRawDog(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlagsPointer
    )
{
    HRESULT Result;
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE_JIT Jit;
    PERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags;

    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    if (!Table->Flags.Created) {
        return PH_E_TABLE_NOT_CREATED;
    }

    if (!ARGUMENT_PRESENT(Table->TableInfoOnDisk)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    if (Table->State.UsingAssigned16) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (!ARGUMENT_PRESENT(Table->TableData)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    if (Table->AlgorithmId != PerfectHashChm01AlgorithmId) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (Table->TableInfoOnDisk->KeySizeInBytes != sizeof(ULONG)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (Table->MaskFunctionId != PerfectHashAndMaskFunctionId) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (Table->HashFunctionId != PerfectHashHashMultiplyShiftRFunctionId) {
        return PH_E_NOT_IMPLEMENTED;
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

    CompileFlags.Jit = TRUE;
    if (CompileFlags.JitIndex64 ||
        CompileFlags.JitIndex32x2 ||
        CompileFlags.JitIndex32x4 ||
        CompileFlags.JitIndex32x8 ||
        CompileFlags.JitIndex32x16 ||
        CompileFlags.JitVectorIndex32x2 ||
        CompileFlags.JitVectorIndex32x4 ||
        CompileFlags.JitVectorIndex32x8) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (Table->Jit) {
        PerfectHashTableJitRundown(Table);
    }

    Allocator = Table->Allocator;
    Jit = (PPERFECT_HASH_TABLE_JIT)(
        Allocator->Vtbl->Calloc(
            Allocator,
            1,
            sizeof(*Jit)
        )
    );

    if (!Jit) {
        return E_OUTOFMEMORY;
    }

    Table->Jit = Jit;
    Jit->SizeOfStruct = sizeof(*Jit);
    Jit->AlgorithmId = Table->AlgorithmId;
    Jit->HashFunctionId = Table->HashFunctionId;
    Jit->MaskFunctionId = Table->MaskFunctionId;
    Jit->OriginalIndex = Table->Vtbl->Index;
    Jit->OriginalQueryInterface = Table->Vtbl->QueryInterface;
    Jit->Flags.BackendRawDog = TRUE;
    Jit->Interface.Table = Table;
    Jit->Interface.Vtbl = &RawDogJitInterfaceVtbl;
    Table->Vtbl->QueryInterface = RawDogTableQueryInterfaceJit;

    Result = CompileChm01IndexJitRawDog(Table, Jit, &CompileFlags);
    if (FAILED(Result)) {
        PerfectHashTableJitRundownRawDog(Table);
        return Result;
    }

    Jit->Flags.Valid = TRUE;
    Table->Flags.JitEnabled = TRUE;
    Table->Vtbl->Index = RawDogTableIndexJit;

    return S_OK;
}

_Use_decl_annotations_
VOID
PerfectHashTableJitRundownRawDog(
    PPERFECT_HASH_TABLE Table
    )
{
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE_JIT Jit;

    if (!ARGUMENT_PRESENT(Table)) {
        return;
    }

    Jit = Table->Jit;
    if (!ARGUMENT_PRESENT(Jit)) {
        return;
    }

    if (Jit->OriginalIndex) {
        Table->Vtbl->Index = Jit->OriginalIndex;
    }

    if (Jit->OriginalQueryInterface) {
        Table->Vtbl->QueryInterface = Jit->OriginalQueryInterface;
    }

    Table->Flags.JitEnabled = FALSE;

    if (Jit->ExecutionEngine) {
        VirtualFree(Jit->ExecutionEngine, 0, MEM_RELEASE);
        Jit->ExecutionEngine = NULL;
    }

    Allocator = Table->Allocator;
    Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&Table->Jit);
}

#endif // PH_HAS_RAWDOG_JIT

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
