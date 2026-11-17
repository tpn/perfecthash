/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableJitInterface.c

Abstract:

    Implements the table-owned index JIT interface. Its COM lifetime follows
    the table, and each index call uses the currently installed backend.

--*/

#include "stdafx.h"
#include <string.h>

typedef
ULONG
(PH_JIT_INDEX32_FUNCTION)(
    _In_ ULONG Key
    );
typedef PH_JIT_INDEX32_FUNCTION *PPH_JIT_INDEX32_FUNCTION;

typedef
ULONG
(PH_JIT_INDEX64_FUNCTION)(
    _In_ ULONGLONG Key
    );
typedef PH_JIT_INDEX64_FUNCTION *PPH_JIT_INDEX64_FUNCTION;

typedef
VOID
(PH_JIT_INDEX32X2_FUNCTION)(
    _In_ ULONG Key1,
    _In_ ULONG Key2,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2
    );
typedef PH_JIT_INDEX32X2_FUNCTION *PPH_JIT_INDEX32X2_FUNCTION;

typedef
VOID
(PH_JIT_INDEX32X4_FUNCTION)(
    _In_ ULONG Key1,
    _In_ ULONG Key2,
    _In_ ULONG Key3,
    _In_ ULONG Key4,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4
    );
typedef PH_JIT_INDEX32X4_FUNCTION *PPH_JIT_INDEX32X4_FUNCTION;

typedef
VOID
(PH_JIT_INDEX32X8_FUNCTION)(
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
    );
typedef PH_JIT_INDEX32X8_FUNCTION *PPH_JIT_INDEX32X8_FUNCTION;

typedef
VOID
(PH_JIT_INDEX32X16_FUNCTION)(
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
    );
typedef PH_JIT_INDEX32X16_FUNCTION *PPH_JIT_INDEX32X16_FUNCTION;

typedef
VOID
(PH_JIT_INDEX64X2_FUNCTION)(
    _In_ ULONGLONG Key1,
    _In_ ULONGLONG Key2,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2
    );
typedef PH_JIT_INDEX64X2_FUNCTION *PPH_JIT_INDEX64X2_FUNCTION;

typedef
VOID
(PH_JIT_INDEX64X4_FUNCTION)(
    _In_ ULONGLONG Key1,
    _In_ ULONGLONG Key2,
    _In_ ULONGLONG Key3,
    _In_ ULONGLONG Key4,
    _Out_ PULONG Index1,
    _Out_ PULONG Index2,
    _Out_ PULONG Index3,
    _Out_ PULONG Index4
    );
typedef PH_JIT_INDEX64X4_FUNCTION *PPH_JIT_INDEX64X4_FUNCTION;

typedef
VOID
(PH_JIT_INDEX64X8_FUNCTION)(
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
    );
typedef PH_JIT_INDEX64X8_FUNCTION *PPH_JIT_INDEX64X8_FUNCTION;

typedef
VOID
(PH_JIT_INDEX64X16_FUNCTION)(
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
    );
typedef PH_JIT_INDEX64X16_FUNCTION *PPH_JIT_INDEX64X16_FUNCTION;

static PERFECT_HASH_TABLE_JIT_INTERFACE_QUERY_INTERFACE
    PerfectHashTableJitInterfaceQueryInterface;
static PERFECT_HASH_TABLE_JIT_INTERFACE_ADD_REF
    PerfectHashTableJitInterfaceAddRef;
static PERFECT_HASH_TABLE_JIT_INTERFACE_RELEASE
    PerfectHashTableJitInterfaceRelease;
static PERFECT_HASH_TABLE_JIT_INTERFACE_CREATE_INSTANCE
    PerfectHashTableJitInterfaceCreateInstance;
static PERFECT_HASH_TABLE_JIT_INTERFACE_LOCK_SERVER
    PerfectHashTableJitInterfaceLockServer;
static PERFECT_HASH_TABLE_JIT_INDEX32 PerfectHashTableJitInterfaceIndex32;
static PERFECT_HASH_TABLE_JIT_INDEX64 PerfectHashTableJitInterfaceIndex64;
static PERFECT_HASH_TABLE_JIT_INDEX32X2 PerfectHashTableJitInterfaceIndex32x2;
static PERFECT_HASH_TABLE_JIT_INDEX32X4 PerfectHashTableJitInterfaceIndex32x4;
static PERFECT_HASH_TABLE_JIT_INDEX32X8 PerfectHashTableJitInterfaceIndex32x8;
static PERFECT_HASH_TABLE_JIT_INDEX32X16 PerfectHashTableJitInterfaceIndex32x16;
static PERFECT_HASH_TABLE_JIT_INDEX64X2 PerfectHashTableJitInterfaceIndex64x2;
static PERFECT_HASH_TABLE_JIT_INDEX64X4 PerfectHashTableJitInterfaceIndex64x4;
static PERFECT_HASH_TABLE_JIT_INDEX64X8 PerfectHashTableJitInterfaceIndex64x8;
static PERFECT_HASH_TABLE_JIT_INDEX64X16 PerfectHashTableJitInterfaceIndex64x16;
static PERFECT_HASH_TABLE_JIT_GET_INFO PerfectHashTableJitInterfaceGetInfo;

PERFECT_HASH_TABLE_JIT_INTERFACE_VTBL PerfectHashTableJitInterfaceVtbl = {
    PerfectHashTableJitInterfaceQueryInterface,
    PerfectHashTableJitInterfaceAddRef,
    PerfectHashTableJitInterfaceRelease,
    PerfectHashTableJitInterfaceCreateInstance,
    PerfectHashTableJitInterfaceLockServer,
    PerfectHashTableJitInterfaceIndex32,
    PerfectHashTableJitInterfaceIndex64,
    PerfectHashTableJitInterfaceIndex32x2,
    PerfectHashTableJitInterfaceIndex32x4,
    PerfectHashTableJitInterfaceIndex32x8,
    PerfectHashTableJitInterfaceIndex32x16,
    PerfectHashTableJitInterfaceIndex64x2,
    PerfectHashTableJitInterfaceIndex64x4,
    PerfectHashTableJitInterfaceIndex64x8,
    PerfectHashTableJitInterfaceIndex64x16,
    PerfectHashTableJitInterfaceGetInfo,
};

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceQueryInterface(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    REFIID InterfaceId,
    PVOID *Interface
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

_Use_decl_annotations_
static
ULONG
PerfectHashTableJitInterfaceAddRef(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit
    )
{
    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Jit->Table)) {
        return 0;
    }

    return Jit->Table->Vtbl->AddRef(Jit->Table);
}

_Use_decl_annotations_
static
ULONG
PerfectHashTableJitInterfaceRelease(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit
    )
{
    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Jit->Table)) {
        return 0;
    }

    return Jit->Table->Vtbl->Release(Jit->Table);
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceCreateInstance(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    PIUNKNOWN UnknownOuter,
    REFIID InterfaceId,
    PVOID *Interface
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

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceLockServer(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    BOOL Lock
    )
{
    if (!ARGUMENT_PRESENT(Jit) || !ARGUMENT_PRESENT(Jit->Table)) {
        return E_POINTER;
    }

    return Jit->Table->Vtbl->LockServer(Jit->Table, Lock);
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceIndex32(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONG Key,
    PULONG Index
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX32_FUNCTION IndexFunction;

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

    IndexFunction = (PPH_JIT_INDEX32_FUNCTION)JitState->Index32Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    *Index = IndexFunction(Key);
    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceIndex64(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONGLONG Key,
    PULONG Index
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX64_FUNCTION IndexFunction;

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
        !JitState->Flags.Index64Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX64_FUNCTION)JitState->Index64Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    *Index = IndexFunction(Key);
    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceIndex32x2(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONG Key1,
    ULONG Key2,
    PULONG Index1,
    PULONG Index2
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX32X2_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index32x2Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX32X2_FUNCTION)JitState->Index32x2Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1, Key2, Index1, Index2);
    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceIndex32x4(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONG Key1,
    ULONG Key2,
    ULONG Key3,
    ULONG Key4,
    PULONG Index1,
    PULONG Index2,
    PULONG Index3,
    PULONG Index4
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX32X4_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2) ||
        !ARGUMENT_PRESENT(Index3) ||
        !ARGUMENT_PRESENT(Index4)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index32x4Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX32X4_FUNCTION)JitState->Index32x4Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1,
                  Key2,
                  Key3,
                  Key4,
                  Index1,
                  Index2,
                  Index3,
                  Index4);
    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceIndex32x8(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONG Key1,
    ULONG Key2,
    ULONG Key3,
    ULONG Key4,
    ULONG Key5,
    ULONG Key6,
    ULONG Key7,
    ULONG Key8,
    PULONG Index1,
    PULONG Index2,
    PULONG Index3,
    PULONG Index4,
    PULONG Index5,
    PULONG Index6,
    PULONG Index7,
    PULONG Index8
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX32X8_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2) ||
        !ARGUMENT_PRESENT(Index3) ||
        !ARGUMENT_PRESENT(Index4) ||
        !ARGUMENT_PRESENT(Index5) ||
        !ARGUMENT_PRESENT(Index6) ||
        !ARGUMENT_PRESENT(Index7) ||
        !ARGUMENT_PRESENT(Index8)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index32x8Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX32X8_FUNCTION)JitState->Index32x8Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1,
                  Key2,
                  Key3,
                  Key4,
                  Key5,
                  Key6,
                  Key7,
                  Key8,
                  Index1,
                  Index2,
                  Index3,
                  Index4,
                  Index5,
                  Index6,
                  Index7,
                  Index8);
    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceIndex32x16(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONG Key1,
    ULONG Key2,
    ULONG Key3,
    ULONG Key4,
    ULONG Key5,
    ULONG Key6,
    ULONG Key7,
    ULONG Key8,
    ULONG Key9,
    ULONG Key10,
    ULONG Key11,
    ULONG Key12,
    ULONG Key13,
    ULONG Key14,
    ULONG Key15,
    ULONG Key16,
    PULONG Index1,
    PULONG Index2,
    PULONG Index3,
    PULONG Index4,
    PULONG Index5,
    PULONG Index6,
    PULONG Index7,
    PULONG Index8,
    PULONG Index9,
    PULONG Index10,
    PULONG Index11,
    PULONG Index12,
    PULONG Index13,
    PULONG Index14,
    PULONG Index15,
    PULONG Index16
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX32X16_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2) ||
        !ARGUMENT_PRESENT(Index3) ||
        !ARGUMENT_PRESENT(Index4) ||
        !ARGUMENT_PRESENT(Index5) ||
        !ARGUMENT_PRESENT(Index6) ||
        !ARGUMENT_PRESENT(Index7) ||
        !ARGUMENT_PRESENT(Index8) ||
        !ARGUMENT_PRESENT(Index9) ||
        !ARGUMENT_PRESENT(Index10) ||
        !ARGUMENT_PRESENT(Index11) ||
        !ARGUMENT_PRESENT(Index12) ||
        !ARGUMENT_PRESENT(Index13) ||
        !ARGUMENT_PRESENT(Index14) ||
        !ARGUMENT_PRESENT(Index15) ||
        !ARGUMENT_PRESENT(Index16)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index32x16Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX32X16_FUNCTION)JitState->Index32x16Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1,
                  Key2,
                  Key3,
                  Key4,
                  Key5,
                  Key6,
                  Key7,
                  Key8,
                  Key9,
                  Key10,
                  Key11,
                  Key12,
                  Key13,
                  Key14,
                  Key15,
                  Key16,
                  Index1,
                  Index2,
                  Index3,
                  Index4,
                  Index5,
                  Index6,
                  Index7,
                  Index8,
                  Index9,
                  Index10,
                  Index11,
                  Index12,
                  Index13,
                  Index14,
                  Index15,
                  Index16);
    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceIndex64x2(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONGLONG Key1,
    ULONGLONG Key2,
    PULONG Index1,
    PULONG Index2
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX64X2_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index64x2Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX64X2_FUNCTION)JitState->Index64x2Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1, Key2, Index1, Index2);
    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceIndex64x4(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONGLONG Key1,
    ULONGLONG Key2,
    ULONGLONG Key3,
    ULONGLONG Key4,
    PULONG Index1,
    PULONG Index2,
    PULONG Index3,
    PULONG Index4
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX64X4_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2) ||
        !ARGUMENT_PRESENT(Index3) ||
        !ARGUMENT_PRESENT(Index4)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index64x4Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX64X4_FUNCTION)JitState->Index64x4Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1,
                  Key2,
                  Key3,
                  Key4,
                  Index1,
                  Index2,
                  Index3,
                  Index4);
    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceIndex64x8(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONGLONG Key1,
    ULONGLONG Key2,
    ULONGLONG Key3,
    ULONGLONG Key4,
    ULONGLONG Key5,
    ULONGLONG Key6,
    ULONGLONG Key7,
    ULONGLONG Key8,
    PULONG Index1,
    PULONG Index2,
    PULONG Index3,
    PULONG Index4,
    PULONG Index5,
    PULONG Index6,
    PULONG Index7,
    PULONG Index8
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX64X8_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2) ||
        !ARGUMENT_PRESENT(Index3) ||
        !ARGUMENT_PRESENT(Index4) ||
        !ARGUMENT_PRESENT(Index5) ||
        !ARGUMENT_PRESENT(Index6) ||
        !ARGUMENT_PRESENT(Index7) ||
        !ARGUMENT_PRESENT(Index8)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index64x8Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX64X8_FUNCTION)JitState->Index64x8Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1,
                  Key2,
                  Key3,
                  Key4,
                  Key5,
                  Key6,
                  Key7,
                  Key8,
                  Index1,
                  Index2,
                  Index3,
                  Index4,
                  Index5,
                  Index6,
                  Index7,
                  Index8);
    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceIndex64x16(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    ULONGLONG Key1,
    ULONGLONG Key2,
    ULONGLONG Key3,
    ULONGLONG Key4,
    ULONGLONG Key5,
    ULONGLONG Key6,
    ULONGLONG Key7,
    ULONGLONG Key8,
    ULONGLONG Key9,
    ULONGLONG Key10,
    ULONGLONG Key11,
    ULONGLONG Key12,
    ULONGLONG Key13,
    ULONGLONG Key14,
    ULONGLONG Key15,
    ULONGLONG Key16,
    PULONG Index1,
    PULONG Index2,
    PULONG Index3,
    PULONG Index4,
    PULONG Index5,
    PULONG Index6,
    PULONG Index7,
    PULONG Index8,
    PULONG Index9,
    PULONG Index10,
    PULONG Index11,
    PULONG Index12,
    PULONG Index13,
    PULONG Index14,
    PULONG Index15,
    PULONG Index16
    )
{
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_TABLE_JIT JitState;
    PPH_JIT_INDEX64X16_FUNCTION IndexFunction;

    if (!ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Index1) ||
        !ARGUMENT_PRESENT(Index2) ||
        !ARGUMENT_PRESENT(Index3) ||
        !ARGUMENT_PRESENT(Index4) ||
        !ARGUMENT_PRESENT(Index5) ||
        !ARGUMENT_PRESENT(Index6) ||
        !ARGUMENT_PRESENT(Index7) ||
        !ARGUMENT_PRESENT(Index8) ||
        !ARGUMENT_PRESENT(Index9) ||
        !ARGUMENT_PRESENT(Index10) ||
        !ARGUMENT_PRESENT(Index11) ||
        !ARGUMENT_PRESENT(Index12) ||
        !ARGUMENT_PRESENT(Index13) ||
        !ARGUMENT_PRESENT(Index14) ||
        !ARGUMENT_PRESENT(Index15) ||
        !ARGUMENT_PRESENT(Index16)) {
        return E_POINTER;
    }

    Table = Jit->Table;
    if (!ARGUMENT_PRESENT(Table)) {
        return E_POINTER;
    }

    JitState = Table->Jit;
    if (!ARGUMENT_PRESENT(JitState) ||
        !JitState->Flags.Valid ||
        !JitState->Flags.Index64x16Compiled) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction = (PPH_JIT_INDEX64X16_FUNCTION)JitState->Index64x16Function;
    if (!ARGUMENT_PRESENT(IndexFunction)) {
        return PH_E_NOT_IMPLEMENTED;
    }

    IndexFunction(Key1,
                  Key2,
                  Key3,
                  Key4,
                  Key5,
                  Key6,
                  Key7,
                  Key8,
                  Key9,
                  Key10,
                  Key11,
                  Key12,
                  Key13,
                  Key14,
                  Key15,
                  Key16,
                  Index1,
                  Index2,
                  Index3,
                  Index4,
                  Index5,
                  Index6,
                  Index7,
                  Index8,
                  Index9,
                  Index10,
                  Index11,
                  Index12,
                  Index13,
                  Index14,
                  Index15,
                  Index16);
    return S_OK;
}

_Use_decl_annotations_
static
HRESULT
PerfectHashTableJitInterfaceGetInfo(
    PPERFECT_HASH_TABLE_JIT_INTERFACE Jit,
    PPERFECT_HASH_TABLE_JIT_INFO Info
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

    ZeroStructPointerInline(Info);
    Info->SizeOfStruct = sizeof(*Info);
    Info->Flags.AsULong = JitState->Flags.AsULong;
    Info->JitMaxIsa = JitState->JitMaxIsa;
    memcpy(Info->TargetCpu, JitState->TargetCpu, sizeof(Info->TargetCpu));
    memcpy(Info->TargetFeatures,
           JitState->TargetFeatures,
           sizeof(Info->TargetFeatures));

    return S_OK;
}
