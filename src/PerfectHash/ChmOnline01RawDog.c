/*++

Copyright (c) 2025 Trent Nelson <trent@trent.me>

Module Name:

    ChmOnline01RawDog.c

Abstract:

    This module implements the RawDog JIT backend for CHM01 online tables.

--*/

#include "stdafx.h"

#if defined(PH_HAS_RAWDOG_JIT)

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "ChmOnline01.h"

#if defined(PH_RAWDOG_X64)
#include <immintrin.h>
#include "PerfectHashJitRawDogMultiplyShiftR_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftR16_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRIndex32x4_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftR16Index32x4_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRIndex32x4Avx2_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftR16Index32x4Avx2_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRIndex32x8_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftR16Index32x8_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRIndex32x16_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftR16Index32x16_x64.h"
#if defined(PH_LINUX)
#include "PerfectHashJitRawDogMultiplyShiftRIndex32x16Avx512_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftR16Index32x16Avx512_x64.h"
#endif
#include "PerfectHashJitRawDogMultiplyShiftRX_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRX16_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRXIndex32x4_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRX16Index32x4_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRXIndex32x4Avx2_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRX16Index32x4Avx2_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRXIndex32x8_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRX16Index32x8_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRXIndex32x16_x64.h"
#include "PerfectHashJitRawDogMultiplyShiftRX16Index32x16_x64.h"
#include "PerfectHashJitRawDogMulshrolate1RX_x64.h"
#include "PerfectHashJitRawDogMulshrolate1RX16_x64.h"
#include "PerfectHashJitRawDogMulshrolate1RXIndex32x4_x64.h"
#include "PerfectHashJitRawDogMulshrolate1RX16Index32x4_x64.h"
#include "PerfectHashJitRawDogMulshrolate1RXIndex32x4Avx2_x64.h"
#include "PerfectHashJitRawDogMulshrolate1RX16Index32x4Avx2_x64.h"
#include "PerfectHashJitRawDogMulshrolate1RXIndex32x8_x64.h"
#include "PerfectHashJitRawDogMulshrolate1RX16Index32x8_x64.h"
#include "PerfectHashJitRawDogMulshrolate1RXIndex32x16_x64.h"
#include "PerfectHashJitRawDogMulshrolate1RX16Index32x16_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RX_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RXIndex32x4_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16Index32x4_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RXIndex32x4Avx2_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16Index32x4Avx2_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RXAvx2_v1_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RXAvx2_v2_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RXAvx2_v3_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RXAvx512_v1_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RXAvx512_v2_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16Avx2_v1_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16Avx2_v2_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16Avx2_v3_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16Avx512_v1_x64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16Avx512_v2_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RX_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RXAvx2_v1_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RXAvx2_v2_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RXAvx2_v3_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RXAvx512_v1_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RXAvx512_v2_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RX16_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RX16Avx2_v1_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RX16Avx2_v2_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RX16Avx2_v3_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RX16Avx512_v1_x64.h"
#include "PerfectHashJitRawDogMulshrolate3RX16Avx512_v2_x64.h"
#elif defined(PH_RAWDOG_ARM64)
#include "PerfectHashJitRawDogMultiplyShiftR_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftR16_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftRX_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftRX16_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftRIndex32x4_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftR16Index32x4_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftRXIndex32x4_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftRX16Index32x4_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftRIndex32x8_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftRIndex32x16_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftR16Index32x8_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftR16Index32x16_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftRXIndex32x8_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftRXIndex32x16_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftRX16Index32x8_arm64.h"
#include "PerfectHashJitRawDogMultiplyShiftRX16Index32x16_arm64.h"
#include "PerfectHashJitRawDogMulshrolate1RX_arm64.h"
#include "PerfectHashJitRawDogMulshrolate1RX16_arm64.h"
#include "PerfectHashJitRawDogMulshrolate1RXIndex32x4_arm64.h"
#include "PerfectHashJitRawDogMulshrolate1RX16Index32x4_arm64.h"
#include "PerfectHashJitRawDogMulshrolate1RXIndex32x8_arm64.h"
#include "PerfectHashJitRawDogMulshrolate1RX16Index32x8_arm64.h"
#include "PerfectHashJitRawDogMulshrolate1RXIndex32x16_arm64.h"
#include "PerfectHashJitRawDogMulshrolate1RX16Index32x16_arm64.h"
#include "PerfectHashJitRawDogMulshrolate2RX_arm64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16_arm64.h"
#include "PerfectHashJitRawDogMulshrolate3RX_arm64.h"
#include "PerfectHashJitRawDogMulshrolate3RX16_arm64.h"
#include "PerfectHashJitRawDogMulshrolate2RXIndex32x4_arm64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16Index32x4_arm64.h"
#include "PerfectHashJitRawDogMulshrolate3RXIndex32x4_arm64.h"
#include "PerfectHashJitRawDogMulshrolate3RX16Index32x4_arm64.h"
#include "PerfectHashJitRawDogMulshrolate2RXIndex32x8_arm64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16Index32x8_arm64.h"
#include "PerfectHashJitRawDogMulshrolate2RXIndex32x16_arm64.h"
#include "PerfectHashJitRawDogMulshrolate2RX16Index32x16_arm64.h"
#include "PerfectHashJitRawDogMulshrolate3RXIndex32x8_arm64.h"
#include "PerfectHashJitRawDogMulshrolate3RX16Index32x8_arm64.h"
#include "PerfectHashJitRawDogMulshrolate3RXIndex32x16_arm64.h"
#include "PerfectHashJitRawDogMulshrolate3RX16Index32x16_arm64.h"
#endif

#define RAWDOG_SENTINEL_ASSIGNED    0xA1A1A1A1A1A1A1A1ULL
#define RAWDOG_SENTINEL_SEED1       0xB1B1B1B1B1B1B1B1ULL
#define RAWDOG_SENTINEL_SEED2       0xC1C1C1C1C1C1C1C1ULL
#define RAWDOG_SENTINEL_SEED3_BYTE1 0xD1D1D1D1D1D1D1D1ULL
#define RAWDOG_SENTINEL_SEED3_BYTE2 0xE1E1E1E1E1E1E1E1ULL
#define RAWDOG_SENTINEL_SEED3_BYTE3 0xD2D2D2D2D2D2D2D2ULL
#define RAWDOG_SENTINEL_SEED4       0xB2B2B2B2B2B2B2B2ULL
#define RAWDOG_SENTINEL_HASH_MASK   0xF1F1F1F1F1F1F1F1ULL
#define RAWDOG_SENTINEL_INDEX_MASK  0x2121212121212121ULL
#define RAWDOG_SENTINEL_IMM32_SEED1       0xB1C2D3E4ULL
#define RAWDOG_SENTINEL_IMM32_SEED2       0xC2D3E4F5ULL
#define RAWDOG_SENTINEL_IMM32_SEED3_BYTE1 0xD31BEF20ULL
#define RAWDOG_SENTINEL_IMM32_SEED3_BYTE2 0xE4F50617ULL
#define RAWDOG_SENTINEL_IMM32_HASH_MASK   0xF5061728ULL
#define RAWDOG_SENTINEL_IMM32_INDEX_MASK  0x06172839ULL
#define RAWDOG_IMM8_TABLE_MAGIC           0x8C4B2A1D9F573E61ULL

#if defined(PH_RAWDOG_X64)
static
BOOLEAN
RawDogOsAvxEnabled(
    _In_opt_ PRTL Rtl
    )
{
    if (!ARGUMENT_PRESENT(Rtl) || Rtl->CpuFeatures.OSXSAVE == FALSE) {
        return FALSE;
    }

#if defined(_M_AMD64) || defined(_M_X64)
    return ((_xgetbv(0) & 0x6) == 0x6);
#else
    return FALSE;
#endif
}

static
BOOLEAN
RawDogOsAvx512Enabled(
    _In_opt_ PRTL Rtl
    )
{
    if (!ARGUMENT_PRESENT(Rtl) || Rtl->CpuFeatures.OSXSAVE == FALSE) {
        return FALSE;
    }

#if defined(_M_AMD64) || defined(_M_X64)
    return ((_xgetbv(0) & 0xE6) == 0xE6);
#else
    return FALSE;
#endif
}

static
BOOLEAN
RawDogHostHasAvx(
    _In_opt_ PRTL Rtl
    )
{
    if (!ARGUMENT_PRESENT(Rtl) ||
        Rtl->CpuFeatures.AVX == FALSE ||
        !RawDogOsAvxEnabled(Rtl)) {
        return FALSE;
    }

    return TRUE;
}

DECLSPEC_NOINLINE
static
VOID
RawDogProbeAvx2(
    VOID
    )
{
    int data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i values = _mm256_i32gather_epi32(data, indices, 4);
    volatile __m256i sink = values;
    (void)sink;
}

static
BOOLEAN
RawDogHostHasAvx2(
    _In_opt_ PRTL Rtl
    )
{
    if (!ARGUMENT_PRESENT(Rtl) ||
        Rtl->CpuFeatures.AVX2 == FALSE ||
        Rtl->CpuFeatures.AVX == FALSE ||
        !RawDogOsAvxEnabled(Rtl)) {
        return FALSE;
    }

    TRY_AVX2 {
        RawDogProbeAvx2();
    } CATCH_EXCEPTION_ILLEGAL_INSTRUCTION {
        return FALSE;
    }

    return TRUE;
}

static
BOOLEAN
RawDogHostHasAvx512(
    _In_opt_ PRTL Rtl
    )
{
    if (!ARGUMENT_PRESENT(Rtl) ||
        Rtl->CpuFeatures.AVX512F == FALSE ||
        !RawDogOsAvx512Enabled(Rtl)) {
        return FALSE;
    }

    return TRUE;
}
#endif

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

typedef struct _RAW_DOG_PATCH_ENTRY {
    ULONGLONG Sentinel;
    ULONGLONG Value;
    PCSZ Name;
    ULONG SizeInBytes;
} RAW_DOG_PATCH_ENTRY;

#if defined(PH_RAWDOG_X64)
static const CHAR RawDogTargetCpu[] = "rawdog-x64";
#elif defined(PH_RAWDOG_ARM64)
static const CHAR RawDogTargetCpu[] = "rawdog-arm64";
#else
static const CHAR RawDogTargetCpu[] = "";
#endif

static
ULONG
GetRawDogVectorVersion(VOID)
{
    const char *Value = NULL;

#if defined(PH_WINDOWS)
    char Buffer[16] = { 0 };
    size_t Length = 0;

    if (getenv_s(&Length,
                 Buffer,
                 sizeof(Buffer),
                 "PH_RAWDOG_VECTOR_VERSION") == 0 &&
        Length > 0) {
        Buffer[sizeof(Buffer) - 1] = '\0';
        Value = Buffer;
    }
#else
    Value = getenv("PH_RAWDOG_VECTOR_VERSION");
#endif
    if (!Value || *Value == '\0') {
        // TODO: Consider per-ISA defaults once AVX-512 v3 is available.
        return 3;
    }

    if (*Value == '3') {
        return 3;
    }

    if (*Value == '2') {
        return 2;
    }

    return 3;
}

#if defined(PH_RAWDOG_X64)
static
BOOLEAN
RawDogTraceCpuFeaturesEnabled(VOID)
{
    const char *Value = NULL;

#if defined(PH_WINDOWS)
    char Buffer[16] = { 0 };
    size_t Length = 0;

    if (getenv_s(&Length,
                 Buffer,
                 sizeof(Buffer),
                 "PH_RAWDOG_TRACE_CPU") == 0 &&
        Length > 0) {
        Buffer[sizeof(Buffer) - 1] = '\0';
        Value = Buffer;
    }
#else
    Value = getenv("PH_RAWDOG_TRACE_CPU");
#endif

    if (!Value || *Value == '\0') {
        return FALSE;
    }

    if (*Value == '1' || *Value == 'y' || *Value == 'Y' ||
        *Value == 't' || *Value == 'T') {
        return TRUE;
    }

    return FALSE;
}

static
VOID
RawDogTraceCpuFeatures(
    _In_opt_ PRTL Rtl,
    _In_ PERFECT_HASH_JIT_MAX_ISA_ID RequestedIsa,
    _In_ ULONG VectorVersion,
    _In_ BOOLEAN HostHasAvx2,
    _In_ BOOLEAN HostHasAvx512,
    _In_ BOOLEAN AllowAvx2,
    _In_ BOOLEAN AllowAvx512
    )
{
    unsigned long long Xcr0 = 0;
    ULONG Avx = 0;
    ULONG Avx2 = 0;
    ULONG Avx512 = 0;
    ULONG Osxsave = 0;

    if (!RawDogTraceCpuFeaturesEnabled()) {
        return;
    }

    if (ARGUMENT_PRESENT(Rtl)) {
        Avx = (ULONG)Rtl->CpuFeatures.AVX;
        Avx2 = (ULONG)Rtl->CpuFeatures.AVX2;
        Avx512 = (ULONG)Rtl->CpuFeatures.AVX512F;
        Osxsave = (ULONG)Rtl->CpuFeatures.OSXSAVE;
        if (Rtl->CpuFeatures.OSXSAVE != FALSE) {
            Xcr0 = (unsigned long long)_xgetbv(0);
        }
    }

    fprintf(stderr,
            "RawDogCpuFeatures: AVX=%lu AVX2=%lu AVX512F=%lu OSXSAVE=%lu "
            "XCR0=0x%llx HostHasAvx2=%lu HostHasAvx512=%lu AllowAvx2=%lu "
            "AllowAvx512=%lu JitMaxIsa=%lu VectorVersion=%lu\n",
            (unsigned long)Avx,
            (unsigned long)Avx2,
            (unsigned long)Avx512,
            (unsigned long)Osxsave,
            Xcr0,
            (unsigned long)HostHasAvx2,
            (unsigned long)HostHasAvx512,
            (unsigned long)AllowAvx2,
            (unsigned long)AllowAvx512,
            (unsigned long)RequestedIsa,
            (unsigned long)VectorVersion);
    fflush(stderr);
}
#endif

static
VOID
CopyRawDogString(
    _Out_writes_(BufferSize) PCHAR Dest,
    _In_ SIZE_T BufferSize,
    _In_opt_ const char *Source
    )
{
    if (!Dest || BufferSize == 0) {
        return;
    }

    if (!Source || *Source == '\0') {
        Dest[0] = '\0';
        return;
    }

#if defined(PH_WINDOWS)
    strncpy_s(Dest, BufferSize, Source, _TRUNCATE);
#else
    strncpy(Dest, Source, BufferSize - 1);
    Dest[BufferSize - 1] = '\0';
#endif
}

static
HRESULT
EnsureRawDogAssigned16Padding(
    _In_ PPERFECT_HASH_TABLE Table,
    _Inout_ PPERFECT_HASH_TABLE_JIT Jit,
    _Out_ PULONGLONG Assigned
    )
{
    PTABLE_INFO_ON_DISK TableInfo;
    ULONGLONG SizeInBytes;
    ULONGLONG AllocSizeInBytes;
    PVOID Buffer;
    PALLOCATOR Allocator;

    if (!ARGUMENT_PRESENT(Table) ||
        !ARGUMENT_PRESENT(Jit) ||
        !ARGUMENT_PRESENT(Assigned)) {
        return E_POINTER;
    }

    if (ARGUMENT_PRESENT(Jit->Assigned16Padded)) {
        *Assigned = (ULONGLONG)(ULONG_PTR)Jit->Assigned16Padded;
        return S_OK;
    }

    if (!ARGUMENT_PRESENT(Table->Assigned16)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    TableInfo = Table->TableInfoOnDisk;
    if (!ARGUMENT_PRESENT(TableInfo)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    SizeInBytes = (
        TableInfo->NumberOfTableElements.QuadPart * sizeof(USHORT)
    );
    AllocSizeInBytes = SizeInBytes + sizeof(ULONG);

    Allocator = Table->Allocator;
    Buffer = Allocator->Vtbl->Calloc(Allocator, 1, AllocSizeInBytes);
    if (!Buffer) {
        return E_OUTOFMEMORY;
    }

    memcpy(Buffer, Table->Assigned16, SizeInBytes);

    Jit->Assigned16Padded = Buffer;
    Jit->Assigned16PaddedSize = AllocSizeInBytes;
    *Assigned = (ULONGLONG)(ULONG_PTR)Buffer;

    return S_OK;
}

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
    _In_ ULONG SizeInBytes,
    _In_ PCSZ Name
    )
{
    SIZE_T Offset;
    ULONG Matches = 0;
    ULONG Size = SizeInBytes ? SizeInBytes : sizeof(ULONGLONG);

    if (Size != 1 && Size != 2 && Size != 4 && Size != 8) {
        UNREFERENCED_PARAMETER(Name);
        return E_INVALIDARG;
    }

    for (Offset = 0; Offset + Size <= CodeSize; Offset++) {
        if (memcmp(Code + Offset, &Sentinel, Size) == 0) {
            memcpy(Code + Offset, &Value, Size);
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
PatchRawDogImm8Table(
    _Inout_ PBYTE Code,
    _In_ SIZE_T CodeSize,
    _In_ ULONG Seed3Byte1,
    _In_ ULONG Seed3Byte2
    )
{
    SIZE_T Offset;
    SIZE_T MatchOffset = 0;
    ULONG Matches = 0;
    ULONGLONG Magic = RAWDOG_IMM8_TABLE_MAGIC;
    ULONG Count = 0;
    ULONG EntryOffset = 0;
    ULONG PatchOffset = 0;
    ULONG RequiredCount = 2;

    if (!ARGUMENT_PRESENT(Code) || CodeSize < sizeof(Magic)) {
        return S_OK;
    }

    for (Offset = 0; Offset + sizeof(Magic) <= CodeSize; Offset++) {
        if (memcmp(Code + Offset, &Magic, sizeof(Magic)) == 0) {
            MatchOffset = Offset;
            Matches++;
            if (Matches > 1) {
                return PH_E_INVARIANT_CHECK_FAILED;
            }
        }
    }

    if (Matches == 0) {
        return S_OK;
    }

    if (MatchOffset + sizeof(Magic) + sizeof(ULONG) * 2 > CodeSize) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    memcpy(&Count, Code + MatchOffset + sizeof(Magic), sizeof(Count));
    if (Count < RequiredCount || (Count & 1)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    if (MatchOffset + sizeof(Magic) + sizeof(ULONG) * 2 +
        (sizeof(ULONG) * Count) > CodeSize) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    EntryOffset = (ULONG)(MatchOffset + sizeof(Magic) + sizeof(ULONG) * 2);
    for (ULONG EntryIndex = 0; EntryIndex < Count; EntryIndex++) {
        const ULONG Seed3Byte = (EntryIndex & 1) ? Seed3Byte2 : Seed3Byte1;

        memcpy(&PatchOffset,
               Code + EntryOffset + (sizeof(ULONG) * EntryIndex),
               sizeof(PatchOffset));
        if (PatchOffset >= CodeSize) {
            return PH_E_INVARIANT_CHECK_FAILED;
        }
        Code[PatchOffset] = (BYTE)(Seed3Byte & 0xffu);
    }

    return S_OK;
}

static
HRESULT
CreateRawDogCode(
    _In_ PBYTE SourceCode,
    _In_ SIZE_T CodeSize,
    _In_reads_(EntryCount) RAW_DOG_PATCH_ENTRY *Entries,
    _In_ ULONG EntryCount,
    _In_ ULONG Seed3Byte1,
    _In_ ULONG Seed3Byte2,
    _Out_ PBYTE *CodeOut
    )
{
    BOOL Success;
    DWORD OldProtect = 0;
    SIZE_T AllocSize;
    PBYTE Code;
    ULONG EntryIndex;
    HRESULT Result;

    if (!ARGUMENT_PRESENT(SourceCode) ||
        CodeSize == 0 ||
        !ARGUMENT_PRESENT(Entries) ||
        EntryCount == 0 ||
        !ARGUMENT_PRESENT(CodeOut)) {
        return E_INVALIDARG;
    }

    *CodeOut = NULL;
    AllocSize = ALIGN_UP(CodeSize, PAGE_SIZE);

    Code = (PBYTE)VirtualAlloc(NULL,
                               AllocSize,
                               MEM_RESERVE | MEM_COMMIT,
                               PAGE_READWRITE);
    if (!Code) {
        SYS_ERROR(VirtualAlloc);
        return E_OUTOFMEMORY;
    }

    memcpy(Code, SourceCode, CodeSize);

    for (EntryIndex = 0; EntryIndex < EntryCount; EntryIndex++) {
        Result = PatchRawDogSentinel(Code,
                                     CodeSize,
                                     Entries[EntryIndex].Sentinel,
                                     Entries[EntryIndex].Value,
                                     Entries[EntryIndex].SizeInBytes,
                                     Entries[EntryIndex].Name);
        if (FAILED(Result)) {
            VirtualFree(Code, 0, MEM_RELEASE);
            return Result;
        }
    }

    Result = PatchRawDogImm8Table(Code,
                                  CodeSize,
                                  Seed3Byte1,
                                  Seed3Byte2);
    if (FAILED(Result)) {
        VirtualFree(Code, 0, MEM_RELEASE);
        return Result;
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

    *CodeOut = Code;
    return S_OK;
}

static
VOID
FreeRawDogCode(
    _Inout_ PVOID *CodePointer
    )
{
    if (ARGUMENT_PRESENT(CodePointer) && *CodePointer) {
        VirtualFree(*CodePointer, 0, MEM_RELEASE);
        *CodePointer = NULL;
    }
}

static
HRESULT
CompileChm01IndexJitRawDog(
    _In_ PPERFECT_HASH_TABLE Table,
    _Inout_ PPERFECT_HASH_TABLE_JIT Jit,
    _In_ PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags
    )
{
    PTABLE_INFO_ON_DISK TableInfo;
    ULONG_BYTES Seed3Bytes;
    ULONGLONG Assigned;
    SIZE_T CodeSize;
    PBYTE Code;
    PBYTE SourceCode;
    RAW_DOG_PATCH_ENTRY Entries[8];
    ULONG EntryCount = 0;
    BOOLEAN UseAssigned16;
    HRESULT Result;

#if defined(PH_RAWDOG_ARM64)
    if ((Table->HashFunctionId != PerfectHashHashMultiplyShiftRFunctionId &&
         Table->HashFunctionId != PerfectHashHashMultiplyShiftRXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate1RXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate2RXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate3RXFunctionId) ||
        Table->MaskFunctionId != PerfectHashAndMaskFunctionId) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (Table->HashFunctionId == PerfectHashHashMulshrolate3RXFunctionId) {
        if (CompileIndex32x4 && !CompileIndex32x8 && !CompileIndex32x16) {
            return PH_E_NOT_IMPLEMENTED;
        }
        CompileIndex32x4 = FALSE;
    }
#else
    if ((Table->HashFunctionId != PerfectHashHashMultiplyShiftRFunctionId &&
         Table->HashFunctionId != PerfectHashHashMultiplyShiftRXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate1RXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate2RXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate3RXFunctionId) ||
        Table->MaskFunctionId != PerfectHashAndMaskFunctionId) {
        return PH_E_NOT_IMPLEMENTED;
    }
#endif

    UseAssigned16 = (Table->State.UsingAssigned16 != FALSE);
#if defined(PH_RAWDOG_ARM64)
    if (UseAssigned16 &&
        Table->HashFunctionId != PerfectHashHashMultiplyShiftRFunctionId &&
        Table->HashFunctionId != PerfectHashHashMultiplyShiftRXFunctionId &&
        Table->HashFunctionId != PerfectHashHashMulshrolate1RXFunctionId &&
        Table->HashFunctionId != PerfectHashHashMulshrolate2RXFunctionId &&
        Table->HashFunctionId != PerfectHashHashMulshrolate3RXFunctionId) {
        return PH_E_NOT_IMPLEMENTED;
    }
#endif

    if (UseAssigned16) {
        if (!ARGUMENT_PRESENT(Table->Assigned16)) {
            return PH_E_INVARIANT_CHECK_FAILED;
        }
    } else if (!ARGUMENT_PRESENT(Table->TableData)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    TableInfo = Table->TableInfoOnDisk;
    if (!ARGUMENT_PRESENT(TableInfo)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    Seed3Bytes.AsULong = TableInfo->Seed3;
    if (UseAssigned16) {
        Result = EnsureRawDogAssigned16Padding(Table, Jit, &Assigned);
        if (FAILED(Result)) {
            return Result;
        }
    } else {
        Assigned = (ULONGLONG)(ULONG_PTR)(PVOID)Table->TableData;
    }

    SourceCode = NULL;
    CodeSize = 0;

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4061)
#endif
    switch (Table->HashFunctionId) {
        case PerfectHashHashMultiplyShiftRFunctionId:
#if defined(PH_RAWDOG_ARM64)
            if (UseAssigned16) {
                SourceCode = (PBYTE)PerfectHashJitRawDogMultiplyShiftR16_arm64;
                CodeSize = sizeof(PerfectHashJitRawDogMultiplyShiftR16_arm64);
            } else {
                SourceCode = (PBYTE)PerfectHashJitRawDogMultiplyShiftR_arm64;
                CodeSize = sizeof(PerfectHashJitRawDogMultiplyShiftR_arm64);
            }
#else
            if (UseAssigned16) {
                SourceCode = (PBYTE)PerfectHashJitRawDogMultiplyShiftR16_x64;
                CodeSize = sizeof(PerfectHashJitRawDogMultiplyShiftR16_x64);
            } else {
                SourceCode = (PBYTE)PerfectHashJitRawDogMultiplyShiftR_x64;
                CodeSize = sizeof(PerfectHashJitRawDogMultiplyShiftR_x64);
            }
#endif
#if defined(PH_RAWDOG_X64)
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_ASSIGNED,
                Assigned,
                "Assigned",
                sizeof(ULONGLONG)
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_SEED1,
                (ULONGLONG)TableInfo->Seed1,
                "Seed1Imm",
                sizeof(ULONG)
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_SEED2,
                (ULONGLONG)TableInfo->Seed2,
                "Seed2Imm",
                sizeof(ULONG)
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_HASH_MASK,
                (ULONGLONG)Table->HashMask,
                "HashMaskImm",
                sizeof(ULONG)
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_INDEX_MASK,
                (ULONGLONG)Table->IndexMask,
                "IndexMaskImm",
                sizeof(ULONG)
            };
#else
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_ASSIGNED,
                Assigned,
                "Assigned"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED1,
                (ULONGLONG)TableInfo->Seed1,
                "Seed1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED2,
                (ULONGLONG)TableInfo->Seed2,
                "Seed2"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE1,
                (ULONGLONG)Seed3Bytes.Byte1,
                "Seed3Byte1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE2,
                (ULONGLONG)Seed3Bytes.Byte2,
                "Seed3Byte2"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_HASH_MASK,
                (ULONGLONG)Table->HashMask,
                "HashMask"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_INDEX_MASK,
                (ULONGLONG)Table->IndexMask,
                "IndexMask"
            };
#endif
            break;

        case PerfectHashHashMultiplyShiftRXFunctionId:
#if defined(PH_RAWDOG_ARM64)
            if (UseAssigned16) {
                SourceCode = (PBYTE)PerfectHashJitRawDogMultiplyShiftRX16_arm64;
                CodeSize = sizeof(PerfectHashJitRawDogMultiplyShiftRX16_arm64);
            } else {
                SourceCode = (PBYTE)PerfectHashJitRawDogMultiplyShiftRX_arm64;
                CodeSize = sizeof(PerfectHashJitRawDogMultiplyShiftRX_arm64);
            }
#elif defined(PH_RAWDOG_X64)
            if (UseAssigned16) {
                SourceCode = (PBYTE)PerfectHashJitRawDogMultiplyShiftRX16_x64;
                CodeSize = sizeof(PerfectHashJitRawDogMultiplyShiftRX16_x64);
            } else {
                SourceCode = (PBYTE)PerfectHashJitRawDogMultiplyShiftRX_x64;
                CodeSize = sizeof(PerfectHashJitRawDogMultiplyShiftRX_x64);
            }
#else
            return PH_E_NOT_IMPLEMENTED;
#endif
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_ASSIGNED,
                Assigned,
                "Assigned"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED1,
                (ULONGLONG)TableInfo->Seed1,
                "Seed1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED2,
                (ULONGLONG)TableInfo->Seed2,
                "Seed2"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE1,
                (ULONGLONG)Seed3Bytes.Byte1,
                "Seed3Byte1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_INDEX_MASK,
                (ULONGLONG)Table->IndexMask,
                "IndexMask"
            };
            break;

        case PerfectHashHashMulshrolate1RXFunctionId:
#if defined(PH_RAWDOG_ARM64)
            if (UseAssigned16) {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate1RX16_arm64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate1RX16_arm64);
            } else {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate1RX_arm64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate1RX_arm64);
            }
#elif defined(PH_RAWDOG_X64)
            if (UseAssigned16) {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate1RX16_x64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate1RX16_x64);
            } else {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate1RX_x64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate1RX_x64);
            }
#endif
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_ASSIGNED,
                Assigned,
                "Assigned"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED1,
                (ULONGLONG)TableInfo->Seed1,
                "Seed1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED2,
                (ULONGLONG)TableInfo->Seed2,
                "Seed2"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE1,
                (ULONGLONG)Seed3Bytes.Byte1,
                "Seed3Byte1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE2,
                (ULONGLONG)Seed3Bytes.Byte2,
                "Seed3Byte2"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_INDEX_MASK,
                (ULONGLONG)Table->IndexMask,
                "IndexMask"
            };
            break;

        case PerfectHashHashMulshrolate2RXFunctionId:
#if defined(PH_RAWDOG_ARM64)
            if (UseAssigned16) {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate2RX16_arm64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate2RX16_arm64);
            } else {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate2RX_arm64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate2RX_arm64);
            }
#elif defined(PH_RAWDOG_X64)
            if (UseAssigned16) {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate2RX16_x64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate2RX16_x64);
            } else {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate2RX_x64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate2RX_x64);
            }
#endif
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_ASSIGNED,
                Assigned,
                "Assigned"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED1,
                (ULONGLONG)TableInfo->Seed1,
                "Seed1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED2,
                (ULONGLONG)TableInfo->Seed2,
                "Seed2"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE1,
                (ULONGLONG)Seed3Bytes.Byte1,
                "Seed3Byte1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE2,
                (ULONGLONG)Seed3Bytes.Byte2,
                "Seed3Byte2"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE3,
                (ULONGLONG)Seed3Bytes.Byte3,
                "Seed3Byte3"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_INDEX_MASK,
                (ULONGLONG)Table->IndexMask,
                "IndexMask"
            };
            break;

#if defined(PH_RAWDOG_ARM64)
        case PerfectHashHashMulshrolate3RXFunctionId:
            if (UseAssigned16) {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate3RX16_arm64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate3RX16_arm64);
            } else {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate3RX_arm64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate3RX_arm64);
            }
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_ASSIGNED,
                Assigned,
                "Assigned"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED1,
                (ULONGLONG)TableInfo->Seed1,
                "Seed1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED2,
                (ULONGLONG)TableInfo->Seed2,
                "Seed2"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE1,
                (ULONGLONG)Seed3Bytes.Byte1,
                "Seed3Byte1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE2,
                (ULONGLONG)Seed3Bytes.Byte2,
                "Seed3Byte2"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE3,
                (ULONGLONG)Seed3Bytes.Byte3,
                "Seed3Byte3"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED4,
                (ULONGLONG)TableInfo->Seed4,
                "Seed4"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_INDEX_MASK,
                (ULONGLONG)Table->IndexMask,
                "IndexMask"
            };
            break;

#endif

#if defined(PH_RAWDOG_X64)
        case PerfectHashHashMulshrolate3RXFunctionId:
            if (UseAssigned16) {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate3RX16_x64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate3RX16_x64);
            } else {
                SourceCode = (PBYTE)PerfectHashJitRawDogMulshrolate3RX_x64;
                CodeSize = sizeof(PerfectHashJitRawDogMulshrolate3RX_x64);
            }
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_ASSIGNED,
                Assigned,
                "Assigned"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED1,
                (ULONGLONG)TableInfo->Seed1,
                "Seed1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED2,
                (ULONGLONG)TableInfo->Seed2,
                "Seed2"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE1,
                (ULONGLONG)Seed3Bytes.Byte1,
                "Seed3Byte1"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE2,
                (ULONGLONG)Seed3Bytes.Byte2,
                "Seed3Byte2"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED3_BYTE3,
                (ULONGLONG)Seed3Bytes.Byte3,
                "Seed3Byte3"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_SEED4,
                (ULONGLONG)TableInfo->Seed4,
                "Seed4"
            };
            Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_INDEX_MASK,
                (ULONGLONG)Table->IndexMask,
                "IndexMask"
            };
            break;
#endif

        default:
            return PH_E_NOT_IMPLEMENTED;
    }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

    if (!SourceCode || CodeSize == 0) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    Result = CreateRawDogCode(SourceCode,
                              CodeSize,
                              Entries,
                              EntryCount,
                              Seed3Bytes.Byte1,
                              Seed3Bytes.Byte2,
                              &Code);
    if (FAILED(Result)) {
        return Result;
    }

    Jit->ExecutionEngine = Code;
    Jit->Index32Function = Code;
    Jit->Flags.Index32Compiled = TRUE;
    Jit->Flags.BackendRawDog = TRUE;
    Jit->JitMaxIsa = CompileFlags->JitMaxIsa;

    CopyRawDogString(Jit->TargetCpu,
                     sizeof(Jit->TargetCpu),
                     RawDogTargetCpu);

    return S_OK;
}

static
HRESULT
CompileChm01IndexVectorJitRawDog(
    _In_ PPERFECT_HASH_TABLE Table,
    _Inout_ PPERFECT_HASH_TABLE_JIT Jit,
    _In_ PPERFECT_HASH_TABLE_COMPILE_FLAGS CompileFlags,
    _In_ BOOLEAN CompileIndex32x4,
    _In_ BOOLEAN CompileIndex32x8,
    _In_ BOOLEAN CompileIndex32x16,
    _Out_opt_ PERFECT_HASH_JIT_MAX_ISA_ID *UsedIsa
    )
{
#if defined(PH_RAWDOG_MSR_ONLY)
#if !defined(PH_RAWDOG_X64)
    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(Jit);
    UNREFERENCED_PARAMETER(CompileFlags);
    UNREFERENCED_PARAMETER(CompileIndex32x4);
    UNREFERENCED_PARAMETER(CompileIndex32x8);
    UNREFERENCED_PARAMETER(CompileIndex32x16);

    if (ARGUMENT_PRESENT(UsedIsa)) {
        *UsedIsa = PerfectHashJitMaxIsaAuto;
    }

    return PH_E_NOT_IMPLEMENTED;
#else
    PTABLE_INFO_ON_DISK TableInfo;
    PRTL Rtl;
    ULONG_BYTES Seed3Bytes;
    ULONGLONG Assigned;
    RAW_DOG_PATCH_ENTRY Entries[8];
    ULONG EntryCount = 0;
    BOOLEAN UseAssigned16;
    BOOLEAN HostHasAvx;
    BOOLEAN HostHasAvx2;
    BOOLEAN HostHasAvx512;
    BOOLEAN AllowAvx;
    BOOLEAN AllowAvx2;
    BOOLEAN AllowAvx512;
    PERFECT_HASH_JIT_MAX_ISA_ID RequestedIsa;
    PERFECT_HASH_JIT_MAX_ISA_ID MaxIsa = PerfectHashJitMaxIsaAuto;
    ULONG VectorVersion = 0;
    PBYTE SourceCode = NULL;
    SIZE_T CodeSize = 0;
    PBYTE Code;
    HRESULT Result;

    if (!CompileIndex32x4 && !CompileIndex32x8 && !CompileIndex32x16) {
        if (ARGUMENT_PRESENT(UsedIsa)) {
            *UsedIsa = MaxIsa;
        }
        return S_OK;
    }

    if ((Table->HashFunctionId != PerfectHashHashMultiplyShiftRFunctionId &&
         Table->HashFunctionId != PerfectHashHashMultiplyShiftRXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate1RXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate2RXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate3RXFunctionId) ||
        Table->MaskFunctionId != PerfectHashAndMaskFunctionId) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (Table->HashFunctionId == PerfectHashHashMulshrolate2RXFunctionId ||
        Table->HashFunctionId == PerfectHashHashMulshrolate3RXFunctionId) {
        if (CompileIndex32x4 && !CompileIndex32x8 && !CompileIndex32x16) {
            return PH_E_NOT_IMPLEMENTED;
        }
        CompileIndex32x4 = FALSE;
    }

    UseAssigned16 = (Table->State.UsingAssigned16 != FALSE);
    if (UseAssigned16) {
        if (!ARGUMENT_PRESENT(Table->Assigned16)) {
            return PH_E_INVARIANT_CHECK_FAILED;
        }
    } else if (!ARGUMENT_PRESENT(Table->TableData)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    TableInfo = Table->TableInfoOnDisk;
    if (!ARGUMENT_PRESENT(TableInfo)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    Rtl = Table->Rtl;
    HostHasAvx = RawDogHostHasAvx(Rtl);
    HostHasAvx2 = RawDogHostHasAvx2(Rtl);
    HostHasAvx512 = RawDogHostHasAvx512(Rtl);
    RequestedIsa = (PERFECT_HASH_JIT_MAX_ISA_ID)CompileFlags->JitMaxIsa;
    AllowAvx512 = (RequestedIsa == PerfectHashJitMaxIsaAuto ||
                   RequestedIsa == PerfectHashJitMaxIsaAvx512);
    AllowAvx = (RequestedIsa == PerfectHashJitMaxIsaAuto ||
                RequestedIsa == PerfectHashJitMaxIsaAvx ||
                RequestedIsa == PerfectHashJitMaxIsaAvx2 ||
                RequestedIsa == PerfectHashJitMaxIsaAvx512);
    AllowAvx2 = (RequestedIsa == PerfectHashJitMaxIsaAuto ||
                 RequestedIsa == PerfectHashJitMaxIsaAvx2 ||
                 RequestedIsa == PerfectHashJitMaxIsaAvx512);
    VectorVersion = GetRawDogVectorVersion();
#if defined(PH_RAWDOG_X64)
    RawDogTraceCpuFeatures(Rtl,
                           RequestedIsa,
                           VectorVersion,
                           HostHasAvx2,
                           HostHasAvx512,
                           AllowAvx2,
                           AllowAvx512);
#endif

    Seed3Bytes.AsULong = TableInfo->Seed3;
    if (UseAssigned16) {
        Result = EnsureRawDogAssigned16Padding(Table, Jit, &Assigned);
        if (FAILED(Result)) {
            return Result;
        }
    } else {
        Assigned = (ULONGLONG)(ULONG_PTR)(PVOID)Table->TableData;
    }

    Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_ASSIGNED,
        Assigned,
        "Assigned"
    };
    Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_SEED1,
        (ULONGLONG)TableInfo->Seed1,
        "Seed1"
    };
    Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_SEED2,
        (ULONGLONG)TableInfo->Seed2,
        "Seed2"
    };
    if (Table->HashFunctionId == PerfectHashHashMultiplyShiftRFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_HASH_MASK,
            (ULONGLONG)Table->HashMask,
            "HashMask"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else if (Table->HashFunctionId ==
               PerfectHashHashMulshrolate1RXFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else if (Table->HashFunctionId ==
               PerfectHashHashMulshrolate2RXFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE3,
            (ULONGLONG)Seed3Bytes.Byte3,
            "Seed3Byte3"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else if (Table->HashFunctionId ==
               PerfectHashHashMulshrolate3RXFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE3,
            (ULONGLONG)Seed3Bytes.Byte3,
            "Seed3Byte3"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED4,
            (ULONGLONG)TableInfo->Seed4,
            "Seed4"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    }

    if (CompileIndex32x16) {
        if (!AllowAvx512 || !HostHasAvx512) {
            return PH_E_NOT_IMPLEMENTED;
        }

        if (Table->HashFunctionId ==
            PerfectHashHashMultiplyShiftRFunctionId) {
            if (UseAssigned16) {
#if defined(PH_LINUX)
                if (VectorVersion >= 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMultiplyShiftR16Index32x16Avx512_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMultiplyShiftR16Index32x16Avx512_x64
                    );
                } else {
#endif
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMultiplyShiftR16Index32x16_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMultiplyShiftR16Index32x16_x64
                    );
#if defined(PH_LINUX)
                }
#endif
            } else {
#if defined(PH_LINUX)
                if (VectorVersion >= 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMultiplyShiftRIndex32x16Avx512_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMultiplyShiftRIndex32x16Avx512_x64
                    );
                } else {
#endif
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMultiplyShiftRIndex32x16_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMultiplyShiftRIndex32x16_x64
                    );
#if defined(PH_LINUX)
                }
#endif
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate1RXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate1RX16Index32x16_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate1RX16Index32x16_x64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate1RXIndex32x16_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate1RXIndex32x16_x64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate2RXFunctionId) {
            if (UseAssigned16) {
                if (VectorVersion >= 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate2RX16Avx512_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Avx512_v2_x64
                    );
                } else {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate2RX16Avx512_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Avx512_v1_x64
                    );
                }
            } else {
                if (VectorVersion >= 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate2RXAvx512_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RXAvx512_v2_x64
                    );
                } else {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate2RXAvx512_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RXAvx512_v1_x64
                    );
                }
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate3RXFunctionId) {
            if (UseAssigned16) {
                if (VectorVersion >= 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate3RX16Avx512_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate3RX16Avx512_v2_x64
                    );
                } else {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate3RX16Avx512_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate3RX16Avx512_v1_x64
                    );
                }
            } else {
                if (VectorVersion >= 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate3RXAvx512_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate3RXAvx512_v2_x64
                    );
                } else {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate3RXAvx512_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate3RXAvx512_v1_x64
                    );
                }
            }
        } else {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x16_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x16_x64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x16_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x16_x64
                );
            }
        }

        Result = CreateRawDogCode(SourceCode,
                                  CodeSize,
                                  Entries,
                                  EntryCount,
                                  Seed3Bytes.Byte1,
                                  Seed3Bytes.Byte2,
                                  &Code);
        if (FAILED(Result)) {
            return Result;
        }

        Jit->Index32x16Function = Code;
        Jit->Flags.Index32x16Compiled = TRUE;
        Jit->Flags.Index32x16Vector = TRUE;
        MaxIsa = PerfectHashJitMaxIsaAvx512;
    }

    if (CompileIndex32x8) {
        if (!AllowAvx2 || !HostHasAvx2) {
            return PH_E_NOT_IMPLEMENTED;
        }

        if (Table->HashFunctionId ==
            PerfectHashHashMultiplyShiftRFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftR16Index32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftR16Index32x8_x64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRIndex32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRIndex32x8_x64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate1RXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate1RX16Index32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate1RX16Index32x8_x64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate1RXIndex32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate1RXIndex32x8_x64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate2RXFunctionId) {
            if (UseAssigned16) {
                if (VectorVersion == 3) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate2RX16Avx2_v3_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Avx2_v3_x64
                    );
                } else if (VectorVersion == 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate2RX16Avx2_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Avx2_v2_x64
                    );
                } else {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate2RX16Avx2_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Avx2_v1_x64
                    );
                }
            } else {
                if (VectorVersion == 3) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate2RXAvx2_v3_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RXAvx2_v3_x64
                    );
                } else if (VectorVersion == 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate2RXAvx2_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RXAvx2_v2_x64
                    );
                } else {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate2RXAvx2_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RXAvx2_v1_x64
                    );
                }
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate3RXFunctionId) {
            if (UseAssigned16) {
                if (VectorVersion == 3) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate3RX16Avx2_v3_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate3RX16Avx2_v3_x64
                    );
                } else if (VectorVersion == 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate3RX16Avx2_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate3RX16Avx2_v2_x64
                    );
                } else {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate3RX16Avx2_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate3RX16Avx2_v1_x64
                    );
                }
            } else {
                if (VectorVersion == 3) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate3RXAvx2_v3_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate3RXAvx2_v3_x64
                    );
                } else if (VectorVersion == 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate3RXAvx2_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate3RXAvx2_v2_x64
                    );
                } else {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMulshrolate3RXAvx2_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate3RXAvx2_v1_x64
                    );
                }
            }
        } else {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x8_x64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x8_x64
                );
            }
        }

        Result = CreateRawDogCode(SourceCode,
                                  CodeSize,
                                  Entries,
                                  EntryCount,
                                  Seed3Bytes.Byte1,
                                  Seed3Bytes.Byte2,
                                  &Code);
        if (FAILED(Result)) {
            return Result;
        }

        Jit->Index32x8Function = Code;
        Jit->Flags.Index32x8Compiled = TRUE;
        Jit->Flags.Index32x8Vector = TRUE;
        if (MaxIsa == PerfectHashJitMaxIsaAuto) {
            MaxIsa = PerfectHashJitMaxIsaAvx2;
        }
    }

    if (CompileIndex32x4) {
        BOOLEAN UseAvx2;
        RAW_DOG_PATCH_ENTRY *EntriesForX4 = Entries;
        ULONG EntryCountForX4 = EntryCount;
#if defined(PH_RAWDOG_X64)
        RAW_DOG_PATCH_ENTRY ImmediateEntries[8];
        ULONG ImmediateEntryCount = 0;
#endif

        if (!AllowAvx || !HostHasAvx) {
            return PH_E_NOT_IMPLEMENTED;
        }

        if ((RequestedIsa == PerfectHashJitMaxIsaAvx2 ||
             RequestedIsa == PerfectHashJitMaxIsaAvx512) &&
            !HostHasAvx2) {
            return PH_E_NOT_IMPLEMENTED;
        }

        UseAvx2 = (AllowAvx2 && HostHasAvx2 &&
                   RequestedIsa != PerfectHashJitMaxIsaAvx);

        if (UseAssigned16) {
            if (Table->HashFunctionId ==
                PerfectHashHashMultiplyShiftRXFunctionId) {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x4Avx2_x64 :
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(PerfectHashJitRawDogMultiplyShiftRX16Index32x4Avx2_x64) :
                    sizeof(PerfectHashJitRawDogMultiplyShiftRX16Index32x4_x64);
            } else if (Table->HashFunctionId ==
                       PerfectHashHashMulshrolate1RXFunctionId) {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMulshrolate1RX16Index32x4Avx2_x64 :
                    PerfectHashJitRawDogMulshrolate1RX16Index32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(PerfectHashJitRawDogMulshrolate1RX16Index32x4Avx2_x64) :
                    sizeof(PerfectHashJitRawDogMulshrolate1RX16Index32x4_x64);
            } else {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMultiplyShiftR16Index32x4Avx2_x64 :
                    PerfectHashJitRawDogMultiplyShiftR16Index32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(PerfectHashJitRawDogMultiplyShiftR16Index32x4Avx2_x64) :
                    sizeof(PerfectHashJitRawDogMultiplyShiftR16Index32x4_x64);
            }
        } else {
            if (Table->HashFunctionId ==
                PerfectHashHashMultiplyShiftRXFunctionId) {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x4Avx2_x64 :
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(PerfectHashJitRawDogMultiplyShiftRXIndex32x4Avx2_x64) :
                    sizeof(PerfectHashJitRawDogMultiplyShiftRXIndex32x4_x64);
            } else if (Table->HashFunctionId ==
                       PerfectHashHashMulshrolate1RXFunctionId) {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMulshrolate1RXIndex32x4Avx2_x64 :
                    PerfectHashJitRawDogMulshrolate1RXIndex32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(PerfectHashJitRawDogMulshrolate1RXIndex32x4Avx2_x64) :
                    sizeof(PerfectHashJitRawDogMulshrolate1RXIndex32x4_x64);
            } else {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMultiplyShiftRIndex32x4Avx2_x64 :
                    PerfectHashJitRawDogMultiplyShiftRIndex32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(PerfectHashJitRawDogMultiplyShiftRIndex32x4Avx2_x64) :
                    sizeof(PerfectHashJitRawDogMultiplyShiftRIndex32x4_x64);
            }
        }

#if defined(PH_RAWDOG_X64)
        if (Table->HashFunctionId ==
            PerfectHashHashMultiplyShiftRFunctionId &&
            UseAvx2) {
            ImmediateEntryCount = 0;
            ImmediateEntries[ImmediateEntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_ASSIGNED,
                Assigned,
                "Assigned",
                sizeof(ULONGLONG)
            };
            ImmediateEntries[ImmediateEntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_SEED1,
                (ULONGLONG)TableInfo->Seed1,
                "Seed1Imm",
                sizeof(ULONG)
            };
            ImmediateEntries[ImmediateEntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_SEED2,
                (ULONGLONG)TableInfo->Seed2,
                "Seed2Imm",
                sizeof(ULONG)
            };
            ImmediateEntries[ImmediateEntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_HASH_MASK,
                (ULONGLONG)Table->HashMask,
                "HashMaskImm",
                sizeof(ULONG)
            };
            ImmediateEntries[ImmediateEntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_INDEX_MASK,
                (ULONGLONG)Table->IndexMask,
                "IndexMaskImm",
                sizeof(ULONG)
            };

            EntriesForX4 = ImmediateEntries;
            EntryCountForX4 = ImmediateEntryCount;
        }
#endif

        Result = CreateRawDogCode(SourceCode,
                                  CodeSize,
                                  EntriesForX4,
                                  EntryCountForX4,
                                  Seed3Bytes.Byte1,
                                  Seed3Bytes.Byte2,
                                  &Code);
        if (FAILED(Result)) {
            return Result;
        }

        Jit->Index32x4Function = Code;
        Jit->Flags.Index32x4Compiled = TRUE;
        Jit->Flags.Index32x4Vector = TRUE;
        if (MaxIsa == PerfectHashJitMaxIsaAuto) {
            MaxIsa = UseAvx2 ? PerfectHashJitMaxIsaAvx2 :
                               PerfectHashJitMaxIsaAvx;
        }
    }

    if (ARGUMENT_PRESENT(UsedIsa)) {
        *UsedIsa = MaxIsa;
    }

    return S_OK;
#endif
#else
#if defined(PH_RAWDOG_ARM64)
    PTABLE_INFO_ON_DISK TableInfo;
    ULONG_BYTES Seed3Bytes;
    ULONGLONG Assigned;
    RAW_DOG_PATCH_ENTRY Entries[8];
    ULONG EntryCount = 0;
    BOOLEAN UseAssigned16;
    PERFECT_HASH_JIT_MAX_ISA_ID MaxIsa = PerfectHashJitMaxIsaAuto;
    PBYTE SourceCode = NULL;
    SIZE_T CodeSize = 0;
    PBYTE Code;
    HRESULT Result;

    if (!CompileIndex32x4 && !CompileIndex32x8 && !CompileIndex32x16) {
        if (ARGUMENT_PRESENT(UsedIsa)) {
            *UsedIsa = MaxIsa;
        }
        return S_OK;
    }

    if ((Table->HashFunctionId != PerfectHashHashMultiplyShiftRFunctionId &&
         Table->HashFunctionId != PerfectHashHashMultiplyShiftRXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate1RXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate2RXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate3RXFunctionId) ||
        Table->MaskFunctionId != PerfectHashAndMaskFunctionId) {
        return PH_E_NOT_IMPLEMENTED;
    }

    UseAssigned16 = (Table->State.UsingAssigned16 != FALSE);
    if (UseAssigned16) {
        if (!ARGUMENT_PRESENT(Table->Assigned16)) {
            return PH_E_INVARIANT_CHECK_FAILED;
        }
    } else if (!ARGUMENT_PRESENT(Table->TableData)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    TableInfo = Table->TableInfoOnDisk;
    if (!ARGUMENT_PRESENT(TableInfo)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    Seed3Bytes.AsULong = TableInfo->Seed3;
    if (UseAssigned16) {
        Result = EnsureRawDogAssigned16Padding(Table, Jit, &Assigned);
        if (FAILED(Result)) {
            return Result;
        }
    } else {
        Assigned = (ULONGLONG)(ULONG_PTR)(PVOID)Table->TableData;
    }

    Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_ASSIGNED,
        Assigned,
        "Assigned"
    };
    Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_SEED1,
        (ULONGLONG)TableInfo->Seed1,
        "Seed1"
    };
    Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_SEED2,
        (ULONGLONG)TableInfo->Seed2,
        "Seed2"
    };

    if (Table->HashFunctionId == PerfectHashHashMultiplyShiftRFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_HASH_MASK,
            (ULONGLONG)Table->HashMask,
            "HashMask"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else if (Table->HashFunctionId ==
               PerfectHashHashMulshrolate1RXFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else if (Table->HashFunctionId ==
               PerfectHashHashMulshrolate2RXFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE3,
            (ULONGLONG)Seed3Bytes.Byte3,
            "Seed3Byte3"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else if (Table->HashFunctionId ==
               PerfectHashHashMulshrolate3RXFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE3,
            (ULONGLONG)Seed3Bytes.Byte3,
            "Seed3Byte3"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED4,
            (ULONGLONG)TableInfo->Seed4,
            "Seed4"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    }

    if (CompileIndex32x4) {
        if (Table->HashFunctionId ==
            PerfectHashHashMultiplyShiftRFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftR16Index32x4_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftR16Index32x4_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRIndex32x4_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRIndex32x4_arm64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMultiplyShiftRXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x4_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x4_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x4_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x4_arm64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate3RXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate3RX16Index32x4_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate3RX16Index32x4_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate3RXIndex32x4_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate3RXIndex32x4_arm64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate2RXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate2RX16Index32x4_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate2RX16Index32x4_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate2RXIndex32x4_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate2RXIndex32x4_arm64
                );
            }
        } else if (UseAssigned16) {
            SourceCode = (PBYTE)
                PerfectHashJitRawDogMulshrolate1RX16Index32x4_arm64;
            CodeSize = sizeof(
                PerfectHashJitRawDogMulshrolate1RX16Index32x4_arm64
            );
        } else {
            SourceCode = (PBYTE)
                PerfectHashJitRawDogMulshrolate1RXIndex32x4_arm64;
            CodeSize = sizeof(
                PerfectHashJitRawDogMulshrolate1RXIndex32x4_arm64
            );
        }

        Result = CreateRawDogCode(SourceCode,
                                  CodeSize,
                                  Entries,
                                  EntryCount,
                                  Seed3Bytes.Byte1,
                                  Seed3Bytes.Byte2,
                                  &Code);
        if (FAILED(Result)) {
            return Result;
        }

        Jit->Index32x4Function = Code;
        Jit->Flags.Index32x4Compiled = TRUE;
        Jit->Flags.Index32x4Vector = TRUE;
        if (MaxIsa == PerfectHashJitMaxIsaAuto) {
            MaxIsa = PerfectHashJitMaxIsaNeon;
        }
    }

    if (CompileIndex32x16) {
        if (Table->HashFunctionId ==
            PerfectHashHashMultiplyShiftRFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftR16Index32x16_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftR16Index32x16_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRIndex32x16_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRIndex32x16_arm64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMultiplyShiftRXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x16_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x16_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x16_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x16_arm64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate3RXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate3RX16Index32x16_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate3RX16Index32x16_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate3RXIndex32x16_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate3RXIndex32x16_arm64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate2RXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate2RX16Index32x16_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate2RX16Index32x16_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate2RXIndex32x16_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate2RXIndex32x16_arm64
                );
            }
        } else if (UseAssigned16) {
            SourceCode = (PBYTE)
                PerfectHashJitRawDogMulshrolate1RX16Index32x16_arm64;
            CodeSize = sizeof(
                PerfectHashJitRawDogMulshrolate1RX16Index32x16_arm64
            );
        } else {
            SourceCode = (PBYTE)
                PerfectHashJitRawDogMulshrolate1RXIndex32x16_arm64;
            CodeSize = sizeof(
                PerfectHashJitRawDogMulshrolate1RXIndex32x16_arm64
            );
        }

        Result = CreateRawDogCode(SourceCode,
                                  CodeSize,
                                  Entries,
                                  EntryCount,
                                  Seed3Bytes.Byte1,
                                  Seed3Bytes.Byte2,
                                  &Code);
        if (FAILED(Result)) {
            return Result;
        }

        Jit->Index32x16Function = Code;
        Jit->Flags.Index32x16Compiled = TRUE;
        Jit->Flags.Index32x16Vector = TRUE;
        MaxIsa = PerfectHashJitMaxIsaNeon;
    }

    if (CompileIndex32x8) {
        if (Table->HashFunctionId ==
            PerfectHashHashMultiplyShiftRFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftR16Index32x8_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftR16Index32x8_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRIndex32x8_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRIndex32x8_arm64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMultiplyShiftRXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x8_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x8_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x8_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x8_arm64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate3RXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate3RX16Index32x8_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate3RX16Index32x8_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate3RXIndex32x8_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate3RXIndex32x8_arm64
                );
            }
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate2RXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate2RX16Index32x8_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate2RX16Index32x8_arm64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate2RXIndex32x8_arm64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate2RXIndex32x8_arm64
                );
            }
        } else if (UseAssigned16) {
            SourceCode = (PBYTE)
                PerfectHashJitRawDogMulshrolate1RX16Index32x8_arm64;
            CodeSize = sizeof(
                PerfectHashJitRawDogMulshrolate1RX16Index32x8_arm64
            );
        } else {
            SourceCode = (PBYTE)
                PerfectHashJitRawDogMulshrolate1RXIndex32x8_arm64;
            CodeSize = sizeof(
                PerfectHashJitRawDogMulshrolate1RXIndex32x8_arm64
            );
        }

        Result = CreateRawDogCode(SourceCode,
                                  CodeSize,
                                  Entries,
                                  EntryCount,
                                  Seed3Bytes.Byte1,
                                  Seed3Bytes.Byte2,
                                  &Code);
        if (FAILED(Result)) {
            return Result;
        }

        Jit->Index32x8Function = Code;
        Jit->Flags.Index32x8Compiled = TRUE;
        Jit->Flags.Index32x8Vector = TRUE;
        if (MaxIsa == PerfectHashJitMaxIsaAuto) {
            MaxIsa = PerfectHashJitMaxIsaNeon;
        }
    }

    if (ARGUMENT_PRESENT(UsedIsa)) {
        *UsedIsa = MaxIsa;
    }

    return S_OK;
#else

    PRTL Rtl;
    PTABLE_INFO_ON_DISK TableInfo;
    ULONG_BYTES Seed3Bytes;
    ULONGLONG Assigned;
    RAW_DOG_PATCH_ENTRY Entries[8];
    ULONG EntryCount = 0;
    BOOLEAN UseAssigned16;
    BOOLEAN HostHasAvx;
    BOOLEAN HostHasAvx2;
    BOOLEAN HostHasAvx512;
    BOOLEAN AllowAvx;
    BOOLEAN AllowAvx2;
    BOOLEAN AllowAvx512;
    ULONG VectorVersion;
    PERFECT_HASH_JIT_MAX_ISA_ID RequestedIsa;
    PERFECT_HASH_JIT_MAX_ISA_ID MaxIsa = PerfectHashJitMaxIsaAuto;
    PBYTE SourceCode = NULL;
    SIZE_T CodeSize = 0;
    PBYTE Code;
    HRESULT Result;

    if (!CompileIndex32x4 && !CompileIndex32x8 && !CompileIndex32x16) {
        if (ARGUMENT_PRESENT(UsedIsa)) {
            *UsedIsa = MaxIsa;
        }
        return S_OK;
    }

    if ((Table->HashFunctionId != PerfectHashHashMultiplyShiftRFunctionId &&
         Table->HashFunctionId != PerfectHashHashMultiplyShiftRXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate1RXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate2RXFunctionId &&
         Table->HashFunctionId != PerfectHashHashMulshrolate3RXFunctionId) ||
        Table->MaskFunctionId != PerfectHashAndMaskFunctionId) {
        return PH_E_NOT_IMPLEMENTED;
    }

    UseAssigned16 = (Table->State.UsingAssigned16 != FALSE);
    if (UseAssigned16) {
        if (!ARGUMENT_PRESENT(Table->Assigned16)) {
            return PH_E_INVARIANT_CHECK_FAILED;
        }
    } else if (!ARGUMENT_PRESENT(Table->TableData)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    TableInfo = Table->TableInfoOnDisk;
    if (!ARGUMENT_PRESENT(TableInfo)) {
        return PH_E_INVARIANT_CHECK_FAILED;
    }

    Seed3Bytes.AsULong = TableInfo->Seed3;
    if (UseAssigned16) {
        Result = EnsureRawDogAssigned16Padding(Table, Jit, &Assigned);
        if (FAILED(Result)) {
            return Result;
        }
    } else {
        Assigned = (ULONGLONG)(ULONG_PTR)(PVOID)Table->TableData;
    }

    Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_ASSIGNED,
        Assigned,
        "Assigned"
    };
    Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_SEED1,
        (ULONGLONG)TableInfo->Seed1,
        "Seed1"
    };
    Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
        RAWDOG_SENTINEL_SEED2,
        (ULONGLONG)TableInfo->Seed2,
        "Seed2"
    };
    if (Table->HashFunctionId == PerfectHashHashMultiplyShiftRFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_HASH_MASK,
            (ULONGLONG)Table->HashMask,
            "HashMask"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else if (Table->HashFunctionId ==
               PerfectHashHashMulshrolate1RXFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else if (Table->HashFunctionId ==
               PerfectHashHashMulshrolate2RXFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE3,
            (ULONGLONG)Seed3Bytes.Byte3,
            "Seed3Byte3"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else if (Table->HashFunctionId ==
               PerfectHashHashMulshrolate3RXFunctionId) {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE2,
            (ULONGLONG)Seed3Bytes.Byte2,
            "Seed3Byte2"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE3,
            (ULONGLONG)Seed3Bytes.Byte3,
            "Seed3Byte3"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED4,
            (ULONGLONG)TableInfo->Seed4,
            "Seed4"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    } else {
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_SEED3_BYTE1,
            (ULONGLONG)Seed3Bytes.Byte1,
            "Seed3Byte1"
        };
        Entries[EntryCount++] = (RAW_DOG_PATCH_ENTRY){
            RAWDOG_SENTINEL_INDEX_MASK,
            (ULONGLONG)Table->IndexMask,
            "IndexMask"
        };
    }

    Rtl = Table->Rtl;
    HostHasAvx = (ARGUMENT_PRESENT(Rtl) && Rtl->CpuFeatures.AVX != FALSE);
    HostHasAvx2 = (ARGUMENT_PRESENT(Rtl) && Rtl->CpuFeatures.AVX2 != FALSE);
    HostHasAvx512 = (ARGUMENT_PRESENT(Rtl) && Rtl->CpuFeatures.AVX512F != FALSE);

    RequestedIsa = (PERFECT_HASH_JIT_MAX_ISA_ID)CompileFlags->JitMaxIsa;
    AllowAvx512 = (RequestedIsa == PerfectHashJitMaxIsaAuto ||
                   RequestedIsa == PerfectHashJitMaxIsaAvx512);
    AllowAvx = (RequestedIsa == PerfectHashJitMaxIsaAuto ||
                RequestedIsa == PerfectHashJitMaxIsaAvx ||
                RequestedIsa == PerfectHashJitMaxIsaAvx2 ||
                RequestedIsa == PerfectHashJitMaxIsaAvx512);
    AllowAvx2 = (RequestedIsa == PerfectHashJitMaxIsaAuto ||
                 RequestedIsa == PerfectHashJitMaxIsaAvx2 ||
                 RequestedIsa == PerfectHashJitMaxIsaAvx512);

    VectorVersion = GetRawDogVectorVersion();
#if defined(PH_RAWDOG_X64)
    RawDogTraceCpuFeatures(Rtl,
                           RequestedIsa,
                           VectorVersion,
                           HostHasAvx2,
                           HostHasAvx512,
                           AllowAvx2,
                           AllowAvx512);
#endif

    if (CompileIndex32x16) {
        if (Table->HashFunctionId ==
            PerfectHashHashMultiplyShiftRFunctionId) {
            if (UseAssigned16) {
#if defined(PH_LINUX)
                if (VectorVersion >= 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMultiplyShiftR16Index32x16Avx512_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMultiplyShiftR16Index32x16Avx512_x64
                    );
                } else {
#endif
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMultiplyShiftR16Index32x16_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMultiplyShiftR16Index32x16_x64
                    );
#if defined(PH_LINUX)
                }
#endif
            } else {
#if defined(PH_LINUX)
                if (VectorVersion >= 2) {
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMultiplyShiftRIndex32x16Avx512_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMultiplyShiftRIndex32x16Avx512_x64
                    );
                } else {
#endif
                    SourceCode = (PBYTE)
                        PerfectHashJitRawDogMultiplyShiftRIndex32x16_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMultiplyShiftRIndex32x16_x64
                    );
#if defined(PH_LINUX)
                }
#endif
            }

            Result = CreateRawDogCode(SourceCode,
                                      CodeSize,
                                      Entries,
                                      EntryCount,
                                      Seed3Bytes.Byte1,
                                      Seed3Bytes.Byte2,
                                      &Code);
            if (FAILED(Result)) {
                return Result;
            }

            Jit->Index32x16Function = Code;
            Jit->Flags.Index32x16Compiled = TRUE;
            Jit->Flags.Index32x16Vector = TRUE;
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate1RXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate1RX16Index32x16_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate1RX16Index32x16_x64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate1RXIndex32x16_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate1RXIndex32x16_x64
                );
            }

            Result = CreateRawDogCode(SourceCode,
                                      CodeSize,
                                      Entries,
                                      EntryCount,
                                      Seed3Bytes.Byte1,
                                      Seed3Bytes.Byte2,
                                      &Code);
            if (FAILED(Result)) {
                return Result;
            }

            Jit->Index32x16Function = Code;
            Jit->Flags.Index32x16Compiled = TRUE;
            Jit->Flags.Index32x16Vector = TRUE;
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMultiplyShiftRXFunctionId) {
            if (!AllowAvx512 || !HostHasAvx512) {
                return PH_E_NOT_IMPLEMENTED;
            }

            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x16_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x16_x64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x16_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x16_x64
                );
            }

            Result = CreateRawDogCode(SourceCode,
                                      CodeSize,
                                      Entries,
                                      EntryCount,
                                      Seed3Bytes.Byte1,
                                      Seed3Bytes.Byte2,
                                      &Code);
            if (FAILED(Result)) {
                return Result;
            }

            Jit->Index32x16Function = Code;
            Jit->Flags.Index32x16Compiled = TRUE;
            Jit->Flags.Index32x16Vector = TRUE;
            MaxIsa = PerfectHashJitMaxIsaAvx512;
        } else {
            if (!AllowAvx512 || !HostHasAvx512) {
                return PH_E_NOT_IMPLEMENTED;
            }

            if (Table->HashFunctionId ==
                PerfectHashHashMulshrolate3RXFunctionId) {
                if (UseAssigned16) {
                    if (VectorVersion >= 2) {
                        SourceCode =
                            (PBYTE)
                            PerfectHashJitRawDogMulshrolate3RX16Avx512_v2_x64;
                        CodeSize = sizeof(
                            PerfectHashJitRawDogMulshrolate3RX16Avx512_v2_x64
                        );
                    } else {
                        SourceCode =
                            (PBYTE)
                            PerfectHashJitRawDogMulshrolate3RX16Avx512_v1_x64;
                        CodeSize = sizeof(
                            PerfectHashJitRawDogMulshrolate3RX16Avx512_v1_x64
                        );
                    }
                } else {
                    if (VectorVersion >= 2) {
                        SourceCode =
                            (PBYTE)
                            PerfectHashJitRawDogMulshrolate3RXAvx512_v2_x64;
                        CodeSize = sizeof(
                            PerfectHashJitRawDogMulshrolate3RXAvx512_v2_x64
                        );
                    } else {
                        SourceCode =
                            (PBYTE)
                            PerfectHashJitRawDogMulshrolate3RXAvx512_v1_x64;
                        CodeSize = sizeof(
                            PerfectHashJitRawDogMulshrolate3RXAvx512_v1_x64
                        );
                    }
                }
            } else if (UseAssigned16) {
                if (VectorVersion >= 2) {
                    SourceCode =
                        (PBYTE)PerfectHashJitRawDogMulshrolate2RX16Avx512_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Avx512_v2_x64
                    );
                } else {
                    SourceCode =
                        (PBYTE)PerfectHashJitRawDogMulshrolate2RX16Avx512_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Avx512_v1_x64
                    );
                }
            } else {
                if (VectorVersion >= 2) {
                    SourceCode =
                        (PBYTE)PerfectHashJitRawDogMulshrolate2RXAvx512_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RXAvx512_v2_x64
                    );
                } else {
                    SourceCode =
                        (PBYTE)PerfectHashJitRawDogMulshrolate2RXAvx512_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RXAvx512_v1_x64
                    );
                }
            }

            Result = CreateRawDogCode(SourceCode,
                                      CodeSize,
                                      Entries,
                                      EntryCount,
                                      Seed3Bytes.Byte1,
                                      Seed3Bytes.Byte2,
                                      &Code);
            if (FAILED(Result)) {
                return Result;
            }

            Jit->Index32x16Function = Code;
            Jit->Flags.Index32x16Compiled = TRUE;
            Jit->Flags.Index32x16Vector = TRUE;
            MaxIsa = PerfectHashJitMaxIsaAvx512;
        }
    }

    if (CompileIndex32x8) {
        if (Table->HashFunctionId ==
            PerfectHashHashMultiplyShiftRFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftR16Index32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftR16Index32x8_x64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRIndex32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRIndex32x8_x64
                );
            }

            Result = CreateRawDogCode(SourceCode,
                                      CodeSize,
                                      Entries,
                                      EntryCount,
                                      Seed3Bytes.Byte1,
                                      Seed3Bytes.Byte2,
                                      &Code);
            if (FAILED(Result)) {
                return Result;
            }

            Jit->Index32x8Function = Code;
            Jit->Flags.Index32x8Compiled = TRUE;
            Jit->Flags.Index32x8Vector = TRUE;
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMulshrolate1RXFunctionId) {
            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate1RX16Index32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate1RX16Index32x8_x64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMulshrolate1RXIndex32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMulshrolate1RXIndex32x8_x64
                );
            }

            Result = CreateRawDogCode(SourceCode,
                                      CodeSize,
                                      Entries,
                                      EntryCount,
                                      Seed3Bytes.Byte1,
                                      Seed3Bytes.Byte2,
                                      &Code);
            if (FAILED(Result)) {
                return Result;
            }

            Jit->Index32x8Function = Code;
            Jit->Flags.Index32x8Compiled = TRUE;
            Jit->Flags.Index32x8Vector = TRUE;
        } else if (Table->HashFunctionId ==
                   PerfectHashHashMultiplyShiftRXFunctionId) {
            if (!AllowAvx2 || !HostHasAvx2) {
                return PH_E_NOT_IMPLEMENTED;
            }

            if (UseAssigned16) {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x8_x64
                );
            } else {
                SourceCode = (PBYTE)
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x8_x64;
                CodeSize = sizeof(
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x8_x64
                );
            }

            Result = CreateRawDogCode(SourceCode,
                                      CodeSize,
                                      Entries,
                                      EntryCount,
                                      Seed3Bytes.Byte1,
                                      Seed3Bytes.Byte2,
                                      &Code);
            if (FAILED(Result)) {
                return Result;
            }

            Jit->Index32x8Function = Code;
            Jit->Flags.Index32x8Compiled = TRUE;
            Jit->Flags.Index32x8Vector = TRUE;
            if (MaxIsa == PerfectHashJitMaxIsaAuto) {
                MaxIsa = PerfectHashJitMaxIsaAvx2;
            }
        } else {
            if (!AllowAvx2 || !HostHasAvx2) {
                return PH_E_NOT_IMPLEMENTED;
            }

            if (Table->HashFunctionId ==
                PerfectHashHashMulshrolate3RXFunctionId) {
                if (UseAssigned16) {
                    if (VectorVersion == 3) {
                        SourceCode =
                            (PBYTE)
                            PerfectHashJitRawDogMulshrolate3RX16Avx2_v3_x64;
                        CodeSize = sizeof(
                            PerfectHashJitRawDogMulshrolate3RX16Avx2_v3_x64
                        );
                    } else if (VectorVersion == 2) {
                        SourceCode =
                            (PBYTE)
                            PerfectHashJitRawDogMulshrolate3RX16Avx2_v2_x64;
                        CodeSize = sizeof(
                            PerfectHashJitRawDogMulshrolate3RX16Avx2_v2_x64
                        );
                    } else {
                        SourceCode =
                            (PBYTE)
                            PerfectHashJitRawDogMulshrolate3RX16Avx2_v1_x64;
                        CodeSize = sizeof(
                            PerfectHashJitRawDogMulshrolate3RX16Avx2_v1_x64
                        );
                    }
                } else {
                    if (VectorVersion == 3) {
                        SourceCode =
                            (PBYTE)
                            PerfectHashJitRawDogMulshrolate3RXAvx2_v3_x64;
                        CodeSize = sizeof(
                            PerfectHashJitRawDogMulshrolate3RXAvx2_v3_x64
                        );
                    } else if (VectorVersion == 2) {
                        SourceCode =
                            (PBYTE)
                            PerfectHashJitRawDogMulshrolate3RXAvx2_v2_x64;
                        CodeSize = sizeof(
                            PerfectHashJitRawDogMulshrolate3RXAvx2_v2_x64
                        );
                    } else {
                        SourceCode =
                            (PBYTE)
                            PerfectHashJitRawDogMulshrolate3RXAvx2_v1_x64;
                        CodeSize = sizeof(
                            PerfectHashJitRawDogMulshrolate3RXAvx2_v1_x64
                        );
                    }
                }
            } else if (UseAssigned16) {
                if (VectorVersion == 3) {
                    SourceCode =
                        (PBYTE)PerfectHashJitRawDogMulshrolate2RX16Avx2_v3_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Avx2_v3_x64
                    );
                } else if (VectorVersion == 2) {
                    SourceCode =
                        (PBYTE)PerfectHashJitRawDogMulshrolate2RX16Avx2_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Avx2_v2_x64
                    );
                } else {
                    SourceCode =
                        (PBYTE)PerfectHashJitRawDogMulshrolate2RX16Avx2_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Avx2_v1_x64
                    );
                }
            } else {
                if (VectorVersion == 3) {
                    SourceCode =
                        (PBYTE)PerfectHashJitRawDogMulshrolate2RXAvx2_v3_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RXAvx2_v3_x64
                    );
                } else if (VectorVersion == 2) {
                    SourceCode =
                        (PBYTE)PerfectHashJitRawDogMulshrolate2RXAvx2_v2_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RXAvx2_v2_x64
                    );
                } else {
                    SourceCode =
                        (PBYTE)PerfectHashJitRawDogMulshrolate2RXAvx2_v1_x64;
                    CodeSize = sizeof(
                        PerfectHashJitRawDogMulshrolate2RXAvx2_v1_x64
                    );
                }
            }

            Result = CreateRawDogCode(SourceCode,
                                      CodeSize,
                                      Entries,
                                      EntryCount,
                                      Seed3Bytes.Byte1,
                                      Seed3Bytes.Byte2,
                                      &Code);
            if (FAILED(Result)) {
                return Result;
            }

            Jit->Index32x8Function = Code;
            Jit->Flags.Index32x8Compiled = TRUE;
            Jit->Flags.Index32x8Vector = TRUE;
            if (MaxIsa == PerfectHashJitMaxIsaAuto) {
                MaxIsa = PerfectHashJitMaxIsaAvx2;
            }
        }
    }

    if (CompileIndex32x4) {
        BOOLEAN UseAvx2;
        RAW_DOG_PATCH_ENTRY *EntriesForX4 = Entries;
        ULONG EntryCountForX4 = EntryCount;
#if defined(PH_RAWDOG_X64)
        RAW_DOG_PATCH_ENTRY ImmediateEntries[8];
        ULONG ImmediateEntryCount = 0;
#endif

        if (!AllowAvx || !HostHasAvx) {
            return PH_E_NOT_IMPLEMENTED;
        }

        if ((RequestedIsa == PerfectHashJitMaxIsaAvx2 ||
             RequestedIsa == PerfectHashJitMaxIsaAvx512) &&
            !HostHasAvx2) {
            return PH_E_NOT_IMPLEMENTED;
        }

        UseAvx2 = (AllowAvx2 && HostHasAvx2 &&
                   RequestedIsa != PerfectHashJitMaxIsaAvx);

        if (UseAssigned16) {
            if (Table->HashFunctionId ==
                PerfectHashHashMultiplyShiftRXFunctionId) {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x4Avx2_x64 :
                    PerfectHashJitRawDogMultiplyShiftRX16Index32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(
                        PerfectHashJitRawDogMultiplyShiftRX16Index32x4Avx2_x64
                    ) :
                    sizeof(
                        PerfectHashJitRawDogMultiplyShiftRX16Index32x4_x64
                    );
            } else if (Table->HashFunctionId ==
                       PerfectHashHashMulshrolate1RXFunctionId) {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMulshrolate1RX16Index32x4Avx2_x64 :
                    PerfectHashJitRawDogMulshrolate1RX16Index32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(
                        PerfectHashJitRawDogMulshrolate1RX16Index32x4Avx2_x64
                    ) :
                    sizeof(
                        PerfectHashJitRawDogMulshrolate1RX16Index32x4_x64
                    );
            } else if (Table->HashFunctionId ==
                       PerfectHashHashMulshrolate2RXFunctionId) {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMulshrolate2RX16Index32x4Avx2_x64 :
                    PerfectHashJitRawDogMulshrolate2RX16Index32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Index32x4Avx2_x64
                    ) :
                    sizeof(
                        PerfectHashJitRawDogMulshrolate2RX16Index32x4_x64
                    );
            } else {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMultiplyShiftR16Index32x4Avx2_x64 :
                    PerfectHashJitRawDogMultiplyShiftR16Index32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(
                        PerfectHashJitRawDogMultiplyShiftR16Index32x4Avx2_x64
                    ) :
                    sizeof(
                        PerfectHashJitRawDogMultiplyShiftR16Index32x4_x64
                    );
            }
        } else {
            if (Table->HashFunctionId ==
                PerfectHashHashMultiplyShiftRXFunctionId) {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x4Avx2_x64 :
                    PerfectHashJitRawDogMultiplyShiftRXIndex32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(
                        PerfectHashJitRawDogMultiplyShiftRXIndex32x4Avx2_x64
                    ) :
                    sizeof(
                        PerfectHashJitRawDogMultiplyShiftRXIndex32x4_x64
                    );
            } else if (Table->HashFunctionId ==
                       PerfectHashHashMulshrolate1RXFunctionId) {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMulshrolate1RXIndex32x4Avx2_x64 :
                    PerfectHashJitRawDogMulshrolate1RXIndex32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(
                        PerfectHashJitRawDogMulshrolate1RXIndex32x4Avx2_x64
                    ) :
                    sizeof(
                        PerfectHashJitRawDogMulshrolate1RXIndex32x4_x64
                    );
            } else if (Table->HashFunctionId ==
                       PerfectHashHashMulshrolate2RXFunctionId) {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMulshrolate2RXIndex32x4Avx2_x64 :
                    PerfectHashJitRawDogMulshrolate2RXIndex32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(
                        PerfectHashJitRawDogMulshrolate2RXIndex32x4Avx2_x64
                    ) :
                    sizeof(
                        PerfectHashJitRawDogMulshrolate2RXIndex32x4_x64
                    );
            } else {
                SourceCode = (PBYTE)(
                    UseAvx2 ?
                    PerfectHashJitRawDogMultiplyShiftRIndex32x4Avx2_x64 :
                    PerfectHashJitRawDogMultiplyShiftRIndex32x4_x64
                );
                CodeSize = UseAvx2 ?
                    sizeof(
                        PerfectHashJitRawDogMultiplyShiftRIndex32x4Avx2_x64
                    ) :
                    sizeof(
                        PerfectHashJitRawDogMultiplyShiftRIndex32x4_x64
                    );
            }
        }

#if defined(PH_RAWDOG_X64)
        if (Table->HashFunctionId ==
            PerfectHashHashMultiplyShiftRFunctionId &&
            UseAvx2) {
            ImmediateEntryCount = 0;
            ImmediateEntries[ImmediateEntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_ASSIGNED,
                Assigned,
                "Assigned",
                sizeof(ULONGLONG)
            };
            ImmediateEntries[ImmediateEntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_SEED1,
                (ULONGLONG)TableInfo->Seed1,
                "Seed1Imm",
                sizeof(ULONG)
            };
            ImmediateEntries[ImmediateEntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_SEED2,
                (ULONGLONG)TableInfo->Seed2,
                "Seed2Imm",
                sizeof(ULONG)
            };
            ImmediateEntries[ImmediateEntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_HASH_MASK,
                (ULONGLONG)Table->HashMask,
                "HashMaskImm",
                sizeof(ULONG)
            };
            ImmediateEntries[ImmediateEntryCount++] = (RAW_DOG_PATCH_ENTRY){
                RAWDOG_SENTINEL_IMM32_INDEX_MASK,
                (ULONGLONG)Table->IndexMask,
                "IndexMaskImm",
                sizeof(ULONG)
            };

            EntriesForX4 = ImmediateEntries;
            EntryCountForX4 = ImmediateEntryCount;
        }
#endif

        Result = CreateRawDogCode(SourceCode,
                                  CodeSize,
                                  EntriesForX4,
                                  EntryCountForX4,
                                  Seed3Bytes.Byte1,
                                  Seed3Bytes.Byte2,
                                  &Code);
        if (FAILED(Result)) {
            return Result;
        }

        Jit->Index32x4Function = Code;
        Jit->Flags.Index32x4Compiled = TRUE;
        Jit->Flags.Index32x4Vector = TRUE;
        if (MaxIsa == PerfectHashJitMaxIsaAuto) {
            MaxIsa = UseAvx2 ? PerfectHashJitMaxIsaAvx2 :
                               PerfectHashJitMaxIsaAvx;
        }
    }

    if (ARGUMENT_PRESENT(UsedIsa)) {
        *UsedIsa = MaxIsa;
    }

    return S_OK;
#endif
#endif
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

    memset(Info, 0, sizeof(*Info));
    Info->SizeOfStruct = sizeof(*Info);
    Info->Flags.AsULong = JitState->Flags.AsULong;
    Info->JitMaxIsa = JitState->JitMaxIsa;
    CopyRawDogString(Info->TargetCpu,
                     sizeof(Info->TargetCpu),
                     JitState->TargetCpu);
    CopyRawDogString(Info->TargetFeatures,
                     sizeof(Info->TargetFeatures),
                     JitState->TargetFeatures);

    return S_OK;
}

static PERFECT_HASH_TABLE_JIT_INTERFACE_VTBL RawDogJitInterfaceVtbl = {
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
    BOOLEAN WantsIndex32x2;
    BOOLEAN WantsIndex32x4;
    BOOLEAN WantsIndex32x8;
    BOOLEAN WantsIndex32x16;
    BOOLEAN WantsVector;
    PERFECT_HASH_JIT_MAX_ISA_ID UsedIsa = PerfectHashJitMaxIsaAuto;

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
        if (!ARGUMENT_PRESENT(Table->Assigned16)) {
            return PH_E_INVARIANT_CHECK_FAILED;
        }
    } else if (!ARGUMENT_PRESENT(Table->TableData)) {
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

    if (Table->HashFunctionId != PerfectHashHashMultiplyShiftRFunctionId &&
        Table->HashFunctionId != PerfectHashHashMultiplyShiftRXFunctionId &&
        Table->HashFunctionId != PerfectHashHashMulshrolate1RXFunctionId &&
        Table->HashFunctionId != PerfectHashHashMulshrolate2RXFunctionId &&
        Table->HashFunctionId != PerfectHashHashMulshrolate3RXFunctionId) {
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
    WantsIndex32x2 = (CompileFlags.JitIndex32x2 != FALSE ||
                      CompileFlags.JitVectorIndex32x2 != FALSE);
    WantsIndex32x4 = (CompileFlags.JitIndex32x4 != FALSE ||
                      CompileFlags.JitVectorIndex32x4 != FALSE);
    WantsIndex32x8 = (CompileFlags.JitIndex32x8 != FALSE ||
                      CompileFlags.JitVectorIndex32x8 != FALSE);
    WantsIndex32x16 = (CompileFlags.JitIndex32x16 != FALSE);
    WantsVector = (WantsIndex32x4 || WantsIndex32x8 || WantsIndex32x16);

    if (CompileFlags.JitIndex64 ||
        WantsIndex32x2) {
        return PH_E_NOT_IMPLEMENTED;
    }

    if (WantsVector &&
        Table->HashFunctionId != PerfectHashHashMultiplyShiftRFunctionId &&
        Table->HashFunctionId != PerfectHashHashMultiplyShiftRXFunctionId &&
        Table->HashFunctionId != PerfectHashHashMulshrolate1RXFunctionId &&
        Table->HashFunctionId != PerfectHashHashMulshrolate2RXFunctionId &&
        Table->HashFunctionId != PerfectHashHashMulshrolate3RXFunctionId) {
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

    if (WantsVector) {
        Result = CompileChm01IndexVectorJitRawDog(Table,
                                                  Jit,
                                                  &CompileFlags,
                                                  WantsIndex32x4,
                                                  WantsIndex32x8,
                                                  WantsIndex32x16,
                                                  &UsedIsa);
        if (FAILED(Result)) {
            PerfectHashTableJitRundownRawDog(Table);
            return Result;
        }

        if (UsedIsa == PerfectHashJitMaxIsaAvx) {
            CopyRawDogString(Jit->TargetFeatures,
                             sizeof(Jit->TargetFeatures),
                             "avx");
            Jit->JitMaxIsa = UsedIsa;
        } else if (UsedIsa == PerfectHashJitMaxIsaAvx2) {
            CopyRawDogString(Jit->TargetFeatures,
                             sizeof(Jit->TargetFeatures),
                             "avx2");
            Jit->JitMaxIsa = UsedIsa;
        } else if (UsedIsa == PerfectHashJitMaxIsaAvx512) {
            CopyRawDogString(Jit->TargetFeatures,
                             sizeof(Jit->TargetFeatures),
                             "avx512f");
            Jit->JitMaxIsa = UsedIsa;
        }
    }

    Jit->Flags.Valid = TRUE;
    Table->Flags.JitEnabled = TRUE;
    if (Jit->Flags.Index32Compiled) {
        Table->Vtbl->Index = RawDogTableIndexJit;
    }

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

    FreeRawDogCode(&Jit->Index32Function);
    FreeRawDogCode(&Jit->Index32x4Function);
    FreeRawDogCode(&Jit->Index32x8Function);
    FreeRawDogCode(&Jit->Index32x16Function);
    Jit->ExecutionEngine = NULL;

    Allocator = Table->Allocator;
    if (Jit->Assigned16Padded) {
        Allocator->Vtbl->FreePointer(Allocator, &Jit->Assigned16Padded);
        Jit->Assigned16PaddedSize = 0;
    }
    Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&Table->Jit);
}

#endif // PH_HAS_RAWDOG_JIT

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
