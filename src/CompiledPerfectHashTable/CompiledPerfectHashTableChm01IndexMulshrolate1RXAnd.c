#define CPH_M1RX_INDEX32X8_ROUTINE_NAME(T) CompiledPerfectHash_##T##_Index32x8
#define CPH_M1RX_INDEX32X16_ROUTINE_NAME(T) CompiledPerfectHash_##T##_Index32x16

#define EXPAND_M1RX_INDEX32X8_ROUTINE_NAME(T) CPH_M1RX_INDEX32X8_ROUTINE_NAME(T)
#define EXPAND_M1RX_INDEX32X16_ROUTINE_NAME(T) CPH_M1RX_INDEX32X16_ROUTINE_NAME(T)

#define INDEX32X8_ROUTINE EXPAND_M1RX_INDEX32X8_ROUTINE_NAME(CPH_TABLENAME)
#define INDEX32X16_ROUTINE EXPAND_M1RX_INDEX32X16_ROUTINE_NAME(CPH_TABLENAME)

#define CPH_HAS_INDEX32X8_ROUTINE 1
#define CPH_HAS_INDEX32X16_ROUTINE 1

DECLARE_INDEX_ROUTINE()
{
    CPHINDEX Index;
    CPHDKEY Vertex1;
    CPHDKEY Vertex2;
    CPHDKEY DownsizedKey;

    //IACA_VC_START();

    DownsizedKey = DOWNSIZE_KEY(Key);

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = RotateRight32(Vertex1, SEED3_BYTE2);
    Vertex1 = Vertex1 >> SEED3_BYTE1;
    Vertex1 = TABLE_DATA[Vertex1];

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE1;
    Vertex2 = TABLE_DATA[Vertex2];

    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);

    //IACA_VC_END();

    return Index;
}

#ifndef CPH_INLINE_ROUTINES

#if defined(_M_X64) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if (defined(__clang__) || defined(__GNUC__)) && \
    (defined(__x86_64__) || defined(__i386__))
#define CPH_M1RX_HAS_TARGET_ATTR 1
#endif

#if defined(CPH_M1RX_HAS_TARGET_ATTR) || defined(__AVX2__)
#define CPH_M1RX_CAN_BUILD_AVX2 1
#endif

#if defined(CPH_M1RX_HAS_TARGET_ATTR) || defined(__AVX512F__)
#define CPH_M1RX_CAN_BUILD_AVX512 1
#endif

#if defined(CPH_M1RX_HAS_TARGET_ATTR)
#define CPH_M1RX_TARGET_AVX2 __attribute__((target("avx2")))
#define CPH_M1RX_TARGET_AVX512 __attribute__((target("avx512f")))
#else
#define CPH_M1RX_TARGET_AVX2
#define CPH_M1RX_TARGET_AVX512
#endif

FORCEINLINE
CPHINDEX
CphMulshrolate1RXFinalizeIndex(
    CPHDKEY Vertex1,
    CPHDKEY Vertex2
    )
{
    CPHDKEY Assigned1;
    CPHDKEY Assigned2;

    Assigned1 = TABLE_DATA[Vertex1];
    Assigned2 = TABLE_DATA[Vertex2];

    return (CPHINDEX)((Assigned1 + Assigned2) & INDEX_MASK);
}

FORCEINLINE
VOID
CphMulshrolate1RXComputeVerticesScalar(
    _In_reads_(Count) PULONG Keys,
    _Out_writes_(Count) PCPHDKEY Vertex1,
    _Out_writes_(Count) PCPHDKEY Vertex2,
    _In_ ULONG Count
    )
{
    ULONG Index;

    for (Index = 0; Index < Count; Index++) {
        CPHDKEY Value1;
        CPHDKEY Value2;

        Value1 = Keys[Index] * SEED1;
        Value1 = RotateRight32(Value1, SEED3_BYTE2);
        Value1 = Value1 >> SEED3_BYTE1;

        Value2 = Keys[Index] * SEED2;
        Value2 = Value2 >> SEED3_BYTE1;

        Vertex1[Index] = Value1;
        Vertex2[Index] = Value2;
    }
}

#if defined(CPH_M1RX_CAN_BUILD_AVX2)
static CPH_M1RX_TARGET_AVX2
VOID
CphMulshrolate1RXComputeVertices8Avx2(
    _In_reads_(8) PULONG Keys,
    _Out_writes_(8) PCPHDKEY Vertex1,
    _Out_writes_(8) PCPHDKEY Vertex2
    )
{
    const __m256i KeysYmm = _mm256_loadu_si256((const __m256i *)(const VOID *)Keys);
    const __m256i Seed1Ymm = _mm256_set1_epi32((int)SEED1);
    const __m256i Seed2Ymm = _mm256_set1_epi32((int)SEED2);

    const __m256i RotateYmm = _mm256_set1_epi32((int)(SEED3_BYTE2 & 31));
    const __m256i RotateInverseYmm =
        _mm256_set1_epi32((int)((32 - (SEED3_BYTE2 & 31)) & 31));
    const __m256i ShiftYmm = _mm256_set1_epi32((int)(SEED3_BYTE1 & 31));

    __m256i Value1Ymm = _mm256_mullo_epi32(KeysYmm, Seed1Ymm);
    __m256i Value2Ymm = _mm256_mullo_epi32(KeysYmm, Seed2Ymm);

    Value1Ymm = _mm256_or_si256(
        _mm256_srlv_epi32(Value1Ymm, RotateYmm),
        _mm256_sllv_epi32(Value1Ymm, RotateInverseYmm)
    );

    Value1Ymm = _mm256_srlv_epi32(Value1Ymm, ShiftYmm);
    Value2Ymm = _mm256_srlv_epi32(Value2Ymm, ShiftYmm);

    _mm256_storeu_si256((__m256i *)(VOID *)Vertex1, Value1Ymm);
    _mm256_storeu_si256((__m256i *)(VOID *)Vertex2, Value2Ymm);
}
#endif

#if defined(CPH_M1RX_CAN_BUILD_AVX512)
static CPH_M1RX_TARGET_AVX512
VOID
CphMulshrolate1RXComputeVertices16Avx512(
    _In_reads_(16) PULONG Keys,
    _Out_writes_(16) PCPHDKEY Vertex1,
    _Out_writes_(16) PCPHDKEY Vertex2
    )
{
    const __m512i KeysZmm = _mm512_loadu_si512((const VOID *)Keys);
    const __m512i Seed1Zmm = _mm512_set1_epi32((int)SEED1);
    const __m512i Seed2Zmm = _mm512_set1_epi32((int)SEED2);

    const __m512i RotateZmm = _mm512_set1_epi32((int)(SEED3_BYTE2 & 31));
    const __m512i RotateInverseZmm =
        _mm512_set1_epi32((int)((32 - (SEED3_BYTE2 & 31)) & 31));
    const __m512i ShiftZmm = _mm512_set1_epi32((int)(SEED3_BYTE1 & 31));

    __m512i Value1Zmm = _mm512_mullo_epi32(KeysZmm, Seed1Zmm);
    __m512i Value2Zmm = _mm512_mullo_epi32(KeysZmm, Seed2Zmm);

    Value1Zmm = _mm512_or_si512(
        _mm512_srlv_epi32(Value1Zmm, RotateZmm),
        _mm512_sllv_epi32(Value1Zmm, RotateInverseZmm)
    );

    Value1Zmm = _mm512_srlv_epi32(Value1Zmm, ShiftZmm);
    Value2Zmm = _mm512_srlv_epi32(Value2Zmm, ShiftZmm);

    _mm512_storeu_si512((VOID *)Vertex1, Value1Zmm);
    _mm512_storeu_si512((VOID *)Vertex2, Value2Zmm);
}
#endif

FORCEINLINE
BOOLEAN
CphMulshrolate1RXCanUseAvx2(
    VOID
    )
{
#if defined(CPH_M1RX_CAN_BUILD_AVX2)
#if defined(CPH_M1RX_HAS_TARGET_ATTR)
    return (__builtin_cpu_supports("avx2") != 0);
#else
    return TRUE;
#endif
#else
    return FALSE;
#endif
}

FORCEINLINE
BOOLEAN
CphMulshrolate1RXCanUseAvx512(
    VOID
    )
{
#if defined(CPH_M1RX_CAN_BUILD_AVX512)
#if defined(CPH_M1RX_HAS_TARGET_ATTR)
    return (__builtin_cpu_supports("avx512f") != 0);
#else
    return TRUE;
#endif
#else
    return FALSE;
#endif
}

CPHAPI
VOID
CPHCALLTYPE
INDEX32X8_ROUTINE(
    _In_ CPHKEY Key1,
    _In_ CPHKEY Key2,
    _In_ CPHKEY Key3,
    _In_ CPHKEY Key4,
    _In_ CPHKEY Key5,
    _In_ CPHKEY Key6,
    _In_ CPHKEY Key7,
    _In_ CPHKEY Key8,
    _Out_ PCPHINDEX Index1,
    _Out_ PCPHINDEX Index2,
    _Out_ PCPHINDEX Index3,
    _Out_ PCPHINDEX Index4,
    _Out_ PCPHINDEX Index5,
    _Out_ PCPHINDEX Index6,
    _Out_ PCPHINDEX Index7,
    _Out_ PCPHINDEX Index8
    )
{
    ULONG Keys[8];
    CPHDKEY Vertex1[8];
    CPHDKEY Vertex2[8];

    Keys[0] = DOWNSIZE_KEY(Key1);
    Keys[1] = DOWNSIZE_KEY(Key2);
    Keys[2] = DOWNSIZE_KEY(Key3);
    Keys[3] = DOWNSIZE_KEY(Key4);
    Keys[4] = DOWNSIZE_KEY(Key5);
    Keys[5] = DOWNSIZE_KEY(Key6);
    Keys[6] = DOWNSIZE_KEY(Key7);
    Keys[7] = DOWNSIZE_KEY(Key8);

    if (CphMulshrolate1RXCanUseAvx2()) {
#if defined(CPH_M1RX_CAN_BUILD_AVX2)
        CphMulshrolate1RXComputeVertices8Avx2(Keys, Vertex1, Vertex2);
#endif
    } else {
        CphMulshrolate1RXComputeVerticesScalar(Keys, Vertex1, Vertex2, 8);
    }

    *Index1 = CphMulshrolate1RXFinalizeIndex(Vertex1[0], Vertex2[0]);
    *Index2 = CphMulshrolate1RXFinalizeIndex(Vertex1[1], Vertex2[1]);
    *Index3 = CphMulshrolate1RXFinalizeIndex(Vertex1[2], Vertex2[2]);
    *Index4 = CphMulshrolate1RXFinalizeIndex(Vertex1[3], Vertex2[3]);
    *Index5 = CphMulshrolate1RXFinalizeIndex(Vertex1[4], Vertex2[4]);
    *Index6 = CphMulshrolate1RXFinalizeIndex(Vertex1[5], Vertex2[5]);
    *Index7 = CphMulshrolate1RXFinalizeIndex(Vertex1[6], Vertex2[6]);
    *Index8 = CphMulshrolate1RXFinalizeIndex(Vertex1[7], Vertex2[7]);
}

CPHAPI
VOID
CPHCALLTYPE
INDEX32X16_ROUTINE(
    _In_ CPHKEY Key1,
    _In_ CPHKEY Key2,
    _In_ CPHKEY Key3,
    _In_ CPHKEY Key4,
    _In_ CPHKEY Key5,
    _In_ CPHKEY Key6,
    _In_ CPHKEY Key7,
    _In_ CPHKEY Key8,
    _In_ CPHKEY Key9,
    _In_ CPHKEY Key10,
    _In_ CPHKEY Key11,
    _In_ CPHKEY Key12,
    _In_ CPHKEY Key13,
    _In_ CPHKEY Key14,
    _In_ CPHKEY Key15,
    _In_ CPHKEY Key16,
    _Out_ PCPHINDEX Index1,
    _Out_ PCPHINDEX Index2,
    _Out_ PCPHINDEX Index3,
    _Out_ PCPHINDEX Index4,
    _Out_ PCPHINDEX Index5,
    _Out_ PCPHINDEX Index6,
    _Out_ PCPHINDEX Index7,
    _Out_ PCPHINDEX Index8,
    _Out_ PCPHINDEX Index9,
    _Out_ PCPHINDEX Index10,
    _Out_ PCPHINDEX Index11,
    _Out_ PCPHINDEX Index12,
    _Out_ PCPHINDEX Index13,
    _Out_ PCPHINDEX Index14,
    _Out_ PCPHINDEX Index15,
    _Out_ PCPHINDEX Index16
    )
{
    ULONG Keys[16];
    CPHDKEY Vertex1[16];
    CPHDKEY Vertex2[16];

    Keys[0] = DOWNSIZE_KEY(Key1);
    Keys[1] = DOWNSIZE_KEY(Key2);
    Keys[2] = DOWNSIZE_KEY(Key3);
    Keys[3] = DOWNSIZE_KEY(Key4);
    Keys[4] = DOWNSIZE_KEY(Key5);
    Keys[5] = DOWNSIZE_KEY(Key6);
    Keys[6] = DOWNSIZE_KEY(Key7);
    Keys[7] = DOWNSIZE_KEY(Key8);
    Keys[8] = DOWNSIZE_KEY(Key9);
    Keys[9] = DOWNSIZE_KEY(Key10);
    Keys[10] = DOWNSIZE_KEY(Key11);
    Keys[11] = DOWNSIZE_KEY(Key12);
    Keys[12] = DOWNSIZE_KEY(Key13);
    Keys[13] = DOWNSIZE_KEY(Key14);
    Keys[14] = DOWNSIZE_KEY(Key15);
    Keys[15] = DOWNSIZE_KEY(Key16);

    if (CphMulshrolate1RXCanUseAvx512()) {
#if defined(CPH_M1RX_CAN_BUILD_AVX512)
        CphMulshrolate1RXComputeVertices16Avx512(Keys, Vertex1, Vertex2);
#endif
    } else {
        CphMulshrolate1RXComputeVerticesScalar(Keys, Vertex1, Vertex2, 16);
    }

    *Index1 = CphMulshrolate1RXFinalizeIndex(Vertex1[0], Vertex2[0]);
    *Index2 = CphMulshrolate1RXFinalizeIndex(Vertex1[1], Vertex2[1]);
    *Index3 = CphMulshrolate1RXFinalizeIndex(Vertex1[2], Vertex2[2]);
    *Index4 = CphMulshrolate1RXFinalizeIndex(Vertex1[3], Vertex2[3]);
    *Index5 = CphMulshrolate1RXFinalizeIndex(Vertex1[4], Vertex2[4]);
    *Index6 = CphMulshrolate1RXFinalizeIndex(Vertex1[5], Vertex2[5]);
    *Index7 = CphMulshrolate1RXFinalizeIndex(Vertex1[6], Vertex2[6]);
    *Index8 = CphMulshrolate1RXFinalizeIndex(Vertex1[7], Vertex2[7]);
    *Index9 = CphMulshrolate1RXFinalizeIndex(Vertex1[8], Vertex2[8]);
    *Index10 = CphMulshrolate1RXFinalizeIndex(Vertex1[9], Vertex2[9]);
    *Index11 = CphMulshrolate1RXFinalizeIndex(Vertex1[10], Vertex2[10]);
    *Index12 = CphMulshrolate1RXFinalizeIndex(Vertex1[11], Vertex2[11]);
    *Index13 = CphMulshrolate1RXFinalizeIndex(Vertex1[12], Vertex2[12]);
    *Index14 = CphMulshrolate1RXFinalizeIndex(Vertex1[13], Vertex2[13]);
    *Index15 = CphMulshrolate1RXFinalizeIndex(Vertex1[14], Vertex2[14]);
    *Index16 = CphMulshrolate1RXFinalizeIndex(Vertex1[15], Vertex2[15]);
}

DECLARE_INDEX_IACA_ROUTINE()
{
    CPHINDEX Index;
    CPHDKEY Vertex1;
    CPHDKEY Vertex2;
    CPHDKEY DownsizedKey;

    IACA_VC_START();

    DownsizedKey = DOWNSIZE_KEY(Key);

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = RotateRight32(Vertex1, SEED3_BYTE2);
    Vertex1 = Vertex1 >> SEED3_BYTE1;
    Vertex1 = TABLE_DATA[Vertex1];

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE1;
    Vertex2 = TABLE_DATA[Vertex2];

    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);

    IACA_VC_END();

    return Index;
}

#endif
