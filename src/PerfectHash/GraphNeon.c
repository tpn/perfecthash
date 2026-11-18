/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    GraphNeon.c

Abstract:

    Four-lane ARM64 key hashing for the MultiplyShift and Mulshrolate solvers.

--*/

#include "stdafx.h"

#ifdef PH_HAS_ARM64_NEON

#include <arm_neon.h>
#include "PerfectHashEventsPrivate.h"

#ifdef PH_LINUX
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

BOOL
GraphHasNeon(VOID)
{
#if defined(PH_WINDOWS)
    return IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE);
#elif defined(PH_LINUX)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#else
    // Advanced SIMD is part of the Apple arm64 ABI.
    return TRUE;
#endif
}

// Each Mulshrolate variant adds one operation to the previous variant.
// A constant step count lets each vector loop omit unused rotates and multiplies.
FORCEINLINE
HRESULT
HashKeysNeon(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys,
    ULONG MulshrolateSteps
    )
{
    PPERFECT_HASH_TABLE Table = Graph->Context->Table;
    BOOL Assigned16 = IsUsingAssigned16(Graph);
    ULONG Mask = Table->HashMask;
    ULONG Seed1 = Graph->Seed1;
    ULONG Seed2 = Graph->Seed2;
    ULONG Seed4 = MulshrolateSteps >= 3 ? Graph->Seed4 : 0;
    ULONG Seed5 = MulshrolateSteps >= 4 ? Graph->Seed5 : 0;
    ULONG_BYTES Seed3;
    ULONG Shift1, Shift2;
    uint32x4_t MaskVector;
    int32x4_t ShiftVector1, ShiftVector2;
    int32x4_t RotateRight1, RotateLeft1, RotateRight2, RotateLeft2;
    ULONG VectorEnd = NumberOfKeys & ~3u;
    EDGE Edge;
    KEY Key = NumberOfKeys ? Keys[NumberOfKeys - 1] : 0;
    HRESULT Result = S_OK;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    Seed3.AsULong = Graph->Seed3;
    Shift1 = Seed3.Byte1 & 31;
    Shift2 = Seed3.Byte2 & 31;
    if (Table->HashFunctionId != PerfectHashHashMultiplyShiftRFunctionId) {
        // RX uses the first shift twice and only truncates to the vertex width.
        Shift2 = Shift1;
        Mask = Assigned16 ? 0xffffu : 0xffffffffu;
    }
    MaskVector = vdupq_n_u32(Mask);
    ShiftVector1 = vdupq_n_s32(-(int32_t)Shift1);
    ShiftVector2 = vdupq_n_s32(-(int32_t)Shift2);
    RotateRight1 = vdupq_n_s32(-(int32_t)(Seed3.Byte2 & 31));
    RotateRight2 = vdupq_n_s32(-(int32_t)(Seed3.Byte3 & 31));
    RotateLeft1 = vaddq_s32(RotateRight1, vdupq_n_s32(32));
    RotateLeft2 = vaddq_s32(RotateRight2, vdupq_n_s32(32));

    Graph->Flags.UsedNeonHashFunction = (VectorEnd != 0);
    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < VectorEnd; Edge += 4) {
        uint32x4_t Input = vld1q_u32((const uint32_t *)&Keys[Edge]);
        uint32x4_t Vertex1 = vmulq_n_u32(Input, Seed1);
        uint32x4_t Vertex2 = vmulq_n_u32(Input, Seed2);

        if (MulshrolateSteps >= 1) {
            Vertex1 = vorrq_u32(vshlq_u32(Vertex1, RotateRight1),
                                vshlq_u32(Vertex1, RotateLeft1));
        }
        if (MulshrolateSteps >= 2) {
            Vertex2 = vorrq_u32(vshlq_u32(Vertex2, RotateRight2),
                                vshlq_u32(Vertex2, RotateLeft2));
        }
        if (MulshrolateSteps >= 3) {
            Vertex1 = vmulq_n_u32(Vertex1, Seed4);
        }
        if (MulshrolateSteps >= 4) {
            Vertex2 = vmulq_n_u32(Vertex2, Seed5);
        }
        Vertex1 = vandq_u32(MaskVector, vshlq_u32(Vertex1, ShiftVector1));
        Vertex2 = vandq_u32(MaskVector, vshlq_u32(Vertex2, ShiftVector2));

        if (vmaxvq_u32(vceqq_u32(Vertex1, Vertex2))) {
            // Replay this group below to report the first colliding key.
            break;
        }
        if (Assigned16) {
            vst1q_u32((uint32_t *)&Graph->Vertex16Pairs[Edge],
                      vsliq_n_u32(Vertex1, Vertex2, 16));
        } else {
            uint32x4x2_t Pairs;
            Pairs.val[0] = Vertex1;
            Pairs.val[1] = Vertex2;
            vst2q_u32((uint32_t *)&Graph->VertexPairs[Edge], Pairs);
        }
    }

    for (; Edge < NumberOfKeys; Edge++) {
        Key = Keys[Edge];
        if (Assigned16) {
            VERTEX16_PAIR Pair;
            Pair.AsULong = SeededHash16ExRoutines[Table->HashFunctionId](
                Key, &Graph->FirstSeed, (USHORT)Table->HashMask);
            if (Pair.Vertex1 == Pair.Vertex2) {
                Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
                break;
            }
            Graph->Vertex16Pairs[Edge] = Pair;
        } else {
            VERTEX_PAIR Pair;
            Pair.AsULongLong = SeededHashExRoutines[Table->HashFunctionId](
                Key, &Graph->FirstSeed, Table->HashMask);
            if (Pair.Vertex1 == Pair.Vertex2) {
                Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
                break;
            }
            Graph->VertexPairs[Edge] = Pair;
        }
    }

    STOP_GRAPH_COUNTER(HashKeys);
    EVENT_WRITE_GRAPH(HashKeys);
    return GraphPostHashKeys(Result, Graph);
}

_Use_decl_annotations_
HRESULT
GraphHashKeys_NEON(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
{
    switch ((ULONG)Graph->Context->Table->HashFunctionId) {
        case PerfectHashHashMultiplyShiftRFunctionId:
        case PerfectHashHashMultiplyShiftRXFunctionId:
            return HashKeysNeon(Graph, NumberOfKeys, Keys, 0);
        case PerfectHashHashMulshrolate1RXFunctionId:
            return HashKeysNeon(Graph, NumberOfKeys, Keys, 1);
        case PerfectHashHashMulshrolate2RXFunctionId:
            return HashKeysNeon(Graph, NumberOfKeys, Keys, 2);
        case PerfectHashHashMulshrolate3RXFunctionId:
            return HashKeysNeon(Graph, NumberOfKeys, Keys, 3);
        case PerfectHashHashMulshrolate4RXFunctionId:
            return HashKeysNeon(Graph, NumberOfKeys, Keys, 4);
        default:
            return IsUsingAssigned16(Graph) ? GraphHashKeys16(Graph, NumberOfKeys, Keys) :
                                              GraphHashKeys(Graph, NumberOfKeys, Keys);
    }
}

#endif
