/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    GraphNeon.c

Abstract:

    Four-lane ARM64 key hashing for the MultiplyShiftR and MultiplyShiftRX
    graph solvers.

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

_Use_decl_annotations_
HRESULT
GraphHashKeysMultiplyShift_NEON(
    PGRAPH Graph,
    ULONG NumberOfKeys,
    PKEY Keys
    )
{
    PPERFECT_HASH_TABLE Table = Graph->Context->Table;
    BOOL Assigned16 = IsUsingAssigned16(Graph);
    ULONG Mask = Table->HashMask;
    ULONG Seed1 = Graph->Seed1;
    ULONG Seed2 = Graph->Seed2;
    ULONG_BYTES Seed3;
    ULONG Shift1, Shift2;
    uint32x4_t MaskVector;
    int32x4_t ShiftVector1, ShiftVector2;
    ULONG VectorEnd = NumberOfKeys & ~3u;
    EDGE Edge;
    KEY Key = NumberOfKeys ? Keys[NumberOfKeys - 1] : 0;
    HRESULT Result = S_OK;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    Seed3.AsULong = Graph->Seed3;
    Shift1 = Seed3.Byte1 & 31;
    Shift2 = Seed3.Byte2 & 31;
    if (Table->HashFunctionId == PerfectHashHashMultiplyShiftRXFunctionId) {
        // RX uses the first shift twice and only truncates to the vertex width.
        Shift2 = Shift1;
        Mask = Assigned16 ? 0xffffu : 0xffffffffu;
    }
    MaskVector = vdupq_n_u32(Mask);
    ShiftVector1 = vdupq_n_s32(-(int32_t)Shift1);
    ShiftVector2 = vdupq_n_s32(-(int32_t)Shift2);

    Graph->Flags.UsedNeonHashFunction = (VectorEnd != 0);
    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < VectorEnd; Edge += 4) {
        uint32x4_t Input = vld1q_u32((const uint32_t *)&Keys[Edge]);
        uint32x4_t Vertex1 = vandq_u32(MaskVector,
            vshlq_u32(vmulq_n_u32(Input, Seed1), ShiftVector1));
        uint32x4_t Vertex2 = vandq_u32(MaskVector,
            vshlq_u32(vmulq_n_u32(Input, Seed2), ShiftVector2));

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
        ULONG Vertex1, Vertex2;
        Key = Keys[Edge];
        Vertex1 = ((Key * Seed1) >> Shift1) & Mask;
        Vertex2 = ((Key * Seed2) >> Shift2) & Mask;
        if (Vertex1 == Vertex2) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            break;
        }
        if (Assigned16) {
            Graph->Vertex16Pairs[Edge].Vertex1 = (USHORT)Vertex1;
            Graph->Vertex16Pairs[Edge].Vertex2 = (USHORT)Vertex2;
        } else {
            Graph->VertexPairs[Edge].Vertex1 = Vertex1;
            Graph->VertexPairs[Edge].Vertex2 = Vertex2;
        }
    }

    STOP_GRAPH_COUNTER(HashKeys);
    EVENT_WRITE_GRAPH(HashKeys);
    return GraphPostHashKeys(Result, Graph);
}

#endif
