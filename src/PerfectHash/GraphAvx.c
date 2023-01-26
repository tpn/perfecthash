/*++

Copyright (c) 2022-2023 Trent Nelson <trent@trent.me>

Module Name:

    GraphAvx.c

Abstract:

    This module implements various AVX-optimized routines related to graphs.
    This includes AVX, AVX2, and AVX-512.

--*/

#include "stdafx.h"
#include "PerfectHashEventsPrivate.h"

_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
GraphHashKeysMultiplyShiftR_AVX2(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PKEY Keys
    )
/*++

Routine Description:

    This routine hashes all keys into vertices without adding the resulting
    vertices to the graph.  It is used by GraphHashKeysThenAdd().

Arguments:

    Graph - Supplies a pointer to the graph for which the hash values will be
        created.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

--*/
{
    KEY Key = 0;
    EDGE Edge;
    ULONG Mask;
    ULONG NumberOfYmmWords;
    ULONG TrailingKeys;
    PULONG Seeds;
    PEDGE Edges;
    HRESULT Result;
    PULONGLONG VertexPairs;
    PPERFECT_HASH_TABLE Table;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    YMMWORD Ymm1;
    YMMWORD Ymm2;
    YMMWORD Ymm3;
    YMMWORD Imm0;
    YMMWORD Imm1;
    YMMWORD HashMaskYmm;
    YMMWORD KeysYmm;
    YMMWORD Seed1Ymm;
    YMMWORD Seed2Ymm;
    YMMWORD Vertex1Ymm;
    YMMWORD Vertex2Ymm;
    YMMWORD VertexCompareYmm;
    LONG VertexCompareMask;
    ULARGE_INTEGER Pair;
    PYMMWORD VertexPairsYmm;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Initialize aliases.
    //

    Result = S_OK;
    Table = Graph->Context->Table;
    Mask = Table->HashMask;
    Edges = (PEDGE)Keys;
    C_ASSERT(sizeof(*VertexPairs) == sizeof(Graph->VertexPairs));
    VertexPairs = (PULONGLONG)Graph->VertexPairs;
    VertexPairsYmm = (PYMMWORD)VertexPairs;;

    //
    // Determine the number of YMM words we'll use to iterate over the keys,
    // 8 x 32-bit keys at a time.  Capture the number of trailing keys that
    // we'll handle at the end with a scalar loop.
    //

    NumberOfYmmWords = NumberOfKeys >> 3;
    TrailingKeys = NumberOfKeys % 8;

    //
    // Initialize seeds.
    //

    Seeds = &Graph->FirstSeed;
    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];
    Seed1Ymm = _mm256_broadcastd_epi32(_mm_set1_epi32(Seed1));
    Seed2Ymm = _mm256_broadcastd_epi32(_mm_set1_epi32(Seed2));

    //
    // Broadcast the hash across 8 x 32-bit YMM lanes.
    //

    HashMaskYmm = _mm256_broadcastd_epi32(_mm_set1_epi32(Mask));

    //
    // Initialize the permute control variables (described below).
    //

    Imm0 = _mm256_setr_epi32(0,  0,  1,  1,  2,  2,  3,  3);
    Imm1 = _mm256_setr_epi32(4,  4,  5,  5,  6,  6,  7,  7);

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfYmmWords; Edge++, Edges += 8) {

        //IACA_VC_START();

        //
        // Load 8 keys into our YMM register.
        //

        KeysYmm = _mm256_loadu_epi32(Edges);

        //
        // Perform vectorized multiply, shift, then and-masking against 8
        // keys at a time, once for each vertex.
        //

        //
        // Vertex 1: (((Key * SEED1) >> SEED3_BYTE1) & Mask)
        //

        Vertex1Ymm = _mm256_mullo_epi32(KeysYmm, Seed1Ymm);
        Vertex1Ymm = _mm256_srli_epi32(Vertex1Ymm, Seed3.Byte1);
        Vertex1Ymm = _mm256_and_epi32(Vertex1Ymm, HashMaskYmm);

        //
        // Vertex 2: (((Key * SEED2) >> SEED3_BYTE2) & Mask)
        //

        Vertex2Ymm = _mm256_mullo_epi32(KeysYmm, Seed2Ymm);
        Vertex2Ymm = _mm256_srli_epi32(Vertex2Ymm, Seed3.Byte2);
        Vertex2Ymm = _mm256_and_epi32(Vertex2Ymm, HashMaskYmm);

        //
        // Compare each pair of vertices against each other to see if there
        // are any conflicts (i.e. a key hashed to the same final vertex value
        // for both seeds).  If there are, abort, indicating vertex collision.
        //

        VertexCompareYmm = _mm256_cmpeq_epi32(Vertex1Ymm, Vertex2Ymm);
        VertexCompareMask = _mm256_movemask_epi8(VertexCompareYmm);
        if (VertexCompareMask > 0) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        }

        //
        // No collisions were detected, so we can save these vertices to memory.
        // We need to do this in two steps, with the help of SIMD permutation
        // and blend intrinsics, processing the first four 32-bit vertices in
        // the lower-half of each YMM register, and interleaving them such that
        // we get a third YMM register whose layout matches what we want to
        // write to memory.
        //
        // The first step is the SIMD permute8x32, and the second is a blend.
        // The permute8x32 is responsible for taking a vertex YMM register with
        // 8 values and yielding a new YMM register with only four values, but
        // with each value repeated twice, e.g.:
        //
        //      # Ymm1 = permute8x32_epi32(Vertex1Ymm, Imm0)
        //      #   Where: Imm0 = (0, 0, 1, 1, 2, 2, 3, 3);
        //      Ymm1 = [
        //          (Vertex1Ymm[0], Vertex1Ymm[0]),
        //          (Vertex1Ymm[1], Vertex1Ymm[1]),
        //          (Vertex1Ymm[2], Vertex1Ymm[2]),
        //          (Vertex1Ymm[3], Vertex1Ymm[3]),
        //      ]
        //
        //      # Ymm2 = permute8x32_epi32(Vertex2Ymm, Imm0)
        //      #   Where: Imm0 = (0, 0, 1, 1, 2, 2, 3, 3);
        //      Ymm2 = [
        //          (Vertex1Ymm[0], Vertex1Ymm[0]),
        //          (Vertex1Ymm[1], Vertex1Ymm[1]),
        //          (Vertex1Ymm[2], Vertex1Ymm[2]),
        //          (Vertex1Ymm[3], Vertex1Ymm[3]),
        //      ]
        //
        //  N.B. We don't actually care about the second values in Ymm1 and the
        //       first values in Ymm2, i.e., if we look at the final values that
        //       actually get used, the Ymm1 and Ymm2 registers would look like
        //       this:
        //          Ymm1 = [
        //              (Vertex1Ymm[0], _),
        //              (Vertex1Ymm[1], _),
        //              (Vertex1Ymm[2], _),
        //              (Vertex1Ymm[3], _),
        //          ]
        //          Ymm2 = [
        //              (_, Vertex1Ymm[0]),
        //              (_, Vertex1Ymm[1]),
        //              (_, Vertex1Ymm[2]),
        //              (_, Vertex1Ymm[3]),
        //          ]
        //
        //  The second step of the permutation is to blend these resulting Ymm1
        //  and Ymm2 registers into a third Ymm3 register, which consists of the
        //  (Vertex1, Vertex2) pairs that we want to write to memory, e.g.:
        //
        //      # Ymm3 = blend_epi32(Ymm1, Ymm2, 0xaa)
        //      #   Where 0xaa = '0b10101010', e.g. alternate between each input
        //      #   YMM register.
        //      Ymm3 = [
        //          (Ymm1[0], Ymm2[0]),
        //          (Ymm1[1], Ymm2[1]),
        //          (Ymm1[2], Ymm2[2]),
        //          (Ymm1[3], Ymm2[3]),
        //      ]
        //      *VertexPairsYmm++ = Ymm3
        //
        //  The process is repeated for the last four 32-bit vertices, using
        //  Imm1, where Imm1 = (4, 4, 5, 5, 6, 6, 7, 7), and the results are
        //  saved to the next 8 byte location.  Thus, each 32-bit key results
        //  in a 64-bit "vertex pair" consisting of (vertex1, vertex2) being
        //  written to memory.
        //

        //
        // Permute, blend, then store the first four vertex pairs.
        //

        Ymm1 = _mm256_permutevar8x32_epi32(Vertex1Ymm, Imm0);
        Ymm2 = _mm256_permutevar8x32_epi32(Vertex2Ymm, Imm0);
        Ymm3 = _mm256_blend_epi32(Ymm1, Ymm2, 0xaa);
        _mm256_storeu_epi32(VertexPairsYmm, Ymm3);
        VertexPairsYmm++;

        //
        // Permute, blend, then store the last four vertex pairs.
        //

        Ymm1 = _mm256_permutevar8x32_epi32(Vertex1Ymm, Imm1);
        Ymm2 = _mm256_permutevar8x32_epi32(Vertex2Ymm, Imm1);
        Ymm3 = _mm256_blend_epi32(Ymm1, Ymm2, 0xaa);
        _mm256_storeu_epi32(VertexPairsYmm, Ymm3);
        VertexPairsYmm++;

        //IACA_VC_END();
    }

    if (TrailingKeys > 0) {

        //
        // Handle the remaining 1-7 keys using a normal, non-SIMD loop.
        //

        VertexPairs = (PULONGLONG)VertexPairsYmm;

        for (Edge = 0; Edge < TrailingKeys; Edge++) {
            Key = *Edges++;

            Vertex1 = Key * Seed1;
            Vertex1 = Vertex1 >> Seed3.Byte1;;
            Vertex1 = Vertex1 & Mask;

            Vertex2 = Key * Seed2;
            Vertex2 = Vertex2 >> Seed3.Byte2;
            Vertex2 = Vertex2 & Mask;

            if (Vertex1 == Vertex2) {
                Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
                goto End;
            }

            Pair.LowPart = Vertex1;
            Pair.HighPart = Vertex2;

            *VertexPairs++ = Pair.QuadPart;
        }
    }

End:

    STOP_GRAPH_COUNTER(HashKeys);

    EVENT_WRITE_GRAPH(HashKeys);

    Result = GraphPostHashKeys(Result, Graph);

    return Result;
}


_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
GraphHashKeysMultiplyShiftR_AVX512(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PKEY Keys
    )
/*++

Routine Description:

    This routine hashes all keys into vertices without adding the resulting
    vertices to the graph.  It is used by GraphHashKeysThenAdd().

Arguments:

    Graph - Supplies a pointer to the graph for which the hash values will be
        created.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

--*/
{
    KEY Key = 0;
    EDGE Edge;
    ULONG Mask;
    ULONG NumberOfZmmWords;
    ULONG TrailingKeys;
    PULONG Seeds;
    PEDGE Edges;
    HRESULT Result;
    PPERFECT_HASH_TABLE Table;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ZMMWORD Zmm1;
    ZMMWORD Zmm2;
    ZMASK16 ZmmMask;
    ZMMWORD KeysZmm;
    ZMMWORD Seed1Zmm;
    ZMMWORD Seed2Zmm;
    ZMMWORD Index1Zmm;
    ZMMWORD Index2Zmm;
    ZMMWORD Vertex1Zmm;
    ZMMWORD Vertex2Zmm;
    ZMMWORD HashMaskZmm;
    ULARGE_INTEGER Pair;
    PULONGLONG VertexPairs;
    PZMMWORD VertexPairsZmm;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // The following structures are fed to _mm512_permutex2var_epi32() below.
    // They form the index tables that direct the SIMD permute operation to
    // unpack and interleave our two vertex registers into the final vertex pair
    // form we want to write to memory.
    //

    ZMM_PERMUTEX2VAR_INDEX32 PermuteIndex1[] = {
        { .Index = 0, .Selector = 0, .Unused = 0 },
        { .Index = 0, .Selector = 1, .Unused = 0 },

        { .Index = 1, .Selector = 0, .Unused = 0 },
        { .Index = 1, .Selector = 1, .Unused = 0 },

        { .Index = 2, .Selector = 0, .Unused = 0 },
        { .Index = 2, .Selector = 1, .Unused = 0 },

        { .Index = 3, .Selector = 0, .Unused = 0 },
        { .Index = 3, .Selector = 1, .Unused = 0 },

        { .Index = 4, .Selector = 0, .Unused = 0 },
        { .Index = 4, .Selector = 1, .Unused = 0 },

        { .Index = 5, .Selector = 0, .Unused = 0 },
        { .Index = 5, .Selector = 1, .Unused = 0 },

        { .Index = 6, .Selector = 0, .Unused = 0 },
        { .Index = 6, .Selector = 1, .Unused = 0 },

        { .Index = 7, .Selector = 0, .Unused = 0 },
        { .Index = 7, .Selector = 1, .Unused = 0 },
    };

    ZMM_PERMUTEX2VAR_INDEX32 PermuteIndex2[] = {
        { .Index = 8, .Selector = 0, .Unused = 0 },
        { .Index = 8, .Selector = 1, .Unused = 0 },

        { .Index = 9, .Selector = 0, .Unused = 0 },
        { .Index = 9, .Selector = 1, .Unused = 0 },

        { .Index = 10, .Selector = 0, .Unused = 0 },
        { .Index = 10, .Selector = 1, .Unused = 0 },

        { .Index = 11, .Selector = 0, .Unused = 0 },
        { .Index = 11, .Selector = 1, .Unused = 0 },

        { .Index = 12, .Selector = 0, .Unused = 0 },
        { .Index = 12, .Selector = 1, .Unused = 0 },

        { .Index = 13, .Selector = 0, .Unused = 0 },
        { .Index = 13, .Selector = 1, .Unused = 0 },

        { .Index = 14, .Selector = 0, .Unused = 0 },
        { .Index = 14, .Selector = 1, .Unused = 0 },

        { .Index = 15, .Selector = 0, .Unused = 0 },
        { .Index = 15, .Selector = 1, .Unused = 0 },
    };

    //
    // Initialize aliases.
    //

    Result = S_OK;
    Table = Graph->Context->Table;
    Mask = Table->HashMask;
    Edges = (PEDGE)Keys;
    C_ASSERT(sizeof(*VertexPairs) == sizeof(Graph->VertexPairs));
    VertexPairs = (PULONGLONG)Graph->VertexPairs;
    VertexPairsZmm = (PZMMWORD)VertexPairs;;

    //
    // Determine the number of ZMM words we'll use to iterate over the keys,
    // 16 x 32-bit keys at a time.  Capture the number of trailing keys that
    // we'll handle at the end with a scalar loop.
    //

    NumberOfZmmWords = NumberOfKeys >> 4;
    TrailingKeys = NumberOfKeys % 16;

    //
    // Initialize seeds.
    //

    Seeds = &Graph->FirstSeed;
    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];
    Seed1Zmm = _mm512_broadcastd_epi32(_mm_set1_epi32(Seed1));
    Seed2Zmm = _mm512_broadcastd_epi32(_mm_set1_epi32(Seed2));

    //
    // Broadcast the hash across 16 x 32-bit ZMM lanes.
    //

    HashMaskZmm = _mm512_broadcastd_epi32(_mm_set1_epi32(Mask));

    //
    // Load the permute indexes.
    //

    Index1Zmm = _mm512_loadu_epi32(&PermuteIndex1[0]);
    Index2Zmm = _mm512_loadu_epi32(&PermuteIndex2[0]);

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfZmmWords; Edge++, Edges += 16) {

        //IACA_VC_START();

        //
        // Load 16 keys into our ZMM register.
        //

        KeysZmm = _mm512_loadu_epi32(Edges);

        //
        // Perform vectorized multiply, shift, then and-masking against 16
        // keys at a time, once for each vertex.
        //

        //
        // Vertex 1: (((Key * SEED1) >> SEED3_BYTE1) & Mask)
        //

        Vertex1Zmm = _mm512_mullo_epi32(KeysZmm, Seed1Zmm);
        Vertex1Zmm = _mm512_srli_epi32(Vertex1Zmm, Seed3.Byte1);
        Vertex1Zmm = _mm512_and_epi32(Vertex1Zmm, HashMaskZmm);

        //
        // Vertex 2: (((Key * SEED2) >> SEED3_BYTE2) & Mask)
        //

        Vertex2Zmm = _mm512_mullo_epi32(KeysZmm, Seed2Zmm);
        Vertex2Zmm = _mm512_srli_epi32(Vertex2Zmm, Seed3.Byte2);
        Vertex2Zmm = _mm512_and_epi32(Vertex2Zmm, HashMaskZmm);

        //
        // Compare each pair of vertices against each other to see if there
        // are any conflicts (i.e. a key hashed to the same final vertex value
        // for both seeds).  If there are, abort, indicating vertex collision.
        //

        ZmmMask = _mm512_cmpeq_epi32_mask(Vertex1Zmm, Vertex2Zmm);
        if (ZmmMask > 0) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        }

        //
        // No collisions were detected, so we can save these vertices to memory.
        // This is a lot easier in AVX512 versus AVX2; the vperm2td instruction
        // allows us to unpack and interleave the values from each vertex in a
        // single instruction (versus the two permutes then blend we need for
        // AVX2).
        //

        //
        // Unpack-and-interleave the first 8 vertex pairs, then store.
        //

        Zmm1 = _mm512_permutex2var_epi32(Vertex1Zmm, Index1Zmm, Vertex2Zmm);
        _mm512_storeu_epi32(VertexPairsZmm, Zmm1);
        VertexPairsZmm++;

        //
        // Unpack-and-interleave the second 8 vertex pairs, then store.
        //

        Zmm2 = _mm512_permutex2var_epi32(Vertex1Zmm, Index2Zmm, Vertex2Zmm);
        _mm512_storeu_epi32(VertexPairsZmm, Zmm2);
        VertexPairsZmm++;

        //IACA_VC_END();
    }

    if (TrailingKeys > 0) {

        //
        // Handle the remaining 1-15 keys using a normal, non-SIMD loop.
        //

        VertexPairs = (PULONGLONG)VertexPairsZmm;

        for (Edge = 0; Edge < TrailingKeys; Edge++) {
            Key = *Edges++;

            Vertex1 = Key * Seed1;
            Vertex1 = Vertex1 >> Seed3.Byte1;;
            Vertex1 = Vertex1 & Mask;

            Vertex2 = Key * Seed2;
            Vertex2 = Vertex2 >> Seed3.Byte2;
            Vertex2 = Vertex2 & Mask;

            if (Vertex1 == Vertex2) {
                Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
                goto End;
            }

            Pair.LowPart = Vertex1;
            Pair.HighPart = Vertex2;

            *VertexPairs++ = Pair.QuadPart;
        }
    }

End:

    STOP_GRAPH_COUNTER(HashKeys);

    EVENT_WRITE_GRAPH(HashKeys);

    Result = GraphPostHashKeys(Result, Graph);

    return Result;
}

_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
GraphHashKeysMultiplyShiftRX_AVX2(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PKEY Keys
    )
/*++

Routine Description:

    This routine hashes all keys into vertices without adding the resulting
    vertices to the graph.  It is used by GraphHashKeysThenAdd().

Arguments:

    Graph - Supplies a pointer to the graph for which the hash values will be
        created.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

--*/
{
    KEY Key = 0;
    EDGE Edge;
    ULONG NumberOfYmmWords;
    ULONG TrailingKeys;
    PULONG Seeds;
    PEDGE Edges;
    HRESULT Result;
    PULONGLONG VertexPairs;
    PPERFECT_HASH_TABLE Table;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    YMMWORD Ymm1;
    YMMWORD Ymm2;
    YMMWORD Ymm3;
    YMMWORD Imm0;
    YMMWORD Imm1;
    YMMWORD KeysYmm;
    YMMWORD Seed1Ymm;
    YMMWORD Seed2Ymm;
    YMMWORD Vertex1Ymm;
    YMMWORD Vertex2Ymm;
    YMMWORD VertexCompareYmm;
    LONG VertexCompareMask;
    ULARGE_INTEGER Pair;
    PYMMWORD VertexPairsYmm;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Initialize aliases.
    //

    Result = S_OK;
    Table = Graph->Context->Table;
    Edges = (PEDGE)Keys;
    C_ASSERT(sizeof(*VertexPairs) == sizeof(Graph->VertexPairs));
    VertexPairs = (PULONGLONG)Graph->VertexPairs;
    VertexPairsYmm = (PYMMWORD)VertexPairs;;

    //
    // Determine the number of YMM words we'll use to iterate over the keys,
    // 8 x 32-bit keys at a time.  Capture the number of trailing keys that
    // we'll handle at the end with a scalar loop.
    //

    NumberOfYmmWords = NumberOfKeys >> 3;
    TrailingKeys = NumberOfKeys % 8;

    //
    // Initialize seeds.
    //

    Seeds = &Graph->FirstSeed;
    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];
    Seed1Ymm = _mm256_broadcastd_epi32(_mm_set1_epi32(Seed1));
    Seed2Ymm = _mm256_broadcastd_epi32(_mm_set1_epi32(Seed2));

    //
    // Initialize the permute control variables (described below).
    //

    Imm0 = _mm256_setr_epi32(0,  0,  1,  1,  2,  2,  3,  3);
    Imm1 = _mm256_setr_epi32(4,  4,  5,  5,  6,  6,  7,  7);

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfYmmWords; Edge++, Edges += 8) {

        //IACA_VC_START();

        //
        // Load 8 keys into our YMM register.
        //

        KeysYmm = _mm256_loadu_epi32(Edges);

        //
        // Perform vectorized multiply and shift ops against 8 keys at a time,
        // once for each vertex.
        //

        //
        // Vertex 1: ((Key * SEED1) >> SEED3_BYTE1)
        //

        Vertex1Ymm = _mm256_mullo_epi32(KeysYmm, Seed1Ymm);
        Vertex1Ymm = _mm256_srli_epi32(Vertex1Ymm, Seed3.Byte1);

        //
        // Vertex 2: ((Key * SEED2) >> SEED3_BYTE2)
        //

        Vertex2Ymm = _mm256_mullo_epi32(KeysYmm, Seed2Ymm);
        Vertex2Ymm = _mm256_srli_epi32(Vertex2Ymm, Seed3.Byte2);

        //
        // Compare each pair of vertices against each other to see if there
        // are any conflicts (i.e. a key hashed to the same final vertex value
        // for both seeds).  If there are, abort, indicating vertex collision.
        //

        VertexCompareYmm = _mm256_cmpeq_epi32(Vertex1Ymm, Vertex2Ymm);
        VertexCompareMask = _mm256_movemask_epi8(VertexCompareYmm);
        if (VertexCompareMask > 0) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        }

        //
        // No collisions were detected, so we can save these vertices to memory.
        //
        // N.B. See the comment in GraphHashKeysMultiplyShiftR_AVX2() for a
        //      detailed explanation of the following logic.
        //

        //
        // Permute, blend, then store the first four vertex pairs.
        //

        Ymm1 = _mm256_permutevar8x32_epi32(Vertex1Ymm, Imm0);
        Ymm2 = _mm256_permutevar8x32_epi32(Vertex2Ymm, Imm0);
        Ymm3 = _mm256_blend_epi32(Ymm1, Ymm2, 0xaa);
        _mm256_storeu_epi32(VertexPairsYmm, Ymm3);
        VertexPairsYmm++;

        //
        // Permute, blend, then store the last four vertex pairs.
        //

        Ymm1 = _mm256_permutevar8x32_epi32(Vertex1Ymm, Imm1);
        Ymm2 = _mm256_permutevar8x32_epi32(Vertex2Ymm, Imm1);
        Ymm3 = _mm256_blend_epi32(Ymm1, Ymm2, 0xaa);
        _mm256_storeu_epi32(VertexPairsYmm, Ymm3);
        VertexPairsYmm++;

        //IACA_VC_END();
    }

    if (TrailingKeys > 0) {

        //
        // Handle the remaining 1-7 keys using a normal, non-SIMD loop.
        //

        VertexPairs = (PULONGLONG)VertexPairsYmm;

        for (Edge = 0; Edge < TrailingKeys; Edge++) {
            Key = *Edges++;

            Vertex1 = Key * Seed1;
            Vertex1 = Vertex1 >> Seed3.Byte1;;

            Vertex2 = Key * Seed2;
            Vertex2 = Vertex2 >> Seed3.Byte2;

            if (Vertex1 == Vertex2) {
                Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
                goto End;
            }

            Pair.LowPart = Vertex1;
            Pair.HighPart = Vertex2;

            *VertexPairs++ = Pair.QuadPart;
        }
    }

End:

    STOP_GRAPH_COUNTER(HashKeys);

    EVENT_WRITE_GRAPH(HashKeys);

    Result = GraphPostHashKeys(Result, Graph);

    return Result;
}

_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
GraphHashKeysMultiplyShiftRX_AVX512(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PKEY Keys
    )
/*++

Routine Description:

    This routine hashes all keys into vertices without adding the resulting
    vertices to the graph.  It is used by GraphHashKeysThenAdd().

Arguments:

    Graph - Supplies a pointer to the graph for which the hash values will be
        created.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

--*/
{
    KEY Key = 0;
    EDGE Edge;
    ULONG NumberOfZmmWords;
    ULONG TrailingKeys;
    PULONG Seeds;
    PEDGE Edges;
    HRESULT Result;
    PPERFECT_HASH_TABLE Table;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ZMMWORD Zmm1;
    ZMMWORD Zmm2;
    ZMASK16 ZmmMask;
    ZMMWORD KeysZmm;
    ZMMWORD Seed1Zmm;
    ZMMWORD Seed2Zmm;
    ZMMWORD Index1Zmm;
    ZMMWORD Index2Zmm;
    ZMMWORD Vertex1Zmm;
    ZMMWORD Vertex2Zmm;
    ULARGE_INTEGER Pair;
    PULONGLONG VertexPairs;
    PZMMWORD VertexPairsZmm;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // The following structures are fed to _mm512_permutex2var_epi32() below.
    // They form the index tables that direct the SIMD permute operation to
    // unpack and interleave our two vertex registers into the final vertex pair
    // form we want to write to memory.
    //

    ZMM_PERMUTEX2VAR_INDEX32 PermuteIndex1[] = {
        { .Index = 0, .Selector = 0, .Unused = 0 },
        { .Index = 0, .Selector = 1, .Unused = 0 },

        { .Index = 1, .Selector = 0, .Unused = 0 },
        { .Index = 1, .Selector = 1, .Unused = 0 },

        { .Index = 2, .Selector = 0, .Unused = 0 },
        { .Index = 2, .Selector = 1, .Unused = 0 },

        { .Index = 3, .Selector = 0, .Unused = 0 },
        { .Index = 3, .Selector = 1, .Unused = 0 },

        { .Index = 4, .Selector = 0, .Unused = 0 },
        { .Index = 4, .Selector = 1, .Unused = 0 },

        { .Index = 5, .Selector = 0, .Unused = 0 },
        { .Index = 5, .Selector = 1, .Unused = 0 },

        { .Index = 6, .Selector = 0, .Unused = 0 },
        { .Index = 6, .Selector = 1, .Unused = 0 },

        { .Index = 7, .Selector = 0, .Unused = 0 },
        { .Index = 7, .Selector = 1, .Unused = 0 },
    };

    ZMM_PERMUTEX2VAR_INDEX32 PermuteIndex2[] = {
        { .Index = 8, .Selector = 0, .Unused = 0 },
        { .Index = 8, .Selector = 1, .Unused = 0 },

        { .Index = 9, .Selector = 0, .Unused = 0 },
        { .Index = 9, .Selector = 1, .Unused = 0 },

        { .Index = 10, .Selector = 0, .Unused = 0 },
        { .Index = 10, .Selector = 1, .Unused = 0 },

        { .Index = 11, .Selector = 0, .Unused = 0 },
        { .Index = 11, .Selector = 1, .Unused = 0 },

        { .Index = 12, .Selector = 0, .Unused = 0 },
        { .Index = 12, .Selector = 1, .Unused = 0 },

        { .Index = 13, .Selector = 0, .Unused = 0 },
        { .Index = 13, .Selector = 1, .Unused = 0 },

        { .Index = 14, .Selector = 0, .Unused = 0 },
        { .Index = 14, .Selector = 1, .Unused = 0 },

        { .Index = 15, .Selector = 0, .Unused = 0 },
        { .Index = 15, .Selector = 1, .Unused = 0 },
    };

    //
    // Initialize aliases.
    //

    Result = S_OK;
    Table = Graph->Context->Table;
    Edges = (PEDGE)Keys;
    C_ASSERT(sizeof(*VertexPairs) == sizeof(Graph->VertexPairs));
    VertexPairs = (PULONGLONG)Graph->VertexPairs;
    VertexPairsZmm = (PZMMWORD)VertexPairs;;

    //
    // Determine the number of ZMM words we'll use to iterate over the keys,
    // 16 x 32-bit keys at a time.  Capture the number of trailing keys that
    // we'll handle at the end with a scalar loop.
    //

    NumberOfZmmWords = NumberOfKeys >> 4;
    TrailingKeys = NumberOfKeys % 16;

    //
    // Initialize seeds.
    //

    Seeds = &Graph->FirstSeed;
    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];
    Seed1Zmm = _mm512_broadcastd_epi32(_mm_set1_epi32(Seed1));
    Seed2Zmm = _mm512_broadcastd_epi32(_mm_set1_epi32(Seed2));

    //
    // Load the permute indexes.
    //

    Index1Zmm = _mm512_loadu_epi32(&PermuteIndex1[0]);
    Index2Zmm = _mm512_loadu_epi32(&PermuteIndex2[0]);

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfZmmWords; Edge++, Edges += 16) {

        //IACA_VC_START();

        //
        // Load 16 keys into our ZMM register.
        //

        KeysZmm = _mm512_loadu_epi32(Edges);

        //
        // Perform vectorized multiply and shift ops against 16 keys at a time,
        // once for each vertex.
        //

        //
        // Vertex 1: ((Key * SEED1) >> SEED3_BYTE1)
        //

        Vertex1Zmm = _mm512_mullo_epi32(KeysZmm, Seed1Zmm);
        Vertex1Zmm = _mm512_srli_epi32(Vertex1Zmm, Seed3.Byte1);

        //
        // Vertex 2: ((Key * SEED2) >> SEED3_BYTE2)
        //

        Vertex2Zmm = _mm512_mullo_epi32(KeysZmm, Seed2Zmm);
        Vertex2Zmm = _mm512_srli_epi32(Vertex2Zmm, Seed3.Byte2);

        //
        // Compare each pair of vertices against each other to see if there
        // are any conflicts (i.e. a key hashed to the same final vertex value
        // for both seeds).  If there are, abort, indicating vertex collision.
        //

        ZmmMask = _mm512_cmpeq_epi32_mask(Vertex1Zmm, Vertex2Zmm);
        if (ZmmMask > 0) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        }

        //
        // No collisions were detected, so we can save these vertices to memory.
        // This is a lot easier in AVX512 versus AVX2; the vperm2td instruction
        // allows us to unpack and interleave the values from each vertex in a
        // single instruction (versus the two permutes then blend we need for
        // AVX2).
        //

        //
        // Unpack-and-interleave the first 8 vertex pairs, then store.
        //

        Zmm1 = _mm512_permutex2var_epi32(Vertex1Zmm, Index1Zmm, Vertex2Zmm);
        _mm512_storeu_epi32(VertexPairsZmm, Zmm1);
        VertexPairsZmm++;

        //
        // Unpack-and-interleave the second 8 vertex pairs, then store.
        //

        Zmm2 = _mm512_permutex2var_epi32(Vertex1Zmm, Index2Zmm, Vertex2Zmm);
        _mm512_storeu_epi32(VertexPairsZmm, Zmm2);
        VertexPairsZmm++;

        //IACA_VC_END();
    }

    if (TrailingKeys > 0) {

        //
        // Handle the remaining 1-15 keys using a normal, non-SIMD loop.
        //

        VertexPairs = (PULONGLONG)VertexPairsZmm;

        for (Edge = 0; Edge < TrailingKeys; Edge++) {
            Key = *Edges++;

            Vertex1 = Key * Seed1;
            Vertex1 = Vertex1 >> Seed3.Byte1;;

            Vertex2 = Key * Seed2;
            Vertex2 = Vertex2 >> Seed3.Byte2;

            if (Vertex1 == Vertex2) {
                Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
                goto End;
            }

            Pair.LowPart = Vertex1;
            Pair.HighPart = Vertex2;

            *VertexPairs++ = Pair.QuadPart;
        }
    }

End:

    STOP_GRAPH_COUNTER(HashKeys);

    EVENT_WRITE_GRAPH(HashKeys);

    Result = GraphPostHashKeys(Result, Graph);

    return Result;
}

_Must_inspect_result_
_Success_(return >= 0)
_Requires_exclusive_lock_held_(Graph->Lock)
HRESULT
GraphHashKeys16MultiplyShiftRX_AVX512(
    _In_ PGRAPH Graph,
    _In_ ULONG NumberOfKeys,
    _In_reads_(NumberOfKeys) PKEY Keys
    )
/*++

Routine Description:

    This routine hashes all keys into vertices without adding the resulting
    vertices to the graph.  It is used by GraphHashKeysThenAdd().

Arguments:

    Graph - Supplies a pointer to the graph for which the hash values will be
        created.

    NumberOfKeys - Supplies the number of keys.

    Keys - Supplies the base address of the keys array.

Return Value:

    S_OK - Success.

    PH_E_GRAPH_VERTEX_COLLISION_FAILURE - The graph encountered two vertices
        that, when masked, were identical.

--*/
{
    KEY Key = 0;
    EDGE Edge;
    ULONG NumberOfZmmWords;
    ULONG TrailingKeys;
    PULONG Seeds;
    PEDGE Edges;
    HRESULT Result;
    PPERFECT_HASH_TABLE Table;
    VERTEX Vertex1;
    VERTEX Vertex2;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ZMMWORD Zmm1;
    ZMMWORD Zmm2;
    ZMMWORD Zmm3;
    ZMMWORD Imm0;
    ZMASK8 ZmmMask;
    ZMASK32 BlendMask;
    ZMMWORD KeysZmm;
    ZMMWORD Seed1Zmm;
    ZMMWORD Seed2Zmm;
    ZMMWORD Vertex1Zmm;
    ZMMWORD Vertex2Zmm;
    ULONG_INTEGER Pair;
    PULONG VertexPairs;
    PZMMWORD VertexPairsZmm;

    DECL_GRAPH_COUNTER_LOCAL_VARS();

    //
    // Initialize aliases.
    //

    Result = S_OK;
    Table = Graph->Context->Table;
    Edges = (PEDGE)Keys;
    C_ASSERT(sizeof(*VertexPairs) == sizeof(*Graph->Vertex16Pairs));
    VertexPairs = (PULONG)Graph->Vertex16Pairs;
    VertexPairsZmm = (PZMMWORD)VertexPairs;

    //
    // Determine the number of ZMM words we'll use to iterate over the keys,
    // 16 x 32-bit keys at a time.  Capture the number of trailing keys that
    // we'll handle at the end with a scalar loop.
    //

    NumberOfZmmWords = NumberOfKeys >> 4;
    TrailingKeys = NumberOfKeys % 16;

    //
    // Initialize seeds.
    //

    Seeds = &Graph->FirstSeed;
    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];
    Seed1Zmm = _mm512_broadcastd_epi32(_mm_set1_epi32(Seed1));
    Seed2Zmm = _mm512_broadcastd_epi32(_mm_set1_epi32(Seed2));

    //
    // Load the permute index and blend mask.
    //

    Imm0 = _mm512_setr_epi16( 0,  0,  2,  2,
                              4,  4,  6,  6,
                              8,  8, 10, 10,
                             12, 12, 14, 14,
                             16, 16, 18, 18,
                             20, 20, 22, 22,
                             24, 24, 26, 26,
                             28, 28, 30, 30);

    BlendMask = 0xaaaaaaaa;

    START_GRAPH_COUNTER();

    for (Edge = 0; Edge < NumberOfZmmWords; Edge++, Edges += 16) {

        //IACA_VC_START();

        //
        // Load 16 keys into our ZMM register.
        //

        KeysZmm = _mm512_loadu_epi32(Edges);

        //
        // Perform vectorized multiply and shift ops against 16 keys at a time,
        // once for each vertex.
        //

        //
        // Vertex 1: ((Key * SEED1) >> SEED3_BYTE1)
        //

        Vertex1Zmm = _mm512_mullo_epi32(KeysZmm, Seed1Zmm);
        Vertex1Zmm = _mm512_srli_epi32(Vertex1Zmm, Seed3.Byte1);

        //
        // Vertex 2: ((Key * SEED2) >> SEED3_BYTE2)
        //

        Vertex2Zmm = _mm512_mullo_epi32(KeysZmm, Seed2Zmm);
        Vertex2Zmm = _mm512_srli_epi32(Vertex2Zmm, Seed3.Byte2);

        //
        // Compare each pair of vertices against each other to see if there
        // are any conflicts (i.e. a key hashed to the same final vertex value
        // for both seeds).  If there are, abort, indicating vertex collision.
        //

        ZmmMask = _mm512_cmpeq_epi32_mask(Vertex1Zmm, Vertex2Zmm);
        if (ZmmMask > 0) {
            Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
            goto End;
        }

        //
        // No collisions were detected, so we can save these vertices to memory.
        //

        Zmm1 = _mm512_permutexvar_epi16(Imm0, Vertex1Zmm);
        Zmm2 = _mm512_permutexvar_epi16(Imm0, Vertex2Zmm);
        Zmm3 = _mm512_mask_blend_epi16(BlendMask, Zmm1, Zmm2);

        _mm512_storeu_epi32(VertexPairsZmm, Zmm3);
        VertexPairsZmm++;

        //IACA_VC_END();
    }

    if (TrailingKeys > 0) {

        //
        // Handle the remaining keys using a normal, non-SIMD loop.
        //

        VertexPairs = (PULONG)VertexPairsZmm;

        for (Edge = 0; Edge < TrailingKeys; Edge++) {
            Key = *Edges++;

            Vertex1 = Key * Seed1;
            Vertex1 >>= Seed3.Byte1;

            Vertex2 = Key * Seed2;
            Vertex2 >>= Seed3.Byte2;

            if (Vertex1 == Vertex2) {
                Result = PH_E_GRAPH_VERTEX_COLLISION_FAILURE;
                goto End;
            }

            Pair.LowPart = (USHORT)Vertex1;
            Pair.HighPart = (USHORT)Vertex2;

            *VertexPairs++ = Pair.LongPart;
        }
    }

End:

    STOP_GRAPH_COUNTER(HashKeys);

    EVENT_WRITE_GRAPH(HashKeys);

    Result = GraphPostHashKeys(Result, Graph);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
