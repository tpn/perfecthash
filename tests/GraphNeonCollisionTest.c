#include "stdafx.h"
#include <stdio.h>
#include <string.h>

static GRAPH Graph;
static PERFECT_HASH_CONTEXT Context;
static PERFECT_HASH_TABLE Table;

int main(void)
{
#ifdef PH_HAS_ARM64_NEON
    const PERFECT_HASH_HASH_FUNCTION_ID Hashes[] = {
        PerfectHashHashMulshrolate1RXFunctionId,
        PerfectHashHashMulshrolate2RXFunctionId,
        PerfectHashHashMulshrolate3RXFunctionId,
        PerfectHashHashMulshrolate4RXFunctionId,
    };
    KEY Keys[] = {1, 3, 5, 7, 9, 11, 13, 15, 17};
    VERTEX_PAIR Scalar[ARRAYSIZE(Keys)], Output[ARRAYSIZE(Keys)];
    VERTEX_PAIR Expected[ARRAYSIZE(Keys)];
    ULONG Count = ARRAYSIZE(Keys);

    if (!GraphHasNeon()) {
        return 77;
    }
    Graph.Context = &Context;
    Context.Table = &Table;
    QueryPerformanceFrequency(&Context.Frequency);
    Table.HashMask = 0xffff;
    Graph.Seed1 = 0xe7dd8da3;
    Graph.Seed2 = 0xa1be901b;
    Graph.Seed3 = 0x000d1f10;  // Shift 16, rotate 31 and 13.
    Graph.Seed4 = 0xd445a349;
    Graph.Seed5 = 0x85bc199b;

    for (ULONG Hash = 0; Hash < ARRAYSIZE(Hashes); Hash++) {
        Table.HashFunctionId = Hashes[Hash];
        for (ULONG Bits = 16; Bits <= 32; Bits += 16) {
            Graph.Flags.UsingAssigned16 = (Bits == 16);
            size_t PairSize = (Bits == 16 ? sizeof(VERTEX16_PAIR) : sizeof(VERTEX_PAIR));
            Graph.VertexPairs = Scalar;
            HRESULT Result = (Bits == 16 ? GraphHashKeys16 : GraphHashKeys)(
                &Graph, Count, Keys);
            if (Result != S_OK) {
                fprintf(stderr, "Scalar fixture failed for hash %u, %u bits\n",
                        (unsigned)Table.HashFunctionId, (unsigned)Bits);
                return 1;
            }

            // Complete one vector, then collide at each lane of the next vector
            // and in the scalar tail. Zero hashes to a self-edge for every seed.
            for (ULONG Edge = 4; Edge < Count; Edge++) {
                KEY Saved = Keys[Edge];
                Keys[Edge] = 0;
                memset(Expected, 0x9d, sizeof(Expected));
                memcpy(Expected, Scalar, Edge * PairSize);
                memset(Output, 0x9d, sizeof(Output));
                Graph.VertexPairs = Output;
                Graph.Flags.UsedNeonHashFunction = FALSE;
                Result = GraphHashKeys_NEON(&Graph, Count, Keys);
                Keys[Edge] = Saved;
                if (Result != PH_E_GRAPH_VERTEX_COLLISION_FAILURE ||
                    !Graph.Flags.UsedNeonHashFunction ||
                    memcmp(Expected, Output, sizeof(Output)) != 0) {
                    fprintf(stderr, "Collision prefix mismatch: hash %u, %u bits, edge %u\n",
                            (unsigned)Table.HashFunctionId, (unsigned)Bits, (unsigned)Edge);
                    return 1;
                }
            }
        }
    }
    return 0;
#else
    return 77;
#endif
}
