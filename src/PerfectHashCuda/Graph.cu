/*++

Copyright (c) 2020 Trent Nelson <trent@trent.me>

Module Name:

    Graph.cu

Abstract:

    CUDA graph implementation.

--*/

extern "C" {

#include <PerfectHashCuda.h>
#include "../PerfectHash/Graph.h"

//##include "Graph.cuh"

//
// Define helper macros for referring to seed constants stored in local
// variables by their uppercase names.  This allows easy copy-and-pasting of
// the algorithm "guts" between the "compiled" perfect hash table routines in
// ../CompiledPerfectHashTable and the SeededHashEx() implementations here.
//

#define SEED1 Seed1
#define SEED2 Seed2
#define SEED3 Seed3
#define SEED4 Seed4
#define SEED5 Seed5
#define SEED6 Seed6
#define SEED7 Seed7
#define SEED8 Seed8

#define SEED3_BYTE1 Seed3.Byte1
#define SEED3_BYTE2 Seed3.Byte2
#define SEED3_BYTE3 Seed3.Byte3
#define SEED3_BYTE4 Seed3.Byte4

#define SEED6_BYTE1 Seed6.Byte1
#define SEED6_BYTE2 Seed6.Byte2
#define SEED6_BYTE3 Seed6.Byte3
#define SEED6_BYTE4 Seed6.Byte4

#if 0
GLOBAL
VOID
PerfectHashCudaSeededHashAllMultiplyShiftR(
    _In_reads_(NumberOfKeys) PULONG Keys,
    _In_ ULONG NumberOfKeys,
    _Out_writes_(NumberOfKeys) PULONGLONG VertexPairs,
    _In_ PULONG Seeds,
    _In_ ULONG Mask
    )
{
    ULONG Index;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG Vertex3;
    ULONG Vertex4;
    ULONG Key1;
    ULONG Key2;
    ULONG Key3;
    ULONG Key4;
    PINT2 Input = (PINT2)Keys;
    PINT4 Output = (PINT4)VertexPairs;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    FOR_EACH_1D(Index, NumberOfKeys) {
        Key1 = Input[Index].x;
        Key2 = Input[Index].y;
        Key3 = Input[Index+1].x;
        Key4 = Input[Index+1].y;

        Vertex1 = ((Key1 * SEED1) >> SEED3_BYTE1);
        Vertex2 = ((Key2 * SEED2) >> SEED3_BYTE2);

        Vertex3 = ((Key3 * SEED1) >> SEED3_BYTE1);
        Vertex4 = ((Key4 * SEED2) >> SEED3_BYTE2);

        Output[Index].x = Vertex1;
        Output[Index].y = Vertex2;
        Output[Index].w = Vertex3;
        Output[Index].z = Vertex4;
    }
}
#endif

GLOBAL
VOID
PerfectHashCudaSeededHashAllMultiplyShiftR2(
    _In_reads_(NumberOfKeys) PULONG Keys,
    _In_ ULONG NumberOfKeys,
    _Out_writes_(NumberOfKeys) PULONGLONG VertexPairs,
    _In_ PULONG Seeds,
    _In_ ULONG Mask
    )
{
    ULONG Index;
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG Key;
    PINT2 Output = (PINT2)VertexPairs;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    FOR_EACH_1D(Index, NumberOfKeys) {
        Key = Keys[Index];

        Vertex1 = ((Key * SEED1) >> SEED3_BYTE1);
        Vertex2 = ((Key * SEED2) >> SEED3_BYTE2);

        Output[Index].x = Vertex1;
        Output[Index].y = Vertex2;
    }
}

GLOBAL
VOID
PerfectHashCudaEnterSolvingLoop(
    _In_ PGRAPH Graph
    )
{
    ClockBlock(1000);
}

}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
