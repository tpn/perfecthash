/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashKernels.cu

Abstract:

    This module implements CUDA kernels used by the perfect hash library.

--*/

#ifdef __cplusplus
extern "C" {
#endif

#include "Cu.cuh"
#include <no_sal2.h>

GLOBAL
VOID
SinglePrecisionAlphaXPlusY(
    _In_ LONG Total,
    _In_ FLOAT Alpha,
    _In_ PFLOAT X,
    _Out_ PFLOAT Y
    )
{
    LONG Index;

    FOR_EACH_1D(Index, Total) {
        Y[Index] = Alpha * X[Index] + Y[Index];
    }
}

GLOBAL
VOID
DeltaTimestamp(
    _In_ ULONG64 Total,
    _In_ PULONG64 Timestamp,
    _Out_ PULONG64 Delta
    )
{
    ULONG64 Index;

    if (ThreadIndex.x % 32 == 0) {
        return;
    }

    FOR_EACH_1D(Index, Total) {
        Delta[Index] = Timestamp[Index] - Timestamp[Index-1];
    }
}

//
// Define helper macros for referring to seed constants stored in local
// variables by their uppercase names.  This allows easy copy-and-pasting of
// the algorithm "guts" between the "compiled" perfect hash table routines in
// ../CompiledPerfectHashTable and the SeededHash() implementations here.
//
// N.B. Only the most recent routines (multiply xor-shift etc) use these macros;
//      existing ones should be updated during the next bulk refactoring.
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
HRESULT
PerfectHashTableSeededHashShiftMultiplyXorShiftCu(
    ULONG NumberOfKeys,
    PULONG Keys,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    Performs a shift, multiply, xor, shift.

Arguments:

    Table - Supplies a pointer to the table for which the hash is being created.

    Key - Supplies the input value to hash.

    NumberOfSeeds - Supplies the number of elements in the Seeds array.

    Seeds - Supplies an array of ULONG seed values.

    Hash - Receives two 32-bit hashes merged into a 64-bit value.

Return Value:

    S_OK on success.  If the two 32-bit hash values are identical, E_FAIL.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);

    ASSERT(NumberOfSeeds >= 3);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    DownsizedKey = Key;

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey >> SEED3_BYTE1;
    Vertex1 *= SEED1;
    Vertex1 ^= Vertex1 >> SEED3_BYTE2;

    Vertex2 = DownsizedKey >> SEED3_BYTE3;
    Vertex2 *= SEED2;
    Vertex2 ^= Vertex2 >> SEED3_BYTE4;

    if (Vertex1 == Vertex2) {
        return E_FAIL;
    }

    Result.LowPart = Vertex1;
    Result.HighPart = Vertex2;

    *Hash = Result.QuadPart;

    return S_OK;
}
#endif

#ifdef __cplusplus
}
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
