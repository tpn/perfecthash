/*++

Copyright (c) 2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableHashExCpp.cpp

Abstract:

    This module implements templated C++ hash routines that are suitable for use
    both host side and device side (via CUDA).  Each routine corresponds to one
    of the PERFECT_HASH_TABLE_HASH_FUNCTION_ID enumerations.

    Not all CPU hash functions are supported; none of the CRC32-based routines
    are implemented, for example (as these are not useful anyway, even in the
    context of CPU hashing).

    This module was based on PerfectHashTableHashEx.c, which was based on the
    module PerfectHashTableHash.c.

    The original seeded hash routines have the signature:

        typedef
        HRESULT
        (STDAPICALLTYPE PERFECT_HASH_TABLE_SEEDED_HASH)(
            _In_ PPERFECT_HASH_TABLE Table,
            _In_ KeyType Key,
            _In_ ULONG NumberOfSeeds,
            _In_reads_(NumberOfSeeds) PULONG Seeds,
            _Out_ PULONGLONG Hash
            );

    The "Ex" versions (implemented in PerfectHashTableHashEx.c) were introduced
    as follows:

        typedef
        ULONGLONG
        (STDAPICALLTYPE PERFECT_HASH_TABLE_SEEDED_HASH_EX)(
            _In_ KeyType Key,
                    _In_ VertexType Mask
            );

    The routines in this module have this signature:

    template<typename ResultType,
             typename KeyType,
             typename VertexType>
    FORCEINLINE
    DEVICE
    ResultType
    (STDAPICALLTYPE PERFECT_HASH_TABLE_SEEDED_HASH_EX_CPP)(
        _In_ KeyType Key,
        _In_ VertexType Mask
        );

--*/

#pragma once

#include "stdafx.h"

DEVICE CONSTANT GRAPH_SEEDS c_GraphSeeds;

//
// Define helper macros for referring to seed constants stored in the
// __constant__ c_GraphSeeds struct.  This allows easy copy-and-pasting of the
// algorithm "guts" between the "compiled" perfect hash table routines in
// ../CompiledPerfectHashTable and the SeededHashEx() implementations here.
//

#define SEEDS c_GraphSeeds.Seeds

#define SEED1 c_GraphSeeds.Seed1
#define SEED2 c_GraphSeeds.Seed2
#define SEED3 c_GraphSeeds.Seed3
#define SEED4 c_GraphSeeds.Seed4
#define SEED5 c_GraphSeeds.Seed5
#define SEED6 c_GraphSeeds.Seed6
#define SEED7 c_GraphSeeds.Seed7
#define SEED8 c_GraphSeeds.Seed8

#define SEED3_BYTE1 c_GraphSeeds.Seed3Bytes.Byte1
#define SEED3_BYTE2 c_GraphSeeds.Seed3Bytes.Byte2
#define SEED3_BYTE3 c_GraphSeeds.Seed3Bytes.Byte3
#define SEED3_BYTE4 c_GraphSeeds.Seed3Bytes.Byte4

#define SEED6_BYTE1 c_GraphSeeds.Seed6Bytes.Byte1
#define SEED6_BYTE2 c_GraphSeeds.Seed6Bytes.Byte2
#define SEED6_BYTE3 c_GraphSeeds.Seed6Bytes.Byte3
#define SEED6_BYTE4 c_GraphSeeds.Seed6Bytes.Byte4

#define DOWNSIZE_KEY(Key) ((KeyType)(Key))

template<typename ValueType,
         typename ShiftType>
FORCEINLINE
DEVICE
ValueType
RotateRight(
    _In_ ValueType Value,
    _In_ ShiftType Shift
    )
{
    constexpr ShiftType Bits = sizeof(ValueType) * 8;

    if (Shift == 0) {
        return Value;
    }

    Shift %= Bits;

    return (Value >> Shift) | (Value << (Bits - Shift));
}

template<typename ValueType,
         typename ShiftType>
FORCEINLINE
DEVICE
ValueType
RotateLeft(
    _In_ ValueType Value,
    _In_ ShiftType Shift
    )
{
    constexpr ShiftType Bits = sizeof(ValueType) * 8;

    if (Shift == 0) {
        return Value;
    }

    Shift %= Bits;

    return (Value << Shift) | (Value >> (Bits - Shift));
}


template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppNull(
    _In_ KeyType Key,
    _In_ VertexType Mask
    )
/*++

Routine Description:

    This is a dummy seeded hash implementation that simply returns S_OK without
    actually doing anything.

Arguments:

    Table - Supplies a pointer to the table for which the hash is being created.

    Key - Supplies the input value to hash.

    NumberOfSeeds - Supplies the number of elements in the Seeds array.

    Hash - Receives two 32-bit hashes merged into a 64-bit value.

Return Value:

    S_OK.

--*/
{
    ResultType Result;
    Result.AsPair = 0;
    return Result;
}


template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppJenkins(
    _In_ KeyType Key,
    _In_ VertexType Mask
    )
/*++

Routine Description:

    This routine implements the Jenkins hash routine.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType A;
    VertexType B;
    VertexType C;
    VertexType D;
    VertexType E;
    VertexType F;
    PBYTE Byte;
    VertexType Vertex1;
    VertexType Vertex2;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Byte = (PBYTE)&Key;

    //
    // Generate the first hash.
    //

    A = B = 0x79b9;
    C = SEED1;

    A += (((VertexType)Byte[3]) << 24);
    A += (((VertexType)Byte[2]) << 16);
    A += (((VertexType)Byte[1]) <<  8);
    A += ((VertexType)Byte[0]);

    A -= B; A -= C; A ^= (C >> 13);
    B -= C; B -= A; B ^= (A <<  8);
    C -= A; C -= B; C ^= (B >> 13);
    A -= B; A -= C; A ^= (C >> 12);
    B -= C; B -= A; B ^= (A << 16);
    C -= A; C -= B; C ^= (B >>  5);
    A -= B; A -= C; A ^= (C >>  3);
    B -= C; B -= A; B ^= (A << 10);
    C -= A; C -= B; C ^= (B >> 15);

    //
    // Generate the second hash.
    //

    D = E = 0x9e37;
    F = SEED2;

    D += (((VertexType)Byte[3]) << 24);
    D += (((VertexType)Byte[2]) << 16);
    D += (((VertexType)Byte[1]) <<  8);
    D += ((VertexType)Byte[0]);

    D -= E; D -= F; D ^= (F >> 13);
    E -= F; E -= D; E ^= (D <<  8);
    F -= D; F -= E; F ^= (E >> 13);
    D -= E; D -= F; D ^= (F >> 12);
    E -= F; E -= D; E ^= (D << 16);
    F -= D; F -= E; F ^= (E >>  5);
    D -= E; D -= F; D ^= (F >>  3);
    E -= F; E -= D; E ^= (D << 10);
    F -= D; F -= E; F ^= (E >> 15);

    Vertex1 = C;
    Vertex2 = F;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppRotateMultiplyXorRotate(
    _In_ KeyType Key,
    _In_ VertexType Mask
    )
/*++

Routine Description:

    Performs a rotate, multiply, xor, rotate.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    ResultType comprised of the two hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = RotateRight(DownsizedKey, SEED3_BYTE1);
    Vertex1 *= SEED1;
    Vertex1 ^= RotateRight(Vertex1, SEED3_BYTE2);

    Vertex2 = RotateRight(DownsizedKey, SEED3_BYTE3);
    Vertex2 *= SEED2;
    Vertex2 ^= RotateRight(Vertex2, SEED3_BYTE4);

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppShiftMultiplyXorShift(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a shift, multiply, xor, shift.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey >> SEED3_BYTE1;
    Vertex1 *= SEED1;
    Vertex1 ^= Vertex1 >> SEED3_BYTE2;

    Vertex2 = DownsizedKey >> SEED3_BYTE3;
    Vertex2 *= SEED2;
    Vertex2 ^= Vertex2 >> SEED3_BYTE4;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppRotateMultiplyXorRotate2(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs two rotate, multiply, xor, rotate combinations.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = RotateRight(DownsizedKey, SEED3_BYTE1);
    Vertex1 *= SEED1;
    Vertex1 ^= RotateRight(Vertex1, SEED3_BYTE2);
    Vertex1 *= SEED2;
    Vertex1 ^= RotateRight(Vertex1, SEED3_BYTE3);
    Vertex1 &= Mask;

    Vertex2 = RotateRight(DownsizedKey, SEED6_BYTE1);
    Vertex2 *= SEED4;
    Vertex2 ^= RotateRight(Vertex2, SEED6_BYTE2);
    Vertex2 *= SEED5;
    Vertex2 ^= RotateRight(Vertex2, SEED6_BYTE3);

    Vertex2 &= Mask;

    Result.LowPart = (decltype(Result.LowPart))Vertex1;
    Result.HighPart = (decltype(Result.HighPart))Vertex2;

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppShiftMultiplyXorShift2(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs two shift, multiply, xor, shift combinations.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey >> SEED3_BYTE1;
    Vertex1 *= SEED1;
    Vertex1 ^= Vertex1 >> SEED3_BYTE2;
    Vertex1 *= SEED2;
    Vertex1 ^= Vertex1 >> SEED3_BYTE3;

    Vertex2 = DownsizedKey >> SEED6_BYTE1;
    Vertex2 *= SEED4;
    Vertex2 ^= Vertex2 >> SEED6_BYTE2;
    Vertex2 *= SEED5;
    Vertex2 ^= Vertex2 >> SEED6_BYTE3;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiplyRotateR(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right rotate.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = RotateRight(Vertex1, SEED3_BYTE1);

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = RotateRight(Vertex2, SEED3_BYTE2);

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiplyRotateLR(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then left rotate on vertex 1, right rotate on vertex 2.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = RotateLeft(Vertex1, SEED3_BYTE1);

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = RotateRight(Vertex2, SEED3_BYTE2);

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiplyShiftR(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE2;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiplyShiftRX(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.  Ignores mask.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Ignored.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE2;

    Result.LowPart = (decltype(Result.LowPart))Vertex1;
    Result.HighPart = (decltype(Result.HighPart))Vertex2;

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiply643ShiftR(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONGLONG Vertex1;
    ULONGLONG Vertex2;
    ULONGLONG DownsizedKey;
    ULARGE_INTEGER Seed1;
    ULARGE_INTEGER Seed2;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1.QuadPart = *((PULONGLONG)&SEED1);
    Seed2.QuadPart = *((PULONGLONG)&SEED2);

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * Seed1.QuadPart;
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * Seed2.QuadPart;
    Vertex2 = Vertex2 >> SEED3_BYTE2;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiply644ShiftR(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONGLONG Vertex1;
    ULONGLONG Vertex2;
    ULONGLONG DownsizedKey;
    ULARGE_INTEGER Seed1;
    ULARGE_INTEGER Seed2;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1.LowPart = SEED1;
    Seed1.HighPart = SEED2;

    Seed2.LowPart = SEED4;
    Seed2.HighPart = SEED5;

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * Seed1.QuadPart;
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * Seed2.QuadPart;
    Vertex2 = Vertex2 >> SEED3_BYTE2;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiplyShiftLR(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then shift left on vertex 2, shift right on vertext 2.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = Vertex1 << SEED3_BYTE1;

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE2;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiply(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a single multiply on each vertex.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex2 = DownsizedKey * SEED2;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiplyXor(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then xor on each vertex.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 ^= SEED2;

    Vertex2 = DownsizedKey * SEED3;
    Vertex2 ^= SEED4;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiplyRotateRMultiply(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right rotate then multiply.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = RotateRight(Vertex1, SEED3_BYTE1);
    Vertex1 *= SEED2;

    Vertex2 = DownsizedKey * SEED4;
    Vertex2 = RotateRight(Vertex2, SEED3_BYTE2);
    Vertex2 *= SEED5;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiplyRotateR2(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right rotate twice.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = RotateRight(Vertex1, SEED3_BYTE1);
    Vertex1 *= SEED2;
    Vertex1 = RotateRight(Vertex1, SEED3_BYTE2);

    Vertex2 = DownsizedKey * SEED4;
    Vertex2 = RotateRight(Vertex2, SEED3_BYTE3);
    Vertex2 *= SEED5;
    Vertex2 = RotateRight(Vertex2, SEED3_BYTE4);

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiplyShiftRMultiply(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift then multiply.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 >>= SEED3_BYTE1;
    Vertex1 *= SEED2;

    Vertex2 = DownsizedKey * SEED4;
    Vertex2 >>= SEED3_BYTE2;
    Vertex2 *= SEED5;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppMultiplyShiftR2(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift twice.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 >>= SEED3_BYTE1;
    Vertex1 *= SEED2;
    Vertex1 >>= SEED3_BYTE2;

    Vertex2 = DownsizedKey * SEED4;
    Vertex2 >>= SEED3_BYTE3;
    Vertex2 *= SEED5;
    Vertex2 >>= SEED3_BYTE4;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppRotateRMultiply(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a right rotate then multiply.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey;
    Vertex1 = RotateRight(Vertex1, SEED3_BYTE1);
    Vertex1 *= SEED1;

    Vertex2 = DownsizedKey;
    Vertex2 = RotateRight(Vertex2, SEED3_BYTE2);
    Vertex2 *= SEED2;

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
ResultType
PerfectHashTableSeededHashExCppRotateRMultiplyRotateR(
    KeyType Key,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a right rotate, then multiply, then another right rotate.

Arguments:

    Key - Supplies the input value to hash.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    DownsizedKey = DOWNSIZE_KEY(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey;
    Vertex1 = RotateRight(Vertex1, SEED3_BYTE1);
    Vertex1 *= SEED1;
    Vertex1 = RotateRight(Vertex1, SEED3_BYTE2);

    Vertex2 = DownsizedKey;
    Vertex2 = RotateRight(Vertex2, SEED3_BYTE3);
    Vertex2 *= SEED2;
    Vertex2 = RotateRight(Vertex2, SEED3_BYTE4);

    Result.LowPart = (decltype(Result.LowPart))(Vertex1 & Mask);
    Result.HighPart = (decltype(Result.HighPart))(Vertex2 & Mask);

    return Result;
}

//
// Helper routines for resolving hash functions from IDs at runtime.
//

template<typename ResultType,
         typename KeyType>
FORCEINLINE
DEVICE
auto
GetHashFunctionForId(
    _In_ PERFECT_HASH_HASH_FUNCTION_ID Id
    )
{
    using VertexType = typename ResultType::VertexType;

    switch (Id) {
        case PerfectHashHashJenkinsFunctionId:
            return PerfectHashTableSeededHashExCppJenkins<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashRotateMultiplyXorRotateFunctionId:
            return PerfectHashTableSeededHashExCppRotateMultiplyXorRotate<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashShiftMultiplyXorShiftFunctionId:
            return PerfectHashTableSeededHashExCppShiftMultiplyXorShift<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashRotateMultiplyXorRotate2FunctionId:
            return PerfectHashTableSeededHashExCppRotateMultiplyXorRotate2<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashShiftMultiplyXorShift2FunctionId:
            return PerfectHashTableSeededHashExCppShiftMultiplyXorShift2<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyRotateRFunctionId:
            return PerfectHashTableSeededHashExCppMultiplyRotateR<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyRotateLRFunctionId:
            return PerfectHashTableSeededHashExCppMultiplyRotateLR<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyShiftRFunctionId:
            return PerfectHashTableSeededHashExCppMultiplyShiftR<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyShiftRXFunctionId:
            return PerfectHashTableSeededHashExCppMultiplyShiftRX<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiply643ShiftRFunctionId:
            return PerfectHashTableSeededHashExCppMultiply643ShiftR<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiply644ShiftRFunctionId:
            return PerfectHashTableSeededHashExCppMultiply644ShiftR<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyShiftLRFunctionId:
            return PerfectHashTableSeededHashExCppMultiplyShiftLR<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyFunctionId:
            return PerfectHashTableSeededHashExCppMultiply<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyXorFunctionId:
            return PerfectHashTableSeededHashExCppMultiplyXor<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyRotateRMultiplyFunctionId:
            return PerfectHashTableSeededHashExCppMultiplyRotateRMultiply<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyRotateR2FunctionId:
            return PerfectHashTableSeededHashExCppMultiplyRotateR2<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyShiftRMultiplyFunctionId:
            return PerfectHashTableSeededHashExCppMultiplyShiftRMultiply<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashMultiplyShiftR2FunctionId:
            return PerfectHashTableSeededHashExCppMultiplyShiftR2<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashRotateRMultiplyFunctionId:
            return PerfectHashTableSeededHashExCppRotateRMultiply<
                ResultType,
                KeyType,
                VertexType>;

        case PerfectHashHashRotateRMultiplyRotateRFunctionId:
            return PerfectHashTableSeededHashExCppRotateRMultiplyRotateR<
                ResultType,
                KeyType,
                VertexType>;

        default:
            return PerfectHashTableSeededHashExCppNull<
                ResultType,
                KeyType,
                VertexType>;
    }
}

template<typename GraphType>
FORCEINLINE
DEVICE
auto
GraphGetHashFunction(
    _In_ GraphType *Graph
    )
{
    using ResultType = typename GraphType::VertexPairType;
    using KeyType = typename GraphType::KeyType;

    return GetHashFunctionForId<ResultType, KeyType>(Graph->HashFunctionId);
}

template <typename GraphType>
FORCEINLINE
DEVICE
HOST
bool
GraphIsHashFunctionSupported(
    _In_ GraphType *Graph
    )
{
    switch (Graph->HashFunctionId) {
        case PerfectHashHashJenkinsFunctionId:
        case PerfectHashHashRotateMultiplyXorRotateFunctionId:
        case PerfectHashHashShiftMultiplyXorShiftFunctionId:
        case PerfectHashHashRotateMultiplyXorRotate2FunctionId:
        case PerfectHashHashShiftMultiplyXorShift2FunctionId:
        case PerfectHashHashMultiplyRotateRFunctionId:
        case PerfectHashHashMultiplyRotateLRFunctionId:
        case PerfectHashHashMultiplyShiftRFunctionId:
        case PerfectHashHashMultiplyShiftRXFunctionId:
        case PerfectHashHashMultiply643ShiftRFunctionId:
        case PerfectHashHashMultiply644ShiftRFunctionId:
        case PerfectHashHashMultiplyShiftLRFunctionId:
        case PerfectHashHashMultiplyFunctionId:
        case PerfectHashHashMultiplyXorFunctionId:
        case PerfectHashHashMultiplyRotateRMultiplyFunctionId:
        case PerfectHashHashMultiplyRotateR2FunctionId:
        case PerfectHashHashMultiplyShiftRMultiplyFunctionId:
        case PerfectHashHashMultiplyShiftR2FunctionId:
        case PerfectHashHashRotateRMultiplyFunctionId:
        case PerfectHashHashRotateRMultiplyRotateRFunctionId:
            return true;
        default:
            return false;
    }
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
