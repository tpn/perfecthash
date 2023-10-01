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
            _In_ PULONG Seeds,
            _In_ VertexType Mask
            );

    The routines in this module have this signature:

    template<typename ResultType,
             typename KeyType,
             typename VertexType>
    FORCEINLINE
    DEVICE
    HOST
    ResultType
    (STDAPICALLTYPE PERFECT_HASH_TABLE_SEEDED_HASH_EX_CPP)(
        _In_ KeyType Key,
        _In_ PULONG Seeds,
        _In_ VertexType Mask
        );

--*/

#pragma once

#include "stdafx.h"

#if 0
#ifndef PH_CUDA
#include "stdafx.h"
#else
#include "PerfectHashPrivate.h"
#endif
#endif

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

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppNull(
    _In_ KeyType Key,
    _In_ PULONG Seeds,
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

    Seeds - Supplies an array of ULONG seed values.

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
HOST
ResultType
PerfectHashTableSeededHashExCppJenkins(
    _In_ KeyType Key,
    _In_ PULONG Seeds,
    _In_ VertexType Mask
    )
/*++

Routine Description:

    This routine implements the Jenkins hash routine.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

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

    A = B = 0x9e3779b9;
    C = Seeds[0];

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

    D = E = 0x9e3779b9;
    F = Seeds[1];

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

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppRotateMultiplyXorRotate(
    _In_ KeyType Key,
    _In_ PULONG Seeds,
    _In_ VertexType Mask
    )
/*++

Routine Description:

    Performs a rotate, multiply, xor, rotate.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    ResultType comprised of the two hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

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

    Vertex1 = _rotr(DownsizedKey, SEED3_BYTE1);
    Vertex1 *= SEED1;
    Vertex1 ^= _rotr(Vertex1, SEED3_BYTE2);

    Vertex2 = _rotr(DownsizedKey, SEED3_BYTE3);
    Vertex2 *= SEED2;
    Vertex2 ^= _rotr(Vertex2, SEED3_BYTE4);

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppShiftMultiplyXorShift(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a shift, multiply, xor, shift.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

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

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppRotateMultiplyXorRotate2(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs two rotate, multiply, xor, rotate combinations.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    ULONG_BYTES Seed6;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    Seed4 = Seeds[3];
    Seed5 = Seeds[4];
    Seed6.AsULong = Seeds[5];

    DownsizedKey = Key;

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _rotr(DownsizedKey, SEED3_BYTE1);
    Vertex1 *= SEED1;
    Vertex1 ^= _rotr(Vertex1, SEED3_BYTE2);
    Vertex1 *= SEED2;
    Vertex1 ^= _rotr(Vertex1, SEED3_BYTE3);

    Vertex2 = _rotr(DownsizedKey, SEED6_BYTE1);
    Vertex2 *= SEED4;
    Vertex2 ^= _rotr(Vertex2, SEED6_BYTE2);
    Vertex2 *= SEED5;
    Vertex2 ^= _rotr(Vertex2, SEED6_BYTE3);

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppShiftMultiplyXorShift2(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs two shift, multiply, xor, shift combinations.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    ULONG_BYTES Seed6;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    Seed4 = Seeds[3];
    Seed5 = Seeds[4];
    Seed6.AsULong = Seeds[5];

    DownsizedKey = Key;

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

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiplyRotateR(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right rotate.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

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

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = _rotr(Vertex1, SEED3_BYTE1);

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = _rotr(Vertex2, SEED3_BYTE2);

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiplyRotateLR(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then left rotate on vertex 1, right rotate on vertex 2.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

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

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = _rotl(Vertex1, SEED3_BYTE1);

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = _rotr(Vertex2, SEED3_BYTE2);

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiplyShiftR(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

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

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE2;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiplyShiftRX(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.  Ignores mask.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Ignored.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

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

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE2;

    Result.LowPart = Vertex1;
    Result.HighPart = Vertex2;

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiply643ShiftR(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG_BYTES Seed3;
    ULONGLONG Vertex1;
    ULONGLONG Vertex2;
    ULONGLONG DownsizedKey;
    ULARGE_INTEGER Seed1;
    ULARGE_INTEGER Seed2;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1.QuadPart = *((PULONGLONG)&Seeds[0]);
    Seed3.AsULong = Seeds[2];
    Seed2.QuadPart = *((PULONGLONG)&Seeds[3]);

    DownsizedKey = Key;

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * Seed1.QuadPart;
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * Seed2.QuadPart;
    Vertex2 = Vertex2 >> SEED3_BYTE2;

    Result.LowPart = (ULONG)(Vertex1 & Mask);
    Result.HighPart = (ULONG)(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiply644ShiftR(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG_BYTES Seed3;
    ULONGLONG Vertex1;
    ULONGLONG Vertex2;
    ULONGLONG DownsizedKey;
    ULARGE_INTEGER Seed1;
    ULARGE_INTEGER Seed2;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1.LowPart = Seeds[0];
    Seed1.HighPart = Seeds[1];

    Seed3.AsULong = Seeds[2];

    Seed2.LowPart = Seeds[3];
    Seed2.HighPart = Seeds[4];

    DownsizedKey = Key;

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * Seed1.QuadPart;
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * Seed2.QuadPart;
    Vertex2 = Vertex2 >> SEED3_BYTE2;

    Result.LowPart = (ULONG)(Vertex1 & Mask);
    Result.HighPart = (ULONG)(Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiplyShiftLR(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then shift left on vertex 2, shift right on vertext 2.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

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

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = Vertex1 << SEED3_BYTE1;

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE2;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiply(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a single multiply on each vertex.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    DownsizedKey = Key;

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * Seed1;
    Vertex2 = DownsizedKey * Seed2;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiplyXor(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then xor on each vertex.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG Seed3;
    ULONG Seed4;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = Seeds[2];
    Seed4 = Seeds[3];
    DownsizedKey = Key;

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * Seed1;
    Vertex1 ^= Seed2;

    Vertex2 = DownsizedKey * Seed3;
    Vertex2 ^= Seed4;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiplyRotateRMultiply(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right rotate then multiply.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];
    Seed4 = Seeds[3];
    Seed5 = Seeds[4];
    DownsizedKey = Key;

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = _rotr(Vertex1, SEED3_BYTE1);
    Vertex1 *= SEED2;

    Vertex2 = DownsizedKey * SEED4;
    Vertex2 = _rotr(Vertex2, SEED3_BYTE2);
    Vertex2 *= SEED5;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiplyRotateR2(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right rotate twice.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];
    Seed4 = Seeds[3];
    Seed5 = Seeds[4];
    DownsizedKey = Key;

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = _rotr(Vertex1, SEED3_BYTE1);
    Vertex1 *= SEED2;
    Vertex1 = _rotr(Vertex1, SEED3_BYTE2);

    Vertex2 = DownsizedKey * SEED4;
    Vertex2 = _rotr(Vertex2, SEED3_BYTE3);
    Vertex2 *= SEED5;
    Vertex2 = _rotr(Vertex2, SEED3_BYTE4);

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiplyShiftRMultiply(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift then multiply.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];
    Seed4 = Seeds[3];
    Seed5 = Seeds[4];
    DownsizedKey = Key;

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 >>= SEED3_BYTE1;
    Vertex1 *= SEED2;

    Vertex2 = DownsizedKey * SEED4;
    Vertex2 >>= SEED3_BYTE2;
    Vertex2 *= SEED5;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppMultiplyShiftR2(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift twice.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];
    Seed4 = Seeds[3];
    Seed5 = Seeds[4];
    DownsizedKey = Key;

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

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppRotateRMultiply(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a right rotate then multiply.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

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

    Vertex1 = DownsizedKey;
    Vertex1 = _rotr(Vertex1, SEED3_BYTE1);
    Vertex1 *= SEED1;

    Vertex2 = DownsizedKey;
    Vertex2 = _rotr(Vertex2, SEED3_BYTE2);
    Vertex2 *= SEED2;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

template<typename ResultType,
         typename KeyType,
         typename VertexType>
FORCEINLINE
DEVICE
HOST
ResultType
PerfectHashTableSeededHashExCppRotateRMultiplyRotateR(
    KeyType Key,
    PULONG Seeds,
    VertexType Mask
    )
/*++

Routine Description:

    Performs a right rotate, then multiply, then another right rotate.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    64-bit ULONGLONG comprised of two 32-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    VertexType Vertex1;
    VertexType Vertex2;
    KeyType DownsizedKey;
    ResultType Result;

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

    Vertex1 = DownsizedKey;
    Vertex1 = _rotr(Vertex1, SEED3_BYTE1);
    Vertex1 *= SEED1;
    Vertex1 = _rotr(Vertex1, SEED3_BYTE2);

    Vertex2 = DownsizedKey;
    Vertex2 = _rotr(Vertex2, SEED3_BYTE3);
    Vertex2 *= SEED2;
    Vertex2 = _rotr(Vertex2, SEED3_BYTE4);

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
