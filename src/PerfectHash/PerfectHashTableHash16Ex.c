/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableHash16Ex.c

Abstract:

    This module implements routines that hash a 32-bit value into a 32-bit
    value comprised of two independent 16-bit hashes.  Each routine corresponds
    to one of the PERFECT_HASH_TABLE_HASH_FUNCTION_ID enumerations.

    This module is based on PerfectHashTableHashEx.c.

--*/

#include "stdafx.h"

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

NOINLINE
ULONG
PerfectHashTableSeededHash16ExNull(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
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
    UNREFERENCED_PARAMETER(Key);
    UNREFERENCED_PARAMETER(Seeds);
    UNREFERENCED_PARAMETER(Mask);
    return 0;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExCrc32Rotate15(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash routine uses a combination of CRC32 and rotates.  It is simple,
    fast, and generates reasonable quality hashes.  It is currently our default.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(SEED1, Key);
    Vertex2 = _mm_crc32_u32(SEED2, _rotl(Key, 15));

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExCrc32RotateX(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash routine is based off Crc32Rotate15, but uses a random value for
    the rotation amount (instead of a fixed 15).

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(SEED1, Key);
    Vertex2 = _mm_crc32_u32(SEED2, _rotl(Key, SEED3_BYTE1));

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExCrc32RotateXY(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash routine is based off Crc32RotateX, but uses a random value to
    rotate the key both left and right before calling Crc32.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG RotatedKey1;
    ULONG RotatedKey2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    //
    // Calculate the individual hash parts.
    //

    RotatedKey1 = _rotr(Key, SEED3_BYTE1);
    RotatedKey2 = _rotl(Key, SEED3_BYTE2);
    Vertex1 = _mm_crc32_u32(SEED1, RotatedKey1);
    Vertex2 = _mm_crc32_u32(SEED2, RotatedKey2);

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExCrc32RotateWXYZ(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash routine is based off Crc32RotateXY with more rotates.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(SEED1, _rotr(Key, SEED3_BYTE1));
    Vertex1 = _rotl(Vertex1, SEED3_BYTE2);

    Vertex2 = _mm_crc32_u32(SEED2, _rotl(Key, SEED3_BYTE3));
    Vertex2 = _rotr(Vertex2, SEED3_BYTE4);

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExRotateXor(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash routine uses some rotates and xor, inspired by nothing in
    particular other than it would be nice having a second hash routine,
    ideally with poor randomness characteristics, such that it makes a marked
    difference on the ability to solve the graphs.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Seed3;
    ULONG Seed4;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = Seeds[2];
    Seed4 = Seeds[3];

    //
    // Calculate the individual hash parts.
    //

    A = _rotl(Key ^ Seed1, 15);
    B = _rotl(Key + Seed2, 7);
    C = _rotr(Key - Seed3, 11);
    D = _rotr(Key ^ Seed4, 20);

    Vertex1 = A ^ C;
    Vertex2 = B ^ D;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExAddSubXor(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash routine is even simpler than the previous versions, using an
    add, sub and xor.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = Key + Seed1;
    Vertex2 = Key - Seed2;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExXor(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This is the simplest possible hash I could think of, simply xor'ing the
    input with each seed value.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Long1;
    ULONG_INTEGER Long2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Long1.LongPart = Vertex1 = Key ^ Seed1;
    Long2.LongPart = Vertex2 = _rotl(Key, 15) ^ Seed2;

    Long1.LowPart ^= Long1.HighPart;
    Long1.HighPart = 0;

    Long2.LowPart ^= Long2.HighPart;
    Long2.HighPart = 0;

    Result.LowPart = (Long1.LowPart & Mask);
    Result.HighPart = (Long2.LowPart & Mask);
    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExJenkins(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This routine implements the Jenkins hash routine.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG E;
    ULONG F;
    PBYTE Byte;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Byte = (PBYTE)&Key;

    //
    // Generate the first hash.
    //

    A = B = 0x9e3779b9;
    C = Seeds[0];

    A += (((ULONG)Byte[3]) << 24);
    A += (((ULONG)Byte[2]) << 16);
    A += (((ULONG)Byte[1]) <<  8);
    A += ((ULONG)Byte[0]);

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

    D += (((ULONG)Byte[3]) << 24);
    D += (((ULONG)Byte[2]) << 16);
    D += (((ULONG)Byte[1]) <<  8);
    D += ((ULONG)Byte[0]);

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExJenkinsMod(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This routine implements the Jenkins hash routine with a dummy modulus op
    thrown in for comparison purposes.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG E;
    ULONG F;
    ULONG Y;
    ULONG Z;
    PBYTE Byte;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Byte = (PBYTE)&Key;

    //
    // Generate the first hash.
    //

    A = B = 0x9e3779b9;
    C = Seeds[0];

    A += (((ULONG)Byte[3]) << 24);
    A += (((ULONG)Byte[2]) << 16);
    A += (((ULONG)Byte[1]) <<  8);
    A += ((ULONG)Byte[0]);

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

    D += (((ULONG)Byte[3]) << 24);
    D += (((ULONG)Byte[2]) << 16);
    D += (((ULONG)Byte[1]) <<  8);
    D += ((ULONG)Byte[0]);

    D -= E; D -= F; D ^= (F >> 13);
    E -= F; E -= D; E ^= (D <<  8);
    F -= D; F -= E; F ^= (E >> 13);
    D -= E; D -= F; D ^= (F >> 12);
    E -= F; E -= D; E ^= (D << 16);
    F -= D; F -= E; F ^= (E >>  5);
    D -= E; D -= F; D ^= (F >>  3);
    E -= F; E -= D; E ^= (D << 10);
    F -= D; F -= E; F ^= (E >> 15);

    Y = C % ((C >> 7) + 3);
    Z = F % ((F >> 5) + 9);

    Vertex1 = C;
    Vertex2 = F;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExCrc32RotateXor(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash routine uses a combination of CRC32 and rotates.  It is simple,
    fast, and generates reasonable quality hashes.  It is currently our default.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = Seeds[2];

    //
    // Calculate the individual hash parts.
    //

    A = _mm_crc32_u32(SEED1, Key);
    B = _mm_crc32_u32(SEED2, _rotl(Key, 15));
    C = Seed3 ^ Key;
    D = _mm_crc32_u32(B, C);

    Vertex1 = A;
    Vertex2 = D;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExScratch(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    A scratch routine that can be used to quickly iterate on hash development.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    XMMWORD KeyXmm;
    XMMWORD Seed1Xmm;
    XMMWORD Seed2Xmm;
    XMMWORD Vertex1Xmm;
    XMMWORD Vertex2Xmm;
    ULONG_INTEGER Result;

    //IACA_VC_START();

#if 0
    //
    // Load seeds.
    //

    Seed1Xmm = _mm_setr_epi32(Seeds[0], Seeds[1], Seeds[2], Seeds[3]);
    Seed2Xmm = _mm_setr_epi32(Seeds[4], Seeds[5], Seeds[6], Seeds[7]);

    //
    // Calculate the individual hash parts.
    //

    Vertex1Xmm = _mm_aesenc_si128(_mm_set1_epi32(Key), Seed1Xmm);
    Vertex2Xmm = _mm_aesenc_si128(_mm_set1_epi32(Key), Seed2Xmm);

    Result.LowPart = (Vertex1Xmm.m128i_u32[0] & Mask);
    Result.HighPart = (Vertex2Xmm.m128i_u32[0] & Mask);
#else

    Seed1Xmm = _mm_set1_epi32(Seeds[0]);
    Seed2Xmm = _mm_set1_epi32(Seeds[1]);

    KeyXmm = _mm_set1_epi32(Key);

    //
    // Calculate the individual hash parts.
    //

    Vertex1Xmm = _mm_aesenc_si128(KeyXmm, Seed1Xmm);
    Vertex2Xmm = _mm_aesenc_si128(KeyXmm, Seed2Xmm);

#ifdef PH_WINDOWS
    Result.LowPart = (Vertex1Xmm.m128i_u32[0] & Mask);
    Result.HighPart = (Vertex2Xmm.m128i_u32[0] & Mask);
#endif

#endif

    //IACA_VC_END();

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExDummy(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    A dummy routine used as a placeholder.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Key2;
    ULONG Seed1;
    ULONG Seed2;
    BYTE Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = (BYTE)(Seeds[2] & 0x1f);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(SEED1, Key);
    Key2 = _rotl(Key, Seed3);
    Vertex2 = _mm_crc32_u32(SEED2, Key2);

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExCrc32(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash routine uses two CRC32 instructions.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(SEED1, Key);
    Vertex2 = _mm_crc32_u32(SEED2, Key);

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExDjb(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash is based on the Daniel Bernstein hash.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG A;
    ULONG B;
    BYTE Byte1;
    BYTE Byte2;
    BYTE Byte3;
    BYTE Byte4;
    PBYTE Byte;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_BYTES Bytes;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Byte = (PBYTE)&Key;
    Bytes.AsULong = Key;

    Byte1 = (BYTE)((ULONG)Byte[0]);
    Byte2 = (BYTE)((ULONG)Byte[1]);
    Byte3 = (BYTE)((ULONG)Byte[2]);
    Byte4 = (BYTE)((ULONG)Byte[3]);

    A = Seed1;
    A = 33 * A + Byte1;
    A = 33 * A + Byte2;
    A = 33 * A + Byte3;
    A = 33 * A + Byte4;

    Vertex1 = A;

    B = Seed2;
    B = 33 * B + Bytes.Byte1;
    B = 33 * B + Bytes.Byte2;
    B = 33 * B + Bytes.Byte3;
    B = 33 * B + Bytes.Byte4;

    Vertex2 = B;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExDjbXor(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash is based on the Daniel Bernstein hash but uses XOR instead of
    add.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG A;
    ULONG B;
    BYTE Byte1;
    BYTE Byte2;
    BYTE Byte3;
    BYTE Byte4;
    PBYTE Byte;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Byte = (PBYTE)&Key;

    Byte1 = (BYTE)((ULONG)Byte[0]);
    Byte2 = (BYTE)((ULONG)Byte[1]);
    Byte3 = (BYTE)((ULONG)Byte[2]);
    Byte4 = (BYTE)((ULONG)Byte[3]);

    A = Seed1;
    A = 33 * A ^ Byte1;
    A = 33 * A ^ Byte2;
    A = 33 * A ^ Byte3;
    A = 33 * A ^ Byte4;

    Vertex1 = A;

    B = Seed2;
    B = 33 * B ^ Byte1;
    B = 33 * B ^ Byte2;
    B = 33 * B ^ Byte3;
    B = 33 * B ^ Byte4;

    Vertex2 = B;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExFnv(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash is based on the FNV (Fowler/Noll/Vo) hash.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG A;
    ULONG B;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_BYTES Bytes;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Bytes.AsULong = Key;

    A = Seed1 ^ 2166136261;
    A = 16777619 * A ^ Bytes.Byte1;
    A = 16777619 * A ^ Bytes.Byte2;
    A = 16777619 * A ^ Bytes.Byte3;
    A = 16777619 * A ^ Bytes.Byte4;

    Vertex1 = A;

    B = Seed2 ^ 2166136261;
    B = 16777619 * B ^ Bytes.Byte1;
    B = 16777619 * B ^ Bytes.Byte2;
    B = 16777619 * B ^ Bytes.Byte3;
    B = 16777619 * B ^ Bytes.Byte4;

    Vertex2 = B;

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExCrc32Not(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    This hash routine uses two CRC32 instructions.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Result;

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(SEED1, Key);
    Vertex2 = _mm_crc32_u32(SEED2, ~Key);

    Result.LowPart = (Vertex1 & Mask);
    Result.HighPart = (Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExRotateMultiplyXorRotate(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a rotate, multiply, xor, rotate.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExShiftMultiplyXorShift(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a shift, multiply, xor, shift.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExRotateMultiplyXorRotate2(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs two rotate, multiply, xor, rotate combinations.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    ULONG_BYTES Seed6;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExShiftMultiplyXorShift2(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs two shift, multiply, xor, shift combinations.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    ULONG_BYTES Seed6;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiplyRotateR(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then right rotate.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiplyRotateLR(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then left rotate on vertex 1, right rotate on vertex 2.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiplyShiftR(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiplyShiftRX(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.  Ignores mask.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Ignored.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

    UNREFERENCED_PARAMETER(Mask);

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
    Vertex2 = Vertex2 >> SEED3_BYTE1;

    Result.LowPart = (USHORT)Vertex1;
    Result.HighPart = (USHORT)Vertex2;

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiply643ShiftR(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG_BYTES Seed3;
    ULONGLONG Vertex1;
    ULONGLONG Vertex2;
    ULONGLONG DownsizedKey;
    ULARGE_INTEGER Seed1;
    ULARGE_INTEGER Seed2;
    ULONG_INTEGER Result;

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

    Result.LowPart = (USHORT)(Vertex1 & Mask);
    Result.HighPart = (USHORT)(Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiply644ShiftR(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG_BYTES Seed3;
    ULONGLONG Vertex1;
    ULONGLONG Vertex2;
    ULONGLONG DownsizedKey;
    ULARGE_INTEGER Seed1;
    ULARGE_INTEGER Seed2;
    ULONG_INTEGER Result;

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

    Result.LowPart = (USHORT)(Vertex1 & Mask);
    Result.HighPart = (USHORT)(Vertex2 & Mask);

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiplyShiftLR(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then shift left on vertex 2, shift right on vertext 2.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiply(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a single multiply on each vertex.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiplyXor(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then xor on each vertex.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG Seed3;
    ULONG Seed4;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiplyRotateRMultiply(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then right rotate then multiply.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiplyRotateR2(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then right rotate twice.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiplyShiftRMultiply(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift then multiply.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMultiplyShiftR2(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a multiply then right shift twice.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExRotateRMultiply(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a right rotate then multiply.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExRotateRMultiplyRotateR(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
/*++

Routine Description:

    Performs a right rotate, then multiply, then another right rotate.

Arguments:

    Key - Supplies the input value to hash.

    Seeds - Supplies an array of ULONG seed values.

    Mask - Supplies the mask to AND each vertex with.

Return Value:

    32-bit ULONG comprised of two 16-bit masked hashes.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

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

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMulshrolate1RX(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

    UNREFERENCED_PARAMETER(Mask);

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
    Vertex1 = _rotr(Vertex1, SEED3_BYTE2);
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE1;

    Result.LowPart = (USHORT)Vertex1;
    Result.HighPart = (USHORT)Vertex2;

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMulshrolate2RX(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

    UNREFERENCED_PARAMETER(Mask);

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
    Vertex1 = _rotr(Vertex1, SEED3_BYTE2);
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = _rotr(Vertex2, SEED3_BYTE3);
    Vertex2 = Vertex2 >> SEED3_BYTE1;

    Result.LowPart = (USHORT)Vertex1;
    Result.HighPart = (USHORT)Vertex2;

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMulshrolate3RX(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

    UNREFERENCED_PARAMETER(Mask);

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];
    Seed4 = Seeds[3];
    DownsizedKey = Key;

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = _rotr(Vertex1, SEED3_BYTE2);
    Vertex1 = Vertex1 * SEED4;
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = _rotr(Vertex2, SEED3_BYTE3);
    Vertex2 = Vertex2 >> SEED3_BYTE1;

    Result.LowPart = (USHORT)Vertex1;
    Result.HighPart = (USHORT)Vertex2;

    return Result.LongPart;
}

_Use_decl_annotations_
ULONG
PerfectHashTableSeededHash16ExMulshrolate4RX(
    ULONG Key,
    PULONG Seeds,
    USHORT Mask
    )
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG_BYTES Seed3;
    ULONG Seed4;
    ULONG Seed5;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG DownsizedKey;
    ULONG_INTEGER Result;

    UNREFERENCED_PARAMETER(Mask);

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
    Vertex1 = _rotr(Vertex1, SEED3_BYTE2);
    Vertex1 = Vertex1 * SEED4;
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = _rotr(Vertex2, SEED3_BYTE3);
    Vertex2 = Vertex2 * SEED5;
    Vertex2 = Vertex2 >> SEED3_BYTE1;

    Result.LowPart = (USHORT)Vertex1;
    Result.HighPart = (USHORT)Vertex2;

    return Result.LongPart;
}

#define EXPAND_AS_HASH16_EX_ROUTINE(Name, NumberOfSeeds, SeedMasks) \
ULONGLONG                                                           \
PerfectHashTableHash16Ex##Name(                                     \
    PPERFECT_HASH_TABLE Table,                                      \
    ULONG Key                                                       \
    )                                                               \
{                                                                   \
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;         \
    return PerfectHashTableSeededHash16Ex##Name(                    \
        Key,                                                        \
        &TableInfo->FirstSeed,                                      \
        (USHORT)TableInfo->HashMask                                 \
    );                                                              \
}

PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_HASH16_EX_ROUTINE);

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
