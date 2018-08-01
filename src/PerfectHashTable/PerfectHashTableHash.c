/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableHash.c

Abstract:

    This module implements routines that hash a 32-bit value into a 64-bit
    value comprised of two independent 32-bit hashes.  Each routine corresponds
    to one of the PERFECT_HASH_TABLE_HASH_FUNCTION_ID enumerations.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashCrc32Rotate(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash routine uses a combination of CRC32 and rotates.  It is simple,
    fast, and generates reasonable quality hashes.  It is currently our default.

Arguments:

    Table - Supplies a pointer to the table for which the hash is being created.

    Input - Supplies the input value to hash.

    NumberOfSeeds - Supplies the number of elements in the Seeds array.

    Seeds - Supplies an array of ULONG seed values.

    Masked - Receives two 32-bit hashes merged into a 64-bit value.

Return Value:

    S_OK on success.  If the two 32-bit hash values are identical, E_FAIL.

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
    ULARGE_INTEGER Result;

    ASSERT(NumberOfSeeds >= 3);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = Seeds[2];

    //
    // Calculate the individual hash parts.
    //

    A = _mm_crc32_u32(Seed1, Input);
    B = _mm_crc32_u32(Seed2, _rotl(Input, 15));
    C = Seed3 ^ Input;
    D = _mm_crc32_u32(B, C);

    //IACA_VC_END();

    Vertex1 = A;
    Vertex2 = D;

    if (Vertex1 == Vertex2) {
        return E_FAIL;
    }

    Result.LowPart = Vertex1;
    Result.HighPart = Vertex2;

    *Hash = Result.QuadPart;

    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableHashCrc32Rotate(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONGLONG Hash
    )
{
    return PerfectHashTableSeededHashCrc32Rotate(Table,
                                                 Input,
                                                 Table->Header->NumberOfSeeds,
                                                 &Table->Header->FirstSeed,
                                                 Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashRotateXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash routine uses some rotates and xor, inspired by nothing in
    particular other than it would be nice having a second hash routine,
    ideally with poor randomness characteristics, such that it makes a marked
    difference on the ability to solve the graphs.

Arguments:

    Table - Supplies a pointer to the table for which the hash is being created.

    Input - Supplies the input value to hash.

    NumberOfSeeds - Supplies the number of elements in the Seeds array.

    Seeds - Supplies an array of ULONG seed values.

    Masked - Receives two 32-bit hashes merged into a 64-bit value.

Return Value:

    S_OK on success.  If the two 32-bit hash values are identical, E_FAIL.

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
    ULARGE_INTEGER Result;

    ASSERT(NumberOfSeeds >= 3);

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

    A = _rotl(Input ^ Seed1, 15);
    B = _rotl(Input + Seed2, 7);
    C = _rotr(Input - Seed3, 11);
    D = _rotr(Input ^ Seed4, 20);

    Vertex1 = A ^ C;
    Vertex2 = B ^ D;

    if (Vertex1 == Vertex2) {
        return E_FAIL;
    }

    Result.LowPart = Vertex1;
    Result.HighPart = Vertex2;

    *Hash = Result.QuadPart;
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableHashRotateXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONGLONG Hash
    )
{
    return PerfectHashTableSeededHashRotateXor(Table,
                                               Input,
                                               Table->Header->NumberOfSeeds,
                                               &Table->Header->FirstSeed,
                                               Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashAddSubXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash routine is even simpler than the previous versions, using an
    add, sub and xor.

Arguments:

    Table - Supplies a pointer to the table for which the hash is being created.

    Input - Supplies the input value to hash.

    NumberOfSeeds - Supplies the number of elements in the Seeds array.

    Seeds - Supplies an array of ULONG seed values.

    Masked - Receives two 32-bit hashes merged into a 64-bit value.

Return Value:

    S_OK on success.  If the two 32-bit hash values are identical, E_FAIL.

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
    ULARGE_INTEGER Result;

    ASSERT(NumberOfSeeds >= 3);

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = Seeds[2];

    //
    // Calculate the individual hash parts.
    //

    A = Input + Seed1;
    B = Input - Seed2;
    C = A ^ B;
    D = C ^ Seed3;

    Vertex1 = A;
    Vertex2 = D;

    if (Vertex1 == Vertex2) {
        return E_FAIL;
    }

    Result.LowPart = Vertex1;
    Result.HighPart = Vertex2;

    *Hash = Result.QuadPart;
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableHashAddSubXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONGLONG Hash
    )
{
    return PerfectHashTableSeededHashAddSubXor(Table,
                                               Input,
                                               Table->Header->NumberOfSeeds,
                                               &Table->Header->FirstSeed,
                                               Hash);
}


_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This is the simplest possible hash I could think of, simply xor'ing the
    input with each seed value.

Arguments:

    Table - Supplies a pointer to the table for which the hash is being created.

    Input - Supplies the input value to hash.

    NumberOfSeeds - Supplies the number of elements in the Seeds array.

    Seeds - Supplies an array of ULONG seed values.

    Masked - Receives two 32-bit hashes merged into a 64-bit value.

Return Value:

    S_OK on success.  If the two 32-bit hash values are identical, E_FAIL.

--*/
{
    ULONG Seed1;
    ULONG Seed2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULARGE_INTEGER Result;

    ASSERT(NumberOfSeeds >= 2);

    //IACA_VC_START();

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = Input ^ Seed1;
    Vertex2 = Input ^ Seed2;

    if (Vertex1 == Vertex2) {
        return E_FAIL;
    }

    Result.LowPart = Vertex1;
    Result.HighPart = Vertex2;

    *Hash = Result.QuadPart;

    //IACA_VC_END();

    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableHashXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONGLONG Hash
    )
{
    return PerfectHashTableSeededHashXor(Table,
                                         Input,
                                         Table->Header->NumberOfSeeds,
                                         &Table->Header->FirstSeed,
                                         Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashJenkins(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This routine implements the Jenkins hash routine.

Arguments:

    Table - Supplies a pointer to the table for which the hash is being created.

    Input - Supplies the input value to hash.

    NumberOfSeeds - Supplies the number of elements in the Seeds array.

    Seeds - Supplies an array of ULONG seed values.

    Masked - Receives two 32-bit hashes merged into a 64-bit value.

Return Value:

    S_OK on success.  If the two 32-bit hash values are identical, E_FAIL.

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
    ULARGE_INTEGER Result;

    ASSERT(NumberOfSeeds >= 2);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Byte = (PBYTE)&Input;

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

    //IACA_VC_END();

    Vertex1 = C;
    Vertex2 = F;

    if (Vertex1 == Vertex2) {
        return E_FAIL;
    }

    Result.LowPart = Vertex1;
    Result.HighPart = Vertex2;

    *Hash = Result.QuadPart;

    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableHashJenkins(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONGLONG Hash
    )
{
    return PerfectHashTableSeededHashJenkins(Table,
                                             Input,
                                             Table->Header->NumberOfSeeds,
                                             &Table->Header->FirstSeed,
                                             Hash);
}


_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashCrc32Rotate2(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash routine uses a combination of CRC32 and rotates.  It is simple,
    fast, and generates reasonable quality hashes.  It is currently our default.

Arguments:

    Table - Supplies a pointer to the table for which the hash is being created.

    Input - Supplies the input value to hash.

    NumberOfSeeds - Supplies the number of elements in the Seeds array.

    Seeds - Supplies an array of ULONG seed values.

    Masked - Receives two 32-bit hashes merged into a 64-bit value.

Return Value:

    S_OK on success.  If the two 32-bit hash values are identical, E_FAIL.

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
    ULARGE_INTEGER Result;

    ASSERT(NumberOfSeeds >= 3);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = Seeds[2];

    //
    // Calculate the individual hash parts.
    //

    A = _mm_crc32_u32(Seed1, Input);
    B = _mm_crc32_u32(Seed2, _rotl(Input, 15));
    C = Seed3 ^ Input;
    D = _mm_crc32_u32(B, C);

    //IACA_VC_END();

    Vertex1 = A;
    Vertex2 = D;

    if (Vertex1 == Vertex2) {
        return E_FAIL;
    }

    Result.LowPart = Vertex1;
    Result.HighPart = Vertex2;

    *Hash = Result.QuadPart;

    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableHashCrc32Rotate2(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONGLONG Hash
    )
{
    return PerfectHashTableSeededHashCrc32Rotate(Table,
                                                 Input,
                                                 Table->Header->NumberOfSeeds,
                                                 &Table->Header->FirstSeed,
                                                 Hash);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
