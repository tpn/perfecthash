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

NOINLINE
HRESULT
PerfectHashTableSeededHashNull(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
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
    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(Key);
    UNREFERENCED_PARAMETER(NumberOfSeeds);
    UNREFERENCED_PARAMETER(Seeds);
    UNREFERENCED_PARAMETER(Hash);
    return S_OK;
}

NOINLINE
HRESULT
PerfectHashTableHashNull(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashNull(Table,
                                          Key,
                                          TableInfo->NumberOfSeeds,
                                          &TableInfo->FirstSeed,
                                          Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashCrc32Rotate15(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
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
    ULONG Vertex1;
    ULONG Vertex2;
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    ASSERT(NumberOfSeeds >= 2);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(Seed1, Key);
    Vertex2 = _mm_crc32_u32(Seed2, _rotl(Key, 15));

    //IACA_VC_END();

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
PerfectHashTableHashCrc32Rotate15(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashCrc32Rotate15(Table,
                                                   Key,
                                                   TableInfo->NumberOfSeeds,
                                                   &TableInfo->FirstSeed,
                                                   Hash);
}


_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashCrc32RotateX(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash routine is based off Crc32Rotate15, but uses a random value for
    the rotation amount (instead of a fixed 15).

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
    ULONG Seed3;
    BYTE Rotate;
    ULONG Vertex1;
    ULONG Vertex2;
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    ASSERT(NumberOfSeeds >= 3);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = Seeds[2];
    Rotate = (BYTE)Seed3;

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(Seed1, Key);
    Vertex2 = _mm_crc32_u32(Seed2, _rotl(Key, Rotate));

    //IACA_VC_END();

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
PerfectHashTableHashCrc32RotateX(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashCrc32RotateX(Table,
                                                  Key,
                                                  TableInfo->NumberOfSeeds,
                                                  &TableInfo->FirstSeed,
                                                  Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashCrc32RotateXY(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash routine is based off Crc32RotateX, but uses a random value to
    rotate the key both left and right before calling Crc32.

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
    ULONG Seed3;
    ULONG Seed4;
    ULONG Vertex1;
    ULONG Vertex2;
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    ASSERT(NumberOfSeeds >= 3);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = Seeds[2];
    Seed4 = Seeds[3];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(Seed1, _rotr(Key, (BYTE)Seed3));
    Vertex2 = _mm_crc32_u32(Seed2, _rotl(Key, (BYTE)Seed4));

    //IACA_VC_END();

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
PerfectHashTableHashCrc32RotateXY(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashCrc32RotateXY(Table,
                                                   Key,
                                                   TableInfo->NumberOfSeeds,
                                                   &TableInfo->FirstSeed,
                                                   Hash);
}


_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashCrc32RotateWXYZ(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash routine is based off Crc32RotateXY with more rotates.

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
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    ASSERT(NumberOfSeeds >= 3);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3.AsULong = Seeds[2];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(Seed1, _rotr(Key, Seed3.Byte1));
    Vertex1 = _rotl(Vertex1, Seed3.Byte3);

    Vertex2 = _mm_crc32_u32(Seed2, _rotl(Key, Seed3.Byte2));
    Vertex2 = _rotr(Vertex2, Seed3.Byte4);

    //IACA_VC_END();

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
PerfectHashTableHashCrc32RotateWXYZ(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashCrc32RotateWXYZ(Table,
                                                     Key,
                                                     TableInfo->NumberOfSeeds,
                                                     &TableInfo->FirstSeed,
                                                     Hash);
}


_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashRotateXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
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

    Key - Supplies the input value to hash.

    NumberOfSeeds - Supplies the number of elements in the Seeds array.

    Seeds - Supplies an array of ULONG seed values.

    Hash - Receives two 32-bit hashes merged into a 64-bit value.

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

    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

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

    A = _rotl(Key ^ Seed1, 15);
    B = _rotl(Key + Seed2, 7);
    C = _rotr(Key - Seed3, 11);
    D = _rotr(Key ^ Seed4, 20);

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
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashRotateXor(Table,
                                               Key,
                                               TableInfo->NumberOfSeeds,
                                               &TableInfo->FirstSeed,
                                               Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashAddSubXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
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
    ULONG Vertex1;
    ULONG Vertex2;
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);

    ASSERT(NumberOfSeeds >= 2);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

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
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashAddSubXor(Table,
                                               Key,
                                               TableInfo->NumberOfSeeds,
                                               &TableInfo->FirstSeed,
                                               Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
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
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Long1;
    ULONG_INTEGER Long2;

    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);

    ASSERT(NumberOfSeeds >= 2);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    //IACA_VC_START();

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

    if (Long1.LowPart == Long2.LowPart) {
        return E_FAIL;
    }

    Result.LowPart = Long1.LowPart;
    Result.HighPart = Long2.LowPart;

    *Hash = Result.QuadPart;

    //IACA_VC_END();

    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableHashXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashXor(Table,
                                         Key,
                                         TableInfo->NumberOfSeeds,
                                         &TableInfo->FirstSeed,
                                         Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashJenkins(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This routine implements the Jenkins hash routine.

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

    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    ASSERT(NumberOfSeeds >= 2);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

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
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashJenkins(Table,
                                             Key,
                                             TableInfo->NumberOfSeeds,
                                             &TableInfo->FirstSeed,
                                             Hash);
}


_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashJenkinsMod(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This routine implements the Jenkins hash routine with a dummy modulus op
    thrown in for comparison purposes.

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
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    ASSERT(NumberOfSeeds >= 2);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

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

    //IACA_VC_END();

    Y = C % ((C >> 7) + 3);
    Z = F % ((F >> 5) + 9);

    Vertex1 = C;
    Vertex2 = F;

    if (Vertex1 == Vertex2) {
        return E_FAIL;
    }

    if (((Y + Z) << 4) < (Vertex1 >> 5)) {
        return E_FAIL;
    }

    Result.LowPart = Vertex1;
    Result.HighPart = Vertex2;

    *Hash = Result.QuadPart;

    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableHashJenkinsMod(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashJenkinsMod(Table,
                                                Key,
                                                TableInfo->NumberOfSeeds,
                                                &TableInfo->FirstSeed,
                                                Hash);
}

PERFECT_HASH_TABLE_SEEDED_HASH PerfectHashTableSeededHashCrc32RotateXor;

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashCrc32RotateXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
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

    Key - Supplies the input value to hash.

    NumberOfSeeds - Supplies the number of elements in the Seeds array.

    Seeds - Supplies an array of ULONG seed values.

    Hash - Receives two 32-bit hashes merged into a 64-bit value.

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

    UNREFERENCED_PARAMETER(Table);

    ASSERT(NumberOfSeeds >= 3);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

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

    A = _mm_crc32_u32(Seed1, Key);
    B = _mm_crc32_u32(Seed2, _rotl(Key, 15));
    C = Seed3 ^ Key;
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

PERFECT_HASH_TABLE_HASH PerfectHashTableHashCrc32RotateXor;

_Use_decl_annotations_
HRESULT
PerfectHashTableHashCrc32RotateXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashCrc32RotateXor(
        Table,
        Key,
        TableInfo->NumberOfSeeds,
        &TableInfo->FirstSeed,
        Hash
    );
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashScratch(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    A scratch routine that can be used to quickly iterate on hash development.

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
    ULONG Key2;
    ULONG Seed1;
    ULONG Seed2;
    BYTE Seed3;
    ULONG Vertex1;
    ULONG Vertex2;
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);

    ASSERT(NumberOfSeeds >= 3);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = (BYTE)(Seeds[2] & 0x1f);

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(Seed1, Key);
    Key2 = _rotl(Key, Seed3);
    Vertex2 = _mm_crc32_u32(Seed2, Key2);

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
PerfectHashTableHashScratch(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashScratch(Table,
                                             Key,
                                             TableInfo->NumberOfSeeds,
                                             &TableInfo->FirstSeed,
                                             Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashCrc32(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash routine uses two CRC32 instructions.

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
    ULONG Vertex1;
    ULONG Vertex2;
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);

    ASSERT(NumberOfSeeds >= 2);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(Seed1, Key);
    Vertex2 = _mm_crc32_u32(Seed2, Key);

    //IACA_VC_END();

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
PerfectHashTableHashCrc32(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashCrc32(Table,
                                           Key,
                                           TableInfo->NumberOfSeeds,
                                           &TableInfo->FirstSeed,
                                           Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashDjb(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash is based on the Daniel Bernstein hash.

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
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    ASSERT(NumberOfSeeds >= 2);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

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

    //IACA_VC_END();

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
PerfectHashTableHashDjb(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashDjb(Table,
                                         Key,
                                         TableInfo->NumberOfSeeds,
                                         &TableInfo->FirstSeed,
                                         Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashDjbXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash is based on the Daniel Bernstein hash but uses XOR instead of
    add.

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
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    ASSERT(NumberOfSeeds >= 2);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

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

    //IACA_VC_END();

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
PerfectHashTableHashDjbXor(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashDjbXor(Table,
                                            Key,
                                            TableInfo->NumberOfSeeds,
                                            &TableInfo->FirstSeed,
                                            Hash);
}


_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashFnv(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash is based on the FNV (Fowler/Noll/Vo) hash.

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
    ULONG A;
    ULONG B;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_BYTES Bytes;
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);

    ASSERT(NumberOfSeeds >= 2);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    //
    // Initialize aliases.
    //

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Bytes.AsULong = Key;

    //IACA_VC_START();

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

    //IACA_VC_END();

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
PerfectHashTableHashFnv(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashFnv(Table,
                                         Key,
                                         TableInfo->NumberOfSeeds,
                                         &TableInfo->FirstSeed,
                                         Hash);
}

_Use_decl_annotations_
HRESULT
PerfectHashTableSeededHashCrc32Not(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    ULONG NumberOfSeeds,
    PULONG Seeds,
    PULONGLONG Hash
    )
/*++

Routine Description:

    This hash routine uses two CRC32 instructions.

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
    ULONG Vertex1;
    ULONG Vertex2;
    ULARGE_INTEGER Result;

    UNREFERENCED_PARAMETER(Table);
    UNREFERENCED_PARAMETER(NumberOfSeeds);

    ASSERT(NumberOfSeeds >= 2);

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seed1 = Seeds[0];
    Seed2 = Seeds[1];

    //
    // Calculate the individual hash parts.
    //

    Vertex1 = _mm_crc32_u32(Seed1, Key);
    Vertex2 = _mm_crc32_u32(Seed2, ~Key);

    //IACA_VC_END();

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
PerfectHashTableHashCrc32Not(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONGLONG Hash
    )
{
    PTABLE_INFO_ON_DISK TableInfo = Table->TableInfoOnDisk;
    return PerfectHashTableSeededHashCrc32Not(Table,
                                              Key,
                                              TableInfo->NumberOfSeeds,
                                              &TableInfo->FirstSeed,
                                              Hash);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
