/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01Index.c

Abstract:

    This module implements the Index() routine for the CHM v1 algorithm, as well
    as custom FastIndex() routines for certain combinations of hash function and
    masking type.

--*/

#include "stdafx.h"

PERFECT_HASH_TABLE_INDEX PerfectHashTableIndexImplChm01;

_Use_decl_annotations_
HRESULT
PerfectHashTableIndexImplChm01(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Index
    )
/*++

Routine Description:

    Looks up given key in a perfect hash table and returns its index.

    N.B. If Key did not appear in the original set the hash table was created
         from, the behavior of this routine is undefined.  (In practice, the
         key will hash to either an existing key's location or an empty slot,
         so there is potential for returning a non-unique index.)

Arguments:

    Table - Supplies a pointer to the table for which the key lookup is to be
        performed.

    Key - Supplies the key to look up.

    Index - Receives the index associated with this key.  The index will be
        between 0 and Table->HashSize-1, and can be safely used to offset
        directly into an appropriately sized array (e.g. Table->Values[]).

Return Value:

    S_OK on success, E_FAIL if the underlying hash function returned a failure.
    This will happen if the two hash values for a key happen to be identical.
    It shouldn't happen once a perfect graph has been created (i.e. it only
    happens when attempting to solve the graph).  The Index parameter will
    be cleared in the case of E_FAIL.

--*/
{
    ULONG Masked;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG MaskedLow;
    ULONG MaskedHigh;
    PULONG Assigned;
    ULONGLONG Combined;
    ULARGE_INTEGER Hash;

    //
    // Hash the incoming key into the 64-bit representation, which is two
    // 32-bit ULONGs in disguise, each one driven by a separate seed value.
    //

    if (FAILED(Table->Vtbl->Hash(Table, Key, &Hash.QuadPart))) {
        goto Error;
    }

    //
    // Mask each hash value such that it falls within the confines of the
    // number of vertices.  That is, make sure the value is between 0 and
    // Table->NumberOfVertices-1.
    //

    if (FAILED(Table->Vtbl->MaskHash(Table, Hash.LowPart, &MaskedLow))) {
        goto Error;
    }

    if (FAILED(Table->Vtbl->MaskHash(Table, Hash.HighPart, &MaskedHigh))) {
        goto Error;
    }

    //
    // Obtain the corresponding vertex values for the masked high and low hash
    // values.  These are derived from the "assigned" array that we construct
    // during the creation routine's assignment step (GraphAssign()).
    //

    Assigned = (PULONG)Table->TableFile->BaseAddress;

    Vertex1 = Assigned[MaskedLow];
    Vertex2 = Assigned[MaskedHigh];

    //
    // Combine the two values, then perform the index masking operation, such
    // that our final index into the array falls within the confines of the
    // number of edges, or keys, in the table.  That is, make sure the index
    // value is between 0 and Table->Keys->NumberOfElements-1.
    //

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    if (FAILED(Table->Vtbl->MaskIndex(Table, Combined, &Masked))) {
        goto Error;
    }

    //
    // Update the caller's pointer and return success.  The resulting index
    // value represents the array offset index for this given key in the
    // underlying table, and is guaranteed to be unique amongst the original
    // keys in the input set.
    //

    *Index = Masked;
    return S_OK;

Error:

    //
    // Clear the caller's pointer and return failure.  We should only hit this
    // point if the caller supplies a key that both: a) wasn't in the original
    // input set, and b) happens to result in a hash value where both the high
    // part and low part are identical, which is rare, but not impossible.
    //

    *Index = 0;
    return E_FAIL;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableFastIndexImplChm01Crc32RotateHashAndMask(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Index
    )
/*++

Routine Description:

    Looks up given key in a perfect hash table and returns its index.  This
    is a fast version of the normal Index() routine that inlines the Crc32Rotate
    hash function and AND masking.

    N.B. If Key did not appear in the original set the hash table was created
         from, the behavior of this routine is undefined.  (In practice, the
         key will hash to either an existing key's location or an empty slot,
         so there is potential for returning a non-unique index.)

Arguments:

    Table - Supplies a pointer to the table for which the key lookup is to be
        performed.

    Key - Supplies the key to look up.

    Index - Receives the index associated with this key.  The index will be
        between 0 and Table->HashSize-1, and can be safely used to offset
        directly into an appropriately sized array (e.g. Table->Values[]).

Return Value:

    S_OK on success, E_FAIL if the underlying hash function returned a failure.
    This will happen if the two hash values for a key happen to be identical.
    It shouldn't happen once a perfect graph has been created (i.e. it only
    happens when attempting to solve the graph).  The Index parameter will
    be cleared in the case of E_FAIL.

--*/
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Seed3;
    ULONG Input;
    PULONG Seeds;
    ULONG Masked;
    ULONG Vertex1;
    ULONG Vertex2;
    PULONG Assigned;
    ULONG MaskedLow;
    ULONG MaskedHigh;
    ULONGLONG Combined;

    //IACA_VC_START();

    //
    // Initialize aliases.
    //

    Seeds = &Table->TableInfoOnDisk->FirstSeed;
    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Seed3 = Seeds[2];
    Input = Key;
    Assigned = Table->Assigned;

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
        goto Error;
    }

    //
    // Mask each hash value such that it falls within the confines of the
    // number of vertices.
    //

    MaskedLow = Vertex1 & Table->HashMask;
    MaskedHigh = Vertex2 & Table->HashMask;

    //
    // Obtain the corresponding vertex values for the masked high and low hash
    // values.  These are derived from the "assigned" array that we construct
    // during the creation routine's assignment step (GraphAssign()).
    //

    Vertex1 = Assigned[MaskedLow];
    Vertex2 = Assigned[MaskedHigh];

    //
    // Combine the two values, then perform the index masking operation, such
    // that our final index into the array falls within the confines of the
    // number of edges, or keys, in the table.  That is, make sure the index
    // value is between 0 and Table->Keys->NumberOfElements-1.
    //

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Masked = Combined & Table->IndexMask;

    //
    // Update the caller's pointer and return success.  The resulting index
    // value represents the array offset index for this given key in the
    // underlying table, and is guaranteed to be unique amongst the original
    // keys in the input set.
    //

    *Index = Masked;

    //IACA_VC_END();

    return S_OK;

Error:

    //
    // Clear the caller's pointer and return failure.  We should only hit this
    // point if the caller supplies a key that both: a) wasn't in the original
    // input set, and b) happens to result in a hash value where both the high
    // part and low part are identical, which is rare, but not impossible.
    //

    *Index = 0;
    return E_FAIL;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableFastIndexImplChm01JenkinsHashAndMask(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Index
    )
/*++

Routine Description:

    Looks up given key in a perfect hash table and returns its index.  This
    is a fast version of the normal Index() routine that inlines the Jenkins
    hash function and AND masking.

    N.B. If Key did not appear in the original set the hash table was created
         from, the behavior of this routine is undefined.  (In practice, the
         key will hash to either an existing key's location or an empty slot,
         so there is potential for returning a non-unique index.)

Arguments:

    Table - Supplies a pointer to the table for which the key lookup is to be
        performed.

    Key - Supplies the key to look up.

    Index - Receives the index associated with this key.  The index will be
        between 0 and Table->HashSize-1, and can be safely used to offset
        directly into an appropriately sized array (e.g. Table->Values[]).

Return Value:

    S_OK on success, E_FAIL if the underlying hash function returned a failure.
    This will happen if the two hash values for a key happen to be identical.
    It shouldn't happen once a perfect graph has been created (i.e. it only
    happens when attempting to solve the graph).  The Index parameter will
    be cleared in the case of E_FAIL.

--*/
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG E;
    ULONG F;
    PBYTE Byte;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Input;
    PULONG Seeds;
    ULONG Masked;
    ULONG Vertex1;
    ULONG Vertex2;
    PULONG Assigned;
    ULONG MaskedLow;
    ULONG MaskedHigh;
    ULONGLONG Combined;

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seeds = &Table->TableInfoOnDisk->FirstSeed;
    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Input = Key;

    Byte = (PBYTE)&Input;

    //
    // Generate the first hash.
    //

    A = B = 0x9e3779b9;
    C = Seed1;

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

    Vertex1 = C;

    //
    // Generate the second hash.
    //

    D = E = 0x9e3779b9;
    F = Seed2;

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

    Vertex2 = F;

    if (Vertex1 == Vertex2) {
        goto Error;
    }

    //
    // Mask each hash value such that it falls within the confines of the
    // number of vertices.
    //

    MaskedLow = Vertex1 & Table->HashMask;
    MaskedHigh = Vertex2 & Table->HashMask;

    //
    // Obtain the corresponding vertex values for the masked high and low hash
    // values.  These are derived from the "assigned" array that we construct
    // during the creation routine's assignment step (GraphAssign()).
    //

    Assigned = (PULONG)Table->TableFile->BaseAddress;

    Vertex1 = Assigned[MaskedLow];
    Vertex2 = Assigned[MaskedHigh];

    //
    // Combine the two values, then perform the index masking operation, such
    // that our final index into the array falls within the confines of the
    // number of edges, or keys, in the table.  That is, make sure the index
    // value is between 0 and Table->Keys->NumberOfElements-1.
    //

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Masked = Combined & Table->IndexMask;

    //
    // Update the caller's pointer and return success.  The resulting index
    // value represents the array offset index for this given key in the
    // underlying table, and is guaranteed to be unique amongst the original
    // keys in the input set.
    //

    *Index = Masked;

    //IACA_VC_END();

    return S_OK;

Error:

    //
    // Clear the caller's pointer and return failure.  We should only hit this
    // point if the caller supplies a key that both: a) wasn't in the original
    // input set, and b) happens to result in a hash value where both the high
    // part and low part are identical, which is rare, but not impossible.
    //

    *Index = 0;
    return E_FAIL;
}

PERFECT_HASH_TABLE_INDEX PerfectHashTableFastIndexImplChm01JenkinsHashModMask;

_Use_decl_annotations_
HRESULT
PerfectHashTableFastIndexImplChm01JenkinsHashModMask(
    PPERFECT_HASH_TABLE Table,
    ULONG Key,
    PULONG Index
    )
/*++

Routine Description:

    Looks up given key in a perfect hash table and returns its index.

    N.B. This version is based off the Jenkins hash function and modulus
         masking.  As we don't use modulus masking at all, it's not intended
         to be used in reality.  However, it's useful to feed to IACA to see
         the impact of the modulus operation.

Arguments:

    Table - Supplies a pointer to the table for which the key lookup is to be
        performed.

    Key - Supplies the key to look up.

    Index - Receives the index associated with this key.  The index will be
        between 0 and Table->HashSize-1, and can be safely used to offset
        directly into an appropriately sized array (e.g. Table->Values[]).

Return Value:

    S_OK on success, E_FAIL if the underlying hash function returned a failure.
    This will happen if the two hash values for a key happen to be identical.
    It shouldn't happen once a perfect graph has been created (i.e. it only
    happens when attempting to solve the graph).  The Index parameter will
    be cleared in the case of E_FAIL.

--*/
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG E;
    ULONG F;
    PBYTE Byte;
    ULONG Seed1;
    ULONG Seed2;
    ULONG Input;
    PULONG Seeds;
    ULONG Masked;
    ULONG Vertex1;
    ULONG Vertex2;
    PULONG Assigned;
    ULONG MaskedLow;
    ULONG MaskedHigh;
    ULONGLONG Combined;

    //
    // Initialize aliases.
    //

    //IACA_VC_START();

    Seeds = &Table->TableInfoOnDisk->FirstSeed;
    Seed1 = Seeds[0];
    Seed2 = Seeds[1];
    Input = Key;

    Byte = (PBYTE)&Input;

    //
    // Generate the first hash.
    //

    A = B = 0x9e3779b9;
    C = Seed1;

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

    Vertex1 = C;

    //
    // Generate the second hash.
    //

    D = E = 0x9e3779b9;
    F = Seed2;

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

    Vertex2 = F;

    if (Vertex1 == Vertex2) {
        goto Error;
    }

    //
    // Mask each hash value such that it falls within the confines of the
    // number of vertices.
    //

    MaskedLow = Vertex1 % Table->HashModulus;
    MaskedHigh = Vertex2 % Table->HashModulus;

    //
    // Obtain the corresponding vertex values for the masked high and low hash
    // values.  These are derived from the "assigned" array that we construct
    // during the creation routine's assignment step (GraphAssign()).
    //

    Assigned = (PULONG)Table->TableFile->BaseAddress;

    Vertex1 = Assigned[MaskedLow];
    Vertex2 = Assigned[MaskedHigh];

    //
    // Combine the two values, then perform the index masking operation, such
    // that our final index into the array falls within the confines of the
    // number of edges, or keys, in the table.  That is, make sure the index
    // value is between 0 and Table->Keys->NumberOfElements-1.
    //

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Masked = Combined % Table->IndexModulus;

    //
    // Update the caller's pointer and return success.  The resulting index
    // value represents the array offset index for this given key in the
    // underlying table, and is guaranteed to be unique amongst the original
    // keys in the input set.
    //

    *Index = Masked;

    //IACA_VC_END();

    return S_OK;

Error:

    //
    // Clear the caller's pointer and return failure.  We should only hit this
    // point if the caller supplies a key that both: a) wasn't in the original
    // input set, and b) happens to result in a hash value where both the high
    // part and low part are identical, which is rare, but not impossible.
    //

    *Index = 0;
    return E_FAIL;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
