/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableMask.c

Abstract:

    This module implements routines that mask a value such that it remains
    within the bounds of a given hash table's underlying size.  Each routine
    corresponds to one of the PERFECT_HASH_TABLE_MASK_FUNCTION_ID enumeration
    values.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskHashModulus(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Returns the input value modulus the table size.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    ULARGE_INTEGER Mask;

    Mask.QuadPart = Input;
    Mask.QuadPart %= (ULONGLONG)Table->HashModulus;

    ASSERT(!Mask.HighPart);

    *Masked = Mask.LowPart;
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskIndexModulus(
    PPERFECT_HASH_TABLE Table,
    ULONGLONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Returns the input value modulus the number of edges.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    *Masked = Input % Table->IndexModulus;
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskHashAnd(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Returns the input value masked (AND'd) by the table shift value minus 1.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    ULARGE_INTEGER Mask;

    Mask.QuadPart = Input;
    Mask.QuadPart &= Table->HashMask;

    *Masked = Mask.LowPart;
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskIndexAnd(
    PPERFECT_HASH_TABLE Table,
    ULONGLONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Returns the input value masked (AND'd) by the table shift value minus 1.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    ULARGE_INTEGER Mask;

    Mask.QuadPart = Input;
    Mask.QuadPart &= Table->IndexMask;

    *Masked = Mask.LowPart;
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskHashXorAnd(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Returns the high dword and low dword of the input XORd, then masked
    by the table size value minus 1.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    ULARGE_INTEGER Mask;

    Mask.QuadPart = Input;

    Mask.LowPart ^= Mask.HighPart;
    Mask.LowPart &= Table->HashMask;

    *Masked = Mask.LowPart;
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskIndexXorAnd(
    PPERFECT_HASH_TABLE Table,
    ULONGLONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Returns the high dword and low dword of the input XORd, then masked
    by the table size value minus 1.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    ULARGE_INTEGER Mask;

    Mask.QuadPart = Input;

    Mask.LowPart ^= Mask.HighPart;
    Mask.LowPart &= Table->IndexMask;

    *Masked = Mask.LowPart;
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskHashFoldOnce(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Folds all of the bits in input such that they participate in the XOR
    for the final masked value.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    ULONG Mask;
    ULONG Extra;
    ULONG Shift;
    ULONG PopCount;
    ULONG InvertedHashMask;

    Mask = Input;
    Mask &= Table->HashMask;

    //
    // Fold in the remaining bits that fall outside the current mask.
    //

    InvertedHashMask = ~Table->HashMask;

    Extra = Input;
    Extra &= InvertedHashMask;

    Shift = TrailingZeros(InvertedHashMask);
    Extra >>= Shift;

    Mask ^= Extra;

    *Masked = Mask;
    return S_OK;

    PopCount = PopulationCount32(Extra);

    if (PopCount & 1) {

        //
        // Extra bits are odd.
        //

        Mask -= Extra;

    } else {

        //
        // Extra bits are even.
        //

        Mask += Extra;

    }

    //Mask ^= Extra;

    *Masked = Mask & Table->HashMask;

    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskIndexFoldOnce(
    PPERFECT_HASH_TABLE Table,
    ULONGLONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Folds all of the bits in input such that they participate in the XOR
    for the final masked value.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    ULONG Mask;
    ULONG Extra;
    ULONG Shift;
    //ULONG PopCount;
    ULONG InvertedIndexMask;
    ULARGE_INTEGER Quad;

    //
    // Fold 64 -> 32.
    //

    Quad.QuadPart = Input;
    Quad.LowPart ^= Quad.HighPart;

    //
    // Mask the input value.
    //

    Mask = Quad.LowPart;
    Mask &= Table->IndexMask;

    //
    // Fold in the remaining bits that fall outside the current mask.
    //

    InvertedIndexMask = ~Table->IndexMask;

    Extra = Quad.LowPart;
    Extra &= InvertedIndexMask;

    Shift = TrailingZeros(InvertedIndexMask);
    Extra >>= Shift;

    Mask ^= Extra;

    *Masked = Mask;

    return S_OK;

#if 0
    PopCount = PopulationCount32(Extra);

    if (PopCount & 1) {

        //
        // Extra bits are odd.
        //

        Mask -= Extra;

    } else {

        //
        // Extra bits are even.
        //

        Mask += Extra;

    }

    Mask ^= Extra;

    *Masked = Mask & Table->Mask;

    return S_OK;
#endif

}

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskHashFoldTwice(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Folds all of the bits in input such that they participate in the XOR
    for the final masked value.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    ASSERT(FALSE);
    return E_FAIL;
#if 0
    ULONG Mask;
    ULONG Extra;
    ULARGE_INTEGER Quad;
    ULONG_INTEGER Long;

    //
    // Fold 64 -> 32.
    //

    Quad.QuadPart = Input;
    Quad.LowPart ^= Quad.HighPart;

    //
    // Fold 32 -> 16.
    //

    Long.LongPart = Quad.LowPart;
    Long.LowPart ^= Long.HighPart;

    Mask = Long.LowPart;
    Mask &= Table->Mask;

    //
    // Fold in the remaining bits that fall outside the current mask.
    //

    Extra = Long.LowPart;
    Extra &= ~Table->Mask;
    if (Extra) {
        __debugbreak();
    }
    Extra >>= TrailingZeros(Extra);

    Mask ^= Extra;

    *Masked = Mask;

    return S_OK;
#endif
}

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskIndexFoldTwice(
    PPERFECT_HASH_TABLE Table,
    ULONGLONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Folds all of the bits in input such that they participate in the XOR
    for the final masked value.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    ASSERT(FALSE);
    return E_FAIL;
#if 0
    ULONG Mask;
    ULONG Extra;
    ULARGE_INTEGER Quad;
    ULONG_INTEGER Long;

    //
    // Fold 64 -> 32.
    //

    Quad.QuadPart = Input;
    Quad.LowPart ^= Quad.HighPart;

    //
    // Fold 32 -> 16.
    //

    Long.LongPart = Quad.LowPart;
    Long.LowPart ^= Long.HighPart;

    Mask = Long.LowPart;
    Mask &= Table->Mask;

    //
    // Fold in the remaining bits that fall outside the current mask.
    //

    Extra = Long.LowPart;
    Extra &= ~Table->Mask;
    if (Extra) {
        __debugbreak();
    }
    Extra >>= TrailingZeros(Extra);

    Mask ^= Extra;

    *Masked = Mask;

    return S_OK;
#endif
}

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskHashFoldThrice(
    PPERFECT_HASH_TABLE Table,
    ULONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Folds all of the bits in input such that they participate in the XOR
    for the final masked value.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    ASSERT(FALSE);
    return E_FAIL;
#if 0
    ULONG Mask;
    ULONG Extra;
    ULARGE_INTEGER Quad;
    ULONG_INTEGER Long;
    USHORT_INTEGER Short;

    Quad.QuadPart = Input;
    Quad.LowPart ^= Quad.HighPart;

    Long.LongPart = Quad.LowPart;
    Long.LowPart ^= Long.HighPart;

    Short.ShortPart = Long.LowPart;
    Short.LowPart ^= Short.HighPart;

    //
    // Fold in the remaining bits that fall outside the current mask.
    //

    Extra = Short.LowPart;
    Extra &= ~Table->Mask;
    Extra >>= TrailingZeros(Extra);

    Mask ^= Extra;

    *Masked = Mask;
    return S_OK;
#endif
}

_Use_decl_annotations_
HRESULT
PerfectHashTableMaskIndexFoldThrice(
    PPERFECT_HASH_TABLE Table,
    ULONGLONG Input,
    PULONG Masked
    )
/*++

Routine Description:

    Folds all of the bits in input such that they participate in the XOR
    for the final masked value.

Arguments:

    Table - Supplies a pointer to the table for which the mask will be derived.

    Input - Supplies the input value to mask.

    Masked - Receives the masked value.

Return Value:

    S_OK.

--*/
{
    ASSERT(FALSE);
    return E_FAIL;
#if 0
    ULONG Mask;
    ULONG Extra;
    ULARGE_INTEGER Quad;
    ULONG_INTEGER Long;
    USHORT_INTEGER Short;

    Quad.QuadPart = Input;
    Quad.LowPart ^= Quad.HighPart;

    Long.LongPart = Quad.LowPart;
    Long.LowPart ^= Long.HighPart;

    Short.ShortPart = Long.LowPart;
    Short.LowPart ^= Short.HighPart;

    //
    // Fold in the remaining bits that fall outside the current mask.
    //

    Extra = Short.LowPart;
    Extra &= ~Table->Mask;
    Extra >>= TrailingZeros(Extra);

    Mask ^= Extra;

    *Masked = Mask;
    return S_OK;
#endif
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
