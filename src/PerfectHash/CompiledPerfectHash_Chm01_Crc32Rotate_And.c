/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    CompiledPerfectHash_Chm01_Crc32Rotate_And.c

Abstract:

    This module implements index, lookup, insert and delete routines for the
    CHM algorithm, CRC32Rotate hash function and And masking type for the
    perfect hash table library.

--*/

#include <CompiledPerfectHashTable.h>

COMPILED_PERFECT_HASH_TABLE_INDEX
    CompiledPerfectHash_Chm01_Crc32Rotate_And_Index;

_Use_decl_annotations_
ULONG
CompiledPerfectHash_Chm01_Crc32Rotate_And_Index(
    ULONG Key
    )
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG Index;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG MaskedLow;
    ULONG MaskedHigh;
    ULONGLONG Combined;

    //IACA_VC_START();

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

    //
    // Mask each hash value such that it falls within the confines of the
    // number of vertices.
    //

    MaskedLow = Vertex1 & HashMask;
    MaskedHigh = Vertex2 & HashMask;

    //
    // Obtain the corresponding vertex values for the masked high and low hash
    // values.  These are derived from the "assigned" array that we construct
    // during the creation routine's assignment step (GraphAssign()).
    //

    Vertex1 = TableData[MaskedLow];
    Vertex2 = TableData[MaskedHigh];

    //
    // Combine the two values, then perform the index masking operation, such
    // that our final index into the array falls within the confines of the
    // number of edges, or keys, in the table.  That is, make sure the index
    // value is between 0 and Table->Keys->NumberOfElements-1.
    //

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Index = Combined & IndexMask;

    return Index;
}

COMPILED_PERFECT_HASH_TABLE_LOOKUP
    CompiledPerfectHash_Chm01_Crc32Rotate_And_Lookup;

_Use_decl_annotations_
ULONG
CompiledPerfectHash_Chm01_Crc32Rotate_And_Lookup(
    ULONG Key
    )
{
    ULONG Index;

    Index = CompiledPerfectHash_Chm01_Crc32Rotate_And_Index(Key);
    return TableValues[Index];
}

COMPILED_PERFECT_HASH_TABLE_INSERT
    CompiledPerfectHash_Chm01_Crc32Rotate_And_Insert;

_Use_decl_annotations_
ULONG
CompiledPerfectHash_Chm01_Crc32Rotate_And_Insert(
    ULONG Key,
    ULONG Value
    )
{
    ULONG Index;
    ULONG Previous;

    Index = CompiledPerfectHash_Chm01_Crc32Rotate_And_Index(Key);
    Previous = TableValues[Index];
    TableValues[Index] = Value;
    return Previous;
}

COMPILED_PERFECT_HASH_TABLE_DELETE
    CompiledPerfectHash_Chm01_Crc32Rotate_And_Delete;

_Use_decl_annotations_
ULONG
CompiledPerfectHash_Chm01_Crc32Rotate_And_Delete(
    ULONG Key
    )
{
    ULONG Index;
    ULONG Previous;

    Index = CompiledPerfectHash_Chm01_Crc32Rotate_And_Index(Key);
    Previous = TableValues[Index];
    TableValues[Index] = 0;
    return Previous;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
