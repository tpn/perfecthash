
//
// Begin CompiledPerfectHashTableChm01IndexRotateXorAnd.c.
//

DECLARE_INDEX_ROUTINE()
{
    ULONG A;
    ULONG B;
    ULONG C;
    ULONG D;
    ULONG Vertex1;
    ULONG Vertex2;
    ULARGE_INTEGER Result;

    A = _rotl(Key ^ SEED1, 15);
    B = _rotl(Key + SEED2, 7);
    C = _rotr(Key - SEED3, 11);
    D = _rotr(Key ^ SEED4, 20);

    Vertex1 = A ^ C;
    Vertex2 = B ^ D;

    MaskedLow = Vertex1 & HASH_MASK;
    MaskedHigh = Vertex2 & HASH_MASK;

    Vertex1 = TABLE_DATA[MaskedLow];
    Vertex2 = TABLE_DATA[MaskedHigh];

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Index = Combined & INDEX_MASK;

    return Index;
}

//
// End CompiledPerfectHashTableChm01IndexRotateXorAnd.c.
//
