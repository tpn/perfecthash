
//
// Begin CompiledPerfectHashTableChm01IndexAddSubXorAnd.c.
//

DECLARE_INDEX_ROUTINE()
{
    ULONG Vertex1;
    ULONG Vertex2;
    ULARGE_INTEGER Result;

    Vertex1 = (Key + Seed1) ^ Seed3;
    Vertex2 = (Key - Seed2) ^ Seed4;

    MaskedLow = Vertex1 & HASH_MASK;
    MaskedHigh = Vertex2 & HASH_MASK;

    Vertex1 = TABLE_DATA[MaskedLow];
    Vertex2 = TABLE_DATA[MaskedHigh];

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Index = Combined & INDEX_MASK;

    return Index;
}

//
// End CompiledPerfectHashTableChm01IndexAddSubXorAnd.c.
//
