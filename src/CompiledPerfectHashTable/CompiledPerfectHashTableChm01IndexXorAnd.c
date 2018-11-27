
//
// Begin CompiledPerfectHashTableChm01IndexXorAnd.c.
//

DECLARE_INDEX_ROUTINE()
{
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG_INTEGER Long1;
    ULONG_INTEGER Long2;

    Long1.LongPart = Key ^ Seed1;
    Long2.LongPart = _rotl(Key, 15) ^ Seed2;

    Long1.LowPart ^= Long1.HighPart;
    Long1.HighPart = 0;

    Long2.LowPart ^= Long2.HighPart;
    Long2.HighPart = 0;


    MaskedLow = Vertex1 & HASH_MASK;
    MaskedHigh = Vertex2 & HASH_MASK;

    Vertex1 = TABLE_DATA[MaskedLow];
    Vertex2 = TABLE_DATA[MaskedHigh];

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Index = Combined & INDEX_MASK;

    if (((Y + Z) << 4) > Index) {
        return Index;
    } else {
        return Y;
    }

}

//
// End CompiledPerfectHashTableChm01IndexXorAnd.c.
//
