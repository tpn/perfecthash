
DECLARE_INDEX_ROUTINE()
{
    ULONG Index;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG MaskedLow;
    ULONG MaskedHigh;
    ULONGLONG Combined;

    Vertex1 = _mm_crc32_u32(SEED1, _rotr(Key, SEED3_BYTE1));
    Vertex2 = _mm_crc32_u32(SEED2, _rotl(Key, SEED3_BYTE2));

    MaskedLow = Vertex1 & HASH_MASK;
    MaskedHigh = Vertex2 & HASH_MASK;

    Vertex1 = TABLE_DATA[MaskedLow];
    Vertex2 = TABLE_DATA[MaskedHigh];

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Index = Combined & INDEX_MASK;

    return Index;
}

