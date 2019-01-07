
DECLARE_INDEX_ROUTINE()
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

    A = _mm_crc32_u32(SEED1, Key);
    B = _mm_crc32_u32(SEED2, _rotl(Key, 15));
    C = SEED3 ^ Input;
    D = _mm_crc32_u32(B, C);

    Vertex1 = A;
    Vertex2 = D;

    MaskedLow = Vertex1 & HASH_MASK;
    MaskedHigh = Vertex2 & HASH_MASK;

    Vertex1 = TABLE_DATA[MaskedLow];
    Vertex2 = TABLE_DATA[MaskedHigh];

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Index = Combined & INDEX_MASK;

    return Index;
}

