
DECLARE_INDEX_ROUTINE()
{
    CPHDKEY A;
    CPHDKEY B;
    CPHDKEY C;
    CPHDKEY D;
    CPHINDEX Index;
    CPHDKEY Vertex1;
    CPHDKEY Vertex2;
    CPHDKEY MaskedLow;
    CPHDKEY MaskedHigh;

    //IACA_VC_START();

    DownsizedKey = DOWNSIZE_KEY(Key);

    A = _mm_crc32_u32(SEED1, DownsizedKey);
    B = _mm_crc32_u32(SEED2, _rotl(DownsizedKey, 15));
    C = SEED3 ^ DownsizedKey;
    D = _mm_crc32_u32(B, C);

    Vertex1 = A;
    Vertex2 = D;

    MaskedLow = Vertex1 & HASH_MASK;
    MaskedHigh = Vertex2 & HASH_MASK;

    Vertex1 = TABLE_DATA[MaskedLow];
    Vertex2 = TABLE_DATA[MaskedHigh];

    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);

    //IACA_VC_END();

    return Index;
}

