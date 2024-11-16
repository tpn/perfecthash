
DECLARE_INDEX_ROUTINE()
{
    CPHINDEX Index;
    CPHDKEY A;
    CPHDKEY B;
    CPHDKEY C;
    CPHDKEY D;
    CPHDKEY Vertex1;
    CPHDKEY Vertex2;
    CPHDKEY DownsizedKey;

    //IACA_VC_START();

    DownsizedKey = DOWNSIZE_KEY(Key);

    A = RotateLeft32(DownsizedKey ^ SEED1, 15);
    B = RotateLeft32(DownsizedKey + SEED2, 7);
    C = RotateRight32(DownsizedKey - SEED3, 11);
    D = RotateRight32(DownsizedKey ^ SEED4, 20);

    Vertex1 = A ^ C;
    Vertex2 = B ^ D;

    MaskedLow = Vertex1 & HASH_MASK;
    MaskedHigh = Vertex2 & HASH_MASK;

    Vertex1 = TABLE_DATA[MaskedLow];
    Vertex2 = TABLE_DATA[MaskedHigh];

    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);

    //IACA_VC_END();

    return Index;
}

#ifndef CPH_INLINE_ROUTINES

DECLARE_INDEX_IACA_ROUTINE()
{
    CPHINDEX Index;
    CPHDKEY A;
    CPHDKEY B;
    CPHDKEY C;
    CPHDKEY D;
    CPHDKEY Vertex1;
    CPHDKEY Vertex2;
    CPHDKEY DownsizedKey;

    IACA_VC_START();

    DownsizedKey = DOWNSIZE_KEY(Key);

    A = RotateLeft32(DownsizedKey ^ SEED1, 15);
    B = RotateLeft32(DownsizedKey + SEED2, 7);
    C = RotateRight32(DownsizedKey - SEED3, 11);
    D = RotateRight32(DownsizedKey ^ SEED4, 20);

    Vertex1 = A ^ C;
    Vertex2 = B ^ D;

    MaskedLow = Vertex1 & HASH_MASK;
    MaskedHigh = Vertex2 & HASH_MASK;

    Vertex1 = TABLE_DATA[MaskedLow];
    Vertex2 = TABLE_DATA[MaskedHigh];

    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);

    IACA_VC_END();

    return Index;
}

#endif