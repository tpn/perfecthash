
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

    A = _rotl(DownsizedKey ^ SEED1, 15);
    B = _rotl(DownsizedKey + SEED2, 7);
    C = _rotr(DownsizedKey - SEED3, 11);
    D = _rotr(DownsizedKey ^ SEED4, 20);

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

