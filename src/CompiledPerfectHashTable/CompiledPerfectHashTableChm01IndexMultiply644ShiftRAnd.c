
DECLARE_INDEX_ROUTINE()
{
    CPHINDEX Index;
    CPHDKEY64 Vertex1;
    CPHDKEY64 Vertex2;
    CPHDKEY Final1;
    CPHDKEY Final2;
    CPHDKEY MaskedLow;
    CPHDKEY MaskedHigh;
    CPHDKEY64 DownsizedKey;

    //IACA_VC_START();

    DownsizedKey = Key;

    Vertex1 = DownsizedKey * (CPHSEED64)SEED1;
    Vertex1 = Vertex1 >> SEED3_BYTE1;

    Vertex2 = DownsizedKey * (CPHSEED64)SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE2;

    MaskedLow = (CPHDKEY)(Vertex1 & HASH_MASK);
    MaskedHigh = (CPHDKEY)(Vertex2 & HASH_MASK);

    Final1 = TABLE_DATA[MaskedLow];
    Final2 = TABLE_DATA[MaskedHigh];

    Index = (CPHINDEX)((Final1 + Final2) & INDEX_MASK);

    //IACA_VC_END();

    return Index;
}

