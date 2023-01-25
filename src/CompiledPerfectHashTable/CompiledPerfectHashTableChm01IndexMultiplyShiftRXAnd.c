
DECLARE_INDEX_ROUTINE()
{
    CPHINDEX Index;
    CPHDKEY Vertex1;
    CPHDKEY Vertex2;
    CPHDKEY DownsizedKey;

    //IACA_VC_START();

    DownsizedKey = DOWNSIZE_KEY(Key);

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = Vertex1 >> SEED3_BYTE1;
    Vertex1 = TABLE_DATA[Vertex1];

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE2;
    Vertex2 = TABLE_DATA[Vertex2];

    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);

    //IACA_VC_END();

    return Index;
}

#ifndef CPH_INLINE_ROUTINES

DECLARE_INDEX_IACA_ROUTINE()
{
    CPHINDEX Index;
    CPHDKEY Vertex1;
    CPHDKEY Vertex2;
    CPHDKEY DownsizedKey;

    IACA_VC_START();

    DownsizedKey = DOWNSIZE_KEY(Key);

    Vertex1 = DownsizedKey * SEED1;
    Vertex1 = Vertex1 >> SEED3_BYTE1;
    Vertex1 = TABLE_DATA[Vertex1];

    Vertex2 = DownsizedKey * SEED2;
    Vertex2 = Vertex2 >> SEED3_BYTE2;
    Vertex2 = TABLE_DATA[Vertex2];

    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);

    IACA_VC_END();

    return Index;
}

#endif
