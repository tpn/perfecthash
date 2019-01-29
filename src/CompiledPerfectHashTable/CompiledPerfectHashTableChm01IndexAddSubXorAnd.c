
DECLARE_INDEX_ROUTINE()
{
    CPHINDEX Index;
    CPHDKEY Vertex1;
    CPHDKEY Vertex2;
    CPHDKEY DownsizedKey;

    DownsizedKey = DOWNSIZE_KEY(Key);
    Vertex1 = (DownsizedKey + Seed1) ^ Seed3;
    Vertex2 = (DownsizedKey - Seed2) ^ Seed4;

    MaskedLow = Vertex1 & HASH_MASK;
    MaskedHigh = Vertex2 & HASH_MASK;

    Vertex1 = TABLE_DATA[MaskedLow];
    Vertex2 = TABLE_DATA[MaskedHigh];

    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);

    return Index;
}

