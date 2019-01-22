
DECLARE_INDEX_ROUTINE()
{
    CPHDKEY A;
    CPHDKEY B;
    CPHINDEX Index;
    CPHDKEY Vertex1;
    CPHDKEY Vertex2;
    CPHDKEY MaskedLow;
    CPHDKEY MaskedHigh;
    CPHDKEY DownsizedKey;
    ULONG_BYTES Bytes;
    ULONGLONG Combined;

    DownsizedKey = DOWNSIZE_KEY(Key);
    Bytes.AsULong = DownsizedKey;

    A = SEED1;
    A = 33 * A ^ Bytes.Byte1;
    A = 33 * A ^ Bytes.Byte2;
    A = 33 * A ^ Bytes.Byte3;
    A = 33 * A ^ Bytes.Byte4;

    Vertex1 = A;

    B = SEED2;
    B = 33 * B ^ Bytes.Byte1;
    B = 33 * B ^ Bytes.Byte2;
    B = 33 * B ^ Bytes.Byte3;
    B = 33 * B ^ Bytes.Byte4;

    Vertex2 = B;

    MaskedLow = Vertex1 & HASH_MASK;
    MaskedHigh = Vertex2 & HASH_MASK;

    Vertex1 = TABLE_DATA[MaskedLow];
    Vertex2 = TABLE_DATA[MaskedHigh];

    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;

    Index = Combined & INDEX_MASK;

    return Index;
}

