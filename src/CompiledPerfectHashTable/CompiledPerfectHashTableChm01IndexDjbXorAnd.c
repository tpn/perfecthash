
DECLARE_INDEX_ROUTINE()
{
    ULONG A;
    ULONG B;
    ULONG Index;
    ULONG Vertex1;
    ULONG Vertex2;
    ULONG MaskedLow;
    ULONG MaskedHigh;
    ULONG_BYTES Bytes;
    ULONGLONG Combined;

    Bytes.AsULong = Key;

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

