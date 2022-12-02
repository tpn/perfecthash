
DECLARE_INDEX_ROUTINE()
{
    CPHDKEY A;
    CPHDKEY B;
    CPHDKEY C;
    CPHDKEY D;
    CPHDKEY E;
    CPHDKEY F;
    PBYTE Byte;
    CPHINDEX Index;
    CPHDKEY Vertex1;
    CPHDKEY Vertex2;
    CPHDKEY MaskedLow;
    CPHDKEY MaskedHigh;
    CPHDKEY DownsizedKey;

    //IACA_VC_START();

    DownsizedKey = DOWNSIZE_KEY(Key);
    Byte = (PBYTE)&DownsizedKey;

    A = B = 0x9e3779b9;
    C = SEED1;

    A += (((ULONG)Byte[3]) << 24);
    A += (((ULONG)Byte[2]) << 16);
    A += (((ULONG)Byte[1]) <<  8);
    A += ((ULONG)Byte[0]);

    A -= B; A -= C; A ^= (C >> 13);
    B -= C; B -= A; B ^= (A <<  8);
    C -= A; C -= B; C ^= (B >> 13);
    A -= B; A -= C; A ^= (C >> 12);
    B -= C; B -= A; B ^= (A << 16);
    C -= A; C -= B; C ^= (B >>  5);
    A -= B; A -= C; A ^= (C >>  3);
    B -= C; B -= A; B ^= (A << 10);
    C -= A; C -= B; C ^= (B >> 15);

    Vertex1 = C;

    D = E = 0x9e3779b9;
    F = SEED2;

    D += (((ULONG)Byte[3]) << 24);
    D += (((ULONG)Byte[2]) << 16);
    D += (((ULONG)Byte[1]) <<  8);
    D += ((ULONG)Byte[0]);

    D -= E; D -= F; D ^= (F >> 13);
    E -= F; E -= D; E ^= (D <<  8);
    F -= D; F -= E; F ^= (E >> 13);
    D -= E; D -= F; D ^= (F >> 12);
    E -= F; E -= D; E ^= (D << 16);
    F -= D; F -= E; F ^= (E >>  5);
    D -= E; D -= F; D ^= (F >>  3);
    E -= F; E -= D; E ^= (D << 10);
    F -= D; F -= E; F ^= (E >> 15);

    Vertex2 = F;

    MaskedLow = Vertex1 % HASH_MODULUS;
    MaskedHigh = Vertex2 % HASH_MODULUS;

    Vertex1 = TABLE_DATA[MaskedLow];
    Vertex2 = TABLE_DATA[MaskedHigh];

    Index = (CPHINDEX)((Vertex1 + Vertex2) % INDEX_MODULUS);

    //IACA_VC_END();

    return Index;
}

