//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexJenkinsModulusCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexJenkinsModulus.c.\n"
    "//\n"
    "\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
    "    CPHDKEY A;\n"
    "    CPHDKEY B;\n"
    "    CPHDKEY C;\n"
    "    CPHDKEY D;\n"
    "    CPHDKEY E;\n"
    "    CPHDKEY F;\n"
    "    PBYTE Byte;\n"
    "    CPHINDEX Index;\n"
    "    CPHDKEY Vertex1;\n"
    "    CPHDKEY Vertex2;\n"
    "    CPHDKEY MaskedLow;\n"
    "    CPHDKEY MaskedHigh;\n"
    "    CPHDKEY DownsizedKey;\n"
    "\n"
    "    //IACA_VC_START();\n"
    "\n"
    "    DownsizedKey = DOWNSIZE_KEY(Key);\n"
    "    Byte = (PBYTE)&DownsizedKey;\n"
    "\n"
    "    A = B = 0x9e3779b9;\n"
    "    C = SEED1;\n"
    "\n"
    "    A += (((ULONG)Byte[3]) << 24);\n"
    "    A += (((ULONG)Byte[2]) << 16);\n"
    "    A += (((ULONG)Byte[1]) <<  8);\n"
    "    A += ((ULONG)Byte[0]);\n"
    "\n"
    "    A -= B; A -= C; A ^= (C >> 13);\n"
    "    B -= C; B -= A; B ^= (A <<  8);\n"
    "    C -= A; C -= B; C ^= (B >> 13);\n"
    "    A -= B; A -= C; A ^= (C >> 12);\n"
    "    B -= C; B -= A; B ^= (A << 16);\n"
    "    C -= A; C -= B; C ^= (B >>  5);\n"
    "    A -= B; A -= C; A ^= (C >>  3);\n"
    "    B -= C; B -= A; B ^= (A << 10);\n"
    "    C -= A; C -= B; C ^= (B >> 15);\n"
    "\n"
    "    Vertex1 = C;\n"
    "\n"
    "    D = E = 0x9e3779b9;\n"
    "    F = SEED2;\n"
    "\n"
    "    D += (((ULONG)Byte[3]) << 24);\n"
    "    D += (((ULONG)Byte[2]) << 16);\n"
    "    D += (((ULONG)Byte[1]) <<  8);\n"
    "    D += ((ULONG)Byte[0]);\n"
    "\n"
    "    D -= E; D -= F; D ^= (F >> 13);\n"
    "    E -= F; E -= D; E ^= (D <<  8);\n"
    "    F -= D; F -= E; F ^= (E >> 13);\n"
    "    D -= E; D -= F; D ^= (F >> 12);\n"
    "    E -= F; E -= D; E ^= (D << 16);\n"
    "    F -= D; F -= E; F ^= (E >>  5);\n"
    "    D -= E; D -= F; D ^= (F >>  3);\n"
    "    E -= F; E -= D; E ^= (D << 10);\n"
    "    F -= D; F -= E; F ^= (E >> 15);\n"
    "\n"
    "    Vertex2 = F;\n"
    "\n"
    "    MaskedLow = Vertex1 % HASH_MODULUS;\n"
    "    MaskedHigh = Vertex2 % HASH_MODULUS;\n"
    "\n"
    "    Vertex1 = TABLE_DATA[MaskedLow];\n"
    "    Vertex2 = TABLE_DATA[MaskedHigh];\n"
    "\n"
    "    Index = (CPHINDEX)((Vertex1 + Vertex2) % INDEX_MODULUS);\n"
    "\n"
    "    //IACA_VC_END();\n"
    "\n"
    "    return Index;\n"
    "}\n"
    "\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableChm01IndexJenkinsModulus.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableChm01IndexJenkinsModulusCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexJenkinsModulusCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexJenkinsModulusCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexJenkinsModulusCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexJenkinsModulusCSourceRawCString)
#endif
