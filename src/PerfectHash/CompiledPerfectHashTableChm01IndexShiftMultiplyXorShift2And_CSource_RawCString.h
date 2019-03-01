//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexShiftMultiplyXorShift2AndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexShiftMultiplyXorShift2And.c.\n"
    "//\n"
    "\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
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
    "\n"
    "    Vertex1 = DownsizedKey >> SEED3_BYTE1;\n"
    "    Vertex1 *= SEED1;\n"
    "    Vertex1 ^= Vertex1 >> SEED3_BYTE2;\n"
    "    Vertex1 *= SEED2;\n"
    "    Vertex1 ^= Vertex1 >> SEED3_BYTE3;\n"
    "\n"
    "    Vertex2 = DownsizedKey >> SEED6_BYTE1;\n"
    "    Vertex2 *= SEED4;\n"
    "    Vertex2 ^= Vertex2 >> SEED6_BYTE2;\n"
    "    Vertex2 *= SEED5;\n"
    "    Vertex2 ^= Vertex2 >> SEED6_BYTE3;\n"
    "\n"
    "    MaskedLow = Vertex1 & HASH_MASK;\n"
    "    MaskedHigh = Vertex2 & HASH_MASK;\n"
    "\n"
    "    Vertex1 = TABLE_DATA[MaskedLow];\n"
    "    Vertex2 = TABLE_DATA[MaskedHigh];\n"
    "\n"
    "    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);\n"
    "\n"
    "    //IACA_VC_END();\n"
    "\n"
    "    return Index;\n"
    "}\n"
    "\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableChm01IndexShiftMultiplyXorShift2And.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableChm01IndexShiftMultiplyXorShift2AndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexShiftMultiplyXorShift2AndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexShiftMultiplyXorShift2AndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexShiftMultiplyXorShift2AndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexShiftMultiplyXorShift2AndCSourceRawCString)
#endif