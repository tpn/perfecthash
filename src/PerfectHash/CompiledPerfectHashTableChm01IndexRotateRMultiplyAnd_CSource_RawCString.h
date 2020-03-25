//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexRotateRMultiplyAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexRotateRMultiplyAnd.c.\n"
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
    "    DownsizedKey = DOWNSIZE_KEY(Key);\n"
    "\n"
    "    Vertex1 = DownsizedKey;\n"
    "    Vertex1 = _rotr(Vertex1, SEED3_BYTE1);\n"
    "    Vertex1 *= SEED1;\n"
    "\n"
    "    Vertex2 = DownsizedKey;\n"
    "    Vertex2 = _rotr(Vertex2, SEED3_BYTE2);\n"
    "    Vertex2 *= SEED2;\n"
    "\n"
    "    MaskedLow = Vertex1 & HASH_MASK;\n"
    "    MaskedHigh = Vertex2 & HASH_MASK;\n"
    "\n"
    "    Vertex1 = TABLE_DATA[MaskedLow];\n"
    "    Vertex2 = TABLE_DATA[MaskedHigh];\n"
    "\n"
    "    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);\n"
    "\n"
    "    return Index;\n"
    "}\n"
    "\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableChm01IndexRotateRMultiplyAnd.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableChm01IndexRotateRMultiplyAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexRotateRMultiplyAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexRotateRMultiplyAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexRotateRMultiplyAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexRotateRMultiplyAndCSourceRawCString)
#endif
