//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexDjbAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexDjbAnd.c.\n"
    "//\n"
    "\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
    "    CPHINDEX Index;\n"
    "    CPHDKEY A;\n"
    "    CPHDKEY B;\n"
    "    CPHDKEY Vertex1;\n"
    "    CPHDKEY Vertex2;\n"
    "    CPHDKEY MaskedLow;\n"
    "    CPHDKEY MaskedHigh;\n"
    "    CPHDKEY DownsizedKey;\n"
    "    ULONG_BYTES Bytes;\n"
    "\n"
    "    DownsizedKey = DOWNSIZE_KEY(Key);\n"
    "    Bytes.AsULong = DownsizedKey;\n"
    "\n"
    "    A = SEED1;\n"
    "    A = 33 * A + Bytes.Byte1;\n"
    "    A = 33 * A + Bytes.Byte2;\n"
    "    A = 33 * A + Bytes.Byte3;\n"
    "    A = 33 * A + Bytes.Byte4;\n"
    "\n"
    "    Vertex1 = A;\n"
    "\n"
    "    B = SEED2;\n"
    "    B = 33 * B + Bytes.Byte1;\n"
    "    B = 33 * B + Bytes.Byte2;\n"
    "    B = 33 * B + Bytes.Byte3;\n"
    "    B = 33 * B + Bytes.Byte4;\n"
    "\n"
    "    Vertex2 = B;\n"
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
    "// End CompiledPerfectHashTableChm01IndexDjbAnd.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableChm01IndexDjbAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexDjbAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexDjbAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexDjbAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexDjbAndCSourceRawCString)
#endif
