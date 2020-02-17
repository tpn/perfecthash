//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexDummyAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexDummyAnd.c.\n"
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
    "    Vertex1 = _mm_crc32_u32(SEED1, DownsizedKey);\n"
    "    Vertex2 = _mm_crc32_u32(SEED2, _rotl(DownsizedKey, (BYTE)(SEED3 & 0x1f)));\n"
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
    "// End CompiledPerfectHashTableChm01IndexDummyAnd.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableChm01IndexDummyAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexDummyAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexDummyAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexDummyAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexDummyAndCSourceRawCString)
#endif
