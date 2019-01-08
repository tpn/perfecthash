//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexCrc32RotateWXYZAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexCrc32RotateWXYZAnd.c.\n"
    "//\n"
    "\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
    "    ULONG Index;\n"
    "    ULONG Vertex1;\n"
    "    ULONG Vertex2;\n"
    "    ULONG MaskedLow;\n"
    "    ULONG MaskedHigh;\n"
    "    ULONGLONG Combined;\n"
    "\n"
    "    Vertex1 = _mm_crc32_u32(SEED1, _rotr(Key, SEED3_BYTE1));\n"
    "    Vertex1 = _rotl(Vertex1, SEED3_BYTE3);\n"
    "\n"
    "    Vertex2 = _mm_crc32_u32(SEED2, _rotl(Key, SEED3_BYTE2));\n"
    "    Vertex2 = _rotr(Vertex2, SEED3_BYTE4);\n"
    "\n"
    "    MaskedLow = Vertex1 & HASH_MASK;\n"
    "    MaskedHigh = Vertex2 & HASH_MASK;\n"
    "\n"
    "    Vertex1 = TABLE_DATA[MaskedLow];\n"
    "    Vertex2 = TABLE_DATA[MaskedHigh];\n"
    "\n"
    "    Combined = (ULONGLONG)Vertex1 + (ULONGLONG)Vertex2;\n"
    "\n"
    "    Index = Combined & INDEX_MASK;\n"
    "\n"
    "    return Index;\n"
    "}\n"
    "\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableChm01IndexCrc32RotateWXYZAnd.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableChm01IndexCrc32RotateWXYZAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexCrc32RotateWXYZAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexCrc32RotateWXYZAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexCrc32RotateWXYZAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexCrc32RotateWXYZAndCSourceRawCString)
#endif
