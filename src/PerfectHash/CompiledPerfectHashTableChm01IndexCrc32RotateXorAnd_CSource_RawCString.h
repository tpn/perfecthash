//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexCrc32RotateXorAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexCrc32RotateXorAnd.c.\n"
    "//\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
    "    ULONG A;\n"
    "    ULONG B;\n"
    "    ULONG C;\n"
    "    ULONG D;\n"
    "    ULONG Index;\n"
    "    ULONG Vertex1;\n"
    "    ULONG Vertex2;\n"
    "    ULONG MaskedLow;\n"
    "    ULONG MaskedHigh;\n"
    "    ULONGLONG Combined;\n"
    "\n"
    "    A = _mm_crc32_u32(SEED1, Key);\n"
    "    B = _mm_crc32_u32(SEED2, _rotl(Key, 15));\n"
    "    C = SEED3 ^ Input;\n"
    "    D = _mm_crc32_u32(B, C);\n"
    "\n"
    "    Vertex1 = A;\n"
    "    Vertex2 = D;\n"
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
    "//\n"
    "// End CompiledPerfectHashTableChm01IndexCrc32RotateXorAnd.c.\n"
    "//\n"
;

const STRING CompiledPerfectHashTableChm01IndexCrc32RotateXorAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexCrc32RotateXorAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexCrc32RotateXorAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexCrc32RotateXorAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexCrc32RotateXorAndCSourceRawCString)
#endif
