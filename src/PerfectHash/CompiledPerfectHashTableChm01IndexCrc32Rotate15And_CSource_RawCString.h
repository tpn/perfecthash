//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexCrc32Rotate15AndCSourceRawCStr[] =
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
    "    Vertex1 = _mm_crc32_u32(SEED1, Key);\n"
    "    Vertex2 = _mm_crc32_u32(SEED2, _rotl(Key, 15));\n"
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
;

const STRING CompiledPerfectHashTableChm01IndexCrc32Rotate15AndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexCrc32Rotate15AndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexCrc32Rotate15AndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexCrc32Rotate15AndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexCrc32Rotate15AndCSourceRawCString)
#endif
