//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexCrc32AndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexCrc32And.c.\n"
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
    "    ULONGLONG Combined;\n"
    "\n"
    "    //IACA_VC_START();\n"
    "\n"
    "    DownsizedKey = DOWNSIZE_KEY(Key);\n"
    "    Vertex1 = Crc32u32(SEED1, DownsizedKey);\n"
    "    Vertex2 = Crc32u32(SEED2, DownsizedKey);\n"
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
    "    //IACA_VC_END();\n"
    "\n"
    "    return Index;\n"
    "}\n"
    "\n"
    "#ifndef CPH_INLINE_ROUTINES\n"
    "\n"
    "DECLARE_INDEX_IACA_ROUTINE()\n"
    "{\n"
    "    CPHINDEX Index;\n"
    "    CPHDKEY Vertex1;\n"
    "    CPHDKEY Vertex2;\n"
    "    CPHDKEY MaskedLow;\n"
    "    CPHDKEY MaskedHigh;\n"
    "    CPHDKEY DownsizedKey;\n"
    "    ULONGLONG Combined;\n"
    "\n"
    "    IACA_VC_START();\n"
    "\n"
    "    DownsizedKey = DOWNSIZE_KEY(Key);\n"
    "    Vertex1 = Crc32u32(SEED1, DownsizedKey);\n"
    "    Vertex2 = Crc32u32(SEED2, DownsizedKey);\n"
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
    "    IACA_VC_END();\n"
    "\n"
    "    return Index;\n"
    "}\n"
    "\n"
    "#endif\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableChm01IndexCrc32And.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableChm01IndexCrc32AndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexCrc32AndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexCrc32AndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexCrc32AndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexCrc32AndCSourceRawCString)
#endif
