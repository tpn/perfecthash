//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexXorAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexXorAnd.c.\n"
    "//\n"
    "\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
    "    CPHINDEX Index;\n"
    "    ULONG Vertex1;\n"
    "    ULONG Vertex2;\n"
    "    ULONG_INTEGER Long1;\n"
    "    ULONG_INTEGER Long2;\n"
    "    CPHDKEY DownsizedKey;\n"
    "\n"
    "    DownsizedKey = DOWNSIZE_KEY(Key);\n"
    "    Long1.LongPart = DownsizedKey ^ Seed1;\n"
    "    Long2.LongPart = _rotl(DownsizedKey, 15) ^ Seed2;\n"
    "\n"
    "    Long1.LowPart ^= Long1.HighPart;\n"
    "    Long1.HighPart = 0;\n"
    "\n"
    "    Long2.LowPart ^= Long2.HighPart;\n"
    "    Long2.HighPart = 0;\n"
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
    "// End CompiledPerfectHashTableChm01IndexXorAnd.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableChm01IndexXorAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexXorAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexXorAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexXorAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexXorAndCSourceRawCString)
#endif
