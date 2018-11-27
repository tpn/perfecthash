//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexAddSubXorAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexAddSubXorAnd.c.\n"
    "//\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
    "    ULONG Vertex1;\n"
    "    ULONG Vertex2;\n"
    "    ULARGE_INTEGER Result;\n"
    "\n"
    "    Vertex1 = (Key + Seed1) ^ Seed3;\n"
    "    Vertex2 = (Key - Seed2) ^ Seed4;\n"
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
    "// End CompiledPerfectHashTableChm01IndexAddSubXorAnd.c.\n"
    "//\n"
;

const STRING CompiledPerfectHashTableChm01IndexAddSubXorAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexAddSubXorAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexAddSubXorAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexAddSubXorAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexAddSubXorAndCSourceRawCString)
#endif
