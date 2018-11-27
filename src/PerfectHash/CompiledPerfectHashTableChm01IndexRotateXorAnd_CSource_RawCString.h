//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexRotateXorAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexRotateXorAnd.c.\n"
    "//\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
    "    ULONG A;\n"
    "    ULONG B;\n"
    "    ULONG C;\n"
    "    ULONG D;\n"
    "    ULONG Vertex1;\n"
    "    ULONG Vertex2;\n"
    "    ULARGE_INTEGER Result;\n"
    "\n"
    "    A = _rotl(Key ^ SEED1, 15);\n"
    "    B = _rotl(Key + SEED2, 7);\n"
    "    C = _rotr(Key - SEED3, 11);\n"
    "    D = _rotr(Key ^ SEED4, 20);\n"
    "\n"
    "    Vertex1 = A ^ C;\n"
    "    Vertex2 = B ^ D;\n"
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
    "// End CompiledPerfectHashTableChm01IndexRotateXorAnd.c.\n"
    "//\n"
;

const STRING CompiledPerfectHashTableChm01IndexRotateXorAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexRotateXorAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexRotateXorAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexRotateXorAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexRotateXorAndCSourceRawCString)
#endif
