//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexDjbXorAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexDjbXorAnd.c.\n"
    "//\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
    "    ULONG A;\n"
    "    ULONG B;\n"
    "    ULONG Index;\n"
    "    ULONG Vertex1;\n"
    "    ULONG Vertex2;\n"
    "    ULONG MaskedLow;\n"
    "    ULONG MaskedHigh;\n"
    "    ULONG_BYTES Bytes;\n"
    "    ULONGLONG Combined;\n"
    "\n"
    "    Bytes.AsULong = Key;\n"
    "\n"
    "    A = SEED1;\n"
    "    A = 33 * A ^ Bytes.Byte1;\n"
    "    A = 33 * A ^ Bytes.Byte2;\n"
    "    A = 33 * A ^ Bytes.Byte3;\n"
    "    A = 33 * A ^ Bytes.Byte4;\n"
    "\n"
    "    Vertex1 = A;\n"
    "\n"
    "    B = SEED2;\n"
    "    B = 33 * B ^ Bytes.Byte1;\n"
    "    B = 33 * B ^ Bytes.Byte2;\n"
    "    B = 33 * B ^ Bytes.Byte3;\n"
    "    B = 33 * B ^ Bytes.Byte4;\n"
    "\n"
    "    Vertex2 = B;\n"
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
    "// End CompiledPerfectHashTableChm01IndexDjbXorAnd.c.\n"
    "//\n"
;

const STRING CompiledPerfectHashTableChm01IndexDjbXorAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexDjbXorAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexDjbXorAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexDjbXorAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexDjbXorAndCSourceRawCString)
#endif
