//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexFnvAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexFnvAnd.c.\n"
    "//\n"
    "\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
    "    CPHDKEY A;\n"
    "    CPHDKEY B;\n"
    "    CPHINDEX Index;\n"
    "    CPHDKEY Vertex1;\n"
    "    CPHDKEY Vertex2;\n"
    "    CPHDKEY MaskedLow;\n"
    "    CPHDKEY MaskedHigh;\n"
    "    CPHDKEY DownsizedKey;\n"
    "    ULONG_BYTES Bytes;\n"
    "    ULONGLONG Combined;\n"
    "\n"
    "    DownsizedKey = DOWNSIZE_KEY(Key);\n"
    "    Bytes.AsULong = DownsizedKey;\n"
    "\n"
    "    A = SEED1 ^ 2166136261;\n"
    "    A = 16777619 * A ^ Bytes.Byte1;\n"
    "    A = 16777619 * A ^ Bytes.Byte2;\n"
    "    A = 16777619 * A ^ Bytes.Byte3;\n"
    "    A = 16777619 * A ^ Bytes.Byte4;\n"
    "\n"
    "    Vertex1 = A;\n"
    "\n"
    "    B = SEED2 ^ 2166136261;\n"
    "    B = 16777619 * B ^ Bytes.Byte1;\n"
    "    B = 16777619 * B ^ Bytes.Byte2;\n"
    "    B = 16777619 * B ^ Bytes.Byte3;\n"
    "    B = 16777619 * B ^ Bytes.Byte4;\n"
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
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableChm01IndexFnvAnd.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableChm01IndexFnvAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexFnvAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexFnvAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexFnvAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexFnvAndCSourceRawCString)
#endif
