//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexMultiply644ShiftRAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexMultiply644ShiftRAnd.c.\n"
    "//\n"
    "\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
    "    CPHINDEX Index;\n"
    "    CPHDKEY64 Vertex1;\n"
    "    CPHDKEY64 Vertex2;\n"
    "    CPHDKEY Final1;\n"
    "    CPHDKEY Final2;\n"
    "    CPHDKEY MaskedLow;\n"
    "    CPHDKEY MaskedHigh;\n"
    "    CPHDKEY64 DownsizedKey;\n"
    "\n"
    "    //IACA_VC_START();\n"
    "\n"
    "    DownsizedKey = Key;\n"
    "\n"
    "    Vertex1 = DownsizedKey * SEED12;\n"
    "    Vertex1 = Vertex1 >> SEED3_BYTE1;\n"
    "\n"
    "    Vertex2 = DownsizedKey * SEED45;\n"
    "    Vertex2 = Vertex2 >> SEED3_BYTE2;\n"
    "\n"
    "    MaskedLow = (CPHDKEY)(Vertex1 & HASH_MASK);\n"
    "    MaskedHigh = (CPHDKEY)(Vertex2 & HASH_MASK);\n"
    "\n"
    "    Final1 = TABLE_DATA[MaskedLow];\n"
    "    Final2 = TABLE_DATA[MaskedHigh];\n"
    "\n"
    "    Index = (CPHINDEX)((Final1 + Final2) & INDEX_MASK);\n"
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
    "    CPHDKEY64 Vertex1;\n"
    "    CPHDKEY64 Vertex2;\n"
    "    CPHDKEY Final1;\n"
    "    CPHDKEY Final2;\n"
    "    CPHDKEY MaskedLow;\n"
    "    CPHDKEY MaskedHigh;\n"
    "    CPHDKEY64 DownsizedKey;\n"
    "\n"
    "    IACA_VC_START();\n"
    "\n"
    "    DownsizedKey = Key;\n"
    "\n"
    "    Vertex1 = DownsizedKey * SEED12;\n"
    "    Vertex1 = Vertex1 >> SEED3_BYTE1;\n"
    "\n"
    "    Vertex2 = DownsizedKey * SEED45;\n"
    "    Vertex2 = Vertex2 >> SEED3_BYTE2;\n"
    "\n"
    "    MaskedLow = (CPHDKEY)(Vertex1 & HASH_MASK);\n"
    "    MaskedHigh = (CPHDKEY)(Vertex2 & HASH_MASK);\n"
    "\n"
    "    Final1 = TABLE_DATA[MaskedLow];\n"
    "    Final2 = TABLE_DATA[MaskedHigh];\n"
    "\n"
    "    Index = (CPHINDEX)((Final1 + Final2) & INDEX_MASK);\n"
    "\n"
    "    IACA_VC_END();\n"
    "\n"
    "    return Index;\n"
    "}\n"
    "\n"
    "#endif\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableChm01IndexMultiply644ShiftRAnd.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableChm01IndexMultiply644ShiftRAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexMultiply644ShiftRAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexMultiply644ShiftRAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexMultiply644ShiftRAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexMultiply644ShiftRAndCSourceRawCString)
#endif
