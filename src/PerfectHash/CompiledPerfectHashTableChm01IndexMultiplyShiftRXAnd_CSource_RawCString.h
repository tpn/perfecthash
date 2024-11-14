//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableChm01IndexMultiplyShiftRXAndCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableChm01IndexMultiplyShiftRXAnd.c.\n"
    "//\n"
    "\n"
    "\n"
    "DECLARE_INDEX_ROUTINE()\n"
    "{\n"
    "    CPHINDEX Index;\n"
    "    CPHDKEY Vertex1;\n"
    "    CPHDKEY Vertex2;\n"
    "    CPHDKEY DownsizedKey;\n"
    "\n"
    "    //IACA_VC_START();\n"
    "\n"
    "    DownsizedKey = DOWNSIZE_KEY(Key);\n"
    "\n"
    "    Vertex1 = DownsizedKey * SEED1;\n"
    "    Vertex1 = Vertex1 >> SEED3_BYTE1;\n"
    "    Vertex1 = TABLE_DATA[Vertex1];\n"
    "\n"
    "    Vertex2 = DownsizedKey * SEED2;\n"
    "    Vertex2 = Vertex2 >> SEED3_BYTE1;\n"
    "    Vertex2 = TABLE_DATA[Vertex2];\n"
    "\n"
    "    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);\n"
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
    "    CPHDKEY DownsizedKey;\n"
    "\n"
    "    IACA_VC_START();\n"
    "\n"
    "    DownsizedKey = DOWNSIZE_KEY(Key);\n"
    "\n"
    "    Vertex1 = DownsizedKey * SEED1;\n"
    "    Vertex1 = Vertex1 >> SEED3_BYTE1;\n"
    "    Vertex1 = TABLE_DATA[Vertex1];\n"
    "\n"
    "    Vertex2 = DownsizedKey * SEED2;\n"
    "    Vertex2 = Vertex2 >> SEED3_BYTE1;\n"
    "    Vertex2 = TABLE_DATA[Vertex2];\n"
    "\n"
    "    Index = (CPHINDEX)((Vertex1 + Vertex2) & INDEX_MASK);\n"
    "\n"
    "    IACA_VC_END();\n"
    "\n"
    "    return Index;\n"
    "}\n"
    "\n"
    "#endif\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableChm01IndexMultiplyShiftRXAnd.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableChm01IndexMultiplyShiftRXAndCSourceRawCString = {
    sizeof(CompiledPerfectHashTableChm01IndexMultiplyShiftRXAndCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableChm01IndexMultiplyShiftRXAndCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableChm01IndexMultiplyShiftRXAndCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableChm01IndexMultiplyShiftRXAndCSourceRawCString)
#endif
