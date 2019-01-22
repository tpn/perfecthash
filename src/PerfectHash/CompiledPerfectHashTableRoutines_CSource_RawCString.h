//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableRoutinesCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableRoutines.c.\n"
    "//\n"
    "\n"
    "\n"
    "DECLARE_LOOKUP_ROUTINE()\n"
    "{\n"
    "    CPHINDEX Index;\n"
    "\n"
    "    Index = INDEX_ROUTINE(Key);\n"
    "    return TABLE_VALUES[Index];\n"
    "}\n"
    "\n"
    "DECLARE_INSERT_ROUTINE()\n"
    "{\n"
    "    CPHINDEX Index;\n"
    "    CPHVALUE Previous;\n"
    "\n"
    "    Index = INDEX_ROUTINE(Key);\n"
    "    Previous = TABLE_VALUES[Index];\n"
    "    TABLE_VALUES[Index] = Value;\n"
    "    return Previous;\n"
    "}\n"
    "\n"
    "DECLARE_DELETE_ROUTINE()\n"
    "{\n"
    "    CPHINDEX Index;\n"
    "    CPHVALUE Previous;\n"
    "\n"
    "    Index = INDEX_ROUTINE(Key);\n"
    "    Previous = TABLE_VALUES[Index];\n"
    "    TABLE_VALUES[Index] = 0;\n"
    "    return Previous;\n"
    "}\n"
    "\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableRoutines.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableRoutinesCSourceRawCString = {
    sizeof(CompiledPerfectHashTableRoutinesCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableRoutinesCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableRoutinesCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableRoutinesCSourceRawCString)
#endif
