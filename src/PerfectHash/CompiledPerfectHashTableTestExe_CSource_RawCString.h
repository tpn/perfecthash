//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableTestExeCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableTestExe.c.\n"
    "//\n"
    "\n"
    "\n"
    "void\n"
    "__stdcall\n"
    "mainCRTStartup(\n"
    "    void\n"
    "    )\n"
    "{\n"
    "    ULONG NumberOfErrors;\n"
    "    BOOLEAN DebugBreakOnFailure = 0;\n"
    "\n"
    "    NumberOfErrors = TEST_CPH_ROUTINE(DebugBreakOnFailure);\n"
    "\n"
    "    ExitProcess(NumberOfErrors);\n"
    "}\n"
    "\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableTestExe.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableTestExeCSourceRawCString = {
    sizeof(CompiledPerfectHashTableTestExeCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableTestExeCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableTestExeCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableTestExeCSourceRawCString)
#endif
