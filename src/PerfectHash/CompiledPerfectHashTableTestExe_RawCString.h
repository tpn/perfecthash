//
// Auto-generated from ../CompiledPerfectHashTable/CompiledPerfectHashTableTestExe.c.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableTestExeRawCStr[] =
    "extern void ExitProcess(ULONG);\n"
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
    "    NumberOfErrors = TestCphTable(DebugBreakOnFailure);\n"
    "\n"
    "    ExitProcess(NumberOfErrors);\n"
    "}\n"
    "\n"
    "// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :\n"
;

const STRING CompiledPerfectHashTableTestExeRawString = {
    sizeof(CompiledPerfectHashTableTestExeRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableTestExeRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableTestExeRawCStr,
};
