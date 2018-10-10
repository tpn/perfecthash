//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableTestExeCSourceRawCStr[] =
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
