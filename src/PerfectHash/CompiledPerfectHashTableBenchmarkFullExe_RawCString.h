//
// Auto-generated from ../CompiledPerfectHashTable/CompiledPerfectHashTableBenchmarkFullExe.c.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableBenchmarkFullExeRawCStr[] =
    "#pragma optimize(\"\", off)\n"
    "\n"
    "extern void ExitProcess(ULONG);\n"
    "\n"
    "volatile ULONG CtrlCPressed = 0;\n"
    "\n"
    "void\n"
    "__stdcall\n"
    "mainCRTStartup(\n"
    "    void\n"
    "    )\n"
    "{\n"
    "    ULONG Cycles;\n"
    "    ULONG Seconds = 0;\n"
    "\n"
    "    Cycles = BenchmarkFullCphTable(Seconds);\n"
    "\n"
    "    ExitProcess(Cycles);\n"
    "}\n"
    "\n"
    "// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :\n"
;

const STRING CompiledPerfectHashTableBenchmarkFullExeRawString = {
    sizeof(CompiledPerfectHashTableBenchmarkFullExeRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableBenchmarkFullExeRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableBenchmarkFullExeRawCStr,
};
