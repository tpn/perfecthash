//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableBenchmarkFullExeCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableBenchmarkFullExe.c.\n"
    "//\n"
    "\n"
    "\n"
    "CPH_MAIN()\n"
    "{\n"
    "    ULONG Cycles;\n"
    "    ULONG Seconds = 0;\n"
    "\n"
    "    Cycles = BENCHMARK_FULL_CPH_ROUTINE(Seconds);\n"
    "\n"
    "    CPH_EXIT(Cycles);\n"
    "}\n"
    "\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableBenchmarkFullExe.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableBenchmarkFullExeCSourceRawCString = {
    sizeof(CompiledPerfectHashTableBenchmarkFullExeCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableBenchmarkFullExeCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableBenchmarkFullExeCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableBenchmarkFullExeCSourceRawCString)
#endif
