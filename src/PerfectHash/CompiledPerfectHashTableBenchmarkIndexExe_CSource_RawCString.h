//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableBenchmarkIndexExeCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableBenchmarkIndexExe.c.\n"
    "//\n"
    "\n"
    "\n"
    "CPH_MAIN()\n"
    "{\n"
    "    ULONG Cycles;\n"
    "    ULONG Seconds = 0;\n"
    "\n"
    "    Cycles = BENCHMARK_INDEX_CPH_ROUTINE(Seconds);\n"
    "\n"
    "    CPH_EXIT(Cycles);\n"
    "}\n"
    "\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableBenchmarkIndexExe.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableBenchmarkIndexExeCSourceRawCString = {
    sizeof(CompiledPerfectHashTableBenchmarkIndexExeCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableBenchmarkIndexExeCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableBenchmarkIndexExeCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableBenchmarkIndexExeCSourceRawCString)
#endif
