//
// Auto-generated from ../CompiledPerfectHashTable/CompiledPerfectHashTableBenchmarkIndex.c.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableBenchmarkIndexRawCStr[] =
    "extern\n"
    "BOOLEAN\n"
    "QueryPerformanceCounter(\n"
    "    _Out_ PLARGE_INTEGER Count\n"
    "    );\n"
    "\n"
    "extern volatile ULONG CtrlCPressed;\n"
    "\n"
    "DECLARE_BENCHMARK_INDEX_COMPILED_PERFECT_HASH_TABLE_ROUTINE_HEADER()\n"
    "{\n"
    "    ULONG Key;\n"
    "    ULONG Index;\n"
    "    ULONG Count;\n"
    "    ULONG Attempt = 1000;\n"
    "    const ULONG Iterations = 100000;\n"
    "    LARGE_INTEGER Start;\n"
    "    LARGE_INTEGER End;\n"
    "    LARGE_INTEGER Delta;\n"
    "    ULONG Best = (ULONG)-1;\n"
    "\n"
    "    Key = *CphTableKeys;\n"
    "\n"
    "    if (Seconds) {\n"
    "\n"
    "        while (!CtrlCPressed) {\n"
    "\n"
    "            QueryPerformanceCounter(&Start);\n"
    "\n"
    "            for (Count = Iterations; Count != 0; Count--) {\n"
    "                Index = CphTableIndex(Key);\n"
    "            }\n"
    "\n"
    "            QueryPerformanceCounter(&End);\n"
    "\n"
    "            Delta.QuadPart = End.QuadPart - Start.QuadPart;\n"
    "\n"
    "            if (Delta.LowPart < Best) {\n"
    "                Best = Delta.LowPart;\n"
    "            }\n"
    "\n"
    "        }\n"
    "\n"
    "    } else {\n"
    "\n"
    "        while (Attempt--) {\n"
    "\n"
    "            QueryPerformanceCounter(&Start);\n"
    "\n"
    "            for (Count = Iterations; Count != 0; Count--) {\n"
    "                Index = CphTableIndex(Key);\n"
    "            }\n"
    "\n"
    "            QueryPerformanceCounter(&End);\n"
    "\n"
    "            Delta.QuadPart = End.QuadPart - Start.QuadPart;\n"
    "\n"
    "            if (Delta.LowPart < Best) {\n"
    "                Best = Delta.LowPart;\n"
    "            }\n"
    "\n"
    "        }\n"
    "\n"
    "    }\n"
    "\n"
    "    return Best;\n"
    "}\n"
    "\n"
    "// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :\n"
;

const STRING CompiledPerfectHashTableBenchmarkIndexRawString = {
    sizeof(CompiledPerfectHashTableBenchmarkIndexRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableBenchmarkIndexRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableBenchmarkIndexRawCStr,
};
