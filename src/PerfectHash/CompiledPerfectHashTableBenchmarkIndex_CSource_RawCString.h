//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableBenchmarkIndexCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableBenchmarkIndex.c.\n"
    "//\n"
    "\n"
    "\n"
    "DECLARE_BENCHMARK_INDEX_CPH_ROUTINE()\n"
    "{\n"
    "    CPHKEY Key;\n"
    "    CPHINDEX Index;\n"
    "    ULONG Count;\n"
    "    ULONG Attempt = 1000;\n"
    "    const ULONG Iterations = 100000;\n"
    "    LARGE_INTEGER Start;\n"
    "    LARGE_INTEGER End;\n"
    "    LARGE_INTEGER Delta;\n"
    "    ULONG Best = (ULONG)-1;\n"
    "\n"
    "    Key = *KEYS;\n"
    "\n"
    "    if (Seconds) {\n"
    "\n"
    "        while (!CtrlCPressed) {\n"
    "\n"
    "            QueryPerformanceCounter(&Start);\n"
    "\n"
    "            for (Count = Iterations; Count != 0; Count--) {\n"
    "                Index = INDEX_ROUTINE(Key);\n"
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
    "                Index = INDEX_ROUTINE(Key);\n"
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
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableBenchmarkIndex.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableBenchmarkIndexCSourceRawCString = {
    sizeof(CompiledPerfectHashTableBenchmarkIndexCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableBenchmarkIndexCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableBenchmarkIndexCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableBenchmarkIndexCSourceRawCString)
#endif
