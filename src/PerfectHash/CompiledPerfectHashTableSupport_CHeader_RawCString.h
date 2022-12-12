//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableSupportCHeaderRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableSupport.h.\n"
    "//\n"
    "\n"
    "\n"
    "\n"
    "extern\n"
    "void\n"
    "CphQueryPerformanceCounter(\n"
    "    _Out_ PLARGE_INTEGER Count\n"
    "    );\n"
    "\n"
    "extern volatile ULONG CtrlCPressed;\n"
    "\n"
    "#define FOR_EACH_KEY \\\n"
    "    for (Index = 0, Source = KEYS; Index < NUMBER_OF_KEYS; Index++)\n"
    "\n"
    "#ifdef _WIN32\n"
    "\n"
    "extern\n"
    "BOOLEAN\n"
    "QueryPerformanceCounter(\n"
    "    _Out_ PLARGE_INTEGER Count\n"
    "    );\n"
    "\n"
    "#define CPH_MAIN() \\\n"
    "VOID               \\\n"
    "__stdcall          \\\n"
    "mainCRTStartup(    \\\n"
    "    void           \\\n"
    "    )\n"
    "\n"
    "extern void ExitProcess(ULONG);\n"
    "#define CPH_EXIT(Code) ExitProcess(Code)\n"
    "\n"
    "#elif defined(__linux__) || defined(__APPLE__)\n"
    "\n"
    "#include <stdio.h>\n"
    "\n"
    "#define CPH_MAIN() \\\n"
    "int                \\\n"
    "main(              \\\n"
    "    int argc,      \\\n"
    "    char **argv    \\\n"
    "    )\n"
    "\n"
    "#define CPH_EXIT(Code)    \\\n"
    "    printf(\"%d\\n\", Code); \\\n"
    "    return Code\n"
    "\n"
    "#endif\n"
    "\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableSupport.h.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableSupportCHeaderRawCString = {
    sizeof(CompiledPerfectHashTableSupportCHeaderRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableSupportCHeaderRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableSupportCHeaderRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableSupportCHeaderRawCString)
#endif
