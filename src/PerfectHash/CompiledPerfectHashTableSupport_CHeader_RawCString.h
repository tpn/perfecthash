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
    "//\n"
    "// Disable the anonymous union/struct warning.\n"
    "//\n"
    "\n"
    "#pragma warning(push)\n"
    "#pragma warning(disable: 4201 4094)\n"
    "\n"
    "typedef union _LARGE_INTEGER {\n"
    "    struct {\n"
    "        ULONG LowPart;\n"
    "        LONG HighPart;\n"
    "    };\n"
    "    LONGLONG QuadPart;\n"
    "} LARGE_INTEGER;\n"
    "typedef LARGE_INTEGER *PLARGE_INTEGER;\n"
    "\n"
    "typedef union _ULARGE_INTEGER {\n"
    "    struct {\n"
    "        ULONG LowPart;\n"
    "        ULONG HighPart;\n"
    "    };\n"
    "    ULONGLONG QuadPart;\n"
    "} ULARGE_INTEGER;\n"
    "typedef ULARGE_INTEGER *PULARGE_INTEGER;\n"
    "\n"
    "#pragma warning(pop)\n"
    "\n"
    "extern\n"
    "BOOLEAN\n"
    "QueryPerformanceCounter(\n"
    "    _Out_ PLARGE_INTEGER Count\n"
    "    );\n"
    "\n"
    "extern volatile ULONG CtrlCPressed;\n"
    "\n"
    "extern void ExitProcess(ULONG);\n"
    "\n"
    "#define FOR_EACH_KEY \\\n"
    "    for (Index = 0, Source = KEYS; Index < NUMBER_OF_KEYS; Index++)\n"
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
