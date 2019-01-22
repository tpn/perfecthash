//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableTypesPreCHeaderRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableTypesPre.h.\n"
    "//\n"
    "\n"
    "\n"
    "//\n"
    "// Define start/end markers for IACA.\n"
    "//\n"
    "\n"
    "#define IACA_VC_START() __writegsbyte(111, 111)\n"
    "#define IACA_VC_END()   __writegsbyte(222, 222)\n"
    "\n"
    "//\n"
    "// Define basic NT types and macros used by this header file.\n"
    "//\n"
    "\n"
    "#define CPHCALLTYPE __stdcall\n"
    "#define FORCEINLINE __forceinline\n"
    "\n"
    "typedef char BOOLEAN;\n"
    "typedef unsigned char BYTE;\n"
    "typedef BYTE *PBYTE;\n"
    "typedef short SHORT;\n"
    "typedef unsigned short USHORT;\n"
    "typedef long LONG;\n"
    "typedef long long LONGLONG;\n"
    "typedef unsigned long ULONG;\n"
    "typedef unsigned long *PULONG;\n"
    "typedef unsigned long long ULONGLONG;\n"
    "typedef void *PVOID;\n"
    "\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableTypesPre.h.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableTypesPreCHeaderRawCString = {
    sizeof(CompiledPerfectHashTableTypesPreCHeaderRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableTypesPreCHeaderRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableTypesPreCHeaderRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableTypesPreCHeaderRawCString)
#endif
