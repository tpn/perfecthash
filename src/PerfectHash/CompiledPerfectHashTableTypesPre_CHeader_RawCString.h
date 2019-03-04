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
    "#ifndef BASETYPES\n"
    "\n"
    "//\n"
    "// Define basic NT types.\n"
    "//\n"
    "\n"
    "typedef int BOOL;\n"
    "typedef char BOOLEAN;\n"
    "typedef unsigned char BYTE;\n"
    "typedef BYTE *PBYTE;\n"
    "typedef short SHORT;\n"
    "typedef short *PSHORT;\n"
    "typedef unsigned short USHORT;\n"
    "typedef unsigned short *PUSHORT;\n"
    "typedef long long LONGLONG;\n"
    "typedef long long *PLONGLONG;\n"
    "typedef unsigned long long ULONGLONG;\n"
    "typedef unsigned long long *PULONGLONG;\n"
    "typedef void *PVOID;\n"
    "\n"
    "#define VOID void\n"
    "\n"
    "#ifdef _WIN32\n"
    "typedef long LONG;\n"
    "typedef long *PLONG;\n"
    "typedef unsigned long ULONG;\n"
    "typedef unsigned long *PULONG;\n"
    "#elif defined(__linux__) || defined(__APPLE__)\n"
    "typedef int LONG;\n"
    "typedef int *PLONG;\n"
    "typedef unsigned int ULONG;\n"
    "typedef unsigned int *PULONG;\n"
    "#endif\n"
    "\n"
    "#define TRUE 1\n"
    "#define FALSE 0\n"
    "\n"
    "\n"
    "#define BASETYPES\n"
    "\n"
    "#endif\n"
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
