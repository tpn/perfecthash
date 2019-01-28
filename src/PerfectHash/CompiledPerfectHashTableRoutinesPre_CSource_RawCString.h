//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableRoutinesPreCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableRoutinesPre.c.\n"
    "//\n"
    "\n"
    "\n"
    "#ifdef CPH_INLINE_ROUTINES\n"
    "\n"
    "#define INDEX_ROUTINE EXPAND_INDEX_INLINE_ROUTINE(CPH_TABLENAME)\n"
    "#define DECLARE_INDEX_ROUTINE() EXPAND_INDEX_INLINE_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "\n"
    "#ifndef CPH_INDEX_ONLY\n"
    "#define LOOKUP_ROUTINE EXPAND_LOOKUP_INLINE_ROUTINE(CPH_TABLENAME)\n"
    "#define INSERT_ROUTINE EXPAND_INSERT_INLINE_ROUTINE(CPH_TABLENAME)\n"
    "#define DELETE_ROUTINE EXPAND_DELETE_INLINE_ROUTINE(CPH_TABLENAME)\n"
    "\n"
    "#define DECLARE_LOOKUP_ROUTINE() EXPAND_LOOKUP_INLINE_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "#define DECLARE_INSERT_ROUTINE() EXPAND_INSERT_INLINE_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "#define DECLARE_DELETE_ROUTINE() EXPAND_DELETE_INLINE_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "#endif\n"
    "\n"
    "#else\n"
    "\n"
    "#define INDEX_ROUTINE EXPAND_INDEX_ROUTINE(CPH_TABLENAME)\n"
    "#define DECLARE_INDEX_ROUTINE() EXPAND_INDEX_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "\n"
    "#ifndef CPH_INDEX_ONLY\n"
    "#define LOOKUP_ROUTINE EXPAND_LOOKUP_ROUTINE(CPH_TABLENAME)\n"
    "#define INSERT_ROUTINE EXPAND_INSERT_ROUTINE(CPH_TABLENAME)\n"
    "#define DELETE_ROUTINE EXPAND_DELETE_ROUTINE(CPH_TABLENAME)\n"
    "\n"
    "#define DECLARE_LOOKUP_ROUTINE() EXPAND_LOOKUP_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "#define DECLARE_INSERT_ROUTINE() EXPAND_INSERT_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "#define DECLARE_DELETE_ROUTINE() EXPAND_DELETE_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "#endif\n"
    "\n"
    "#endif\n"
    "\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableRoutinesPre.c.\n"
    "//\n"
    "\n"
;

const STRING CompiledPerfectHashTableRoutinesPreCSourceRawCString = {
    sizeof(CompiledPerfectHashTableRoutinesPreCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableRoutinesPreCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableRoutinesPreCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableRoutinesPreCSourceRawCString)
#endif
