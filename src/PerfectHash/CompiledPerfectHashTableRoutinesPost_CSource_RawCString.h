//
// Auto-generated.
//

DECLSPEC_ALIGN(16)
const CHAR CompiledPerfectHashTableRoutinesPostCSourceRawCStr[] =
    "\n"
    "//\n"
    "// Begin CompiledPerfectHashTableRoutinesPost.c.\n"
    "//\n"
    "\n"
    "#ifdef CPH_INLINE_ROUTINES\n"
    "\n"
    "#undef CPH_INLINE_ROUTINES\n"
    "\n"
    "#undef INDEX_ROUTINE\n"
    "#undef LOOKUP_ROUTINE\n"
    "#undef INSERT_ROUTINE\n"
    "#undef DELETE_ROUTINE\n"
    "\n"
    "#undef DECLARE_INDEX_ROUTINE\n"
    "#undef DECLARE_LOOKUP_ROUTINE\n"
    "#undef DECLARE_INSERT_ROUTINE\n"
    "#undef DECLARE_DELETE_ROUTINE\n"
    "\n"
    "#define INDEX_ROUTINE EXPAND_INDEX_ROUTINE(CPH_TABLENAME)\n"
    "#define LOOKUP_ROUTINE EXPAND_LOOKUP_ROUTINE(CPH_TABLENAME)\n"
    "#define INSERT_ROUTINE EXPAND_INSERT_ROUTINE(CPH_TABLENAME)\n"
    "#define DELETE_ROUTINE EXPAND_DELETE_ROUTINE(CPH_TABLENAME)\n"
    "\n"
    "#define DECLARE_INDEX_ROUTINE() EXPAND_INDEX_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "#define DECLARE_LOOKUP_ROUTINE() EXPAND_LOOKUP_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "#define DECLARE_INSERT_ROUTINE() EXPAND_INSERT_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "#define DECLARE_DELETE_ROUTINE() EXPAND_DELETE_ROUTINE_HEADER(CPH_TABLENAME)\n"
    "\n"
    "#endif\n"
    "\n"
    "//\n"
    "// End CompiledPerfectHashTableRoutinesPost.c.\n"
    "//\n"
;

const STRING CompiledPerfectHashTableRoutinesPostCSourceRawCString = {
    sizeof(CompiledPerfectHashTableRoutinesPostCSourceRawCStr) - sizeof(CHAR),
    sizeof(CompiledPerfectHashTableRoutinesPostCSourceRawCStr),
#ifdef _WIN64
    0,
#endif
    (PCHAR)&CompiledPerfectHashTableRoutinesPostCSourceRawCStr,
};

#ifndef RawCString
#define RawCString (&CompiledPerfectHashTableRoutinesPostCSourceRawCString)
#endif
