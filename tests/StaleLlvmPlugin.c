// Model a plugin with the current index ABI but no hash JIT, or an older
// plugin with all four unversioned entry points. Both must be rejected.

#ifdef _WIN32
#define PH_PLUGIN_EXPORT __declspec(dllexport)
#else
#define PH_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

PH_PLUGIN_EXPORT
int
PerfectHashTableCompileJit(
    void *Table,
    void *CompileFlags
    )
{
    (void)Table;
    (void)CompileFlags;
    return -1;
}

PH_PLUGIN_EXPORT
int
PerfectHashTableJitRundown(
    void *Table
    )
{
    (void)Table;
    return -1;
}

#ifdef PH_TEST_UNVERSIONED_LLVM
PH_PLUGIN_EXPORT
int
PerfectHashTableCompileHashJit(void *Table, void *CompileFlags)
{
    return PerfectHashTableCompileJit(Table, CompileFlags);
}

PH_PLUGIN_EXPORT
void
PerfectHashTableHashJitRundown(void *Table)
{
    (void)Table;
}
#endif
