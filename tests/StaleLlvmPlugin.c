// A PerfectHashLLVM plugin from before the hash JIT existed. It exports the
// two index JIT entry points and neither hash JIT one, so the loader must
// reject it without calling either.

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
