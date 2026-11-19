#include <PerfectHash/PerfectHashOnlineJit.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static int IsPluginLoaded(const char *path)
{
#ifdef _WIN32
    return GetModuleHandleA(path) != NULL;
#else
    void *plugin = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
    if (!plugin) {
        return 0;
    }
    dlclose(plugin);
    return 1;
#endif
}

int main(int argc, char **argv)
{
    const uint32_t keys[16] = {1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
    const PH_ONLINE_JIT_BACKEND backends[] = {PhOnlineJitBackendRawDogJit, PhOnlineJitBackendLlvmJit};
    PH_ONLINE_JIT_CONTEXT *context = NULL;
    PH_ONLINE_JIT_TABLE *table = NULL;
    uint32_t native[16], batch[16];
    int result = 1;

#ifdef _WIN32
    DWORD hostTls = TlsAlloc();
    if (hostTls == TLS_OUT_OF_INDEXES) {
        return 1;
    }
    if (!TlsSetValue(hostTls, argv)) {
        TlsFree(hostTls);
        return 1;
    }
#endif
    if (argc != 2 || PhOnlineJitOpen(&context) < 0 ||
        PhOnlineJitCreateTable32(context, PhOnlineJitHashMultiplyShiftR, keys, 16, &table) < 0) {
        fputs("Could not create the installed Core test table\n", stderr);
        goto done;
    }
    for (unsigned i = 0; i < 16; ++i) {
        if (PhOnlineJitIndex32(table, keys[i], &native[i]) < 0) {
            fputs("Native lookup failed\n", stderr);
            goto done;
        }
    }
    for (unsigned i = 0; i < sizeof(backends) / sizeof(backends[0]); ++i) {
        PH_ONLINE_JIT_BACKEND active = PhOnlineJitBackendAuto;
        uint32_t width = 0;
        int32_t status = PhOnlineJitCompileTableEx(
            context, table, backends[i], 16, PhOnlineJitMaxIsaNeon,
            PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH, &active, &width);
        if (status < 0 || active != backends[i] || width != 16) {
            fprintf(stderr, "Backend %d returned %#x, active backend %d, width %u\n",
                    (int)backends[i], (unsigned)status, (int)active, (unsigned)width);
            goto done;
        }
        if (PhOnlineJitIndex32x16(table, keys, batch) < 0 || memcmp(batch, native, sizeof(native))) {
            fprintf(stderr, "Backend %d x16 lookup failed or disagreed with native\n", (int)active);
            goto done;
        }
    }
    if (!IsPluginLoaded(argv[1])) {
        fprintf(stderr, "The expected installed plugin was not loaded: %s\n", argv[1]);
        goto done;
    }
    result = 0;
done:
    PhOnlineJitReleaseTable(table);
    PhOnlineJitClose(context);
#ifdef _WIN32
    if (TlsGetValue(hostTls) != argv) {
        fputs("Client thread-local state changed during JIT lookup\n", stderr);
        result = 1;
    }
    TlsFree(hostTls);
#endif
    return result;
}
