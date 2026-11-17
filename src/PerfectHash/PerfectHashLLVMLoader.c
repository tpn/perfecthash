/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashLLVMLoader.c

Abstract:

    Load the optional LLVM index and solver hash backend on first use.

--*/

#include "stdafx.h"

#ifdef PH_USE_LLVMLIB

extern LOAD_SYMBOLS LoadSymbols;
static PERFECT_HASH_LLVM_FUNCTIONS Functions;
static const PERFECT_HASH_LLVM_FUNCTIONS *LoadedFunctions;

#ifdef PH_WINDOWS

// Maximum length of a Win32 extended-length path including its terminator.
#define PH_LLVM_PATH_CAPACITY 32768
#define PH_LLVM_LOAD_FLAGS \
    (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)

static const WCHAR PluginName[] = L"PerfectHashLLVM.dll";

static HMODULE
LoadLlvmCRuntimeFromPath(VOID)
{
    static const WCHAR RuntimeName[] = L"LLVM-C.dll";
    DWORD Size;
    DWORD Length;
    PWSTR Path;
    PWSTR Directory;
    PWSTR Next;
    PWSTR Candidate;
    HMODULE Runtime = NULL;

    Size = GetEnvironmentVariableW(L"PATH", NULL, 0);
    if (!Size) {
        return NULL;
    }
    Path = malloc((SIZE_T)Size * sizeof(*Path));
    Candidate = malloc(PH_LLVM_PATH_CAPACITY * sizeof(*Candidate));
    if (!Path || !Candidate) {
        goto End;
    }
    Length = GetEnvironmentVariableW(L"PATH", Path, Size);
    if (!Length || Length >= Size) {
        goto End;
    }

    // Only explicitly configured absolute directories participate. Keep
    // dependency resolution scoped to each load, including LLVM's own DLLs.
    for (Directory = wcstok_s(Path, L";", &Next);
         Directory;
         Directory = wcstok_s(NULL, L";", &Next)) {
        SIZE_T Count = wcslen(Directory);
        if (Count >= 2 && Directory[0] == L'"' && Directory[Count-1] == L'"') {
            Directory[--Count] = L'\0';
            Directory++;
            Count--;
        }
        if (!((Count >= 3 && Directory[1] == L':' &&
               (Directory[2] == L'\\' || Directory[2] == L'/')) ||
              (Count >= 2 && Directory[0] == L'\\' && Directory[1] == L'\\'))) {
            continue;
        }
        if (Count + 1 + ARRAYSIZE(RuntimeName) > PH_LLVM_PATH_CAPACITY) {
            continue;
        }
        memcpy(Candidate, Directory, Count * sizeof(WCHAR));
        if (Directory[Count-1] != L'\\' && Directory[Count-1] != L'/') {
            Candidate[Count++] = L'\\';
        }
        memcpy(Candidate + Count, RuntimeName, sizeof(RuntimeName));
        Runtime = LoadLibraryExW(Candidate, NULL, PH_LLVM_LOAD_FLAGS);
        if (Runtime) {
            break;
        }
    }
End:
    free(Candidate);
    free(Path);
    return Runtime;
}

static HMODULE
LoadPerfectHashLLVMModule(VOID)
{
    HMODULE Host;
    HMODULE Module = NULL;
    HMODULE Runtime;
    DWORD Length;
    PWSTR Path;
    PWSTR Name;

    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            PluginName, &Host)) {
        return NULL;
    }
    Path = malloc(PH_LLVM_PATH_CAPACITY * sizeof(*Path));
    if (!Path) {
        return NULL;
    }
    Length = GetModuleFileNameW(Host, Path, PH_LLVM_PATH_CAPACITY);
    if (!Length || Length >= PH_LLVM_PATH_CAPACITY) {
        goto End;
    }
    Name = Path + Length;
    while (Name > Path && Name[-1] != L'\\' && Name[-1] != L'/') {
        Name--;
    }
    if (Name == Path || (Name - Path) + ARRAYSIZE(PluginName) > PH_LLVM_PATH_CAPACITY) {
        goto End;
    }
    memcpy(Name, PluginName, sizeof(PluginName));
    Module = LoadLibraryExW(Path, NULL, PH_LLVM_LOAD_FLAGS);
    if (!Module && GetLastError() == ERROR_MOD_NOT_FOUND &&
        GetFileAttributesW(Path) != INVALID_FILE_ATTRIBUTES) {
        // The official LLVM archive ships LLVM-C.dll in its own bin directory.
        // Hold it by absolute path while the plugin resolves its import.
        Runtime = LoadLlvmCRuntimeFromPath();
        if (Runtime) {
            Module = LoadLibraryExW(Path, NULL, PH_LLVM_LOAD_FLAGS);
            FreeLibrary(Runtime);
        }
    }
End:
    free(Path);
    return Module;
}
#endif

static VOID
LoadPerfectHashLLVMFunctions(VOID)
{
    CONST PCSZ Names[] = {
        "PerfectHashTableCompileJit",
        "PerfectHashTableJitRundown",
        "PerfectHashTableCompileHashJit",
        "PerfectHashTableHashJitRundown",
    };
    ULONG FailedBits = 0;
    RTL_BITMAP FailedBitmap;
    ULONG Resolved = 0;
    HMODULE Module;

    FailedBitmap.SizeOfBitMap = ARRAYSIZE(Names) + 1;
    FailedBitmap.Buffer = &FailedBits;
#if defined(PH_WINDOWS)
    Module = LoadPerfectHashLLVMModule();
#elif defined(PH_MAC)
    Module = LoadLibraryA("@rpath/libPerfectHashLLVM.dylib");
#else
    Module = LoadLibraryA("libPerfectHashLLVM.so");
#endif
    if (!IsValidHandle(Module)) {
        return;
    }
    if (!LoadSymbols(Names, ARRAYSIZE(Names), (PULONG_PTR)&Functions,
                     sizeof(Functions) / sizeof(ULONG_PTR), Module,
                     &FailedBitmap, &Resolved) || Resolved != ARRAYSIZE(Names)) {
        FreeLibrary(Module);
        return;
    }

    // Keep the module reference for the lifetime of this dispatch table.
    LoadedFunctions = &Functions;
}

#ifdef PH_WINDOWS
static BOOL CALLBACK
InitPerfectHashLLVM(PINIT_ONCE Once, PVOID Parameter, PVOID *Context)
{
    UNREFERENCED_PARAMETER(Once);
    UNREFERENCED_PARAMETER(Parameter);
    UNREFERENCED_PARAMETER(Context);
    LoadPerfectHashLLVMFunctions();
    return TRUE;
}
#endif

const PERFECT_HASH_LLVM_FUNCTIONS *
GetPerfectHashLLVMFunctions(VOID)
{
#ifdef PH_WINDOWS
    static INIT_ONCE Once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&Once, InitPerfectHashLLVM, NULL, NULL);
#else
    static pthread_once_t Once = PTHREAD_ONCE_INIT;
    pthread_once(&Once, LoadPerfectHashLLVMFunctions);
#endif
    return LoadedFunctions;
}

#endif // PH_USE_LLVMLIB
