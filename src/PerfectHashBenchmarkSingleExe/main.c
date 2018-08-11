/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    main.c

Abstract:

    This is the main file for the PerfectHashTableBenchmarkSingle executable.
    It implements mainCRTStartup(), which simply loads "PerfectHashTable.dll"
    and calls the PerfectHashTableBenchmarkSingleMain().

--*/

#include "stdafx.h"

typedef
ULONG
(BENCHMARK_MAIN)(
    VOID
    );
typedef BENCHMARK_MAIN *PBENCHMARK_MAIN;

DECLSPEC_NORETURN
VOID
WINAPI
mainCRTStartup()
{
    ULONG ExitCode;
    HMODULE Module;
    PROC Proc;
    PBENCHMARK_MAIN Main;


    Module = LoadLibraryA("PerfectHashTable.dll");
    Proc = GetProcAddress(Module, "PerfectHashTableBenchmarkSingleMain");
    if (!Proc) {
        __debugbreak();
        ExitCode = 1;
    } else {
        Main = (PBENCHMARK_MAIN)Proc;
        ExitCode = Main();
    }

    ExitProcess(ExitCode);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
