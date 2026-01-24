/*++

Copyright (c) 2018-2025 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashServerExe.c

Abstract:

    This module implements the main entry point for the perfect hash server.
    It loads the perfect hash library, creates a server instance, applies any
    command line options, then starts and waits on the server.

--*/

#include "stdafx.h"
#include <stdio.h>
#include <string.h>
#include <wchar.h>

#ifdef PH_WINDOWS

typedef DWORD MINIDUMP_TYPE;

#define MiniDumpWithDataSegs ((MINIDUMP_TYPE)0x00000001)
#define MiniDumpIgnoreInaccessibleMemory ((MINIDUMP_TYPE)0x00020000)

#pragma warning(push)
#pragma warning(disable: 4820)

typedef struct _MINIDUMP_EXCEPTION_INFORMATION {
    DWORD ThreadId;
    PEXCEPTION_POINTERS ExceptionPointers;
    BOOL ClientPointers;
} MINIDUMP_EXCEPTION_INFORMATION, *PMINIDUMP_EXCEPTION_INFORMATION;

#pragma warning(pop)

typedef PVOID PMINIDUMP_USER_STREAM_INFORMATION;
typedef PVOID PMINIDUMP_CALLBACK_INFORMATION;

#endif

typedef struct _PERFECT_HASH_SERVER_CLI_OPTIONS {
    UNICODE_STRING Endpoint;
    PERFECT_HASH_NUMA_NODE_MASK NumaNodeMask;
    ULONG IocpConcurrency;
    ULONG MaxThreads;
    BOOLEAN EndpointPresent;
    BOOLEAN IocpConcurrencyPresent;
    BOOLEAN MaxThreadsPresent;
    BOOLEAN NumaNodeMaskPresent;
    BOOLEAN LocalOnly;
    BOOLEAN LocalOnlyPresent;
    BOOLEAN Verbose;
    BOOLEAN VerbosePresent;
    BOOLEAN NoFileIo;
    BOOLEAN NoFileIoPresent;
    BOOLEAN IocpBufferGuardPages;
    BOOLEAN IocpBufferGuardPagesPresent;
    UCHAR Padding1[4];
} PERFECT_HASH_SERVER_CLI_OPTIONS;
typedef PERFECT_HASH_SERVER_CLI_OPTIONS *PPERFECT_HASH_SERVER_CLI_OPTIONS;

static PPERFECT_HASH_SERVER GlobalServer;

#ifdef PH_WINDOWS

typedef BOOL (WINAPI *PMINIDUMP_WRITE_DUMP)(
    HANDLE Process,
    DWORD ProcessId,
    HANDLE File,
    MINIDUMP_TYPE DumpType,
    PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam,
    PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam,
    PMINIDUMP_CALLBACK_INFORMATION CallbackParam
    );

static HMODULE ServerDbgHelpModule = NULL;
static PMINIDUMP_WRITE_DUMP ServerMiniDumpWriteDump = NULL;
static WCHAR ServerCrashDir[MAX_PATH];

static
VOID
BuildCrashPath(
    _In_opt_ PCWSTR CrashDir,
    _In_ PCWSTR FileName,
    _Out_writes_(BufferChars) PWSTR Buffer,
    _In_ ULONG BufferChars
    )
{
    size_t Length;

    if (!CrashDir || !*CrashDir) {
        wcscpy_s(Buffer, BufferChars, FileName);
        return;
    }

    Length = wcslen(CrashDir);
    if (CrashDir[Length - 1] == L'\\' || CrashDir[Length - 1] == L'/') {
        wcscpy_s(Buffer, BufferChars, CrashDir);
        wcscat_s(Buffer, BufferChars, FileName);
    } else {
        wcscpy_s(Buffer, BufferChars, CrashDir);
        wcscat_s(Buffer, BufferChars, L"\\");
        wcscat_s(Buffer, BufferChars, FileName);
    }
}

static
VOID
InitializeServerCrashDir(
    VOID
    )
{
    DWORD Length;

    ServerCrashDir[0] = L'\0';
    Length = GetEnvironmentVariableW(L"PH_SERVER_CRASH_DIR",
                                     ServerCrashDir,
                                     ARRAYSIZE(ServerCrashDir));
    if (Length == 0 || Length >= ARRAYSIZE(ServerCrashDir)) {
        ServerCrashDir[0] = L'\0';
    } else {
        ServerCrashDir[Length] = L'\0';
    }
}

static
VOID
PreloadServerDbgHelp(
    VOID
    )
{
    if (ServerMiniDumpWriteDump) {
        return;
    }

    ServerDbgHelpModule = LoadLibraryW(L"DbgHelp.dll");
    if (!ServerDbgHelpModule) {
        return;
    }

#pragma warning(push)
#pragma warning(disable: 4191)
    ServerMiniDumpWriteDump = (PMINIDUMP_WRITE_DUMP)(
        GetProcAddress(ServerDbgHelpModule, "MiniDumpWriteDump")
    );
#pragma warning(pop)
}

static
LONG
WINAPI
PerfectHashServerUnhandledExceptionFilter(
    _In_ struct _EXCEPTION_POINTERS *ExceptionPointers
    )
{
    HMODULE DbgHelpModule;
    HANDLE FileHandle = INVALID_HANDLE_VALUE;
    PMINIDUMP_WRITE_DUMP MiniDumpWriteDump;
    MINIDUMP_EXCEPTION_INFORMATION ExceptionInfo;
    PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam = NULL;
    MINIDUMP_TYPE DumpType;
    BOOL DumpResult = FALSE;
    BOOL FallbackResult = FALSE;
    BOOL TriedFallback = FALSE;
    DWORD DumpError = 0;
    DWORD FallbackError = 0;
    HANDLE LogHandle = INVALID_HANDLE_VALUE;
    DWORD BytesWritten;
    PVOID BackTrace[64];
    USHORT Index;
    USHORT Frames;
    CHAR Buffer[256];
    HMODULE Module;
    HMODULE ExeBase;
    ULONG_PTR Offset;
    BOOLEAN ForceFallback = FALSE;
    WCHAR ModulePathW[MAX_PATH];
    CHAR ModulePathA[MAX_PATH];
    DWORD ModulePathLength;
    INT ModulePathChars;
    WCHAR DumpPath[MAX_PATH];
    WCHAR LogPath[MAX_PATH];

    BuildCrashPath(ServerCrashDir,
                   L"PerfectHashServerCrash.dmp",
                   DumpPath,
                   ARRAYSIZE(DumpPath));

    BuildCrashPath(ServerCrashDir,
                   L"PerfectHashServerCrash.log",
                   LogPath,
                   ARRAYSIZE(LogPath));

    LogHandle = CreateFileW(LogPath,
                            GENERIC_WRITE,
                            FILE_SHARE_READ,
                            NULL,
                            CREATE_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL,
                            NULL);

    if (LogHandle != INVALID_HANDLE_VALUE) {
        if (ExceptionPointers) {
            _snprintf_s(Buffer,
                        sizeof(Buffer),
                        _TRUNCATE,
                        "ExceptionCode: 0x%08lX\r\n",
                        ExceptionPointers->ExceptionRecord->ExceptionCode);

            WriteFile(LogHandle,
                      Buffer,
                      (DWORD)strlen(Buffer),
                      &BytesWritten,
                      NULL);

            _snprintf_s(Buffer,
                        sizeof(Buffer),
                        _TRUNCATE,
                        "ExceptionAddress: 0x%p\r\n",
                        ExceptionPointers->ExceptionRecord->ExceptionAddress);

            WriteFile(LogHandle,
                      Buffer,
                      (DWORD)strlen(Buffer),
                      &BytesWritten,
                      NULL);

            Module = NULL;
            if (GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    (LPCWSTR)ExceptionPointers->ExceptionRecord->ExceptionAddress,
                    &Module)) {
                Offset = (ULONG_PTR)ExceptionPointers->ExceptionRecord->ExceptionAddress -
                         (ULONG_PTR)Module;

                _snprintf_s(Buffer,
                            sizeof(Buffer),
                            _TRUNCATE,
                            "ExceptionModule: 0x%p Offset: 0x%Ix\r\n",
                            Module,
                            Offset);

                WriteFile(LogHandle,
                          Buffer,
                          (DWORD)strlen(Buffer),
                          &BytesWritten,
                          NULL);
            }
        } else {
            _snprintf_s(Buffer,
                        sizeof(Buffer),
                        _TRUNCATE,
                        "ExceptionCode: <none>\r\n");

            WriteFile(LogHandle,
                      Buffer,
                      (DWORD)strlen(Buffer),
                      &BytesWritten,
                      NULL);

            _snprintf_s(Buffer,
                        sizeof(Buffer),
                        _TRUNCATE,
                        "ExceptionAddress: <none>\r\n");

            WriteFile(LogHandle,
                      Buffer,
                      (DWORD)strlen(Buffer),
                      &BytesWritten,
                      NULL);
        }

        _snprintf_s(Buffer,
                    sizeof(Buffer),
                    _TRUNCATE,
                    "ThreadId: %lu\r\n",
                    GetCurrentThreadId());

        WriteFile(LogHandle,
                  Buffer,
                  (DWORD)strlen(Buffer),
                  &BytesWritten,
                  NULL);
    }

    if (!ServerMiniDumpWriteDump) {
        PreloadServerDbgHelp();
    }

    DbgHelpModule = ServerDbgHelpModule;
    MiniDumpWriteDump = ServerMiniDumpWriteDump;

    if (!MiniDumpWriteDump) {
        DumpError = GetLastError();
        goto LogOnly;
    }

    FileHandle = CreateFileW(DumpPath,
                             GENERIC_WRITE,
                             FILE_SHARE_READ,
                             NULL,
                             CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL,
                             NULL);

    if (FileHandle == INVALID_HANDLE_VALUE) {
        DumpError = GetLastError();
        goto LogOnly;
    }

    if (ExceptionPointers) {
        ExceptionInfo.ThreadId = GetCurrentThreadId();
        ExceptionInfo.ExceptionPointers = ExceptionPointers;
        ExceptionInfo.ClientPointers = FALSE;
        ExceptionParam = &ExceptionInfo;
    }

    DumpType = MiniDumpWithDataSegs | MiniDumpIgnoreInaccessibleMemory;

    ForceFallback = (GetEnvironmentVariableW(
        L"PH_SERVER_MINIDUMP_FORCE_FALLBACK",
        NULL,
        0
    ) > 0);

    if (!ForceFallback) {
        DumpResult = MiniDumpWriteDump(GetCurrentProcess(),
                                       GetCurrentProcessId(),
                                       FileHandle,
                                       DumpType,
                                       ExceptionParam,
                                       NULL,
                                       NULL);
    } else {
        DumpResult = FALSE;
        DumpError = ERROR_NOACCESS;
    }

    if (!DumpResult) {
        if (!ForceFallback) {
            DumpError = GetLastError();
        }

        if (DumpError == ERROR_NOACCESS && ExceptionParam) {
            DumpResult = MiniDumpWriteDump(GetCurrentProcess(),
                                           GetCurrentProcessId(),
                                           FileHandle,
                                           DumpType,
                                           NULL,
                                           NULL,
                                           NULL);
            if (!DumpResult) {
                DumpError = GetLastError();
            }
        }

        if (DumpResult) {
            goto LogOnly;
        }

        TriedFallback = TRUE;

        CloseHandle(FileHandle);
        FileHandle = CreateFileW(DumpPath,
                                 GENERIC_WRITE,
                                 FILE_SHARE_READ,
                                 NULL,
                                 CREATE_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL,
                                 NULL);

        if (FileHandle != INVALID_HANDLE_VALUE) {
            DumpType = MiniDumpIgnoreInaccessibleMemory;
            FallbackResult = MiniDumpWriteDump(GetCurrentProcess(),
                                               GetCurrentProcessId(),
                                               FileHandle,
                                               DumpType,
                                               ExceptionParam,
                                               NULL,
                                               NULL);
            if (!FallbackResult) {
                FallbackError = GetLastError();
            }
        }
    }

LogOnly:

    if (LogHandle == INVALID_HANDLE_VALUE) {
        LogHandle = CreateFileW(LogPath,
                                FILE_APPEND_DATA,
                                FILE_SHARE_READ,
                                NULL,
                                OPEN_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL,
                                NULL);
    }

    if (LogHandle == INVALID_HANDLE_VALUE) {
        goto End;
    }

    SetFilePointer(LogHandle, 0, NULL, FILE_END);

    ExeBase = GetModuleHandleW(NULL);

    _snprintf_s(Buffer,
                sizeof(Buffer),
                _TRUNCATE,
                "ExeBase: 0x%p\r\n",
                ExeBase);

    WriteFile(LogHandle, Buffer, (DWORD)strlen(Buffer), &BytesWritten, NULL);

    _snprintf_s(Buffer,
                sizeof(Buffer),
                _TRUNCATE,
                "MiniDumpForceFallback: %s\r\n",
                ForceFallback ? "true" : "false");

    WriteFile(LogHandle, Buffer, (DWORD)strlen(Buffer), &BytesWritten, NULL);

    _snprintf_s(Buffer,
                sizeof(Buffer),
                _TRUNCATE,
                "MiniDumpWriteDump: %s (Error=0x%08lX Win32=%lu)\r\n",
                DumpResult ? "OK" : "FAILED",
                DumpError,
                (DumpError & 0xFFFF));

    WriteFile(LogHandle, Buffer, (DWORD)strlen(Buffer), &BytesWritten, NULL);

    if (TriedFallback) {
        _snprintf_s(Buffer,
                    sizeof(Buffer),
                    _TRUNCATE,
                    "MiniDumpWriteDumpFallback: %s (Error=0x%08lX Win32=%lu)\r\n",
                    FallbackResult ? "OK" : "FAILED",
                    FallbackError,
                    (FallbackError & 0xFFFF));

        WriteFile(LogHandle,
                  Buffer,
                  (DWORD)strlen(Buffer),
                  &BytesWritten,
                  NULL);
    }

    _snprintf_s(Buffer,
                sizeof(Buffer),
                _TRUNCATE,
                "ExceptionCode: 0x%08lX\r\n",
                ExceptionPointers->ExceptionRecord->ExceptionCode);

    WriteFile(LogHandle, Buffer, (DWORD)strlen(Buffer), &BytesWritten, NULL);

    _snprintf_s(Buffer,
                sizeof(Buffer),
                _TRUNCATE,
                "ExceptionAddress: 0x%p\r\n",
                ExceptionPointers->ExceptionRecord->ExceptionAddress);

    WriteFile(LogHandle, Buffer, (DWORD)strlen(Buffer), &BytesWritten, NULL);

    Frames = RtlCaptureStackBackTrace(0,
                                      ARRAYSIZE(BackTrace),
                                      BackTrace,
                                      NULL);

    for (Index = 0; Index < Frames; Index++) {
        Module = NULL;
        Offset = 0;
        ModulePathLength = 0;
        ModulePathChars = 0;
        ModulePathA[0] = '?';
        ModulePathA[1] = '\0';

        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCWSTR)BackTrace[Index],
                               &Module)) {
            Offset = (ULONG_PTR)BackTrace[Index] - (ULONG_PTR)Module;

            ModulePathLength = GetModuleFileNameW(Module,
                                                  ModulePathW,
                                                  ARRAYSIZE(ModulePathW));

            if (ModulePathLength > 0) {
                ModulePathW[ARRAYSIZE(ModulePathW) - 1] = L'\0';
                ModulePathChars = WideCharToMultiByte(CP_ACP,
                                                      0,
                                                      ModulePathW,
                                                      ModulePathLength,
                                                      ModulePathA,
                                                      ARRAYSIZE(ModulePathA) - 1,
                                                      NULL,
                                                      NULL);

                if (ModulePathChars > 0) {
                    ModulePathA[ModulePathChars] = '\0';
                }
            }
        }

        _snprintf_s(Buffer,
                    sizeof(Buffer),
                    _TRUNCATE,
                    "Frame %02hu: 0x%p (base=0x%p offset=0x%Ix module=%s)\r\n",
                    Index,
                    BackTrace[Index],
                    Module,
                    Offset,
                    ModulePathA);

        WriteFile(LogHandle,
                  Buffer,
                  (DWORD)strlen(Buffer),
                  &BytesWritten,
                  NULL);
    }

End:

    if (FileHandle != INVALID_HANDLE_VALUE) {
        CloseHandle(FileHandle);
    }

    if (LogHandle != INVALID_HANDLE_VALUE) {
        CloseHandle(LogHandle);
    }

    if (DbgHelpModule) {
        FreeLibrary(DbgHelpModule);
    }

    return EXCEPTION_EXECUTE_HANDLER;
}

static
VOID
InstallPerfectHashServerCrashHandler(
    VOID
    )
{
    DWORD Length;

    Length = GetEnvironmentVariableW(L"PH_LOG_SERVER_CRASH", NULL, 0);
    if (Length == 0) {
        return;
    }

    InitializeServerCrashDir();
    PreloadServerDbgHelp();

    SetUnhandledExceptionFilter(PerfectHashServerUnhandledExceptionFilter);
}

static
VOID
TriggerServerCrashTest(
    _In_ ULONG ExceptionCode
    )
{
    UNREFERENCED_PARAMETER(ExceptionCode);
    PerfectHashServerUnhandledExceptionFilter(NULL);
    ExitProcess((ULONG)E_UNEXPECTED);
}

static
VOID
MaybeTriggerServerCrashTest(
    VOID
    )
{
    DWORD Length;
    WCHAR Buffer[32];
    Length = GetEnvironmentVariableW(L"PH_SERVER_CRASH_TEST",
                                     Buffer,
                                     ARRAYSIZE(Buffer));

    if (Length == 0 || Length >= ARRAYSIZE(Buffer)) {
        return;
    }

    Buffer[Length] = L'\0';

    InitializeServerCrashDir();

    if (_wcsicmp(Buffer, L"AV") == 0 ||
        _wcsicmp(Buffer, L"ACCESS_VIOLATION") == 0) {
        TriggerServerCrashTest(STATUS_ACCESS_VIOLATION);
    } else {
        TriggerServerCrashTest(0xE0000001);
    }
}

#endif
static
VOID
PrintUsage(
    VOID
    )
{
    wprintf(L"Usage: PerfectHashServer [--IocpConcurrency=<N>] "
            L"[--MaxThreads=<N>] [--Numa=All|<list>] [--Endpoint=<pipe>] "
            L"[--AllowRemote|--LocalOnly] [--Verbose] [--NoFileIo] "
            L"[--IocpBufferGuardPages]\n");
    wprintf(L"  --IocpConcurrency=<N> IOCP concurrency per NUMA node\n");
    wprintf(L"  --MaxConcurrency=<N>  Alias for --IocpConcurrency\n");
    wprintf(L"  --MaxThreads=<N>      Worker threads per NUMA node\n");
    wprintf(L"                        Default: IocpConcurrency * 2 when set\n");
    wprintf(L"  --Numa=All|0,1|0-3      NUMA node selection mask\n");
    wprintf(L"  --Endpoint=<pipe>      Named pipe endpoint\n");
    wprintf(L"  --AllowRemote          Allow remote named pipe clients\n");
    wprintf(L"  --LocalOnly            Reject remote named pipe clients\n");
    wprintf(L"  --Verbose              Enable per-request console output\n");
    wprintf(L"  --NoFileIo             Disable file I/O for requests\n");
    wprintf(L"  --IocpBufferGuardPages Enable guard pages for IOCP buffers\n");
}

static
HRESULT
ParseUnsignedInteger(
    _In_ PCWSTR Value,
    _Out_ PULONG Result
    )
{
    PWSTR End = NULL;
    ULONG Local;

    Local = wcstoul(Value, &End, 10);
    if (!End || End == Value || *End != L'\0') {
        return E_INVALIDARG;
    }

    *Result = Local;
    return S_OK;
}

static
HRESULT
ParseNumaNodeMaskValue(
    _In_ PCWSTR Value,
    _Out_ PPERFECT_HASH_NUMA_NODE_MASK Mask
    )
{
    PCWSTR Ptr;
    PERFECT_HASH_NUMA_NODE_MASK LocalMask;

    if (_wcsicmp(Value, L"All") == 0) {
        *Mask = PERFECT_HASH_NUMA_NODE_MASK_ALL;
        return S_OK;
    }

    LocalMask = 0;
    Ptr = Value;

    while (*Ptr) {
        ULONG Start;
        ULONG End;
        PWSTR Next = NULL;

        Start = wcstoul(Ptr, &Next, 10);
        if (!Next || Next == Ptr) {
            return E_INVALIDARG;
        }

        End = Start;

        if (*Next == L'-') {
            Ptr = Next + 1;
            End = wcstoul(Ptr, &Next, 10);
            if (!Next || Next == Ptr) {
                return E_INVALIDARG;
            }
        }

        if (End < Start) {
            return E_INVALIDARG;
        }

        if (End >= 64) {
            return E_INVALIDARG;
        }

        for (; Start <= End; Start++) {
            LocalMask |= (1ULL << Start);
        }

        if (*Next == L',') {
            Ptr = Next + 1;
            continue;
        } else if (*Next == L'\0') {
            break;
        } else {
            return E_INVALIDARG;
        }
    }

    if (LocalMask == 0) {
        return E_INVALIDARG;
    }

    *Mask = LocalMask;
    return S_OK;
}

static
HRESULT
ParseServerArgs(
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW,
    _Inout_ PPERFECT_HASH_SERVER_CLI_OPTIONS Options
    )
{
    ULONG Index;

    Options->IocpConcurrency = 0;
    Options->MaxThreads = 0;
    Options->NumaNodeMask = PERFECT_HASH_NUMA_NODE_MASK_ALL;
    Options->Endpoint.Buffer = NULL;
    Options->Endpoint.Length = 0;
    Options->Endpoint.MaximumLength = 0;
    Options->EndpointPresent = FALSE;
    Options->IocpConcurrencyPresent = FALSE;
    Options->MaxThreadsPresent = FALSE;
    Options->NumaNodeMaskPresent = FALSE;
    Options->LocalOnly = TRUE;
    Options->LocalOnlyPresent = FALSE;
    Options->Verbose = FALSE;
    Options->VerbosePresent = FALSE;
    Options->NoFileIo = FALSE;
    Options->NoFileIoPresent = FALSE;
    Options->IocpBufferGuardPages = FALSE;
    Options->IocpBufferGuardPagesPresent = FALSE;

    for (Index = 1; Index < NumberOfArguments; Index++) {
        PCWSTR Arg = ArgvW[Index];

        if (!Arg) {
            continue;
        }

        if (Arg[0] != L'-' || Arg[1] != L'-') {
            return PH_E_INVALID_COMMANDLINE_ARG;
        }

        Arg += 2;

        if (_wcsicmp(Arg, L"Help") == 0 ||
            _wcsicmp(Arg, L"?") == 0) {
            PrintUsage();
            return S_FALSE;
        }

        if (_wcsnicmp(Arg, L"IocpConcurrency=", 16) == 0) {
            HRESULT Result;
            Arg += 16;
            Result = ParseUnsignedInteger(Arg, &Options->IocpConcurrency);
            if (FAILED(Result)) {
                return Result;
            }
            Options->IocpConcurrencyPresent = TRUE;
            continue;
        }

        if (_wcsnicmp(Arg, L"MaxConcurrency=", 15) == 0) {
            HRESULT Result;
            Arg += 15;
            Result = ParseUnsignedInteger(Arg, &Options->IocpConcurrency);
            if (FAILED(Result)) {
                return Result;
            }
            Options->IocpConcurrencyPresent = TRUE;
            continue;
        }

        if (_wcsnicmp(Arg, L"MaxThreads=", 11) == 0) {
            HRESULT Result;
            Arg += 11;
            Result = ParseUnsignedInteger(Arg, &Options->MaxThreads);
            if (FAILED(Result)) {
                return Result;
            }
            Options->MaxThreadsPresent = TRUE;
            continue;
        }

        if (_wcsnicmp(Arg, L"Numa=", 5) == 0) {
            HRESULT Result;
            Arg += 5;
            Result = ParseNumaNodeMaskValue(Arg, &Options->NumaNodeMask);
            if (FAILED(Result)) {
                return Result;
            }
            Options->NumaNodeMaskPresent = TRUE;
            continue;
        }

        if (_wcsnicmp(Arg, L"Endpoint=", 9) == 0) {
            PCWSTR Value;
            ULONG Length;

            Value = Arg + 9;
            Length = (ULONG)wcslen(Value) * sizeof(WCHAR);

            Options->Endpoint.Buffer = (PWSTR)Value;
            Options->Endpoint.Length = (USHORT)Length;
            Options->Endpoint.MaximumLength = (USHORT)Length + sizeof(WCHAR);
            Options->EndpointPresent = TRUE;
            continue;
        }

        if (_wcsicmp(Arg, L"AllowRemote") == 0) {
            Options->LocalOnly = FALSE;
            Options->LocalOnlyPresent = TRUE;
            continue;
        }

        if (_wcsicmp(Arg, L"LocalOnly") == 0) {
            Options->LocalOnly = TRUE;
            Options->LocalOnlyPresent = TRUE;
            continue;
        }

        if (_wcsicmp(Arg, L"Verbose") == 0) {
            Options->Verbose = TRUE;
            Options->VerbosePresent = TRUE;
            continue;
        }

        if (_wcsicmp(Arg, L"NoFileIo") == 0) {
            Options->NoFileIo = TRUE;
            Options->NoFileIoPresent = TRUE;
            continue;
        }

        if (_wcsicmp(Arg, L"IocpBufferGuardPages") == 0) {
            Options->IocpBufferGuardPages = TRUE;
            Options->IocpBufferGuardPagesPresent = TRUE;
            continue;
        }

        return PH_E_INVALID_COMMANDLINE_ARG;
    }

    return S_OK;
}

#ifdef PH_WINDOWS
static
BOOL
WINAPI
PerfectHashServerConsoleCtrlHandler(
    _In_ DWORD ControlType
    )
{
    switch (ControlType) {
        case CTRL_C_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            if (GlobalServer) {
                GlobalServer->Vtbl->Stop(GlobalServer);
            }
            return TRUE;
        default:
            return FALSE;
    }
}
#endif

//
// Main entry point.
//

#ifdef PH_WINDOWS
DECLSPEC_NORETURN
VOID
WINAPI
mainCRTStartup(
    VOID
    )
{
    HMODULE Module = NULL;
    HRESULT Result = S_OK;
    LPWSTR *ArgvW;
    LPWSTR CommandLineW;
    PICLASSFACTORY ClassFactory;
    PPERFECT_HASH_SERVER Server = NULL;
    PERFECT_HASH_SERVER_CLI_OPTIONS Options = { 0 };
    PPERFECT_HASH_PRINT_ERROR PerfectHashPrintError = NULL;
    PPERFECT_HASH_PRINT_MESSAGE PerfectHashPrintMessage;
    PICLASSFACTORY_CREATE_INSTANCE CreateInstance;
    INT NumberOfArguments = 0;

    CommandLineW = GetCommandLineW();
    ArgvW = CommandLineToArgvW(CommandLineW, &NumberOfArguments);

    InstallPerfectHashServerCrashHandler();
    MaybeTriggerServerCrashTest();

    Result = PerfectHashBootstrap(&ClassFactory,
                                  &PerfectHashPrintError,
                                  &PerfectHashPrintMessage,
                                  &Module);

    if (FAILED(Result)) {
        if (PerfectHashPrintError != NULL) {
            PH_ERROR(PerfectHashBootstrap, Result);
        }
        goto Error;
    }

    CreateInstance = ClassFactory->Vtbl->CreateInstance;

    Result = CreateInstance(ClassFactory,
                            NULL,
                            &IID_PERFECT_HASH_SERVER,
                            &Server);

    if (FAILED(Result)) {
        if (PerfectHashPrintError != NULL) {
            PH_ERROR(PerfectHashServerCreateInstance, Result);
        }
        goto Error;
    }

    Result = ParseServerArgs(NumberOfArguments, ArgvW, &Options);
    if (Result == S_FALSE) {
        Result = S_OK;
        goto End;
    } else if (FAILED(Result)) {
        PrintUsage();
        goto Error;
    }

    if (Options.IocpConcurrencyPresent) {
        Result = Server->Vtbl->SetMaximumConcurrency(
            Server,
            Options.IocpConcurrency
        );
        if (FAILED(Result)) {
            goto Error;
        }

        if (!Options.MaxThreadsPresent) {
            ULONGLONG DefaultThreads;

            DefaultThreads = (ULONGLONG)Options.IocpConcurrency * 2;
            if (DefaultThreads == 0 || DefaultThreads > ULONG_MAX) {
                Result = E_INVALIDARG;
                goto Error;
            }

            Options.MaxThreads = (ULONG)DefaultThreads;
            Options.MaxThreadsPresent = TRUE;
        }
    }

    if (Options.MaxThreadsPresent) {
        Result = Server->Vtbl->SetMaximumThreads(
            Server,
            Options.MaxThreads
        );
        if (FAILED(Result)) {
            goto Error;
        }
    }

    if (Options.NumaNodeMaskPresent) {
        Result = Server->Vtbl->SetNumaNodeMask(Server,
                                               Options.NumaNodeMask);
        if (FAILED(Result)) {
            goto Error;
        }
    }

    if (Options.EndpointPresent) {
        Result = Server->Vtbl->SetEndpoint(Server, &Options.Endpoint);
        if (FAILED(Result)) {
            goto Error;
        }
    }

    if (Options.LocalOnlyPresent) {
        Result = Server->Vtbl->SetLocalOnly(Server, Options.LocalOnly);
        if (FAILED(Result)) {
            goto Error;
        }
    }

    if (Options.VerbosePresent) {
        Result = Server->Vtbl->SetVerbose(Server, Options.Verbose);
        if (FAILED(Result)) {
            goto Error;
        }
    }

    if (Options.NoFileIoPresent) {
        Result = Server->Vtbl->SetNoFileIo(Server, Options.NoFileIo);
        if (FAILED(Result)) {
            goto Error;
        }
    }

    if (Options.IocpBufferGuardPagesPresent) {
        Result = Server->Vtbl->SetIocpBufferGuardPages(
            Server,
            Options.IocpBufferGuardPages
        );
        if (FAILED(Result)) {
            goto Error;
        }
    }

    GlobalServer = Server;
    SetConsoleCtrlHandler(PerfectHashServerConsoleCtrlHandler, TRUE);

    Result = Server->Vtbl->Start(Server);
    if (FAILED(Result)) {
        goto Error;
    }

    Result = Server->Vtbl->Wait(Server);
    if (FAILED(Result)) {
        goto Error;
    }

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    if (PerfectHashPrintError != NULL) {
        PH_ERROR(PerfectHashServer, Result);
    }

End:

    if (Server) {
        Server->Vtbl->Release(Server);
    }

    if (ClassFactory) {
        ClassFactory->Vtbl->Release(ClassFactory);
    }

    if (Module) {
        FreeLibrary(Module);
    }

    ExitProcess((ULONG)Result);
}

#else // PH_WINDOWS

int
main(
    int NumberOfArguments,
    char **ArgvA
    )
{
    UNREFERENCED_PARAMETER(NumberOfArguments);
    UNREFERENCED_PARAMETER(ArgvA);

    return 0;
}

#endif // PH_WINDOWS

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
