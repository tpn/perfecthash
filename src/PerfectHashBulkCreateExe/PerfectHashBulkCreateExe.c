/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashBulkCreateExe.c

Abstract:

    This module implements the main entry point for the perfect hash library's
    bulk-create functionality.  It loads the perfect hash library, obtains a
    class factory, creates a context, then calls the bulk-create function
    against the current executable's command line parameters.

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

//
// Main entry point.
//

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

static HMODULE BulkCreateDbgHelpModule = NULL;
static PMINIDUMP_WRITE_DUMP BulkCreateMiniDumpWriteDump = NULL;
static WCHAR BulkCreateCrashDir[MAX_PATH];

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
InitializeBulkCreateCrashDir(
    VOID
    )
{
    DWORD Length;

    BulkCreateCrashDir[0] = L'\0';
    Length = GetEnvironmentVariableW(L"PH_BULK_CREATE_CRASH_DIR",
                                     BulkCreateCrashDir,
                                     ARRAYSIZE(BulkCreateCrashDir));
    if (Length == 0 || Length >= ARRAYSIZE(BulkCreateCrashDir)) {
        BulkCreateCrashDir[0] = L'\0';
    } else {
        BulkCreateCrashDir[Length] = L'\0';
    }
}

static
VOID
PreloadBulkCreateDbgHelp(
    VOID
    )
{
    if (BulkCreateMiniDumpWriteDump) {
        return;
    }

    BulkCreateDbgHelpModule = LoadLibraryW(L"DbgHelp.dll");
    if (!BulkCreateDbgHelpModule) {
        return;
    }

#pragma warning(push)
#pragma warning(disable: 4191)
    BulkCreateMiniDumpWriteDump = (PMINIDUMP_WRITE_DUMP)(
        GetProcAddress(BulkCreateDbgHelpModule, "MiniDumpWriteDump")
    );
#pragma warning(pop)
}

static
LONG
WINAPI
BulkCreateUnhandledExceptionFilter(
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

    BuildCrashPath(BulkCreateCrashDir,
                   L"PerfectHashBulkCreateCrash.dmp",
                   DumpPath,
                   ARRAYSIZE(DumpPath));

    BuildCrashPath(BulkCreateCrashDir,
                   L"PerfectHashBulkCreateCrash.log",
                   LogPath,
                   ARRAYSIZE(LogPath));

    if (!BulkCreateMiniDumpWriteDump) {
        PreloadBulkCreateDbgHelp();
    }

    DbgHelpModule = BulkCreateDbgHelpModule;
    MiniDumpWriteDump = BulkCreateMiniDumpWriteDump;

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
        L"PH_BULK_CREATE_MINIDUMP_FORCE_FALLBACK",
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

    LogHandle = CreateFileW(LogPath,
                            GENERIC_WRITE,
                            FILE_SHARE_READ,
                            NULL,
                            CREATE_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL,
                            NULL);

    if (LogHandle == INVALID_HANDLE_VALUE) {
        goto End;
    }

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
InstallBulkCreateCrashHandler(
    VOID
    )
{
    DWORD Length;

    Length = GetEnvironmentVariableW(L"PH_LOG_BULK_CREATE_CRASH", NULL, 0);
    if (Length == 0) {
        return;
    }

    InitializeBulkCreateCrashDir();
    PreloadBulkCreateDbgHelp();

    SetUnhandledExceptionFilter(BulkCreateUnhandledExceptionFilter);
}

static
VOID
TriggerBulkCreateCrashTest(
    _In_ ULONG ExceptionCode
    )
{
    UNREFERENCED_PARAMETER(ExceptionCode);
    BulkCreateUnhandledExceptionFilter(NULL);
    ExitProcess((ULONG)E_UNEXPECTED);
}

static
VOID
MaybeTriggerBulkCreateCrashTest(
    VOID
    )
{
    DWORD Length;
    WCHAR Buffer[32];
    Length = GetEnvironmentVariableW(L"PH_BULK_CREATE_CRASH_TEST",
                                     Buffer,
                                     ARRAYSIZE(Buffer));

    if (Length == 0 || Length >= ARRAYSIZE(Buffer)) {
        return;
    }

    Buffer[Length] = L'\0';

    InitializeBulkCreateCrashDir();

    if (_wcsicmp(Buffer, L"AV") == 0 ||
        _wcsicmp(Buffer, L"ACCESS_VIOLATION") == 0) {
        TriggerBulkCreateCrashTest(STATUS_ACCESS_VIOLATION);
    } else {
        TriggerBulkCreateCrashTest(0xE0000001);
    }
}
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
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_PRINT_ERROR PerfectHashPrintError = NULL;
    PPERFECT_HASH_PRINT_MESSAGE PerfectHashPrintMessage;
    PICLASSFACTORY_CREATE_INSTANCE CreateInstance;
    INT NumberOfArguments = 0;

    CommandLineW = GetCommandLineW();
    ArgvW = CommandLineToArgvW(CommandLineW, &NumberOfArguments);

    InstallBulkCreateCrashHandler();
    MaybeTriggerBulkCreateCrashTest();

    Result = PerfectHashBootstrap(&ClassFactory,
                                  &PerfectHashPrintError,
                                  &PerfectHashPrintMessage,
                                  &Module);

    if (FAILED(Result)) {

        //
        // We can only use PH_ERROR() if PerfectHashPrintError is available.
        //

        if (PerfectHashPrintError != NULL) {
            PH_ERROR(PerfectHashBootstrap, Result);
        }

        goto Error;
    }

    CreateInstance = ClassFactory->Vtbl->CreateInstance;

    Result = CreateInstance(ClassFactory,
                            NULL,
                            &IID_PERFECT_HASH_CONTEXT,
                            &Context);

    if (FAILED(Result)) {
        if (PerfectHashPrintError != NULL) {
            PH_ERROR(PerfectHashContextCreateInstance, Result);
        }
        goto Error;
    }

    Result = Context->Vtbl->BulkCreateArgvW(Context,
                                            NumberOfArguments,
                                            ArgvW,
                                            CommandLineW);

    //
    // Print the usage string if the create routine failed due to invalid number
    // of arguments.  Otherwise, as long as we were able to resolve the error
    // routine, print an error message.
    //

    if (FAILED(Result)) {
        if (Result == PH_E_CONTEXT_BULK_CREATE_INVALID_NUM_ARGS) {
            PH_USAGE();
        } else if (PerfectHashPrintError != NULL) {
            PH_ERROR(BulkCreate, Result);
        }
    }

    Context->Vtbl->Release(Context);

    ClassFactory->Vtbl->Release(ClassFactory);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

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
    HMODULE Module = NULL;
    HRESULT Result = S_OK;
    LPWSTR *ArgvW;
    LPWSTR CommandLineW;
    PICLASSFACTORY ClassFactory;
    PPERFECT_HASH_CONTEXT Context;
    PPERFECT_HASH_PRINT_ERROR PerfectHashPrintError = NULL;
    PPERFECT_HASH_PRINT_MESSAGE PerfectHashPrintMessage;
    PICLASSFACTORY_CREATE_INSTANCE CreateInstance;

    Result = PerfectHashBootstrap(&ClassFactory,
                                  &PerfectHashPrintError,
                                  &PerfectHashPrintMessage,
                                  &Module);

    if (FAILED(Result)) {

        //
        // We can only use PH_ERROR() if PerfectHashPrintError is available.
        //

        if (PerfectHashPrintError != NULL) {
            PH_ERROR(PerfectHashBootstrap, Result);
        }

        goto Error;
    }

    CreateInstance = ClassFactory->Vtbl->CreateInstance;

    Result = CreateInstance(ClassFactory,
                            NULL,
                            &IID_PERFECT_HASH_CONTEXT,
                            &Context);

    if (FAILED(Result)) {
        if (PerfectHashPrintError != NULL) {
            PH_ERROR(PerfectHashContextCreateInstance, Result);
        }
        goto Error;
    }

    Result = Context->Vtbl->BulkCreateArgvA(Context,
                                            NumberOfArguments,
                                            ArgvA);

    //
    // Print the usage string if the create routine failed due to invalid number
    // of arguments.  Otherwise, as long as we were able to resolve the error
    // routine, print an error message.
    //

    if (FAILED(Result)) {
        if (Result == PH_E_CONTEXT_BULK_CREATE_INVALID_NUM_ARGS) {
            PH_USAGE();
        } else if (PerfectHashPrintError != NULL) {
            PH_ERROR(TableCreate, Result);
        }
    }

    Context->Vtbl->Release(Context);

    ClassFactory->Vtbl->Release(ClassFactory);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    ExitProcess((ULONG)Result);
}
#endif // !PH_WINDOWS


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
