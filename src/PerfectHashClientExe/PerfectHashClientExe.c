/*++

Copyright (c) 2018-2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashClientExe.c

Abstract:

    This module implements the main entry point for the perfect hash client.
    It loads the perfect hash library, creates a client instance, then attempts
    to connect and optionally submit a request.

--*/

#include "stdafx.h"
#include <stdio.h>

typedef union _PERFECT_HASH_CLIENT_CLI_FLAGS {
    struct {
        ULONG EndpointPresent:1;
        ULONG CommandLinePresent:1;
        ULONG RequestTypePresent:1;
        ULONG WaitForServer:1;
        ULONG ConnectTimeoutPresent:1;
        ULONG Unused:27;
    };
    ULONG AsULong;
} PERFECT_HASH_CLIENT_CLI_FLAGS;
typedef PERFECT_HASH_CLIENT_CLI_FLAGS *PPERFECT_HASH_CLIENT_CLI_FLAGS;

typedef struct _PERFECT_HASH_CLIENT_CLI_OPTIONS {
    UNICODE_STRING Endpoint;
    UNICODE_STRING CommandLine;
    PERFECT_HASH_SERVER_REQUEST_TYPE RequestType;
    ULONG ConnectTimeoutInMilliseconds;
    PERFECT_HASH_CLIENT_CLI_FLAGS Flags;
    ULONG Padding1;
} PERFECT_HASH_CLIENT_CLI_OPTIONS;
typedef PERFECT_HASH_CLIENT_CLI_OPTIONS *PPERFECT_HASH_CLIENT_CLI_OPTIONS;

static
VOID
PrintUsage(
    VOID
    )
{
    wprintf(L"Usage: PerfectHashClient [--Endpoint=<name>] "
            L"[--Shutdown|--TableCreate=<cmd>|--BulkCreate=<cmd>|"
            L"--BulkCreateDirectory=<cmd>|--Ping]\n");
    wprintf(L"  --TableCreate=<cmd>     PerfectHashCreate-style arguments\n");
    wprintf(L"  --BulkCreate=<cmd>      PerfectHashBulkCreate-style arguments\n");
    wprintf(L"  --BulkCreateDirectory=<cmd>  BulkCreate args with single token\n");
    wprintf(L"  --Ping                  Wait for server readiness\n");
    wprintf(L"  --WaitForServer          Wait for server to appear\n");
    wprintf(L"  --ConnectTimeout=<ms>    Cap wait time for --WaitForServer\n");
}

static
HRESULT
ParseClientArgs(
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW,
    _Inout_ PPERFECT_HASH_CLIENT_CLI_OPTIONS Options
    )
{
    ULONG Index;

    Options->Endpoint.Buffer = NULL;
    Options->Endpoint.Length = 0;
    Options->Endpoint.MaximumLength = 0;
    Options->Flags.AsULong = 0;
    Options->CommandLine.Buffer = NULL;
    Options->CommandLine.Length = 0;
    Options->CommandLine.MaximumLength = 0;
    Options->RequestType = PerfectHashNullServerRequestType;
    Options->ConnectTimeoutInMilliseconds = 0;

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

        if (_wcsnicmp(Arg, L"Endpoint=", 9) == 0) {
            PCWSTR Value;
            ULONG Length;

            Value = Arg + 9;
            Length = (ULONG)wcslen(Value) * sizeof(WCHAR);

            Options->Endpoint.Buffer = (PWSTR)Value;
            Options->Endpoint.Length = (USHORT)Length;
            Options->Endpoint.MaximumLength = (USHORT)Length + sizeof(WCHAR);
            Options->Flags.EndpointPresent = TRUE;
            continue;
        }

        if (_wcsicmp(Arg, L"Shutdown") == 0) {
            if (Options->Flags.RequestTypePresent) {
                return PH_E_INVALID_COMMANDLINE_ARG;
            }
            Options->RequestType = PerfectHashShutdownServerRequestType;
            Options->Flags.RequestTypePresent = TRUE;
            continue;
        }

        if (_wcsicmp(Arg, L"Ping") == 0) {
            if (Options->Flags.RequestTypePresent) {
                return PH_E_INVALID_COMMANDLINE_ARG;
            }
            Options->RequestType = PerfectHashPingServerRequestType;
            Options->Flags.RequestTypePresent = TRUE;
            continue;
        }

        if (_wcsicmp(Arg, L"WaitForServer") == 0) {
            Options->Flags.WaitForServer = TRUE;
            continue;
        }

        if (_wcsnicmp(Arg, L"ConnectTimeout=", 15) == 0) {
            PCWSTR Value;
            ULONG Timeout;

            Value = Arg + 15;
            if (!Value || *Value == L'\0') {
                return PH_E_INVALID_COMMANDLINE_ARG;
            }

            Timeout = wcstoul(Value, NULL, 10);
            if (Timeout == 0) {
                return PH_E_INVALID_COMMANDLINE_ARG;
            }

            Options->ConnectTimeoutInMilliseconds = Timeout;
            Options->Flags.ConnectTimeoutPresent = TRUE;
            continue;
        }

        if (_wcsnicmp(Arg, L"TableCreate=", 12) == 0) {
            PCWSTR Value;
            ULONG Length;

            if (Options->Flags.RequestTypePresent) {
                return PH_E_INVALID_COMMANDLINE_ARG;
            }

            Value = Arg + 12;
            Length = (ULONG)wcslen(Value) * sizeof(WCHAR);

            Options->CommandLine.Buffer = (PWSTR)Value;
            Options->CommandLine.Length = (USHORT)Length;
            Options->CommandLine.MaximumLength = (USHORT)Length +
                                                 sizeof(WCHAR);
            Options->Flags.CommandLinePresent = TRUE;
            Options->RequestType = PerfectHashTableCreateServerRequestType;
            Options->Flags.RequestTypePresent = TRUE;
            continue;
        }

        if (_wcsnicmp(Arg, L"BulkCreate=", 11) == 0) {
            PCWSTR Value;
            ULONG Length;

            if (Options->Flags.RequestTypePresent) {
                return PH_E_INVALID_COMMANDLINE_ARG;
            }

            Value = Arg + 11;
            Length = (ULONG)wcslen(Value) * sizeof(WCHAR);

            Options->CommandLine.Buffer = (PWSTR)Value;
            Options->CommandLine.Length = (USHORT)Length;
            Options->CommandLine.MaximumLength = (USHORT)Length +
                                                 sizeof(WCHAR);
            Options->Flags.CommandLinePresent = TRUE;
            Options->RequestType = PerfectHashBulkCreateServerRequestType;
            Options->Flags.RequestTypePresent = TRUE;
            continue;
        }

        if (_wcsnicmp(Arg, L"BulkCreateDirectory=", 20) == 0) {
            PCWSTR Value;
            ULONG Length;

            if (Options->Flags.RequestTypePresent) {
                return PH_E_INVALID_COMMANDLINE_ARG;
            }

            Value = Arg + 20;
            Length = (ULONG)wcslen(Value) * sizeof(WCHAR);

            Options->CommandLine.Buffer = (PWSTR)Value;
            Options->CommandLine.Length = (USHORT)Length;
            Options->CommandLine.MaximumLength = (USHORT)Length +
                                                 sizeof(WCHAR);
            Options->Flags.CommandLinePresent = TRUE;
            Options->RequestType =
                PerfectHashBulkCreateDirectoryServerRequestType;
            Options->Flags.RequestTypePresent = TRUE;
            continue;
        }

        return PH_E_INVALID_COMMANDLINE_ARG;
    }

    if (Options->Flags.RequestTypePresent) {
        if (Options->RequestType == PerfectHashShutdownServerRequestType ||
            Options->RequestType == PerfectHashPingServerRequestType) {
            if (Options->Flags.CommandLinePresent) {
                return PH_E_INVALID_COMMANDLINE_ARG;
            }
        } else if (!Options->Flags.CommandLinePresent) {
            return PH_E_INVALID_COMMANDLINE_ARG;
        }
    }

    if (Options->Flags.ConnectTimeoutPresent && !Options->Flags.WaitForServer) {
        return PH_E_INVALID_COMMANDLINE_ARG;
    }

    return S_OK;
}

static
HRESULT
PerfectHashClientConnectWithWait(
    _In_ PPERFECT_HASH_CLIENT Client,
    _In_opt_ PCUNICODE_STRING Endpoint,
    _In_ BOOLEAN WaitForServer,
    _In_ BOOLEAN TimeoutPresent,
    _In_ ULONG TimeoutInMilliseconds
    )
{
    HRESULT Result;
    ULONGLONG StartTicks;

    if (!WaitForServer) {
        return Client->Vtbl->Connect(Client, Endpoint);
    }

    StartTicks = GetTickCount64();

    for (;;) {
        Result = Client->Vtbl->Connect(Client, Endpoint);
        if (SUCCEEDED(Result)) {
            return S_OK;
        }

        if (Result != HRESULT_FROM_WIN32(ERROR_PIPE_BUSY) &&
            Result != HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND)) {
            return Result;
        }

        if (TimeoutPresent) {
            ULONGLONG Elapsed = GetTickCount64() - StartTicks;
            if (Elapsed >= TimeoutInMilliseconds) {
                return HRESULT_FROM_WIN32(WAIT_TIMEOUT);
            }
        }

        Sleep(50);
    }
}

static
HRESULT
PerfectHashClientPing(
    _In_ PPERFECT_HASH_CLIENT Client
    )
{
    HRESULT Result;
    ULONG ResponseFlags = 0;
    UNICODE_STRING ResponsePayload;
    PERFECT_HASH_SERVER_REQUEST Request = { 0 };

    Request.SizeOfStruct = sizeof(Request);
    Request.RequestType = PerfectHashPingServerRequestType;
    Request.RequestId = 1;

    Result = Client->Vtbl->SubmitRequest(Client, &Request);
    if (FAILED(Result)) {
        return Result;
    }

    Result = Client->Vtbl->GetLastResponse(Client,
                                           &ResponsePayload,
                                           &ResponseFlags);
    if (FAILED(Result)) {
        return Result;
    }

    if (!(ResponseFlags & PERFECT_HASH_SERVER_RESPONSE_FLAG_PONG)) {
        return E_UNEXPECTED;
    }

    if (!ResponsePayload.Buffer ||
        _wcsicmp(ResponsePayload.Buffer, L"PONG") != 0) {
        return E_UNEXPECTED;
    }

    return S_OK;
}

static
HRESULT
PerfectHashClientWaitForBulkCreateToken(
    _In_ PPERFECT_HASH_CLIENT Client,
    _Out_opt_ PPERFECT_HASH_SERVER_BULK_RESULT BulkResult
    )
{
    HRESULT Result;
    DWORD WaitResult;
    int Count;
    ULONG ResponseFlags;
    ULONGLONG EventHandleValue = 0;
    ULONGLONG ResultHandleValue = 0;
    HANDLE EventHandle = NULL;
    HANDLE ResultHandle = NULL;
    PPERFECT_HASH_SERVER_BULK_RESULT Mapping = NULL;
    UNICODE_STRING ResponsePayload;

    if (!ARGUMENT_PRESENT(Client)) {
        return E_POINTER;
    }

    if (ARGUMENT_PRESENT(BulkResult)) {
        ZeroMemory(BulkResult, sizeof(*BulkResult));
    }

    Result = Client->Vtbl->GetLastResponse(Client,
                                           &ResponsePayload,
                                           &ResponseFlags);
    if (FAILED(Result)) {
        return Result;
    }

    if (!(ResponseFlags &
          PERFECT_HASH_SERVER_RESPONSE_FLAG_BULK_CREATE_TOKEN)) {
        return E_UNEXPECTED;
    }

    if (!ResponsePayload.Buffer || ResponsePayload.Length == 0) {
        return E_UNEXPECTED;
    }

    Count = swscanf_s(ResponsePayload.Buffer,
                      PERFECT_HASH_SERVER_BULK_CREATE_TOKEN_FORMAT,
                      &EventHandleValue,
                      &ResultHandleValue);
    if (Count != 2) {
        return E_INVALIDARG;
    }

    if (EventHandleValue == 0 || ResultHandleValue == 0) {
        return E_INVALIDARG;
    }

    EventHandle = (HANDLE)(ULONG_PTR)EventHandleValue;
    ResultHandle = (HANDLE)(ULONG_PTR)ResultHandleValue;

    WaitResult = WaitForSingleObject(EventHandle, INFINITE);
    if (WaitResult != WAIT_OBJECT_0) {
        if (WaitResult == WAIT_FAILED) {
            Result = HRESULT_FROM_WIN32(GetLastError());
        } else {
            Result = HRESULT_FROM_WIN32(WaitResult);
        }
        goto End;
    }

    Mapping = (PPERFECT_HASH_SERVER_BULK_RESULT)(
        MapViewOfFile(ResultHandle, FILE_MAP_READ, 0, 0, 0)
    );
    if (!Mapping) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto End;
    }

    if (Mapping->SizeOfStruct != sizeof(*Mapping) ||
        Mapping->Version != PERFECT_HASH_SERVER_BULK_RESULT_VERSION) {
        Result = E_UNEXPECTED;
        goto End;
    }

    if (ARGUMENT_PRESENT(BulkResult)) {
        *BulkResult = *Mapping;
        Result = S_OK;
    } else {
        Result = Mapping->Result;
    }

End:

    if (Mapping) {
        UnmapViewOfFile(Mapping);
    }

    if (ResultHandle) {
        CloseHandle(ResultHandle);
    }

    if (EventHandle) {
        CloseHandle(EventHandle);
    }

    return Result;
}

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
    PPERFECT_HASH_CLIENT Client = NULL;
    PERFECT_HASH_CLIENT_CLI_OPTIONS Options = { 0 };
    PERFECT_HASH_SERVER_REQUEST Request = { 0 };
    PERFECT_HASH_SERVER_BULK_RESULT BulkResult = { 0 };
    PPERFECT_HASH_PRINT_ERROR PerfectHashPrintError = NULL;
    PPERFECT_HASH_PRINT_MESSAGE PerfectHashPrintMessage;
    PICLASSFACTORY_CREATE_INSTANCE CreateInstance;
    INT NumberOfArguments = 0;

    CommandLineW = GetCommandLineW();
    ArgvW = CommandLineToArgvW(CommandLineW, &NumberOfArguments);

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
                            &IID_PERFECT_HASH_CLIENT,
                            &Client);

    if (FAILED(Result)) {
        if (PerfectHashPrintError != NULL) {
            PH_ERROR(PerfectHashClientCreateInstance, Result);
        }
        goto Error;
    }

    Result = ParseClientArgs(NumberOfArguments, ArgvW, &Options);
    if (Result == S_FALSE) {
        Result = S_OK;
        goto End;
    } else if (FAILED(Result)) {
        PrintUsage();
        goto Error;
    }

    Result = PerfectHashClientConnectWithWait(
        Client,
        Options.Flags.EndpointPresent ? &Options.Endpoint : NULL,
        (BOOLEAN)!!Options.Flags.WaitForServer,
        (BOOLEAN)!!Options.Flags.ConnectTimeoutPresent,
        Options.ConnectTimeoutInMilliseconds
    );
    if (FAILED(Result)) {
        goto Error;
    }

    if (Options.Flags.WaitForServer ||
        Options.RequestType == PerfectHashPingServerRequestType) {
        Result = PerfectHashClientPing(Client);
        if (FAILED(Result)) {
            goto Error;
        }

        if (Options.RequestType == PerfectHashPingServerRequestType ||
            !Options.Flags.RequestTypePresent) {
            goto End;
        }

        Client->Vtbl->Disconnect(Client);
        Result = PerfectHashClientConnectWithWait(
            Client,
            Options.Flags.EndpointPresent ? &Options.Endpoint : NULL,
            (BOOLEAN)!!Options.Flags.WaitForServer,
            (BOOLEAN)!!Options.Flags.ConnectTimeoutPresent,
            Options.ConnectTimeoutInMilliseconds
        );
        if (FAILED(Result)) {
            goto Error;
        }
    }

    if (Options.Flags.RequestTypePresent) {
        Request.SizeOfStruct = sizeof(Request);
        Request.RequestType = Options.RequestType;
        Request.RequestId = 1;
        if (Options.Flags.CommandLinePresent) {
            Request.CommandLine = Options.CommandLine;
        }

        Result = Client->Vtbl->SubmitRequest(Client, &Request);
        if (FAILED(Result)) {
            UNICODE_STRING ResponsePayload;
            ULONG ResponseFlags = 0;
            HRESULT PayloadResult;

            PayloadResult = Client->Vtbl->GetLastResponse(Client,
                                                          &ResponsePayload,
                                                          &ResponseFlags);
            if (SUCCEEDED(PayloadResult) &&
                ResponsePayload.Buffer &&
                (ResponseFlags &
                 PERFECT_HASH_SERVER_RESPONSE_FLAG_ERROR_MESSAGE)) {
                wprintf(L"%.*s\n",
                        (int)(ResponsePayload.Length / sizeof(WCHAR)),
                        ResponsePayload.Buffer);
            }
            goto Error;
        }

        if (Options.RequestType ==
            PerfectHashBulkCreateDirectoryServerRequestType) {
            Result = PerfectHashClientWaitForBulkCreateToken(Client,
                                                             &BulkResult);
            if (FAILED(Result)) {
                goto Error;
            }

            Result = BulkResult.Result;
            if (FAILED(Result)) {
                if (BulkResult.TotalFiles || BulkResult.FailedFiles) {
                    wprintf(L"Bulk result: total=%lu succeeded=%lu failed=%lu "
                            L"first=0x%08lX\n",
                            BulkResult.TotalFiles,
                            BulkResult.SucceededFiles,
                            BulkResult.FailedFiles,
                            (ULONG)BulkResult.FirstFailure);
                }
                if (PerfectHashPrintError != NULL) {
                    PH_ERROR(BulkCreateDirectory, Result);
                }
            }
            goto End;
        }
    }

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    if (PerfectHashPrintError != NULL) {
        PH_ERROR(PerfectHashClient, Result);
    }

End:

    if (Client) {
        Client->Vtbl->Release(Client);
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
