/*++

Copyright (c) 2018-2025 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashClient.c

Abstract:

    This module implements the PERFECT_HASH_CLIENT component.  The initial
    implementation focuses on wiring the COM interface and reserving slots
    for connection and request submission logic.

--*/

#include "stdafx.h"
#include <limits.h>

#ifdef PH_WINDOWS
static const UNICODE_STRING PerfectHashClientDefaultPipeName =
    RTL_CONSTANT_STRING(L"\\\\.\\pipe\\PerfectHashServer");
#endif

static
HRESULT
PerfectHashClientWriteAll(
    _In_ HANDLE Handle,
    _In_reads_bytes_(Length) PVOID Buffer,
    _In_ ULONG Length
    )
{
#ifdef PH_WINDOWS
    ULONG Offset;

    Offset = 0;
    while (Offset < Length) {
        DWORD Written = 0;
        BOOL Success;
        ULONG Remaining = Length - Offset;

        Success = WriteFile(Handle,
                            (PUCHAR)Buffer + Offset,
                            Remaining,
                            &Written,
                            NULL);
        if (!Success) {
            return HRESULT_FROM_WIN32(GetLastError());
        }

        if (Written == 0) {
            return HRESULT_FROM_WIN32(ERROR_BROKEN_PIPE);
        }

        Offset += Written;
    }

    return S_OK;
#else
    UNREFERENCED_PARAMETER(Handle);
    UNREFERENCED_PARAMETER(Buffer);
    UNREFERENCED_PARAMETER(Length);
    return E_NOTIMPL;
#endif
}

static
HRESULT
PerfectHashClientReadAll(
    _In_ HANDLE Handle,
    _Out_writes_bytes_(Length) PVOID Buffer,
    _In_ ULONG Length
    )
{
#ifdef PH_WINDOWS
    ULONG Offset;

    Offset = 0;
    while (Offset < Length) {
        DWORD Read = 0;
        BOOL Success;
        ULONG Remaining = Length - Offset;

        Success = ReadFile(Handle,
                           (PUCHAR)Buffer + Offset,
                           Remaining,
                           &Read,
                           NULL);
        if (!Success) {
            return HRESULT_FROM_WIN32(GetLastError());
        }

        if (Read == 0) {
            return HRESULT_FROM_WIN32(ERROR_BROKEN_PIPE);
        }

        Offset += Read;
    }

    return S_OK;
#else
    UNREFERENCED_PARAMETER(Handle);
    UNREFERENCED_PARAMETER(Buffer);
    UNREFERENCED_PARAMETER(Length);
    return E_NOTIMPL;
#endif
}

PERFECT_HASH_CLIENT_INITIALIZE PerfectHashClientInitialize;

_Use_decl_annotations_
HRESULT
PerfectHashClientInitialize(
    PPERFECT_HASH_CLIENT Client
    )
/*++

Routine Description:

    Initializes a PERFECT_HASH_CLIENT instance.

Arguments:

    Client - Supplies a pointer to a PERFECT_HASH_CLIENT structure for which
        initialization is to be performed.

Return Value:

    S_OK on success, an appropriate error code on failure.

--*/
{
    HRESULT Result;

    if (!ARGUMENT_PRESENT(Client)) {
        return E_POINTER;
    }

    Result = Client->Vtbl->CreateInstance(Client,
                                          NULL,
                                          &IID_PERFECT_HASH_RTL,
                                          &Client->Rtl);
    if (FAILED(Result)) {
        return Result;
    }

    Result = Client->Vtbl->CreateInstance(Client,
                                          NULL,
                                          &IID_PERFECT_HASH_ALLOCATOR,
                                          &Client->Allocator);
    if (FAILED(Result)) {
        return Result;
    }

    Client->ResponsePayload.Buffer = NULL;
    Client->ResponsePayload.Length = 0;
    Client->ResponsePayload.MaximumLength = 0;
    Client->ResponsePayloadBufferSize = 0;
    Client->ResponseFlags = 0;
    Client->State.Initialized = TRUE;

    return S_OK;
}

PERFECT_HASH_CLIENT_RUNDOWN PerfectHashClientRundown;

_Use_decl_annotations_
VOID
PerfectHashClientRundown(
    PPERFECT_HASH_CLIENT Client
    )
/*++

Routine Description:

    Releases resources associated with a PERFECT_HASH_CLIENT instance.

Arguments:

    Client - Supplies a pointer to a PERFECT_HASH_CLIENT structure for which
        rundown is to be performed.

Return Value:

    None.

--*/
{
    if (!ARGUMENT_PRESENT(Client)) {
        return;
    }

#ifdef PH_WINDOWS
    if (Client->ConnectionHandle) {
        CloseHandle(Client->ConnectionHandle);
        Client->ConnectionHandle = NULL;
    }
#endif

    if (Client->ResponsePayload.Buffer && Client->Allocator) {
        Client->Allocator->Vtbl->FreePointer(
            Client->Allocator,
            (PVOID *)&Client->ResponsePayload.Buffer
        );
        Client->ResponsePayload.Length = 0;
        Client->ResponsePayload.MaximumLength = 0;
        Client->ResponsePayloadBufferSize = 0;
        Client->ResponseFlags = 0;
    }

    RELEASE(Client->Allocator);
    RELEASE(Client->Rtl);
}

PERFECT_HASH_CLIENT_CONNECT PerfectHashClientConnect;

_Use_decl_annotations_
HRESULT
PerfectHashClientConnect(
    PPERFECT_HASH_CLIENT Client,
    PCUNICODE_STRING Endpoint
    )
{
#ifndef PH_WINDOWS
    UNREFERENCED_PARAMETER(Client);
    UNREFERENCED_PARAMETER(Endpoint);
    return E_NOTIMPL;
#else
    BOOL Success;
    HANDLE PipeHandle;
    DWORD Mode;
    DWORD LastError;
    PCUNICODE_STRING TargetEndpoint;

    if (!ARGUMENT_PRESENT(Client)) {
        return E_POINTER;
    }

    if (Client->ConnectionHandle) {
        CloseHandle(Client->ConnectionHandle);
        Client->ConnectionHandle = NULL;
    }

    TargetEndpoint = Endpoint ? Endpoint : &PerfectHashClientDefaultPipeName;

    while (TRUE) {
        PipeHandle = CreateFileW(TargetEndpoint->Buffer,
                                 GENERIC_READ | GENERIC_WRITE,
                                 0,
                                 NULL,
                                 OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL,
                                 NULL);

        if (PipeHandle != INVALID_HANDLE_VALUE) {
            break;
        }

        LastError = GetLastError();
        if (LastError != ERROR_PIPE_BUSY) {
            return HRESULT_FROM_WIN32(LastError);
        }

        Success = WaitNamedPipeW(TargetEndpoint->Buffer,
                                 NMPWAIT_WAIT_FOREVER);
        if (!Success) {
            return HRESULT_FROM_WIN32(GetLastError());
        }
    }

    Mode = PIPE_READMODE_BYTE;
    Success = SetNamedPipeHandleState(PipeHandle,
                                      &Mode,
                                      NULL,
                                      NULL);
    if (!Success) {
        CloseHandle(PipeHandle);
        return HRESULT_FROM_WIN32(GetLastError());
    }

    Client->ConnectionHandle = PipeHandle;
    Client->Endpoint = *TargetEndpoint;
    Client->State.Connected = TRUE;
    Client->State.Disconnected = FALSE;

    return S_OK;
#endif
}

PERFECT_HASH_CLIENT_DISCONNECT PerfectHashClientDisconnect;

_Use_decl_annotations_
HRESULT
PerfectHashClientDisconnect(
    PPERFECT_HASH_CLIENT Client
    )
{
    if (!ARGUMENT_PRESENT(Client)) {
        return E_POINTER;
    }

#ifdef PH_WINDOWS
    if (Client->ConnectionHandle) {
        CloseHandle(Client->ConnectionHandle);
        Client->ConnectionHandle = NULL;
    }
#endif

    if (Client->ResponsePayload.Buffer && Client->Allocator) {
        Client->Allocator->Vtbl->FreePointer(
            Client->Allocator,
            (PVOID *)&Client->ResponsePayload.Buffer
        );
        Client->ResponsePayload.Length = 0;
        Client->ResponsePayload.MaximumLength = 0;
        Client->ResponsePayloadBufferSize = 0;
        Client->ResponseFlags = 0;
    }

    Client->State.Disconnected = TRUE;

    return S_OK;
}

PERFECT_HASH_CLIENT_SUBMIT_REQUEST PerfectHashClientSubmitRequest;

_Use_decl_annotations_
HRESULT
PerfectHashClientSubmitRequest(
    PPERFECT_HASH_CLIENT Client,
    PPERFECT_HASH_SERVER_REQUEST Request
    )
{
#ifndef PH_WINDOWS
    UNREFERENCED_PARAMETER(Client);
    UNREFERENCED_PARAMETER(Request);
    return E_NOTIMPL;
#else
    HRESULT Result;
    ULONG PayloadLength;
    HANDLE Handle;
    WCHAR Terminator;
    PRTL Rtl;
    PALLOCATOR Allocator;
    PERFECT_HASH_SERVER_REQUEST_HEADER RequestHeader;
    PERFECT_HASH_SERVER_RESPONSE_HEADER ResponseHeader;

    if (!ARGUMENT_PRESENT(Client)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Request)) {
        return E_POINTER;
    }

    if (!IsValidPerfectHashServerRequestType(Request->RequestType)) {
        return E_INVALIDARG;
    }

    Handle = Client->ConnectionHandle;
    if (!Handle) {
        return E_UNEXPECTED;
    }

    Rtl = Client->Rtl;
    if (!Rtl) {
        return E_UNEXPECTED;
    }

    Allocator = Client->Allocator;
    if (!Allocator) {
        return E_UNEXPECTED;
    }

    if (Client->ResponsePayload.Buffer) {
        Allocator->Vtbl->FreePointer(
            Allocator,
            (PVOID *)&Client->ResponsePayload.Buffer
        );
        Client->ResponsePayload.Length = 0;
        Client->ResponsePayload.MaximumLength = 0;
        Client->ResponsePayloadBufferSize = 0;
        Client->ResponseFlags = 0;
    }

    PayloadLength = 0;
    if (Request->CommandLine.Length > 0) {
        if (!Request->CommandLine.Buffer) {
            return E_INVALIDARG;
        }
        if (Request->CommandLine.Length >
            (PERFECT_HASH_SERVER_MAX_MESSAGE_SIZE - sizeof(WCHAR))) {
            return E_INVALIDARG;
        }
        PayloadLength = Request->CommandLine.Length + sizeof(WCHAR);
    }

    ZeroMemory(&RequestHeader, sizeof(RequestHeader));
    RequestHeader.SizeOfStruct = sizeof(RequestHeader);
    RequestHeader.Version = PERFECT_HASH_SERVER_MESSAGE_VERSION;
    RequestHeader.RequestType = Request->RequestType;
    RequestHeader.RequestId = Request->RequestId;
    RequestHeader.PayloadLength = PayloadLength;

    Result = PerfectHashClientWriteAll(Handle,
                                       &RequestHeader,
                                       sizeof(RequestHeader));
    if (FAILED(Result)) {
        return Result;
    }

    if (PayloadLength > 0) {
        Result = PerfectHashClientWriteAll(Handle,
                                           Request->CommandLine.Buffer,
                                           Request->CommandLine.Length);
        if (FAILED(Result)) {
            return Result;
        }

        Terminator = L'\0';
        Result = PerfectHashClientWriteAll(Handle,
                                           &Terminator,
                                           sizeof(Terminator));
        if (FAILED(Result)) {
            return Result;
        }
    }

    Result = PerfectHashClientReadAll(Handle,
                                      &ResponseHeader,
                                      sizeof(ResponseHeader));
    if (FAILED(Result)) {
        return Result;
    }

    if (ResponseHeader.SizeOfStruct != sizeof(ResponseHeader) ||
        ResponseHeader.Version != PERFECT_HASH_SERVER_MESSAGE_VERSION) {
        return E_UNEXPECTED;
    }

    Client->ResponseFlags = ResponseHeader.Flags;

    if (ResponseHeader.PayloadLength > 0) {
        PVOID PayloadBuffer;

        PayloadLength = ResponseHeader.PayloadLength;

        if (PayloadLength > PERFECT_HASH_SERVER_MAX_MESSAGE_SIZE) {
            return E_INVALIDARG;
        }

        if (PayloadLength > (ULONG)USHRT_MAX) {
            return E_INVALIDARG;
        }

        PayloadBuffer = Allocator->Vtbl->Calloc(Allocator,
                                                1,
                                                PayloadLength);
        if (!PayloadBuffer) {
            return E_OUTOFMEMORY;
        }

        Result = PerfectHashClientReadAll(Handle,
                                          PayloadBuffer,
                                          PayloadLength);
        if (FAILED(Result)) {
            Allocator->Vtbl->FreePointer(Allocator, &PayloadBuffer);
            return Result;
        }

        Client->ResponsePayload.Buffer = (PWSTR)PayloadBuffer;
        Client->ResponsePayload.Length = (USHORT)(
            PayloadLength >= sizeof(WCHAR) ?
            PayloadLength - sizeof(WCHAR) :
            0
        );
        Client->ResponsePayload.MaximumLength = (USHORT)PayloadLength;
        Client->ResponsePayloadBufferSize = PayloadLength;
    }

    return ResponseHeader.Result;
#endif
}

PERFECT_HASH_CLIENT_GET_LAST_RESPONSE PerfectHashClientGetLastResponse;

_Use_decl_annotations_
HRESULT
PerfectHashClientGetLastResponse(
    PPERFECT_HASH_CLIENT Client,
    PUNICODE_STRING ResponsePayload,
    PULONG ResponseFlags
    )
{
    if (!ARGUMENT_PRESENT(Client)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(ResponsePayload)) {
        return E_POINTER;
    }

    ResponsePayload->Buffer = Client->ResponsePayload.Buffer;
    ResponsePayload->Length = Client->ResponsePayload.Length;
    ResponsePayload->MaximumLength = Client->ResponsePayload.MaximumLength;

    if (ARGUMENT_PRESENT(ResponseFlags)) {
        *ResponseFlags = Client->ResponseFlags;
    }

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
