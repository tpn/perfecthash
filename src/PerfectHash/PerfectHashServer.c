/*++

Copyright (c) 2018-2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashServer.c

Abstract:

    This module implements the PERFECT_HASH_SERVER component.  The initial
    implementation provides configuration plumbing and scaffolding for the
    IOCP runtime and request processing.

--*/

#include "stdafx.h"
#include "PerfectHashTls.h"
#include <limits.h>
#include <stdio.h>

#ifdef PH_WINDOWS

static const UNICODE_STRING PerfectHashServerDefaultPipeName =
    RTL_CONSTANT_STRING(L"\\\\.\\pipe\\PerfectHashServer");

typedef enum _PERFECT_HASH_SERVER_PIPE_STATE {
    PerfectHashServerPipeStateInvalid = 0,
    PerfectHashServerPipeStateAccepting,
    PerfectHashServerPipeStateReadingHeader,
    PerfectHashServerPipeStateReadingPayload,
    PerfectHashServerPipeStateWritingResponseHeader,
    PerfectHashServerPipeStateWritingResponsePayload
} PERFECT_HASH_SERVER_PIPE_STATE;

typedef struct _PERFECT_HASH_SERVER_PIPE {
    PERFECT_HASH_IOCP_WORK Iocp;
    PPERFECT_HASH_SERVER Server;
    PPERFECT_HASH_IOCP_NODE Node;
    HANDLE Pipe;
    PERFECT_HASH_SERVER_PIPE_STATE State;
    ULONG Flags;
    ULONG BytesTransferred;
    ULONG PayloadLength;
    BOOLEAN ShutdownAfterSend;
    UCHAR Padding1[7];
    PERFECT_HASH_SERVER_REQUEST_HEADER RequestHeader;
    PERFECT_HASH_SERVER_RESPONSE_HEADER ResponseHeader;
    PVOID PayloadBuffer;
    ULONG PayloadBufferSize;
    ULONG Padding2;
} PERFECT_HASH_SERVER_PIPE;
typedef PERFECT_HASH_SERVER_PIPE *PPERFECT_HASH_SERVER_PIPE;

typedef struct _PERFECT_HASH_SERVER_BULK_REQUEST {
    ULONG SizeOfStruct;
    volatile LONG OutstandingWorkItems;
    volatile LONG FailedWorkItems;
    volatile LONG PendingNodes;
    volatile LONG CompletionSignaled;
    ULONG TotalWorkItems;
    ULONG PerFileMaximumConcurrency;
    ULONG Padding1;
    PPERFECT_HASH_SERVER Server;
    LPWSTR *ArgvW;
    PWSTR CommandLineBuffer;
    PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags;
    PERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags;
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags;
    ULONG Padding2;
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags;
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags;
    ULONG Padding3;
    PERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters;
    PERFECT_HASH_ALGORITHM_ID AlgorithmId;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;
    ULONG Padding4;
    UNICODE_STRING BaseOutputDirectory;
    PWSTR BaseOutputDirectoryBuffer;
    HANDLE CompletionEvent;
    HANDLE ResultMappingHandle;
    PPERFECT_HASH_SERVER_BULK_RESULT ResultMapping;
    volatile LONG FirstFailure;
    volatile LONG DispatchComplete;
    PPERFECT_HASH_IOCP_NODE Nodes;
    ULONG NodeCount;
    ULONG NextNodeIndex;
    PLONG NodeOutstandingCounts;
} PERFECT_HASH_SERVER_BULK_REQUEST;
typedef PERFECT_HASH_SERVER_BULK_REQUEST *PPERFECT_HASH_SERVER_BULK_REQUEST;

typedef struct _PERFECT_HASH_SERVER_BULK_WORK_ITEM {
    PERFECT_HASH_IOCP_WORK Iocp;
    PPERFECT_HASH_SERVER_BULK_REQUEST Request;
    PPERFECT_HASH_IOCP_NODE Node;
    ULONG NodeIndex;
    ULONG Padding1;
    UNICODE_STRING KeysPath;
    PWSTR KeysPathBuffer;
    ULONG LastError;
    ULONG Padding2;
} PERFECT_HASH_SERVER_BULK_WORK_ITEM;
typedef PERFECT_HASH_SERVER_BULK_WORK_ITEM *PPERFECT_HASH_SERVER_BULK_WORK_ITEM;

static
HRESULT
PerfectHashServerCreatePipes(
    _In_ PPERFECT_HASH_SERVER Server
    );

static
VOID
PerfectHashServerDestroyPipes(
    _In_ PPERFECT_HASH_SERVER Server
    );

static
HRESULT
PerfectHashServerIssueConnect(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe
    );

static
HRESULT
PerfectHashServerIssueReadHeader(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe
    );

static
HRESULT
PerfectHashServerIssueReadPayload(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe
    );

static
HRESULT
PerfectHashServerIssueWriteResponseHeader(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe
    );

static
HRESULT
PerfectHashServerIssueWriteResponsePayload(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe
    );

static
VOID
PerfectHashServerResetPipe(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe,
    _In_ BOOLEAN Reconnect
    );

static
HRESULT
PerfectHashServerDispatchRequest(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe
    );

static
HRESULT
PerfectHashServerDispatchBulkCreateDirectoryRequest(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe,
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW,
    _In_ LPWSTR CommandLine,
    _Out_ PBOOLEAN ArgvWOwned
    );

static
HRESULT
PerfectHashServerDispatchTableCreateRequest(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe,
    _In_ ULONG NumberOfArguments,
    _In_ LPWSTR *ArgvW,
    _In_ LPWSTR CommandLine
    );

static
HRESULT
PerfectHashServerPrepareErrorPayload(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe,
    _In_ HRESULT Error
    );

static
HRESULT
PerfectHashServerPreparePingPayload(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe
    );

static
HRESULT
PerfectHashServerPrepareBulkCreateDirectoryPayload(
    _In_ PPERFECT_HASH_SERVER_PIPE Pipe,
    _In_ HANDLE EventHandle,
    _In_ HANDLE ResultHandle
    );

static
HRESULT
PerfectHashServerEnqueueBulkRequest(
    _In_ PPERFECT_HASH_SERVER_BULK_REQUEST Request,
    _In_ PUNICODE_STRING KeysDirectory
    );

static
VOID
PerfectHashServerCompleteBulkRequest(
    _In_ PPERFECT_HASH_SERVER_BULK_REQUEST Request
    );

static
HRESULT
PerfectHashServerBulkCreateWorkItemCallback(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    _In_ ULONG_PTR CompletionKey,
    _In_ LPOVERLAPPED Overlapped,
    _In_ DWORD NumberOfBytesTransferred,
    _In_ BOOL Success
    );

static
VOID
PerfectHashServerLogBulkCreateException(
    _In_ ULONG Stage,
    _In_opt_ PPERFECT_HASH_SERVER_BULK_WORK_ITEM WorkItem,
    _In_ struct _EXCEPTION_POINTERS *ExceptionPointers
    );

static
VOID
PerfectHashServerLogBulkCreateCounts(
    _In_ ULONG Stage,
    _In_opt_ PPERFECT_HASH_SERVER_BULK_WORK_ITEM WorkItem,
    _In_opt_ PPERFECT_HASH_SERVER_BULK_REQUEST Request,
    _In_ LONG Outstanding
    );

static
LONG
PerfectHashServerBulkCreateExceptionFilter(
    _In_ ULONG Stage,
    _In_opt_ PPERFECT_HASH_SERVER_BULK_WORK_ITEM WorkItem,
    _In_ struct _EXCEPTION_POINTERS *ExceptionPointers
    );

static
HRESULT
PerfectHashServerIocpCompletionCallback(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    _In_ ULONG_PTR CompletionKey,
    _In_ LPOVERLAPPED Overlapped,
    _In_ DWORD NumberOfBytesTransferred,
    _In_ BOOL Success
    );

#endif

PERFECT_HASH_SERVER_INITIALIZE PerfectHashServerInitialize;

_Use_decl_annotations_
HRESULT
PerfectHashServerInitialize(
    PPERFECT_HASH_SERVER Server
    )
/*++

Routine Description:

    Initializes a PERFECT_HASH_SERVER instance.  This routine creates the
    Rtl, Allocator, and IOCP context components and initializes default
    configuration values.

Arguments:

    Server - Supplies a pointer to a PERFECT_HASH_SERVER structure for which
        initialization is to be performed.

Return Value:

    S_OK on success, an appropriate error code on failure.

--*/
{
#ifndef PH_WINDOWS
    UNREFERENCED_PARAMETER(Server);
    return E_NOTIMPL;
#else
    HRESULT Result = E_UNEXPECTED;
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp;

    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    Result = Server->Vtbl->CreateInstance(Server,
                                          NULL,
                                          &IID_PERFECT_HASH_RTL,
                                          &Server->Rtl);
    if (FAILED(Result)) {
        return Result;
    }

    Result = Server->Vtbl->CreateInstance(Server,
                                          NULL,
                                          &IID_PERFECT_HASH_ALLOCATOR,
                                          &Server->Allocator);
    if (FAILED(Result)) {
        return Result;
    }

    Result = Server->Vtbl->CreateInstance(Server,
                                          NULL,
                                          &IID_PERFECT_HASH_CONTEXT_IOCP,
                                          &Server->ContextIocp);
    if (FAILED(Result)) {
        return Result;
    }

    ContextIocp = Server->ContextIocp;

    Server->IocpConcurrency = ContextIocp->IocpConcurrency;
    Server->MaxWorkerThreads = ContextIocp->MaxWorkerThreads;
    Server->NumaNodeMask = ContextIocp->NumaNodeMask;
    Server->NumaNodeCount = ContextIocp->NumaNodeCount;

    Server->StartedEvent = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!Server->StartedEvent) {
        return PH_E_SYSTEM_CALL_FAILED;
    }

    Server->ShutdownEvent = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!Server->ShutdownEvent) {
        return PH_E_SYSTEM_CALL_FAILED;
    }

    Server->Endpoint = PerfectHashServerDefaultPipeName;
    Server->Flags.LocalOnly = TRUE;
    Server->Flags.EndpointAllocated = FALSE;
    Server->Flags.Verbose = FALSE;
    Server->Flags.NoFileIo = FALSE;
    Server->State.Initialized = TRUE;

    return S_OK;
#endif
}

PERFECT_HASH_SERVER_RUNDOWN PerfectHashServerRundown;

_Use_decl_annotations_
VOID
PerfectHashServerRundown(
    PPERFECT_HASH_SERVER Server
    )
/*++

Routine Description:

    Releases resources associated with a PERFECT_HASH_SERVER instance.

Arguments:

    Server - Supplies a pointer to a PERFECT_HASH_SERVER structure for which
        rundown is to be performed.

Return Value:

    None.

--*/
{
    if (!ARGUMENT_PRESENT(Server)) {
        return;
    }

    Server->State.Stopping = TRUE;

    if (Server->ContextIocp) {
#ifdef PH_WINDOWS
        PerfectHashServerDestroyPipes(Server);
#endif
        PerfectHashContextIocpStop(Server->ContextIocp);
    }

#ifdef PH_WINDOWS
    if (Server->ShutdownEvent) {
        CloseHandle(Server->ShutdownEvent);
        Server->ShutdownEvent = NULL;
    }
    if (Server->StartedEvent) {
        CloseHandle(Server->StartedEvent);
        Server->StartedEvent = NULL;
    }
#endif

    if (Server->Flags.EndpointAllocated && Server->Endpoint.Buffer) {
        if (Server->Allocator) {
            Server->Allocator->Vtbl->FreePointer(
                Server->Allocator,
                (PVOID *)&Server->Endpoint.Buffer
            );
        }
        Server->Endpoint.Length = 0;
        Server->Endpoint.MaximumLength = 0;
        Server->Flags.EndpointAllocated = FALSE;
    }

    RELEASE(Server->ContextIocp);
    RELEASE(Server->Allocator);
    RELEASE(Server->Rtl);
}

PERFECT_HASH_SERVER_SET_MAXIMUM_CONCURRENCY
    PerfectHashServerSetMaximumConcurrency;

_Use_decl_annotations_
HRESULT
PerfectHashServerSetMaximumConcurrency(
    PPERFECT_HASH_SERVER Server,
    ULONG MaximumConcurrency
    )
{
    HRESULT Result = E_UNEXPECTED;

    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (MaximumConcurrency == 0) {
        return E_INVALIDARG;
    }

    if (Server->ContextIocp) {
        Result = Server->ContextIocp->Vtbl->SetMaximumConcurrency(
            Server->ContextIocp,
            MaximumConcurrency
        );
        if (FAILED(Result)) {
            return Result;
        }
    }

    Server->IocpConcurrency = MaximumConcurrency;
    return S_OK;
}

PERFECT_HASH_SERVER_GET_MAXIMUM_CONCURRENCY
    PerfectHashServerGetMaximumConcurrency;

_Use_decl_annotations_
HRESULT
PerfectHashServerGetMaximumConcurrency(
    PPERFECT_HASH_SERVER Server,
    PULONG MaximumConcurrency
    )
{
    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(MaximumConcurrency)) {
        return E_POINTER;
    }

    *MaximumConcurrency = Server->IocpConcurrency;
    return S_OK;
}

PERFECT_HASH_SERVER_SET_MAXIMUM_THREADS
    PerfectHashServerSetMaximumThreads;

_Use_decl_annotations_
HRESULT
PerfectHashServerSetMaximumThreads(
    PPERFECT_HASH_SERVER Server,
    ULONG MaximumThreads
    )
{
    HRESULT Result = E_UNEXPECTED;

    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (MaximumThreads == 0) {
        return E_INVALIDARG;
    }

    if (Server->ContextIocp) {
        Result = Server->ContextIocp->Vtbl->SetMaximumThreads(
            Server->ContextIocp,
            MaximumThreads
        );
        if (FAILED(Result)) {
            return Result;
        }
    }

    Server->MaxWorkerThreads = MaximumThreads;
    return S_OK;
}

PERFECT_HASH_SERVER_GET_MAXIMUM_THREADS
    PerfectHashServerGetMaximumThreads;

_Use_decl_annotations_
HRESULT
PerfectHashServerGetMaximumThreads(
    PPERFECT_HASH_SERVER Server,
    PULONG MaximumThreads
    )
{
    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(MaximumThreads)) {
        return E_POINTER;
    }

    *MaximumThreads = Server->MaxWorkerThreads;
    return S_OK;
}

PERFECT_HASH_SERVER_SET_NUMA_NODE_MASK PerfectHashServerSetNumaNodeMask;

_Use_decl_annotations_
HRESULT
PerfectHashServerSetNumaNodeMask(
    PPERFECT_HASH_SERVER Server,
    PERFECT_HASH_NUMA_NODE_MASK NumaNodeMask
    )
{
    HRESULT Result;

    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (Server->ContextIocp) {
        Result = Server->ContextIocp->Vtbl->SetNumaNodeMask(
            Server->ContextIocp,
            NumaNodeMask
        );
        if (FAILED(Result)) {
            return Result;
        }
    }

    Server->NumaNodeMask = NumaNodeMask;
    return S_OK;
}

PERFECT_HASH_SERVER_GET_NUMA_NODE_MASK PerfectHashServerGetNumaNodeMask;

_Use_decl_annotations_
HRESULT
PerfectHashServerGetNumaNodeMask(
    PPERFECT_HASH_SERVER Server,
    PPERFECT_HASH_NUMA_NODE_MASK NumaNodeMask
    )
{
    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(NumaNodeMask)) {
        return E_POINTER;
    }

    *NumaNodeMask = Server->NumaNodeMask;
    return S_OK;
}

PERFECT_HASH_SERVER_SET_ENDPOINT PerfectHashServerSetEndpoint;

_Use_decl_annotations_
HRESULT
PerfectHashServerSetEndpoint(
    PPERFECT_HASH_SERVER Server,
    PCUNICODE_STRING Endpoint
    )
{
    PALLOCATOR Allocator = NULL;
    PRTL Rtl = NULL;
    PWSTR Buffer;
    ULONG BytesToAllocate;

    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Endpoint)) {
        return E_POINTER;
    }

    if (Server->State.Running) {
        return E_UNEXPECTED;
    }

    if (!IsValidNullTerminatedUnicodeString(Endpoint)) {
        return E_INVALIDARG;
    }

    Allocator = Server->Allocator;
    Rtl = Server->Rtl;

    if (!Allocator || !Rtl) {
        return E_UNEXPECTED;
    }

    BytesToAllocate = Endpoint->Length + sizeof(WCHAR);
    if (BytesToAllocate < Endpoint->Length) {
        return E_INVALIDARG;
    }
    if (BytesToAllocate > (ULONG)USHRT_MAX) {
        return E_INVALIDARG;
    }

    Buffer = (PWSTR)Allocator->Vtbl->Calloc(Allocator, 1, BytesToAllocate);
    if (!Buffer) {
        return E_OUTOFMEMORY;
    }

    CopyMemory(Buffer, Endpoint->Buffer, Endpoint->Length);

    if (Server->Flags.EndpointAllocated && Server->Endpoint.Buffer) {
        Allocator->Vtbl->FreePointer(Allocator,
                                     (PVOID *)&Server->Endpoint.Buffer);
    }

    Server->Endpoint.Buffer = Buffer;
    Server->Endpoint.Length = Endpoint->Length;
    Server->Endpoint.MaximumLength = (USHORT)BytesToAllocate;
    Server->Flags.EndpointAllocated = TRUE;

    return S_OK;
}

PERFECT_HASH_SERVER_GET_ENDPOINT PerfectHashServerGetEndpoint;

_Use_decl_annotations_
HRESULT
PerfectHashServerGetEndpoint(
    PPERFECT_HASH_SERVER Server,
    PUNICODE_STRING Endpoint
    )
{
    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Endpoint)) {
        return E_POINTER;
    }

    *Endpoint = Server->Endpoint;
    return S_OK;
}

PERFECT_HASH_SERVER_SET_LOCAL_ONLY PerfectHashServerSetLocalOnly;

_Use_decl_annotations_
HRESULT
PerfectHashServerSetLocalOnly(
    PPERFECT_HASH_SERVER Server,
    BOOLEAN LocalOnly
    )
{
    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (Server->State.Running) {
        return E_UNEXPECTED;
    }

    Server->Flags.LocalOnly = LocalOnly ? 1 : 0;
    return S_OK;
}

PERFECT_HASH_SERVER_GET_LOCAL_ONLY PerfectHashServerGetLocalOnly;

_Use_decl_annotations_
HRESULT
PerfectHashServerGetLocalOnly(
    PPERFECT_HASH_SERVER Server,
    PBOOLEAN LocalOnly
    )
{
    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(LocalOnly)) {
        return E_POINTER;
    }

    *LocalOnly = (Server->Flags.LocalOnly != 0);
    return S_OK;
}

PERFECT_HASH_SERVER_SET_VERBOSE PerfectHashServerSetVerbose;

_Use_decl_annotations_
HRESULT
PerfectHashServerSetVerbose(
    PPERFECT_HASH_SERVER Server,
    BOOLEAN Verbose
    )
{
    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (Server->State.Running) {
        return E_UNEXPECTED;
    }

    Server->Flags.Verbose = Verbose ? 1 : 0;
    return S_OK;
}

PERFECT_HASH_SERVER_GET_VERBOSE PerfectHashServerGetVerbose;

_Use_decl_annotations_
HRESULT
PerfectHashServerGetVerbose(
    PPERFECT_HASH_SERVER Server,
    PBOOLEAN Verbose
    )
{
    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Verbose)) {
        return E_POINTER;
    }

    *Verbose = (Server->Flags.Verbose != 0);
    return S_OK;
}

PERFECT_HASH_SERVER_SET_NO_FILE_IO PerfectHashServerSetNoFileIo;

_Use_decl_annotations_
HRESULT
PerfectHashServerSetNoFileIo(
    PPERFECT_HASH_SERVER Server,
    BOOLEAN NoFileIo
    )
{
    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (Server->State.Running) {
        return E_UNEXPECTED;
    }

    Server->Flags.NoFileIo = NoFileIo ? 1 : 0;
    return S_OK;
}

PERFECT_HASH_SERVER_GET_NO_FILE_IO PerfectHashServerGetNoFileIo;

_Use_decl_annotations_
HRESULT
PerfectHashServerGetNoFileIo(
    PPERFECT_HASH_SERVER Server,
    PBOOLEAN NoFileIo
    )
{
    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(NoFileIo)) {
        return E_POINTER;
    }

    *NoFileIo = (Server->Flags.NoFileIo != 0);
    return S_OK;
}

PERFECT_HASH_SERVER_START PerfectHashServerStart;

_Use_decl_annotations_
HRESULT
PerfectHashServerStart(
    PPERFECT_HASH_SERVER Server
    )
{
#ifndef PH_WINDOWS
    UNREFERENCED_PARAMETER(Server);
    return E_NOTIMPL;
#else
    HRESULT Result;

    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (Server->ContextIocp) {
        Result = PerfectHashContextIocpStart(Server->ContextIocp);
        if (FAILED(Result)) {
            return Result;
        }
    }

    Result = PerfectHashServerCreatePipes(Server);
    if (FAILED(Result)) {
        PerfectHashContextIocpStop(Server->ContextIocp);
        return Result;
    }

    Server->State.Running = TRUE;
    if (Server->StartedEvent) {
        SetEvent(Server->StartedEvent);
    }

    return S_OK;
#endif
}

PERFECT_HASH_SERVER_STOP PerfectHashServerStop;

_Use_decl_annotations_
HRESULT
PerfectHashServerStop(
    PPERFECT_HASH_SERVER Server
    )
{
#ifndef PH_WINDOWS
    UNREFERENCED_PARAMETER(Server);
    return E_NOTIMPL;
#else
    HRESULT Result;

    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    Server->State.Stopping = TRUE;

    if (Server->ContextIocp) {
        PerfectHashServerDestroyPipes(Server);
        Result = PerfectHashContextIocpStop(Server->ContextIocp);
        if (FAILED(Result)) {
            return Result;
        }
    }

    if (Server->ShutdownEvent) {
        SetEvent(Server->ShutdownEvent);
    }

    Server->State.Stopped = TRUE;
    return S_OK;
#endif
}

PERFECT_HASH_SERVER_WAIT PerfectHashServerWait;

_Use_decl_annotations_
HRESULT
PerfectHashServerWait(
    PPERFECT_HASH_SERVER Server
    )
{
#ifndef PH_WINDOWS
    UNREFERENCED_PARAMETER(Server);
    return E_NOTIMPL;
#else
    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (Server->ShutdownEvent) {
        WaitForSingleObject(Server->ShutdownEvent, INFINITE);
    }

    return S_OK;
#endif
}

PERFECT_HASH_SERVER_SUBMIT_REQUEST PerfectHashServerSubmitRequest;

_Use_decl_annotations_
HRESULT
PerfectHashServerSubmitRequest(
    PPERFECT_HASH_SERVER Server,
    PPERFECT_HASH_SERVER_REQUEST Request
    )
{
    UNREFERENCED_PARAMETER(Server);
    UNREFERENCED_PARAMETER(Request);

    return E_NOTIMPL;
}

#ifdef PH_WINDOWS

static
HRESULT
PerfectHashServerCreatePipes(
    PPERFECT_HASH_SERVER Server
    )
{
    ULONG NodeIndex;
    ULONG PipeIndex;
    ULONG PipeCount;
    HRESULT Result;
    PALLOCATOR Allocator;
    PRTL Rtl;
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp;

    if (!ARGUMENT_PRESENT(Server)) {
        return E_POINTER;
    }

    if (Server->Pipes) {
        return S_OK;
    }

    ContextIocp = Server->ContextIocp;
    Allocator = Server->Allocator;
    Rtl = Server->Rtl;

    if (!ContextIocp || !Allocator || !Rtl) {
        return E_UNEXPECTED;
    }

    PipeCount = ContextIocp->TotalWorkerThreadCount;
    if (PipeCount == 0) {
        PipeCount = 1;
    }

    Server->Pipes = (PPERFECT_HASH_SERVER_PIPE)(
        Allocator->Vtbl->Calloc(Allocator,
                                PipeCount,
                                sizeof(*Server->Pipes))
    );
    if (!Server->Pipes) {
        return E_OUTOFMEMORY;
    }

    Server->PipeCount = PipeCount;
    PipeIndex = 0;

    for (NodeIndex = 0; NodeIndex < ContextIocp->NodeCount; NodeIndex++) {
        PPERFECT_HASH_IOCP_NODE Node = &ContextIocp->Nodes[NodeIndex];
        ULONG InstanceIndex;

        for (InstanceIndex = 0;
             InstanceIndex < Node->WorkerThreadCount;
             InstanceIndex++) {
            DWORD OpenMode;
            DWORD PipeMode;
            HANDLE PipeHandle;
            HANDLE PortHandle;
            PPERFECT_HASH_SERVER_PIPE Pipe;

            if (PipeIndex >= Server->PipeCount) {
                break;
            }

            Pipe = &Server->Pipes[PipeIndex++];
            Pipe->Iocp.Signature = PH_IOCP_WORK_SIGNATURE;
            Pipe->Iocp.Flags = PH_IOCP_WORK_FLAG_PIPE;
            Pipe->Iocp.CompletionCallback =
                PerfectHashServerIocpCompletionCallback;
            Pipe->Iocp.CompletionContext = Pipe;
            Pipe->Server = Server;
            Pipe->Node = Node;

            OpenMode = PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED;
            PipeMode = PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT;

#ifdef PIPE_REJECT_REMOTE_CLIENTS
            if (Server->Flags.LocalOnly) {
                PipeMode |= PIPE_REJECT_REMOTE_CLIENTS;
            }
#endif

            PipeHandle = CreateNamedPipeW(Server->Endpoint.Buffer,
                                          OpenMode,
                                          PipeMode,
                                          PIPE_UNLIMITED_INSTANCES,
                                          0,
                                          0,
                                          0,
                                          NULL);
            if (!PipeHandle || PipeHandle == INVALID_HANDLE_VALUE) {
                SYS_ERROR(CreateNamedPipeW);
                Result = PH_E_SYSTEM_CALL_FAILED;
                goto Error;
            }

            Pipe->Pipe = PipeHandle;

            PortHandle = CreateIoCompletionPort(PipeHandle,
                                                Node->IoCompletionPort,
                                                (ULONG_PTR)Pipe,
                                                0);
            if (!PortHandle) {
                SYS_ERROR(CreateIoCompletionPort);
                Result = PH_E_SYSTEM_CALL_FAILED;
                goto Error;
            }

            Result = PerfectHashServerIssueConnect(Pipe);
            if (FAILED(Result)) {
                goto Error;
            }
        }
    }

    return S_OK;

Error:

    PerfectHashServerDestroyPipes(Server);
    return Result;
}

static
VOID
PerfectHashServerDestroyPipes(
    PPERFECT_HASH_SERVER Server
    )
{
    ULONG Index;
    PALLOCATOR Allocator;

    if (!ARGUMENT_PRESENT(Server)) {
        return;
    }

    if (!Server->Pipes) {
        return;
    }

    Allocator = Server->Allocator;

    for (Index = 0; Index < Server->PipeCount; Index++) {
        PPERFECT_HASH_SERVER_PIPE Pipe = &Server->Pipes[Index];

        if (Pipe->Pipe) {
            CancelIoEx(Pipe->Pipe, NULL);
            CloseHandle(Pipe->Pipe);
            Pipe->Pipe = NULL;
        }

        if (Pipe->PayloadBuffer && Allocator) {
            Allocator->Vtbl->FreePointer(Allocator,
                                         &Pipe->PayloadBuffer);
            Pipe->PayloadBufferSize = 0;
        }
    }

    if (Allocator) {
        Allocator->Vtbl->FreePointer(Allocator,
                                     (PVOID *)&Server->Pipes);
    }

    Server->Pipes = NULL;
    Server->PipeCount = 0;
}

static
HRESULT
PerfectHashServerIssueConnect(
    PPERFECT_HASH_SERVER_PIPE Pipe
    )
{
    BOOL Success;
    DWORD LastError;
    PRTL Rtl;

    if (!ARGUMENT_PRESENT(Pipe)) {
        return E_POINTER;
    }

    Rtl = Pipe->Server ? Pipe->Server->Rtl : NULL;
    if (!Rtl) {
        return E_UNEXPECTED;
    }

    ZeroMemory(&Pipe->Iocp.Overlapped, sizeof(Pipe->Iocp.Overlapped));
    Pipe->State = PerfectHashServerPipeStateAccepting;
    Pipe->BytesTransferred = 0;
    Pipe->PayloadLength = 0;
    Pipe->ShutdownAfterSend = FALSE;

    Success = ConnectNamedPipe(Pipe->Pipe, &Pipe->Iocp.Overlapped);
    if (Success) {
        return S_OK;
    }

    LastError = GetLastError();
    if (LastError == ERROR_IO_PENDING) {
        return S_OK;
    }

    if (LastError == ERROR_PIPE_CONNECTED) {
        if (!PostQueuedCompletionStatus(Pipe->Node->IoCompletionPort,
                                        0,
                                        (ULONG_PTR)Pipe,
                                        &Pipe->Iocp.Overlapped)) {
            SYS_ERROR(PostQueuedCompletionStatus);
            return PH_E_SYSTEM_CALL_FAILED;
        }
        return S_OK;
    }

    SYS_ERROR(ConnectNamedPipe);
    return PH_E_SYSTEM_CALL_FAILED;
}

static
HRESULT
PerfectHashServerIssueReadHeader(
    PPERFECT_HASH_SERVER_PIPE Pipe
    )
{
    BOOL Success;
    DWORD LastError;
    ULONG Remaining;
    PUCHAR Buffer;
    PRTL Rtl;

    if (!ARGUMENT_PRESENT(Pipe)) {
        return E_POINTER;
    }

    Rtl = Pipe->Server ? Pipe->Server->Rtl : NULL;
    if (!Rtl) {
        return E_UNEXPECTED;
    }

    Remaining = sizeof(PERFECT_HASH_SERVER_REQUEST_HEADER) -
                Pipe->BytesTransferred;
    Buffer = (PUCHAR)&Pipe->RequestHeader + Pipe->BytesTransferred;

    ZeroMemory(&Pipe->Iocp.Overlapped, sizeof(Pipe->Iocp.Overlapped));
    Pipe->State = PerfectHashServerPipeStateReadingHeader;

    Success = ReadFile(Pipe->Pipe,
                       Buffer,
                       Remaining,
                       NULL,
                       &Pipe->Iocp.Overlapped);
    if (Success) {
        return S_OK;
    }

    LastError = GetLastError();
    if (LastError == ERROR_IO_PENDING) {
        return S_OK;
    }

    return HRESULT_FROM_WIN32(LastError);
}

static
HRESULT
PerfectHashServerIssueReadPayload(
    PPERFECT_HASH_SERVER_PIPE Pipe
    )
{
    BOOL Success;
    DWORD LastError;
    ULONG Remaining;
    PUCHAR Buffer;
    PRTL Rtl;

    if (!ARGUMENT_PRESENT(Pipe)) {
        return E_POINTER;
    }

    Rtl = Pipe->Server ? Pipe->Server->Rtl : NULL;
    if (!Rtl) {
        return E_UNEXPECTED;
    }

    Remaining = Pipe->PayloadLength - Pipe->BytesTransferred;
    Buffer = (PUCHAR)Pipe->PayloadBuffer + Pipe->BytesTransferred;

    ZeroMemory(&Pipe->Iocp.Overlapped, sizeof(Pipe->Iocp.Overlapped));
    Pipe->State = PerfectHashServerPipeStateReadingPayload;

    Success = ReadFile(Pipe->Pipe,
                       Buffer,
                       Remaining,
                       NULL,
                       &Pipe->Iocp.Overlapped);
    if (Success) {
        return S_OK;
    }

    LastError = GetLastError();
    if (LastError == ERROR_IO_PENDING) {
        return S_OK;
    }

    return HRESULT_FROM_WIN32(LastError);
}

static
HRESULT
PerfectHashServerIssueWriteResponseHeader(
    PPERFECT_HASH_SERVER_PIPE Pipe
    )
{
    BOOL Success;
    DWORD LastError;
    ULONG Remaining;
    PUCHAR Buffer;
    PRTL Rtl;

    if (!ARGUMENT_PRESENT(Pipe)) {
        return E_POINTER;
    }

    Rtl = Pipe->Server ? Pipe->Server->Rtl : NULL;
    if (!Rtl) {
        return E_UNEXPECTED;
    }

    Remaining = sizeof(PERFECT_HASH_SERVER_RESPONSE_HEADER) -
                Pipe->BytesTransferred;
    Buffer = (PUCHAR)&Pipe->ResponseHeader + Pipe->BytesTransferred;

    ZeroMemory(&Pipe->Iocp.Overlapped, sizeof(Pipe->Iocp.Overlapped));
    Pipe->State = PerfectHashServerPipeStateWritingResponseHeader;

    Success = WriteFile(Pipe->Pipe,
                        Buffer,
                        Remaining,
                        NULL,
                        &Pipe->Iocp.Overlapped);
    if (Success) {
        return S_OK;
    }

    LastError = GetLastError();
    if (LastError == ERROR_IO_PENDING) {
        return S_OK;
    }

    return HRESULT_FROM_WIN32(LastError);
}

static
HRESULT
PerfectHashServerIssueWriteResponsePayload(
    PPERFECT_HASH_SERVER_PIPE Pipe
    )
{
    BOOL Success;
    DWORD LastError;
    ULONG Remaining;
    PUCHAR Buffer;
    PRTL Rtl;

    if (!ARGUMENT_PRESENT(Pipe)) {
        return E_POINTER;
    }

    Rtl = Pipe->Server ? Pipe->Server->Rtl : NULL;
    if (!Rtl) {
        return E_UNEXPECTED;
    }

    Remaining = Pipe->ResponseHeader.PayloadLength - Pipe->BytesTransferred;
    Buffer = (PUCHAR)Pipe->PayloadBuffer + Pipe->BytesTransferred;

    ZeroMemory(&Pipe->Iocp.Overlapped, sizeof(Pipe->Iocp.Overlapped));
    Pipe->State = PerfectHashServerPipeStateWritingResponsePayload;

    Success = WriteFile(Pipe->Pipe,
                        Buffer,
                        Remaining,
                        NULL,
                        &Pipe->Iocp.Overlapped);
    if (Success) {
        return S_OK;
    }

    LastError = GetLastError();
    if (LastError == ERROR_IO_PENDING) {
        return S_OK;
    }

    return HRESULT_FROM_WIN32(LastError);
}

static
VOID
PerfectHashServerResetPipe(
    PPERFECT_HASH_SERVER_PIPE Pipe,
    BOOLEAN Reconnect
    )
{
    PALLOCATOR Allocator = NULL;
    PRTL Rtl = NULL;
    PPERFECT_HASH_SERVER Server;

    if (!ARGUMENT_PRESENT(Pipe)) {
        return;
    }

    Server = Pipe->Server;
    Allocator = Server ? Server->Allocator : NULL;
    Rtl = Server ? Server->Rtl : NULL;

    Pipe->BytesTransferred = 0;
    Pipe->PayloadLength = 0;
    Pipe->ShutdownAfterSend = FALSE;

    if (Pipe->PayloadBuffer && Allocator) {
        Allocator->Vtbl->FreePointer(Allocator,
                                     &Pipe->PayloadBuffer);
        Pipe->PayloadBufferSize = 0;
    }

    if (Rtl) {
        ZeroMemory(&Pipe->RequestHeader, sizeof(Pipe->RequestHeader));
        ZeroMemory(&Pipe->ResponseHeader, sizeof(Pipe->ResponseHeader));
        ZeroMemory(&Pipe->Iocp.Overlapped, sizeof(Pipe->Iocp.Overlapped));
    }

    if (Pipe->Pipe) {
        DisconnectNamedPipe(Pipe->Pipe);
    }

    if (Reconnect && Server &&
        !Server->State.Stopping &&
        !Server->State.Stopped) {
        PerfectHashServerIssueConnect(Pipe);
    }
}

static
HRESULT
PerfectHashServerPrepareErrorPayload(
    PPERFECT_HASH_SERVER_PIPE Pipe,
    HRESULT Error
    )
{
    BOOL FreeMessage;
    DWORD Flags;
    DWORD Count;
    ULONG PayloadLength;
    ULONG BytesToAllocate;
    PRTL Rtl;
    PALLOCATOR Allocator;
    PWSTR Buffer;
    PWSTR Message;
    WCHAR Fallback[64];

    if (!ARGUMENT_PRESENT(Pipe)) {
        return E_POINTER;
    }

    Rtl = Pipe->Server ? Pipe->Server->Rtl : NULL;
    Allocator = Pipe->Server ? Pipe->Server->Allocator : NULL;

    if (!Rtl || !Allocator) {
        return E_UNEXPECTED;
    }

    FreeMessage = FALSE;
    Message = NULL;
    Fallback[0] = L'\0';

    Flags = (
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_HMODULE |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS
    );

    Count = FormatMessageW(Flags,
                           PerfectHashModule,
                           (DWORD)Error,
                           0,
                           (LPWSTR)&Message,
                           0,
                           NULL);

    if (Count == 0 || !Message) {
        swprintf_s(Fallback,
                   ARRAYSIZE(Fallback),
                   L"Error 0x%08lX",
                   (ULONG)Error);
        Message = Fallback;
        Count = (DWORD)wcslen(Message);
    } else {
        FreeMessage = TRUE;
    }

    while (Count > 0) {
        WCHAR Ch = Message[Count - 1];
        if (Ch == L'\r' || Ch == L'\n' || Ch == L' ' || Ch == L'\t') {
            Message[Count - 1] = L'\0';
            Count--;
            continue;
        }
        break;
    }

    PayloadLength = (Count + 1) * sizeof(WCHAR);
    if (PayloadLength > PERFECT_HASH_SERVER_MAX_MESSAGE_SIZE) {
        ULONG MaxChars;

        MaxChars = (PERFECT_HASH_SERVER_MAX_MESSAGE_SIZE / sizeof(WCHAR));
        if (MaxChars == 0) {
            PayloadLength = 0;
        } else {
            MaxChars -= 1;
            PayloadLength = (MaxChars + 1) * sizeof(WCHAR);
            Message[MaxChars] = L'\0';
            Count = MaxChars;
        }
    }

    if (PayloadLength == 0) {
        if (FreeMessage) {
            LocalFree(Message);
        }
        return E_UNEXPECTED;
    }

    BytesToAllocate = PayloadLength;

    if (!Pipe->PayloadBuffer || Pipe->PayloadBufferSize < BytesToAllocate) {
        if (Pipe->PayloadBuffer) {
            Allocator->Vtbl->FreePointer(Allocator,
                                         &Pipe->PayloadBuffer);
        }

        Pipe->PayloadBuffer = Allocator->Vtbl->Calloc(Allocator,
                                                      1,
                                                      BytesToAllocate);
        if (!Pipe->PayloadBuffer) {
            if (FreeMessage) {
                LocalFree(Message);
            }
            return E_OUTOFMEMORY;
        }

        Pipe->PayloadBufferSize = BytesToAllocate;
    } else {
        ZeroMemory(Pipe->PayloadBuffer, BytesToAllocate);
    }

    Buffer = (PWSTR)Pipe->PayloadBuffer;
    CopyMemory(Buffer, Message, Count * sizeof(WCHAR));
    Buffer[Count] = L'\0';

    Pipe->ResponseHeader.PayloadLength = PayloadLength;
    Pipe->ResponseHeader.Flags |= PERFECT_HASH_SERVER_RESPONSE_FLAG_ERROR_MESSAGE;

    if (FreeMessage) {
        LocalFree(Message);
    }

    return S_OK;
}

static
HRESULT
PerfectHashServerPreparePingPayload(
    PPERFECT_HASH_SERVER_PIPE Pipe
    )
{
    PALLOCATOR Allocator = NULL;
    PRTL Rtl = NULL;
    WCHAR Pong[] = L"PONG";
    ULONG PayloadLength;

    if (!ARGUMENT_PRESENT(Pipe)) {
        return E_POINTER;
    }

    Allocator = Pipe->Server ? Pipe->Server->Allocator : NULL;
    Rtl = Pipe->Server ? Pipe->Server->Rtl : NULL;
    if (!Allocator || !Rtl) {
        return E_UNEXPECTED;
    }

    PayloadLength = (ULONG)sizeof(Pong);
    if (PayloadLength > PERFECT_HASH_SERVER_MAX_MESSAGE_SIZE) {
        return E_INVALIDARG;
    }

    if (!Pipe->PayloadBuffer || Pipe->PayloadBufferSize < PayloadLength) {
        if (Pipe->PayloadBuffer) {
            Allocator->Vtbl->FreePointer(Allocator,
                                         &Pipe->PayloadBuffer);
        }

        Pipe->PayloadBuffer = Allocator->Vtbl->Calloc(Allocator,
                                                      1,
                                                      PayloadLength);
        if (!Pipe->PayloadBuffer) {
            return E_OUTOFMEMORY;
        }

        Pipe->PayloadBufferSize = PayloadLength;
    } else {
        ZeroMemory(Pipe->PayloadBuffer, PayloadLength);
    }

    CopyMemory(Pipe->PayloadBuffer, Pong, PayloadLength);

    Pipe->ResponseHeader.PayloadLength = PayloadLength;
    Pipe->ResponseHeader.Flags |= PERFECT_HASH_SERVER_RESPONSE_FLAG_PONG;

    return S_OK;
}

static
HRESULT
PerfectHashServerPrepareBulkCreateDirectoryPayload(
    PPERFECT_HASH_SERVER_PIPE Pipe,
    HANDLE EventHandle,
    HANDLE ResultHandle
    )
{
    PALLOCATOR Allocator = NULL;
    PRTL Rtl = NULL;
    WCHAR Payload[128];
    int Count;
    ULONG PayloadLength;

    if (!ARGUMENT_PRESENT(Pipe)) {
        return E_POINTER;
    }

    Allocator = Pipe->Server ? Pipe->Server->Allocator : NULL;
    Rtl = Pipe->Server ? Pipe->Server->Rtl : NULL;
    if (!Allocator || !Rtl) {
        return E_UNEXPECTED;
    }

    Count = swprintf_s(Payload,
                       ARRAYSIZE(Payload),
                       PERFECT_HASH_SERVER_BULK_CREATE_TOKEN_FORMAT,
                       (ULONGLONG)(ULONG_PTR)EventHandle,
                       (ULONGLONG)(ULONG_PTR)ResultHandle);
    if (Count <= 0) {
        return E_UNEXPECTED;
    }

    PayloadLength = (ULONG)((Count + 1) * sizeof(WCHAR));
    if (PayloadLength > PERFECT_HASH_SERVER_MAX_MESSAGE_SIZE) {
        return E_INVALIDARG;
    }

    if (!Pipe->PayloadBuffer || Pipe->PayloadBufferSize < PayloadLength) {
        if (Pipe->PayloadBuffer) {
            Allocator->Vtbl->FreePointer(Allocator,
                                         &Pipe->PayloadBuffer);
        }

        Pipe->PayloadBuffer = Allocator->Vtbl->Calloc(Allocator,
                                                      1,
                                                      PayloadLength);
        if (!Pipe->PayloadBuffer) {
            return E_OUTOFMEMORY;
        }

        Pipe->PayloadBufferSize = PayloadLength;
    } else {
        ZeroMemory(Pipe->PayloadBuffer, PayloadLength);
    }

    CopyMemory(Pipe->PayloadBuffer, Payload, PayloadLength);

    Pipe->ResponseHeader.PayloadLength = PayloadLength;
    Pipe->ResponseHeader.Flags |=
        PERFECT_HASH_SERVER_RESPONSE_FLAG_BULK_CREATE_TOKEN;

    return S_OK;
}

static
VOID
PerfectHashServerCompleteBulkRequest(
    PPERFECT_HASH_SERVER_BULK_REQUEST Request
    )
{
    ULONG Failed;
    ULONG Total;
    ULONG Succeeded;
    HRESULT Result;
    HRESULT FirstFailure;
    PPERFECT_HASH_SERVER_BULK_RESULT BulkResult;
    PALLOCATOR Allocator;
    LONG Outstanding;

    if (!ARGUMENT_PRESENT(Request)) {
        return;
    }

    if (InterlockedCompareExchange(&Request->CompletionSignaled, 1, 0) != 0) {
        return;
    }

    Outstanding = Request->OutstandingWorkItems;
    PerfectHashServerLogBulkCreateCounts(3, NULL, Request, Outstanding);

    Failed = (ULONG)Request->FailedWorkItems;
    Total = Request->TotalWorkItems;
    Succeeded = (Total >= Failed) ? (Total - Failed) : 0;
    FirstFailure = (HRESULT)Request->FirstFailure;

    if (Total == 0) {
        Result = (FirstFailure != S_OK) ?
                 FirstFailure :
                 PH_E_NO_KEYS_FOUND_IN_DIRECTORY;
    } else if (Failed == 0) {
        Result = PH_S_SERVER_BULK_CREATE_ALL_SUCCEEDED;
    } else {
        Result = PH_E_SERVER_BULK_CREATE_FAILED;
    }

    BulkResult = Request->ResultMapping;
    if (BulkResult) {
        BulkResult->SizeOfStruct = sizeof(*BulkResult);
        BulkResult->Version = PERFECT_HASH_SERVER_BULK_RESULT_VERSION;
        BulkResult->Result = Result;
        BulkResult->Flags = 0;
        BulkResult->TotalFiles = Total;
        BulkResult->SucceededFiles = Succeeded;
        BulkResult->FailedFiles = Failed;
        BulkResult->FirstFailure = FirstFailure;
    }

    if (Request->CompletionEvent) {
        SetEvent(Request->CompletionEvent);
    }

    if (Request->ResultMapping) {
        UnmapViewOfFile(Request->ResultMapping);
    }

    if (Request->ResultMappingHandle) {
        CloseHandle(Request->ResultMappingHandle);
    }

    if (Request->CompletionEvent) {
        CloseHandle(Request->CompletionEvent);
    }

    if (Request->TableCreateParameters.SizeOfStruct != 0) {
        HRESULT CleanupResult;

        CleanupResult = CleanupTableCreateParameters(
            &Request->TableCreateParameters
        );
        if (FAILED(CleanupResult)) {
            PH_ERROR(CleanupTableCreateParameters, CleanupResult);
        }
    }

    if (Request->ArgvW) {
        LocalFree(Request->ArgvW);
        Request->ArgvW = NULL;
    }

    Allocator = Request->Server ? Request->Server->Allocator : NULL;
    if (Allocator) {
        if (Request->NodeOutstandingCounts) {
            Allocator->Vtbl->FreePointer(
                Allocator,
                (PVOID *)&Request->NodeOutstandingCounts
            );
        }

        if (Request->BaseOutputDirectoryBuffer) {
            Allocator->Vtbl->FreePointer(
                Allocator,
                (PVOID *)&Request->BaseOutputDirectoryBuffer
            );
        }

        if (Request->CommandLineBuffer) {
            Allocator->Vtbl->FreePointer(
                Allocator,
                (PVOID *)&Request->CommandLineBuffer
            );
        }

        Allocator->Vtbl->FreePointer(Allocator,
                                     (PVOID *)&Request);
    }
}

static
VOID
PerfectHashServerLogBulkCreateException(
    ULONG Stage,
    PPERFECT_HASH_SERVER_BULK_WORK_ITEM WorkItem,
    struct _EXCEPTION_POINTERS *ExceptionPointers
    )
{
#ifdef PH_WINDOWS
    HANDLE LogHandle;
    DWORD BytesWritten;
    ULONG NodeIndex = 0;
    ULONG_PTR Address = 0;
    ULONG_PTR Offset = 0;
    DWORD Code = 0;
    DWORD ThreadId;
    CHAR Buffer[256];
    HMODULE Module = NULL;

    if (!ARGUMENT_PRESENT(ExceptionPointers)) {
        return;
    }

    if (GetEnvironmentVariableW(L"PH_LOG_SERVER_CRASH", NULL, 0) == 0) {
        return;
    }

    if (WorkItem) {
        NodeIndex = WorkItem->NodeIndex;
    }

    LogHandle = CreateFileW(L"PerfectHashServerBulkCreateCrash.log",
                            FILE_APPEND_DATA,
                            FILE_SHARE_READ,
                            NULL,
                            OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL,
                            NULL);
    if (!IsValidHandle(LogHandle)) {
        return;
    }

    ThreadId = GetCurrentThreadId();
    Code = ExceptionPointers->ExceptionRecord->ExceptionCode;
    Address = (ULONG_PTR)ExceptionPointers->ExceptionRecord->ExceptionAddress;

    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCWSTR)ExceptionPointers->ExceptionRecord->ExceptionAddress,
            &Module)) {
        Offset = Address - (ULONG_PTR)Module;
    }

    _snprintf_s(
        Buffer,
        sizeof(Buffer),
        _TRUNCATE,
        "Stage=%lu Code=0x%08lX Address=0x%p Module=0x%p Offset=0x%Ix "
        "ThreadId=%lu Node=%lu\r\n",
        Stage,
        Code,
        (PVOID)Address,
        Module,
        Offset,
        ThreadId,
        NodeIndex
    );

    WriteFile(LogHandle,
              Buffer,
              (DWORD)strlen(Buffer),
              &BytesWritten,
              NULL);

    CloseHandle(LogHandle);
#else
    UNREFERENCED_PARAMETER(Stage);
    UNREFERENCED_PARAMETER(WorkItem);
    UNREFERENCED_PARAMETER(ExceptionPointers);
#endif
}

static
VOID
PerfectHashServerLogBulkCreateCounts(
    _In_ ULONG Stage,
    _In_opt_ PPERFECT_HASH_SERVER_BULK_WORK_ITEM WorkItem,
    _In_opt_ PPERFECT_HASH_SERVER_BULK_REQUEST Request,
    _In_ LONG Outstanding
    )
{
#ifdef PH_WINDOWS
    int Count;
    DWORD BytesWritten;
    HANDLE LogHandle;
    CHAR Buffer[512];
    ULONG PathChars = 0;
    LONG DispatchComplete = 0;
    LONG Failed = 0;
    LONG PendingNodes = 0;
    ULONG Total = 0;
    ULONG NodeIndex = 0;

    if (GetEnvironmentVariableW(L"PH_LOG_BULK_CREATE_COUNTS", NULL, 0) == 0) {
        return;
    }

    if (Request) {
        DispatchComplete = Request->DispatchComplete;
        Failed = Request->FailedWorkItems;
        PendingNodes = Request->PendingNodes;
        Total = Request->TotalWorkItems;
    }

    if (WorkItem) {
        NodeIndex = WorkItem->NodeIndex;
        if (WorkItem->KeysPath.Buffer) {
            PathChars = WorkItem->KeysPath.Length / sizeof(WCHAR);
        }
    }

    LogHandle = CreateFileW(L"PerfectHashServerBulkCreateCounts.log",
                            FILE_APPEND_DATA,
                            FILE_SHARE_READ,
                            NULL,
                            OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL,
                            NULL);
    if (!IsValidHandle(LogHandle)) {
        return;
    }

    Count = _snprintf_s(
        Buffer,
        sizeof(Buffer),
        _TRUNCATE,
        "Stage=%lu Outstanding=%ld Dispatch=%ld Total=%lu Failed=%ld "
        "PendingNodes=%ld Node=%lu Path=%.*S\r\n",
        Stage,
        Outstanding,
        DispatchComplete,
        Total,
        Failed,
        PendingNodes,
        NodeIndex,
        (int)PathChars,
        WorkItem ? WorkItem->KeysPath.Buffer : L""
    );

    if (Count > 0) {
        WriteFile(LogHandle,
                  Buffer,
                  (DWORD)strlen(Buffer),
                  &BytesWritten,
                  NULL);
    }

    CloseHandle(LogHandle);
#else
    UNREFERENCED_PARAMETER(Stage);
    UNREFERENCED_PARAMETER(WorkItem);
    UNREFERENCED_PARAMETER(Request);
    UNREFERENCED_PARAMETER(Outstanding);
#endif
}

static
VOID
PerfectHashServerLogBulkCreateFailure(
    _In_ ULONG Stage,
    _In_opt_ PPERFECT_HASH_SERVER_BULK_WORK_ITEM WorkItem,
    _In_ HRESULT Result
    )
{
#ifdef PH_WINDOWS
    int Count;
    DWORD BytesWritten;
    HANDLE LogHandle;
    CHAR Buffer[1024];
    ULONG PathChars = 0;

    if (GetEnvironmentVariableW(L"PH_LOG_BULK_CREATE_FAILURES", NULL, 0) == 0) {
        return;
    }

    if (!WorkItem || !WorkItem->KeysPath.Buffer) {
        return;
    }

    PathChars = WorkItem->KeysPath.Length / sizeof(WCHAR);

    LogHandle = CreateFileW(L"PerfectHashServerBulkCreateFailures.log",
                            FILE_APPEND_DATA,
                            FILE_SHARE_READ,
                            NULL,
                            OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL,
                            NULL);
    if (!IsValidHandle(LogHandle)) {
        return;
    }

    Count = _snprintf_s(Buffer,
                        sizeof(Buffer),
                        _TRUNCATE,
                        "Stage=%lu Result=0x%08lX LastError=%lu (0x%08lX) "
                        "Path=%.*S\r\n",
                        Stage,
                        Result,
                        WorkItem->LastError,
                        WorkItem->LastError,
                        (int)PathChars,
                        WorkItem->KeysPath.Buffer);
    if (Count > 0) {
        WriteFile(LogHandle,
                  Buffer,
                  (DWORD)strlen(Buffer),
                  &BytesWritten,
                  NULL);
    }

    CloseHandle(LogHandle);
#else
    UNREFERENCED_PARAMETER(Stage);
    UNREFERENCED_PARAMETER(WorkItem);
    UNREFERENCED_PARAMETER(Result);
#endif
}

static
LONG
PerfectHashServerBulkCreateExceptionFilter(
    ULONG Stage,
    PPERFECT_HASH_SERVER_BULK_WORK_ITEM WorkItem,
    struct _EXCEPTION_POINTERS *ExceptionPointers
    )
{
    PerfectHashServerLogBulkCreateException(Stage,
                                            WorkItem,
                                            ExceptionPointers);
    return EXCEPTION_EXECUTE_HANDLER;
}

static
HRESULT
PerfectHashServerBulkCreateWorkItemCallback(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    ULONG_PTR CompletionKey,
    LPOVERLAPPED Overlapped,
    DWORD NumberOfBytesTransferred,
    BOOL Success
    )
{
    HRESULT Result;
    HRESULT Failure;
    LONG Outstanding;
    LONG NodeOutstanding;
    PALLOCATOR Allocator = NULL;
    PPERFECT_HASH_CONTEXT Context = NULL;
    PPERFECT_HASH_SERVER_BULK_WORK_ITEM WorkItem;
    PPERFECT_HASH_SERVER_BULK_REQUEST Request;
    PERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags;
    PPERFECT_HASH_TLS_CONTEXT TlsContext;
    PERFECT_HASH_TLS_CONTEXT LocalTlsContext = { 0 };
    ULONG Stage = 0;
    BOOLEAN ExceptionRaised = FALSE;

    UNREFERENCED_PARAMETER(NumberOfBytesTransferred);

    WorkItem = CONTAINING_RECORD(Overlapped,
                                 PERFECT_HASH_SERVER_BULK_WORK_ITEM,
                                 Iocp.Overlapped);
    Request = WorkItem->Request;
    Allocator = (Request && Request->Server) ? Request->Server->Allocator : NULL;
    WorkItem->LastError = ERROR_SUCCESS;

    if (!Success) {
        WorkItem->LastError = GetLastError();
        Failure = HRESULT_FROM_WIN32(WorkItem->LastError);
    } else {
        Failure = S_OK;
    }

    __try {
        Stage = 1;

        if ((ULONG_PTR)WorkItem != CompletionKey) {
            Failure = E_UNEXPECTED;
        }

        if (FAILED(Failure)) {
            InterlockedIncrement(&Request->FailedWorkItems);
            InterlockedCompareExchange(&Request->FirstFailure,
                                       (LONG)Failure,
                                       (LONG)S_OK);
            goto Complete;
        }

        Stage = 2;
        Result = PerfectHashContextIocpCreateTableContext(ContextIocp,
                                                          &Context);
        if (FAILED(Result)) {
            Failure = Result;
            InterlockedIncrement(&Request->FailedWorkItems);
            InterlockedCompareExchange(&Request->FirstFailure,
                                       (LONG)Failure,
                                       (LONG)S_OK);
            goto Complete;
        }

        if (Request->CommandLineBuffer) {
            Context->CommandLineW = Request->CommandLineBuffer;
        }

        if (WorkItem->Node && WorkItem->Node->IoCompletionPort) {
            Context->FileWorkIoCompletionPort = WorkItem->Node->IoCompletionPort;
        }

        SetContextSkipContextFileWork(Context);

        if (Request->PerFileMaximumConcurrency > 0) {
            Result = Context->Vtbl->SetMaximumConcurrency(
                Context,
                Request->PerFileMaximumConcurrency
            );
            if (FAILED(Result)) {
                if (Result == PH_E_SET_MAXIMUM_CONCURRENCY_FAILED &&
                    GetLastError() == ERROR_ACCESS_DENIED) {
                    //
                    // Ignore access-denied threadpool sizing failures; proceed
                    // with the default concurrency configured by initialization.
                    //
                    Result = S_OK;
                }

                if (FAILED(Result)) {
                    Failure = Result;
                    InterlockedIncrement(&Request->FailedWorkItems);
                    InterlockedCompareExchange(&Request->FirstFailure,
                                               (LONG)Failure,
                                               (LONG)S_OK);
                    goto Complete;
                }
            }
        }

        Stage = 3;
        Result = PerfectHashContextInitializeFunctionHookCallbackDll(
            Context,
            &Request->TableCreateFlags,
            &Request->TableCreateParameters
        );
        if (FAILED(Result)) {
            Failure = Result;
            InterlockedIncrement(&Request->FailedWorkItems);
            InterlockedCompareExchange(&Request->FirstFailure,
                                       (LONG)Failure,
                                       (LONG)S_OK);
            goto Complete;
        }

        Stage = 4;
        Result = PerfectHashContextInitializeRng(
            Context,
            &Request->TableCreateFlags,
            &Request->TableCreateParameters
        );
        if (FAILED(Result)) {
            Failure = Result;
            InterlockedIncrement(&Request->FailedWorkItems);
            InterlockedCompareExchange(&Request->FirstFailure,
                                       (LONG)Failure,
                                       (LONG)S_OK);
            goto Complete;
        }

        ContextTableCreateFlags = Request->ContextTableCreateFlags;

        Stage = 5;
        TlsContext = PerfectHashTlsGetOrSetContext(&LocalTlsContext);
        TlsContext->Context = Context;

        Stage = 6;
        Result = Context->Vtbl->TableCreate(
            Context,
            &WorkItem->KeysPath,
            &Request->BaseOutputDirectory,
            Request->AlgorithmId,
            Request->HashFunctionId,
            Request->MaskFunctionId,
            &ContextTableCreateFlags,
            &Request->KeysLoadFlags,
            &Request->TableCreateFlags,
            &Request->TableCompileFlags,
            &Request->TableCreateParameters
        );

        Stage = 7;
        PerfectHashTlsClearContextIfActive(&LocalTlsContext);

        if (FAILED(Result)) {
            WorkItem->LastError = GetLastError();
            Failure = Result;
            InterlockedIncrement(&Request->FailedWorkItems);
            InterlockedCompareExchange(&Request->FirstFailure,
                                       (LONG)Failure,
                                       (LONG)S_OK);
        }

    } __except (PerfectHashServerBulkCreateExceptionFilter(
                    Stage,
                    WorkItem,
                    GetExceptionInformation())) {
        ExceptionRaised = TRUE;
    }

    if (ExceptionRaised) {
        Failure = E_UNEXPECTED;
        if (Stage >= 6) {
            __try {
                PerfectHashTlsClearContextIfActive(&LocalTlsContext);
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                NOTHING;
            }
        }
        if (Request) {
            InterlockedIncrement(&Request->FailedWorkItems);
            InterlockedCompareExchange(&Request->FirstFailure,
                                       (LONG)Failure,
                                       (LONG)S_OK);
        }
    }

Complete:

    if (FAILED(Failure)) {
        PerfectHashServerLogBulkCreateFailure(Stage, WorkItem, Failure);
    }

    if (Context) {
        Context->Vtbl->Release(Context);
    }

    if (WorkItem->KeysPathBuffer && Request->Server &&
        Request->Server->Allocator) {
        Request->Server->Allocator->Vtbl->FreePointer(
            Request->Server->Allocator,
            (PVOID *)&WorkItem->KeysPathBuffer
        );
    }

    if (Request->NodeOutstandingCounts) {
        NodeOutstanding = InterlockedDecrement(
            &Request->NodeOutstandingCounts[WorkItem->NodeIndex]
        );
        if (NodeOutstanding == 0) {
            InterlockedDecrement(&Request->PendingNodes);
        }
    }

    Outstanding = InterlockedDecrement(&Request->OutstandingWorkItems);
    PerfectHashServerLogBulkCreateCounts(1, WorkItem, Request, Outstanding);

    if (Outstanding == 0 && Request->DispatchComplete != 0) {
        PerfectHashServerCompleteBulkRequest(Request);
        Request = NULL;
    }

    if (Allocator) {
        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&WorkItem);
    }

    return S_OK;
}

static
HRESULT
PerfectHashServerEnqueueBulkRequest(
    PPERFECT_HASH_SERVER_BULK_REQUEST Request,
    PUNICODE_STRING KeysDirectory
    )
{
    BOOL Success;
    HRESULT Result = S_OK;
    DWORD LastError;
    ULONG NodeIndex;
    ULONG BytesToAllocate;
    ULONG WildcardBytes;
    ULONG KeysPathBytes;
    LONG Outstanding;
    PALLOCATOR Allocator = NULL;
    PRTL Rtl = NULL;
    PWSTR WildcardBuffer = NULL;
    PWSTR KeysPathBuffer = NULL;
    BOOLEAN DispatchInitialized = FALSE;
    PWSTR Dest;
    UNICODE_STRING WildcardPath;
    WIN32_FIND_DATAW FindData;
    HANDLE FindHandle = NULL;

    if (!ARGUMENT_PRESENT(Request)) {
        return E_POINTER;
    }

    Request->OutstandingWorkItems = 1;
    Request->TotalWorkItems = 0;
    Request->DispatchComplete = 0;
    DispatchInitialized = TRUE;

    if (!ARGUMENT_PRESENT(KeysDirectory)) {
        Result = E_POINTER;
        InterlockedIncrement(&Request->FailedWorkItems);
        InterlockedCompareExchange(&Request->FirstFailure,
                                   (LONG)Result,
                                   (LONG)S_OK);
        goto End;
    }

    if (!IsValidMinimumDirectoryUnicodeString(KeysDirectory)) {
        Result = E_INVALIDARG;
        InterlockedIncrement(&Request->FailedWorkItems);
        InterlockedCompareExchange(&Request->FirstFailure,
                                   (LONG)Result,
                                   (LONG)S_OK);
        goto End;
    }

    Allocator = Request->Server ? Request->Server->Allocator : NULL;
    if (!Allocator) {
        Result = E_UNEXPECTED;
        InterlockedIncrement(&Request->FailedWorkItems);
        InterlockedCompareExchange(&Request->FirstFailure,
                                   (LONG)Result,
                                   (LONG)S_OK);
        goto End;
    }

    Rtl = Request->Server ? Request->Server->Rtl : NULL;
    if (!Rtl) {
        Result = E_UNEXPECTED;
        InterlockedIncrement(&Request->FailedWorkItems);
        InterlockedCompareExchange(&Request->FirstFailure,
                                   (LONG)Result,
                                   (LONG)S_OK);
        goto End;
    }

    BytesToAllocate = KeysDirectory->Length +
                      sizeof(WCHAR) +
                      KeysWildcardSuffix.Length +
                      sizeof(WCHAR);
    if (BytesToAllocate < KeysDirectory->Length) {
        Result = E_INVALIDARG;
        InterlockedIncrement(&Request->FailedWorkItems);
        InterlockedCompareExchange(&Request->FirstFailure,
                                   (LONG)Result,
                                   (LONG)S_OK);
        goto End;
    }

    WildcardBuffer = (PWSTR)Allocator->Vtbl->Calloc(Allocator,
                                                    1,
                                                    BytesToAllocate);
    if (!WildcardBuffer) {
        Result = E_OUTOFMEMORY;
        InterlockedIncrement(&Request->FailedWorkItems);
        InterlockedCompareExchange(&Request->FirstFailure,
                                   (LONG)Result,
                                   (LONG)S_OK);
        goto End;
    }

    CopyMemory(WildcardBuffer,
               KeysDirectory->Buffer,
               KeysDirectory->Length);

    Dest = (PWSTR)RtlOffsetToPointer(WildcardBuffer,
                                     KeysDirectory->Length);
    *Dest++ = PATHSEP;
    CopyMemory(Dest, KeysWildcardSuffix.Buffer, KeysWildcardSuffix.Length);

    WildcardBytes = BytesToAllocate - sizeof(WCHAR);
    WildcardPath.Buffer = WildcardBuffer;
    WildcardPath.Length = (USHORT)WildcardBytes;
    WildcardPath.MaximumLength = (USHORT)BytesToAllocate;

    FindHandle = FindFirstFileW(WildcardPath.Buffer, &FindData);
    if (!IsValidHandle(FindHandle)) {
        LastError = GetLastError();
        if (LastError == ERROR_FILE_NOT_FOUND) {
            Result = PH_E_NO_KEYS_FOUND_IN_DIRECTORY;
        } else {
            Result = HRESULT_FROM_WIN32(LastError);
        }
        if (Result != PH_E_NO_KEYS_FOUND_IN_DIRECTORY) {
            InterlockedIncrement(&Request->FailedWorkItems);
        }
        InterlockedCompareExchange(&Request->FirstFailure,
                                   (LONG)Result,
                                   (LONG)S_OK);
        goto End;
    }

    NodeIndex = Request->NextNodeIndex;

    do {
        ULONG FileNameLength;
        PPERFECT_HASH_IOCP_NODE Node;
        PPERFECT_HASH_SERVER_BULK_WORK_ITEM WorkItem;
        LONG NodeOutstanding;

        if (FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            continue;
        }

        FileNameLength = (ULONG)wcslen(FindData.cFileName) * sizeof(WCHAR);

        KeysPathBytes = KeysDirectory->Length +
                        sizeof(WCHAR) +
                        FileNameLength +
                        sizeof(WCHAR);
        if (KeysPathBytes < KeysDirectory->Length ||
            KeysPathBytes > (ULONG)USHRT_MAX) {
            Result = E_INVALIDARG;
            InterlockedIncrement(&Request->FailedWorkItems);
            InterlockedCompareExchange(&Request->FirstFailure,
                                       (LONG)Result,
                                       (LONG)S_OK);
            break;
        }

        WorkItem = (PPERFECT_HASH_SERVER_BULK_WORK_ITEM)(
            Allocator->Vtbl->Calloc(Allocator, 1, sizeof(*WorkItem))
        );
        if (!WorkItem) {
            Result = E_OUTOFMEMORY;
            InterlockedIncrement(&Request->FailedWorkItems);
            InterlockedCompareExchange(&Request->FirstFailure,
                                       (LONG)Result,
                                       (LONG)S_OK);
            break;
        }

        KeysPathBuffer = (PWSTR)Allocator->Vtbl->Calloc(Allocator,
                                                        1,
                                                        KeysPathBytes);
        if (!KeysPathBuffer) {
            Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&WorkItem);
            Result = E_OUTOFMEMORY;
            InterlockedIncrement(&Request->FailedWorkItems);
            InterlockedCompareExchange(&Request->FirstFailure,
                                       (LONG)Result,
                                       (LONG)S_OK);
            break;
        }

        CopyMemory(KeysPathBuffer,
                   KeysDirectory->Buffer,
                   KeysDirectory->Length);

        Dest = (PWSTR)RtlOffsetToPointer(KeysPathBuffer,
                                         KeysDirectory->Length);
        *Dest++ = PATHSEP;
        CopyMemory(Dest, FindData.cFileName, FileNameLength);

        WorkItem->Request = Request;
        WorkItem->NodeIndex = NodeIndex;
        WorkItem->Node = &Request->Nodes[NodeIndex];
        WorkItem->KeysPathBuffer = KeysPathBuffer;
        WorkItem->KeysPath.Buffer = KeysPathBuffer;
        WorkItem->KeysPath.Length = (USHORT)(KeysPathBytes - sizeof(WCHAR));
        WorkItem->KeysPath.MaximumLength = (USHORT)KeysPathBytes;
        WorkItem->Iocp.Signature = PH_IOCP_WORK_SIGNATURE;
        WorkItem->Iocp.Flags = PH_IOCP_WORK_FLAG_BULK;
        WorkItem->Iocp.CompletionCallback =
            PerfectHashServerBulkCreateWorkItemCallback;
        WorkItem->Iocp.CompletionContext = WorkItem;

        Node = WorkItem->Node;
        InterlockedIncrement(&Request->OutstandingWorkItems);
        Request->TotalWorkItems++;
        NodeOutstanding = InterlockedIncrement(
            &Request->NodeOutstandingCounts[NodeIndex]
        );
        if (NodeOutstanding == 1) {
            InterlockedIncrement(&Request->PendingNodes);
        }

        Success = PostQueuedCompletionStatus(Node->IoCompletionPort,
                                             0,
                                             (ULONG_PTR)WorkItem,
                                             &WorkItem->Iocp.Overlapped);
        if (!Success) {
            Result = HRESULT_FROM_WIN32(GetLastError());

            InterlockedDecrement(&Request->OutstandingWorkItems);
            Request->TotalWorkItems--;
            NodeOutstanding = InterlockedDecrement(
                &Request->NodeOutstandingCounts[NodeIndex]
            );
            if (NodeOutstanding == 0) {
                InterlockedDecrement(&Request->PendingNodes);
            }

            InterlockedIncrement(&Request->FailedWorkItems);
            InterlockedCompareExchange(&Request->FirstFailure,
                                       (LONG)Result,
                                       (LONG)S_OK);

            if (WorkItem->KeysPathBuffer) {
                Allocator->Vtbl->FreePointer(Allocator,
                                             (PVOID *)&WorkItem->KeysPathBuffer);
            }
            Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&WorkItem);
        }

        NodeIndex++;
        if (NodeIndex >= Request->NodeCount) {
            NodeIndex = 0;
        }

    } while (FindNextFileW(FindHandle, &FindData));

    if (Result == S_OK) {
        LastError = GetLastError();
        if (LastError != ERROR_NO_MORE_FILES) {
            Result = HRESULT_FROM_WIN32(LastError);
            InterlockedIncrement(&Request->FailedWorkItems);
            InterlockedCompareExchange(&Request->FirstFailure,
                                       (LONG)Result,
                                       (LONG)S_OK);
        }
    }

    Request->NextNodeIndex = NodeIndex;

    if (Request->TotalWorkItems == 0 && Result == S_OK) {
        Result = PH_E_NO_KEYS_FOUND_IN_DIRECTORY;
        InterlockedCompareExchange(&Request->FirstFailure,
                                   (LONG)Result,
                                   (LONG)S_OK);
    }

End:

    if (FindHandle && FindHandle != INVALID_HANDLE_VALUE) {
        FindClose(FindHandle);
    }

    if (DispatchInitialized) {
        Request->DispatchComplete = 1;
        Outstanding = InterlockedDecrement(&Request->OutstandingWorkItems);
        PerfectHashServerLogBulkCreateCounts(2, NULL, Request, Outstanding);
        if (Outstanding == 0) {
            PerfectHashServerCompleteBulkRequest(Request);
        }
    }

    if (WildcardBuffer && Allocator) {
        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&WildcardBuffer);
    }

    return Result;
}

static
HRESULT
PerfectHashServerDispatchBulkCreateDirectoryRequest(
    PPERFECT_HASH_SERVER_PIPE Pipe,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW,
    LPWSTR CommandLine,
    PBOOLEAN ArgvWOwned
    )
{
    BOOL Success;
    HRESULT Result;
    HRESULT CleanupResult;
    ULONG BytesToAllocate;
    ULONG CommandLineChars;
    ULONG CommandLineBytes;
    DWORD ClientProcessId = 0;
    HANDLE ClientProcess = NULL;
    HANDLE EventHandle = NULL;
    HANDLE ResultHandle = NULL;
    HANDLE ClientEventHandle = NULL;
    HANDLE ClientResultHandle = NULL;
    ULONG AdjustedArguments;
    PALLOCATOR Allocator;
    PRTL Rtl;
    PPERFECT_HASH_SERVER Server;
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp;
    PPERFECT_HASH_SERVER_BULK_REQUEST Request = NULL;
    PPERFECT_HASH_SERVER_BULK_RESULT BulkResult = NULL;
    UNICODE_STRING KeysDirectory = { 0 };
    UNICODE_STRING BaseOutputDirectory = { 0 };
    PERFECT_HASH_ALGORITHM_ID AlgorithmId = 0;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId = 0;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId = 0;
    ULONG MaximumConcurrency = 0;
    PERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags = { 0 };
    PERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags = { 0 };
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags = { 0 };
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags = { 0 };
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags = { 0 };
    PERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters = { 0 };
    LPWSTR *AdjustedArgvW = NULL;
    BOOLEAN ParsedArgs = FALSE;

    if (!ARGUMENT_PRESENT(Pipe)) {
        return E_POINTER;
    }

    Server = Pipe->Server;
    if (!Server) {
        return E_UNEXPECTED;
    }

    ContextIocp = Server->ContextIocp;
    Allocator = Server->Allocator;
    Rtl = Server->Rtl;

    if (!ContextIocp || !Allocator || !Rtl) {
        return E_UNEXPECTED;
    }

    if (!ARGUMENT_PRESENT(ArgvWOwned)) {
        return E_POINTER;
    }

    *ArgvWOwned = FALSE;

    Result = LoadDefaultTableCreateFlags(&TableCreateFlags);
    if (FAILED(Result)) {
        goto Error;
    }

    TableCreateParameters.SizeOfStruct = sizeof(TableCreateParameters);
    TableCreateParameters.Allocator = Allocator;

    AdjustedArguments = NumberOfArguments + 1;
    AdjustedArgvW = Allocator->Vtbl->Calloc(Allocator,
                                            AdjustedArguments,
                                            sizeof(*AdjustedArgvW));
    if (!AdjustedArgvW) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    AdjustedArgvW[0] = L"PerfectHashBulkCreate";
    CopyMemory(&AdjustedArgvW[1],
               ArgvW,
               NumberOfArguments * sizeof(*AdjustedArgvW));

    Result = ContextIocp->Vtbl->ExtractBulkCreateArgsFromArgvW(
        ContextIocp,
        AdjustedArguments,
        AdjustedArgvW,
        CommandLine,
        &KeysDirectory,
        &BaseOutputDirectory,
        &AlgorithmId,
        &HashFunctionId,
        &MaskFunctionId,
        &MaximumConcurrency,
        &ContextBulkCreateFlags,
        &KeysLoadFlags,
        &TableCreateFlags,
        &TableCompileFlags,
        &TableCreateParameters
    );
    if (FAILED(Result)) {
        goto Error;
    }
    ParsedArgs = TRUE;

    ContextTableCreateFlags.AsULong = 0;
    ContextTableCreateFlags.SkipTestAfterCreate =
        ContextBulkCreateFlags.SkipTestAfterCreate;
    ContextTableCreateFlags.Compile = ContextBulkCreateFlags.Compile;
    ContextTableCreateFlags.MonitorLowMemory =
        ContextBulkCreateFlags.MonitorLowMemory;

    //
    // Suppress per-file CSV output for async bulk operations.
    //

    TableCreateFlags.DisableCsvOutputFile = TRUE;
    if (Server->Flags.NoFileIo) {
        TableCreateFlags.NoFileIo = TRUE;
    }
    if (!Server->Flags.Verbose) {
        TableCreateFlags.Silent = TRUE;
        TableCreateFlags.Quiet = FALSE;
    }

    Request = (PPERFECT_HASH_SERVER_BULK_REQUEST)(
        Allocator->Vtbl->Calloc(Allocator, 1, sizeof(*Request))
    );
    if (!Request) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    Request->SizeOfStruct = sizeof(*Request);
    Request->Server = Server;
    Request->ContextBulkCreateFlags = ContextBulkCreateFlags;
    Request->ContextTableCreateFlags = ContextTableCreateFlags;
    Request->KeysLoadFlags = KeysLoadFlags;
    Request->TableCreateFlags = TableCreateFlags;
    Request->TableCompileFlags = TableCompileFlags;
    Request->TableCreateParameters = TableCreateParameters;
    Request->AlgorithmId = AlgorithmId;
    Request->HashFunctionId = HashFunctionId;
    Request->MaskFunctionId = MaskFunctionId;
    Request->PerFileMaximumConcurrency = MaximumConcurrency;
    Request->FirstFailure = (LONG)S_OK;
    Request->ArgvW = ArgvW;
    *ArgvWOwned = TRUE;

    CommandLineChars = (ULONG)wcslen(CommandLine);
    if (CommandLineChars > ((ULONG_MAX / sizeof(WCHAR)) - 1)) {
        Result = E_INVALIDARG;
        goto Error;
    }

    CommandLineBytes = (CommandLineChars + 1) * sizeof(WCHAR);
    Request->CommandLineBuffer = (PWSTR)(
        Allocator->Vtbl->Calloc(Allocator, 1, CommandLineBytes)
    );
    if (!Request->CommandLineBuffer) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    CopyMemory(Request->CommandLineBuffer,
               CommandLine,
               CommandLineBytes);

    Request->Nodes = ContextIocp->Nodes;
    Request->NodeCount = ContextIocp->NodeCount;

    if (!Request->Nodes || Request->NodeCount == 0) {
        Result = E_UNEXPECTED;
        goto Error;
    }

    Request->NodeOutstandingCounts = (PLONG)(
        Allocator->Vtbl->Calloc(Allocator,
                                Request->NodeCount,
                                sizeof(*Request->NodeOutstandingCounts))
    );
    if (!Request->NodeOutstandingCounts) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    BytesToAllocate = BaseOutputDirectory.Length + sizeof(WCHAR);
    if (BytesToAllocate < BaseOutputDirectory.Length ||
        BytesToAllocate > (ULONG)USHRT_MAX) {
        Result = E_INVALIDARG;
        goto Error;
    }

    Request->BaseOutputDirectoryBuffer = (PWSTR)(
        Allocator->Vtbl->Calloc(Allocator, 1, BytesToAllocate)
    );
    if (!Request->BaseOutputDirectoryBuffer) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    CopyMemory(Request->BaseOutputDirectoryBuffer,
               BaseOutputDirectory.Buffer,
               BaseOutputDirectory.Length);

    Request->BaseOutputDirectory.Buffer = Request->BaseOutputDirectoryBuffer;
    Request->BaseOutputDirectory.Length = BaseOutputDirectory.Length;
    Request->BaseOutputDirectory.MaximumLength = (USHORT)BytesToAllocate;

    EventHandle = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!EventHandle) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    Request->CompletionEvent = EventHandle;

    ResultHandle = CreateFileMappingW(INVALID_HANDLE_VALUE,
                                      NULL,
                                      PAGE_READWRITE,
                                      0,
                                      sizeof(PERFECT_HASH_SERVER_BULK_RESULT),
                                      NULL);
    if (!ResultHandle) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    Request->ResultMappingHandle = ResultHandle;

    BulkResult = (PPERFECT_HASH_SERVER_BULK_RESULT)(
        MapViewOfFile(ResultHandle, FILE_MAP_WRITE, 0, 0, 0)
    );
    if (!BulkResult) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    ZeroMemory(BulkResult, sizeof(*BulkResult));
    Request->ResultMapping = BulkResult;

    Success = GetNamedPipeClientProcessId(Pipe->Pipe, &ClientProcessId);
    if (!Success) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    ClientProcess = OpenProcess(PROCESS_DUP_HANDLE, FALSE, ClientProcessId);
    if (!ClientProcess) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Success = DuplicateHandle(GetCurrentProcess(),
                              EventHandle,
                              ClientProcess,
                              &ClientEventHandle,
                              0,
                              FALSE,
                              DUPLICATE_SAME_ACCESS);
    if (!Success) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Success = DuplicateHandle(GetCurrentProcess(),
                              ResultHandle,
                              ClientProcess,
                              &ClientResultHandle,
                              0,
                              FALSE,
                              DUPLICATE_SAME_ACCESS);
    if (!Success) {
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    CloseHandle(ClientProcess);
    ClientProcess = NULL;

    Result = PerfectHashServerPrepareBulkCreateDirectoryPayload(
        Pipe,
        ClientEventHandle,
        ClientResultHandle
    );
    if (FAILED(Result)) {
        goto Error;
    }

    Result = PerfectHashServerEnqueueBulkRequest(Request, &KeysDirectory);
    if (FAILED(Result)) {
        //
        // Enqueue failures are recorded and signaled by the enqueue routine.
        //
    }

    if (AdjustedArgvW) {
        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&AdjustedArgvW);
    }

    return S_OK;

Error:

    if (AdjustedArgvW && Allocator) {
        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&AdjustedArgvW);
    }

    if (ClientProcess) {
        CloseHandle(ClientProcess);
    }

    if (BulkResult) {
        UnmapViewOfFile(BulkResult);
    }

    if (ResultHandle) {
        CloseHandle(ResultHandle);
    }

    if (EventHandle) {
        CloseHandle(EventHandle);
    }

    if (Request) {
        if (Request->ArgvW) {
            LocalFree(Request->ArgvW);
            Request->ArgvW = NULL;
        }

        if (Request->NodeOutstandingCounts) {
            Allocator->Vtbl->FreePointer(
                Allocator,
                (PVOID *)&Request->NodeOutstandingCounts
            );
        }

        if (Request->BaseOutputDirectoryBuffer) {
            Allocator->Vtbl->FreePointer(
                Allocator,
                (PVOID *)&Request->BaseOutputDirectoryBuffer
            );
        }

        if (Request->CommandLineBuffer) {
            Allocator->Vtbl->FreePointer(
                Allocator,
                (PVOID *)&Request->CommandLineBuffer
            );
        }

        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&Request);
    }

    if (ParsedArgs) {
        CleanupResult = CleanupTableCreateParameters(&TableCreateParameters);
        if (FAILED(CleanupResult)) {
            PH_ERROR(CleanupTableCreateParameters, CleanupResult);
            Result = CleanupResult;
        }
    }

    return Result;
}

static
HRESULT
PerfectHashServerDispatchTableCreateRequest(
    PPERFECT_HASH_SERVER_PIPE Pipe,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW,
    LPWSTR CommandLine
    )
{
    HRESULT Result;
    HRESULT CleanupResult;
    ULONG CommandLineChars;
    ULONG CommandLineBytes;
    ULONG AdjustedArguments;
    ULONG MaximumConcurrency = 0;
    PALLOCATOR Allocator;
    PRTL Rtl;
    PPERFECT_HASH_SERVER Server;
    PPERFECT_HASH_CONTEXT Context = NULL;
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp;
    PPERFECT_HASH_IOCP_NODE Node = NULL;
    LPWSTR *AdjustedArgvW = NULL;
    LPWSTR *ArgvToUse = NULL;
    PWSTR CommandLineBuffer = NULL;
    UNICODE_STRING KeysPath = { 0 };
    UNICODE_STRING BaseOutputDirectory = { 0 };
    PERFECT_HASH_ALGORITHM_ID AlgorithmId = 0;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId = 0;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId = 0;
    PERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags = { 0 };
    PERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags = { 0 };
    PERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags = { 0 };
    PERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags = { 0 };
    PERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters = { 0 };
    PPERFECT_HASH_TLS_CONTEXT TlsContext;
    PERFECT_HASH_TLS_CONTEXT LocalTlsContext = { 0 };
    BOOLEAN ParsedArgs = FALSE;
    BOOLEAN TlsContextSet = FALSE;

    if (!ARGUMENT_PRESENT(Pipe)) {
        return E_POINTER;
    }

    Server = Pipe->Server;
    if (!Server) {
        return E_UNEXPECTED;
    }

    ContextIocp = Server->ContextIocp;
    Allocator = Server->Allocator;
    Rtl = Server->Rtl;

    if (!ContextIocp || !Allocator || !Rtl) {
        return E_UNEXPECTED;
    }

    if (!ARGUMENT_PRESENT(ArgvW) || NumberOfArguments == 0) {
        return E_INVALIDARG;
    }

    Result = LoadDefaultTableCreateFlags(&TableCreateFlags);
    if (FAILED(Result)) {
        return Result;
    }

    TableCreateParameters.SizeOfStruct = sizeof(TableCreateParameters);
    TableCreateParameters.Allocator = Allocator;

    ArgvToUse = ArgvW;
    AdjustedArguments = NumberOfArguments;

    if (ArgvW[0][0] == L'-' || ArgvW[0][0] == L'/') {
        AdjustedArgvW = Allocator->Vtbl->Calloc(Allocator,
                                                NumberOfArguments + 1,
                                                sizeof(*AdjustedArgvW));
        if (!AdjustedArgvW) {
            Result = E_OUTOFMEMORY;
            goto Error;
        }

        AdjustedArgvW[0] = L"PerfectHashTableCreate";
        CopyMemory(&AdjustedArgvW[1],
                   ArgvW,
                   NumberOfArguments * sizeof(*AdjustedArgvW));
        ArgvToUse = AdjustedArgvW;
        AdjustedArguments = NumberOfArguments + 1;
    }

    Result = ContextIocp->Vtbl->ExtractTableCreateArgsFromArgvW(
        ContextIocp,
        AdjustedArguments,
        ArgvToUse,
        CommandLine,
        &KeysPath,
        &BaseOutputDirectory,
        &AlgorithmId,
        &HashFunctionId,
        &MaskFunctionId,
        &MaximumConcurrency,
        &ContextTableCreateFlags,
        &KeysLoadFlags,
        &TableCreateFlags,
        &TableCompileFlags,
        &TableCreateParameters
    );
    if (FAILED(Result)) {
        goto Error;
    }
    ParsedArgs = TRUE;

    if (Server->Flags.NoFileIo) {
        TableCreateFlags.NoFileIo = TRUE;
    }
    if (!Server->Flags.Verbose) {
        TableCreateFlags.Silent = TRUE;
        TableCreateFlags.Quiet = FALSE;
    }

    Result = PerfectHashContextIocpCreateTableContext(ContextIocp,
                                                      &Context);
    if (FAILED(Result)) {
        goto Error;
    }

    if (CommandLine) {
        CommandLineChars = (ULONG)wcslen(CommandLine);
        if (CommandLineChars > ((ULONG_MAX / sizeof(WCHAR)) - 1)) {
            Result = E_INVALIDARG;
            goto Error;
        }

        CommandLineBytes = (CommandLineChars + 1) * sizeof(WCHAR);
        CommandLineBuffer = (PWSTR)Allocator->Vtbl->Calloc(Allocator,
                                                           1,
                                                           CommandLineBytes);
        if (!CommandLineBuffer) {
            Result = E_OUTOFMEMORY;
            goto Error;
        }

        CopyMemory(CommandLineBuffer, CommandLine, CommandLineBytes);
        Context->CommandLineW = CommandLineBuffer;
    }

    if (ContextIocp->NodeCount > 0 && ContextIocp->Nodes) {
        Node = &ContextIocp->Nodes[0];
        if (Node->IoCompletionPort) {
            Context->FileWorkIoCompletionPort = Node->IoCompletionPort;
        }
    }

    SetContextSkipContextFileWork(Context);

    Result = PerfectHashContextInitializeFunctionHookCallbackDll(
        Context,
        &TableCreateFlags,
        &TableCreateParameters
    );
    if (FAILED(Result)) {
        goto Error;
    }

    if (MaximumConcurrency > 0) {
        Result = Context->Vtbl->SetMaximumConcurrency(Context,
                                                      MaximumConcurrency);
        if (FAILED(Result)) {
            if (Result == PH_E_SET_MAXIMUM_CONCURRENCY_FAILED &&
                GetLastError() == ERROR_ACCESS_DENIED) {
                Result = S_OK;
            }
        }
        if (FAILED(Result)) {
            goto Error;
        }
    }

    Result = PerfectHashContextInitializeRng(Context,
                                             &TableCreateFlags,
                                             &TableCreateParameters);
    if (FAILED(Result)) {
        goto Error;
    }

    TlsContext = PerfectHashTlsGetOrSetContext(&LocalTlsContext);
    TlsContext->Context = Context;
    TlsContextSet = TRUE;

    Result = Context->Vtbl->TableCreate(Context,
                                        &KeysPath,
                                        &BaseOutputDirectory,
                                        AlgorithmId,
                                        HashFunctionId,
                                        MaskFunctionId,
                                        &ContextTableCreateFlags,
                                        &KeysLoadFlags,
                                        &TableCreateFlags,
                                        &TableCompileFlags,
                                        &TableCreateParameters);

Error:

    if (TlsContextSet) {
        PerfectHashTlsClearContextIfActive(&LocalTlsContext);
    }

    if (ParsedArgs) {
        CleanupResult = CleanupTableCreateParameters(&TableCreateParameters);
        if (FAILED(CleanupResult)) {
            PH_ERROR(CleanupTableCreateParameters, CleanupResult);
            if (SUCCEEDED(Result)) {
                Result = CleanupResult;
            }
        }
    }

    if (Context) {
        Context->Vtbl->Release(Context);
    }

    if (CommandLineBuffer && Allocator) {
        Allocator->Vtbl->FreePointer(Allocator,
                                     (PVOID *)&CommandLineBuffer);
    }

    if (AdjustedArgvW && Allocator) {
        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&AdjustedArgvW);
    }

    return Result;
}

static
HRESULT
PerfectHashServerDispatchRequest(
    PPERFECT_HASH_SERVER_PIPE Pipe
    )
{
    HRESULT Result = E_UNEXPECTED;
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp;
    PERFECT_HASH_SERVER_REQUEST_TYPE RequestType;
    PALLOCATOR Allocator;
    LPWSTR CommandLine;
    LPWSTR *ArgvW;
    ULONG NumberOfArguments;
    PRTL Rtl;
    BOOLEAN ArgvWOwned;

    if (!ARGUMENT_PRESENT(Pipe)) {
        return E_POINTER;
    }

    Rtl = Pipe->Server ? Pipe->Server->Rtl : NULL;
    if (!Rtl) {
        return E_UNEXPECTED;
    }

    ContextIocp = Pipe->Server ? Pipe->Server->ContextIocp : NULL;
    RequestType = Pipe->RequestHeader.RequestType;
    Allocator = Pipe->Server ? Pipe->Server->Allocator : NULL;

    ZeroMemory(&Pipe->ResponseHeader, sizeof(Pipe->ResponseHeader));
    Pipe->ResponseHeader.SizeOfStruct =
        sizeof(PERFECT_HASH_SERVER_RESPONSE_HEADER);
    Pipe->ResponseHeader.Version = PERFECT_HASH_SERVER_MESSAGE_VERSION;
    Pipe->ResponseHeader.RequestId = Pipe->RequestHeader.RequestId;
    Pipe->ResponseHeader.PayloadLength = 0;

    ArgvW = NULL;
    NumberOfArguments = 0;
    CommandLine = NULL;
    ArgvWOwned = FALSE;

    switch (RequestType) {
        case PerfectHashNullServerRequestType:
        case PerfectHashInvalidServerRequestType:
            Result = E_INVALIDARG;
            break;

        case PerfectHashPingServerRequestType:
            if (Pipe->PayloadLength != 0) {
                Result = E_INVALIDARG;
                break;
            }
            if (Pipe->Server && Pipe->Server->StartedEvent) {
                WaitForSingleObject(Pipe->Server->StartedEvent, INFINITE);
            }
            Result = PerfectHashServerPreparePingPayload(Pipe);
            break;

        case PerfectHashShutdownServerRequestType:
            Result = S_OK;
            Pipe->ShutdownAfterSend = TRUE;
            break;

        case PerfectHashTableCreateServerRequestType:
        case PerfectHashBulkCreateServerRequestType:
        case PerfectHashBulkCreateDirectoryServerRequestType:
            if (!ContextIocp || !Allocator) {
                Result = E_UNEXPECTED;
                break;
            }

            if (Pipe->PayloadLength == 0 ||
                (Pipe->PayloadLength % sizeof(WCHAR)) != 0) {
                Result = E_INVALIDARG;
                break;
            }

            if (!Pipe->PayloadBuffer) {
                Result = E_UNEXPECTED;
                break;
            }

            CommandLine = (LPWSTR)Pipe->PayloadBuffer;
            if (CommandLine[0] == L'\0') {
                Result = E_INVALIDARG;
                break;
            }

            ArgvW = CommandLineToArgvW(CommandLine,
                                       (PINT)&NumberOfArguments);
            if (!ArgvW || NumberOfArguments == 0) {
                Result = HRESULT_FROM_WIN32(GetLastError());
                if (Result == S_OK) {
                    Result = E_INVALIDARG;
                }
                break;
            }

            if (RequestType == PerfectHashTableCreateServerRequestType) {
                Result = PerfectHashServerDispatchTableCreateRequest(
                    Pipe,
                    NumberOfArguments,
                    ArgvW,
                    CommandLine
                );
            } else if (RequestType ==
                       PerfectHashBulkCreateDirectoryServerRequestType ||
                       RequestType == PerfectHashBulkCreateServerRequestType) {
                Result = PerfectHashServerDispatchBulkCreateDirectoryRequest(
                    Pipe,
                    NumberOfArguments,
                    ArgvW,
                    CommandLine,
                    &ArgvWOwned
                );
            }

            break;

        default:
            Result = E_INVALIDARG;
            break;
    }

    if (ArgvW && !ArgvWOwned) {
        LocalFree(ArgvW);
    }

    Pipe->ResponseHeader.Result = Result;
    if (FAILED(Result)) {
        HRESULT PayloadResult;

        PayloadResult = PerfectHashServerPrepareErrorPayload(Pipe, Result);
        if (FAILED(PayloadResult)) {
            Pipe->ResponseHeader.PayloadLength = 0;
            Pipe->ResponseHeader.Flags = 0;
        }
    }
    Pipe->BytesTransferred = 0;

    return PerfectHashServerIssueWriteResponseHeader(Pipe);
}

static
HRESULT
PerfectHashServerIocpCompletionCallback(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    ULONG_PTR CompletionKey,
    LPOVERLAPPED Overlapped,
    DWORD NumberOfBytesTransferred,
    BOOL Success
    )
{
    HRESULT Result;
    DWORD LastError;
    ULONG Expected;
    PPERFECT_HASH_SERVER Server;
    PPERFECT_HASH_SERVER_PIPE Pipe;

    UNREFERENCED_PARAMETER(ContextIocp);

    if (!Overlapped) {
        return E_POINTER;
    }

    Pipe = CONTAINING_RECORD(Overlapped,
                             PERFECT_HASH_SERVER_PIPE,
                             Iocp.Overlapped);
    Server = Pipe->Server;

    if ((ULONG_PTR)Pipe != CompletionKey) {
        return E_UNEXPECTED;
    }

    if (!Success) {
        LastError = GetLastError();
        switch (LastError) {
            case ERROR_BROKEN_PIPE:
            case ERROR_PIPE_NOT_CONNECTED:
            case ERROR_NO_DATA:
            case ERROR_OPERATION_ABORTED:
                PerfectHashServerResetPipe(Pipe, TRUE);
                return S_OK;
            default:
                SYS_ERROR(GetQueuedCompletionStatus);
                PerfectHashServerResetPipe(Pipe, TRUE);
                return HRESULT_FROM_WIN32(LastError);
        }
    }

    switch (Pipe->State) {
        case PerfectHashServerPipeStateInvalid:
            PerfectHashServerResetPipe(Pipe, TRUE);
            return E_UNEXPECTED;

        case PerfectHashServerPipeStateAccepting:
            Pipe->BytesTransferred = 0;
            Result = PerfectHashServerIssueReadHeader(Pipe);
            if (FAILED(Result)) {
                PerfectHashServerResetPipe(Pipe, TRUE);
            }
            return Result;

        case PerfectHashServerPipeStateReadingHeader:
            if (NumberOfBytesTransferred == 0) {
                PerfectHashServerResetPipe(Pipe, TRUE);
                return S_OK;
            }

            Pipe->BytesTransferred += NumberOfBytesTransferred;
            Expected = sizeof(PERFECT_HASH_SERVER_REQUEST_HEADER);

            if (Pipe->BytesTransferred < Expected) {
                Result = PerfectHashServerIssueReadHeader(Pipe);
                if (FAILED(Result)) {
                    PerfectHashServerResetPipe(Pipe, TRUE);
                }
                return Result;
            }

            if (Pipe->RequestHeader.SizeOfStruct != Expected ||
                Pipe->RequestHeader.Version !=
                PERFECT_HASH_SERVER_MESSAGE_VERSION ||
                !IsValidPerfectHashServerRequestType(
                    Pipe->RequestHeader.RequestType) ||
                Pipe->RequestHeader.PayloadLength >
                PERFECT_HASH_SERVER_MAX_MESSAGE_SIZE) {
                Pipe->ResponseHeader.SizeOfStruct =
                    sizeof(PERFECT_HASH_SERVER_RESPONSE_HEADER);
                Pipe->ResponseHeader.Version =
                    PERFECT_HASH_SERVER_MESSAGE_VERSION;
                Pipe->ResponseHeader.RequestId =
                    Pipe->RequestHeader.RequestId;
                Pipe->ResponseHeader.Result = E_INVALIDARG;
                Pipe->ResponseHeader.PayloadLength = 0;
                Pipe->BytesTransferred = 0;
                Result = PerfectHashServerIssueWriteResponseHeader(Pipe);
                if (FAILED(Result)) {
                    PerfectHashServerResetPipe(Pipe, TRUE);
                }
                return Result;
            }

            Pipe->PayloadLength = Pipe->RequestHeader.PayloadLength;

            if (Pipe->PayloadLength == 0) {
                Result = PerfectHashServerDispatchRequest(Pipe);
                if (FAILED(Result)) {
                    PerfectHashServerResetPipe(Pipe, TRUE);
                }
                return Result;
            }

            if (!Pipe->PayloadBuffer ||
                Pipe->PayloadBufferSize < Pipe->PayloadLength) {
                SIZE_T AllocationSize;
                PALLOCATOR Allocator;

                Allocator = Server ? Server->Allocator : NULL;
                if (!Allocator) {
                    PerfectHashServerResetPipe(Pipe, TRUE);
                    return E_UNEXPECTED;
                }

                if (Pipe->PayloadBuffer) {
                    Allocator->Vtbl->FreePointer(
                        Allocator,
                        &Pipe->PayloadBuffer
                    );
                }

                AllocationSize = (SIZE_T)Pipe->PayloadLength +
                                 sizeof(WCHAR);
                if (AllocationSize < Pipe->PayloadLength) {
                    PerfectHashServerResetPipe(Pipe, TRUE);
                    return E_INVALIDARG;
                }

                Pipe->PayloadBuffer = Allocator->Vtbl->Calloc(
                    Allocator,
                    1,
                    AllocationSize
                );
                if (!Pipe->PayloadBuffer) {
                    PerfectHashServerResetPipe(Pipe, TRUE);
                    return E_OUTOFMEMORY;
                }

                Pipe->PayloadBufferSize = (ULONG)AllocationSize;
            }

            Pipe->BytesTransferred = 0;
            Result = PerfectHashServerIssueReadPayload(Pipe);
            if (FAILED(Result)) {
                PerfectHashServerResetPipe(Pipe, TRUE);
            }
            return Result;

        case PerfectHashServerPipeStateReadingPayload:
            if (NumberOfBytesTransferred == 0) {
                PerfectHashServerResetPipe(Pipe, TRUE);
                return S_OK;
            }

            Pipe->BytesTransferred += NumberOfBytesTransferred;
            Expected = Pipe->PayloadLength;

            if (Pipe->BytesTransferred < Expected) {
                Result = PerfectHashServerIssueReadPayload(Pipe);
                if (FAILED(Result)) {
                    PerfectHashServerResetPipe(Pipe, TRUE);
                }
                return Result;
            }

            Result = PerfectHashServerDispatchRequest(Pipe);
            if (FAILED(Result)) {
                PerfectHashServerResetPipe(Pipe, TRUE);
            }
            return Result;

        case PerfectHashServerPipeStateWritingResponseHeader:
            Pipe->BytesTransferred += NumberOfBytesTransferred;
            Expected = sizeof(PERFECT_HASH_SERVER_RESPONSE_HEADER);

            if (Pipe->BytesTransferred < Expected) {
                Result = PerfectHashServerIssueWriteResponseHeader(Pipe);
                if (FAILED(Result)) {
                    PerfectHashServerResetPipe(Pipe, TRUE);
                }
                return Result;
            }

            if (Pipe->ResponseHeader.PayloadLength > 0) {
                Pipe->BytesTransferred = 0;
                Result = PerfectHashServerIssueWriteResponsePayload(Pipe);
                if (FAILED(Result)) {
                    PerfectHashServerResetPipe(Pipe, TRUE);
                }
                return Result;
            }

            if (Pipe->ShutdownAfterSend && Server) {
                if (Server->ShutdownEvent) {
                    SetEvent(Server->ShutdownEvent);
                }
            }

            PerfectHashServerResetPipe(Pipe, TRUE);
            return S_OK;

        case PerfectHashServerPipeStateWritingResponsePayload:
            Pipe->BytesTransferred += NumberOfBytesTransferred;
            Expected = Pipe->ResponseHeader.PayloadLength;

            if (Pipe->BytesTransferred < Expected) {
                Result = PerfectHashServerIssueWriteResponsePayload(Pipe);
                if (FAILED(Result)) {
                    PerfectHashServerResetPipe(Pipe, TRUE);
                }
                return Result;
            }

            if (Pipe->ShutdownAfterSend && Server) {
                if (Server->ShutdownEvent) {
                    SetEvent(Server->ShutdownEvent);
                }
            }

            PerfectHashServerResetPipe(Pipe, TRUE);
            return S_OK;

        default:
            PerfectHashServerResetPipe(Pipe, TRUE);
            return E_UNEXPECTED;
    }
}

#endif // PH_WINDOWS

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
