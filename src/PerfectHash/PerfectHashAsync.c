/*++

Copyright (c) 2018-2025 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashAsync.c

Abstract:

    This module implements the asynchronous work framework used by IOCP-based
    execution paths.  Work items are driven by per-item completion callbacks
    and can cooperatively yield by re-posting to the completion port.

--*/

#include "stdafx.h"

static
HRESULT
PerfectHashAsyncRequeueWork(
    _In_ PPERFECT_HASH_ASYNC_WORK Work
    )
{
    BOOL Success;
    PPERFECT_HASH_ASYNC_CONTEXT AsyncContext;

    AsyncContext = Work->AsyncContext;
    if (!AsyncContext || !AsyncContext->IoCompletionPort) {
        return E_UNEXPECTED;
    }

    Success = PostQueuedCompletionStatus(AsyncContext->IoCompletionPort,
                                         0,
                                         0,
                                         &Work->Iocp.Overlapped);
    if (!Success) {
        SYS_ERROR(PostQueuedCompletionStatus_AsyncRequeue);
        return PH_E_SYSTEM_CALL_FAILED;
    }

    return S_OK;
}

static
HRESULT
PerfectHashAsyncCompleteWork(
    _In_ PPERFECT_HASH_ASYNC_WORK Work,
    _In_ HRESULT Result
    )
{
    PPERFECT_HASH_ASYNC_CONTEXT AsyncContext;

    AsyncContext = Work->AsyncContext;
    Work->LastResult = Result;

    if (Work->Complete) {
        Work->Complete(Work, Result);
    }

    if (AsyncContext) {
        if (InterlockedDecrement(&AsyncContext->Outstanding) == 0) {
            if (AsyncContext->OutstandingEvent) {
                SetEvent(AsyncContext->OutstandingEvent);
            }
        }
    }

    return Result;
}

static
HRESULT
PerfectHashAsyncIocpCompletionCallback(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    _In_ ULONG_PTR CompletionKey,
    _In_ LPOVERLAPPED Overlapped,
    _In_ DWORD NumberOfBytesTransferred,
    _In_ BOOL Success
    )
{
    HRESULT Result;
    PPERFECT_HASH_ASYNC_WORK Work;

    UNREFERENCED_PARAMETER(ContextIocp);
    UNREFERENCED_PARAMETER(CompletionKey);
    UNREFERENCED_PARAMETER(NumberOfBytesTransferred);
    UNREFERENCED_PARAMETER(Success);

    Work = (PPERFECT_HASH_ASYNC_WORK)Overlapped;
    if (!Work || !Work->Step) {
        return E_POINTER;
    }

    Result = Work->Step(Work);

    if (Result == S_FALSE) {
        Result = PerfectHashAsyncRequeueWork(Work);
        if (FAILED(Result)) {
            return PerfectHashAsyncCompleteWork(Work, Result);
        }
        return S_OK;
    }

    return PerfectHashAsyncCompleteWork(Work, Result);
}

_Use_decl_annotations_
HRESULT
PerfectHashAsyncInitialize(
    PPERFECT_HASH_ASYNC_CONTEXT AsyncContext,
    PPERFECT_HASH_CONTEXT Context,
    HANDLE IoCompletionPort
    )
/*++

Routine Description:

    Initializes an async context for IOCP work submission.

Arguments:

    AsyncContext - Supplies a pointer to the async context to initialize.

    Context - Supplies a pointer to the owning perfect hash context.

    IoCompletionPort - Supplies the IOCP handle to post work items to.

Return Value:

    S_OK on success, or an appropriate error code on failure.

--*/
{
    if (!ARGUMENT_PRESENT(AsyncContext) ||
        !ARGUMENT_PRESENT(Context) ||
        !IoCompletionPort) {
        return E_POINTER;
    }

    ZeroStructPointerInline(AsyncContext);
    AsyncContext->Context = Context;
    AsyncContext->Allocator = Context->Allocator;
    AsyncContext->IoCompletionPort = IoCompletionPort;
    AsyncContext->Outstanding = 0;
    AsyncContext->OutstandingEvent = NULL;

    return S_OK;
}

_Use_decl_annotations_
VOID
PerfectHashAsyncRundown(
    PPERFECT_HASH_ASYNC_CONTEXT AsyncContext
    )
/*++

Routine Description:

    Releases resources associated with an async context.

Arguments:

    AsyncContext - Supplies a pointer to the async context to rundown.

Return Value:

    None.

--*/
{
    if (!ARGUMENT_PRESENT(AsyncContext)) {
        return;
    }

    if (AsyncContext->OutstandingEvent) {
        if (!CloseEvent(AsyncContext->OutstandingEvent)) {
            SYS_ERROR(CloseHandle);
        }
        AsyncContext->OutstandingEvent = NULL;
    }

    AsyncContext->Outstanding = 0;
}

_Use_decl_annotations_
HRESULT
PerfectHashAsyncSubmit(
    PPERFECT_HASH_ASYNC_CONTEXT AsyncContext,
    PPERFECT_HASH_ASYNC_WORK Work
    )
/*++

Routine Description:

    Submits an async work item to the IOCP dispatch queue.  If submission
    fails, the work is executed inline.

Arguments:

    AsyncContext - Supplies a pointer to the async context.

    Work - Supplies a pointer to the async work item.

Return Value:

    S_OK if the work was queued, otherwise the result of inline execution.

--*/
{
    BOOL Success;
    LONG Outstanding;
    HRESULT Result;

    if (!ARGUMENT_PRESENT(AsyncContext) ||
        !ARGUMENT_PRESENT(Work) ||
        !Work->Step) {
        return E_POINTER;
    }

    if (!AsyncContext->IoCompletionPort) {
        return E_UNEXPECTED;
    }

    Work->AsyncContext = AsyncContext;
    Work->Iocp.Signature = PH_IOCP_WORK_SIGNATURE;
    Work->Iocp.Flags = PH_IOCP_WORK_FLAG_ASYNC;
    Work->Iocp.CompletionCallback = PerfectHashAsyncIocpCompletionCallback;

    if (!AsyncContext->OutstandingEvent) {
        AsyncContext->OutstandingEvent = CreateEventW(NULL, TRUE, TRUE, NULL);
        if (!AsyncContext->OutstandingEvent) {
            SYS_ERROR(CreateEventW_AsyncOutstandingEvent);
            goto InlineWork;
        }
    }

    Outstanding = InterlockedIncrement(&AsyncContext->Outstanding);
    if (Outstanding == 1 && AsyncContext->OutstandingEvent) {
        ResetEvent(AsyncContext->OutstandingEvent);
    }

    Success = PostQueuedCompletionStatus(AsyncContext->IoCompletionPort,
                                         0,
                                         0,
                                         &Work->Iocp.Overlapped);
    if (!Success) {
        SYS_ERROR(PostQueuedCompletionStatus_AsyncSubmit);
        goto InlineWork;
    }

    return S_OK;

InlineWork:

    Work->Flags.InlineDispatch = TRUE;
    Result = Work->Step(Work);

    if (Result == S_FALSE) {
        Result = PerfectHashAsyncRequeueWork(Work);
        if (FAILED(Result)) {
            return PerfectHashAsyncCompleteWork(Work, Result);
        }
        return S_OK;
    }

    return PerfectHashAsyncCompleteWork(Work, Result);
}

_Use_decl_annotations_
VOID
PerfectHashAsyncWait(
    PPERFECT_HASH_ASYNC_CONTEXT AsyncContext
    )
/*++

Routine Description:

    Waits for all outstanding async work to complete.

Arguments:

    AsyncContext - Supplies a pointer to the async context.

Return Value:

    None.

--*/
{
    if (!ARGUMENT_PRESENT(AsyncContext)) {
        return;
    }

    if (AsyncContext->OutstandingEvent) {
        WaitForSingleObject(AsyncContext->OutstandingEvent, INFINITE);
    }
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
