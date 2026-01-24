/*++

Copyright (c) 2024-2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashIocpBufferPool.c

Abstract:

    This module implements IOCP buffer pool routines used by the IOCP backend
    for overlapped file I/O.

--*/

#include "stdafx.h"
#include "PerfectHashIocpBufferPool.h"

PERFECT_HASH_IOCP_BUFFER_POOL_CREATE PerfectHashIocpBufferPoolCreate;
PERFECT_HASH_IOCP_BUFFER_POOL_DESTROY PerfectHashIocpBufferPoolDestroy;
PERFECT_HASH_IOCP_BUFFER_POOL_POP PerfectHashIocpBufferPoolPop;
PERFECT_HASH_IOCP_BUFFER_POOL_PUSH PerfectHashIocpBufferPoolPush;

_Use_decl_annotations_
HRESULT
PerfectHashIocpBufferPoolCreate(
    PRTL Rtl,
    PHANDLE ProcessHandle,
    ULONG PageSize,
    ULONG NumberOfBuffers,
    ULONG NumberOfPagesPerBuffer,
    PULONG AdditionalProtectionFlags,
    PULONG AdditionalAllocationTypeFlags,
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool
    )
{
    ULONG Index;
    ULONG PayloadOffset;
    PBYTE Base;
    HRESULT Result;
    ULONGLONG UsableBufferSizeInBytes;
    ULONGLONG TotalBufferSizeInBytes;
    ULONGLONG BufferStrideInBytes;
    ULONGLONG PayloadSizeInBytes;
    PPERFECT_HASH_IOCP_BUFFER Buffer;
    PRTL_VTBL Vtbl;
    HANDLE TargetProcessHandle;
    PHANDLE TargetProcessHandlePointer;

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Pool)) {
        return E_POINTER;
    }

    if (!NumberOfBuffers || !NumberOfPagesPerBuffer) {
        return E_INVALIDARG;
    }

    if (!PageSize) {
        return E_INVALIDARG;
    }

    Vtbl = Rtl->Vtbl;

    ZeroMemory(Pool, sizeof(*Pool));
    InitializeSListHead(&Pool->ListHead);

    TargetProcessHandle = NULL;
    if (ARGUMENT_PRESENT(ProcessHandle)) {
        TargetProcessHandlePointer = ProcessHandle;
    } else {
        TargetProcessHandlePointer = &TargetProcessHandle;
    }

    Result = Vtbl->CreateMultipleBuffers(Rtl,
                                         TargetProcessHandlePointer,
                                         PageSize,
                                         NumberOfBuffers,
                                         NumberOfPagesPerBuffer,
                                         AdditionalProtectionFlags,
                                         AdditionalAllocationTypeFlags,
                                         &UsableBufferSizeInBytes,
                                         &TotalBufferSizeInBytes,
                                         &Pool->BaseAddress);

    if (FAILED(Result)) {
        return Result;
    }

    Pool->ProcessHandle = *TargetProcessHandlePointer;
    PayloadOffset = PERFECT_HASH_IOCP_BUFFER_HEADER_SIZE;

    if ((ULONGLONG)PayloadOffset >= UsableBufferSizeInBytes) {
        Vtbl->DestroyBuffer(Rtl,
                            TargetProcessHandle,
                            &Pool->BaseAddress,
                            TotalBufferSizeInBytes);
        return E_FAIL;
    }

    PayloadSizeInBytes = UsableBufferSizeInBytes - (ULONGLONG)PayloadOffset;
    BufferStrideInBytes = UsableBufferSizeInBytes + (ULONGLONG)PageSize;

    Pool->SizeOfStruct = sizeof(*Pool);
    Pool->PageSize = PageSize;
    Pool->NumberOfBuffers = NumberOfBuffers;
    Pool->NumberOfPagesPerBuffer = NumberOfPagesPerBuffer;
    Pool->PayloadOffset = PayloadOffset;
    Pool->UsableBufferSizeInBytes = UsableBufferSizeInBytes;
    Pool->PayloadSizeInBytes = PayloadSizeInBytes;
    Pool->BufferStrideInBytes = BufferStrideInBytes;
    Pool->TotalAllocationSizeInBytes = TotalBufferSizeInBytes;
    Pool->ProcessHandle = TargetProcessHandle;

    Base = (PBYTE)Pool->BaseAddress;
    for (Index = 0; Index < NumberOfBuffers; Index++) {
        Buffer = (PPERFECT_HASH_IOCP_BUFFER)Base;
        Buffer->SizeOfStruct = sizeof(*Buffer);
        Buffer->PayloadOffset = PayloadOffset;
        Buffer->PayloadSize = PayloadSizeInBytes;
        Buffer->BytesWritten = 0;
        Buffer->Flags = 0;
        Buffer->BucketIndex = 0;
        Buffer->FileId = 0;
        Buffer->NumaNode = 0;
        InterlockedPushEntrySList(&Pool->ListHead, &Buffer->ListEntry);
        Base += BufferStrideInBytes;
    }

    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashIocpBufferPoolDestroy(
    PRTL Rtl,
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool
    )
{
    HRESULT Result;

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Pool)) {
        return E_POINTER;
    }

    if (!Pool->BaseAddress) {
        return S_OK;
    }

    Result = Rtl->Vtbl->DestroyBuffer(Rtl,
                                      Pool->ProcessHandle,
                                      &Pool->BaseAddress,
                                      Pool->TotalAllocationSizeInBytes);
    if (FAILED(Result)) {
        return Result;
    }

    ZeroMemory(Pool, sizeof(*Pool));
    return S_OK;
}

_Use_decl_annotations_
PPERFECT_HASH_IOCP_BUFFER
PerfectHashIocpBufferPoolPop(
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool
    )
{
    PSLIST_ENTRY Entry;

    if (!ARGUMENT_PRESENT(Pool)) {
        return NULL;
    }

    Entry = InterlockedPopEntrySList(&Pool->ListHead);
    if (!Entry) {
        return NULL;
    }

    return CONTAINING_RECORD(Entry, PERFECT_HASH_IOCP_BUFFER, ListEntry);
}

_Use_decl_annotations_
VOID
PerfectHashIocpBufferPoolPush(
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool,
    PPERFECT_HASH_IOCP_BUFFER Buffer
    )
{
    if (!ARGUMENT_PRESENT(Pool)) {
        return;
    }

    if (!ARGUMENT_PRESENT(Buffer)) {
        return;
    }

    Buffer->BytesWritten = 0;
    InterlockedPushEntrySList(&Pool->ListHead, &Buffer->ListEntry);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
