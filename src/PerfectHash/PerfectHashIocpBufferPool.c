/*++

Copyright (c) 2024-2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashIocpBufferPool.c

Abstract:

    This module implements IOCP buffer pool routines used by the IOCP backend
    for overlapped file I/O.  Buffers are allocated on demand from size-class
    pools and returned to SLIST free lists for reuse.

--*/

#include "stdafx.h"
#include "PerfectHashIocpBufferPool.h"

PERFECT_HASH_IOCP_BUFFER_POOL_INITIALIZE PerfectHashIocpBufferPoolInitialize;
PERFECT_HASH_IOCP_BUFFER_POOL_ACQUIRE PerfectHashIocpBufferPoolAcquire;
PERFECT_HASH_IOCP_BUFFER_POOL_RELEASE PerfectHashIocpBufferPoolRelease;
PERFECT_HASH_IOCP_BUFFER_POOL_RUNDOWN PerfectHashIocpBufferPoolRundown;

static
_Must_inspect_result_
HRESULT
PerfectHashIocpBufferPoolAllocate(
    _In_ PRTL Rtl,
    _In_opt_ PALLOCATOR Allocator,
    _In_ PPERFECT_HASH_IOCP_BUFFER_POOL Pool,
    _Outptr_ PPERFECT_HASH_IOCP_BUFFER *BufferPointer
    )
{
    ULONG Pages;
    HRESULT Result;
    HANDLE ProcessHandle;
    PVOID BaseAddress;
    ULONGLONG AllocationSize;
    ULONGLONG UsableBufferSize;
    ULONGLONG PayloadSize;
    PPERFECT_HASH_IOCP_BUFFER Buffer;

    UNREFERENCED_PARAMETER(Allocator);

    if (!ARGUMENT_PRESENT(Rtl) ||
        !ARGUMENT_PRESENT(Pool) ||
        !ARGUMENT_PRESENT(BufferPointer)) {
        return E_POINTER;
    }

    *BufferPointer = NULL;
    BaseAddress = NULL;
    UsableBufferSize = 0;
    PayloadSize = Pool->PayloadSize;

    ProcessHandle = Pool->ProcessHandle;
    if (!ProcessHandle) {
        ProcessHandle = GetCurrentProcess();
    }

    AllocationSize = Pool->AllocationSize;
    if (AllocationSize == 0) {
        return E_INVALIDARG;
    }

    if (Pool->Flags & PERFECT_HASH_IOCP_BUFFER_POOL_FLAG_GUARD_PAGES) {
        Pages = (ULONG)BYTES_TO_PAGES(AllocationSize);
        if (Pages == 0) {
            Pages = 1;
        }

        Result = Rtl->Vtbl->CreateBuffer(Rtl,
                                         &ProcessHandle,
                                         Pages,
                                         NULL,
                                         &UsableBufferSize,
                                         &BaseAddress);
        if (FAILED(Result)) {
            return Result;
        }

        AllocationSize = (
            ((ULONGLONG)Pages + 1) * (ULONGLONG)PAGE_SIZE
        );
        if (UsableBufferSize <= PERFECT_HASH_IOCP_BUFFER_HEADER_SIZE) {
            Rtl->Vtbl->DestroyBuffer(Rtl,
                                     ProcessHandle,
                                     &BaseAddress,
                                     AllocationSize);
            return E_FAIL;
        }

        PayloadSize = UsableBufferSize -
                      (ULONGLONG)PERFECT_HASH_IOCP_BUFFER_HEADER_SIZE;

    } else {

        BaseAddress = VirtualAllocEx(ProcessHandle,
                                     NULL,
                                     AllocationSize,
                                     MEM_RESERVE | MEM_COMMIT,
                                     PAGE_READWRITE);
        if (!BaseAddress) {
            SYS_ERROR(VirtualAllocEx);
            return E_OUTOFMEMORY;
        }
    }

    Buffer = (PPERFECT_HASH_IOCP_BUFFER)BaseAddress;
    Rtl->RtlZeroMemory(Buffer, PERFECT_HASH_IOCP_BUFFER_HEADER_SIZE);
    Buffer->SizeOfStruct = sizeof(*Buffer);
    Buffer->PayloadOffset = PERFECT_HASH_IOCP_BUFFER_HEADER_SIZE;
    Buffer->PayloadSize = PayloadSize;
    Buffer->AllocationSize = AllocationSize;
    Buffer->BytesWritten = 0;
    Buffer->OwnerPool = Pool;
    Buffer->Flags = 0;

    if (Pool->Flags & PERFECT_HASH_IOCP_BUFFER_POOL_FLAG_GUARD_PAGES) {
        Buffer->Flags |= PERFECT_HASH_IOCP_BUFFER_FLAG_GUARD_PAGES;
    }

    if (Pool->Flags & PERFECT_HASH_IOCP_BUFFER_POOL_FLAG_OVERSIZE) {
        Buffer->Flags |= PERFECT_HASH_IOCP_BUFFER_FLAG_OVERSIZE;
    }

    if (Pool->BufferList) {
        Pool->BufferList->Vtbl->InsertTail(Pool->BufferList,
                                           &Buffer->ListEntry);
    }

    *BufferPointer = Buffer;
    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashIocpBufferPoolInitialize(
    PRTL Rtl,
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool,
    ULONGLONG PayloadSize,
    USHORT NumaNode,
    ULONG Flags,
    HANDLE ProcessHandle,
    PGUARDED_LIST BufferList
    )
{
    if (!ARGUMENT_PRESENT(Rtl) || !ARGUMENT_PRESENT(Pool)) {
        return E_POINTER;
    }

    if (!PayloadSize) {
        return E_INVALIDARG;
    }

    Rtl->RtlZeroMemory(Pool, sizeof(*Pool));
    InitializeSListHead(&Pool->FreeList);
    InitializeListHead(&Pool->ListEntry);

    Pool->SizeOfStruct = sizeof(*Pool);
    Pool->PayloadSize = PayloadSize;
    Pool->AllocationSize = PERFECT_HASH_IOCP_BUFFER_HEADER_SIZE + PayloadSize;
    Pool->Flags = Flags | PERFECT_HASH_IOCP_BUFFER_POOL_FLAG_INITIALIZED;
    Pool->NumaNode = NumaNode;
    Pool->ProcessHandle = ProcessHandle;
    Pool->BufferList = BufferList;

    return S_OK;
}

_Use_decl_annotations_
HRESULT
PerfectHashIocpBufferPoolAcquire(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool,
    PPERFECT_HASH_IOCP_BUFFER *BufferPointer
    )
{
    PSLIST_ENTRY Entry;
    PPERFECT_HASH_IOCP_BUFFER Buffer;

    if (!ARGUMENT_PRESENT(Pool) || !ARGUMENT_PRESENT(BufferPointer)) {
        return E_POINTER;
    }

    *BufferPointer = NULL;

    Entry = InterlockedPopEntrySList(&Pool->FreeList);
    if (Entry) {
        Buffer = CONTAINING_RECORD(Entry,
                                   PERFECT_HASH_IOCP_BUFFER,
                                   FreeListEntry);
        *BufferPointer = Buffer;
        return S_OK;
    }

    return PerfectHashIocpBufferPoolAllocate(Rtl,
                                             Allocator,
                                             Pool,
                                             BufferPointer);
}

_Use_decl_annotations_
VOID
PerfectHashIocpBufferPoolRelease(
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool,
    PPERFECT_HASH_IOCP_BUFFER Buffer
    )
{
    if (!ARGUMENT_PRESENT(Pool) || !ARGUMENT_PRESENT(Buffer)) {
        return;
    }

    Buffer->BytesWritten = 0;
    InterlockedPushEntrySList(&Pool->FreeList, &Buffer->FreeListEntry);
}

_Use_decl_annotations_
VOID
PerfectHashIocpBufferPoolRundown(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool
    )
{
    UNREFERENCED_PARAMETER(Rtl);
    UNREFERENCED_PARAMETER(Allocator);

    if (!ARGUMENT_PRESENT(Pool)) {
        return;
    }

    InitializeSListHead(&Pool->FreeList);
    InitializeListHead(&Pool->ListEntry);
    ZeroMemory(Pool, sizeof(*Pool));
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
