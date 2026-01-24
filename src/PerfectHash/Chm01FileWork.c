/*++

Copyright (c) 2018-2024 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWork.c

Abstract:

    This module implement the file work callback routine for the CHM v1 algo
    implementation of the perfect hash library.

    The FileWorkCallbackChm01 routine is the main entry point for all file work
    that has been requested via FILE_WORK_ITEM structs and submitted via the
    PERFECT_HASH_CONTEXT's "file work" threadpool (in Chm01.c).

    Generic preparation, unmapping and closing functionality is also implemented
    by way of PrepareFileChm01, UnmapFileChm01 and CloseFileChm01 routines.

--*/

#include "stdafx.h"
#include "PerfectHashContextIocp.h"
#include "PerfectHashIocpBufferPool.h"

//
// File work callback array.
//

#define EXPAND_AS_CALLBACK(         \
    Verb, VUpper, Name, Upper,      \
    EofType, EofValue,              \
    Suffix, Extension, Stream, Base \
)                                   \
    Verb##Name##Chm01,

FILE_WORK_CALLBACK_IMPL *FileCallbacks[] = {
    NULL,
    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CALLBACK)
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CALLBACK)
    NULL
};

static
VOID
Chm01GetFileWorkBufferPoolState(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _Outptr_ PPERFECT_HASH_IOCP_BUFFER_POOL **PoolsPointer,
    _Out_ PULONG *PoolCountPointer,
    _Outptr_ PGUARDED_LIST *BufferListPointer,
    _Outptr_ PLIST_ENTRY *OversizeListPointer,
    _Outptr_ PULONG *OversizeCountPointer,
    _Outptr_ PSRWLOCK *PoolLockPointer,
    _Out_ PUSHORT NumaNodePointer
    )
{
    PPERFECT_HASH_IOCP_NODE Node;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(PoolsPointer) ||
        !ARGUMENT_PRESENT(PoolCountPointer) ||
        !ARGUMENT_PRESENT(BufferListPointer) ||
        !ARGUMENT_PRESENT(OversizeListPointer) ||
        !ARGUMENT_PRESENT(OversizeCountPointer) ||
        !ARGUMENT_PRESENT(PoolLockPointer) ||
        !ARGUMENT_PRESENT(NumaNodePointer)) {
        return;
    }

    Node = Context->IocpNode;
    if (Node) {
        *PoolsPointer = &Node->FileWorkBufferPools;
        *PoolCountPointer = &Node->FileWorkBufferPoolCount;
        *BufferListPointer = Node->FileWorkBufferList;
        *OversizeListPointer = &Node->FileWorkOversizePools;
        *OversizeCountPointer = &Node->FileWorkOversizePoolCount;
        *PoolLockPointer = &Node->FileWorkBufferPoolLock;
        *NumaNodePointer = (USHORT)Node->NodeId;
    } else {
        *PoolsPointer = &Context->FileWorkBufferPools;
        *PoolCountPointer = &Context->FileWorkBufferPoolCount;
        *BufferListPointer = Context->FileWorkBufferList;
        *OversizeListPointer = &Context->FileWorkOversizePools;
        *OversizeCountPointer = &Context->FileWorkOversizePoolCount;
        *PoolLockPointer = &Context->Lock;
        *NumaNodePointer = 0;
    }
}

static
HRESULT
Chm01EnsureFileWorkBufferPools(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _Outptr_ PPERFECT_HASH_IOCP_BUFFER_POOL *PoolsPointer,
    _Out_ PULONG PoolCountPointer,
    _Outptr_ PGUARDED_LIST *BufferListPointer,
    _Out_ PUSHORT NumaNodePointer
    )
{
    HRESULT Result;
    ULONG Flags;
    ULONG PoolIndex;
    ULONG PoolCount;
    ULONGLONG PayloadSize;
    PALLOCATOR Allocator;
    PRTL Rtl;
    PPERFECT_HASH_IOCP_BUFFER_POOL Pools;
    PPERFECT_HASH_IOCP_BUFFER_POOL *PoolArrayPointer;
    PULONG PoolCountField;
    PGUARDED_LIST BufferList;
    PLIST_ENTRY OversizeList;
    PULONG OversizeCount;
    PSRWLOCK PoolLock;
    USHORT NumaNode;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(PoolsPointer) ||
        !ARGUMENT_PRESENT(PoolCountPointer) ||
        !ARGUMENT_PRESENT(BufferListPointer) ||
        !ARGUMENT_PRESENT(NumaNodePointer)) {
        return E_POINTER;
    }

    Allocator = Context->Allocator;
    Rtl = Context->Rtl;
    if (!Allocator || !Rtl) {
        return E_UNEXPECTED;
    }

    Chm01GetFileWorkBufferPoolState(Context,
                                    &PoolArrayPointer,
                                    &PoolCountField,
                                    &BufferList,
                                    &OversizeList,
                                    &OversizeCount,
                                    &PoolLock,
                                    &NumaNode);

    if (!PoolArrayPointer || !PoolCountField || !BufferList) {
        return E_UNEXPECTED;
    }

    Pools = *PoolArrayPointer;
    if (!Pools) {
        AcquireSRWLockExclusive(PoolLock);
        Pools = *PoolArrayPointer;
        if (!Pools) {
            PoolCount = PERFECT_HASH_IOCP_BUFFER_CLASS_COUNT;
            Pools = (PPERFECT_HASH_IOCP_BUFFER_POOL)Allocator->Vtbl->Calloc(
                Allocator,
                PoolCount,
                sizeof(*Pools)
            );
            if (!Pools) {
                ReleaseSRWLockExclusive(PoolLock);
                return E_OUTOFMEMORY;
            }

            Flags = UseIocpBufferGuardPages(Context) ?
                PERFECT_HASH_IOCP_BUFFER_POOL_FLAG_GUARD_PAGES :
                0;
            for (PoolIndex = 0; PoolIndex < PoolCount; PoolIndex++) {
                PayloadSize =
                    PerfectHashIocpBufferGetPayloadSizeFromClassIndex(
                        (LONG)PoolIndex
                    );

                Result = PerfectHashIocpBufferPoolInitialize(
                    Rtl,
                    &Pools[PoolIndex],
                    PayloadSize,
                    NumaNode,
                    Flags,
                    Context->ProcessHandle,
                    BufferList
                );

                if (FAILED(Result)) {
                    ReleaseSRWLockExclusive(PoolLock);
                    Allocator->Vtbl->FreePointer(
                        Allocator,
                        (PVOID *)&Pools
                    );
                    return Result;
                }
            }

            *PoolArrayPointer = Pools;
            *PoolCountField = PoolCount;
        }
        ReleaseSRWLockExclusive(PoolLock);
    }

    *PoolsPointer = Pools;
    *PoolCountPointer = *PoolCountField;
    *BufferListPointer = BufferList;
    *NumaNodePointer = NumaNode;
    return S_OK;
}

static
HRESULT
Chm01GetOversizeFileWorkBufferPool(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONGLONG PayloadSize,
    _Outptr_ PPERFECT_HASH_IOCP_BUFFER_POOL *PoolPointer
    )
{
    HRESULT Result;
    ULONG Flags;
    PALLOCATOR Allocator;
    PRTL Rtl;
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool;
    PPERFECT_HASH_IOCP_BUFFER_POOL *PoolArrayPointer;
    PULONG PoolCountField;
    PGUARDED_LIST BufferList;
    PLIST_ENTRY OversizeList;
    PLIST_ENTRY Entry;
    PULONG OversizeCount;
    PSRWLOCK PoolLock;
    USHORT NumaNode;

    if (!ARGUMENT_PRESENT(Context) || !ARGUMENT_PRESENT(PoolPointer)) {
        return E_POINTER;
    }

    Allocator = Context->Allocator;
    Rtl = Context->Rtl;
    if (!Allocator || !Rtl) {
        return E_UNEXPECTED;
    }

    Chm01GetFileWorkBufferPoolState(Context,
                                    &PoolArrayPointer,
                                    &PoolCountField,
                                    &BufferList,
                                    &OversizeList,
                                    &OversizeCount,
                                    &PoolLock,
                                    &NumaNode);

    if (!OversizeList || !OversizeCount || !PoolLock) {
        return E_UNEXPECTED;
    }

    AcquireSRWLockExclusive(PoolLock);

    Entry = OversizeList->Flink;
    while (Entry != OversizeList) {
        Pool = CONTAINING_RECORD(Entry,
                                 PERFECT_HASH_IOCP_BUFFER_POOL,
                                 ListEntry);
        if (Pool->PayloadSize == PayloadSize) {
            ReleaseSRWLockExclusive(PoolLock);
            *PoolPointer = Pool;
            return S_OK;
        }

        Entry = Entry->Flink;
    }

    Pool = (PPERFECT_HASH_IOCP_BUFFER_POOL)Allocator->Vtbl->Calloc(
        Allocator,
        1,
        sizeof(*Pool)
    );
    if (!Pool) {
        ReleaseSRWLockExclusive(PoolLock);
        return E_OUTOFMEMORY;
    }

    Flags = PERFECT_HASH_IOCP_BUFFER_POOL_FLAG_OVERSIZE;
    if (UseIocpBufferGuardPages(Context)) {
        Flags |= PERFECT_HASH_IOCP_BUFFER_POOL_FLAG_GUARD_PAGES;
    }
    Result = PerfectHashIocpBufferPoolInitialize(Rtl,
                                                 Pool,
                                                 PayloadSize,
                                                 NumaNode,
                                                 Flags,
                                                 Context->ProcessHandle,
                                                 BufferList);

    if (FAILED(Result)) {
        Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&Pool);
        ReleaseSRWLockExclusive(PoolLock);
        return Result;
    }

    InsertTailList(OversizeList, &Pool->ListEntry);
    (*OversizeCount)++;
    ReleaseSRWLockExclusive(PoolLock);

    *PoolPointer = Pool;
    return S_OK;
}

static
HRESULT
Chm01GetFileWorkBufferPool(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONGLONG RequiredPayloadBytes,
    _Outptr_ PPERFECT_HASH_IOCP_BUFFER_POOL *PoolPointer
    )
{
    HRESULT Result;
    LONG ClassIndex;
    ULONG PoolCount;
    ULONGLONG PayloadSize;
    PRTL Rtl;
    PPERFECT_HASH_IOCP_BUFFER_POOL Pools;
    PGUARDED_LIST BufferList;
    USHORT NumaNode;

    if (!ARGUMENT_PRESENT(Context) || !ARGUMENT_PRESENT(PoolPointer)) {
        return E_POINTER;
    }

    Rtl = Context->Rtl;
    if (!Rtl) {
        return E_UNEXPECTED;
    }

    ClassIndex = PerfectHashIocpBufferGetClassIndex(Rtl,
                                                    RequiredPayloadBytes);
    if (ClassIndex >= 0) {
        Result = Chm01EnsureFileWorkBufferPools(Context,
                                                &Pools,
                                                &PoolCount,
                                                &BufferList,
                                                &NumaNode);
        if (FAILED(Result)) {
            return Result;
        }

        if ((ULONG)ClassIndex >= PoolCount) {
            return E_INVALIDARG;
        }

        *PoolPointer = &Pools[ClassIndex];
        return S_OK;
    }

    PayloadSize = Rtl->RoundUpPowerOfTwo64(RequiredPayloadBytes);
    if (PayloadSize == 0) {
        return E_INVALIDARG;
    }

    return Chm01GetOversizeFileWorkBufferPool(Context,
                                              PayloadSize,
                                              PoolPointer);
}

static
HRESULT
Chm01AcquireFileWorkBuffer(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG FileIndex,
    _In_ ULONGLONG RequiredPayloadBytes,
    _Outptr_ PPERFECT_HASH_IOCP_BUFFER_POOL *PoolPointer,
    _Outptr_ PPERFECT_HASH_IOCP_BUFFER *BufferPointer
    )
{
    HRESULT Result;
    PALLOCATOR Allocator;
    PRTL Rtl;
    PPERFECT_HASH_IOCP_BUFFER Buffer;
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool;

    UNREFERENCED_PARAMETER(FileIndex);

    if (!ARGUMENT_PRESENT(BufferPointer) || !ARGUMENT_PRESENT(PoolPointer)) {
        return E_POINTER;
    }

    *BufferPointer = NULL;
    *PoolPointer = NULL;

    if (!ARGUMENT_PRESENT(Context)) {
        return E_POINTER;
    }

    Allocator = Context->Allocator;
    Rtl = Context->Rtl;
    if (!Allocator || !Rtl) {
        return E_UNEXPECTED;
    }

    Result = Chm01GetFileWorkBufferPool(Context,
                                        RequiredPayloadBytes,
                                        &Pool);
    if (FAILED(Result)) {
        return Result;
    }

    Result = PerfectHashIocpBufferPoolAcquire(Rtl,
                                              Allocator,
                                              Pool,
                                              &Buffer);
    if (FAILED(Result)) {
        return Result;
    }

    if (Buffer->PayloadSize < RequiredPayloadBytes) {
        PerfectHashIocpBufferPoolRelease(Pool, Buffer);
        return E_FAIL;
    }

    *PoolPointer = Pool;
    *BufferPointer = Buffer;
    return S_OK;
}

static
VOID
Chm01ClearFileWorkBuffer(
    _In_ PPERFECT_HASH_FILE File
    )
{
    if (!ARGUMENT_PRESENT(File)) {
        return;
    }

    File->IocpBufferPool = NULL;
    File->IocpBuffer = NULL;
    File->BaseAddress = NULL;
}

static
VOID
Chm01ReleaseIocpBuffer(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPERFECT_HASH_IOCP_BUFFER_POOL Pool,
    _In_ PPERFECT_HASH_IOCP_BUFFER Buffer
    )
{
    UNREFERENCED_PARAMETER(Context);

    if (!ARGUMENT_PRESENT(Pool) || !ARGUMENT_PRESENT(Buffer)) {
        return;
    }

    PerfectHashIocpBufferPoolRelease(Pool, Buffer);
}

static
VOID
Chm01ReleaseFileWorkBuffer(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PPERFECT_HASH_FILE File
    )
{
    if (!ARGUMENT_PRESENT(File)) {
        return;
    }

    if (File->IocpBufferPool && File->IocpBuffer) {
        Chm01ReleaseIocpBuffer(Context,
                               File->IocpBufferPool,
                               File->IocpBuffer);
    }

    Chm01ClearFileWorkBuffer(File);
}

static
VOID
Chm01RequeueFileWorkItem(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PFILE_WORK_ITEM Item
    )
{
    if (!ARGUMENT_PRESENT(Context) || !ARGUMENT_PRESENT(Item)) {
        return;
    }

    InsertTailFileWork(Context, &Item->ListEntry);
    PerfectHashContextSubmitFileWork(Context);
}

static
HRESULT
Chm01EnsureFileWorkBuffer(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ ULONG FileIndex,
    _In_ ULONGLONG RequiredPayloadBytes,
    _In_ PPERFECT_HASH_FILE File
    )
{
    HRESULT Result;
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool;
    PPERFECT_HASH_IOCP_BUFFER Buffer;

    if (!ARGUMENT_PRESENT(Context) || !ARGUMENT_PRESENT(File)) {
        return E_POINTER;
    }

    if (File->IocpBuffer) {
        if (File->IocpBuffer->PayloadSize < RequiredPayloadBytes) {
            Chm01ReleaseFileWorkBuffer(Context, File);
        } else {
            File->BaseAddress = PerfectHashIocpBufferPayload(File->IocpBuffer);
            return S_OK;
        }
    }

    Result = Chm01AcquireFileWorkBuffer(Context,
                                        FileIndex,
                                        RequiredPayloadBytes,
                                        &Pool,
                                        &Buffer);
    if (FAILED(Result)) {
        return Result;
    }

    File->IocpBufferPool = Pool;
    File->IocpBuffer = Buffer;
    File->BaseAddress = PerfectHashIocpBufferPayload(Buffer);

    return S_OK;
}

static
ULONGLONG
Chm01AdjustRequiredPayloadBytes(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ FILE_ID FileId,
    _In_ ULONGLONG RequiredPayloadBytes
    )
{
    ULONGLONG Adjusted;
    ULONGLONG BucketCount;
    PRTL Rtl;
    PCEOF_INIT Eof;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_KEYS Keys;

    if (!ARGUMENT_PRESENT(Context)) {
        return RequiredPayloadBytes;
    }

    Rtl = Context->Rtl;
    Table = Context->Table;
    Keys = Table ? Table->Keys : NULL;

    if (!Rtl || !Table || !Keys) {
        return RequiredPayloadBytes;
    }

    if (!IsValidFileId(FileId)) {
        return RequiredPayloadBytes;
    }

    Eof = &EofInits[FileId];
    switch (Eof->Type) {
        case EofInitTypeNumberOfKeysMultiplier:
            BucketCount = Rtl->RoundUpPowerOfTwo64(Keys->NumberOfKeys.QuadPart);
            if (BucketCount == 0 || Eof->Multiplier == 0) {
                return RequiredPayloadBytes;
            }
            if (BucketCount > ((ULONGLONG)-1) / Eof->Multiplier) {
                return RequiredPayloadBytes;
            }
            Adjusted = BucketCount * Eof->Multiplier;
            break;

        case EofInitTypeNumberOfTableElementsMultiplier:
            if (!Table->TableInfoOnDisk) {
                return RequiredPayloadBytes;
            }
            BucketCount = Rtl->RoundUpPowerOfTwo64(
                Table->TableInfoOnDisk->NumberOfTableElements.QuadPart
            );
            if (BucketCount == 0 || Eof->Multiplier == 0) {
                return RequiredPayloadBytes;
            }
            if (BucketCount > ((ULONGLONG)-1) / Eof->Multiplier) {
                return RequiredPayloadBytes;
            }
            Adjusted = BucketCount * Eof->Multiplier;
            break;

        case EofInitTypeNull:
        case EofInitTypeDefault:
        case EofInitTypeAssignedSize:
        case EofInitTypeFixed:
        case EofInitTypeNumberOfPages:
        case EofInitTypeInvalid:
        default:
            return RequiredPayloadBytes;
    }

    if (Adjusted < Context->SystemAllocationGranularity) {
        Adjusted = Context->SystemAllocationGranularity;
    }

    if (Adjusted < RequiredPayloadBytes) {
        Adjusted = RequiredPayloadBytes;
    }

    return Adjusted;
}

typedef struct _CHM01_FILE_IOCP_WRITE {
    PERFECT_HASH_IOCP_WORK Iocp;
    PPERFECT_HASH_CONTEXT Context;
    PFILE_WORK_ITEM Item;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_IOCP_BUFFER_POOL Pool;
    PPERFECT_HASH_IOCP_BUFFER Buffer;
    ULONG BytesToWrite;
    ULONG Padding1;
} CHM01_FILE_IOCP_WRITE;
typedef CHM01_FILE_IOCP_WRITE *PCHM01_FILE_IOCP_WRITE;

static
HRESULT
Chm01FileWriteIocpCompletionCallback(
    _In_ PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
    _In_ ULONG_PTR CompletionKey,
    _In_ LPOVERLAPPED Overlapped,
    _In_ DWORD NumberOfBytesTransferred,
    _In_ BOOL Success
    )
{
    ULONG EventIndex;
    ULONG LastError;
    HANDLE Event;
    HRESULT Result;
    PFILE_WORK_ITEM Item;
    PPERFECT_HASH_CONTEXT Context;
    PCHM01_FILE_IOCP_WRITE WorkItem;

    UNREFERENCED_PARAMETER(ContextIocp);
    UNREFERENCED_PARAMETER(CompletionKey);

    WorkItem = (PCHM01_FILE_IOCP_WRITE)Overlapped;
    if (!WorkItem) {
        return E_POINTER;
    }

    Context = WorkItem->Context;
    Item = WorkItem->Item;
    Event = NULL;
    Result = S_OK;
    LastError = ERROR_SUCCESS;

    if (!Success) {
        LastError = GetLastError();
        SYS_ERROR(WriteFile_IocpCompletion);
        Result = PH_E_SYSTEM_CALL_FAILED;
    } else if (NumberOfBytesTransferred != WorkItem->BytesToWrite) {
        LastError = ERROR_WRITE_FAULT;
        SetLastError(LastError);
        Result = PH_E_SYSTEM_CALL_FAILED;
    }

    if (WorkItem->Buffer) {
        WorkItem->Buffer->BytesWritten = NumberOfBytesTransferred;
    }

    if (WorkItem->Pool && WorkItem->Buffer) {
        Chm01ReleaseIocpBuffer(Context, WorkItem->Pool, WorkItem->Buffer);
    }

    if (WorkItem->File) {
        Chm01ClearFileWorkBuffer(WorkItem->File);
    }

    if (Item) {
        Item->LastResult = Result;
        Item->LastError = (LONG)LastError;
        if (FAILED(Result)) {
            InterlockedIncrement(&Item->NumberOfErrors);
        }
    }

    if (Context && Item && !IsCloseFileWorkId(Item->FileWorkId)) {
        EventIndex = FileWorkIdToEventIndex(Item->FileWorkId);
        Event = *(&Context->FirstPreparedEvent + EventIndex);
        if (Event) {
            SetEvent(Event);
        }
    }

    if (Context && Context->Allocator) {
        Context->Allocator->Vtbl->FreePointer(
            Context->Allocator,
            (PVOID *)&WorkItem
        );
    }

    return S_OK;
}

static
HRESULT
Chm01WriteFileFromBuffer(
    _In_ PPERFECT_HASH_CONTEXT Context,
    _In_ PFILE_WORK_ITEM Item,
    _In_ PPERFECT_HASH_FILE File,
    _In_ PPERFECT_HASH_IOCP_BUFFER_POOL Pool,
    _In_ PPERFECT_HASH_IOCP_BUFFER Buffer
    )
{
    BOOL Success;
    DWORD BytesWritten;
    DWORD BytesToWrite;
    ULONG LastError;
    HRESULT Result = S_OK;
    LARGE_INTEGER BytesRemaining;
    PVOID BaseAddress;
    PALLOCATOR Allocator;
    PCHM01_FILE_IOCP_WRITE WorkItem;

    if (!ARGUMENT_PRESENT(Context) ||
        !ARGUMENT_PRESENT(File) ||
        !ARGUMENT_PRESENT(Item) ||
        !ARGUMENT_PRESENT(Pool) ||
        !ARGUMENT_PRESENT(Buffer)) {
        return E_POINTER;
    }

    BytesRemaining.QuadPart = File->NumberOfBytesWritten.QuadPart;
    if (BytesRemaining.QuadPart <= 0) {
        PerfectHashIocpBufferPoolRelease(Pool, Buffer);
        return S_OK;
    }

#ifdef PH_WINDOWS
    if (BytesRemaining.QuadPart > (LONGLONG)Buffer->PayloadSize ||
        BytesRemaining.QuadPart >= (LONGLONG)(1ULL << 30)) {
        BOOL LogLargeWrite;
        WCHAR LogBuffer[512];
        int CharCount;
        HANDLE LogHandle;
        DWORD BytesWrittenLog;

        LogLargeWrite = (GetEnvironmentVariableW(
            L"PH_LOG_LARGE_FILE_WRITE", NULL, 0) != 0);

        if (LogLargeWrite) {
            CharCount = _snwprintf_s(
                LogBuffer,
                ARRAYSIZE(LogBuffer),
                _TRUNCATE,
                L"PH_LARGE_WRITE: FileId=%lu FileWorkId=%lu "
                L"BytesRemaining=%lld PayloadSize=%llu "
                L"EndOfFile=%lld Path=%wZ\n",
                (ULONG)Item->FileId,
                (ULONG)Item->FileWorkId,
                BytesRemaining.QuadPart,
                Buffer->PayloadSize,
                File->FileInfo.EndOfFile.QuadPart,
                (File->Path ? &File->Path->FullPath : &NullUnicodeString)
            );

            if (CharCount > 0) {
                OutputDebugStringW(LogBuffer);
                LogHandle = CreateFileW(L"PerfectHashLargeWrite.log",
                                        FILE_APPEND_DATA,
                                        FILE_SHARE_READ,
                                        NULL,
                                        OPEN_ALWAYS,
                                        FILE_ATTRIBUTE_NORMAL,
                                        NULL);
                if (IsValidHandle(LogHandle)) {
                    WriteFile(LogHandle,
                              LogBuffer,
                              (DWORD)(CharCount * sizeof(WCHAR)),
                              &BytesWrittenLog,
                              NULL);
                    CloseHandle(LogHandle);
                }
            }
        }

        PH_RAISE(PH_E_INVALID_END_OF_FILE);
    }
#endif

    if (BytesRemaining.HighPart != 0) {
        PH_RAISE(PH_E_INVALID_END_OF_FILE);
    }

    BaseAddress = PerfectHashIocpBufferPayload(Buffer);
    BytesToWrite = BytesRemaining.LowPart;
    BytesWritten = 0;

    Allocator = Context->Allocator;
    if (!Allocator) {
        Result = E_UNEXPECTED;
        goto End;
    }

    WorkItem = (PCHM01_FILE_IOCP_WRITE)Allocator->Vtbl->Calloc(
        Allocator,
        1,
        sizeof(*WorkItem)
    );
    if (!WorkItem) {
        Result = E_OUTOFMEMORY;
        goto End;
    }

    WorkItem->Iocp.Signature = PH_IOCP_WORK_SIGNATURE;
    WorkItem->Iocp.Flags = PH_IOCP_WORK_FLAG_FILE_IO;
    WorkItem->Iocp.CompletionCallback =
        Chm01FileWriteIocpCompletionCallback;
    WorkItem->Context = Context;
    WorkItem->Item = Item;
    WorkItem->File = File;
    WorkItem->Pool = Pool;
    WorkItem->Buffer = Buffer;
    WorkItem->BytesToWrite = BytesToWrite;

    Success = WriteFile(File->FileHandle,
                        BaseAddress,
                        BytesToWrite,
                        &BytesWritten,
                        &WorkItem->Iocp.Overlapped);

    if (!Success) {
        LastError = GetLastError();
        if (LastError != ERROR_IO_PENDING) {
            SetLastError(LastError);
            SYS_ERROR(WriteFile);
            Result = PH_E_SYSTEM_CALL_FAILED;
            Allocator->Vtbl->FreePointer(Allocator, (PVOID *)&WorkItem);
            goto End;
        }
    }

    return S_FALSE;

End:

    Chm01ReleaseIocpBuffer(Context, Pool, Buffer);
    Chm01ClearFileWorkBuffer(File);
    return Result;
}

//
// Forward decls.
//

PREPARE_FILE PrepareFileChm01;
UNMAP_FILE UnmapFileChm01;
CLOSE_FILE CloseFileChm01;

PERFECT_HASH_FILE_WORK_ITEM_CALLBACK FileWorkItemCallbackChm01;

//
// Begin method implementations.
//

#ifdef PH_WINDOWS
PERFECT_HASH_FILE_WORK_CALLBACK FileWorkCallbackChm01;

_Use_decl_annotations_
VOID
FileWorkCallbackChm01(
    PTP_CALLBACK_INSTANCE Instance,
    PPERFECT_HASH_CONTEXT Context,
    PLIST_ENTRY ListEntry
    )
/*++

Routine Description:

    This routine is the callback entry point for file-oriented work we want
    to perform in the file work threadpool context.

Arguments:

    Instance - Supplies a pointer to the callback instance for this invocation.

    Context - Supplies a pointer to the active context for the graph solving.

    ListEntry - Supplies a pointer to the list entry that was removed from the
        context's file work list head.

Return Value:

    None.

--*/
{
    PFILE_WORK_ITEM Item;

    Item = CONTAINING_RECORD(ListEntry, FILE_WORK_ITEM, ListEntry);
    Item->Instance = Instance;
    Item->Context = Context;

    FileWorkItemCallbackChm01(Item);
}
#endif

_Use_decl_annotations_
VOID
FileWorkItemCallbackChm01(
    PFILE_WORK_ITEM Item
    )
/*++

Routine Description:

    This routine is the callback entry point for file-oriented work we want
    to perform in the file work threadpool context.

Arguments:

    Item - Supplies a pointer to the file work item for this callback.

Return Value:

    None.

--*/
{
    PRTL Rtl;
    ULONG FileIndex;
    ULONG EventIndex;
    ULONG ContextFileIndex;
    ULONG DependentEventIndex;
    HRESULT Result = S_OK;
    PGRAPH_INFO Info;
    FILE_ID FileId;
    FILE_WORK_ID FileWorkId;
    FILE_WORK_ID PrepareWorkId;
    CONTEXT_FILE_ID ContextFileId;
    PFILE_WORK_CALLBACK_IMPL Impl = NULL;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_KEYS Keys;
    HANDLE Event;
    HANDLE DependentEvent = NULL;
    PPERFECT_HASH_PATH Path = NULL;
    PTABLE_INFO_ON_DISK TableInfo;
    PPERFECT_HASH_FILE *File = NULL;
    PPERFECT_HASH_FILE ContextFile = NULL;
    PPERFECT_HASH_CONTEXT Context;
    BOOLEAN IsContextFile = FALSE;
    BOOLEAN IsMakefileOrMainMkOrCMakeListsTextFile = FALSE;
    BOOLEAN HasSaveCallback = FALSE;
    BOOLEAN HasPrepareCallback = FALSE;

    //
    // Initialize aliases.
    //

    Context = Item->Context;
    Rtl = Context->Rtl;
    Info = (PGRAPH_INFO)Context->AlgorithmContext;
    Table = Context->Table;
    TableInfo = Table->TableInfoOnDisk;
    Keys = Table->Keys;

    ASSERT(!NoFileIo(Table));

    FileWorkId = Item->FileWorkId;
    ASSERT(IsValidFileWorkId(FileWorkId));

    //
    // Resolve the relevant file and event indices and associated pointers.
    // (Note that the close file work type does not use events.)
    //

    FileIndex = FileWorkIdToFileIndex(FileWorkId);
    File = &Table->FirstFile + FileIndex;

    if (!IsCloseFileWorkId(FileWorkId)) {
        EventIndex = FileWorkIdToEventIndex(FileWorkId);
        Event = *(&Context->FirstPreparedEvent + EventIndex);
    } else {
        EventIndex = (ULONG)-1;
        Event = NULL;
    }

    //
    // Set the file ID.
    //

    Item->FileId = FileId = FileWorkIdToFileId(FileWorkId);

    if (SkipContextFileWork(Context) &&
        IsValidContextFileId((CONTEXT_FILE_ID)FileId)) {
        Item->Flags.IsContextFile = TRUE;
        Item->LastResult = S_OK;
        Item->LastError = 0;
        if (Event) {
#ifdef PH_WINDOWS
            if (Item->Instance) {
                SetEventWhenCallbackReturns(Item->Instance, Event);
            } else {
                SetEvent(Event);
            }
#else
            SetEvent(Event);
#endif
        }
        return;
    }

    //
    // Determine if this is a context file.  Context files get treated
    // differently to normal table output files in that they are only
    // prepared and saved once per context instance.
    //

    ContextFileId = (CONTEXT_FILE_ID)FileId;

    if (!IsValidContextFileId(ContextFileId)) {

        //
        // This is not a context file.
        //

        ContextFileId = ContextFileNullId;

        //
        // Toggle a boolean if this is the Makefile, main.mk, or CMakeLists.txt
        // file; we need this later when creating the path.
        //

        IsMakefileOrMainMkOrCMakeListsTextFile = (
            FileId == FileMakefileFileId        ||
            FileId == FileMakefileMainMkFileId  ||
            FileId == FileCMakeListsTextFileId  ||
            FileId == FileRustCargoTomlFileId
        );

    } else {

        Item->Flags.IsContextFile = IsContextFile = TRUE;

        //
        // Override the table's file pointer with the context's one.
        //

        ContextFileIndex = ContextFileIdToContextFileIndex(ContextFileId);
        File = &Context->FirstFile + ContextFileIndex;

        if (IsPrepareFileWorkId(FileWorkId)) {

            Item->Flags.PrepareOnce = TRUE;

            ContextFile = *File;

            //
            // Context files only get one prepare call.  If ContextFile is not
            // NULL here, it means the file has already been prepared once, so
            // we can jump straight to the end.
            //

            if (ContextFile) {
                goto End;
            }
        }
    }

    Item->FilePointer = File;

    if (!IsCloseFileWorkId(FileWorkId)) {
        Impl = FileCallbacks[FileWorkId];
    } else {
        Impl = NULL;
    }

    if (IsPrepareFileWorkId(FileWorkId)) {

        PCEOF_INIT Eof;
        LARGE_INTEGER EndOfFile;
        ULONG NumberOfResizeEvents;
        ULARGE_INTEGER NumberOfTableElements;
        PPERFECT_HASH_PATH ExistingPath = NULL;
        PCUNICODE_STRING NewExtension = NULL;
        PCUNICODE_STRING NewDirectory = NULL;
        PCUNICODE_STRING NewBaseName = NULL;
        PCUNICODE_STRING NewStreamName = NULL;
        PCUNICODE_STRING AdditionalSuffix = NULL;
        UNICODE_STRING TestBaseName = { 0 };
        WCHAR TestBaseNameBuffer[MAX_PATH];
        PCSTRING TableNameA = NULL;
        USHORT TableNameChars = 0;
        USHORT TestBaseNameChars = 0;
        USHORT NameIndex = 0;

        Eof = &EofInits[FileWorkId];
        NewBaseName = GetFileWorkItemBaseName(FileWorkId);
        NewExtension = GetFileWorkItemExtension(FileWorkId);

        if (IsContextFile) {

            //
            // All context files are rooted within in the context's base
            // output directory.
            //

            NewDirectory = &Context->BaseOutputDirectory->Path->FullPath;

        } else {

            //
            // All table output files are rooted within in the table's output
            // directory.
            //

            NewDirectory = &Table->OutputDirectory->Path->FullPath;

            ASSERT(IsValidUnicodeString(NewDirectory));

            //
            // Initialize variables specific to the file work ID.
            //

            AdditionalSuffix = GetFileWorkItemSuffix(FileWorkId);
            NewStreamName = GetFileWorkItemStreamName(FileWorkId);

            if (FileId == FilePythonTestFileId) {
                TableNameA = &Table->Keys->File->Path->TableNameA;
                if (TableNameA->Length > 0) {
                    TableNameChars = (USHORT)TableNameA->Length;
                    TestBaseNameChars = (USHORT)(5 + TableNameChars);
                    if ((USHORT)(TestBaseNameChars + 1) <=
                        (USHORT)ARRAYSIZE(TestBaseNameBuffer)) {
                        TestBaseName.Buffer = TestBaseNameBuffer;
                        TestBaseName.Length = (
                            TestBaseNameChars * sizeof(WCHAR)
                        );
                        TestBaseName.MaximumLength = (
                            (TestBaseNameChars + 1) * sizeof(WCHAR)
                        );
                        TestBaseNameBuffer[0] = L't';
                        TestBaseNameBuffer[1] = L'e';
                        TestBaseNameBuffer[2] = L's';
                        TestBaseNameBuffer[3] = L't';
                        TestBaseNameBuffer[4] = L'_';
                        for (NameIndex = 0;
                             NameIndex < TableNameChars;
                             NameIndex++) {
                            TestBaseNameBuffer[5 + NameIndex] = (WCHAR)(
                                TableNameA->Buffer[NameIndex]
                            );
                        }
                        TestBaseNameBuffer[TestBaseNameChars] = L'\0';
                        NewBaseName = &TestBaseName;
                    }
                }
            }

            if (IsValidUnicodeString(NewStreamName)) {

                Item->Flags.PrepareOnce = TRUE;

                //
                // Streams don't need more than one prepare call, as their path
                // hangs off their owning file's path (and thus, will inherit
                // its scheduled rename), and their size never changes.  If we
                // dereference *File and it's non-NULL, it means the stream has
                // already been prepared, in which case, we can jump straight to
                // the end.
                //

                ASSERT(Eof->Type == EofInitTypeFixed);

                if (*File) {
                    goto End;
                }

                //
                // Streams are dependent upon their "owning" file, which always
                // reside before them.
                //

                DependentEvent = *(
                    &Context->FirstPreparedEvent +
                    (EventIndex - 1)
                );

            } else if (FileRequiresUuid(FileId) && !*File) {

                //
                // Generate a UUID the first time we prepare a VC Project file
                // or VS Solution file.
                //

                Result = RtlCreateUuidString(Rtl, &Item->Uuid);
                if (FAILED(Result)) {
                    goto End;
                }

            }
        }

        NumberOfResizeEvents = (ULONG)Context->NumberOfTableResizeEvents;
        NumberOfTableElements.QuadPart = (
            TableInfo->NumberOfTableElements.QuadPart
        );

        //
        // Default size for end-of-file is the system allocation granularity.
        //

        EndOfFile.QuadPart = Context->SystemAllocationGranularity;

        //
        // Initialize the end-of-file based on the relevant file work ID's
        // EOF_INIT structure.
        //

        switch (Eof->Type) {

            case EofInitTypeDefault:
                break;

            case EofInitTypeAssignedSize:
                EndOfFile.QuadPart = Info->AssignedSizeInBytes;
                break;

            case EofInitTypeFixed:
                EndOfFile.QuadPart = Eof->FixedValue;
                break;

            case EofInitTypeNumberOfKeysMultiplier:
                {
                    LONGLONG Computed = (LONGLONG)(
                        Keys->NumberOfKeys.QuadPart * Eof->Multiplier
                    );
                    if (Computed > EndOfFile.QuadPart) {
                        EndOfFile.QuadPart = Computed;
                    }
                }
                break;

            case EofInitTypeNumberOfTableElementsMultiplier:
                {
                    LONGLONG Computed = (LONGLONG)(
                        NumberOfTableElements.QuadPart * Eof->Multiplier
                    );
                    if (Computed > EndOfFile.QuadPart) {
                        EndOfFile.QuadPart = Computed;
                    }
                }
                break;

            case EofInitTypeNumberOfPages:
                EndOfFile.QuadPart = (
                    (ULONG_PTR)PAGE_SIZE *
                    (ULONG_PTR)Eof->NumberOfPages
                );
                break;

            case EofInitTypeNull:
            case EofInitTypeInvalid:
            default:
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
                return;
        }

        //
        // We create the path differently depending on whether or not it's
        // a context file (rooted in the context's base output directory) or
        // a table file (rooted in the table's output directory).  E.g.
        //
        //  Context's base output directory:
        //
        //      C:\Temp\output
        //
        //  Table's base output directory:
        //
        //      C:\Temp\output\KernelBase_2415_8192_Chm01_Crc32Rotate_And
        //

        if (IsContextFile) {

            //
            // Create a new path instance.
            //

            Result = Context->Vtbl->CreateInstance(Context,
                                                   NULL,
                                                   &IID_PERFECT_HASH_PATH,
                                                   (PVOID *)&Path);

            if (FAILED(Result)) {
                PH_ERROR(PerfectHashTableCreateInstance, Result);
                goto End;
            }

            //
            // Create the underlying path.
            //

            ExistingPath = Context->BaseOutputDirectory->Path;

            Result = Path->Vtbl->Create(Path,
                                        ExistingPath,   // ExistingPath
                                        NewDirectory,   // NewDirectory
                                        NULL,           // DirectorySuffix
                                        NewBaseName,    // NewBaseName
                                        NULL,           // BaseNameSuffix
                                        NewExtension,   // NewExtension
                                        NULL,           // NewStreamName
                                        NULL,           // Parts
                                        NULL);          // Reserved

            if (FAILED(Result)) {
                PH_ERROR(PerfectHashPathCreate, Result);
                goto End;
            }

        } else if (IsMakefileOrMainMkOrCMakeListsTextFile) {

            //
            // The Makefile and main.mk files are special as they're the only
            // files rooted in the table's output directory that don't have the
            // usual algo/hash/mask suffix appended to them; they need to be
            // named "Makefile" and "main.mk".  As with the context file logic
            // above, we handle this requirement by creating the path manually
            // instead of using PerfectHashTableCreatePath().
            //
            // Update: this also applies to CMakeLists.txt files.
            //

            //
            // Create a new path instance.
            //

            Result = Context->Vtbl->CreateInstance(Context,
                                                   NULL,
                                                   &IID_PERFECT_HASH_PATH,
                                                   (PVOID *)&Path);

            if (FAILED(Result)) {
                PH_ERROR(PerfectHashTableCreateInstance, Result);
                goto End;
            }

            //
            // Create the underlying path.
            //

            //
            // N.B. The ExistingPath parameter for PerfectHashPathCreate() is
            //      mandatory (and must be a valid path).  We have two choices
            //      here: we can use either the keys file path or the context's
            //      base output directory.  It doesn't matter which one, as the
            //      underlying path isn't used (because we supply overrides for
            //      NewDirectory, NewBaseName and NewExtension).
            //

            ExistingPath = Context->BaseOutputDirectory->Path;

            Result = Path->Vtbl->Create(Path,
                                        ExistingPath,   // ExistingPath
                                        NewDirectory,   // NewDirectory
                                        NULL,           // DirectorySuffix
                                        NewBaseName,    // NewBaseName
                                        NULL,           // BaseNameSuffix
                                        NewExtension,   // NewExtension
                                        NULL,           // NewStreamName
                                        NULL,           // Parts
                                        NULL);          // Reserved

            if (FAILED(Result)) {
                PH_ERROR(PerfectHashPathCreate, Result);
                goto End;
            }

        } else {

            //
            // This is a normal table file.
            //

            ExistingPath = Table->Keys->File->Path;

            Result = PerfectHashTableCreatePath(Table,
                                                ExistingPath,
                                                &NumberOfResizeEvents,
                                                &NumberOfTableElements,
                                                Table->AlgorithmId,
                                                Table->HashFunctionId,
                                                Table->MaskFunctionId,
                                                NewDirectory,
                                                NewBaseName,
                                                AdditionalSuffix,
                                                NewExtension,
                                                NewStreamName,
                                                &Path,
                                                NULL);


            if (FAILED(Result)) {
                PH_ERROR(PerfectHashTableCreatePath, Result);
                goto End;
            }

        }

        //
        // Sanity check we're indicating success at this point.
        //

        if (FAILED(Result)) {
            PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        }

#ifdef PH_WINDOWS
        if (EndOfFile.QuadPart >= (LONGLONG)(1ULL << 30)) {
            BOOL LogLargeEof;
            BOOL FailLargeEof;
            ULONGLONG NumberOfKeys;
            ULONGLONG LargeEofTableElements;
            ULONGLONG AssignedSizeInBytes;
            ULONG AllocationGranularity;
            PCUNICODE_STRING Extension;
            PCUNICODE_STRING Suffix;
            WCHAR LogBuffer[512];
            int CharCount;
            HANDLE LogHandle;
            DWORD BytesWritten;

            LogLargeEof = (GetEnvironmentVariableW(
                L"PH_LOG_LARGE_EOF", NULL, 0) != 0);
            FailLargeEof = (GetEnvironmentVariableW(
                L"PH_FAIL_LARGE_EOF", NULL, 0) != 0);

            if (LogLargeEof || FailLargeEof) {
                NumberOfKeys = Keys ? Keys->NumberOfKeys.QuadPart : 0;
                LargeEofTableElements = (
                    TableInfo ?
                    TableInfo->NumberOfTableElements.QuadPart :
                    0
                );
                AssignedSizeInBytes = (
                    Info ? Info->AssignedSizeInBytes : 0
                );
                AllocationGranularity = (
                    Context ? Context->SystemAllocationGranularity : 0
                );
                Extension = GetFileWorkItemExtension(FileWorkId);
                Suffix = GetFileWorkItemSuffix(FileWorkId);

                CharCount = _snwprintf_s(
                    LogBuffer,
                    ARRAYSIZE(LogBuffer),
                    _TRUNCATE,
                    L"PH_LARGE_EOF: FileWorkId=%lu FileId=%lu "
                    L"EofType=%lu Multiplier=%llu EndOfFile=%lld "
                    L"Keys=%llu TableElements=%llu AssignedSize=%llu "
                    L"Granularity=%lu Suffix=%wZ Ext=%wZ Path=%wZ\n",
                    (ULONG)FileWorkId,
                    (ULONG)FileId,
                    (ULONG)Eof->Type,
                    (ULONGLONG)Eof->Multiplier,
                    EndOfFile.QuadPart,
                    NumberOfKeys,
                    LargeEofTableElements,
                    AssignedSizeInBytes,
                    AllocationGranularity,
                    Suffix ? Suffix : &NullUnicodeString,
                    Extension ? Extension : &NullUnicodeString,
                    Path ? &Path->FullPath : &NullUnicodeString
                );

                if (CharCount > 0) {
                    OutputDebugStringW(LogBuffer);

                    LogHandle = CreateFileW(L"PerfectHashLargeEof.log",
                                            FILE_APPEND_DATA,
                                            FILE_SHARE_READ,
                                            NULL,
                                            OPEN_ALWAYS,
                                            FILE_ATTRIBUTE_NORMAL,
                                            NULL);
                    if (IsValidHandle(LogHandle)) {
                        WriteFile(LogHandle,
                                  LogBuffer,
                                  (DWORD)(CharCount * sizeof(WCHAR)),
                                  &BytesWritten,
                                  NULL);
                        CloseHandle(LogHandle);
                    }
                }
            }

            if (FailLargeEof) {
                Result = PH_E_INVALID_END_OF_FILE;
                PH_ERROR(PrepareFileChm01_LargeEof, Result);
                goto End;
            }
        }
#endif

        Result = PrepareFileChm01(Table,
                                  Item,
                                  Path,
                                  &EndOfFile,
                                  DependentEvent);

        if (Result == S_FALSE) {
            Chm01RequeueFileWorkItem(Context, Item);
            return;
        }

        if (FAILED(Result)) {
            PH_ERROR(PrepareFileChm01, Result);
            goto End;
        }

        if (UseOverlappedIo(Context) && Impl) {
            ULONGLONG BufferPayloadBytes;

            if (!*File) {
                Result = E_UNEXPECTED;
                goto End;
            }

            BufferPayloadBytes = Chm01AdjustRequiredPayloadBytes(
                Context,
                FileId,
                (ULONGLONG)EndOfFile.QuadPart
            );

            Result = Chm01EnsureFileWorkBuffer(
                Context,
                FileIndex,
                BufferPayloadBytes,
                *File
            );
            if (FAILED(Result)) {
                goto End;
            }

            //
            // Reset bytes written for IOCP buffers.  Prepare routines must
            // explicitly advance this as they generate output.
            //

            (*File)->NumberOfBytesWritten.QuadPart = 0;
        }

        if (Impl) {
            Result = Impl(Context, Item);
            if (Result == S_FALSE) {
                Chm01RequeueFileWorkItem(Context, Item);
                return;
            }
            if (FAILED(Result)) {

                //
                // Nothing needs doing here.  The Result will bubble back up
                // via the normal mechanisms.
                //

                NOTHING;
            }
        }

        if (UseOverlappedIo(Context) && Impl) {
            FILE_WORK_ID SaveWorkId;

            SaveWorkId = FileWorkIdToDependentId(FileWorkId);
            HasSaveCallback = (FileCallbacks[SaveWorkId] != NULL);

            if (!HasSaveCallback) {
                Result = Chm01WriteFileFromBuffer(
                    Context,
                    Item,
                    *File,
                    (*File)->IocpBufferPool,
                    (*File)->IocpBuffer
                );
                if (Result == S_FALSE) {
                    Chm01ClearFileWorkBuffer(*File);
                    return;
                }
                Chm01ClearFileWorkBuffer(*File);
                if (FAILED(Result)) {
                    goto End;
                }
            }
        }

    } else if (IsSaveFileWorkId(FileWorkId)) {

        ULONG WaitResult;

        //
        // All save events are dependent on their previous prepare events.
        //

        DependentEventIndex = FileWorkIdToDependentEventIndex(FileWorkId);
        DependentEvent = *(&Context->FirstPreparedEvent + DependentEventIndex);

        WaitResult = WaitForSingleObject(
            DependentEvent,
            UseOverlappedIo(Context) ? 0 : INFINITE
        );
        if (WaitResult == WAIT_TIMEOUT && UseOverlappedIo(Context)) {
            Chm01RequeueFileWorkItem(Context, Item);
            return;
        }
        if (WaitResult != WAIT_OBJECT_0) {
            SYS_ERROR(WaitForSingleObject);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto End;
        }

        //
        // If the file hasn't been set at this point, the prepare routine
        // was not successful.  We use E_UNEXPECTED as the error code here
        // as we just need *something* to be set when End: is jumped to in
        // order for the interlocked increment of the error count to occur.
        // The parent Chm01.c routine will adjust this to the proper error
        // code as necessary.
        //

        if (!*File) {
            Result = E_UNEXPECTED;
            goto End;
        }

        PrepareWorkId = FileWorkIdToDependentId(FileWorkId);
        HasPrepareCallback = (FileCallbacks[PrepareWorkId] != NULL);

        if (UseOverlappedIo(Context)) {
            if (Impl || (*File)->IocpBuffer) {
                ULONGLONG BufferPayloadBytes;

                BufferPayloadBytes = Chm01AdjustRequiredPayloadBytes(
                    Context,
                    FileId,
                    (ULONGLONG)(*File)->FileInfo.EndOfFile.QuadPart
                );

                Result = Chm01EnsureFileWorkBuffer(
                    Context,
                    FileIndex,
                    BufferPayloadBytes,
                    *File
                );
                if (FAILED(Result)) {
                    goto End;
                }
            }
        }

        if (UseOverlappedIo(Context) && Impl && !HasPrepareCallback) {
            (*File)->NumberOfBytesWritten.QuadPart = 0;
        }

        if (Impl) {
            Result = Impl(Context, Item);
            if (Result == S_FALSE) {
                Chm01RequeueFileWorkItem(Context, Item);
                return;
            }
            if (FAILED(Result)) {
                goto End;
            }
        }

        if (UseOverlappedIo(Context) && (*File)->IocpBuffer) {
            Result = Chm01WriteFileFromBuffer(Context,
                                              Item,
                                              *File,
                                              (*File)->IocpBufferPool,
                                              (*File)->IocpBuffer);
            if (Result == S_FALSE) {
                Chm01ClearFileWorkBuffer(*File);
                return;
            }
            Chm01ClearFileWorkBuffer(*File);
            if (FAILED(Result)) {
                goto End;
            }
        }

        //
        // Unmap the file (which has the effect of flushing the file buffers),
        // but don't close it.  We do this here, as part of the save file work,
        // in order to reduce the amount of work each file's Close() routine
        // has to do when the close work items are submitted in parallel.
        //

        if (!UseOverlappedIo(Context)) {
            Result = UnmapFileChm01(Table, Item);
            if (FAILED(Result)) {

                //
                // Nothing needs doing here.  The Result will bubble back up
                // via the normal mechanisms.
                //

                NOTHING;
            }
        }

    } else {

        //
        // Invariant check: our file work ID should be of type 'Close' here.
        //

        if (!IsCloseFileWorkId(FileWorkId)) {
            PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        }

        //
        // As above (in the save logic), if *File is NULL, use E_UNEXPECTED
        // as our Result.
        //

        if (!*File) {
            Result = E_UNEXPECTED;
            goto End;
        }

        Result = CloseFileChm01(Table, Item);
        if (FAILED(Result)) {

            //
            // Nothing needs doing here.  The Result will bubble back up
            // via the normal mechanisms.
            //

            NOTHING;
        }

    }

    //
    // Intentional follow-on to End.
    //

End:

    RELEASE(Path);

    if (UseOverlappedIo(Context) &&
        File &&
        *File &&
        (*File)->IocpBuffer) {
        if (FAILED(Result) || IsCloseFileWorkId(FileWorkId)) {
            Chm01ReleaseFileWorkBuffer(Context, *File);
        }
    }

    //
    // If the item's UUID string buffer is non-NULL here, the downstream routine
    // did not successfully take ownership of it, and thus, we're responsbile
    // for freeing it.
    //

    if (Item->Uuid.Buffer) {
        ASSERT(FileRequiresUuid(FileId));
        if (File && *File) {
            ASSERT((*File)->Uuid.Buffer == NULL);
        }
        Result = RtlFreeUuidString(Rtl, &Item->Uuid);
    }

    Item->LastResult = Result;

    if (FAILED(Result)) {
        InterlockedIncrement(&Item->NumberOfErrors);
        Item->LastError = GetLastError();
    }

    //
    // Register the relevant event to be set when this threadpool callback
    // returns, then return.
    //


    if (Event) {
#ifdef PH_WINDOWS
        if (Item->Instance) {
            SetEventWhenCallbackReturns(Item->Instance, Event);
        } else {
            SetEvent(Event);
        }
#else
        SetEvent(Event);
#endif
    }

    return;
}


PREPARE_FILE PrepareFileChm01;

_Use_decl_annotations_
HRESULT
PrepareFileChm01(
    PPERFECT_HASH_TABLE Table,
    PFILE_WORK_ITEM Item,
    PPERFECT_HASH_PATH Path,
    PLARGE_INTEGER EndOfFile,
    HANDLE DependentEvent
    )
/*++

Routine Description:

    Performs common file preparation work for a given file instance associated
    with a table.  If this is the first call to the function, indicated by a
    NULL value pointed to by the FilePointer argument, then a new file instance
    is created, and a Create() call is issued with the path and mapping size
    parameters.  Otherwise, if it is not NULL, a rename is scheduled for the
    new path name and the mapping size is extended (this involves unmapping the
    existing map, closing the mapping handle, extending the file by setting the
    file pointer and then end-of-file, and then creating a new mapping handle
    and re-mapping the address).

Arguments:

    Table - Supplies a pointer to the table owning the file to be prepared.

    Item - Supplies a pointer to the active file work item associated with
        the preparation.

    Path - Supplies a pointer to the path to use for the file.  If the file
        has already been prepared at least once, this path is scheduled for
        rename.

    EndOfFile - Supplies a pointer to a LARGE_INTEGER that contains the
        desired file size.

    DependentEvent - Optionally supplies a handle to an event that must be
        signaled prior to this routine proceeding.  This is used, for example,
        to wait for the perfect hash table file to be created before creating
        the :Info stream that hangs off it.

Return Value:

    S_OK - File prepared successfully.  Otherwise, an appropriate error code.

--*/
{
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File = NULL;
    PPERFECT_HASH_DIRECTORY Directory = NULL;
    PPERFECT_HASH_CONTEXT Context = NULL;
    PERFECT_HASH_FILE_CREATE_FLAGS CreateFlags = { 0 };
    PPERFECT_HASH_FILE_CREATE_FLAGS CreateFlagsPointer = NULL;

    //
    // Dereference the file pointer provided by the caller.  If NULL, this
    // is the first preparation request for the given file instance.  Otherwise,
    // a table resize event has occurred, which means a file rename needs to be
    // scheduled (as we include the number of table elements in the file name),
    // and the mapping size needs to be extended (as a larger table size means
    // larger files are required to capture table data).
    //

    File = *Item->FilePointer;
    Context = Table->Context;

    //
    // If a dependent event has been provided, wait for this object to become
    // signaled first before proceeding.
    //

    if (IsValidHandle(DependentEvent)) {
        ULONG WaitResult;
        DWORD Timeout;

        Timeout = (Context && UseOverlappedIo(Context)) ? 0 : INFINITE;

        WaitResult = WaitForSingleObject(DependentEvent, Timeout);
        if (WaitResult == WAIT_TIMEOUT && Timeout == 0) {
            return S_FALSE;
        }
        if (WaitResult != WAIT_OBJECT_0) {
            SYS_ERROR(WaitForSingleObject);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }
    }

    //
    // Dereference the file pointer provided by the caller.  If NULL, this

    if (!File) {

        //
        // File does not exist, so create a new instance, then issue a Create()
        // call with the desired path and mapping size parameters provided by
        // the caller.
        //

        Result = Table->Vtbl->CreateInstance(Table,
                                             NULL,
                                             &IID_PERFECT_HASH_FILE,
                                             (PVOID *)&File);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileCreateInstance, Result);
            goto Error;
        }

        if (!IsContextFileWorkItem(Item)) {

            //
            // Table files always get associated with the table's output
            // directory (i.e. the Create() call coming up will call the
            // directory's AddFile() method if Directory is not NULL).
            //

            Directory = Table->OutputDirectory;
        }

        if (Context && UseOverlappedIo(Context)) {
            CreateFlags.SkipMapping = TRUE;
            CreateFlagsPointer = &CreateFlags;
        }

        Result = File->Vtbl->Create(File,
                                    Path,
                                    EndOfFile,
                                    Directory,
                                    CreateFlagsPointer);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileCreate, Result);
            RELEASE(File);
            goto Error;
        }

        if (Context &&
            UseOverlappedIo(Context) &&
            Context->FileWorkIoCompletionPort) {
            HANDLE PortHandle;

            PortHandle = CreateIoCompletionPort(
                File->FileHandle,
                Context->FileWorkIoCompletionPort,
                (ULONG_PTR)File,
                0
            );

            if (!PortHandle) {
                SYS_ERROR(CreateIoCompletionPort_FileWork);
                Result = PH_E_SYSTEM_CALL_FAILED;
                RELEASE(File);
                goto Error;
            }
        }

        //
        // Set the file ID and then update the file pointer.
        //

        File->FileId = Item->FileId;
        *Item->FilePointer = File;

        if (!FileRequiresUuid(File->FileId)) {

            //
            // No UUID string buffer should be set if the file hasn't been
            // marked as requiring a UUID.
            //

            ASSERT(Item->Uuid.Buffer == NULL);

        } else {

            //
            // Verify the Item->Uuid string has been filled out.
            //

            if (!IsValidUuidString(&Item->Uuid)) {
                Result = PH_E_INVARIANT_CHECK_FAILED;
                PH_ERROR(PrepareFileChm01_VCProjectItemMissingUuid, Result);
                goto Error;
            }

            //
            // Copy the details over to the file instance, which will now "own"
            // the underlying UUID string buffer (this is freed in the file's
            // rundown routine), and zero the Item->Uuid representation.
            //

            CopyInline(&File->Uuid, &Item->Uuid, sizeof(File->Uuid));
            ZeroStructInline(Item->Uuid);
        }

    } else {

        //
        // The file already exists.
        //

        //
        // Invariant check: context files should only be prepared once.
        //

        if (IsContextFileWorkItem(Item)) {
            Result = PH_E_CONTEXT_FILE_ALREADY_PREPARED;
            PH_ERROR(PrepareFileChm01, Result);
            goto Error;
        }

        //
        // Invariant check: no UUID should be set.
        //

        if (Item->Uuid.Buffer) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareFileChm01_ItemUuidBufferNotNull, Result);
            goto Error;
        }

        //
        // Invariant check: File->FileId should already be set, and it should
        // match the file ID specified in Item.
        //

        if (!IsValidFileId(File->FileId)) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareFileChm01_SaveFile_InvalidFileId, Result);
            goto Error;
        }

        if (File->FileId != Item->FileId) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareFileChm01_SaveFile_FileIdMismatch, Result);
            goto Error;
        }

        //
        // If the table indicates a table resize requires rename, schedule one.
        //

        if (TableResizeRequiresRename(Table)) {
            Result = File->Vtbl->ScheduleRename(File, Path);
            if (FAILED(Result)) {
                PH_ERROR(PerfectHashFileScheduleRename, Result);
                goto Error;
            }
        }

        //
        // If the existing mapping size differs from the desired size, extend
        // the file accordingly.
        //

        if (File->FileInfo.EndOfFile.QuadPart < EndOfFile->QuadPart) {
            AcquirePerfectHashFileLockExclusive(File);
            Result = File->Vtbl->Extend(File, EndOfFile);
            ReleasePerfectHashFileLockExclusive(File);
            if (FAILED(Result)) {
                PH_ERROR(PerfectHashFileExtend, Result);
                goto Error;
            }
        }

    }

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

    return Result;
}

UNMAP_FILE UnmapFileChm01;

_Use_decl_annotations_
HRESULT
UnmapFileChm01(
    PPERFECT_HASH_TABLE Table,
    PFILE_WORK_ITEM Item
    )
/*++

Routine Description:

    Unmaps a file instance associated with a table.

Arguments:

    Table - Supplies a pointer to the table owning the file to be unmapped.

    Item - Supplies a pointer to the file work item for this unmap action.

Return Value:

    S_OK - File unmapped successfully.  Otherwise, an appropriate error code.

--*/
{
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;

    UNREFERENCED_PARAMETER(Table);

    File = *Item->FilePointer;

    //
    // Unmap the file if it's either a) not a context file, or b) if it is a
    // context file, only if it hasn't already been unmapped.
    //

    if (!IsContextFileWorkItem(Item) || !IsFileUnmapped(File)) {
        Result = File->Vtbl->Unmap(File);
        if (FAILED(Result)) {
            PH_ERROR(UnmapFileChm01, Result);
        }
    }

    return Result;
}

CLOSE_FILE CloseFileChm01;

_Use_decl_annotations_
HRESULT
CloseFileChm01(
    PPERFECT_HASH_TABLE Table,
    PFILE_WORK_ITEM Item
    )
/*++

Routine Description:

    Closes a file instance associated with a table.

    N.B.  If an error has occurred, Item->EndOfFile will point to a
          LARGE_INTEGER with value 0, which informs the file's Close()
          machinery to delete the file.  (Otherwise, the file will be
          truncated based on the value of File->NumberOfBytesWritten.)

Arguments:

    Table - Supplies a pointer to the table owning the file to be closed.

    Item - Supplies a pointer to the file work item for this close action.

Return Value:

    S_OK - File closed successfully.  Otherwise, an appropriate error code.

--*/
{
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;

    UNREFERENCED_PARAMETER(Table);

    File = *Item->FilePointer;

    //
    // Close the file if it's either a) not a context file, or b) if it is a
    // context file, only if it hasn't already been closed.
    //

    if (!IsContextFileWorkItem(Item) || !IsFileClosed(File)) {
        Result = File->Vtbl->Close(File, Item->EndOfFile);
        if (FAILED(Result)) {
            PH_ERROR(CloseFileChm01, Result);
        }
    }

    return Result;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
