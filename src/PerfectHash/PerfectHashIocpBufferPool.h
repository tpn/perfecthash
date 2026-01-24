/*++

Copyright (c) 2024-2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashIocpBufferPool.h

Abstract:

    This is the private header file for IOCP buffer pool support. It defines
    buffer header and pool structures used by the IOCP backend for overlapped
    file I/O, along with helper routines for pool management.

--*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <PerfectHash.h>

#ifndef ALIGN_UP
#define ALIGN_UP(Address, Alignment) (                        \
    (((ULONG_PTR)(Address)) + (Alignment) - 1) &              \
    ~((ULONG_PTR)(Alignment) - 1)                             \
)
#endif

typedef struct _RTL RTL;
typedef RTL *PRTL;

//
// IOCP buffer buckets keyed by AlignUpPow2(NumberOfKeys).
//

#define PERFECT_HASH_IOCP_BUFFER_BUCKET_COUNT 32
#define PERFECT_HASH_IOCP_BUFFER_MAX_BUCKET_INDEX \
    (PERFECT_HASH_IOCP_BUFFER_BUCKET_COUNT - 1)

//
// Buffer pool flags.
//

#define PERFECT_HASH_IOCP_BUFFER_POOL_FLAG_ONE_OFF 0x00000001

//
// IOCP buffer header used by IOCP file work. Payload begins at PayloadOffset.
//

typedef struct DECLSPEC_ALIGN(16) _PERFECT_HASH_IOCP_BUFFER {
    SLIST_ENTRY ListEntry;
    ULONGLONG BytesWritten;
    ULONGLONG PayloadSize;
    ULONG SizeOfStruct;
    ULONG PayloadOffset;
    ULONG BucketIndex;
    ULONG FileId;
    USHORT NumaNode;
    USHORT Flags;
    ULONG Padding1;
    ULONG Padding2;
    ULONG Padding3;
} PERFECT_HASH_IOCP_BUFFER;
typedef PERFECT_HASH_IOCP_BUFFER *PPERFECT_HASH_IOCP_BUFFER;

#define PERFECT_HASH_IOCP_BUFFER_HEADER_ALIGNMENT MEMORY_ALLOCATION_ALIGNMENT

#define PERFECT_HASH_IOCP_BUFFER_HEADER_SIZE (                                    \
    (ULONG)ALIGN_UP(sizeof(PERFECT_HASH_IOCP_BUFFER),                             \
                    PERFECT_HASH_IOCP_BUFFER_HEADER_ALIGNMENT)                   \
)

FORCEINLINE
PVOID
PerfectHashIocpBufferPayload(
    _In_ PPERFECT_HASH_IOCP_BUFFER Buffer
    )
{
    return (PVOID)((PCHAR)Buffer + Buffer->PayloadOffset);
}

//
// IOCP buffer pool.
//

typedef struct DECLSPEC_ALIGN(16) _PERFECT_HASH_IOCP_BUFFER_POOL {
    ULONG SizeOfStruct;
    ULONG Flags;
    ULONG PageSize;
    ULONG NumberOfBuffers;
    ULONG NumberOfPagesPerBuffer;
    ULONG PayloadOffset;
    ULONGLONG UsableBufferSizeInBytes;
    ULONGLONG PayloadSizeInBytes;
    ULONGLONG BufferStrideInBytes;
    ULONGLONG TotalAllocationSizeInBytes;
    HANDLE ProcessHandle;
    PVOID BaseAddress;
    ULONGLONG ListHeadPadding;
    SLIST_HEADER ListHead;
} PERFECT_HASH_IOCP_BUFFER_POOL;
typedef PERFECT_HASH_IOCP_BUFFER_POOL *PPERFECT_HASH_IOCP_BUFFER_POOL;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_IOCP_BUFFER_POOL_CREATE)(
    _In_ PRTL Rtl,
    _In_opt_ PHANDLE ProcessHandle,
    _In_ ULONG PageSize,
    _In_ ULONG NumberOfBuffers,
    _In_ ULONG NumberOfPagesPerBuffer,
    _In_opt_ PULONG AdditionalProtectionFlags,
    _In_opt_ PULONG AdditionalAllocationTypeFlags,
    _Out_ PPERFECT_HASH_IOCP_BUFFER_POOL Pool
    );
typedef PERFECT_HASH_IOCP_BUFFER_POOL_CREATE
      *PPERFECT_HASH_IOCP_BUFFER_POOL_CREATE;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_IOCP_BUFFER_POOL_DESTROY)(
    _In_ PRTL Rtl,
    _Inout_ PPERFECT_HASH_IOCP_BUFFER_POOL Pool
    );
typedef PERFECT_HASH_IOCP_BUFFER_POOL_DESTROY
      *PPERFECT_HASH_IOCP_BUFFER_POOL_DESTROY;

typedef
_Must_inspect_result_
_Success_(return != NULL)
PPERFECT_HASH_IOCP_BUFFER
(NTAPI PERFECT_HASH_IOCP_BUFFER_POOL_POP)(
    _In_ PPERFECT_HASH_IOCP_BUFFER_POOL Pool
    );
typedef PERFECT_HASH_IOCP_BUFFER_POOL_POP
      *PPERFECT_HASH_IOCP_BUFFER_POOL_POP;

typedef
VOID
(NTAPI PERFECT_HASH_IOCP_BUFFER_POOL_PUSH)(
    _In_ PPERFECT_HASH_IOCP_BUFFER_POOL Pool,
    _In_ PPERFECT_HASH_IOCP_BUFFER Buffer
    );
typedef PERFECT_HASH_IOCP_BUFFER_POOL_PUSH
      *PPERFECT_HASH_IOCP_BUFFER_POOL_PUSH;

extern PERFECT_HASH_IOCP_BUFFER_POOL_CREATE PerfectHashIocpBufferPoolCreate;
extern PERFECT_HASH_IOCP_BUFFER_POOL_DESTROY PerfectHashIocpBufferPoolDestroy;
extern PERFECT_HASH_IOCP_BUFFER_POOL_POP PerfectHashIocpBufferPoolPop;
extern PERFECT_HASH_IOCP_BUFFER_POOL_PUSH PerfectHashIocpBufferPoolPush;

#ifdef __cplusplus
} // extern "C"
#endif
