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
// IOCP buffer size-class configuration (payload sizes in bytes).
//

#define PERFECT_HASH_IOCP_BUFFER_MIN_SHIFT 12  // 4KB
#define PERFECT_HASH_IOCP_BUFFER_MAX_SHIFT 24  // 16MB

#define PERFECT_HASH_IOCP_BUFFER_MIN_SIZE \
    (1ULL << PERFECT_HASH_IOCP_BUFFER_MIN_SHIFT)

#define PERFECT_HASH_IOCP_BUFFER_MAX_SIZE \
    (1ULL << PERFECT_HASH_IOCP_BUFFER_MAX_SHIFT)

#define PERFECT_HASH_IOCP_BUFFER_CLASS_COUNT (                                   \
    PERFECT_HASH_IOCP_BUFFER_MAX_SHIFT -                                        \
    PERFECT_HASH_IOCP_BUFFER_MIN_SHIFT +                                        \
    1                                                                           \
)

//
// Buffer flags.
//

#define PERFECT_HASH_IOCP_BUFFER_FLAG_GUARD_PAGES 0x00000001
#define PERFECT_HASH_IOCP_BUFFER_FLAG_OVERSIZE    0x00000002

//
// Pool flags.
//

#define PERFECT_HASH_IOCP_BUFFER_POOL_FLAG_INITIALIZED 0x00000001
#define PERFECT_HASH_IOCP_BUFFER_POOL_FLAG_GUARD_PAGES 0x00000002
#define PERFECT_HASH_IOCP_BUFFER_POOL_FLAG_OVERSIZE    0x00000004

//
// IOCP buffer header used by IOCP file work. Payload begins at PayloadOffset.
//

typedef struct DECLSPEC_ALIGN(16) _PERFECT_HASH_IOCP_BUFFER {
    SLIST_ENTRY FreeListEntry;
    LIST_ENTRY ListEntry;
    struct _PERFECT_HASH_IOCP_BUFFER_POOL *OwnerPool;
    ULONGLONG BytesWritten;
    ULONGLONG PayloadSize;
    ULONGLONG AllocationSize;
    ULONG SizeOfStruct;
    ULONG PayloadOffset;
    ULONG Flags;
    ULONG Padding1;
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

FORCEINLINE
LONG
PerfectHashIocpBufferGetClassIndex(
    _In_ PRTL Rtl,
    _In_ ULONGLONG PayloadBytes
    )
{
    ULONGLONG Rounded;
    ULONG TrailingZeros;

    if (!ARGUMENT_PRESENT(Rtl)) {
        return -1;
    }

    if (PayloadBytes <= PERFECT_HASH_IOCP_BUFFER_MIN_SIZE) {
        return 0;
    }

    Rounded = Rtl->RoundUpPowerOfTwo64(PayloadBytes);
    if (Rounded == 0) {
        return -1;
    }

    if (Rounded > PERFECT_HASH_IOCP_BUFFER_MAX_SIZE) {
        return -1;
    }

    TrailingZeros = (ULONG)Rtl->TrailingZeros64(Rounded);
    return (LONG)TrailingZeros - PERFECT_HASH_IOCP_BUFFER_MIN_SHIFT;
}

FORCEINLINE
ULONGLONG
PerfectHashIocpBufferGetPayloadSizeFromClassIndex(
    _In_ LONG ClassIndex
    )
{
    ULONGLONG Shift;

    if (ClassIndex < 0) {
        return 0;
    }

    Shift = (ULONGLONG)ClassIndex + PERFECT_HASH_IOCP_BUFFER_MIN_SHIFT;
    return (1ULL << Shift);
}

//
// IOCP buffer pool.
//

typedef struct DECLSPEC_ALIGN(16) _PERFECT_HASH_IOCP_BUFFER_POOL {
    LIST_ENTRY ListEntry;
    SLIST_HEADER FreeList;
    PGUARDED_LIST BufferList;
    HANDLE ProcessHandle;
    ULONGLONG PayloadSize;
    ULONGLONG AllocationSize;
    ULONG SizeOfStruct;
    ULONG Flags;
    ULONG Padding1;
    USHORT NumaNode;
    USHORT Padding2;
} PERFECT_HASH_IOCP_BUFFER_POOL;
typedef PERFECT_HASH_IOCP_BUFFER_POOL *PPERFECT_HASH_IOCP_BUFFER_POOL;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_IOCP_BUFFER_POOL_INITIALIZE)(
    _In_ PRTL Rtl,
    _In_ PPERFECT_HASH_IOCP_BUFFER_POOL Pool,
    _In_ ULONGLONG PayloadSize,
    _In_ USHORT NumaNode,
    _In_ ULONG Flags,
    _In_opt_ HANDLE ProcessHandle,
    _In_opt_ PGUARDED_LIST BufferList
    );
typedef PERFECT_HASH_IOCP_BUFFER_POOL_INITIALIZE
      *PPERFECT_HASH_IOCP_BUFFER_POOL_INITIALIZE;

typedef
_Must_inspect_result_
HRESULT
(NTAPI PERFECT_HASH_IOCP_BUFFER_POOL_ACQUIRE)(
    _In_ PRTL Rtl,
    _In_opt_ PALLOCATOR Allocator,
    _In_ PPERFECT_HASH_IOCP_BUFFER_POOL Pool,
    _Outptr_ PPERFECT_HASH_IOCP_BUFFER *BufferPointer
    );
typedef PERFECT_HASH_IOCP_BUFFER_POOL_ACQUIRE
      *PPERFECT_HASH_IOCP_BUFFER_POOL_ACQUIRE;

typedef
VOID
(NTAPI PERFECT_HASH_IOCP_BUFFER_POOL_RELEASE)(
    _In_ PPERFECT_HASH_IOCP_BUFFER_POOL Pool,
    _In_ PPERFECT_HASH_IOCP_BUFFER Buffer
    );
typedef PERFECT_HASH_IOCP_BUFFER_POOL_RELEASE
      *PPERFECT_HASH_IOCP_BUFFER_POOL_RELEASE;

typedef
VOID
(NTAPI PERFECT_HASH_IOCP_BUFFER_POOL_RUNDOWN)(
    _In_ PRTL Rtl,
    _In_opt_ PALLOCATOR Allocator,
    _Inout_ PPERFECT_HASH_IOCP_BUFFER_POOL Pool
    );
typedef PERFECT_HASH_IOCP_BUFFER_POOL_RUNDOWN
      *PPERFECT_HASH_IOCP_BUFFER_POOL_RUNDOWN;

extern PERFECT_HASH_IOCP_BUFFER_POOL_INITIALIZE
    PerfectHashIocpBufferPoolInitialize;
extern PERFECT_HASH_IOCP_BUFFER_POOL_ACQUIRE
    PerfectHashIocpBufferPoolAcquire;
extern PERFECT_HASH_IOCP_BUFFER_POOL_RELEASE
    PerfectHashIocpBufferPoolRelease;
extern PERFECT_HASH_IOCP_BUFFER_POOL_RUNDOWN
    PerfectHashIocpBufferPoolRundown;

#ifdef __cplusplus
} // extern "C"
#endif
