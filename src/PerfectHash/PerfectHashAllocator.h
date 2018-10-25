/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashAllocator.h

Abstract:

    This is the private header file for the allocator component of the perfect
    hash library.  It defines the ALLOCATOR structure, and function pointer
    typedefs for the initialize and rundown functions.

--*/

#pragma once

#include "stdafx.h"

DEFINE_UNUSED_STATE(ALLOCATOR);
DEFINE_UNUSED_FLAGS(ALLOCATOR);

typedef struct _ALLOCATOR {
    COMMON_COMPONENT_HEADER(ALLOCATOR);
    HANDLE HeapHandle;
    ALLOCATOR_VTBL Interface;
} ALLOCATOR;
typedef ALLOCATOR *PALLOCATOR;

typedef
HRESULT
(NTAPI ALLOCATOR_INITIALIZE)(
    _In_ PALLOCATOR Allocator
    );
typedef ALLOCATOR_INITIALIZE *PALLOCATOR_INITIALIZE;

typedef
VOID
(NTAPI ALLOCATOR_RUNDOWN)(
    _In_ _Post_ptr_invalid_ PALLOCATOR Allocator
    );
typedef ALLOCATOR_RUNDOWN *PALLOCATOR_RUNDOWN;

extern ALLOCATOR_INITIALIZE AllocatorInitialize;
extern ALLOCATOR_RUNDOWN AllocatorRundown;
extern ALLOCATOR_MALLOC AllocatorMalloc;
extern ALLOCATOR_CALLOC AllocatorCalloc;
extern ALLOCATOR_REALLOC AllocatorReAlloc;
extern ALLOCATOR_RECALLOC AllocatorReCalloc;
extern ALLOCATOR_FREE AllocatorFree;
extern ALLOCATOR_FREE_POINTER AllocatorFreePointer;
extern ALLOCATOR_FREE_STRING_BUFFER AllocatorFreeStringBuffer;
extern ALLOCATOR_FREE_UNICODE_STRING_BUFFER AllocatorFreeUnicodeStringBuffer;
extern ALLOCATOR_ALIGNED_MALLOC AllocatorAlignedMalloc;
extern ALLOCATOR_ALIGNED_CALLOC AllocatorAlignedCalloc;
extern ALLOCATOR_ALIGNED_REALLOC AllocatorAlignedReAlloc;
extern ALLOCATOR_ALIGNED_RECALLOC AllocatorAlignedReCalloc;
extern ALLOCATOR_ALIGNED_FREE AllocatorAlignedFree;
extern ALLOCATOR_ALIGNED_FREE_POINTER AllocatorAlignedFreePointer;
extern ALLOCATOR_ALIGNED_OFFSET_MALLOC AllocatorAlignedOffsetMalloc;
extern ALLOCATOR_ALIGNED_OFFSET_CALLOC AllocatorAlignedOffsetCalloc;
extern ALLOCATOR_ALIGNED_OFFSET_REALLOC AllocatorAlignedOffsetReAlloc;
extern ALLOCATOR_ALIGNED_OFFSET_RECALLOC AllocatorAlignedOffsetReCalloc;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
