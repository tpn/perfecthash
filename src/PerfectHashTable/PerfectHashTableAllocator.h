/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableAllocator.h

Abstract:

    This is the private header file for the allocator component of the perfect
    hash table library.  It defines the ALLOCATOR structure, and function
    pointer typedefs for the initialize and rundown functions.

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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
