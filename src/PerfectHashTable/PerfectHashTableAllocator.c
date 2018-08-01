/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableAllocator.c

Abstract:

    This module implements the perfect hash allocator initialization routines.
    We currently use the Rtl component's heap-backed allocator, although this
    can be changed easily at a later date.

    N.B. Two allocators are provided, one bootstrap allocator and one normal
         one, as this allows a downstream component to obtain an allocator
         prior to initializing the RTL structure.  This is useful in unit tests.

--*/

#include "stdafx.h"

_Use_decl_annotations_
BOOL
InitializePerfectHashTableAllocatorFromRtlBootstrap(
    PRTL_BOOTSTRAP RtlBootstrap,
    PALLOCATOR Allocator
    )
{
    return RtlBootstrap->InitializeHeapAllocator(Allocator);
}

_Use_decl_annotations_
BOOL
InitializePerfectHashTableAllocator(
    PRTL Rtl,
    PALLOCATOR Allocator
    )
{
    return Rtl->InitializeHeapAllocator(Allocator);
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
