/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashAllocator.c

Abstract:

    This module implements the perfect hash allocator component.

--*/

#include "stdafx.h"

ALLOCATOR_MALLOC AllocatorMalloc;

_Use_decl_annotations_
PVOID
AllocatorMalloc(
    PALLOCATOR Allocator,
    SIZE_T Size
    )
{
    return HeapAlloc(Allocator->HeapHandle, 0, Size);
}


ALLOCATOR_CALLOC AllocatorCalloc;

_Use_decl_annotations_
PVOID
AllocatorCalloc(
    PALLOCATOR Allocator,
    SIZE_T NumberOfElements,
    SIZE_T ElementSize
    )
{
    SIZE_T Size = NumberOfElements * ElementSize;
    PVOID Address;

    Address = HeapAlloc(Allocator->HeapHandle, HEAP_ZERO_MEMORY, Size);
    return Address;
}


ALLOCATOR_FREE AllocatorFree;

_Use_decl_annotations_
VOID
AllocatorFree(
    PALLOCATOR Allocator,
    PVOID Address
    )
{
    HeapFree(Allocator->HeapHandle, 0, Address);
    return;
}


ALLOCATOR_FREE_POINTER AllocatorFreePointer;

_Use_decl_annotations_
VOID
AllocatorFreePointer(
    PALLOCATOR Allocator,
    PVOID *AddressPointer
    )
{
    if (!ARGUMENT_PRESENT(AddressPointer)) {
        return;
    }

    if (!ARGUMENT_PRESENT(*AddressPointer)) {
        return;
    }

    AllocatorFree(Allocator, *AddressPointer);
    *AddressPointer = NULL;

    return;
}

ALLOCATOR_FREE_STRING_BUFFER AllocatorFreeStringBuffer;

_Use_decl_annotations_
VOID
AllocatorFreeStringBuffer(
    PALLOCATOR Allocator,
    PSTRING String
    )
{
    if (String->Buffer) {
        AllocatorFree(Allocator, String->Buffer);
    }

    String->Buffer = NULL;
    String->Length = 0;
    String->MaximumLength = 0;
}

ALLOCATOR_FREE_UNICODE_STRING_BUFFER AllocatorFreeUnicodeStringBuffer;

_Use_decl_annotations_
VOID
AllocatorFreeUnicodeStringBuffer(
    PALLOCATOR Allocator,
    PUNICODE_STRING String
    )
{
    if (String->Buffer) {
        AllocatorFree(Allocator, String->Buffer);
    }

    String->Buffer = NULL;
    String->Length = 0;
    String->MaximumLength = 0;
}

ALLOCATOR_INITIALIZE AllocatorInitialize;

_Use_decl_annotations_
HRESULT
AllocatorInitialize(
    PALLOCATOR Allocator
    )
{
    ULONG Flags = 0;
    ULONG_PTR MinimumSize = 0;
    ULONG_PTR MaximumSize = 0;
    PPERFECT_HASH_TLS_CONTEXT TlsContext;

    TlsContext = PerfectHashTlsGetContext();

    if (TlsContextCustomAllocatorDetailsPresent(TlsContext)) {
        Flags = TlsContext->HeapCreateFlags;
        MinimumSize = TlsContext->HeapMinimumSize;
    }

    Allocator->HeapHandle = HeapCreate(Flags, MinimumSize, MaximumSize);

    if (!Allocator->HeapHandle) {
        return PH_E_HEAP_CREATE_FAILED;
    }

    return S_OK;
}


ALLOCATOR_RUNDOWN AllocatorRundown;

_Use_decl_annotations_
VOID
AllocatorRundown(
    PALLOCATOR Allocator
    )
{
    HeapDestroy(Allocator->HeapHandle);
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
