/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashAllocator.c

Abstract:

    This module implements the perfect hash allocator component.

--*/

#include "stdafx.h"

#ifdef _WIN64
#define PTR_SZ 8
#else
#define PTR_SZ 4
#endif


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


ALLOCATOR_REALLOC AllocatorReAlloc;

_Use_decl_annotations_
PVOID
AllocatorReAlloc(
    PALLOCATOR Allocator,
    PVOID Address,
    SIZE_T Size
    )
{
    PVOID NewAddress;

    NewAddress = HeapReAlloc(Allocator->HeapHandle,
                             0,
                             Address,
                             Size);

    return NewAddress;
}


ALLOCATOR_RECALLOC AllocatorReCalloc;

_Use_decl_annotations_
PVOID
AllocatorReCalloc(
    PALLOCATOR Allocator,
    PVOID Address,
    SIZE_T NumberOfElements,
    SIZE_T ElementSize
    )
{
    PVOID NewAddress;

    NewAddress = HeapReAlloc(Allocator->HeapHandle,
                             HEAP_ZERO_MEMORY,
                             Address,
                             NumberOfElements * ElementSize);

    return NewAddress;
}


ALLOCATOR_FREE AllocatorFree;

_Use_decl_annotations_
VOID
AllocatorFree(
    PALLOCATOR Allocator,
    PVOID Address
    )
{
    if (!HeapFree(Allocator->HeapHandle, 0, Address)) {
        SYS_ERROR(HeapFree);
        PH_RAISE(E_UNEXPECTED);
    }
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

//
// Base aligned offset allocation routine.
//

_Must_inspect_result_
_Success_(return != 0)
_Ret_maybenull_
_Post_writable_byte_size_(Size)
PVOID
AlignedOffsetAllocBase(
    _In_ PALLOCATOR Allocator,
    _In_ SIZE_T Size,
    _In_ SIZE_T Alignment,
    _In_ SIZE_T Offset,
    _In_ BOOLEAN ZeroMemory
    )
{
    ULONG Flags = 0;
    ULONG_PTR Buffer;
    ULONG_PTR Padding;
    ULONG_PTR ReturnAddress;
    ULONG_PTR AddressPointer;
    SIZE_T Overhead;
    SIZE_T AllocSize;
    SIZE_T Align;

    if (ZeroMemory) {
        Flags = HEAP_ZERO_MEMORY;
    }

    Align = (Alignment > PTR_SZ ? Alignment : PTR_SZ) - 1;

    Padding = (0 - Offset) & (PTR_SZ - 1);
    Overhead = (PTR_SZ + Padding + Align);
    AllocSize = Overhead + Size;

    Buffer = (ULONG_PTR)HeapAlloc(Allocator->HeapHandle, Flags, AllocSize);
    if (!Buffer) {
        return NULL;
    }

    ReturnAddress = ((Buffer + Overhead + Offset) & ~Align) - Offset;
    AddressPointer = (ReturnAddress - Padding) - sizeof(ULONG_PTR);
    ((ULONG_PTR *)(ReturnAddress - Padding))[-1] = Buffer;

    return (PVOID)ReturnAddress;
}


_Must_inspect_result_
_Ret_maybenull_
_Ret_reallocated_bytes_(Previous, Size)
PVOID
AlignedOffsetReAllocBase(
    _In_ PALLOCATOR Allocator,
    _Frees_ptr_opt_ PVOID Previous,
    _In_ SIZE_T Size,
    _In_ SIZE_T Alignment,
    _In_ SIZE_T Offset,
    _In_ BOOLEAN ZeroMemory
    )
{
    ULONG Flags = 0;
    ULONG_PTR Buffer;
    ULONG_PTR Padding;
    ULONG_PTR ReturnAddress;
    ULONG_PTR AddressPointer;
    ULONG_PTR PreviousAddress;
    SIZE_T Overhead;
    SIZE_T AllocSize;
    SIZE_T Align;

    if (ZeroMemory) {
        Flags = HEAP_ZERO_MEMORY;
    }

    Align = (Alignment > PTR_SZ ? Alignment : PTR_SZ) - 1;

    Padding = (0 - Offset) & (PTR_SZ - 1);
    Overhead = (PTR_SZ + Padding + Align);
    AllocSize = Overhead + Size;

    PreviousAddress = (ULONG_PTR)Previous;
    PreviousAddress = (PreviousAddress & ~(PTR_SZ - 1)) - PTR_SZ;
    PreviousAddress = *((ULONG_PTR *)PreviousAddress);

    Buffer = (ULONG_PTR)HeapReAlloc(Allocator->HeapHandle,
                                    Flags,
                                    (PVOID)PreviousAddress,
                                    AllocSize);

    if (!Buffer) {
        return NULL;
    }

    ReturnAddress = ((Buffer + Overhead + Offset) & ~Align) - Offset;
    AddressPointer = (ReturnAddress - Padding) - sizeof(ULONG_PTR);
    ((ULONG_PTR *)(ReturnAddress - Padding))[-1] = Buffer;

    return (PVOID)ReturnAddress;
}

//
// Aligned routines.
//

ALLOCATOR_ALIGNED_MALLOC AllocatorAlignedMalloc;

_Use_decl_annotations_
PVOID
AllocatorAlignedMalloc(
    PALLOCATOR Allocator,
    SIZE_T Size,
    SIZE_T Alignment
    )
{
    return AlignedOffsetAllocBase(Allocator,
                                  Size,
                                  Alignment,
                                  0,
                                  FALSE);
}


ALLOCATOR_ALIGNED_CALLOC AllocatorAlignedCalloc;

_Use_decl_annotations_
PVOID
AllocatorAlignedCalloc(
    PALLOCATOR Allocator,
    SIZE_T NumberOfElements,
    SIZE_T ElementSize,
    SIZE_T Alignment
    )
{
    return AlignedOffsetAllocBase(Allocator,
                                  NumberOfElements * ElementSize,
                                  Alignment,
                                  0,
                                  TRUE);
}


ALLOCATOR_ALIGNED_REALLOC AllocatorAlignedReAlloc;

_Use_decl_annotations_
PVOID
AllocatorAlignedReAlloc(
    PALLOCATOR Allocator,
    PVOID Address,
    SIZE_T Size,
    SIZE_T Alignment
    )
{
    return AlignedOffsetReAllocBase(Allocator,
                                    Address,
                                    Size,
                                    Alignment,
                                    0,
                                    FALSE);
}


ALLOCATOR_ALIGNED_RECALLOC AllocatorAlignedReCalloc;

_Use_decl_annotations_
PVOID
AllocatorAlignedReCalloc(
    PALLOCATOR Allocator,
    PVOID Address,
    SIZE_T NumberOfElements,
    SIZE_T ElementSize,
    SIZE_T Alignment
    )
{
    return AlignedOffsetReAllocBase(Allocator,
                                    Address,
                                    NumberOfElements * ElementSize,
                                    Alignment,
                                    0,
                                    TRUE);
}


ALLOCATOR_ALIGNED_FREE AllocatorAlignedFree;

_Use_decl_annotations_
VOID
AllocatorAlignedFree(
    PALLOCATOR Allocator,
    PVOID Buffer
    )
{
    ULONG_PTR Address;

    if (!Buffer) {
        return;
    }

    Address = (ULONG_PTR)Buffer;
    Address = (Address & ~(PTR_SZ - 1)) - PTR_SZ;
    Address = *((ULONG_PTR *)Address);

    ASSERT(Address != 0);

    AllocatorFree(Allocator, (PVOID)Address);
    return;
}


ALLOCATOR_ALIGNED_FREE_POINTER AllocatorAlignedFreePointer;

_Use_decl_annotations_
VOID
AllocatorAlignedFreePointer(
    PALLOCATOR Allocator,
    PVOID *BufferPointer
    )
{
    if (!ARGUMENT_PRESENT(BufferPointer)) {
        return;
    }

    if (!ARGUMENT_PRESENT(*BufferPointer)) {
        return;
    }

    AllocatorAlignedFree(Allocator, *BufferPointer);
    *BufferPointer = NULL;

    return;
}


//
// Offset versions.
//

ALLOCATOR_ALIGNED_OFFSET_MALLOC AllocatorAlignedOffsetMalloc;

_Use_decl_annotations_
PVOID
AllocatorAlignedOffsetMalloc(
    PALLOCATOR Allocator,
    SIZE_T Size,
    SIZE_T Alignment,
    SIZE_T Offset
    )
{
    return AlignedOffsetAllocBase(Allocator,
                                  Size,
                                  Alignment,
                                  Offset,
                                  FALSE);
}


ALLOCATOR_ALIGNED_OFFSET_CALLOC AllocatorAlignedOffsetCalloc;

_Use_decl_annotations_
PVOID
AllocatorAlignedOffsetCalloc(
    PALLOCATOR Allocator,
    SIZE_T NumberOfElements,
    SIZE_T ElementSize,
    SIZE_T Alignment,
    SIZE_T Offset
    )
{
    return AlignedOffsetAllocBase(Allocator,
                                  NumberOfElements * ElementSize,
                                  Alignment,
                                  Offset,
                                  TRUE);
}


ALLOCATOR_ALIGNED_OFFSET_REALLOC AllocatorAlignedOffsetReAlloc;

_Use_decl_annotations_
PVOID
AllocatorAlignedOffsetReAlloc(
    PALLOCATOR Allocator,
    PVOID Address,
    SIZE_T Size,
    SIZE_T Alignment,
    SIZE_T Offset
    )
{
    return AlignedOffsetReAllocBase(Allocator,
                                    Address,
                                    Size,
                                    Alignment,
                                    Offset,
                                    FALSE);
}


ALLOCATOR_ALIGNED_OFFSET_RECALLOC AllocatorAlignedOffsetReCalloc;

_Use_decl_annotations_
PVOID
AllocatorAlignedOffsetReCalloc(
    PALLOCATOR Allocator,
    PVOID Address,
    SIZE_T NumberOfElements,
    SIZE_T ElementSize,
    SIZE_T Alignment,
    SIZE_T Offset
    )
{
    return AlignedOffsetReAllocBase(Allocator,
                                    Address,
                                    NumberOfElements * ElementSize,
                                    Alignment,
                                    Offset,
                                    TRUE);
}


//
// Non-vtbl methods.
//

ALLOCATOR_INITIALIZE AllocatorInitialize;

_Use_decl_annotations_
HRESULT
AllocatorInitialize(
    PALLOCATOR Allocator
    )
{
    ULONG Flags = 0;
    ULONG LastError;
    HRESULT Result = S_OK;
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
        LastError = GetLastError();
        if (LastError != ERROR_NOT_ENOUGH_MEMORY) {
            SYS_ERROR(HeapCreate);
            Result = PH_E_SYSTEM_CALL_FAILED;
        } else {
            Result = E_OUTOFMEMORY;
        }
    }

    return Result;
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
