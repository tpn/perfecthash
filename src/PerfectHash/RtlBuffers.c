/*++

Copyright (c) 2016 Trent Nelson <trent@trent.me>

Module Name:

    RtlBuffers.c

Abstract:

    This module implements routines related to creating and destroying
    generic buffers backed by guard pages.

--*/

#include "stdafx.h"

RTL_CREATE_BUFFER RtlCreateBuffer;

_Use_decl_annotations_
HRESULT
RtlCreateBuffer(
    PRTL Rtl,
    PHANDLE TargetProcessHandle,
    ULONG NumberOfPages,
    PULONG AdditionalProtectionFlags,
    PULONGLONG UsableBufferSizeInBytes,
    PVOID *BufferAddress
    )
{
    BOOL Success;
    PVOID Buffer;
    PBYTE Unusable;
    HRESULT Result;
    HANDLE ProcessHandle;
    ULONG ProtectionFlags;
    ULONG OldProtectionFlags;
    ULARGE_INTEGER TotalNumberOfPages;
    ULARGE_INTEGER AllocSizeInBytes;
    ULARGE_INTEGER UsableSizeInBytes;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TargetProcessHandle)) {
        ProcessHandle = GetCurrentProcess();
    } else {
        if (!*TargetProcessHandle) {
            *TargetProcessHandle = GetCurrentProcess();
        }
        ProcessHandle = *TargetProcessHandle;
    }

    if (!ARGUMENT_PRESENT(UsableBufferSizeInBytes)) {
        return E_POINTER;
    } else {
        *UsableBufferSizeInBytes = 0;
    }

    if (!ARGUMENT_PRESENT(BufferAddress)) {
        return E_POINTER;
    } else {
        *BufferAddress = NULL;
    }

    TotalNumberOfPages.QuadPart = (ULONGLONG)NumberOfPages + 1;

    //
    // Convert total number of pages into total number of bytes (alloc size)
    // and verify it hasn't overflowed either (thus, 4GB is the current maximum
    // size allowed by this routine).
    //

    AllocSizeInBytes.QuadPart = TotalNumberOfPages.QuadPart;
    AllocSizeInBytes.QuadPart <<= (ULONGLONG)PAGE_SHIFT;

    ProtectionFlags = PAGE_READWRITE;
    if (ARGUMENT_PRESENT(AdditionalProtectionFlags)) {
        ProtectionFlags |= *AdditionalProtectionFlags;
    }

    //
    // Validation of parameters complete.  Proceed with buffer allocation.
    //

    Buffer = VirtualAllocEx(ProcessHandle,
                            NULL,
                            AllocSizeInBytes.QuadPart,
                            MEM_COMMIT,
                            ProtectionFlags);

    if (!Buffer) {
        SYS_ERROR(VirtualAllocEx);
        return E_OUTOFMEMORY;
    }

    //
    // Buffer was successfully allocated.  Any failures after this point should
    // `goto Error` to ensure the memory is freed.
    //
    // Calculate the usable size and corresponding unusable address.
    //

    UsableSizeInBytes.QuadPart = (
        (ULONGLONG)NumberOfPages <<
        (ULONGLONG)PAGE_SHIFT
    );

    Unusable = (PBYTE)Buffer;
    Unusable += UsableSizeInBytes.QuadPart;

    //
    // Change the protection of the trailing page to PAGE_NOACCESS.
    //

    ProtectionFlags = PAGE_NOACCESS;
    Success = VirtualProtectEx(ProcessHandle,
                               Unusable,
                               PAGE_SIZE,
                               ProtectionFlags,
                               &OldProtectionFlags);

    if (!Success) {
        SYS_ERROR(VirtualProtectEx);
        goto Error;
    }

    //
    // We're done, goto End.
    //

    Result = S_OK;
    goto End;

Error:

    Result = E_FAIL;

    //
    // Buffer should be non-NULL at this point.  Assert this invariant, free the
    // allocated memory, clear the buffer pointer and set the alloc size to 0.
    //

    ASSERT(Buffer);
    if (!VirtualFreeEx(ProcessHandle, Buffer, 0, MEM_RELEASE)) {
        SYS_ERROR(VirtualFreeEx);
    }
    Buffer = NULL;

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Update caller's parameters and return.
    //

    *BufferAddress = Buffer;

    if (SUCCEEDED(Result)) {
        *UsableBufferSizeInBytes = UsableSizeInBytes.QuadPart;
    }

    return Result;
}

RTL_CREATE_MULTIPLE_BUFFERS RtlCreateMultipleBuffers;

_Use_decl_annotations_
HRESULT
RtlCreateMultipleBuffers(
    PRTL Rtl,
    PHANDLE TargetProcessHandle,
    ULONG PageSize,
    ULONG NumberOfBuffers,
    ULONG NumberOfPagesPerBuffer,
    PULONG AdditionalProtectionFlags,
    PULONG AdditionalAllocationTypeFlags,
    PULONGLONG UsableBufferSizeInBytesPerBuffer,
    PULONGLONG TotalBufferSizeInBytes,
    PVOID *BufferAddress
    )
/*++

Routine Description:

    Allocates a single buffer using VirtualAllocEx() sized using the following:

        (PageSize * NumberOfPagesPerBuffer * NumberOfBuffers) +
        (NumberOfBuffers * PageSize)

    This includes a guard page following each buffer's set of pages.  Memory
    protection is set on the guard page such that any access will trap.

Arguments:

    Rtl - Supplies a pointer to an initialized RTL structure.

    TargetProcessHandle - Optionally supplies a pointer to a variable that
        contains a process handle for which the memory is to be allocated.
        If non-NULL, but pointed-to value is 0, this will receive the handle
        of the current process.

    PageSize - Indicates the page size indicated by the NumberOfPagesPerBuffer
        parameter.  This should be either 4096 or 2MB.  If 2MB, the caller
        should also ensure a) MEM_LARGE_PAGES is also provided as the value in
        the parameter AdditionalAllocationTypeFlags, and b) relevant privileges
        have been obtained (e.g. Rtl->EnableManageVolumePrivilege() and
        Rtl->EnableLockMemoryPrivilege()).

    NumberOfBuffers - Supplies the number of individual buffers being serviced
        by this request.  This will also reflect the number of implicit guard
        pages included in the allocation.

    NumberOfPagesPerBuffer - Supplies the number of pages to assign for each
        buffer.

    AdditionalProtectionFlags - Optionally supplies flags that will be OR'd
        with the flProtect flags parameter of VirtualAllocEx() (e.g.
        PAGE_NOCACHE or PAGE_WRITECOMBINE).

    AdditionalAllocationTypeFlags - Optionally supplies flags that will be OR'd
        with the flAllocationType flags parameter of VirtualAllocEx().  This
        should include MEM_LARGE_PAGES if large pages are desired.

    UsableBufferSizeInBytesPerBuffer - Supplies the address of a variable that
        receives the number of usable bytes per buffer.

    TotalBufferSizeInBytes - Supplies the address of a variable that receives
        the total allocation size provided to VirtualAllocEx().

    BufferAddress - Supplies the address of a variable that receives the
        base address of the allocation if successful.

Return Value:

    TRUE on success, FALSE on failure.

--*/

{
    BOOL Success;
    ULONG Index;
    PVOID Buffer;
    PBYTE Unusable;
    HRESULT Result;
    HANDLE ProcessHandle;
    ULONG ProtectionFlags;
    ULONG OldProtectionFlags;
    ULONG AllocationFlags;
    ULARGE_INTEGER TotalNumberOfPages;
    ULARGE_INTEGER AllocSizeInBytes;
    ULONGLONG UsableBufferSizeInBytes;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    if (!NumberOfPagesPerBuffer) {
        return E_POINTER;
    }

    if (!NumberOfBuffers) {
        return E_POINTER;
    }

    //
    // Verify page size is either 4KB or 2MB.
    //

    if (PageSize != 4096 && PageSize != (1 << 21)) {
        return E_INVALIDARG;
    }

    //
    // If page size is 2MB, verify MEM_LARGE_PAGES has also been requested.
    //

    if (PageSize == (1 << 21)) {
        if (!ARGUMENT_PRESENT(AdditionalAllocationTypeFlags)) {
            return E_INVALIDARG;
        }
        if (!(*AdditionalAllocationTypeFlags & MEM_LARGE_PAGES)) {
            return E_INVALIDARG;
        }
    }

    if (!ARGUMENT_PRESENT(TargetProcessHandle)) {
        ProcessHandle = GetCurrentProcess();
    } else {
        if (!*TargetProcessHandle) {
            *TargetProcessHandle = GetCurrentProcess();
        }
        ProcessHandle = *TargetProcessHandle;
    }

    if (!ARGUMENT_PRESENT(UsableBufferSizeInBytesPerBuffer)) {
        return E_POINTER;
    } else {
        *UsableBufferSizeInBytesPerBuffer = 0;
    }

    if (!ARGUMENT_PRESENT(TotalBufferSizeInBytes)) {
        return E_POINTER;
    } else {
        *TotalBufferSizeInBytes = 0;
    }

    if (!ARGUMENT_PRESENT(BufferAddress)) {
        return E_POINTER;
    } else {
        *BufferAddress = NULL;
    }

    TotalNumberOfPages.QuadPart = (

        //
        // Account for the buffer related pages.
        //

        ((ULONGLONG)NumberOfPagesPerBuffer * (ULONGLONG)NumberOfBuffers) +

        //
        // Account for the guard pages; one for each buffer.
        //

        ((ULONGLONG)NumberOfBuffers)
    );

    //
    // Calculate the total allocation size required.
    //

    AllocSizeInBytes.QuadPart = (
        TotalNumberOfPages.QuadPart * (ULONGLONG)PageSize
    );

    ProtectionFlags = PAGE_READWRITE;
    if (ARGUMENT_PRESENT(AdditionalProtectionFlags)) {
        ProtectionFlags |= *AdditionalProtectionFlags;
    }

    AllocationFlags = MEM_RESERVE | MEM_COMMIT;
    if (ARGUMENT_PRESENT(AdditionalAllocationTypeFlags)) {
        AllocationFlags |= *AdditionalAllocationTypeFlags;
    }

    //
    // Validation of parameters complete.  Proceed with buffer allocation.
    //

    Buffer = VirtualAllocEx(ProcessHandle,
                            NULL,
                            AllocSizeInBytes.QuadPart,
                            AllocationFlags,
                            ProtectionFlags);

    if (!Buffer) {
        SYS_ERROR(VirtualAllocEx);
        return E_OUTOFMEMORY;
    }

    //
    // Buffer was successfully allocated.  Any failures after this point should
    // `goto Error` to ensure the memory is freed.
    //

    UsableBufferSizeInBytes = (ULONGLONG)(
        (ULONGLONG)NumberOfPagesPerBuffer *
        (ULONGLONG)PageSize
    );

    //
    // Loop through the assigned memory and toggle protection to PAGE_NOACCESS
    // for the guard pages.
    //

    Unusable = (PBYTE)Buffer;
    ProtectionFlags = PAGE_NOACCESS;

    for (Index = 0; Index < NumberOfBuffers; Index++) {

        Unusable += UsableBufferSizeInBytes;

        Success = VirtualProtectEx(ProcessHandle,
                                   Unusable,
                                   PageSize,
                                   ProtectionFlags,
                                   &OldProtectionFlags);

        if (!Success) {
            SYS_ERROR(VirtualProtectEx);
            goto Error;
        }

        //
        // Advance past this guard page.
        //

        Unusable += PageSize;
    }

    //
    // We're done, indicate success and finish up.
    //

    Result = S_OK;
    goto End;

Error:

    Result = E_FAIL;

    //
    // Buffer should be non-NULL at this point.  Assert this invariant, free the
    // allocated memory, clear the buffer pointer and set the alloc size to 0.
    //

    ASSERT(Buffer);
    if (!VirtualFreeEx(ProcessHandle, Buffer, 0, MEM_RELEASE)) {
        SYS_ERROR(VirtualFreeEx);
    }
    Buffer = NULL;

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Update caller's parameters and return.
    //

    *BufferAddress = Buffer;

    if (SUCCEEDED(Result)) {
        *TotalBufferSizeInBytes = AllocSizeInBytes.QuadPart;
        *UsableBufferSizeInBytesPerBuffer = UsableBufferSizeInBytes;
    }

    return Result;
}

RTL_DESTROY_BUFFER RtlDestroyBuffer;

_Use_decl_annotations_
HRESULT
RtlDestroyBuffer(
    PRTL Rtl,
    HANDLE ProcessHandle,
    PVOID *Address,
    SIZE_T Size
    )
{
    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

#ifdef PH_WINDOWS
    if (!ARGUMENT_PRESENT(ProcessHandle)) {
        return E_POINTER;
    }
#endif

    if (!ARGUMENT_PRESENT(Address)) {
        return E_POINTER;
    }
   
    if (!VirtualFreeEx(ProcessHandle, *Address, Size, MEM_RELEASE)) {
        SYS_ERROR(VirtualFreeEx);
        return E_FAIL;
    }

    *Address = NULL;
    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
