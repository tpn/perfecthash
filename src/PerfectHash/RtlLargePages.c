/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    RtlLargePages.c

Abstract:

    This module implements routines related to large page allocations for
    the Rtl component of the perfect hash table library.  Routines are
    provided for trying large page allocations, and trying large page
    memory maps.

--*/

#include "stdafx.h"

RTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC RtlTryLargePageVirtualAlloc;

_Use_decl_annotations_
LPVOID
RtlTryLargePageVirtualAlloc(
    PRTL     Rtl,
    LPVOID   lpAddress,
    SIZE_T   dwSize,
    DWORD    flAllocationType,
    DWORD    flProtect,
    PBOOLEAN LargePages
    )
{
    PVOID BaseAddress;

    UNREFERENCED_PARAMETER(Rtl);

    if (!*LargePages) {
        goto Fallback;
    }

    //
    // Attempt a large page VirtualAlloc().
    //

    BaseAddress = VirtualAlloc(lpAddress,
                               max(dwSize, GetLargePageMinimum()),
                               flAllocationType | MEM_LARGE_PAGES,
                               flProtect);

    if (BaseAddress) {
        return BaseAddress;
    }

    //
    // Indicate large pages failed.
    //

    *LargePages = FALSE;

    //
    // Try again.
    //

Fallback:

    BaseAddress = VirtualAlloc(lpAddress,
                               dwSize,
                               flAllocationType,
                               flProtect);

    return BaseAddress;
}

RTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC RtlNoLargePageVirtualAlloc;

_Use_decl_annotations_
LPVOID
RtlNoLargePageVirtualAlloc(
    PRTL     Rtl,
    LPVOID   lpAddress,
    SIZE_T   dwSize,
    DWORD    flAllocationType,
    DWORD    flProtect,
    PBOOLEAN LargePages
    )
{
    PVOID BaseAddress;

    UNREFERENCED_PARAMETER(Rtl);

    *LargePages = FALSE;

    BaseAddress = VirtualAlloc(lpAddress,
                               dwSize,
                               flAllocationType,
                               flProtect);

    return BaseAddress;
}

RTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC_EX RtlTryLargePageVirtualAllocEx;

_Use_decl_annotations_
LPVOID
RtlTryLargePageVirtualAllocEx(
    PRTL     Rtl,
    HANDLE   hProcess,
    LPVOID   lpAddress,
    SIZE_T   dwSize,
    DWORD    flAllocationType,
    DWORD    flProtect,
    PBOOLEAN LargePages
    )
{
    PVOID BaseAddress;

    UNREFERENCED_PARAMETER(Rtl);

    if (!*LargePages) {
        goto Fallback;
    }

    //
    // Attempt a large page VirtualAllocEx().
    //

    BaseAddress = VirtualAllocEx(hProcess,
                                 lpAddress,
                                 max(dwSize, GetLargePageMinimum()),
                                 flAllocationType | MEM_LARGE_PAGES,
                                 flProtect);

    if (BaseAddress) {
        return BaseAddress;
    }

    //
    // Indicate large pages failed.
    //

    *LargePages = FALSE;

    //
    // Try again.
    //

Fallback:

    BaseAddress = VirtualAllocEx(hProcess,
                                 lpAddress,
                                 dwSize,
                                 flAllocationType,
                                 flProtect);

    return BaseAddress;
}

RTL_TRY_LARGE_PAGE_VIRTUAL_ALLOC_EX RtlNoLargePageVirtualAllocEx;

_Use_decl_annotations_
LPVOID
RtlNoLargePageVirtualAllocEx(
    PRTL     Rtl,
    HANDLE   hProcess,
    LPVOID   lpAddress,
    SIZE_T   dwSize,
    DWORD    flAllocationType,
    DWORD    flProtect,
    PBOOLEAN LargePages
    )
{
    PVOID BaseAddress;

    UNREFERENCED_PARAMETER(Rtl);

    *LargePages = FALSE;

    BaseAddress = VirtualAllocEx(hProcess,
                                 lpAddress,
                                 dwSize,
                                 flAllocationType,
                                 flProtect);

    return BaseAddress;
}

RTL_TRY_LARGE_PAGE_CREATE_FILE_MAPPING_W RtlTryLargePageCreateFileMappingW;

_Use_decl_annotations_
HANDLE
RtlTryLargePageCreateFileMappingW(
    PRTL Rtl,
    HANDLE hFile,
    LPSECURITY_ATTRIBUTES lpFileMappingAttributes,
    DWORD flProtect,
    DWORD dwMaximumSizeHigh,
    DWORD dwMaximumSizeLow,
    LPCWSTR lpName,
    PBOOLEAN LargePages
    )
{
    HANDLE Handle;
    ULARGE_INTEGER Size;

    UNREFERENCED_PARAMETER(Rtl);

    if (!*LargePages) {
        goto Fallback;
    }

    Size.HighPart = dwMaximumSizeHigh;

    if (!Size.HighPart && dwMaximumSizeLow) {
        Size.LowPart = max(dwMaximumSizeLow, (DWORD)GetLargePageMinimum());
    } else {
        Size.LowPart = dwMaximumSizeLow;
    }

    //
    // Attempt to create a file mapping using large pages.
    //

    Handle = CreateFileMappingW(hFile,
                                lpFileMappingAttributes,
                                flProtect | SEC_LARGE_PAGES,
                                Size.HighPart,
                                Size.LowPart,
                                lpName);

    if (Handle && Handle != INVALID_HANDLE_VALUE) {
        return Handle;
    }

    //
    // Indicate large pages failed.
    //

    *LargePages = FALSE;

    //
    // Try again without large pages.
    //

Fallback:

    Handle = CreateFileMappingW(hFile,
                                lpFileMappingAttributes,
                                flProtect,
                                dwMaximumSizeHigh,
                                dwMaximumSizeLow,
                                lpName);

    return Handle;
}

RTL_TRY_LARGE_PAGE_CREATE_FILE_MAPPING_W RtlNoLargePageCreateFileMappingW;

_Use_decl_annotations_
HANDLE
RtlNoLargePageCreateFileMappingW(
    PRTL Rtl,
    HANDLE hFile,
    LPSECURITY_ATTRIBUTES lpFileMappingAttributes,
    DWORD flProtect,
    DWORD dwMaximumSizeHigh,
    DWORD dwMaximumSizeLow,
    LPCWSTR lpName,
    PBOOLEAN LargePages
    )
{
    HANDLE Handle;

    UNREFERENCED_PARAMETER(Rtl);

    *LargePages = FALSE;

    Handle = CreateFileMappingW(hFile,
                                lpFileMappingAttributes,
                                flProtect,
                                dwMaximumSizeHigh,
                                dwMaximumSizeLow,
                                lpName);

    return Handle;
}

RTL_INITIALIZE_LARGE_PAGES RtlInitializeLargePages;
extern ENABLE_LOCK_MEMORY_PRIVILEGE EnableLockMemoryPrivilege;

_Use_decl_annotations_
HRESULT
RtlInitializeLargePages(
    PRTL Rtl
    )
{
    Rtl->Flags.IsLargePageEnabled = (
        EnableLockMemoryPrivilege(Rtl) == S_OK ? TRUE : FALSE
    );
    Rtl->LargePageMinimum = GetLargePageMinimum();

    if (Rtl->Flags.IsLargePageEnabled) {
        Rtl->Vtbl->TryLargePageVirtualAlloc = RtlTryLargePageVirtualAlloc;
        Rtl->Vtbl->TryLargePageVirtualAllocEx = RtlTryLargePageVirtualAllocEx;
        Rtl->Vtbl->TryLargePageCreateFileMappingW =
            RtlTryLargePageCreateFileMappingW;
    } else {
        Rtl->Vtbl->TryLargePageVirtualAlloc = RtlNoLargePageVirtualAlloc;
        Rtl->Vtbl->TryLargePageVirtualAllocEx = RtlNoLargePageVirtualAllocEx;
        Rtl->Vtbl->TryLargePageCreateFileMappingW =
            RtlNoLargePageCreateFileMappingW;
    }

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
