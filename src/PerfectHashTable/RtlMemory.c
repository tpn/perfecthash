/*++

Copyright (c) 2016 Trent Nelson <trent@trent.me>

Module Name:

    RtlMemory.c

Abstract:

    This module implements routines related to memory for the Rtl component of
    the perfect hash table library.  Routines are provided for copying pages
    and filling pages with a specific byte pattern.

    N.B. Optimized assembly versions of these routines may be available for
         certain platforms (i.e. x64), which will take precedence over the
         routines in this module.

--*/

#include "stdafx.h"

RTL_COPY_PAGES RtlCopyPages;

_Use_decl_annotations_
HRESULT
RtlCopyPages(
    PRTL Rtl,
    PCHAR Dest,
    const PCHAR Source,
    ULONG NumberOfPages
    )
{
    SIZE_T NumberOfBytes;

    NumberOfBytes = (SIZE_T)NumberOfPages << (SIZE_T)PAGE_SHIFT;

    CopyMemory(Dest, Source, NumberOfBytes);

    return S_OK;
}

RTL_FILL_PAGES RtlFillPages;

_Use_decl_annotations_
HRESULT
RtlFillPages(
    PRTL Rtl,
    PCHAR Dest,
    BYTE Byte,
    ULONG NumberOfPages
    )
{
    SIZE_T NumberOfBytes;

    NumberOfBytes = (SIZE_T)NumberOfPages << (SIZE_T)PAGE_SHIFT;

    FillMemory(Dest, NumberOfBytes, Byte);

    return S_OK;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
