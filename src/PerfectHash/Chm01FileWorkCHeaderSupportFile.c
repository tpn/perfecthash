/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCHeaderSupportFile.c

Abstract:

    This module implements the prepare file work callback routine for the
    support C header file as part of the CHM v1 algorithm implementation
    for the perfect hash library.

--*/

#include "stdafx.h"
#include "CompiledPerfectHashTableSupport_CHeader_RawCString.h"

_Use_decl_annotations_
HRESULT
PrepareCHeaderSupportFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PCHAR Base;
    PCHAR Output;
    PPERFECT_HASH_FILE File;

    UNREFERENCED_PARAMETER(Context);

    //
    // Initialize aliases.
    //

    File = *Item->FilePointer;
    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the text and finish up.
    //

    OUTPUT_INCLUDE_STDAFX_H();
    OUTPUT_STRING(RawCString);

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
