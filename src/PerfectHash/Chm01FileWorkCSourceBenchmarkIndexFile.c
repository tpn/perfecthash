/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCSourceBenchmarkIndexFile.c

Abstract:

    This module implements the prepare file work callback routine for the
    benchmark index C source file as part of the CHM v1 algorithm implementation
    for the perfect hash library.

--*/

#include "stdafx.h"
#include "CompiledPerfectHashTableBenchmarkIndex_CSource_RawCString.h"

_Use_decl_annotations_
HRESULT
PrepareCSourceBenchmarkIndexFileChm01(
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

    OUTPUT_INCLUDE_STDAFX_AND_SUPPORT_H();

    OUTPUT_PRAGMA_WARNING_DISABLE_FUNC_SELECTED_FOR_INLINE_EXP_WARNING();

    OUTPUT_STRING(RawCString);

    OUTPUT_PRAGMA_WARNING_POP();

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
