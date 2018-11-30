/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkBuildSolutionBatchFile.c

Abstract:

    This module implements the prepare file work callback routine for the
    build solution batch file as part of the CHM v1 algorithm implementation
    for the perfect hash library.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareBatchBuildSolutionFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PCHAR Base;
    PCHAR Output;

    UNREFERENCED_PARAMETER(Context);

    //
    // Initialize aliases.
    //

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the text and finish up.
    //

    OUTPUT_RAW("msbuild /nologo /m "
               "/t:Build "
               "/p:Configuration=Release;"
               "Platform=x64 ");

    OUTPUT_STRING(&(GetActivePath((*Item->FilePointer))->TableNameA));

    OUTPUT_RAW(".sln\r\n");

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
