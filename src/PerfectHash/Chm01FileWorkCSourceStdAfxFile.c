/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCSourceStdAfxFile.c

Abstract:

    This module implements the prepare file work callback routine for the
    stdafx.c C source file as part of the CHM v1 algorithm implementation
    for the perfect hash library.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareCSourceStdAfxFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PCHAR Base;
    PCHAR Output;
    PCSTRING Name;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_FILE File;

    UNREFERENCED_PARAMETER(Context);

    //
    // Initialize aliases.
    //

    File = *Item->FilePointer;
    Path = GetActivePath(File);
    Base = (PCHAR)File->BaseAddress;
    Name = &Path->BaseNameA;
    Output = Base;

    //
    // Write the include line and finish up.
    //

    OUTPUT_RAW("#include \"");
    OUTPUT_STRING(Name);
    OUTPUT_RAW(".h\"\n\n");

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
