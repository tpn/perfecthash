/*++

Copyright (c) 2019 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkTestMkFileChm01.c

Abstract:

    This module implements the prepare file work callback routine for a
    compiled perfect hash table's Test.mk file.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareMakefileTestMkFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PCHAR Base;
    PCHAR Output;
    PCSTRING Name;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH Path;

    //
    // Initialize aliases.
    //

    File = *Item->FilePointer;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the Test.mk file.
    //

    OUTPUT_MAKEFILE_SPLASH_COMMENT(Test);

    OUTPUT_MAKEFILE_TARGET(Test);

    OUTPUT_MAKEFILE_TGT_VARS();

    OUTPUT_MAKEFILE_SOURCES(Test);

    //
    // We're done; update number of bytes written and finish up.
    //

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
