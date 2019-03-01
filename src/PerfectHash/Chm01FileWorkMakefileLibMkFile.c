/*++

Copyright (c) 2019 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkLibMkFileChm01.c

Abstract:

    This module implements the prepare file work callback routine for a
    compiled perfect hash table's Lib.mk file.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareMakefileLibMkFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PCHAR Base;
    PCHAR Output;
    PCSTRING Name;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH Path;

    UNREFERENCED_PARAMETER(Context);

    //
    // Initialize aliases.
    //

    File = *Item->FilePointer;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the Lib.mk file.
    //

    OUTPUT_RAW("# Compiled Perfect Hash Table Lib Makefile.\n"
               "# Auto-generated.\n\n");

    OUTPUT_RAW("TARGET := ");
    OUTPUT_STRING(&LibTargetPrefix);
    OUTPUT_STRING(Name);
    OUTPUT_RAW(".a\n\n");

    OUTPUT_RAW("SOURCES := ");

    //
    // Main .c file.
    //

    OUTPUT_STRING(Name);
    OUTPUT_DOT_C();
    OUTPUT_SPACE_SLASH_NEWLINE_TAB();

    MAYBE_OUTPUT_INCLUDE_TABLE_VALUES_DOT_C();

    //
    // TableData.c
    //

    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableData.c\n");

    //
    // We're done; update number of bytes written and finish up.
    //

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
