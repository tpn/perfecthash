/*++

Copyright (c) 2019-2023. Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkSoMkFileChm01.c

Abstract:

    This module implements the prepare file work callback routine for a
    compiled perfect hash table's So.mk file.

    N.B. "So" stands for shared object (i.e. .so suffix); the UNIX-equivalent
         to Windows' DLLs.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareMakefileSoMkFileChm01(
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
    // Write the So.mk file.
    //

    OUTPUT_RAW("# Compiled Perfect Hash Table Shared Object Makefile.\n"
               "# Auto-generated.\n\n");

    OUTPUT_RAW("TGT_CFLAGS := -fPIC\n");
    OUTPUT_RAW("TGT_LDFLAGS := -shared\n\n");

    OUTPUT_RAW("TARGET := lib");
    OUTPUT_STRING(Name);
    OUTPUT_RAW(".so\n\n");

    OUTPUT_RAW("SOURCES := \\\n\t");

    //
    // Main .c file.
    //

    OUTPUT_STRING(Name);
    OUTPUT_DOT_C();
    OUTPUT_SPACE_SLASH_NEWLINE_TAB();

    MAYBE_OUTPUT_INCLUDE_TABLE_VALUES_DOT_C();

    MAYBE_OUTPUT_INCLUDE_KEYS_DOT_C();

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
