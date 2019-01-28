/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCHeaderTypesFile.c

Abstract:

    This module implements the prepare file work callback routine for the
    compiled perfect hash types.h C header file as part of the CHM v1 algorithm
    implementation for the perfect hash library.

    The types file (e.g. KernelBase_2486_Chm01_Crc32Rotate_And_Types.h) is
    responsible for defining the standard C->Nt-style typedefs (i.e. ULONG etc)
    as well as the CPHKEY and CPHVALUE types.

--*/

#include "stdafx.h"

extern const STRING CompiledPerfectHashTableTypesPreCHeaderRawCString;
extern const STRING CompiledPerfectHashTableTypesPostCHeaderRawCString;

_Use_decl_annotations_
HRESULT
PrepareCHeaderTypesFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PCHAR Base;
    PCHAR Output;
    PCSTRING Name;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;

    //
    // Initialize aliases.
    //

    File = *Item->FilePointer;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    Table = Context->Table;

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the header.
    //

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table C Header Types File.  "
               "Auto-generated.\n//\n\n"
               "#pragma once\n\n");

    if (IsIndexOnly(Table)) {
        OUTPUT_RAW("#define CPH_INDEX_ONLY 1\n\n");
    }

    //
    // Write the pre glue.
    //

    OUTPUT_STRING(&CompiledPerfectHashTableTypesPreCHeaderRawCString);

    //
    // Write the CPHKEY, CPHDKEY, CPHVALUE, and CPHINDEX types.
    //

    OUTPUT_RAW("typedef ");
    OUTPUT_STRING(Table->OriginalKeySizeTypeName);
    OUTPUT_RAW(" CPHKEY;\n");

    OUTPUT_RAW("typedef ");
    OUTPUT_STRING(Table->KeySizeTypeName);
    OUTPUT_RAW(" CPHDKEY;\n");

    OUTPUT_RAW("typedef ");
    OUTPUT_STRING(Table->ValueTypeName);
    OUTPUT_RAW(" CPHVALUE;\n");

    OUTPUT_RAW("typedef ");
    OUTPUT_STRING(Table->SeedTypeName);
    OUTPUT_RAW(" CPHSEED;\n");

    OUTPUT_RAW("typedef ");
    OUTPUT_STRING(Table->IndexTypeName);
    OUTPUT_RAW(" CPHINDEX;\n\n");

    //
    // Write the post glue.
    //

    OUTPUT_STRING(&CompiledPerfectHashTableTypesPostCHeaderRawCString);

    //
    // Finish up.
    //

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return S_OK;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
