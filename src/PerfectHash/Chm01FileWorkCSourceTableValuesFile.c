/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCSourceTableValuesFile.c

Abstract:

    This module implements the save file work callback routine for the C source
    table values file as part of the CHM v1 algorithm implementation for the
    perfect hash library.

    The C source table values file (extension _TableValues.c) is simply a C
    array of the requested table value type.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareCSourceTableValuesFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    PCSTRING Name;
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfo;
    ULONGLONG NumberOfElements;
    ULONGLONG TotalNumberOfElements;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = *Item->FilePointer;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    TableInfo = Table->TableInfoOnDisk;
    TotalNumberOfElements = TableInfo->NumberOfTableElements.QuadPart;
    NumberOfElements = TotalNumberOfElements >> 1;
    Output = Base = (PCHAR)File->BaseAddress;

    //
    // Write header.
    //

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table C Source Table Values File."
               "  Auto-generated.\n//\n\n");

    OUTPUT_INCLUDE_STDAFX_H();

    //
    // Write the table values array.
    //

    OUTPUT_RAW("#ifndef CPH_INDEX_ONLY\n\n");

    OUTPUT_RAW("const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableValueSizeInBytes = ");
    OUTPUT_INT(Table->ValueSizeInBytes == 0 ?
               sizeof(ULONG) : Table->ValueSizeInBytes);
    OUTPUT_RAW(";\n");

    OUTPUT_RAW("const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_NumberOfTableValues = ");
    OUTPUT_INT(NumberOfElements);
    OUTPUT_RAW(";\n\n");

    OUTPUT_RAW("#ifdef _WIN32\n"
               "#pragma data_seg(\".cphval\")\n"
               "#endif\n");

    OUTPUT_RAW("CPHVALUE ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableValues[");
    OUTPUT_INT(NumberOfElements);
    OUTPUT_RAW("] = { 0, };\n"
               "#ifdef _WIN32\n"
               "#pragma data_seg()\n"
               "#pragma comment(linker, "
               "\"/section:.cphval,rw");
    if (UseRwsSectionForTableValues(Table)) {
        *Output++ = 's';
    }
    OUTPUT_RAW("\")\n#endif\n#endif");

    //
    // Update the number of bytes written.
    //

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
