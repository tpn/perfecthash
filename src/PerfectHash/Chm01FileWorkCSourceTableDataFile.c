/*++

Copyright (c) 2018-2019 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCSourceTableDataFile.c

Abstract:

    This module implements the save file work callback routine for the C source
    table data file as part of the CHM v1 algorithm implementation for the
    perfect hash library.

    The C source table data file (extension _TableData.c) is simply a C array of
    the "assigned" array that is obtained during the graph solving step (i.e. it
    is identical in nature to the .pht1 table data file).

    This file has no preparation step (unlike, say, the C header file), as there
    is no work that can be done until the graph has been solved and table data
    is available to save to disk.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
SaveCSourceTableDataFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    ULONG Value;
    ULONG Count;
    PULONG Long;
    PULONG Seed;
    PGRAPH Graph;
    PULONG Source;
    ULONG NumberOfSeeds;
    PCSTRING Name;
    PCSTRING Upper;
    ULONGLONG Index;
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfo;
    ULONGLONG NumberOfElements;
    ULONGLONG TotalNumberOfElements;
    const ULONG Indent = 0x20202020;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = *Item->FilePointer;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    Upper = &Path->TableNameUpperA;
    TableInfo = Table->TableInfoOnDisk;
    TotalNumberOfElements = TableInfo->NumberOfTableElements.QuadPart;
    NumberOfElements = TotalNumberOfElements >> 1;
    Graph = (PGRAPH)Context->SolvedContext;
    NumberOfSeeds = Graph->NumberOfSeeds;
    Source = Graph->Assigned;
    Output = Base = (PCHAR)File->BaseAddress;

    //
    // Write header.
    //

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table C Source Table Data File.  "
               "Auto-generated.\n//\n\n");

    OUTPUT_INCLUDE_STDAFX_H();

    //
    // Write seed and mask data.
    //

    OUTPUT_RAW("#pragma const_seg(\".cphsm\")\n");
    OUTPUT_RAW("const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Seeds[");
    OUTPUT_INT(NumberOfSeeds);
    OUTPUT_RAW("] = {\n");

    Seed = &Graph->FirstSeed;

    for (Index = 0, Count = 0; Index < NumberOfSeeds; Index++) {

        if (Count == 0) {
            INDENT();
        }

        OUTPUT_HEX(*Seed++);

        *Output++ = ',';

        if (++Count == 4) {
            Count = 0;
            *Output++ = '\n';
        } else {
            *Output++ = ' ';
        }
    }

    //
    // If the last character written was a trailing space, replace
    // it with a newline.
    //

    if (*(Output - 1) == ' ') {
        *(Output - 1) = '\n';
    }

    OUTPUT_RAW("};\n\n");

    Seed = &Graph->FirstSeed;

    for (Index = 0, Count = 1; Index < NumberOfSeeds; Index++, Count++) {
        OUTPUT_RAW("const ULONG ");
        OUTPUT_STRING(Name);
        OUTPUT_RAW("_Seed");
        OUTPUT_INT(Count);
        OUTPUT_RAW(" = ");
        OUTPUT_HEX(*Seed++);
        OUTPUT_RAW(";\n");
    }

    OUTPUT_RAW("\nconst ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_HashMask = ");
    OUTPUT_HEX(TableInfo->HashMask);
    OUTPUT_RAW(";\nconst ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_IndexMask = ");
    OUTPUT_HEX(TableInfo->IndexMask);

    OUTPUT_RAW(";\n#pragma const_seg()\n\n");

    //
    // Write a table values array.
    //

    OUTPUT_RAW("#pragma data_seg(\".cphval\")\n");
    OUTPUT_RAW("CPHVALUE ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableValues[");
    OUTPUT_INT(NumberOfElements);
    OUTPUT_RAW("] = { 0, };\n#pragma data_seg()\n"
               "#pragma comment(linker, "
               "\"/section:.cphval,rw\")\n\n");

    //
    // Write the table data.
    //

    OUTPUT_RAW("#pragma const_seg(\".cphdata\")\n");

    OUTPUT_RAW("const ");
    OUTPUT_STRING(Table->TableDataArrayTypeName);
    OUTPUT_RAW(" ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableData[");
    OUTPUT_INT(TotalNumberOfElements);
    OUTPUT_RAW("] = {\n\n    //\n    // 1st half.\n    //\n\n");

    for (Index = 0, Count = 0; Index < TotalNumberOfElements; Index++) {

        if (Count == 0) {
            INDENT();
        }

        Value = *Source++;

        OUTPUT_HEX(Value);

        *Output++ = ',';

        if (++Count == 4) {
            Count = 0;
            *Output++ = '\n';
        } else {
            *Output++ = ' ';
        }

        if (Index == NumberOfElements-1) {
            OUTPUT_RAW("\n    //\n    // 2nd half.\n    //\n\n");
        }
    }

    //
    // If the last character written was a trailing space, replace
    // it with a newline.
    //

    if (*(Output - 1) == ' ') {
        *(Output - 1) = '\n';
    }

    OUTPUT_RAW("};\n");

    //
    // Update the number of bytes written.
    //

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
