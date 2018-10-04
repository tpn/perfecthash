/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCSourceFile.c

Abstract:

    This module implements the prepare file work callback routine for the C
    source file as part of the CHM v1 algorithm implementation for the perfect
    hash library.

    As the C source file contains no references to solution-specific information
    (e.g. what seed values were used, what the index or hash masks were, etc),
    everything can be prepared in the initial prepare routine; no separate save
    file routine is required.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareCSourceFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    ULONG Count;
    ULONGLONG Index;
    ULONG NumberOfSeeds;
    PCSTRING Name;
    PCSTRING Upper;
    STRING Algo = { 0 };
    HANDLE Event;
    ULONG WaitResult;
    HRESULT Result = S_OK;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH HeaderPath;
    PPERFECT_HASH_FILE HeaderFile;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;

    UNREFERENCED_PARAMETER(Item);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = Table->CSourceFile;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    Upper = &Path->TableNameUpperA;
    TableInfoOnDisk = Table->TableInfoOnDisk;
    NumberOfSeeds = TableInfoOnDisk->NumberOfSeeds;

    Algo.Buffer = (PSTR)(
        RtlOffsetToPointer(
            Path->TableNameUpperA.Buffer,
            Path->AdditionalSuffixAOffset
        )
    );

    Algo.Length = (
        Path->TableNameUpperA.Length -
        (USHORT)RtlPointerToOffset(Path->TableNameUpperA.Buffer, Algo.Buffer)
    );
    Algo.MaximumLength = Algo.Length;

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Wait for the header file to be prepared before referencing the file
    // instance.
    //

    Event = Context->PreparedCHeaderFileEvent;
    WaitResult = WaitForSingleObject(Event, INFINITE);
    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForSingleObject);
        goto Error;
    }

    HeaderFile = Table->CHeaderFile;
    HeaderPath = GetActivePath(HeaderFile);

    //
    // Write the keys.
    //

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table C Source File.  "
               "Auto-generated.\n//\n\n");

    OUTPUT_RAW("#include \"");
    OUTPUT_STRING(&HeaderPath->BaseNameA);

    OUTPUT_RAW(".h\"\n\nDECLARE_");
    OUTPUT_STRING(&Algo);
    OUTPUT_RAW("_INDEX_ROUTINE(\n    ");

    //
    // TableName
    //

    OUTPUT_STRING(Name);
    OUTPUT_RAW(",\n    ");

    //
    // Seed1 .. Seed[n]
    //

    for (Index = 0, Count = 1; Index < NumberOfSeeds; Index++, Count++) {
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_SEED");
        OUTPUT_INT(Count);
        OUTPUT_RAW(",\n    ");
    }

    //
    // HashMask
    //

    OUTPUT_STRING(Upper);
    OUTPUT_RAW("_HASH_MASK,\n    ");

    //
    // IndexMask
    //

    OUTPUT_STRING(Upper);
    OUTPUT_RAW("_INDEX_MASK\n);\n\n");

    //
    // C functions.
    //

    OUTPUT_RAW("//\n// Disable \"function ... selected for inline expansion\" "
               "warning.\n"
               "//\n\n#pragma warning(push)\n#pragma warning(disable: 4711)\n"
               "DECLARE_COMPILED_PERFECT_HASH_ROUTINES(\n    ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("\n);\n#pragma warning(pop)\n\n");

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
