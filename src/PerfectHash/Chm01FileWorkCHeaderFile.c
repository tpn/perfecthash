/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCHeaderFile.c

Abstract:

    This module implements the prepare and save file work callback routines for
    the C header file as part of the CHM v1 algorithm implementation for the
    perfect hash library.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareCHeaderFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    ULONG Count;
    ULONG NumberOfSeeds;
    ULONGLONG Index;
    PCSTRING Name;
    PPERFECT_HASH_KEYS Keys;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;

    UNREFERENCED_PARAMETER(Item);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    Keys = Table->Keys;
    File = Table->CHeaderFile;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    TableInfoOnDisk = Table->TableInfoOnDisk;
    NumberOfSeeds = TableInfoOnDisk->NumberOfSeeds;

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the keys.
    //

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table C Header File.  "
               "Auto-generated.\n//\n\n"
               "#pragma once\n\n"
               "#include <CompiledPerfectHash.h>\n\n"
               "#ifdef __cplusplus\n"
               "extern \"C\" {\n"
               "#endif\n\n");

    OUTPUT_RAW("extern const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Seeds[");
    OUTPUT_INT(NumberOfSeeds);
    OUTPUT_RAW("];\n");

    for (Index = 0, Count = 1; Index < NumberOfSeeds; Index++, Count++) {
        OUTPUT_RAW("extern const ULONG ");
        OUTPUT_STRING(Name);
        OUTPUT_RAW("_Seed");
        OUTPUT_INT(Count);
        OUTPUT_RAW(";\n");
    }

    OUTPUT_RAW("\nextern const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_HashMask;\nextern const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_IndexMask;\n\nextern const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableData[];\nextern ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableValues[];\n\n");

    OUTPUT_RAW("extern const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Keys[];\n\n");

    OUTPUT_RAW("DEFINE_COMPILED_PERFECT_HASH_ROUTINES(\n    ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("\n);\n\n");

    //
    // Data macros.
    //

    OUTPUT_RAW("#ifndef PhTableKeys\n#define PhTableKeys ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Keys\n#endif\n\n"
               "#ifndef PhTableData\n#define PhTableData ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableData\n#endif\n\n"
               "#ifndef PhTableValues\n#define PhTableValues ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_TableValues\n#endif\n\n");

    //
    // C function macros.
    //

    OUTPUT_RAW("#ifndef PhTableIndex\n"
               "#define PhTableIndex CompiledPerfectHash_");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Index\n#endif\n\n"
               "#ifndef PhTableLookup\n"
               "#define PhTableLookup CompiledPerfectHash_");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Lookup\n#endif\n\n"
               "#ifndef PhTableInsert\n"
               "#define PhTableInsert CompiledPerfectHash_");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Insert\n#endif\n\n"
               "#ifndef PhTableDelete\n"
               "#define PhTableDelete CompiledPerfectHash_");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Delete\n#endif\n\n");

    //
    // Inline C function macros.
    //

    OUTPUT_RAW("#ifndef PhTableIndexInline\n"
               "#define PhTableIndexInline CompiledPerfectHash_");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_IndexInline\n#endif\n\n"
               "#ifndef PhTableLookupInline\n"
               "#define PhTableLookupInline CompiledPerfectHash_");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_LookupInline\n#endif\n\n"
               "#ifndef PhTableInsertInline\n"
               "#define PhTableInsertInline CompiledPerfectHash_");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_InsertInline\n#endif\n\n"
               "#ifndef PhTableDeleteInline\n"
               "#define PhTableDeleteInline CompiledPerfectHash_");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_DeleteInline\n#endif\n\n");

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return S_OK;
}

_Use_decl_annotations_
HRESULT
SaveCHeaderFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Output;
    ULONG Count;
    PULONG Seed;
    PGRAPH Graph;
    PULONG Source;
    ULONG NumberOfSeeds;
    PCSTRING Name;
    PCSTRING Upper;
    STRING Algo = { 0 };
    ULONGLONG Index;
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfo;
    ULONGLONG NumberOfElements;
    ULONGLONG TotalNumberOfElements;
    LARGE_INTEGER EndOfFile;

    UNREFERENCED_PARAMETER(Item);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = Table->CHeaderFile;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    Upper = &Path->TableNameUpperA;
    TableInfo = Table->TableInfoOnDisk;
    TotalNumberOfElements = TableInfo->NumberOfTableElements.QuadPart;
    NumberOfElements = TotalNumberOfElements >> 1;
    Graph = (PGRAPH)Context->SolvedContext;
    NumberOfSeeds = Graph->NumberOfSeeds;
    Source = Graph->Assigned;

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

    //
    // Pick up the offset from where we left off.
    //

    ASSERT(File->NumberOfBytesWritten.QuadPart > 0);

    Output = (PSTR)(
        RtlOffsetToPointer(
            File->BaseAddress,
            File->NumberOfBytesWritten.QuadPart
        )
    );

    //
    // Write seeds.
    //

    Seed = &Graph->FirstSeed;

    for (Index = 0, Count = 1; Index < NumberOfSeeds; Index++, Count++) {
        OUTPUT_RAW("#define ");
        OUTPUT_STRING(Upper);
        OUTPUT_RAW("_SEED");
        OUTPUT_INT(Count);
        *Output++ = ' ';
        OUTPUT_HEX(*Seed++);
        *Output++ = '\n';
    }

    //
    // Write masks.
    //

    OUTPUT_RAW("\n#define ");
    OUTPUT_STRING(Upper);
    OUTPUT_RAW("_HASH_MASK ");
    OUTPUT_HEX(TableInfo->HashMask);
    OUTPUT_RAW("\n#define ");
    OUTPUT_STRING(Upper);
    OUTPUT_RAW("_INDEX_MASK ");
    OUTPUT_HEX(TableInfo->IndexMask);
    OUTPUT_RAW("\n\n");

    OUTPUT_RAW("//\n// Disable \"unreferenced inline function has been "
               "removed\" warning.\n"
               "//\n\n#pragma warning(push)\n#pragma warning(disable: 4514)\n");

    OUTPUT_RAW("DECLARE_");
    OUTPUT_STRING(&Algo);
    OUTPUT_RAW("_INDEX_ROUTINE_INLINE(\n    ");

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
    // Inline functions.
    //

    OUTPUT_RAW("DECLARE_COMPILED_PERFECT_HASH_ROUTINES_INLINE(\n    ");
    OUTPUT_STRING(Name);

    OUTPUT_RAW("\n);\n#pragma warning(pop)\n\n");

    //
    // Close the extern "C" scope.
    //

    OUTPUT_RAW("\n\n"
               "#ifdef __cplusplus\n"
               "} // extern C\n"
               "#endif\n\n");

    EndOfFile.QuadPart = RtlPointerToOffset(File->BaseAddress, Output);

    Result = File->Vtbl->Close(File, &EndOfFile);
    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileClose, Result);
        goto Error;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_ERROR_SAVING_C_HEADER_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
