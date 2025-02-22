/*++

Copyright (c) 2022-2024 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkModuleDefFile.c

Abstract:

    This module implements the prepare file work callback routine for the
    <library>.def file as part of the CHM v1 algorithm implementation for
    the perfect hash library.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareModuleDefFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PCHAR Base;
    PCHAR Output;
    PCSTRING BaseName;
    BOOLEAN IncludeKeys;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;

    //
    // Initialize aliases.
    //

    File = *Item->FilePointer;
    Path = GetActivePath(File);
    BaseName = &Path->BaseNameA;
    Table = Context->Table;
    IncludeKeys = (Table->TableCreateFlags.IncludeKeysInCompiledDll != FALSE);

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the module definition file.
    //

    OUTPUT_RAW("EXPORTS\n");

    //
    // Raw exports of table-related symbols using their full name.
    //

#define WRITE_PH_EXPORT(Export) \
    OUTPUT_RAW("    ");         \
    OUTPUT_STRING(BaseName);    \
    OUTPUT_RAW("_");            \
    OUTPUT_RAW(#Export);        \
    OUTPUT_RAW("\n")

    WRITE_PH_EXPORT(TableData);
    WRITE_PH_EXPORT(TableValues);
    WRITE_PH_EXPORT(NumberOfTableValues);
    WRITE_PH_EXPORT(TableValueSizeInBytes);

    WRITE_PH_EXPORT(Seeds);
    WRITE_PH_EXPORT(NumberOfSeeds);

    if (IncludeKeys) {
        WRITE_PH_EXPORT(Keys);
        WRITE_PH_EXPORT(NumberOfKeys);
        WRITE_PH_EXPORT(KeySizeInBytes);
        WRITE_PH_EXPORT(OriginalKeySizeInBytes);
        WRITE_PH_EXPORT(DownsizedKeySizeInBytes);
    }

    //
    // Raw exports of function-related symbols using their full name.
    //

#define WRITE_CPH_EXPORT(Export)            \
    OUTPUT_RAW("    CompiledPerfectHash_"); \
    OUTPUT_STRING(BaseName);                \
    OUTPUT_RAW("_");                        \
    OUTPUT_RAW(#Export);                    \
    OUTPUT_RAW("\n");

    WRITE_CPH_EXPORT(Index);
    WRITE_CPH_EXPORT(IndexIaca);
    if (IncludeKeys) {
        WRITE_CPH_EXPORT(IndexBsearch);
    }
    WRITE_CPH_EXPORT(Insert);
    WRITE_CPH_EXPORT(Lookup);
    WRITE_CPH_EXPORT(Delete);
    WRITE_CPH_EXPORT(InterlockedIncrement);

    //
    // Aliases for the table-related exports.
    //

#define WRITE_PH_EXPORT_ALIAS(Export) \
    OUTPUT_RAW("    ");               \
    OUTPUT_RAW(#Export);              \
    OUTPUT_RAW(" = ");                \
    OUTPUT_STRING(BaseName);          \
    OUTPUT_RAW("_");                  \
    OUTPUT_RAW(#Export);              \
    OUTPUT_RAW("\n")

    WRITE_PH_EXPORT_ALIAS(TableData);
    WRITE_PH_EXPORT_ALIAS(TableValues);
    WRITE_PH_EXPORT_ALIAS(NumberOfTableValues);
    WRITE_PH_EXPORT_ALIAS(TableValueSizeInBytes);

    WRITE_PH_EXPORT_ALIAS(Seeds);
    WRITE_PH_EXPORT_ALIAS(NumberOfSeeds);

    if (IncludeKeys) {
        WRITE_PH_EXPORT_ALIAS(Keys);
        WRITE_PH_EXPORT_ALIAS(NumberOfKeys);
        WRITE_PH_EXPORT_ALIAS(KeySizeInBytes);
        WRITE_PH_EXPORT_ALIAS(OriginalKeySizeInBytes);
        WRITE_PH_EXPORT_ALIAS(DownsizedKeySizeInBytes);
    }

    //
    // Aliases for the function-related exports.
    //

#define WRITE_CPH_EXPORT_ALIAS(Export)     \
    OUTPUT_RAW("    ");                    \
    OUTPUT_RAW(#Export);                   \
    OUTPUT_RAW(" = CompiledPerfectHash_"); \
    OUTPUT_STRING(BaseName);               \
    OUTPUT_RAW("_");                       \
    OUTPUT_RAW(#Export);                   \
    OUTPUT_RAW("\n");

    WRITE_CPH_EXPORT_ALIAS(Index);
    WRITE_CPH_EXPORT_ALIAS(IndexIaca);
    if (IncludeKeys) {
        WRITE_CPH_EXPORT_ALIAS(IndexBsearch);
    }
    WRITE_CPH_EXPORT_ALIAS(Insert);
    WRITE_CPH_EXPORT_ALIAS(Lookup);
    WRITE_CPH_EXPORT_ALIAS(Delete);
    WRITE_CPH_EXPORT_ALIAS(InterlockedIncrement);

    //
    // Finish up.
    //

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return S_OK;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
