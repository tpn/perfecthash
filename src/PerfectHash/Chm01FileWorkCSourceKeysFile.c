/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCSourceKeysFile.c

Abstract:

    This module implements the prepare file work callback routine for the C
    source keys file as part of the CHM v1 algorithm implementation for the
    perfect hash library.

    As the C source keys file is simply a C array representation of the keys
    array, and thus contains no references to solution-specific information
    (e.g. what seed values were used, what the index or hash masks were, etc),
    everything can be prepared in the initial prepare routine; no separate save
    file routine is required.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareCSourceKeysFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PCHAR Base;
    PCHAR Output;
    ULONG Count;
    PULONG Long;
    ULONG Key;
    PULONG SourceKeys;
    ULONGLONG Index;
    ULONGLONG NumberOfKeys;
    PCSTRING Name;
    HRESULT Result = S_OK;
    PPERFECT_HASH_KEYS Keys;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;
    const ULONG Indent = 0x20202020;

    UNREFERENCED_PARAMETER(Item);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    Keys = Table->Keys;
    File = Table->CSourceKeysFile;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    NumberOfKeys = Keys->NumberOfElements.QuadPart;
    SourceKeys = (PULONG)Keys->File->BaseAddress;

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the keys.
    //

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table Keys File.  "
               "Auto-generated.\n//\n\n"
               "#include <CompiledPerfectHash.h>\n\n");

    OUTPUT_RAW("#pragma const_seg(\".phkeys\")\n");
    OUTPUT_RAW("const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Keys[");
    OUTPUT_INT(NumberOfKeys);
    OUTPUT_RAW("] = {\n");

    for (Index = 0, Count = 0; Index < NumberOfKeys; Index++) {

        if (Count == 0) {
            INDENT();
        }

        Key = *SourceKeys++;

        OUTPUT_HEX(Key);

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

    OUTPUT_RAW("};\n#pragma const_seg()\n");

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
