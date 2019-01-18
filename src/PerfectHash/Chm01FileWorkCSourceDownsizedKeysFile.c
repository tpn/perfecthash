/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkCSourceDownsizedKeysFile.c

Abstract:

    This module implements the prepare file work callback routine for the C
    source keys file as part of the CHM v1 algorithm implementation for the
    perfect hash library.

    This module is identical to Chm01FileWorkCSourceKeysFile.c, except that it
    works against downsized keys.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareCSourceDownsizedKeysFileChm01(
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

    //
    // If the keys were not downsized, there's nothing to do.
    //

    Table = Context->Table;
    Keys = Table->Keys;

    if (!KeysWereDownsized(Keys)) {
        return S_OK;
    }

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    File = *Item->FilePointer;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    NumberOfKeys = Keys->NumberOfElements.QuadPart;
    SourceKeys = (PULONG)Keys->KeyArrayBaseAddress;

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the header.
    //

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table Keys File.  "
               "Auto-generated.\n//\n\n");

    OUTPUT_INCLUDE_STDAFX_H();

    //
    // Write the keys.
    //

    OUTPUT_RAW("#pragma const_seg(\".cphdkeys\")\n");

    OUTPUT_RAW("const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_NumberOfDownsizedKeys = ");
    OUTPUT_INT(NumberOfKeys);
    OUTPUT_RAW(";\n");

    OUTPUT_RAW("const ULONG ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_DownsizedKeys[");
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
