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
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    Keys = Table->Keys;
    File = *Item->FilePointer;
    Path = GetActivePath(File);
    Name = &Path->TableNameA;
    NumberOfKeys = Keys->NumberOfElements.QuadPart;

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

    OUTPUT_RAW("#ifdef _WIN32\n#pragma const_seg(\".cphkeys\")\n#endif\n");

    OUTPUT_RAW("const CPHDKEY ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_NumberOfKeys = ");
    OUTPUT_INT(NumberOfKeys);
    OUTPUT_RAW(";\n");

    OUTPUT_RAW("const CPHKEY ");
    OUTPUT_STRING(Name);
    OUTPUT_RAW("_Keys[");
    OUTPUT_INT(NumberOfKeys);
    OUTPUT_RAW("] = {\n");

    if (Keys->OriginalKeySizeType == LongType) {

        ULONG Key;
        PULONG SourceKeys;

        SourceKeys = (PULONG)Keys->KeyArrayBaseAddress;

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

    } else if (Keys->OriginalKeySizeType == LongLongType) {

        ULONGLONG Key;
        PULONGLONG SourceKeys;

        SourceKeys = (PULONGLONG)Keys->File->BaseAddress;

        for (Index = 0, Count = 0; Index < NumberOfKeys; Index++) {

            if (Count == 0) {
                INDENT();
            }

            Key = *SourceKeys++;

            OUTPUT_HEX64(Key);

            *Output++ = ',';

            if (++Count == 4) {
                Count = 0;
                *Output++ = '\n';
            } else {
                *Output++ = ' ';
            }
        }

    } else {

        Result = PH_E_UNREACHABLE_CODE;
        PH_ERROR(PrepareCSourceKeysFileChm01_UnknownKeyType, Result);
        PH_RAISE(Result);

    }

    //
    // If the last character written was a trailing space, replace
    // it with a newline.
    //

    if (*(Output - 1) == ' ') {
        *(Output - 1) = '\n';
    }

    OUTPUT_RAW("};\n#ifdef _WIN32\n#pragma const_seg()\n#endif\n");

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
