/*++

Copyright (c) 2019 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkMakefileMainMkFile.c

Abstract:

    This module implements the prepare file work callback routine for a
    compiled perfect hash table's main.mk file.  This routine is similar to
    PrepareVSSolutionFileChm01() in that it has to wait for dependent sub-
    Makefile files to have their prepare event signaled before it can proceed.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
PrepareMakefileMainMkFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    BYTE Index;
    PCHAR Base;
    PCHAR Output;
    ULONG WaitResult;
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH SubMakefilePath;
    PPERFECT_HASH_FILE SubMakefileFile;
    PPERFECT_HASH_TABLE Table;
    HANDLE PrepareEvents[NUMBER_OF_SUBMAKEFILE_FILES];
    PHANDLE PrepareEvent = PrepareEvents;
    const BYTE NumberOfEvents = ARRAYSIZE(PrepareEvents);
    const BOOL WaitForAllEvents = TRUE;

    //
    // Initialize aliases.
    //

    Table = Context->Table;
    File = *Item->FilePointer;
    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Wire up the event array.  We need to wait on all of the sub-Makefiles'
    // prepare event before we can access their file and path instances.
    //

#define EXPAND_AS_ASSIGN_EVENT(Verb, VUpper, Name, Upper) \
    *##Verb##Event++ = Context->##Verb##d##Name##Event;

    PREPARE_SUBMAKEFILE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_ASSIGN_EVENT);

    //
    // Wait on all the events.
    //

    ASSERT(ARRAYSIZE(PrepareEvents) == (SIZE_T)NumberOfEvents);

    WaitResult = WaitForMultipleObjects(NumberOfEvents,
                                        PrepareEvents,
                                        WaitForAllEvents,
                                        INFINITE);

    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForMultipleObjects);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Write the header, CFLAGS and SUBMAKEFILES stub.
    //

    OUTPUT_RAW("# Compiled Perfect Hash Table Main Makefile.\n"
               "# Auto-generated.\n\n");

    OUTPUT_RAW("CFLAGS := -O3 -pipe "
                         "-Wall "
                         "-Wno-unknown-pragmas "
                         "-Wno-unused-but-set-variable"
                         "\n\n");

    OUTPUT_RAW("SUBMAKEFILES :=");

    //
    // Write all of the sub-Makefile include lines.  Note that we first check
    // if the file instance pointer is NULL; if it is, it means that file's
    // prepare event failed, in which case, we immediately exit this routine.
    //

#define EXPAND_AS_WRITE_SUBMAKEFILE(Verb, VUpper, Name, Upper) \
    SubMakefileFile = Table->##Name;                           \
    if (!SubMakefileFile) {                                    \
        goto Error;                                            \
    }                                                          \
    SubMakefilePath = GetActivePath(SubMakefileFile);          \
    OUTPUT_RAW(" \\\n\t");                                     \
    OUTPUT_UNICODE_STRING_FAST(&SubMakefilePath->FileName);

    Index = 0;
    SUBMAKEFILE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_WRITE_SUBMAKEFILE);

    //
    // Write the INCDIR and SRC_INCDIR variables.
    //

    OUTPUT_RAW("\n\n"
               "INCDIRS := ..\n"
               "SRC_INCDIRS := .\n");

    //
    // We're done; update number of bytes written and finish up.
    //

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

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
