/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWork.c

Abstract:

    This module implements file work callback routines related to the CHM v1
    algorithm implementation for the perfect hash library.

    Prepare and save routines are implemented for table files, :Info streams,
    C header files, C source files, C source keys files, and C source table
    data files.

--*/

#include "stdafx.h"
#include "Chm_01.h"

HRESULT
PrepareFileChm01(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_FILE *FilePointer,
    PPERFECT_HASH_PATH Path,
    PULARGE_INTEGER MappingSize,
    HANDLE DependentEvent
    )
{
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File = NULL;

    if (IsValidHandle(DependentEvent)) {
        ULONG WaitResult;

        WaitResult = WaitForSingleObject(DependentEvent, INFINITE);
        if (WaitResult != WAIT_OBJECT_0) {
            SYS_ERROR(WaitForSingleObject);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }
    }

    File = *FilePointer;

    if (File) {

        Result = File->Vtbl->ScheduleRename(File, Path);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileScheduleRename, Result);
            goto Error;
        }

        Result = File->Vtbl->Extend(File, *MappingSize);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileExtend, Result);
            goto Error;
        }

    } else {

        Result = Table->Vtbl->CreateInstance(Table,
                                             NULL,
                                             &IID_PERFECT_HASH_FILE,
                                             &File);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileCreateInstance, Result);
            goto Error;
        }

        Result = File->Vtbl->Create(File,
                                    Path,
                                    MappingSize,
                                    NULL);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileCreate, Result);
            File->Vtbl->Release(File);
            File = NULL;
            goto Error;
        }

    }

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

    if (File) {
        *FilePointer = File;
    }

    return Result;
}

_Use_decl_annotations_
VOID
FileWorkCallbackChm01(
    PTP_CALLBACK_INSTANCE Instance,
    PPERFECT_HASH_CONTEXT Context,
    PSLIST_ENTRY ListEntry
    )
/*++

Routine Description:

    This routine is the callback entry point for file-oriented work we want
    to perform in the file work threadpool context.

Arguments:

    Instance - Supplies a pointer to the callback instance for this invocation.

    Context - Supplies a pointer to the active context for the graph solving.

    ListEntry - Supplies a pointer to the list entry that was popped off the
        context's file work interlocked singly-linked list head.

Return Value:

    None.

--*/
{
    PRTL Rtl;
    HRESULT Result = S_OK;
    PGRAPH_INFO Info;
    PFILE_WORK_ITEM Item;
    PFILE_WORK_CALLBACK_IMPL Impl = NULL;
    PPERFECT_HASH_PATH Path = NULL;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_KEYS Keys;
    HANDLE DependentEvent = NULL;
    PTABLE_INFO_ON_DISK TableInfo;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Info = (PGRAPH_INFO)Context->AlgorithmContext;
    Table = Context->Table;
    TableInfo = Table->TableInfoOnDisk;
    Keys = Table->Keys;

    //
    // Resolve the work item base address from the list entry.
    //

    Item = CONTAINING_RECORD(ListEntry, FILE_WORK_ITEM, ListEntry);

    ASSERT(IsValidFileWorkId(Item->FileWorkId));

    if (IsPrepareFileWorkId(Item->FileWorkId)) {

        PCUNICODE_STRING NewBaseName = NULL;
        PCUNICODE_STRING NewExtension = NULL;
        PCUNICODE_STRING NewStreamName = NULL;
        PCUNICODE_STRING AdditionalSuffix = NULL;
        ULARGE_INTEGER MappingSize;
        SYSTEM_INFO SystemInfo;
        PPERFECT_HASH_FILE *File;

        GetSystemInfo(&SystemInfo);
        MappingSize.QuadPart = SystemInfo.dwAllocationGranularity;

        switch (Item->FileWorkId) {

            case FileWorkPrepareTableFileId:
                File = &Table->TableFile;
                NewExtension = &TableExtension;
                MappingSize.QuadPart = Info->AssignedSizeInBytes;
                break;

            case FileWorkPrepareTableInfoStreamId:
                File = &Table->InfoStream;
                NewExtension = &TableExtension;
                NewStreamName = &TableInfoStreamName;
                DependentEvent = Context->PreparedTableFileEvent;
                MappingSize.QuadPart = sizeof(GRAPH_INFO_ON_DISK);
                break;

            case FileWorkPrepareCHeaderFileId:
                File = &Table->CHeaderFile;
                NewExtension = &CHeaderExtension;
                Impl = PrepareCHeaderCallbackChm01;
                break;

            case FileWorkPrepareCSourceFileId:
                File = &Table->CSourceFile;
                NewExtension = &CSourceExtension;
                Impl = PrepareCSourceCallbackChm01;
                break;

            case FileWorkPrepareCSourceKeysFileId:
                File = &Table->CSourceKeysFile;
                NewExtension = &CSourceExtension;
                AdditionalSuffix = &CSourceKeysSuffix;
                MappingSize.QuadPart += Keys->NumberOfElements.QuadPart * 16;
                Impl = PrepareCSourceKeysCallbackChm01;
                break;

            case FileWorkPrepareCSourceTableDataFileId:
                File = &Table->CSourceTableDataFile;
                NewExtension = &CSourceExtension;
                AdditionalSuffix = &CSourceTableDataSuffix;
                MappingSize.QuadPart += (
                    TableInfo->NumberOfTableElements.QuadPart * 16
                );
                break;

            default:
                Result = PH_E_INVALID_FILE_WORK_ID;
                goto End;
        }

        Result = PerfectHashTableCreatePath(Table,
                                            Table->Keys->File->Path,
                                            &TableInfo->NumberOfTableElements,
                                            Table->AlgorithmId,
                                            Table->MaskFunctionId,
                                            Table->HashFunctionId,
                                            Table->OutputDirectory,
                                            NewBaseName,
                                            AdditionalSuffix,
                                            NewExtension,
                                            NewStreamName,
                                            &Path,
                                            NULL);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashTableCreatePath, Result);
            goto End;
        }

        Result = PrepareFileChm01(Table,
                                  File,
                                  Path,
                                  &MappingSize,
                                  DependentEvent);

        if (FAILED(Result)) {
            PH_ERROR(PrepareFileChm01, Result);
            goto End;
        }

    } else {

        ULONG WaitResult;

        if (!IsSaveFileWorkId(Item->FileWorkId)) {
            ASSERT(FALSE);
            Result = PH_E_INVARIANT_CHECK_FAILED;
            goto End;
        }

        switch (Item->FileWorkId) {

            case FileWorkSaveTableFileId:
                Impl = SaveTableCallbackChm01;
                DependentEvent = Context->PreparedTableFileEvent;
                break;

            case FileWorkSaveTableInfoStreamId:
                Impl = SaveTableInfoStreamCallbackChm01;
                DependentEvent = Context->PreparedTableInfoStreamEvent;
                break;

            case FileWorkSaveCSourceTableDataFileId:
                Impl = SaveCSourceTableDataCallbackChm01;
                DependentEvent = Context->PreparedCSourceTableDataFileEvent;
                break;

            default:
                break;
        }

        if (DependentEvent) {
            WaitResult = WaitForSingleObject(DependentEvent, INFINITE);
            if (WaitResult != WAIT_OBJECT_0) {
                SYS_ERROR(WaitForSingleObject);
                Result = PH_E_SYSTEM_CALL_FAILED;
                goto End;
            }
        }
    }

    if (Impl) {
        Result = Impl(Context, Item);
    }

    //
    // Intentional follow-on to End.
    //

End:

    if (Path) {
        Path->Vtbl->Release(Path);
        Path = NULL;
    }

    Item->LastResult = Result;

    if (FAILED(Result)) {
        InterlockedIncrement(&Item->NumberOfErrors);
        Item->LastError = GetLastError();
    }

    //
    // Register the relevant event to be set when this threadpool callback
    // returns, then return.
    //

    SetEventWhenCallbackReturns(Instance, Item->Event);

    return;
}

_Use_decl_annotations_
HRESULT
SaveTableCallbackChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PULONG Dest;
    PGRAPH Graph;
    PULONG Source;
    HRESULT Result = S_OK;
    ULONGLONG SizeInBytes;
    LARGE_INTEGER EndOfFile;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_FILE File;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;

    UNREFERENCED_PARAMETER(Item);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = Table->TableFile;
    Dest = (PULONG)File->BaseAddress;
    Graph = (PGRAPH)Context->SolvedContext;
    Source = Graph->Assigned;
    TableInfoOnDisk = Table->TableInfoOnDisk;

    SizeInBytes = (
        TableInfoOnDisk->NumberOfTableElements.QuadPart *
        TableInfoOnDisk->KeySizeInBytes
    );

    if (SizeInBytes != File->MappingSize.QuadPart) {
        ASSERT(FALSE);
        Result = PH_E_INVARIANT_CHECK_FAILED;
        goto Error;
    }

    //
    // The graph has been solved.  Copy the array of assigned values to the
    // backing memory map.
    //

    CopyMemory(Dest, Source, SizeInBytes);

    EndOfFile.QuadPart = (LONGLONG)SizeInBytes;

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
        Result = PH_E_ERROR_SAVING_TABLE_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

_Use_decl_annotations_
HRESULT
SaveTableInfoStreamCallbackChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PULONG Dest;
    PGRAPH Graph;
    ULONG WaitResult;
    HRESULT Result = S_OK;
    LARGE_INTEGER EndOfFile;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;
    PGRAPH_INFO_ON_DISK GraphInfoOnDisk;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;

    UNREFERENCED_PARAMETER(Item);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = Table->TableFile;
    Dest = (PULONG)File->BaseAddress;
    Graph = (PGRAPH)Context->SolvedContext;
    GraphInfoOnDisk = (PGRAPH_INFO_ON_DISK)File->BaseAddress;
    TableInfoOnDisk = &GraphInfoOnDisk->TableInfoOnDisk;

    //
    // Copy the in-memory representation of the on-disk structure to the memory
    // map of the backing :Info stream (that is actually on-disk).
    //

    CopyMemory(GraphInfoOnDisk,
               Table->TableInfoOnDisk,
               sizeof(*GraphInfoOnDisk));

    //
    // Save the seed values used by this graph.  (Everything else in the on-disk
    // info representation was saved earlier.)
    //

    TableInfoOnDisk->Seed1 = Graph->Seed1;
    TableInfoOnDisk->Seed2 = Graph->Seed2;
    TableInfoOnDisk->Seed3 = Graph->Seed3;
    TableInfoOnDisk->Seed4 = Graph->Seed4;

    //
    // Wait on the verification complete event.  This is done in the
    // main thread straight after it dispatches our file work callback
    // (that ended up here).  We need to block on this event as we want
    // to save the timings for verification to the header.
    //

    WaitResult = WaitForSingleObject(Context->VerifiedTableEvent, INFINITE);
    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForSingleObject);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Save the number of attempts and number of finished solutions.
    //

    TableInfoOnDisk->NumberOfAttempts = Context->Attempts;
    TableInfoOnDisk->NumberOfFailedAttempts = Context->FailedAttempts;
    TableInfoOnDisk->NumberOfSolutionsFound = Context->FinishedCount;

    //
    // Copy timer values.
    //

    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(Solve);
    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(Verify);

    //CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(PrepareTableFile);
    //CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(SaveTableFile);

    //
    // Update the number of bytes written and close the file.
    //

    EndOfFile.QuadPart = sizeof(*GraphInfoOnDisk);

    Result = File->Vtbl->Close(File, &EndOfFile);

    if (FAILED(Result)) {
        PH_ERROR(PerfectHashFileClose, Result);
        goto Error;
    }

    //
    // We're done, jump to the end.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_ERROR_SAVING_TABLE_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

#define INDENT() {            \
    Long = (PULONG)Output;    \
    *Long = Indent;           \
    Output += sizeof(Indent); \
}

_Use_decl_annotations_
HRESULT
PrepareCSourceKeysCallbackChm01(
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
    LARGE_INTEGER EndOfFile;
    const ULONG Indent = 0x20202020;

    UNREFERENCED_PARAMETER(Item);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    Keys = Table->Keys;
    File = Table->CSourceKeysFile;
    Path = File->Path;
    Name = &Path->BaseNameA;
    NumberOfKeys = Keys->NumberOfElements.QuadPart;
    SourceKeys = (PULONG)Keys->File->BaseAddress;

    Base = (PCHAR)File->BaseAddress;
    Output = Base;

    //
    // Write the keys.
    //

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table.  Auto-generated.\n//\n\n");

    OUTPUT_RAW("#ifdef COMPILED_PERFECT_HASH_TABLE_INCLUDE_KEYS\n");
    OUTPUT_RAW("#pragma const_seg(\".cpht_keys\")\n");
    OUTPUT_RAW("const unsigned long TableKeys[");
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

    OUTPUT_RAW("};\n#pragma const_seg()\n"
               "#endif "
               "/* COMPILED_PERFECT_HASH_TABLE_INCLUDE_KEYS */\n");

    EndOfFile.QuadPart = ((ULONG_PTR)Output - (ULONG_PTR)Base);

    Result = File->Vtbl->Close(File, &EndOfFile);
    if (FAILED(Result)) {
        PH_ERROR(PrepareCSourceKeysCallbackChm01, Result);
        goto Error;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_ERROR_PREPARING_C_HEADER_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

_Use_decl_annotations_
HRESULT
SaveCSourceTableDataCallbackChm01(
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
    PCSTRING Name;
    ULONGLONG Index;
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_PATH Path;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfo;
    ULONGLONG NumberOfElements;
    ULONGLONG TotalNumberOfElements;
    LARGE_INTEGER EndOfFile;
    const ULONG Indent = 0x20202020;

    UNREFERENCED_PARAMETER(Item);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = Table->CSourceTableDataFile;
    Path = File->Path;
    Name = &Table->TableFile->Path->BaseNameA;
    TableInfo = Table->TableInfoOnDisk;
    TotalNumberOfElements = TableInfo->NumberOfTableElements.QuadPart;
    NumberOfElements = TotalNumberOfElements >> 1;
    Graph = (PGRAPH)Context->SolvedContext;
    Source = Graph->Assigned;
    Base = (PCHAR)File->BaseAddress;

    //
    // Write seed data.
    //

    OUTPUT_RAW("#pragma const_seg(\".cpht_seeds\")\n");
    OUTPUT_RAW("const unsigned long Seeds[");
    OUTPUT_INT(TableInfo->NumberOfSeeds);
    OUTPUT_RAW("] = {\n");

    Seed = &TableInfo->FirstSeed;

    for (Index = 0, Count = 0; Index < TableInfo->NumberOfSeeds; Index++) {

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

    ASSERT(TableInfo->NumberOfSeeds == 4);

    Seed = &TableInfo->FirstSeed;

    OUTPUT_RAW("};\nstatic const unsigned long Seed1 = ");
    OUTPUT_HEX(*Seed++);
    OUTPUT_RAW(";\nstatic const unsigned long Seed2 = ");
    OUTPUT_HEX(*Seed++);
    OUTPUT_RAW(";\nstatic const unsigned long Seed3 = ");
    OUTPUT_HEX(*Seed++);
    OUTPUT_RAW(";\nstatic const unsigned long Seed4 = ");
    OUTPUT_HEX(*Seed++);

    Seed = &TableInfo->FirstSeed;

    OUTPUT_RAW(";\n#define SEED1 ");
    OUTPUT_HEX(*Seed++);
    OUTPUT_RAW("\n#define SEED2 ");
    OUTPUT_HEX(*Seed++);
    OUTPUT_RAW("\n#define SEED3 ");
    OUTPUT_HEX(*Seed++);
    OUTPUT_RAW("\n#define SEED4 ");
    OUTPUT_HEX(*Seed++);

    OUTPUT_RAW("\n#pragma const_seg()\n\n");

    //
    // Write the table data.
    //

    OUTPUT_RAW("\n\n#pragma const_seg(\".cpht_data\")\n");
    OUTPUT_RAW("static const unsigned long HashMask = ");
    OUTPUT_HEX(TableInfo->HashMask);
    OUTPUT_RAW(";\nstatic const unsigned long IndexMask = ");
    OUTPUT_HEX(TableInfo->IndexMask);
    OUTPUT_RAW(";\n#define HASH_MASK ");
    OUTPUT_HEX(TableInfo->HashMask);
    OUTPUT_RAW("\n#define INDEX_MASK ");
    OUTPUT_HEX(TableInfo->IndexMask);

    OUTPUT_RAW("\nstatic const unsigned long TableData[");
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

    OUTPUT_RAW("};\n#define TABLE_DATA TableData\n#pragma const_seg()\n\n");

    EndOfFile.QuadPart = ((ULONG_PTR)Output - (ULONG_PTR)Base);

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
        Result = PH_E_ERROR_SAVING_C_SOURCE_TABLE_DATA_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
