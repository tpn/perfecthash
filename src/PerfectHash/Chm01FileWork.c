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

PREPARE_FILE PrepareFileChm01;

_Use_decl_annotations_
HRESULT
PrepareFileChm01(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_FILE *FilePointer,
    PPERFECT_HASH_PATH Path,
    PLARGE_INTEGER EndOfFile,
    HANDLE DependentEvent
    )
/*++

Routine Description:

    Performs common file preparation work for a given file instance associated
    with a table.  If this is the first call to the function, indicated by a
    NULL value pointed to by the FilePointer argument, then a new file instance
    is created, and a Create() call is issued with the path and mapping size
    parameters.  Otherwise, if it is not NULL, a rename is scheduled for the
    new path name and the mapping size is extended (this involves unmapping the
    existing map, closing the mapping handle, extending the file by setting the
    file pointer and then end-of-file, and then creating a new mapping handle
    and re-mapping the address).

Arguments:

    Table - Supplies a pointer to the table owning the file to be prepared.

    FilePointer - Supplies the address of a variable that contains a pointer
        to the relevant PERFECT_HASH_FILE instance for this file within the
        PERFECT_HASH_TABLE structure.  If this value points to a NULL, it is
        assumed this is the first time the routine is being called.  Otherwise,
        it is assumed that a resize event has occurred and a new preparation
        request is being furnished.  In the case of the former, a new file
        instance is created and saved to the address specified by this param.

    Path - Supplies a pointer to the path to use for the file.  If the file
        has already been prepared at least once, this path is scheduled for
        rename.

    EndOfFile - Supplies a pointer to a LARGE_INTEGER that contains the
        desired file size.

    DependentEvent - Optionally supplies a handle to an event that must be
        signaled prior to this routine proceeding.  This is used, for example,
        to wait for the perfect hash table file to be created before creating
        the :Info stream that hangs off it.

Return Value:

    S_OK - File prepared successfully.  Otherwise, an appropriate error code.

--*/
{
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File = NULL;

    //
    // If a dependent event has been provided, wait for this object to become
    // signaled first before proceeding.
    //

    if (IsValidHandle(DependentEvent)) {
        ULONG WaitResult;

        WaitResult = WaitForSingleObject(DependentEvent, INFINITE);
        if (WaitResult != WAIT_OBJECT_0) {
            SYS_ERROR(WaitForSingleObject);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }
    }

    //
    // Dereference the file pointer provided by the caller.  If NULL, this
    // is the first preparation request for the given file instance.  Otherwise,
    // a table resize event has occurred, which means a file rename needs to be
    // scheduled (as we include the number of table elements in the file name),
    // and the mapping size needs to be extended (as a larger table size means
    // larger files are required to capture table data).
    //

    File = *FilePointer;

    if (!File) {

        //
        // File does not exist, so create a new instance, then issue a Create()
        // call with the desired path and mapping size parameters provided by
        // the caller.
        //

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
                                    EndOfFile,
                                    NULL);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileCreate, Result);
            File->Vtbl->Release(File);
            File = NULL;
            goto Error;
        }

        //
        // Update the table's pointer to this file instance.
        //

        *FilePointer = File;

    } else {

        //
        // File already exists.  Schedule a rename and then extend the file
        // according to the requested mapping size, assuming they differ.
        //

        Result = File->Vtbl->ScheduleRename(File, Path);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileScheduleRename, Result);
            goto Error;
        }

        if (File->FileInfo.EndOfFile.QuadPart < EndOfFile->QuadPart) {
            AcquirePerfectHashFileLockExclusive(File);
            Result = File->Vtbl->Extend(File, EndOfFile);
            ReleasePerfectHashFileLockExclusive(File);
            if (FAILED(Result)) {
                PH_ERROR(PerfectHashFileExtend, Result);
                goto Error;
            }
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

    return Result;
}

SAVE_FILE SaveFileChm01;

_Use_decl_annotations_
HRESULT
SaveFileChm01(
    PPERFECT_HASH_TABLE Table,
    PPERFECT_HASH_FILE File
    )
/*++

Routine Description:

    Performs common file save work for a given file instance associated with a
    table.  This routine is typically called for files that are not dependent
    upon table data (e.g. a C header file).  They are written in their entirety
    in the prepare callback, however, we don't Close() the file at that point,
    as it prevents our rundown logic kicking in whereby we delete the file if
    an error occurred.

    Each file preparation routine should update File->NumberOfBytesWritten with
    the number of bytes they wrote in order to ensure the file is successfully
    truncated to this size during Close().

Arguments:

    Table - Supplies a pointer to the table owning the file to be saved.

    File - Supplies a pointer to the file to save.

Return Value:

    S_OK - File saved successfully.  Otherwise, an appropriate error code.

--*/
{
    HRESULT Result;

    UNREFERENCED_PARAMETER(Table);

    Result = File->Vtbl->Close(File, NULL);
    if (FAILED(Result)) {
        PH_ERROR(SaveFileChm01_CloseFile, Result);
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
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_KEYS Keys;
    HANDLE DependentEvent = NULL;
    PPERFECT_HASH_PATH Path = NULL;
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
        LARGE_INTEGER EndOfFile;
        ULARGE_INTEGER NumTableElements;
        SYSTEM_INFO SystemInfo;
        PPERFECT_HASH_FILE *File;

        GetSystemInfo(&SystemInfo);
        EndOfFile.QuadPart = SystemInfo.dwAllocationGranularity;
        NumTableElements.QuadPart = TableInfo->NumberOfTableElements.QuadPart;

        switch (Item->FileWorkId) {

            case FileWorkPrepareTableFileId:
                File = &Table->TableFile;
                NewExtension = &TableExtension;
                EndOfFile.QuadPart = Info->AssignedSizeInBytes;
                break;

            case FileWorkPrepareTableInfoStreamId:
                File = &Table->InfoStream;

                if (*File) {

                    //
                    // :Info streams don't need more than one prepare call, as
                    // their path hangs off the main table path (and thus, will
                    // automatically inherit its scheduled rename), and their
                    // size never changes.
                    //

                    goto End;
                }

                NewExtension = &TableExtension;
                NewStreamName = &TableInfoStreamName;
                DependentEvent = Context->PreparedTableFileEvent;
                EndOfFile.QuadPart = sizeof(GRAPH_INFO_ON_DISK);
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
                EndOfFile.QuadPart += Keys->NumberOfElements.QuadPart * 16;
                Impl = PrepareCSourceKeysCallbackChm01;
                break;

            case FileWorkPrepareCSourceTableDataFileId:
                File = &Table->CSourceTableDataFile;
                NewExtension = &CSourceExtension;
                AdditionalSuffix = &CSourceTableDataSuffix;
                EndOfFile.QuadPart += NumTableElements.QuadPart * 16;
                break;

            default:
                Result = PH_E_INVALID_FILE_WORK_ID;
                PH_ERROR(FileWorkCallbackChm01, Result);
                goto End;
        }

        Result = PerfectHashTableCreatePath(Table,
                                            Table->Keys->File->Path,
                                            &NumTableElements,
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
                                  &EndOfFile,
                                  DependentEvent);

        if (FAILED(Result)) {
            PH_ERROR(PrepareFileChm01, Result);
            goto End;
        }

    } else {

        ULONG WaitResult;
        PPERFECT_HASH_FILE File = NULL;

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

            case FileWorkSaveCHeaderFileId:
                File = Table->CHeaderFile;
                DependentEvent = Context->PreparedCHeaderFileEvent;
                break;

            case FileWorkSaveCSourceFileId:
                File = Table->CSourceFile;
                DependentEvent = Context->PreparedCSourceFileEvent;
                break;

            case FileWorkSaveCSourceKeysFileId:
                File = Table->CSourceKeysFile;
                DependentEvent = Context->PreparedCSourceKeysFileEvent;
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

        if (!Impl) {
            ASSERT(File);
            Result = SaveFileChm01(Table, File);
            if (FAILED(Result)) {

                //
                // Nothing needs doing here.  The Result will bubble back up
                // via the normal mechanisms.
                //

                NOTHING;
            }
        } else {
            ASSERT(!File);
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
    ULONG LastError;
    PVOID BaseAddress;
    HRESULT Result = S_OK;
    LONGLONG SizeInBytes;
    LARGE_INTEGER EndOfFile;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_FILE File;
    BOOLEAN LargePagesForTableData;
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

    if (SizeInBytes != File->FileInfo.EndOfFile.QuadPart) {
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

    //
    // Allocate and copy the table data to an in-memory copy so that the table
    // can be used after Create() completes successfully.  See the comment in
    // the SaveTableInfoStreamCallbackChm01() routine for more information.
    //

    LargePagesForTableData = TRUE;

    BaseAddress = Rtl->Vtbl->TryLargePageVirtualAlloc(Rtl,
                                                      NULL,
                                                      SizeInBytes,
                                                      MEM_RESERVE | MEM_COMMIT,
                                                      PAGE_READWRITE,
                                                      &LargePagesForTableData);

    Table->TableDataBaseAddress = BaseAddress;

    if (!BaseAddress) {
        LastError = GetLastError();
        SYS_ERROR(VirtualAlloc);
        if (LastError == ERROR_OUTOFMEMORY) {
            Result = E_OUTOFMEMORY;
        } else {
            Result = PH_E_SYSTEM_CALL_FAILED;
        }
        goto Error;
    }

    //
    // Update flags with large page result for values array.
    //

    Table->Flags.TableDataUsesLargePages = LargePagesForTableData;

    //
    // Copy the table data over to the newly allocated buffer.
    //

    CopyMemory(Table->TableDataBaseAddress, Source, SizeInBytes);

    //
    // Proceed with closing the file.
    //

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
    PALLOCATOR Allocator;
    HRESULT Result = S_OK;
    LARGE_INTEGER EndOfFile;
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PGRAPH_INFO_ON_DISK GraphInfoOnDisk;
    PGRAPH_INFO_ON_DISK NewGraphInfoOnDisk;

    UNREFERENCED_PARAMETER(Item);

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = Table->InfoStream;
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
    // This next part is a bit hacky.  Originally, this library provided no
    // facility for obtaining a table after creation -- you would have to
    // explicitly create a table instance and call Load() on the desired path.
    // As this restriction has now been removed and tables can be interacted
    // with directly after their Create() method has been called, we need to
    // provide a way to make the on-disk table info available after the :Info
    // stream has been closed.  So, we simply do a heap-based alloc and memcpy
    // the structure over.  The table rundown routine knows to free this memory
    // if Table->TableInfoOnDisk is not NULL and Table->Flags.Created == TRUE.
    //

    Allocator = Table->Allocator;

    NewGraphInfoOnDisk = (PGRAPH_INFO_ON_DISK)(
        Allocator->Vtbl->Calloc(
            Allocator,
            1,
            sizeof(*GraphInfoOnDisk)
        )
    );

    if (!NewGraphInfoOnDisk) {
        SYS_ERROR(HeapAlloc);
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    CopyMemory(NewGraphInfoOnDisk,
               GraphInfoOnDisk,
               sizeof(*GraphInfoOnDisk));

    //
    // Switch the pointers.
    //

    Table->TableInfoOnDisk = &NewGraphInfoOnDisk->TableInfoOnDisk;

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
PrepareCHeaderCallbackChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    UNREFERENCED_PARAMETER(Context);
    UNREFERENCED_PARAMETER(Item);

    return S_OK;
    //return PH_E_NOT_IMPLEMENTED;
}

_Use_decl_annotations_
HRESULT
PrepareCSourceCallbackChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    UNREFERENCED_PARAMETER(Context);
    UNREFERENCED_PARAMETER(Item);

    return S_OK;
    //return PH_E_NOT_IMPLEMENTED;
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

    File->NumberOfBytesWritten.QuadPart = RtlPointerToOffset(Base, Output);

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
    Output = Base = (PCHAR)File->BaseAddress;

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
