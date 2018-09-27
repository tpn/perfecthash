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
    HANDLE Event;
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;

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

        Result = File->Vtbl->Extend(File, MappingSize->QuadPart);
        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileExtend, Result);
            goto Error;
        }

    } else {

        Result = Table->Vtbl->CreateInstance(Table,
                                             NULL,
                                             &IID_PERFECT_HASH_FILE,
                                             File);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileCreateInstance, Result);
            goto Error;
        }

        Result = File->Vtbl->Create(File,
                                    Path,
                                    &MappingSize,
                                    NULL);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileCreate, Result);
            File->Vtbl->Release(File);
            File = NULL;
            goto Error;
        }

    }

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
    PHANDLE Event;
    HRESULT Result;
    PGRAPH_INFO Info;
    PFILE_WORK_ITEM Item;
    PERFECT_HASH_TLS_CONTEXT TlsContext;
    PFILE_WORK_CALLBACK_IMPL Impl;
    PFILE_WORK_CALLBACK_WRAPPER Wrapper = NULL;
    PPERFECT_HASH_PATH Path = NULL;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_KEYS Keys;
    HANDLE DependentEvent = NULL;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Info = (PGRAPH_INFO)Context->AlgorithmContext;
    Table = Context->Table;
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
        ULARGE_INTEGER NumberOfTableElements;
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
                Impl = PrepareTableInfoStreamCallbackChm01;
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
                goto Error;
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
    BOOL Success;
    PULONG Dest;
    PGRAPH Graph;
    PULONG Source;
    ULONG WaitResult;
    HRESULT Result = S_OK;
    ULONGLONG SizeInBytes;
    LARGE_INTEGER EndOfFile;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_FILE File;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;

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

    //
    // Save the seed values used by this graph.  (Everything else in
    // the on-disk info representation was saved earlier.)
    //

    TableInfoOnDisk->Seed1 = Graph->Seed1;
    TableInfoOnDisk->Seed2 = Graph->Seed2;
    TableInfoOnDisk->Seed3 = Graph->Seed3;
    TableInfoOnDisk->Seed4 = Graph->Seed4;

    //
    // Kick off a flush file buffers now before we wait on the verified
    // event.  The flush will be a blocking call.  The wait on verified
    // will be blocking if the event isn't signaled.  So, we may as well
    // get some useful blocking work done, before potentially going into
    // another wait state where we're not doing anything useful.
    //

    if (!FlushFileBuffers(Table->FileHandle)) {
        SYS_ERROR(FlushFileBuffers);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Stop the save file timer here, after flushing the file buffers,
    // but before we potentially wait on the verified state.
    //

    CONTEXT_END_TIMERS(SaveTableFile);

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
    // When we mapped the array in the work item above, we used a size
    // that was aligned with the system allocation granularity.  We now
    // want to set the end of file explicitly to the exact size of the
    // underlying array.  To do this, we unmap the view, delete the
    // section, set the file pointer to where we want, set the end of
    // file (which will apply the file pointer position as EOF), then
    // close the file handle.
    //

    if (!UnmapViewOfFile(Table->BaseAddress)) {
        SYS_ERROR(UnmapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    Table->BaseAddress = NULL;

    if (!CloseHandle(Table->MappingHandle)) {
        SYS_ERROR(UnmapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    Table->MappingHandle = NULL;

    EndOfFile.QuadPart = SizeInBytes;

    Success = SetFilePointerEx(Table->FileHandle,
                               EndOfFile,
                               NULL,
                               FILE_BEGIN);

    if (!Success) {
        SYS_ERROR(SetFilePointerEx);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!SetEndOfFile(Table->FileHandle)) {
        SYS_ERROR(SetEndOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!CloseHandle(Table->FileHandle)) {
        SYS_ERROR(CloseHandle);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Table->FileHandle = NULL;

    //
    // Save the number of attempts and number of finished solutions.
    //

    TableInfoOnDisk->NumberOfAttempts = Context->Attempts;
    TableInfoOnDisk->NumberOfFailedAttempts = Context->FailedAttempts;
    TableInfoOnDisk->NumberOfSolutionsFound = Context->FinishedCount;

    //
    // Copy timer values for everything except the save header event.
    //

    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(Solve);
    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(Verify);
    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(PrepareTableFile);
    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(SaveTableFile);

    //
    // We need to wait on the header saved event before we can capture
    // the header timers.
    //

    WaitResult = WaitForSingleObject(Context->SavedHeaderFileEvent,
                                     INFINITE);

    if (WaitResult != WAIT_OBJECT_0) {
        SYS_ERROR(WaitForSingleObject);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(PrepareHeaderFile);
    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(SaveHeaderFile);

    if (!FlushFileBuffers(Table->InfoStreamFileHandle)) {
        SYS_ERROR(FlushFileBuffers);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Finalize the :Info stream the same way we handled the backing
    // file above; unmap, delete section, set file pointer, set eof,
    // close file.
    //

    if (!UnmapViewOfFile(Table->InfoStreamBaseAddress)) {
        SYS_ERROR(UnmapViewOfFile);
        goto Error;
    }
    Table->InfoStreamBaseAddress = NULL;

    if (!CloseHandle(Table->InfoStreamMappingHandle)) {
        SYS_ERROR(CloseHandle);
        goto Error;
    }
    Table->InfoStreamMappingHandle = NULL;

    //
    // The file size for the :Info stream will be the size of our
    // on-disk graph info structure.
    //

    EndOfFile.QuadPart = sizeof(GRAPH_INFO_ON_DISK);

    Success = SetFilePointerEx(Table->InfoStreamFileHandle,
                               EndOfFile,
                               NULL,
                               FILE_BEGIN);

    if (!Success) {
        SYS_ERROR(SetFilePointerEx);
        goto Error;
    }

    if (!SetEndOfFile(Table->InfoStreamFileHandle)) {
        SYS_ERROR(SetEndOfFile);
        goto Error;
    }

    if (!CloseHandle(Table->InfoStreamFileHandle)) {
        SYS_ERROR(CloseHandle);
        goto Error;
    }

    Table->InfoStreamFileHandle = NULL;

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

_Use_decl_annotations_
HRESULT
PrepareHeaderFileChm01(
    PPERFECT_HASH_CONTEXT Context
    )
/*++

Routine Description:

    Creates the underlying C header file, extends it to an appropriate size,
    and maps it into memory.

Arguments:

    Context - Supplies the active context for which the header file is to be
        prepared.

Return Value:

    S_OK - Header file prepared successfully.

    PH_E_SYSTEM_CALL_FAILED - A system call failed.

    PH_E_ERROR_PREPARING_HEADER_FILE - Encountered an error during preparation.

    PH_E_OVERFLOWED_HEADER_FILE_MAPPING_SIZE - The maximum calculated size for
        the header exceeded 4GB, and we're a 32-bit executable.

--*/
{
    PRTL Rtl;
    BOOL Success;
    ULONG Status;
    ULONG LastError;
    ULONG ShareMode;
    ULONG DesiredAccess;
    SIZE_T ViewSize;
    HANDLE FileHandle;
    HANDLE MappingHandle;
    ULONG FlagsAndAttributes;
    ULONG AllocationGranularity;
    SYSTEM_INFO SystemInfo;
    HRESULT Result = S_OK;
    PVOID BaseAddress;
    LARGE_INTEGER MappingSize;
    PPERFECT_HASH_TABLE Table;
    ULONGLONG NumberOfKeys;
    ULONGLONG MaxTotalNumberOfEdges;
    ULONGLONG MaxTotalNumberOfVertices;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    NumberOfKeys = Table->Keys->NumberOfElements.QuadPart;

    //
    // Create the file.
    //

    ShareMode = (
        FILE_SHARE_READ  |
        FILE_SHARE_WRITE |
        FILE_SHARE_DELETE
    );

    DesiredAccess = (
        GENERIC_READ |
        GENERIC_WRITE
    );

    FlagsAndAttributes = FILE_FLAG_OVERLAPPED;

    FileHandle = CreateFileW(Table->HeaderPath.Buffer,
                             DesiredAccess,
                             ShareMode,
                             NULL,
                             OPEN_ALWAYS,
                             FlagsAndAttributes,
                             NULL);

    LastError = GetLastError();

    Table->HeaderFileHandle = FileHandle;

    if (!FileHandle || FileHandle == INVALID_HANDLE_VALUE) {

        //
        // Failed to open the file successfully.
        //

        SYS_ERROR(CreateFileW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;

    } else if (LastError == ERROR_ALREADY_EXISTS) {

        //
        // The file was opened successfully, but it already existed.  Clear the
        // local last error variable then truncate the file.
        //

        LastError = ERROR_SUCCESS;

        Status = SetFilePointer(FileHandle, 0, NULL, FILE_BEGIN);
        if (Status == INVALID_SET_FILE_POINTER) {
            SYS_ERROR(SetFilePointer);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }

        Success = SetEndOfFile(FileHandle);
        if (!Success) {
            SYS_ERROR(SetEndOfFile);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }

        //
        // We've successfully truncated the file.
        //

    }

    //
    // Calculate (a very generous) size to use for the file.  As we map the
    // entire file up-front and then simply write to the memory map without
    // closely tracking bytes written, we want to ensure the allocated space
    // is bigger than the largest possible size we'll consume.  When we save
    // the file, we update the end-of-file accordingly.
    //

    GetSystemInfo(&SystemInfo);
    AllocationGranularity = SystemInfo.dwAllocationGranularity;

    MaxTotalNumberOfEdges = (
        ((ULONGLONG)RoundUpPowerOf2((ULONG)NumberOfKeys) << 1ULL) <<
        GRAPH_SOLVING_RESIZE_TABLE_LIMIT
    );

    MaxTotalNumberOfVertices = (
        ((ULONGLONG)RoundUpNextPowerOf2((ULONG)NumberOfKeys)) <<
        GRAPH_SOLVING_RESIZE_TABLE_LIMIT
    );

    MappingSize.QuadPart = NumberOfKeys * 16;

    MappingSize.QuadPart += (MaxTotalNumberOfEdges * 16);

    MappingSize.QuadPart += (MaxTotalNumberOfVertices * 16);

    //
    // Add in an additional 64KB for all other text/code, then align up to a
    // 64KB boundary.
    //

    MappingSize.QuadPart += AllocationGranularity;

    MappingSize.QuadPart = ALIGN_UP(MappingSize.QuadPart,
                                    AllocationGranularity);

#ifdef _WIN64
    ViewSize = MappingSize.QuadPart;
#else

    //
    // Verify we haven't overflowed MAX_ULONG.
    //

    if (MappingSize.HighPart) {
        Result = PH_E_OVERFLOWED_HEADER_FILE_MAPPING_SIZE;
        PH_ERROR(PrepareHeaderFileChm01, Result);
        goto Error;
    }

    ViewSize = MappingSize.LowPart;
#endif

    //
    // Extend the file to the mapping size.
    //

    Success = SetFilePointerEx(FileHandle,
                               MappingSize,
                               NULL,
                               FILE_BEGIN);

    if (!Success) {
        SYS_ERROR(SetFilePointerEx);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!SetEndOfFile(FileHandle)) {
        SYS_ERROR(SetEndOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // Create a file mapping.
    //

    MappingHandle = CreateFileMappingW(FileHandle,
                                       NULL,
                                       PAGE_READWRITE,
                                       MappingSize.HighPart,
                                       MappingSize.LowPart,
                                       NULL);

    Table->HeaderMappingHandle = MappingHandle;
    Table->HeaderMappingSizeInBytes.QuadPart = MappingSize.QuadPart;

    if (!MappingHandle || MappingHandle == INVALID_HANDLE_VALUE) {
        SYS_ERROR(CreateFileMappingW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // We successfully created a file mapping.  Proceed with mapping it into
    // memory.
    //

    BaseAddress = MapViewOfFile(MappingHandle,
                                FILE_MAP_READ | FILE_MAP_WRITE,
                                0,
                                0,
                                ViewSize);

    Table->HeaderBaseAddress = BaseAddress;

    if (!BaseAddress) {
        SYS_ERROR(MapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_ERROR_PREPARING_TABLE_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

HRESULT
PerfectHashTablePrepareInfoFile(
    PPERFECT_HASH_CONTEXT Context
    )
{
    PRTL Rtl;
    PPERFECT_HASH_TABLE Table;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;

    ShareMode = (
        FILE_SHARE_READ  |
        FILE_SHARE_WRITE |
        FILE_SHARE_DELETE
    );

    DesiredAccess = (
        GENERIC_READ |
        GENERIC_WRITE
    );

    FlagsAndAttributes = FILE_FLAG_OVERLAPPED;

    FileHandle = CreateFileW(Table->InfoStreamPath.Buffer,
                             DesiredAccess,
                             ShareMode,
                             NULL,
                             OPEN_ALWAYS,
                             FlagsAndAttributes,
                             NULL);

    Table->InfoStreamFileHandle = FileHandle;

    LastError = GetLastError();

    if (!FileHandle || FileHandle == INVALID_HANDLE_VALUE) {

        //
        // Failed to open the file successfully.
        //

        SYS_ERROR(CreateFileW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;

    } else if (LastError == ERROR_ALREADY_EXISTS) {

        //
        // The file was opened successfully, but it already existed.  Clear the
        // local last error variable then truncate the file.
        //

        LastError = ERROR_SUCCESS;

        Status = SetFilePointer(FileHandle, 0, NULL, FILE_BEGIN);
        if (Status == INVALID_SET_FILE_POINTER) {
            SYS_ERROR(SetFilePointer);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }

        Success = SetEndOfFile(FileHandle);
        if (!Success) {
            SYS_ERROR(SetEndOfFile);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto Error;
        }

        //
        // We've successfully truncated the :Info file.
        //

    }

    //
    // Get the system allocation granularity, as we use this to govern the size
    // we request of the underlying file mapping.
    //

    GetSystemInfo(&SystemInfo);

    InfoMappingSize = SystemInfo.dwAllocationGranularity;
    ASSERT(InfoMappingSize >= PAGE_SIZE);

    //
    // Create a file mapping for the :Info stream.
    //

    MappingHandle = CreateFileMappingW(FileHandle,
                                       NULL,
                                       PAGE_READWRITE,
                                       0,
                                       InfoMappingSize,
                                       NULL);

    Table->InfoStreamMappingHandle = MappingHandle;
    Table->InfoMappingSizeInBytes.QuadPart = InfoMappingSize;

    if (!MappingHandle || MappingHandle == INVALID_HANDLE_VALUE) {
        SYS_ERROR(CreateFileMappingW);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // We successfully created a file mapping.  Proceed with mapping it into
    // memory.
    //

    BaseAddress = MapViewOfFile(MappingHandle,
                                FILE_MAP_READ | FILE_MAP_WRITE,
                                0,
                                0,
                                InfoMappingSize);

    Table->InfoStreamBaseAddress = BaseAddress;

    if (!BaseAddress) {
        SYS_ERROR(MapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_ERROR_PREPARING_TABLE_INFO_STREAM;
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
PrepareHeaderCallbackChm01(
    PPERFECT_HASH_CONTEXT Context
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
    PPERFECT_HASH_TABLE Table;
    const ULONG Indent = 0x20202020;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    Keys = Table->Keys;
    Name = &Table->TableNameA;
    NumberOfKeys = Keys->NumberOfElements.QuadPart;
    SourceKeys = (PULONG)Keys->File->BaseAddress;

    //
    // Prepare the underlying file and memory maps.
    //

    Result = PrepareHeaderFileChm01(Context);
    if (FAILED(Result)) {
        PH_ERROR(PrepareHeaderCallbackChm01, Result);
        goto Error;
    }

    Base = (PCHAR)Table->HeaderBaseAddress;
    Output = Base;

    //
    // Write the keys.
    //

    OUTPUT_RAW("//\n// Compiled Perfect Hash Table.  Auto-generated.\n//\n\n");

    OUTPUT_RAW("#ifdef COMPILED_PERFECT_HASH_TABLE_INCLUDE_KEYS\n");
    OUTPUT_RAW("#pragma const_seg(\".cpht_keys\")\n");
    OUTPUT_RAW("static const unsigned long TableKeys[");
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

    Table->HeaderSizeInBytes = ((ULONG_PTR)Output - (ULONG_PTR)Base);

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_ERROR_PREPARING_HEADER_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    CONTEXT_END_TIMERS(PrepareHeaderFile);

    return Result;
}

_Use_decl_annotations_
HRESULT
SaveHeaderCallbackChm01(
    PPERFECT_HASH_CONTEXT Context
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
    TableInfo = Table->TableInfoOnDisk;
    Name = &Table->TableNameA;
    TotalNumberOfElements = TableInfo->NumberOfTableElements.QuadPart;
    NumberOfElements = TotalNumberOfElements >> 1;
    Graph = (PGRAPH)Context->SolvedContext;
    Source = Graph->Assigned;

    //
    // Write the table data.
    //

    Base = (PCHAR)Table->HeaderBaseAddress;
    Output = RtlOffsetToPointer(Base, Table->HeaderSizeInBytes);

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

    //
    // Write seed data.
    //

    OUTPUT_RAW("#pragma const_seg(\".cpht_seeds\")\n");
    OUTPUT_RAW("static const unsigned long Seeds[");
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
    // Update the header size, then save it.
    //

    Table->HeaderSizeInBytes = ((ULONG_PTR)Output - (ULONG_PTR)Base);

    Result = SaveHeaderFileChm01(Context);
    if (FAILED(Result)) {
        PH_ERROR(SaveHeaderCallbackChm01, Result);
        goto Error;
    }

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_ERROR_SAVING_HEADER_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    CONTEXT_END_TIMERS(SaveHeaderFile);

    return Result;
}

_Use_decl_annotations_
HRESULT
SaveHeaderFileChm01(
    PPERFECT_HASH_CONTEXT Context
    )
{
    PRTL Rtl;
    BOOL Success;
    HRESULT Result = S_OK;
    LARGE_INTEGER EndOfFile;
    PPERFECT_HASH_TABLE Table;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;

    //
    // Save the header file: flush file buffers, unmap the view, close the
    // mapping handle, set the file pointer, set EOF, and then close the handle.
    //

    if (!FlushFileBuffers(Table->HeaderFileHandle)) {
        SYS_ERROR(FlushFileBuffers);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!UnmapViewOfFile(Table->HeaderBaseAddress)) {
        SYS_ERROR(UnmapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    Table->HeaderBaseAddress = NULL;

    if (!CloseHandle(Table->HeaderMappingHandle)) {
        SYS_ERROR(UnmapViewOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }
    Table->HeaderMappingHandle = NULL;

    EndOfFile.QuadPart = Table->HeaderSizeInBytes;

    Success = SetFilePointerEx(Table->HeaderFileHandle,
                               EndOfFile,
                               NULL,
                               FILE_BEGIN);

    if (!Success) {
        SYS_ERROR(SetFilePointerEx);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!SetEndOfFile(Table->HeaderFileHandle)) {
        SYS_ERROR(SetEndOfFile);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    if (!CloseHandle(Table->HeaderFileHandle)) {
        SYS_ERROR(CloseHandle);
        Result = PH_E_SYSTEM_CALL_FAILED;
        goto Error;
    }

    Table->HeaderFileHandle = NULL;

    //
    // We're done, finish up.
    //

    goto End;

Error:

    if (Result == S_OK) {
        Result = PH_E_ERROR_SAVING_HEADER_FILE;
    }

    //
    // Intentional follow-on to End.
    //

End:

    CONTEXT_END_TIMERS(SaveHeaderFile);

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
