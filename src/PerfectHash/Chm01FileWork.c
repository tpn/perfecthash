/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWork.c

Abstract:

    This module implement the file work callback routine for the CHM v1 algo
    implementation of the perfect hash library.

    The FileWorkCallbackChm01 routine is the main entry point for all file work
    that has been requested via FILE_WORK_ITEM structs and submitted via the
    PERFECT_HASH_CONTEXT's "file work" threadpool (in Chm01.c).

    Generic preparation and saving functionality is also implemented by way of
    PrepareFileChm01 and SaveFileChm01 routines.

--*/

#include "stdafx.h"

//
// File work callback array.
//

#define EXPAND_AS_CALLBACK(Verb, VUpper, Name, Upper) Verb##Name##Chm01,

FILE_WORK_CALLBACK_IMPL *FileCallbacks[] = {
    NULL,
    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CALLBACK)
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CALLBACK)
    NULL
};

//
// Forward decls of routines and arrays in this file.
//

PERFECT_HASH_FILE_WORK_CALLBACK FileWorkCallbackChm01;
PREPARE_FILE PrepareFileChm01;
SAVE_FILE SaveFileChm01;

//
// Begin method implementations.
//

_Use_decl_annotations_
VOID
FileWorkCallbackChm01(
    PTP_CALLBACK_INSTANCE Instance,
    PPERFECT_HASH_CONTEXT Context,
    PLIST_ENTRY ListEntry
    )
/*++

Routine Description:

    This routine is the callback entry point for file-oriented work we want
    to perform in the file work threadpool context.

Arguments:

    Instance - Supplies a pointer to the callback instance for this invocation.

    Context - Supplies a pointer to the active context for the graph solving.

    ListEntry - Supplies a pointer to the list entry that was removed from the
        context's file work list head.

Return Value:

    None.

--*/
{
    PRTL Rtl;
    ULONG FileIndex;
    ULONG EventIndex;
    ULONG DependentEventIndex;
    HRESULT Result = S_OK;
    PGRAPH_INFO Info;
    FILE_WORK_ID FileWorkId;
    PFILE_WORK_ITEM Item;
    PFILE_WORK_CALLBACK_IMPL Impl = NULL;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_KEYS Keys;
    HANDLE Event;
    HANDLE DependentEvent = NULL;
    PPERFECT_HASH_PATH Path = NULL;
    PTABLE_INFO_ON_DISK TableInfo;
    PPERFECT_HASH_FILE *File = NULL;

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

    FileWorkId = Item->FileWorkId;

    ASSERT(IsValidFileWorkId(FileWorkId));

    EventIndex = FileWorkIdToEventIndex(FileWorkId);
    Event = *(&Context->FirstPreparedEvent + EventIndex);

    FileIndex = FileWorkIdToFileIndex(FileWorkId);
    File = &Table->FirstFile + FileIndex;

    Item->FilePointer = File;

    Impl = FileCallbacks[FileWorkId];

    if (IsPrepareFileWorkId(FileWorkId)) {

        PCEOF_INIT Eof;
        LARGE_INTEGER EndOfFile;
        ULARGE_INTEGER NumberOfTableElements;
        PCUNICODE_STRING NewExtension;
        PCUNICODE_STRING NewDirectory;
        PCUNICODE_STRING NewStreamName = NULL;
        PCUNICODE_STRING NewBaseName = NULL;
        PCUNICODE_STRING AdditionalSuffix;

        //
        // All output files are rooted within in the table's output directory.
        //

        NewDirectory = &Table->OutputDirectory->Path->FullPath;

        //
        // Initialize variables specific to the file work ID.
        //

        NewExtension = FileWorkItemExtensions[FileWorkId];
        AdditionalSuffix = FileWorkItemSuffixes[FileWorkId];

        NewStreamName = FileWorkItemStreamNames[FileWorkId];

        if (NewStreamName) {


            //
            // Streams don't need more than one prepare call, as their path
            // hangs off their owning file's path (and thus, will automatically
            // inherit its scheduled rename), and their size never changes.  If
            // *File is non-NULL, it means the stream has already been prepared,
            // in which case, we can jump straight to the end.
            //

            if (*File) {
                goto End;
            }

            //
            // Streams are dependent upon their "owning" file, which always
            // reside before them.
            //

            DependentEvent = *(&Context->FirstPreparedEvent + (EventIndex - 1));
        }

        NumberOfTableElements.QuadPart = (
            TableInfo->NumberOfTableElements.QuadPart
        );

        //
        // Default size for end-of-file is the system allocation granularity.
        //

        EndOfFile.QuadPart = Context->SystemAllocationGranularity;

        //
        // Initialize the end-of-file based on the relevant file work ID's
        // EOF_INIT structure.
        //

        Eof = &EofInits[FileWorkId];

        switch (Eof->Type) {

            case EofInitTypeDefault:
                break;

            case EofInitTypeZero:
                EndOfFile.QuadPart = 1;
                break;

            case EofInitTypeAssignedSize:
                EndOfFile.QuadPart = Info->AssignedSizeInBytes;
                break;

            case EofInitTypeFixed:
                EndOfFile.QuadPart = Eof->FixedValue;
                break;

            case EofInitTypeNumberOfKeysMultiplier:
                EndOfFile.QuadPart += (
                    (LONGLONG)Keys->NumberOfElements.QuadPart *
                    Eof->Multiplier
                );
                break;

            case EofInitTypeNumberOfTableElementsMultiplier:
                EndOfFile.QuadPart += (
                    NumberOfTableElements.QuadPart *
                    Eof->Multiplier
                );
                break;

            case EofInitTypeNumberOfPages:
                EndOfFile.QuadPart = (
                    (ULONG_PTR)PAGE_SIZE *
                    (ULONG_PTR)Eof->NumberOfPages
                );
                break;

            case EofInitTypeNull:
            case EofInitTypeInvalid:
            default:
                PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
                return;
        }

        Result = PerfectHashTableCreatePath(Table,
                                            Table->Keys->File->Path,
                                            &NumberOfTableElements,
                                            Table->AlgorithmId,
                                            Table->MaskFunctionId,
                                            Table->HashFunctionId,
                                            NewDirectory,
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

        if (!IsSaveFileWorkId(FileWorkId)) {
            ASSERT(FALSE);
            Result = PH_E_INVARIANT_CHECK_FAILED;
            goto End;
        }

        //
        // All save events are dependent on their previous prepare events.
        //

        DependentEventIndex = FileWorkIdToDependentEventIndex(FileWorkId);
        DependentEvent = *(&Context->FirstPreparedEvent + DependentEventIndex);

        WaitResult = WaitForSingleObject(DependentEvent, INFINITE);
        if (WaitResult != WAIT_OBJECT_0) {
            SYS_ERROR(WaitForSingleObject);
            Result = PH_E_SYSTEM_CALL_FAILED;
            goto End;
        }

        //
        // If the file hasn't been set at this point, the prepare routine
        // was not successful.  We use E_UNEXPECTED as the error code here
        // as we just need *something* to be set when End: is jumped to in
        // order for the interlocked increment of the error count to occur.
        // The parent Chm01.c routine will adjust this to the proper error
        // code as necessary.
        //

        if (!*File) {
            Result = E_UNEXPECTED;
            goto End;
        }

        if (!Impl) {
            Result = SaveFileChm01(Table, *File);
            if (FAILED(Result)) {

                //
                // Nothing needs doing here.  The Result will bubble back up
                // via the normal mechanisms.
                //

                NOTHING;
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

    SetEventWhenCallbackReturns(Instance, Event);

    return;
}


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
                                    Table->OutputDirectory,
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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
