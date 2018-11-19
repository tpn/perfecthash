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

    Generic preparation, unmapping and closing functionality is also implemented
    by way of PrepareFileChm01, UnmapFileChm01 and CloseFileChm01 routines.

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
// Forward decls of prepare, unmap and close routines.
//

PREPARE_FILE PrepareFileChm01;
UNMAP_FILE UnmapFileChm01;
CLOSE_FILE CloseFileChm01;

//
// Begin method implementations.
//

PERFECT_HASH_FILE_WORK_CALLBACK FileWorkCallbackChm01;

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
    ULONG ContextFileIndex;
    ULONG DependentEventIndex;
    HRESULT Result = S_OK;
    PGRAPH_INFO Info;
    FILE_ID FileId;
    FILE_WORK_ID FileWorkId;
    CONTEXT_FILE_ID ContextFileId;
    PFILE_WORK_ITEM Item;
    PFILE_WORK_CALLBACK_IMPL Impl = NULL;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_KEYS Keys;
    HANDLE Event;
    HANDLE DependentEvent = NULL;
    PPERFECT_HASH_PATH Path = NULL;
    PTABLE_INFO_ON_DISK TableInfo;
    PPERFECT_HASH_FILE *File = NULL;
    PPERFECT_HASH_FILE ContextFile = NULL;
    BOOLEAN IsContextFile = FALSE;

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

    //
    // Resolve the relevant file and event indices and associated pointers.
    // (Note that the close file work type does not use events.)
    //

    FileIndex = FileWorkIdToFileIndex(FileWorkId);
    File = &Table->FirstFile + FileIndex;

    if (!IsCloseFileWorkId(FileWorkId)) {
        EventIndex = FileWorkIdToEventIndex(FileWorkId);
        Event = *(&Context->FirstPreparedEvent + EventIndex);
    } else {
        EventIndex = (ULONG)-1;
        Event = NULL;
    }

    //
    // Set the file ID.
    //

    Item->FileId = FileId = FileWorkIdToFileId(FileWorkId);

    //
    // Determine if this is a context file.  Context files get treated
    // differently to normal table output files in that they are only
    // prepared and saved once per context instance.
    //

    ContextFileId = (CONTEXT_FILE_ID)FileId;

    if (!IsValidContextFileId(ContextFileId)) {

        ContextFileId = ContextFileNullId;

    } else {

        Item->Flags.IsContextFile = IsContextFile = TRUE;

        //
        // Override the table's file pointer with the context's one.
        //

        ContextFileIndex = ContextFileIdToContextFileIndex(ContextFileId);
        File = &Context->FirstFile + ContextFileIndex;

        if (IsPrepareFileWorkId(FileWorkId)) {

            Item->Flags.PrepareOnce = TRUE;

            ContextFile = *File;

            //
            // Context files only get one prepare call.  If ContextFile is not
            // NULL here, it means the file has already been prepared once, so
            // we can jump straight to the end.
            //

            if (ContextFile) {
                goto End;
            }
        }
    }

    Item->FilePointer = File;

    if (!IsCloseFileWorkId(FileWorkId)) {
        Impl = FileCallbacks[FileWorkId];
    } else {
        Impl = NULL;
    }

    if (IsPrepareFileWorkId(FileWorkId)) {

        PCEOF_INIT Eof;
        LARGE_INTEGER EndOfFile;
        ULONG NumberOfResizeEvents;
        ULARGE_INTEGER NumberOfTableElements;
        PCUNICODE_STRING NewExtension = NULL;
        PCUNICODE_STRING NewDirectory = NULL;
        PCUNICODE_STRING NewBaseName = NULL;
        PCUNICODE_STRING NewStreamName = NULL;
        PCUNICODE_STRING AdditionalSuffix = NULL;

        NewExtension = FileWorkItemExtensions[FileWorkId];

        if (IsContextFile) {

            //
            // All context files are rooted within in the context's base
            // output directory.
            //

            NewDirectory = &Context->BaseOutputDirectory->Path->FullPath;
            NewBaseName = FileWorkItemBaseNames[FileWorkId];

        } else {

            //
            // All table output files are rooted within in the table's output
            // directory.
            //

            NewDirectory = &Table->OutputDirectory->Path->FullPath;

            ASSERT(IsValidUnicodeString(NewDirectory));

            //
            // Initialize variables specific to the file work ID.
            //

            AdditionalSuffix = FileWorkItemSuffixes[FileWorkId];

            NewStreamName = FileWorkItemStreamNames[FileWorkId];

            if (NewStreamName) {

                Item->Flags.PrepareOnce = TRUE;

                //
                // Streams don't need more than one prepare call, as their path
                // hangs off their owning file's path (and thus, will inherit
                // its scheduled rename), and their size never changes.  If we
                // dereference *File and it's non-NULL, it means the stream has
                // already been prepared, in which case, we can jump straight to
                // the end.
                //

                if (*File) {
                    goto End;
                }

                //
                // Streams are dependent upon their "owning" file, which always
                // reside before them.
                //

                DependentEvent = *(
                    &Context->FirstPreparedEvent +
                    (EventIndex - 1)
                );

            } else if (FileRequiresUuid(FileId) && !*File) {

                //
                // Generate a UUID the first time we prepare a VC Project file
                // or VS Solution file.
                //

                Result = RtlCreateUuidString(Rtl, &Item->Uuid);
                if (FAILED(Result)) {
                    goto End;
                }

            }
        }

        NumberOfResizeEvents = (ULONG)Context->NumberOfTableResizeEvents;
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

        //
        // We create the path differently depending on whether or not it's
        // a context file (rooted in the context's base output directory) or
        // a table file (rooted in the table's output directory).  E.g.
        //
        //  Context's base output directory:
        //
        //      C:\Temp\output
        //
        //  Table's base output directory:
        //
        //      C:\Temp\output\KernelBase_2415_8192_Chm01_Crc32Rotate_And
        //

        if (IsContextFile) {

            //
            // Create a new path instance.
            //

            Result = Context->Vtbl->CreateInstance(Context,
                                                   NULL,
                                                   &IID_PERFECT_HASH_PATH,
                                                   &Path);

            if (FAILED(Result)) {
                PH_ERROR(PerfectHashTableCreateInstance, Result);
                goto End;
            }

            //
            // Create the underlying path.
            //

            Result = Path->Vtbl->Create(Path,
                                        Context->BaseOutputDirectory->Path,
                                        NewDirectory,   // NewDirectory
                                        NULL,           // DirectorySuffix
                                        NewBaseName,    // NewBaseName
                                        NULL,           // BaseNameSuffix
                                        NewExtension,   // NewExtension
                                        NULL,           // NewStreamName
                                        NULL,           // Parts
                                        NULL);          // Reserved

            if (FAILED(Result)) {
                PH_ERROR(PerfectHashPathCreate, Result);
                goto End;
            }

        } else {

            //
            // This is a normal table file.
            //

            Result = PerfectHashTableCreatePath(Table,
                                                Table->Keys->File->Path,
                                                &NumberOfResizeEvents,
                                                &NumberOfTableElements,
                                                Table->AlgorithmId,
                                                Table->HashFunctionId,
                                                Table->MaskFunctionId,
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

        }

        ASSERT(SUCCEEDED(Result));

        Result = PrepareFileChm01(Table,
                                  Item,
                                  Path,
                                  &EndOfFile,
                                  DependentEvent);

        if (FAILED(Result)) {
            PH_ERROR(PrepareFileChm01, Result);
            goto End;
        }

        if (Impl) {
            Result = Impl(Context, Item);
        }

    } else if (IsSaveFileWorkId(FileWorkId)) {

        ULONG WaitResult;

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

        if (Impl) {
            Result = Impl(Context, Item);
            if (FAILED(Result)) {
                goto End;
            }
        }

        //
        // Unmap the file (which has the effect of flushing the file buffers),
        // but don't close it.  We do this here, as part of the save file work,
        // in order to reduce the amount of work each file's Close() routine
        // has to do when the close work items are submitted in parallel.
        //

        Result = UnmapFileChm01(Table, Item);
        if (FAILED(Result)) {

            //
            // Nothing needs doing here.  The Result will bubble back up
            // via the normal mechanisms.
            //

            NOTHING;
        }

    } else {

        //
        // Invariant check: our file work ID should be of type 'Close' here.
        //

        if (!IsCloseFileWorkId(FileWorkId)) {
            PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        }

        //
        // As above (in the save logic), if *File is NULL, use E_UNEXPECTED
        // as our Result.
        //

        if (!*File) {
            Result = E_UNEXPECTED;
            goto End;
        }

        Result = CloseFileChm01(Table, Item);
        if (FAILED(Result)) {

            //
            // Nothing needs doing here.  The Result will bubble back up
            // via the normal mechanisms.
            //

            NOTHING;
        }

    }

    //
    // Intentional follow-on to End.
    //

End:

    if (Path) {
        Path->Vtbl->Release(Path);
        Path = NULL;
    }

    //
    // If the item's UUID string buffer is non-NULL here, the downstream routine
    // did not successfully take ownership of it, and thus, we're responsbile
    // for freeing it.
    //

    if (Item->Uuid.Buffer) {
        ASSERT(FileRequiresUuid(FileId));
        if (File && *File) {
            ASSERT((*File)->Uuid.Buffer == NULL);
        }
        Result = RtlFreeUuidString(Rtl, &Item->Uuid);
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

    if (Event) {
        SetEventWhenCallbackReturns(Instance, Event);
    }

    return;
}


PREPARE_FILE PrepareFileChm01;

_Use_decl_annotations_
HRESULT
PrepareFileChm01(
    PPERFECT_HASH_TABLE Table,
    PFILE_WORK_ITEM Item,
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

    Item - Supplies a pointer to the active file work item associated with
        the preparation.

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
    PPERFECT_HASH_DIRECTORY Directory = NULL;

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

    File = *Item->FilePointer;

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

        if (!IsContextFileWorkItem(Item)) {

            //
            // Table files always get associated with the table's output
            // directory (i.e. the Create() call coming up will call the
            // directory's AddFile() method if Directory is not NULL).
            //

            Directory = Table->OutputDirectory;
        }

        Result = File->Vtbl->Create(File,
                                    Path,
                                    EndOfFile,
                                    Directory,
                                    NULL);

        if (FAILED(Result)) {
            PH_ERROR(PerfectHashFileCreate, Result);
            File->Vtbl->Release(File);
            File = NULL;
            goto Error;
        }

        //
        // Set the file ID and then update the file pointer.
        //

        File->FileId = Item->FileId;
        *Item->FilePointer = File;

        if (!FileRequiresUuid(File->FileId)) {

            //
            // No UUID string buffer should be set if the file hasn't been
            // marked as requiring a UUID.
            //

            ASSERT(Item->Uuid.Buffer == NULL);

        } else {

            //
            // Verify the Item->Uuid string has been filled out.
            //

            if (!IsValidUuidString(&Item->Uuid)) {
                Result = PH_E_INVARIANT_CHECK_FAILED;
                PH_ERROR(PrepareFileChm01_VCProjectItemMissingUuid, Result);
                goto Error;
            }

            //
            // Copy the details over to the file instance, which will now "own"
            // the underlying UUID string buffer (this is freed in the file's
            // rundown routine), and zero the Item->Uuid representation.
            //

            CopyInline(&File->Uuid, &Item->Uuid, sizeof(File->Uuid));
            ZeroStructInline(Item->Uuid);
        }

    } else {

        //
        // Invariant check: context files should only be prepared once.
        //

        if (IsContextFileWorkItem(Item)) {
            Result = PH_E_CONTEXT_FILE_ALREADY_PREPARED;
            PH_ERROR(PrepareFileChm01, Result);
            goto Error;
        }

        //
        // Invariant check: no UUID should be set.
        //

        if (Item->Uuid.Buffer) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareFileChm01_ItemUuidBufferNotNull, Result);
            goto Error;
        }

        //
        // Invariant check: File->FileId should already be set, and it should
        // match the file ID specified in Item.
        //

        if (!IsValidFileId(File->FileId)) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareFileChm01_SaveFile_InvalidFileId, Result);
            goto Error;
        }

        if (File->FileId != Item->FileId) {
            Result = PH_E_INVARIANT_CHECK_FAILED;
            PH_ERROR(PrepareFileChm01_SaveFile_FileIdMismatch, Result);
            goto Error;
        }

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

UNMAP_FILE UnmapFileChm01;

_Use_decl_annotations_
HRESULT
UnmapFileChm01(
    PPERFECT_HASH_TABLE Table,
    PFILE_WORK_ITEM Item
    )
/*++

Routine Description:

    Unmaps a file instance associated with a table.

Arguments:

    Table - Supplies a pointer to the table owning the file to be unmapped.

    Item - Supplies a pointer to the file work item for this unmap action.

Return Value:

    S_OK - File unmapped successfully.  Otherwise, an appropriate error code.

--*/
{
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;

    UNREFERENCED_PARAMETER(Table);

    File = *Item->FilePointer;

    //
    // Unmap the file if it's either a) not a context file, or b) if it is a
    // context file, only if it hasn't already been unmapped.
    //

    if (!IsContextFileWorkItem(Item) || !IsFileUnmapped(File)) {
        Result = File->Vtbl->Unmap(File);
        if (FAILED(Result)) {
            PH_ERROR(UnmapFileChm01, Result);
        }
    }

    return Result;
}

CLOSE_FILE CloseFileChm01;

_Use_decl_annotations_
HRESULT
CloseFileChm01(
    PPERFECT_HASH_TABLE Table,
    PFILE_WORK_ITEM Item
    )
/*++

Routine Description:

    Closes a file instance associated with a table.

    N.B.  If an error has occurred, Item->EndOfFile will point to a
          LARGE_INTEGER with value 0, which informs the file's Close()
          machinery to delete the file.  (Otherwise, the file will be
          truncated based on the value of File->NumberOfBytesWritten.)

Arguments:

    Table - Supplies a pointer to the table owning the file to be closed.

    Item - Supplies a pointer to the file work item for this close action.

Return Value:

    S_OK - File closed successfully.  Otherwise, an appropriate error code.

--*/
{
    HRESULT Result = S_OK;
    PPERFECT_HASH_FILE File;

    UNREFERENCED_PARAMETER(Table);

    File = *Item->FilePointer;

    //
    // Close the file if it's either a) not a context file, or b) if it is a
    // context file, only if it hasn't already been closed.
    //

    if (!IsContextFileWorkItem(Item) || !IsFileClosed(File)) {
        Result = File->Vtbl->Close(File, Item->EndOfFile);
        if (FAILED(Result)) {
            PH_ERROR(CloseFileChm01, Result);
        }
    }

    return Result;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
