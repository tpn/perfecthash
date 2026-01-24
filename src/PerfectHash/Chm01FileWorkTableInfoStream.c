/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkTableInfoStream.c

Abstract:

    This module implements the save file work callback routine for the table
    info stream as part of the CHM v1 algorithm implementation for the perfect
    hash library.

    The table file info stream is an NTFS stream associated with the main table
    file (e.g. Table.pht1:Info) and contains metadata about the table (such as
    algorithm used, hash function ID etc) that allows it to be loaded from disk
    and used (via the Table->Vtbl->Load() interface).

    The info stream has no preparation step (unlike, say, the C header file),
    as there is no work that can be done until the graph has been solved and
    the relevant table metadata is available to be saved to disk.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
SaveTableInfoStreamChm01(
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
    PPERFECT_HASH_FILE File;
    PPERFECT_HASH_TABLE Table;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;
    PGRAPH_INFO_ON_DISK GraphInfoOnDisk;
    PGRAPH_INFO_ON_DISK NewGraphInfoOnDisk;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = *Item->FilePointer;
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
    // Copy the seed data from the graph to the table info structure.
    //

    ASSERT(Graph->FirstSeed);

    CopyMemory(&TableInfoOnDisk->FirstSeed,
               &Graph->FirstSeed,
               Graph->NumberOfSeeds * sizeof(Graph->FirstSeed));

    //
    // Wait on the verification complete event.  This is done in the
    // main thread straight after it dispatches our file work callback
    // (that ended up here).  We need to block on this event as we want
    // to save the timings for verification to the header.
    //

    WaitResult = WaitForSingleObject(
        Context->VerifiedTableEvent,
        UseOverlappedIo(Context) ? 0 : INFINITE
    );

    if (WaitResult == WAIT_TIMEOUT && UseOverlappedIo(Context)) {
        return S_FALSE;
    }
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

    TableInfoOnDisk->NumberOfTableResizeEvents =
        Context->NumberOfTableResizeEvents;

    TableInfoOnDisk->TotalNumberOfAttemptsWithSmallerTableSizes =
        Context->TotalNumberOfAttemptsWithSmallerTableSizes;

    TableInfoOnDisk->InitialTableSize = Context->InitialTableSize;

    TableInfoOnDisk->ClosestWeCameToSolvingGraphWithSmallerTableSizes =
        Context->ClosestWeCameToSolvingGraphWithSmallerTableSizes;

    //
    // Copy timer values.
    //

    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(Solve);
    CONTEXT_SAVE_TIMERS_TO_TABLE_INFO_ON_DISK(Verify);

    //
    // Update the number of bytes written.
    //

    File->NumberOfBytesWritten.QuadPart = sizeof(*GraphInfoOnDisk);

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
    // Update state indicating table info has been heap-allocated.
    //

    Table->State.TableInfoOnDiskWasHeapAllocated = TRUE;

    //
    // We're done, jump to the end.
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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
