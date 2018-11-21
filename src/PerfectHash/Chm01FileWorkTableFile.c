/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkTableFile.c

Abstract:

    This module implements the save file work callback routine for the table
    file as part of the CHM v1 algorithm implementation for the perfect hash
    library.

    The table file has the extension .pht1 and is simply an on-disk version of
    the "assigned" array that is obtained during the graph solving step.  This
    file has no preparation step (unlike, say, the C header file), as there is
    no work that can be done until the graph has been solved and the table data
    is available to save to disk.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
SaveTableFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PULONG Dest;
    PGRAPH Graph;
    PULONG Source;
    PVOID BaseAddress;
    HRESULT Result = S_OK;
    LONGLONG SizeInBytes;
    LARGE_INTEGER EndOfFile;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_FILE File;
    BOOLEAN LargePagesForTableData;
    PTABLE_INFO_ON_DISK TableInfoOnDisk;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = *Item->FilePointer;
    Dest = (PULONG)File->BaseAddress;
    Graph = (PGRAPH)Context->SolvedContext;
    Source = Graph->Assigned;
    TableInfoOnDisk = Table->TableInfoOnDisk;

    SizeInBytes = (
        TableInfoOnDisk->NumberOfTableElements.QuadPart *
        TableInfoOnDisk->KeySizeInBytes
    );

    if (SizeInBytes != File->FileInfo.EndOfFile.QuadPart) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    //
    // The graph has been solved.  Copy the array of assigned values to the
    // backing memory map.
    //

    CopyMemory(Dest, Source, SizeInBytes);

    EndOfFile.QuadPart = (LONGLONG)SizeInBytes;

    //
    // Update the number of bytes written.
    //

    File->NumberOfBytesWritten.QuadPart = EndOfFile.QuadPart;

    if (IsTableCreateOnly(Table)) {

        //
        // Nothing left to do, finish up.
        //

        goto End;
    }

    //
    // Allocate and copy the table data to an in-memory copy so that the table
    // can be used after Create() completes successfully.  See the comment in
    // the SaveTableInfoStreamChm01() routine for more information about why
    // this is necessary.
    //

    LargePagesForTableData = (
        Table->TableCreateFlags.TryLargePagesForTableData == TRUE
    );

    BaseAddress = Rtl->Vtbl->TryLargePageVirtualAlloc(Rtl,
                                                      NULL,
                                                      SizeInBytes,
                                                      MEM_RESERVE | MEM_COMMIT,
                                                      PAGE_READWRITE,
                                                      &LargePagesForTableData);

    Table->TableDataBaseAddress = BaseAddress;

    if (!BaseAddress) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // Update state indicating table data has been heap-allocated.
    //

    Table->State.TableDataWasHeapAllocated = TRUE;

    //
    // Update flags with large page result for values array.
    //

    Table->Flags.TableDataUsesLargePages = LargePagesForTableData;

    //
    // Copy the table data over to the newly allocated buffer.
    //

    CopyMemory(Table->TableDataBaseAddress, Source, SizeInBytes);

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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
