/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    Chm01FileWorkGraphVertexPairsFile.c

Abstract:

    This module implements the save file work callback routine for the graph
    vertex pairs file as part of the CHM v1 algorithm implementation for the
    perfect hash library.

--*/

#include "stdafx.h"

_Use_decl_annotations_
HRESULT
SaveGraphVertexPairsFileChm01(
    PPERFECT_HASH_CONTEXT Context,
    PFILE_WORK_ITEM Item
    )
{
    PRTL Rtl;
    PULONG Dest;
    PGRAPH Graph;
    PVOID Source;
    PGRAPH_INFO Info;
    HRESULT Result = S_OK;
    LONGLONG SizeInBytes;
    LARGE_INTEGER EndOfFile;
    PPERFECT_HASH_TABLE Table;
    PPERFECT_HASH_FILE File;

    //
    // Initialize aliases.
    //

    Rtl = Context->Rtl;
    Table = Context->Table;
    File = *Item->FilePointer;
    Dest = (PULONG)File->BaseAddress;
    Graph = (PGRAPH)Context->SolvedContext;
    Info = Graph->Info;
    Source = Graph->VertexPairs;

    if (Graph->_SavedVertexPairs) {
        return S_OK;
    }

    SizeInBytes = Info->VertexPairsSizeInBytes;

    if (SizeInBytes > File->FileInfo.EndOfFile.QuadPart) {
        PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
    }

    //
    // The graph has been solved.  Copy the array of vertex pairs to the
    // backing memory map.
    //

    CopyMemory(Dest, Source, SizeInBytes);

    EndOfFile.QuadPart = (LONGLONG)SizeInBytes;

    //
    // Update the number of bytes written.
    //

    File->NumberOfBytesWritten.QuadPart = EndOfFile.QuadPart;

    //
    // We're done, finish up.
    //

    Graph->_SavedVertexPairs = 1;
    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
