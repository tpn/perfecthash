/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01.h

Abstract:

    This is the header file for the Chm01.c module, which is our first pass
    at the CHM perfect hash table algorithm.  It defines types related to the
    implementation of the CHM algorithm.

--*/

#include "stdafx.h"

//
// Declare the main work and file work callback functions.
//

extern PERFECT_HASH_MAIN_WORK_CALLBACK ProcessGraphCallbackChm01;
extern PERFECT_HASH_FILE_WORK_CALLBACK FileWorkCallbackChm01;

extern SHOULD_WE_CONTINUE_TRYING_TO_SOLVE_GRAPH
    ShouldWeContinueTryingToSolveGraphChm01;

typedef
_Check_return_
_Success_(return >= 0)
_When_(
    Item->Flags.PrepareOnce == TRUE,
    _Pre_satisfies_(*Item->FilePointer == NULL)
)
HRESULT
(NTAPI PREPARE_FILE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PFILE_WORK_ITEM Item,
    _In_ PPERFECT_HASH_PATH Path,
    _In_ PLARGE_INTEGER EndOfFile,
    _In_opt_ HANDLE DependentEvent
    );
typedef PREPARE_FILE *PPREPARE_FILE;

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(NTAPI UNMAP_FILE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PFILE_WORK_ITEM Item
    );
typedef UNMAP_FILE *PUNMAP_FILE;

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(NTAPI CLOSE_FILE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PFILE_WORK_ITEM Item
    );
typedef CLOSE_FILE *PCLOSE_FILE;

extern PREPARE_FILE PrepareFileChm01;
extern UNMAP_FILE UnmapFileChm01;
extern CLOSE_FILE CloseFileChm01;

//
// Helper macro for adding a 4-space indent to the output stream.
//

#define INDENT() {            \
    Long = (PULONG)Output;    \
    *Long = Indent;           \
    Output += sizeof(Indent); \
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
