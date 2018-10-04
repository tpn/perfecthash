/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    Chm01.h

Abstract:

    This is the header file for the Chm_01.c module, which is our first pass
    at the CHM perfect hash table algorithm.  It defines types related to the
    implementation of the CHM algorithm.

--*/

#include "stdafx.h"

//
// Declare the main work and file work callback functions.
//

extern PERFECT_HASH_MAIN_WORK_CALLBACK ProcessGraphCallbackChm01;
extern PERFECT_HASH_FILE_WORK_CALLBACK FileWorkCallbackChm01;

extern FILE_WORK_CALLBACK_IMPL SaveTableFileChm01;
extern FILE_WORK_CALLBACK_IMPL SaveTableInfoStreamChm01;
extern FILE_WORK_CALLBACK_IMPL SaveCSourceTableDataCallbackChm01;

extern FILE_WORK_CALLBACK_IMPL PrepareCHeaderFileChm01;
extern FILE_WORK_CALLBACK_IMPL SaveCHeaderFileChm01;

extern FILE_WORK_CALLBACK_IMPL PrepareCSourceFileChm01;
extern FILE_WORK_CALLBACK_IMPL PrepareCSourceKeysCallbackChm01;

extern SHOULD_WE_CONTINUE_TRYING_TO_SOLVE_GRAPH
    ShouldWeContinueTryingToSolveGraphChm01;

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(NTAPI PREPARE_FILE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PPERFECT_HASH_FILE *FilePointer,
    _In_ PPERFECT_HASH_PATH Path,
    _In_ PLARGE_INTEGER EndOfFile,
    _In_opt_ HANDLE DependentEvent
    );
typedef PREPARE_FILE *PPREPARE_FILE;

typedef
_Check_return_
_Success_(return >= 0)
HRESULT
(NTAPI SAVE_FILE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PPERFECT_HASH_FILE File
    );
typedef SAVE_FILE *PSAVE_FILE;

extern PREPARE_FILE PrepareFileChm01;
extern SAVE_FILE SaveFileChm01;

//
// Helper macro for adding a 4-space indent to the output stream.
//

#define INDENT() {            \
    Long = (PULONG)Output;    \
    *Long = Indent;           \
    Output += sizeof(Indent); \
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
