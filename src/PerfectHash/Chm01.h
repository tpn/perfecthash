/*++

Copyright (c) 2018-2022 Trent Nelson <trent@trent.me>

Module Name:

    Chm01.h

Abstract:

    This is the header file for the Chm01.c module, which is our first pass
    at the CHM perfect hash table algorithm.  It defines types related to the
    implementation of the CHM algorithm.  Chm02.c is based on Chm01.c, and is
    used to explore the viability of CUDA support.  Thus, some symbols in this
    file have 02 appended to them if they're specific to the Chm02.c module
    (instead of the 01 suffix for Chm01.c functionality).

--*/

#include "stdafx.h"

//
// Declare the main work and file work callback functions.
//

extern PERFECT_HASH_MAIN_WORK_CALLBACK ProcessGraphCallbackChm01;
extern PERFECT_HASH_MAIN_WORK_CALLBACK ProcessGraphCallbackChm02;
extern PERFECT_HASH_CONSOLE_WORK_CALLBACK ProcessConsoleCallbackChm01;
extern PERFECT_HASH_FILE_WORK_CALLBACK FileWorkCallbackChm01;

typedef
_Must_inspect_result_
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
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI UNMAP_FILE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PFILE_WORK_ITEM Item
    );
typedef UNMAP_FILE *PUNMAP_FILE;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI CLOSE_FILE)(
    _In_ PPERFECT_HASH_TABLE Table,
    _In_ PFILE_WORK_ITEM Item
    );
typedef CLOSE_FILE *PCLOSE_FILE;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PREPARE_GRAPH_INFO)(
    _In_ PPERFECT_HASH_TABLE Table,
    _When_(PrevInfo == NULL, _Out_)
    _When_(PrevInfo != NULL, _Inout_)
        PGRAPH_INFO Info,
    _Out_opt_ PGRAPH_INFO PrevInfo
    );
typedef PREPARE_GRAPH_INFO *PPREPARE_GRAPH_INFO;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PREPARE_TABLE_OUTPUT_DIRECTORY)(
    _In_ PPERFECT_HASH_TABLE Table
    );
typedef PREPARE_TABLE_OUTPUT_DIRECTORY *PPREPARE_TABLE_OUTPUT_DIRECTORY;

extern PREPARE_FILE PrepareFileChm01;
extern UNMAP_FILE UnmapFileChm01;
extern CLOSE_FILE CloseFileChm01;

extern PREPARE_TABLE_OUTPUT_DIRECTORY PrepareTableOutputDirectory;

extern PREPARE_GRAPH_INFO PrepareGraphInfoChm01;
extern PREPARE_GRAPH_INFO PrepareGraphInfoChm02;

//
//
// Internal methods private to Chm01.c and Chm01Compat.c.
//

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PREPARE_GRAPH_INFO)(
    _In_ PPERFECT_HASH_TABLE Table,
    _When_(PrevInfo == NULL, _Out_)
    _When_(PrevInfo != NULL, _Inout_)
        PGRAPH_INFO Info,
    _Out_opt_ PGRAPH_INFO PrevInfo
    );
typedef PREPARE_GRAPH_INFO *PPREPARE_GRAPH_INFO;

extern PREPARE_GRAPH_INFO PrepareGraphInfoChm01;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI PREPARE_TABLE_OUTPUT_DIRECTORY)(
    _In_ PPERFECT_HASH_TABLE Table
    );
typedef PREPARE_TABLE_OUTPUT_DIRECTORY *PPREPARE_TABLE_OUTPUT_DIRECTORY;

extern PREPARE_TABLE_OUTPUT_DIRECTORY PrepareTableOutputDirectory;

_Must_inspect_result_
_Success_(return >= 0)
HRESULT
PrintCurrentContextStatsChm01(
    _In_ PPERFECT_HASH_CONTEXT Context
    );

//
// Helper macro for adding a 4-space indent to the output stream.
//

#define INDENT() {            \
    Long = (PULONG)Output;    \
    *Long = Indent;           \
    Output += sizeof(Indent); \
}

//
// Define helper macros for checking prepare and save file work errors.
//

#define EXPAND_AS_CHECK_ERRORS(                                      \
    Verb, VUpper, Name, Upper,                                       \
    EofType, EofValue,                                               \
    Suffix, Extension, Stream, Base                                  \
)                                                                    \
    if (Verb##Name.NumberOfErrors > 0) {                             \
        Result = Verb##Name.LastResult;                              \
        if (Result == S_OK || Result == E_UNEXPECTED) {              \
            Result = PH_E_ERROR_DURING_##VUpper##_##Upper;           \
        }                                                            \
        PH_ERROR(                                                    \
            CreatePerfectHashTableImplChm01_ErrorDuring##Verb##Name, \
            Result                                                   \
        );                                                           \
        goto Error;                                                  \
    }

#define CHECK_ALL_PREPARE_ERRORS() \
    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CHECK_ERRORS)

#define CHECK_ALL_SAVE_ERRORS() \
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_CHECK_ERRORS)


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
