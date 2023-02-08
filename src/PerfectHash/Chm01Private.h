/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    Chm01Private.h

Abstract:

    Private header file for the Chm01 components.

--*/

#include "stdafx.h"

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
