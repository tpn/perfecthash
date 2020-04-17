/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    ExtractArg.h

Abstract:

    This is the private header file for the command line argument extraction
    functionality.  This refers to converting a PCUNICODE_STRING representing
    a command line argument into its corresponding flag or table create param
    representation.

--*/

#pragma once

#include "stdafx.h"

//
// Define function pointer typedefs.
//

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI TRY_EXTRACT_ARG_KEYS_DIRECTORY)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ PCUNICODE_STRING Argument,
    _Inout_ PUNICODE_STRING KeysDirectory
    );
typedef TRY_EXTRACT_ARG_KEYS_DIRECTORY
      *PTRY_EXTRACT_ARG_KEYS_DIRECTORY;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI TRY_EXTRACT_ARG_CONTEXT_BULK_CREATE_FLAGS)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ PCUNICODE_STRING Argument,
    _Inout_ PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS Flags
    );
typedef TRY_EXTRACT_ARG_CONTEXT_BULK_CREATE_FLAGS
      *PTRY_EXTRACT_ARG_CONTEXT_BULK_CREATE_FLAGS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI TRY_EXTRACT_ARG_CONTEXT_TABLE_CREATE_FLAGS)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ PCUNICODE_STRING Argument,
    _Inout_ PPERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS Flags
    );
typedef TRY_EXTRACT_ARG_CONTEXT_TABLE_CREATE_FLAGS
      *PTRY_EXTRACT_ARG_CONTEXT_TABLE_CREATE_FLAGS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI TRY_EXTRACT_ARG_KEYS_LOAD_FLAGS)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ PCUNICODE_STRING Argument,
    _Inout_ PPERFECT_HASH_KEYS_LOAD_FLAGS Flags
    );
typedef TRY_EXTRACT_ARG_KEYS_LOAD_FLAGS *PTRY_EXTRACT_ARG_KEYS_LOAD_FLAGS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI TRY_EXTRACT_ARG_TABLE_CREATE_FLAGS)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ PCUNICODE_STRING Argument,
    _Inout_ PPERFECT_HASH_TABLE_CREATE_FLAGS Flags
    );
typedef TRY_EXTRACT_ARG_TABLE_CREATE_FLAGS *PTRY_EXTRACT_ARG_TABLE_CREATE_FLAGS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI TRY_EXTRACT_ARG_TABLE_LOAD_FLAGS)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ PCUNICODE_STRING Argument,
    _Inout_ PPERFECT_HASH_TABLE_LOAD_FLAGS Flags
    );
typedef TRY_EXTRACT_ARG_TABLE_LOAD_FLAGS *PTRY_EXTRACT_ARG_TABLE_LOAD_FLAGS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI TRY_EXTRACT_ARG_TABLE_COMPILE_FLAGS)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ PCUNICODE_STRING Argument,
    _Inout_ PPERFECT_HASH_TABLE_COMPILE_FLAGS Flags
    );
typedef TRY_EXTRACT_ARG_TABLE_COMPILE_FLAGS
      *PTRY_EXTRACT_ARG_TABLE_COMPILE_FLAGS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI TRY_EXTRACT_ARG_TABLE_CREATE_PARAMETERS)(
    _In_ PRTL Rtl,
    _In_ PCUNICODE_STRING Argument,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );
typedef TRY_EXTRACT_ARG_TABLE_CREATE_PARAMETERS
      *PTRY_EXTRACT_ARG_TABLE_CREATE_PARAMETERS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI TRY_EXTRACT_VALUE_ARRAY)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ PCUNICODE_STRING InputString,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETER Param,
    _In_opt_ BOOLEAN EnsureSortedAndUnique
    );
typedef TRY_EXTRACT_VALUE_ARRAY *PTRY_EXTRACT_VALUE_ARRAY;

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI TRY_EXTRACT_SEED_MASK_COUNTS)(
    _In_ PRTL Rtl,
    _In_ PALLOCATOR Allocator,
    _In_ PCUNICODE_STRING InputString,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETER Param,
    _In_range_(1, 8) BYTE SeedNumber,
    _In_range_(1, 4) BYTE ByteNumber,
    _In_range_(32, 32) BYTE NumberOfCounts
    );
typedef TRY_EXTRACT_SEED_MASK_COUNTS *PTRY_EXTRACT_SEED_MASK_COUNTS;

typedef
_Must_inspect_result_
_Success_(return >= 0)
_Post_satisfies_(TableCreateParameters->NumberOfElements == 0)
_Post_satisfies_(TableCreateParameters->Params == NULL)
HRESULT
(NTAPI CLEANUP_TABLE_CREATE_PARAMETERS)(
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    );
typedef CLEANUP_TABLE_CREATE_PARAMETERS *PCLEANUP_TABLE_CREATE_PARAMETERS;

typedef
_Success_(return == 0)
HRESULT
(NTAPI GET_TABLE_CREATE_PARAMETER_FOR_ID)(
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters,
    _In_ PERFECT_HASH_TABLE_CREATE_PARAMETER_ID ParameterId,
    _Out_ PPERFECT_HASH_TABLE_CREATE_PARAMETER *Parameter
    );
typedef GET_TABLE_CREATE_PARAMETER_FOR_ID *PGET_TABLE_CREATE_PARAMETER_FOR_ID;

//
// Declare functions.
//

#ifndef __INTELLISENSE__
extern TRY_EXTRACT_ARG_KEYS_DIRECTORY TryExtractArgKeysDirectory;
extern TRY_EXTRACT_VALUE_ARRAY TryExtractValueArray;
extern TRY_EXTRACT_ARG_CONTEXT_BULK_CREATE_FLAGS TryExtractArgContextBulkCreateFlags;
extern TRY_EXTRACT_ARG_CONTEXT_TABLE_CREATE_FLAGS TryExtractArgContextTableCreateFlags;
extern TRY_EXTRACT_ARG_KEYS_LOAD_FLAGS TryExtractArgKeysLoadFlags;
extern TRY_EXTRACT_ARG_TABLE_CREATE_FLAGS TryExtractArgTableCreateFlags;
extern TRY_EXTRACT_ARG_TABLE_COMPILE_FLAGS TryExtractArgTableCompileFlags;
extern TRY_EXTRACT_ARG_TABLE_CREATE_PARAMETERS TryExtractArgTableCreateParameters;
extern CLEANUP_TABLE_CREATE_PARAMETERS CleanupTableCreateParameters;
extern GET_TABLE_CREATE_PARAMETER_FOR_ID GetTableCreateParameterForId;
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
