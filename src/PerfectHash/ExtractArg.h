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
    _In_ PALLOCATOR Allocator,
    _In_ PCUNICODE_STRING Argument,
    _Inout_ PULONG NumberOfTableCreateParameters,
    _Inout_ PPERFECT_HASH_TABLE_CREATE_PARAMETER *TableCreateParameters
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
(NTAPI DESTROY_TABLE_CREATE_PARAMETERS)(
    _In_ PALLOCATOR Allocator,
    _In_ ULONG NumberOfTableCreateParametersPointer,
    _Inout_ _Post_invalid_ PPERFECT_HASH_TABLE_CREATE_PARAMETER
        *TableCreateParametersPointer
    );
typedef DESTROY_TABLE_CREATE_PARAMETERS *PDESTROY_TABLE_CREATE_PARAMETERS;


//
// Declare functions.
//

#ifndef __INTELLISENSE__
extern TRY_EXTRACT_VALUE_ARRAY TryExtractValueArray;
extern TRY_EXTRACT_ARG_CONTEXT_BULK_CREATE_FLAGS TryExtractArgContextBulkCreateFlags;
extern TRY_EXTRACT_ARG_CONTEXT_TABLE_CREATE_FLAGS TryExtractArgContextTableCreateFlags;
extern TRY_EXTRACT_ARG_KEYS_LOAD_FLAGS TryExtractArgKeysLoadFlags;
extern TRY_EXTRACT_ARG_TABLE_CREATE_FLAGS TryExtractArgTableCreateFlags;
extern TRY_EXTRACT_ARG_TABLE_COMPILE_FLAGS TryExtractArgTableCompileFlags;
extern TRY_EXTRACT_ARG_TABLE_CREATE_PARAMETERS TryExtractArgTableCreateParameters;
extern DESTROY_TABLE_CREATE_PARAMETERS DestroyTableCreateParameters;
#endif

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
