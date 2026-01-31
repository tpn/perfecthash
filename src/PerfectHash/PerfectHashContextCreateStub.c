/*++

Copyright (c) 2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContextCreateStub.c

Abstract:

    Stub implementations for CSV-based context creation routines in
    online-only builds.

--*/

#include "stdafx.h"

#ifdef PH_ONLINE_ONLY

PERFECT_HASH_CONTEXT_TABLE_CREATE PerfectHashContextTableCreate;

_Use_decl_annotations_
HRESULT
PerfectHashContextTableCreate(
    PPERFECT_HASH_CONTEXT Context,
    PCUNICODE_STRING KeysPath,
    PCUNICODE_STRING BaseOutputDirectory,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PPERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlagsPointer,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlagsPointer,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
{
    UNREFERENCED_PARAMETER(Context);
    UNREFERENCED_PARAMETER(KeysPath);
    UNREFERENCED_PARAMETER(BaseOutputDirectory);
    UNREFERENCED_PARAMETER(AlgorithmId);
    UNREFERENCED_PARAMETER(HashFunctionId);
    UNREFERENCED_PARAMETER(MaskFunctionId);
    UNREFERENCED_PARAMETER(ContextTableCreateFlagsPointer);
    UNREFERENCED_PARAMETER(KeysLoadFlagsPointer);
    UNREFERENCED_PARAMETER(TableCreateFlagsPointer);
    UNREFERENCED_PARAMETER(TableCompileFlagsPointer);
    UNREFERENCED_PARAMETER(TableCreateParameters);

    return PH_E_NOT_IMPLEMENTED;
}

PERFECT_HASH_CONTEXT_EXTRACT_TABLE_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextExtractTableCreateArgsFromArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextExtractTableCreateArgsFromArgvW(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW,
    LPWSTR CommandLineW,
    PUNICODE_STRING KeysPath,
    PUNICODE_STRING BaseOutputDirectory,
    PPERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PPERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PULONG MaximumConcurrency,
    PPERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS ContextTableCreateFlags,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
{
    UNREFERENCED_PARAMETER(Context);
    UNREFERENCED_PARAMETER(NumberOfArguments);
    UNREFERENCED_PARAMETER(ArgvW);
    UNREFERENCED_PARAMETER(CommandLineW);
    UNREFERENCED_PARAMETER(KeysPath);
    UNREFERENCED_PARAMETER(BaseOutputDirectory);
    UNREFERENCED_PARAMETER(AlgorithmId);
    UNREFERENCED_PARAMETER(HashFunctionId);
    UNREFERENCED_PARAMETER(MaskFunctionId);
    UNREFERENCED_PARAMETER(MaximumConcurrency);
    UNREFERENCED_PARAMETER(ContextTableCreateFlags);
    UNREFERENCED_PARAMETER(KeysLoadFlags);
    UNREFERENCED_PARAMETER(TableCreateFlags);
    UNREFERENCED_PARAMETER(TableCompileFlags);
    UNREFERENCED_PARAMETER(TableCreateParameters);

    return PH_E_NOT_IMPLEMENTED;
}

PERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVW PerfectHashContextTableCreateArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextTableCreateArgvW(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW,
    LPWSTR CommandLineW
    )
{
    UNREFERENCED_PARAMETER(Context);
    UNREFERENCED_PARAMETER(NumberOfArguments);
    UNREFERENCED_PARAMETER(ArgvW);
    UNREFERENCED_PARAMETER(CommandLineW);

    return PH_E_NOT_IMPLEMENTED;
}

#ifdef PH_COMPAT
PERFECT_HASH_CONTEXT_TABLE_CREATE_ARGVA PerfectHashContextTableCreateArgvA;

_Use_decl_annotations_
HRESULT
PerfectHashContextTableCreateArgvA(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPSTR *ArgvA
    )
{
    UNREFERENCED_PARAMETER(Context);
    UNREFERENCED_PARAMETER(NumberOfArguments);
    UNREFERENCED_PARAMETER(ArgvA);

    return PH_E_NOT_IMPLEMENTED;
}
#endif

PERFECT_HASH_CONTEXT_BULK_CREATE PerfectHashContextBulkCreate;

_Use_decl_annotations_
HRESULT
PerfectHashContextBulkCreate(
    PPERFECT_HASH_CONTEXT Context,
    PCUNICODE_STRING KeysDirectory,
    PCUNICODE_STRING BaseOutputDirectory,
    PERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlagsPointer,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlagsPointer,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlagsPointer,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
{
    UNREFERENCED_PARAMETER(Context);
    UNREFERENCED_PARAMETER(KeysDirectory);
    UNREFERENCED_PARAMETER(BaseOutputDirectory);
    UNREFERENCED_PARAMETER(AlgorithmId);
    UNREFERENCED_PARAMETER(HashFunctionId);
    UNREFERENCED_PARAMETER(MaskFunctionId);
    UNREFERENCED_PARAMETER(ContextBulkCreateFlagsPointer);
    UNREFERENCED_PARAMETER(KeysLoadFlagsPointer);
    UNREFERENCED_PARAMETER(TableCreateFlagsPointer);
    UNREFERENCED_PARAMETER(TableCompileFlagsPointer);
    UNREFERENCED_PARAMETER(TableCreateParameters);

    return PH_E_NOT_IMPLEMENTED;
}

PERFECT_HASH_CONTEXT_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextExtractBulkCreateArgsFromArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextExtractBulkCreateArgsFromArgvW(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW,
    LPWSTR CommandLineW,
    PUNICODE_STRING KeysDirectory,
    PUNICODE_STRING BaseOutputDirectory,
    PPERFECT_HASH_ALGORITHM_ID AlgorithmId,
    PPERFECT_HASH_HASH_FUNCTION_ID HashFunctionId,
    PPERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId,
    PULONG MaximumConcurrency,
    PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS ContextBulkCreateFlags,
    PPERFECT_HASH_KEYS_LOAD_FLAGS KeysLoadFlags,
    PPERFECT_HASH_TABLE_CREATE_FLAGS TableCreateFlags,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS TableCompileFlags,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
{
    UNREFERENCED_PARAMETER(Context);
    UNREFERENCED_PARAMETER(NumberOfArguments);
    UNREFERENCED_PARAMETER(ArgvW);
    UNREFERENCED_PARAMETER(CommandLineW);
    UNREFERENCED_PARAMETER(KeysDirectory);
    UNREFERENCED_PARAMETER(BaseOutputDirectory);
    UNREFERENCED_PARAMETER(AlgorithmId);
    UNREFERENCED_PARAMETER(HashFunctionId);
    UNREFERENCED_PARAMETER(MaskFunctionId);
    UNREFERENCED_PARAMETER(MaximumConcurrency);
    UNREFERENCED_PARAMETER(ContextBulkCreateFlags);
    UNREFERENCED_PARAMETER(KeysLoadFlags);
    UNREFERENCED_PARAMETER(TableCreateFlags);
    UNREFERENCED_PARAMETER(TableCompileFlags);
    UNREFERENCED_PARAMETER(TableCreateParameters);

    return PH_E_NOT_IMPLEMENTED;
}

PERFECT_HASH_CONTEXT_BULK_CREATE_ARGVW PerfectHashContextBulkCreateArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextBulkCreateArgvW(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPWSTR *ArgvW,
    LPWSTR CommandLineW
    )
{
    UNREFERENCED_PARAMETER(Context);
    UNREFERENCED_PARAMETER(NumberOfArguments);
    UNREFERENCED_PARAMETER(ArgvW);
    UNREFERENCED_PARAMETER(CommandLineW);

    return PH_E_NOT_IMPLEMENTED;
}

#ifdef PH_COMPAT
PERFECT_HASH_CONTEXT_BULK_CREATE_ARGVA PerfectHashContextBulkCreateArgvA;

_Use_decl_annotations_
HRESULT
PerfectHashContextBulkCreateArgvA(
    PPERFECT_HASH_CONTEXT Context,
    ULONG NumberOfArguments,
    LPSTR *ArgvA
    )
{
    UNREFERENCED_PARAMETER(Context);
    UNREFERENCED_PARAMETER(NumberOfArguments);
    UNREFERENCED_PARAMETER(ArgvA);

    return PH_E_NOT_IMPLEMENTED;
}
#endif

#endif // PH_ONLINE_ONLY

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :