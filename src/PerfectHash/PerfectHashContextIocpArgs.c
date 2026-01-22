/*++

Copyright (c) 2018-2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashContextIocpArgs.c

Abstract:

    This module implements IOCP-native argument extraction helpers for
    bulk-create and table-create operations.  These routines mirror the
    legacy context parsing logic but operate directly on the IOCP context's
    RTL/allocator, avoiding the legacy context dependency.

--*/

#include "stdafx.h"

#define GET_LENGTH(Name) \
    ((USHORT)wcslen((Name)->Buffer) * (USHORT)sizeof(WCHAR))
#define GET_MAX_LENGTH(Name) ((Name)->Length + sizeof(WCHAR))

#define VALIDATE_ID(Name, Upper)                                       \
    if (FAILED(Rtl->RtlUnicodeStringToInteger(String,                  \
                                              10,                      \
                                              (PULONG)Name##Id))) {    \
        return PH_E_INVALID_##Upper##_ID;                              \
    } else if (*Name##Id == 0) {                                       \
        Result = PerfectHashLookupIdForName(Rtl,                       \
                                            PerfectHash##Name##EnumId, \
                                            String,                    \
                                            (PULONG)Name##Id);         \
        if (FAILED(Result)) {                                          \
            return PH_E_INVALID_##Upper##_ID;                          \
        }                                                              \
    }                                                                  \
    if (!IsValidPerfectHash##Name##Id(*Name##Id)) {                    \
        return PH_E_INVALID_##Upper##_ID;                              \
    }

#define EXTRACT_ID(Name, Upper)                     \
    CurrentArg++;                                   \
    String->Buffer = *ArgW++;                       \
    String->Length = GET_LENGTH(String);            \
    String->MaximumLength = GET_MAX_LENGTH(String); \
    VALIDATE_ID(Name, Upper)

PERFECT_HASH_CONTEXT_IOCP_EXTRACT_BULK_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextIocpExtractBulkCreateArgsFromArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpExtractBulkCreateArgsFromArgvW(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
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
/*++

Routine Description:

    Extracts arguments for the bulk-create functionality from an argument
    vector array, typically obtained from a commandline invocation.

Arguments:

    ContextIocp - Supplies a pointer to the PERFECT_HASH_CONTEXT_IOCP instance
        for which the arguments are to be extracted.

    NumberOfArguments - Supplies the number of elements in the ArgvW array.

    ArgvW - Supplies a pointer to an array of wide C string arguments.

    CommandLineW - Supplies a pointer to the original command line used to
        construct the ArgvW array above.  This is only used for inclusion in
        things like CSV output; it is not used programmatically (and is not
        checked for correctness against ArgvW).

    KeysDirectory - Supplies a pointer to a UNICODE_STRING structure that
        will be filled out with the keys directory.

    BaseOutputDirectory - Supplies a pointer to a UNICODE_STRING structure that
        will be filled out with the output directory.

    AlgorithmId - Supplies the address of a variable that will receive the
        algorithm ID.

    HashFunctionId - Supplies the address of a variable that will receive the
        hash function ID.

    MaskFunctionId - Supplies the address of a variable that will receive the
        mask function ID.

    MaximumConcurrency - Supplies the address of a variable that will receive
        the maximum concurrency.

    ContextBulkCreateFlags - Supplies the address of a variable that will
        receive the bulk-create flags.

    KeysLoadFlags - Supplies the address of a variable that will receive the
        keys load flags.

    TableCreateFlags - Supplies the address of a variable that will receive
        the table create flags.

    TableCompileFlags - Supplies the address of a variable that will receive
        the table compile flags.

    TableCreateParameters - Supplies the address of a variable that will
        receive a pointer to a table create parameters structure.

Return Value:

    S_OK - Arguments extracted successfully.

    E_POINTER - One or more mandatory parameters were NULL pointers.

    PH_E_CONTEXT_BULK_CREATE_INVALID_NUM_ARGS - Invalid number of arguments.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_MAXIMUM_CONCURRENCY - Invalid maximum concurrency.

--*/
{
    PRTL Rtl;
    LPWSTR *ArgW;
    LPWSTR Arg;
    HRESULT Result = S_OK;
    HRESULT CleanupResult;
    ULONG CurrentArg = 1;
    PALLOCATOR Allocator;
    UNICODE_STRING Temp;
    PUNICODE_STRING String;
    BOOLEAN InvalidPrefix;
    BOOLEAN ValidNumberOfArguments;

    String = &Temp;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(ArgvW)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(CommandLineW)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(KeysDirectory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(BaseOutputDirectory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(AlgorithmId)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(HashFunctionId)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(MaskFunctionId)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(MaximumConcurrency)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(ContextBulkCreateFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(KeysLoadFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCreateFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCompileFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCreateParameters)) {
        return E_POINTER;
    }

    ValidNumberOfArguments = (NumberOfArguments >= 7);

    if (!ValidNumberOfArguments) {
        return PH_E_CONTEXT_BULK_CREATE_INVALID_NUM_ARGS;
    }

    //
    // Argument validation complete, continue.
    //

    ArgW = &ArgvW[1];
    Rtl = ContextIocp->Rtl;
    Allocator = ContextIocp->Allocator;

    if (!Rtl || !Allocator) {
        return E_UNEXPECTED;
    }

    //
    // The first six arguments (keys directory, base output directory, algo ID,
    // hash function ID, mask function ID and maximum concurrency) are special
    // in that they're mandatory and expected to appear sequentially, prior to
    // any additional arguments (i.e. table create parameters) appearing.
    //

    //
    // Extract keys directory.
    //

    CurrentArg++;
    KeysDirectory->Buffer = *ArgW++;
    KeysDirectory->Length = GET_LENGTH(KeysDirectory);
    KeysDirectory->MaximumLength = GET_MAX_LENGTH(KeysDirectory);

    //
    // Extract base output directory.
    //

    CurrentArg++;
    BaseOutputDirectory->Buffer = *ArgW++;
    BaseOutputDirectory->Length = GET_LENGTH(BaseOutputDirectory);
    BaseOutputDirectory->MaximumLength = GET_MAX_LENGTH(BaseOutputDirectory);

    //
    // Extract algo, hash function and mask function IDs.
    //

    EXTRACT_ID(Algorithm, ALGORITHM);
    EXTRACT_ID(HashFunction, HASH_FUNCTION);
    EXTRACT_ID(MaskFunction, MASK_FUNCTION);

    //
    // Extract maximum concurrency.
    //

    CurrentArg++;
    String->Buffer = *ArgW++;
    String->Length = GET_LENGTH(String);
    String->MaximumLength = GET_MAX_LENGTH(String);

    if (FAILED(Rtl->RtlUnicodeStringToInteger(String,
                                              10,
                                              MaximumConcurrency))) {
        return PH_E_INVALID_MAXIMUM_CONCURRENCY;
    }

    //
    // Zero all flags (except for table create flags, as these may have
    // default values) and table create parameters.
    //

    KeysLoadFlags->AsULong = 0;
    TableCompileFlags->AsULong = 0;
    ContextBulkCreateFlags->AsULong = 0;

    for (; CurrentArg < NumberOfArguments; CurrentArg++, ArgW++) {

        String->Buffer = Arg = *ArgW;
        String->Length = GET_LENGTH(String);
        String->MaximumLength = GET_MAX_LENGTH(String);

        //
        // If the argument doesn't start with two dashes, report it.
        //

        InvalidPrefix = (
            (String->Length <= (sizeof(L'-') + sizeof(L'-'))) ||
            (!(*Arg++ == L'-' && *Arg++ == L'-'))
        );

        if (InvalidPrefix) {
            goto InvalidArg;
        }

        //
        // Advance the buffer past the two dashes and update lengths
        // accordingly.
        //

        String->Buffer += 2;
        String->Length -= (sizeof(WCHAR) * 2);
        String->MaximumLength -= (sizeof(WCHAR) * 2);

        //
        // Try each argument extraction routine for this argument; if it
        // indicates an error, report it and break out of the loop.  If it
        // indicates it successfully extracted the argument (Result == S_OK),
        // continue onto the next argument.  Otherwise, verify it indicates
        // that no argument was extracted (S_FALSE), then try the next routine.
        //

#define TRY_EXTRACT_ARG(Name)                                                 \
    Result = TryExtractArg##Name##Flags(Rtl, Allocator, String, Name##Flags); \
    if (FAILED(Result)) {                                                     \
        PH_ERROR(ExtractBulkCreateArgs_TryExtractArg##Name##Flags, Result);   \
        break;                                                                \
    } else if (Result == S_OK) {                                              \
        continue;                                                             \
    } else {                                                                  \
        ASSERT(Result == S_FALSE);                                            \
    }

        TRY_EXTRACT_ARG(ContextBulkCreate);
        TRY_EXTRACT_ARG(KeysLoad);
        TRY_EXTRACT_ARG(TableCreate);
        TRY_EXTRACT_ARG(TableCompile);

        //
        // If we get here, none of the previous extraction routines claimed the
        // argument, so, provide the table create parameters extraction routine
        // an opportunity to run.
        //

        Result = TryExtractArgTableCreateParameters(Rtl,
                                                    String,
                                                    TableCreateParameters);

        if (FAILED(Result)) {
            PH_ERROR(ExtractBulkCreateArgs_TryExtractTableCreateParams, Result);
            break;
        }

        if (Result == S_OK) {
            continue;
        }

        if (Result == PH_E_COMMANDLINE_ARG_MISSING_VALUE) {
            PH_MESSAGE_ARGS(Result, String);
            break;
        }

        if (FAILED(Result)) {
            PH_ERROR(ExtractBulkCreateArgs_TryExtractTableCreateParams, Result);
            break;
        }

        ASSERT(Result == S_FALSE);

InvalidArg:

        //
        // If we get here, we don't recognize the argument.
        //

        Result = PH_E_INVALID_COMMANDLINE_ARG;
        PH_MESSAGE_ARGS(Result, String);
        break;
    }

    //
    // If we failed, clean up the table create parameters.  If that fails,
    // report the error, then replace our return value error code with that
    // error code.
    //

    if (FAILED(Result)) {
        CleanupResult = CleanupTableCreateParameters(TableCreateParameters);
        if (FAILED(CleanupResult)) {
            PH_ERROR(CleanupTableCreateParameters, CleanupResult);
            Result = CleanupResult;
        }
    }

    return Result;
}

#undef TRY_EXTRACT_ARG

PERFECT_HASH_CONTEXT_IOCP_EXTRACT_TABLE_CREATE_ARGS_FROM_ARGVW
    PerfectHashContextIocpExtractTableCreateArgsFromArgvW;

_Use_decl_annotations_
HRESULT
PerfectHashContextIocpExtractTableCreateArgsFromArgvW(
    PPERFECT_HASH_CONTEXT_IOCP ContextIocp,
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
/*++

Routine Description:

    Extracts arguments for the table create functionality from an argument
    vector array, typically obtained from a commandline invocation.

Arguments:

    ContextIocp - Supplies a pointer to the PERFECT_HASH_CONTEXT_IOCP instance
        for which the arguments are to be extracted.

    NumberOfArguments - Supplies the number of elements in the ArgvW array.

    ArgvW - Supplies a pointer to an array of wide C string arguments.

    CommandLineW - Supplies a pointer to the original command line used to
        construct the ArgvW array above.  This is only used for inclusion in
        things like CSV output; it is not used programmatically (and is not
        checked for correctness against ArgvW).

    KeysPath - Supplies a pointer to a UNICODE_STRING structure that will be
        filled out with the keys path.

    BaseOutputDirectory - Supplies a pointer to a UNICODE_STRING structure that
        will be filled out with the output directory.

    AlgorithmId - Supplies the address of a variable that will receive the
        algorithm ID.

    HashFunctionId - Supplies the address of a variable that will receive the
        hash function ID.

    MaskFunctionId - Supplies the address of a variable that will receive the
        mask function ID.

    MaximumConcurrency - Supplies the address of a variable that will receive
        the maximum concurrency.

    ContextTableCreateFlags - Supplies the address of a variable that will
        receive the context table-create flags.

    KeysLoadFlags - Supplies the address of a variable that will receive the
        keys load flags.

    TableCreateFlags - Supplies the address of a variable that will receive
        the table create flags.

    TableCompileFlags - Supplies the address of a variable that will receive
        the table compile flags.

    TableCreateParameters - Supplies the address of a table create params
        structure that will receive any extracted params.

Return Value:

    S_OK - Arguments extracted successfully.

    E_POINTER - One or more mandatory parameters were NULL pointers.

    PH_E_CONTEXT_TABLE_CREATE_INVALID_NUM_ARGS - Invalid number of arguments.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_MAXIMUM_CONCURRENCY - Invalid maximum concurrency.

--*/
{
    PRTL Rtl;
    LPWSTR *ArgW;
    LPWSTR Arg;
    HRESULT Result = S_OK;
    HRESULT CleanupResult;
    HRESULT DebuggerResult;
    ULONG CurrentArg = 1;
    PALLOCATOR Allocator;
    UNICODE_STRING Temp;
    PUNICODE_STRING String;
    BOOLEAN InvalidPrefix;
    DEBUGGER_CONTEXT_FLAGS Flags;
    BOOLEAN ValidNumberOfArguments;
    PDEBUGGER_CONTEXT DebuggerContext;

    String = &Temp;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(ContextIocp)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(ArgvW)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(CommandLineW)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(KeysPath)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(BaseOutputDirectory)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(AlgorithmId)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(HashFunctionId)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(MaskFunctionId)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(MaximumConcurrency)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(ContextTableCreateFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(KeysLoadFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCreateFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCompileFlags)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(TableCreateParameters)) {
        return E_POINTER;
    }

    ValidNumberOfArguments = (NumberOfArguments >= 7);

    if (!ValidNumberOfArguments) {
        return PH_E_CONTEXT_TABLE_CREATE_INVALID_NUM_ARGS;
    }

    //
    // Argument validation complete, continue.
    //

    ArgW = &ArgvW[1];
    Rtl = ContextIocp->Rtl;
    Allocator = ContextIocp->Allocator;

    if (!Rtl || !Allocator) {
        return E_UNEXPECTED;
    }

    //
    // Extract keys path.
    //

    CurrentArg++;
    KeysPath->Buffer = *ArgW++;
    KeysPath->Length = GET_LENGTH(KeysPath);
    KeysPath->MaximumLength = GET_MAX_LENGTH(KeysPath);

    //
    // Extract base output directory.
    //

    CurrentArg++;
    BaseOutputDirectory->Buffer = *ArgW++;
    BaseOutputDirectory->Length = GET_LENGTH(BaseOutputDirectory);
    BaseOutputDirectory->MaximumLength = GET_MAX_LENGTH(BaseOutputDirectory);

    //
    // Extract algorithm ID, hash function and mask function.
    //

    EXTRACT_ID(Algorithm, ALGORITHM);
    EXTRACT_ID(HashFunction, HASH_FUNCTION);
    EXTRACT_ID(MaskFunction, MASK_FUNCTION);

    //
    // Extract maximum concurrency.
    //

    CurrentArg++;
    String->Buffer = *ArgW++;
    String->Length = GET_LENGTH(String);
    String->MaximumLength = GET_MAX_LENGTH(String);

    if (FAILED(Rtl->RtlUnicodeStringToInteger(String,
                                              10,
                                              MaximumConcurrency))) {
        return PH_E_INVALID_MAXIMUM_CONCURRENCY;
    }

    //
    // Zero all flags (except for table create flags, as these may have
    // default values) and table create parameters.
    //

    KeysLoadFlags->AsULong = 0;
    TableCompileFlags->AsULong = 0;

    for (; CurrentArg < NumberOfArguments; CurrentArg++, ArgW++) {

        String->Buffer = Arg = *ArgW;
        String->Length = GET_LENGTH(String);
        String->MaximumLength = GET_MAX_LENGTH(String);

        //
        // If the argument doesn't start with two dashes, report it.
        //

        InvalidPrefix = (
            (String->Length <= (sizeof(L'-') + sizeof(L'-'))) ||
            (!(*Arg++ == L'-' && *Arg++ == L'-'))
        );

        if (InvalidPrefix) {
            goto InvalidArg;
        }

        //
        // Advance the buffer past the two dashes and update lengths
        // accordingly.
        //

        String->Buffer += 2;
        String->Length -= (sizeof(WCHAR) * 2);
        String->MaximumLength -= (sizeof(WCHAR) * 2);

        //
        // Try each argument extraction routine for this argument; if it
        // indicates an error, report it and break out of the loop.  If it
        // indicates it successfully extracted the argument (Result == S_OK),
        // continue onto the next argument.  Otherwise, verify it indicates
        // that no argument was extracted (S_FALSE), then try the next routine.
        //

#define TRY_EXTRACT_ARG(Name)                                                 \
    Result = TryExtractArg##Name##Flags(Rtl, Allocator, String, Name##Flags); \
    if (FAILED(Result)) {                                                     \
        PH_ERROR(ExtractTableCreateArgs_TryExtractArg##Name##Flags, Result);  \
        break;                                                                \
    } else if (Result == S_OK) {                                              \
        continue;                                                             \
    } else {                                                                  \
        ASSERT(Result == S_FALSE);                                            \
    }

        TRY_EXTRACT_ARG(ContextTableCreate);
        TRY_EXTRACT_ARG(KeysLoad);
        TRY_EXTRACT_ARG(TableCreate);
        TRY_EXTRACT_ARG(TableCompile);

        //
        // If we get here, none of the previous extraction routines claimed the
        // argument, so, provide the table create parameters extraction routine
        // an opportunity to run.
        //

        Result = TryExtractArgTableCreateParameters(Rtl,
                                                    String,
                                                    TableCreateParameters);

        if (Result == S_OK) {
            continue;
        }

        if (Result == PH_E_COMMANDLINE_ARG_MISSING_VALUE) {
            PH_MESSAGE_ARGS(Result, String);
            break;
        }

        if (FAILED(Result)) {
            PH_ERROR(ExtractBulkCreateArgs_TryExtractTableCreateParams, Result);
            break;
        }

        ASSERT(Result == S_FALSE);

InvalidArg:

        //
        // If we get here, we don't recognize the argument.
        //

        Result = PH_E_INVALID_COMMANDLINE_ARG;
        PH_MESSAGE_ARGS(Result, String);
        break;
    }

    if (SUCCEEDED(Result)) {

        //
        // Initialize the debugger flags from the table create flags, initialize
        // the debugger context, then, maybe wait for a debugger attach.  This
        // is a no-op on Windows, or if no debugger has been requested.
        //

        Flags.AsULong = 0;
        Flags.WaitForGdb = (TableCreateFlags->WaitForGdb != FALSE);
        Flags.WaitForCudaGdb = (TableCreateFlags->WaitForCudaGdb != FALSE);
        Flags.UseGdbForHostDebugging = (
            TableCreateFlags->UseGdbForHostDebugging != FALSE
        );

        DebuggerContext = &Rtl->DebuggerContext;
        DebuggerResult = InitializeDebuggerContext(DebuggerContext, &Flags);

        if (FAILED(DebuggerResult)) {

            PH_ERROR(InitializeDebuggerContext, DebuggerResult);

            //
            // *Now* we can propagate the debugger result back as the primary
            // result, which ensures the cleanup code below runs.
            //

            Result = DebuggerResult;

        } else {

            //
            // Debugger context was successfully initialized, so, maybe wait for
            // a debugger to attach (depending on what flags were supplied).
            //

            Result = MaybeWaitForDebuggerAttach(DebuggerContext);
            if (FAILED(Result)) {
                PH_ERROR(MaybeWaitForDebuggerAttach, Result);
            }

        }
    }

    //
    // If we failed, clean up the table create parameters.  If that fails,
    // report the error, then replace our return value error code with that
    // error code.
    //

    if (FAILED(Result)) {
        CleanupResult = CleanupTableCreateParameters(TableCreateParameters);
        if (FAILED(CleanupResult)) {
            PH_ERROR(CleanupTableCreateParameters, CleanupResult);
            Result = CleanupResult;
        }
    }

    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
