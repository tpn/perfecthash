/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    ExtractArg.c

Abstract:

    This module implements the various argument extraction routines.

--*/

#include "stdafx.h"

//
// Helper macro for defining local UNICODE_STRING structures.
//

#ifndef RCS
#define RCS RTL_CONSTANT_STRING
#endif

#define DECL_ARG(Name) const UNICODE_STRING Name = RCS(L#Name)

//
// Helper macro for just the Rtl->RtlEqualUnicodeString() comparison.
//

#define IS_EQUAL(Name) Rtl->RtlEqualUnicodeString(Argument, &Name, TRUE)

//
// Helper macro for Rtl->RtlPrefixUnicodeString().  This is used instead of the
// IS_EQUAL() macro above when the argument may contain an equal sign, e.g.
//
//      --TryLargePagesForKeysData vs --AttemptsBeforeTableResize=100
//

#define IS_PREFIX(Name) Rtl->RtlPrefixUnicodeString(&Name, Argument, TRUE)

//
// Helper macro for toggling the given flag if the current argument matches
// the given UNICODE_STRING.
//

#define SET_FLAG_AND_RETURN_IF_EQUAL(Name) \
    if (IS_EQUAL(Name)) {                  \
        Flags->##Name = TRUE;              \
        return S_OK;                       \
    }

//
// Method implementations.
//

TRY_EXTRACT_ARG_CONTEXT_BULK_CREATE_FLAGS TryExtractArgContextBulkCreateFlags;

_Use_decl_annotations_
HRESULT
TryExtractArgContextBulkCreateFlags(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PCUNICODE_STRING Argument,
    PPERFECT_HASH_CONTEXT_BULK_CREATE_FLAGS Flags
    )
{
    DECL_ARG(SkipTestAfterCreate);
    DECL_ARG(Compile);

    UNREFERENCED_PARAMETER(Allocator);

    SET_FLAG_AND_RETURN_IF_EQUAL(SkipTestAfterCreate);
    SET_FLAG_AND_RETURN_IF_EQUAL(Compile);

    return S_FALSE;
}


TRY_EXTRACT_ARG_CONTEXT_TABLE_CREATE_FLAGS TryExtractArgContextTableCreateFlags;

_Use_decl_annotations_
HRESULT
TryExtractArgContextTableCreateFlags(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PCUNICODE_STRING Argument,
    PPERFECT_HASH_CONTEXT_TABLE_CREATE_FLAGS Flags
    )
{
    DECL_ARG(SkipTestAfterCreate);
    DECL_ARG(Compile);

    UNREFERENCED_PARAMETER(Allocator);

    SET_FLAG_AND_RETURN_IF_EQUAL(SkipTestAfterCreate);
    SET_FLAG_AND_RETURN_IF_EQUAL(Compile);

    return S_FALSE;
}


TRY_EXTRACT_ARG_KEYS_LOAD_FLAGS TryExtractArgKeysLoadFlags;

_Use_decl_annotations_
HRESULT
TryExtractArgKeysLoadFlags(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PCUNICODE_STRING Argument,
    PPERFECT_HASH_KEYS_LOAD_FLAGS Flags
    )
{
    DECL_ARG(TryLargePagesForKeysData);
    DECL_ARG(SkipKeysVerification);

    UNREFERENCED_PARAMETER(Allocator);

    SET_FLAG_AND_RETURN_IF_EQUAL(TryLargePagesForKeysData);
    SET_FLAG_AND_RETURN_IF_EQUAL(SkipKeysVerification);

    return S_FALSE;
}


TRY_EXTRACT_ARG_TABLE_CREATE_FLAGS TryExtractArgTableCreateFlags;

_Use_decl_annotations_
HRESULT
TryExtractArgTableCreateFlags(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PCUNICODE_STRING Argument,
    PPERFECT_HASH_TABLE_CREATE_FLAGS Flags
    )
{
    DECL_ARG(FirstGraphWins);
    DECL_ARG(FindBestGraph);
    DECL_ARG(SkipGraphVerification);
    DECL_ARG(CreateOnly);
    DECL_ARG(TryLargePagesForTableData);
    DECL_ARG(TryLargePagesForValuesArray);
    DECL_ARG(IgnorePreviousTableSize);
    DECL_ARG(IncludeNumberOfTableResizeEventsInOutputPath);
    DECL_ARG(IncludeNumberOfTableElementsInOutputPath);
    DECL_ARG(NoFileIo);
    DECL_ARG(Silent);
    DECL_ARG(Paranoid);
    DECL_ARG(SkipMemoryCoverageInFirstGraphWinsMode);
    DECL_ARG(TryLargePagesForGraphTableData);
    DECL_ARG(TryLargePagesForGraphEdgeAndVertexArrays);
    DECL_ARG(OmitCsvRowIfTableCreateFailed);
    DECL_ARG(OmitCsvRowIfTableCreateSucceeded);

    UNREFERENCED_PARAMETER(Allocator);

    //
    // FirstGraphWins isn't actually a flag as it's the default behavior.
    // However, it's explicitly referenced in the usage string, so, add a
    // test for it that simply returns S_OK.
    //

    if (IS_EQUAL(FirstGraphWins)) {
        return S_OK;
    }

    //
    // Continue with additional flag extraction.
    //

    SET_FLAG_AND_RETURN_IF_EQUAL(FindBestGraph);
    SET_FLAG_AND_RETURN_IF_EQUAL(SkipGraphVerification);
    SET_FLAG_AND_RETURN_IF_EQUAL(CreateOnly);
    SET_FLAG_AND_RETURN_IF_EQUAL(TryLargePagesForTableData);
    SET_FLAG_AND_RETURN_IF_EQUAL(TryLargePagesForValuesArray);
    SET_FLAG_AND_RETURN_IF_EQUAL(IgnorePreviousTableSize);
    SET_FLAG_AND_RETURN_IF_EQUAL(IncludeNumberOfTableResizeEventsInOutputPath);
    SET_FLAG_AND_RETURN_IF_EQUAL(IncludeNumberOfTableElementsInOutputPath);
    SET_FLAG_AND_RETURN_IF_EQUAL(NoFileIo);
    SET_FLAG_AND_RETURN_IF_EQUAL(Silent);
    SET_FLAG_AND_RETURN_IF_EQUAL(Paranoid);
    SET_FLAG_AND_RETURN_IF_EQUAL(SkipMemoryCoverageInFirstGraphWinsMode);
    SET_FLAG_AND_RETURN_IF_EQUAL(TryLargePagesForGraphTableData);
    SET_FLAG_AND_RETURN_IF_EQUAL(TryLargePagesForGraphEdgeAndVertexArrays);
    SET_FLAG_AND_RETURN_IF_EQUAL(OmitCsvRowIfTableCreateFailed);
    SET_FLAG_AND_RETURN_IF_EQUAL(OmitCsvRowIfTableCreateSucceeded);

    return S_FALSE;
}


TRY_EXTRACT_ARG_TABLE_LOAD_FLAGS TryExtractArgTableLoadFlags;

_Use_decl_annotations_
HRESULT
TryExtractArgTableLoadFlags(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PCUNICODE_STRING Argument,
    PPERFECT_HASH_TABLE_LOAD_FLAGS Flags
    )
{
    DECL_ARG(TryLargePagesForTableData);
    DECL_ARG(TryLargePagesForValuesArray);

    UNREFERENCED_PARAMETER(Allocator);

    SET_FLAG_AND_RETURN_IF_EQUAL(TryLargePagesForTableData);
    SET_FLAG_AND_RETURN_IF_EQUAL(TryLargePagesForValuesArray);

    return S_FALSE;
}


TRY_EXTRACT_ARG_TABLE_COMPILE_FLAGS TryExtractArgTableCompileFlags;

_Use_decl_annotations_
HRESULT
TryExtractArgTableCompileFlags(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PCUNICODE_STRING Argument,
    PPERFECT_HASH_TABLE_COMPILE_FLAGS Flags
    )
{
    UNREFERENCED_PARAMETER(Rtl);
    UNREFERENCED_PARAMETER(Allocator);
    DBG_UNREFERENCED_PARAMETER(Argument);
    DBG_UNREFERENCED_PARAMETER(Flags);

    return S_FALSE;
}

typedef enum _EXTRACT_VALUE_ARRAY_STATE {
    LookingForValue,
    LookingForComma,
} EXTRACT_VALUE_ARRAY_STATE;

TRY_EXTRACT_VALUE_ARRAY TryExtractValueArray;

_Use_decl_annotations_
HRESULT
TryExtractValueArray(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PCUNICODE_STRING InputString,
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param,
    BOOLEAN EnsureSortedAndUnique
    )
{
    USHORT Index;
    USHORT NumberOfInputStringChars;
    ULONG Commas = 0;
    ULONG NumberOfValues;
    ULONG NumberOfValidValues;
    PWCHAR Wide;
    PWCHAR ValueStart = NULL;
    PULONG Value;
    PULONG Values = NULL;
    UNICODE_STRING ValueString;
    EXTRACT_VALUE_ARRAY_STATE State;
    BOOLEAN IsLastChar;
    NTSTATUS Status;
    HRESULT Result = S_OK;
    PVALUE_ARRAY ValueArray;
    PRTL_UNICODE_STRING_TO_INTEGER RtlUnicodeStringToInteger;

    //
    // Initialize aliases.
    //

    RtlUnicodeStringToInteger = Rtl->RtlUnicodeStringToInteger;

    //
    // Loop through the string and count the number of commas we see.
    //

    Wide = InputString->Buffer;
    NumberOfInputStringChars = InputString->Length >> 1;

    for (Index = 0; Index < NumberOfInputStringChars; Index++, Wide++) {
        if (*Wide == L',') {
            Commas++;
        }
    }

    if (Commas == 0) {
        return S_FALSE;
    }

    //
    // Allocate memory to contain an array of ULONGs whose length matches the
    // number of commas we saw, plus 1.
    //

    NumberOfValues = Commas + 1;
    Value = Values = Allocator->Vtbl->Calloc(Allocator,
                                             NumberOfValues,
                                             sizeof(ULONG));
    if (!Values) {
        return E_OUTOFMEMORY;
    }

    State = LookingForValue;
    NumberOfValidValues = 0;
    ValueStart = NULL;

    Wide = InputString->Buffer;

    for (Index = 0; Index < NumberOfInputStringChars; Index++, Wide++) {

        IsLastChar = (Index == (NumberOfInputStringChars - 1));

        if (IsLastChar) {
            if (*Wide == L',') {
                break;
            }
            if (!ValueStart) {
                ValueStart = Wide;
            }
            Wide++;
            goto ProcessValue;
        }

        if (State == LookingForValue) {

            ASSERT(ValueStart == NULL);

            if (*Wide != L',') {
                ValueStart = Wide;
                State = LookingForComma;
            }

            continue;

        } else {

            ASSERT(State == LookingForComma);
            ASSERT(ValueStart != NULL);

            if (*Wide != L',') {
                continue;
            }

ProcessValue:

            ValueString.Buffer = ValueStart;
            ValueString.Length = (USHORT)RtlPointerToOffset(ValueStart, Wide);
            ValueString.MaximumLength = ValueString.Length;

            Status = RtlUnicodeStringToInteger(&ValueString, 0, Value);
            if (!SUCCEEDED(Status)) {
                Result = E_FAIL;
                goto Error;
            }

            if (EnsureSortedAndUnique && Value != Values) {
                ULONG Previous;
                ULONG This;

                Previous = *(Value - 1);
                This = *Value;

                if (Previous > This) {
                    Result = PH_E_NOT_SORTED;
                    goto Error;
                } else if (Previous == This) {
                    Result = PH_E_DUPLICATE_DETECTED;
                    goto Error;
                }
            }

            Value++;
            ValueStart = NULL;
            NumberOfValidValues++;
            State = LookingForValue;

            continue;
        }
    }

    if (NumberOfValidValues == 0) {
        Result = E_FAIL;
        goto Error;
    } else if (NumberOfValidValues != NumberOfValues) {
        Result = PH_E_INVARIANT_CHECK_FAILED;
        PH_ERROR(TryExtractValueArray_NumValidValuesMismatch, Result);
        goto Error;
    }

    //
    // We've successfully extracted the array.  Update the parameter and return
    // success.
    //

    ValueArray = &Param->AsValueArray;
    ValueArray->Values = Values;
    ValueArray->NumberOfValues = NumberOfValidValues;
    ValueArray->ValueSizeInBytes = sizeof(ULONG);

    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    if (Values) {
        Allocator->Vtbl->FreePointer(Allocator, &Values);
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}

FORCEINLINE
VOID
MaybeDeallocateTableCreateParameter(
    _In_ PALLOCATOR Allocator,
    _In_ PPERFECT_HASH_TABLE_CREATE_PARAMETER Param
    )
{
    if (DoesTableCreateParameterRequireDeallocation(Param->Id)) {
        Param->Id = PerfectHashNullTableCreateParameterId;
        Allocator->Vtbl->FreePointer(Allocator,
                                     (PVOID *)&Param->AsVoidPointer);
    }
}

//
// Processing table create parameters is more involved than the flag-oriented
// routines above as we need to allocate or reallocate the parameters array
// when we find a match.
//

TRY_EXTRACT_ARG_TABLE_CREATE_PARAMETERS TryExtractArgTableCreateParameters;

_Use_decl_annotations_
HRESULT
TryExtractArgTableCreateParameters(
    PRTL Rtl,
    PCUNICODE_STRING Argument,
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    This routine tries to extract table create parameters from a given argument
    string.

    Table create parameters must adhere to the format "<Name>=<Value>"; i.e.
    an equal sign must be present and a valid value must trail it.

Arguments:

    Rtl - Supplies a pointer to an Rtl instance.

    Allocator - Supplies a pointer to an Allocator instance that will be used
        for all memory allocations.

        N.B. The same allocator should always be passed when parsing a command
             line string, as previous allocations may need to be reallocated if
             additional table create parameters are detected.

    Argument - Supplies a pointer to a UNICODE_STRING structure capturing the
        argument to process.  The string should *not* include leading dashes,
        e.g. if the command line argument was "--Seeds=0,0,15" the Argument
        parameter should be adjusted to point to "Seeds=0,0,15" by the time
        this routine is called.

    TableCreateParameters - Supplies a pointer to the table create parameters
        structure to be used for capturing params.

Return Value:

    S_OK - A table create parameter was successfully extracted.

    S_FALSE - No table create parameter was detected.

    PH_E_COMMANDLINE_ARG_MISSING_VALUE - A table create parameter or best
        coverage type was detected, however, no equal sign or value was
        detected.

    E_OUTOFMEMORY - Out of memory.

    PH_E_INVALID_TABLE_CREATE_PARAMETERS - Invalid table create parameters.

    PH_E_KEYS_SUBSET_NOT_SORTED - The keys subset list of values was not sorted
        in ascending order.

    PH_E_DUPLICATE_VALUE_DETECTED_IN_KEYS_SUBSET - Duplicate value detected in
        keys subset.

    PH_E_INVALID_KEYS_SUBSET - Invalid keys subset value provided.

    PH_E_INVALID_MAIN_WORK_THREADPOOL_PRIORITY - Invalid main work threadpool
        priority.

    PH_E_INVALID_FILE_WORK_THREADPOOL_PRIORITY - Invalid file work threadpool
        priority.

    PH_E_INVALID_SEEDS - Invalid seeds.

--*/
{
    USHORT Count;
    USHORT Index;
    ULONG Value = 0;
    PWSTR Source;
    PALLOCATOR Allocator;
    BOOLEAN ValueIsInteger = FALSE;
    BOOLEAN EqualSignFound = FALSE;
    BOOLEAN TableParamFound = FALSE;
    HRESULT Result = S_FALSE;
    UNICODE_STRING Temp = { 0 };
    PUNICODE_STRING ValueString;
    ULONG NumberOfTableCreateParameters;
    PERFECT_HASH_TABLE_CREATE_PARAMETER LocalParam;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER NewParams;

    //
    // Use two X-macro expansions for declaring local variables for the table
    // create parameters and best coverage types.
    //

#define EXPAND_AS_TABLE_PARAM_DECL_ARG(Name) \
    DECL_ARG(Name);

    TABLE_CREATE_PARAMETER_TABLE_ENTRY(EXPAND_AS_TABLE_PARAM_DECL_ARG);

#define EXPAND_AS_BEST_COVERAGE_DECL_ARG(Name, Comparison, Comparator) \
    DECL_ARG(Comparison##Name);

    BEST_COVERAGE_TYPE_TABLE_ENTRY(EXPAND_AS_BEST_COVERAGE_DECL_ARG)

    //
    // Declare local variables for the threadpool priority values.
    //

    DECL_ARG(High);
    DECL_ARG(Normal);
    DECL_ARG(Low);

    //
    // Invariant check: if number of table create parameters is 0, the array
    // pointer should be null, and vice versa.
    //

    NumberOfTableCreateParameters = TableCreateParameters->NumberOfElements;
    if (NumberOfTableCreateParameters == 0) {
        if (TableCreateParameters->Params != NULL) {
            return PH_E_INVALID_TABLE_CREATE_PARAMETERS;
        }
    } else {
        if (TableCreateParameters->Params == NULL) {
            return PH_E_INVALID_TABLE_CREATE_PARAMETERS;
        }
    }

    Allocator = TableCreateParameters->Allocator;

    ZeroStructInline(LocalParam);

    //
    // Find where the equals sign occurs, if at all.
    //

    Source = Argument->Buffer;
    Count = Argument->Length >> 1;

    for (Index = 0; Index < Count; Index++) {
        if (*Source == L'=') {
            Source++;
            EqualSignFound = TRUE;
            break;
        }
        Source++;
    }

    //
    // Initially, when this routine was first written, it didn't report an
    // error if a table create parameter or best coverage type name was
    // detected but no equal sign was present.  We've got a sloppy fix in
    // place for now: we check the incoming argument string against known
    // parameters and set a boolean flag if we find a match.  Then, if no
    // equal sign is found but we've found a param match, we can report this
    // error code back to the user in a more friendly manner.
    //
    // It's sloppy as we then repeat all the string comparisons later as part
    // of the macros (e.g. ADD_PARAM_IF_PREFIX_AND_VALUE_IS_INTEGER()).
    //
    // This routine is only called at the start of the program in order to
    // parse command line arguments, so, eh, we can live with some sloppiness
    // for now.
    //

    //
    // Determine if the incoming argument matches any of our known parameters.
    //

#define EXPAND_AS_IS_TABLE_CREATE_PARAM(Name) \
    if (IS_PREFIX(Name)) {                    \
        TableParamFound = TRUE;               \
        break;                                \
    }

    do {
        TABLE_CREATE_PARAMETER_TABLE_ENTRY(EXPAND_AS_IS_TABLE_CREATE_PARAM)
    } while (0);

    if (!EqualSignFound) {

        //
        // No equal sign was found.  If a table create parameter *was* found,
        // report an error, otherwise, return S_FALSE, indicating that no param
        // was detected.
        //

        if (TableParamFound) {
            return PH_E_COMMANDLINE_ARG_MISSING_VALUE;
        } else {
            return S_FALSE;
        }
    }

    ASSERT(*(Source - 1) == L'=');
    ASSERT(Argument->Buffer[Index] == L'=');

    //
    // Wire up the ValueString variable to point immediately after the equals
    // sign.  The +1 in (Index + 1) accounts for (subtracts) the NULL.
    //

    ValueString = &Temp;
    ValueString->Buffer = Source;
    ValueString->Length = Argument->Length - ((Index + 1) << 1);
    ValueString->MaximumLength = ValueString->Length + 2;

    ASSERT(ValueString->Buffer[ValueString->Length >> 1] == L'\0');

    //
    // Attempt to convert the value string into an integer representation.
    //

    Result = Rtl->RtlUnicodeStringToInteger(ValueString, 10, &Value);
    if (SUCCEEDED(Result)) {
        ValueIsInteger = TRUE;
    }

#define SET_PARAM_ID(Name)                         \
    LocalParam.Id = TableCreateParameter##Name##Id

#define ADD_PARAM_IF_PREFIX_AND_VALUE_IS_INTEGER(Name) \
    if (IS_PREFIX(Name) && ValueIsInteger) {           \
        SET_PARAM_ID(Name);                            \
        LocalParam.AsULong = Value;                    \
        goto AddParam;                                 \
    }

    ADD_PARAM_IF_PREFIX_AND_VALUE_IS_INTEGER(AttemptsBeforeTableResize);

    ADD_PARAM_IF_PREFIX_AND_VALUE_IS_INTEGER(MaxNumberOfTableResizes);

    ADD_PARAM_IF_PREFIX_AND_VALUE_IS_INTEGER(BestCoverageAttempts);

#define IS_VALUE_EQUAL(ValueName) \
    Rtl->RtlEqualUnicodeString(ValueString, &ValueName, TRUE)

#define ADD_PARAM_IF_PREFIX_AND_VALUE_EQUAL(Name, ValueName) \
    if (IS_PREFIX(Name) && IS_VALUE_EQUAL(ValueName)) {      \
        SET_PARAM_ID(Name);                                  \
        LocalParam.AsULongLong = Name##ValueName##Id;        \
        goto AddParam;                                       \
    }

#define EXPAND_AS_ADD_PARAM(Name, Comparison, Comparator)  \
    ADD_PARAM_IF_PREFIX_AND_VALUE_EQUAL(BestCoverageType,  \
                                        Comparison##Name);

    BEST_COVERAGE_TYPE_TABLE_ENTRY(EXPAND_AS_ADD_PARAM);

#define ADD_PARAM_IF_PREFIX_AND_VALUE_IS_CSV_OF_ASCENDING_INTEGERS(Name,  \
                                                                   Upper) \
    if (IS_PREFIX(Name)) {                                                \
        Result = TryExtractValueArray(Rtl,                                \
                                      Allocator,                          \
                                      ValueString,                        \
                                      &LocalParam,                        \
                                      TRUE);                              \
                                                                          \
        if (Result == S_OK) {                                             \
            SET_PARAM_ID(Name);                                           \
            goto AddParam;                                                \
        } else {                                                          \
            if (Result == PH_E_NOT_SORTED) {                              \
                Result = PH_E_##Upper##_NOT_SORTED;                       \
            } else if (Result == PH_E_DUPLICATE_DETECTED) {               \
                Result = PH_E_DUPLICATE_VALUE_DETECTED_IN_##Upper;        \
            } else if (Result != E_OUTOFMEMORY) {                         \
                Result = PH_E_INVALID_##Upper;                            \
            }                                                             \
            goto Error;                                                   \
        }                                                                 \
    }

    ADD_PARAM_IF_PREFIX_AND_VALUE_IS_CSV_OF_ASCENDING_INTEGERS(KeysSubset,
                                                               KEYS_SUBSET);

#define ADD_PARAM_IF_PREFIX_AND_VALUE_IS_CSV(Name, Upper) \
    if (IS_PREFIX(Name)) {                                \
        Result = TryExtractValueArray(Rtl,                \
                                      Allocator,          \
                                      ValueString,        \
                                      &LocalParam,        \
                                      FALSE);             \
                                                          \
        if (Result == S_OK) {                             \
            SET_PARAM_ID(Name);                           \
            goto AddParam;                                \
        } else {                                          \
            if (Result != E_OUTOFMEMORY) {                \
                Result = PH_E_INVALID_##Upper;            \
            }                                             \
            goto Error;                                   \
        }                                                 \
    }

    ADD_PARAM_IF_PREFIX_AND_VALUE_IS_CSV(Seeds, SEEDS);

#define ADD_PARAM_IF_PREFIX_AND_VALUE_IS_TP_PRIORITY(Name, Upper)          \
    if (IS_PREFIX(Name##ThreadpoolPriority)) {                             \
        if (IS_VALUE_EQUAL(High)) {                                        \
            SET_PARAM_ID(Name##ThreadpoolPriority);                        \
            LocalParam.AsTpCallbackPriority = TP_CALLBACK_PRIORITY_HIGH;   \
            goto AddParam;                                                 \
        } else if (IS_VALUE_EQUAL(Normal)) {                               \
            SET_PARAM_ID(Name##ThreadpoolPriority);                        \
            LocalParam.AsTpCallbackPriority = TP_CALLBACK_PRIORITY_NORMAL; \
            goto AddParam;                                                 \
        } else if (IS_VALUE_EQUAL(Low)) {                                  \
            SET_PARAM_ID(Name##ThreadpoolPriority);                        \
            LocalParam.AsTpCallbackPriority = TP_CALLBACK_PRIORITY_LOW;    \
            goto AddParam;                                                 \
        } else {                                                           \
            Result = PH_E_INVALID_##Upper##_THREADPOOL_PRIORITY;           \
            goto Error;                                                    \
        }                                                                  \
    }

    ADD_PARAM_IF_PREFIX_AND_VALUE_IS_TP_PRIORITY(MainWork, MAIN_WORK);
    ADD_PARAM_IF_PREFIX_AND_VALUE_IS_TP_PRIORITY(FileWork, FILE_WORK);

    //
    // We shouldn't ever get here; we've already determined that a valid param
    // was detected earlier in the routine.
    //

    Result = PH_E_UNREACHABLE_CODE;
    PH_ERROR(TryExtractArgTableCreateParameters, Result);
    PH_RAISE(Result);

AddParam:

    //
    // If there are already table create parameters, we may have already seen
    // a parameter with the same ID.  Loop through all the existing ones and
    // see if we can find a match.  If so, update the value directly and return;
    // avoiding the allocation/reallocation logic altogether.
    //

    Param = TableCreateParameters->Params;

    for (Index = 0; Index < NumberOfTableCreateParameters; Index++, Param++) {
        if (Param->Id == LocalParam.Id) {

            //
            // Make sure we potentially deallocate the existing param if
            // applicable.
            //

            MaybeDeallocateTableCreateParameter(Allocator, Param);

            Param->AsULongLong = LocalParam.AsULongLong;
            return S_OK;
        }
    }

    //
    // If no parameters have been allocated yet, allocate from scratch.
    // Otherwise, reallocate.
    //

    if (NumberOfTableCreateParameters == 0) {

        ASSERT(TableCreateParameters->Params == NULL);

        NewParams = Allocator->Vtbl->Calloc(Allocator, 1, sizeof(*Param));

    } else {

        ASSERT(TableCreateParameters->Params != NULL);

        NewParams = (
            Allocator->Vtbl->ReCalloc(
                Allocator,
                TableCreateParameters->Params,
                (ULONG_PTR)NumberOfTableCreateParameters + 1,
                sizeof(*Param)
            )
        );

    }

    if (!NewParams) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // Write our new parameter to the end of the array.
    //

    Param = &NewParams[NumberOfTableCreateParameters];
    CopyMemory(Param, &LocalParam, sizeof(*Param));

    //
    // Update the structure and finish up.
    //

    TableCreateParameters->NumberOfElements++;
    TableCreateParameters->Params = NewParams;

    Result = S_OK;
    goto End;

Error:

    if (Result == S_FALSE) {
        Result = E_UNEXPECTED;
    }

    //
    // Intentional follow-on to End.
    //

End:

    return Result;
}


CLEANUP_TABLE_CREATE_PARAMETERS CleanupTableCreateParameters;

_Use_decl_annotations_
HRESULT
CleanupTableCreateParameters(
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters
    )
/*++

Routine Description:

    Walks the array of individual table create parameter pointers and, if
    applicable for the given parameter type, releases any memory that was
    allocated for the parameter.

Arguments:

    TableCreateParameters - Supplies a pointer to the table create parameters
        structure for which the cleanup is to be performed.

Return Value:

    S_OK - Success.

    E_POINTER - TableCreateParameters was NULL.

    PH_E_INVALID_TABLE_CREATE_PARAMETERS - Invalid table create parameters.

--*/
{
    ULONG Index;
    ULONG Count;
    PALLOCATOR Allocator;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Params;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(TableCreateParameters)) {
        return E_POINTER;
    }

    Count = TableCreateParameters->NumberOfElements;
    Param = Params = TableCreateParameters->Params;

    //
    // Invariant checks: if count is 0, params should be NULL, and vice versa.
    //

    if (Count == 0) {

        if (Params != NULL) {
            return PH_E_INVALID_TABLE_CREATE_PARAMETERS;
        }

        //
        // There are no parameters, nothing to clean up.
        //

        goto End;
    }

    if (Params == NULL) {
        return PH_E_INVALID_TABLE_CREATE_PARAMETERS;
    }

    //
    // Argument validation complete; there is at least one or more table create
    // parameter that needs to be checked for potential deallocation.
    //

    Allocator = TableCreateParameters->Allocator;

    for (Index = 0; Index < Count; Index++, Param++) {
        MaybeDeallocateTableCreateParameter(Allocator, Param);
    }

    Allocator->Vtbl->FreePointer(Allocator, &TableCreateParameters->Params);

    //
    // Intentional follow-on to End.
    //

End:

    //
    // Explicitly clear the number of elements and params pointer before
    // returning success to the caller.
    //

    TableCreateParameters->NumberOfElements = 0;
    TableCreateParameters->Params = NULL;

    return S_OK;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
