/*++

Copyright (c) 2018-2020 Trent Nelson <trent@trent.me>

Module Name:

    ExtractArg.c

Abstract:

    This module implements the various argument extraction routines.

--*/

#include "stdafx.h"

//
// Forward decls.
//

double wstrtod(wchar_t *string, wchar_t **endPtr);

//
// Helper macro for defining local UNICODE_STRING structures.
//

#ifndef RCS
#define RCS RTL_CONSTANT_STRING
#endif

#define DECL_ARG(Name) const UNICODE_STRING Name = RCS(L#Name)

//
// Helper macro for Rtl->RtlEqualUnicodeString() comparison.
//

#define IS_EQUAL(Name) Rtl->RtlEqualUnicodeString(Arg, &Name, TRUE)

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
    PCUNICODE_STRING Arg = Argument;
    DECL_ARG(SkipTestAfterCreate);
    DECL_ARG(Compile);
    DECL_ARG(TryCuda);

    UNREFERENCED_PARAMETER(Allocator);

    SET_FLAG_AND_RETURN_IF_EQUAL(SkipTestAfterCreate);
    SET_FLAG_AND_RETURN_IF_EQUAL(Compile);
    SET_FLAG_AND_RETURN_IF_EQUAL(TryCuda);

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
    PCUNICODE_STRING Arg = Argument;
    DECL_ARG(SkipTestAfterCreate);
    DECL_ARG(Compile);
    DECL_ARG(TryCuda);

    UNREFERENCED_PARAMETER(Allocator);

    SET_FLAG_AND_RETURN_IF_EQUAL(SkipTestAfterCreate);
    SET_FLAG_AND_RETURN_IF_EQUAL(Compile);
    SET_FLAG_AND_RETURN_IF_EQUAL(TryCuda);

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
    PCUNICODE_STRING Arg = Argument;
    DECL_ARG(TryLargePagesForKeysData);
    DECL_ARG(SkipKeysVerification);
    DECL_ARG(DisableImplicitKeyDownsizing);
    DECL_ARG(TryInferKeySizeFromKeysFilename);

    UNREFERENCED_PARAMETER(Allocator);

    SET_FLAG_AND_RETURN_IF_EQUAL(TryLargePagesForKeysData);
    SET_FLAG_AND_RETURN_IF_EQUAL(SkipKeysVerification);
    SET_FLAG_AND_RETURN_IF_EQUAL(DisableImplicitKeyDownsizing);
    SET_FLAG_AND_RETURN_IF_EQUAL(TryInferKeySizeFromKeysFilename);

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
    PCUNICODE_STRING Arg = Argument;
    DECL_ARG(FirstGraphWins);
    DECL_ARG(FindBestGraph);
    DECL_ARG(SkipGraphVerification);
    DECL_ARG(CreateOnly);
    DECL_ARG(TryLargePagesForTableData);
    DECL_ARG(TryLargePagesForValuesArray);
    DECL_ARG(UsePreviousTableSize);
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
    DECL_ARG(IndexOnly);
    DECL_ARG(UseRwsSectionForTableValues);
    DECL_ARG(UseNonTemporalAvx2Routines);
    DECL_ARG(DisableCsvOutputFile);
    DECL_ARG(ClampNumberOfEdges);
    DECL_ARG(UseOriginalSeededHashRoutines);
    DECL_ARG(HashAllKeysFirst);
    DECL_ARG(EnableWriteCombineForVertexPairs);
    DECL_ARG(RemoveWriteCombineAfterSuccessfulHashKeys);
    DECL_ARG(TryLargePagesForVertexPairs);
    DECL_ARG(TryUsePredictedAttemptsToLimitMaxConcurrency);
    DECL_ARG(RngUseRandomStartSeed);

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
    SET_FLAG_AND_RETURN_IF_EQUAL(UsePreviousTableSize);
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
    SET_FLAG_AND_RETURN_IF_EQUAL(IndexOnly);
    SET_FLAG_AND_RETURN_IF_EQUAL(UseRwsSectionForTableValues);
    SET_FLAG_AND_RETURN_IF_EQUAL(UseNonTemporalAvx2Routines);
    SET_FLAG_AND_RETURN_IF_EQUAL(DisableCsvOutputFile);
    SET_FLAG_AND_RETURN_IF_EQUAL(ClampNumberOfEdges);
    SET_FLAG_AND_RETURN_IF_EQUAL(UseOriginalSeededHashRoutines);
    SET_FLAG_AND_RETURN_IF_EQUAL(HashAllKeysFirst);
    SET_FLAG_AND_RETURN_IF_EQUAL(EnableWriteCombineForVertexPairs);
    SET_FLAG_AND_RETURN_IF_EQUAL(RemoveWriteCombineAfterSuccessfulHashKeys);
    SET_FLAG_AND_RETURN_IF_EQUAL(TryLargePagesForVertexPairs);
    SET_FLAG_AND_RETURN_IF_EQUAL(TryUsePredictedAttemptsToLimitMaxConcurrency);
    SET_FLAG_AND_RETURN_IF_EQUAL(RngUseRandomStartSeed);

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
    PCUNICODE_STRING Arg = Argument;
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

TRY_EXTRACT_SEED_MASK_COUNTS TryExtractSeedMaskCounts;

_Use_decl_annotations_
HRESULT
TryExtractSeedMaskCounts(
    PRTL Rtl,
    PALLOCATOR Allocator,
    PCUNICODE_STRING InputString,
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param,
    BYTE SeedNumber,
    BYTE ByteNumber,
    BYTE NumberOfCounts
    )
{
    HRESULT Result;
    USHORT Index;
    ULONG Count;
    ULONG Cumulative;
    ULONGLONG Total;
    ULARGE_INTEGER Large;
    PSEED_MASK_COUNTS SeedMaskCounts = NULL;
    USHORT NumberOfInputStringChars;
    WCHAR Wide;
    PWCHAR Source;
    PWCHAR Dest;
    PVOID Buffer;

    if (NumberOfCounts != 32) {

        //
        // All our seed masks are 0x1f (31), so the only permitted value for
        // counts at the moment is 32.
        //

        return PH_E_INVALID_NUMBER_OF_SEED_MASK_COUNTS;
    }

    //
    // Try and extract the raw VALUE_ARRAY first.
    //

    Result = TryExtractValueArray(Rtl,
                                  Allocator,
                                  InputString,
                                  Param,
                                  FALSE);

    if (FAILED(Result)) {

        //
        // If we failed, nothing more to do, go straight to the end.
        //

        goto Error;
    }

    //
    // We extracted a VALUE_ARRAY successfully.  Ensure the correct number of
    // elements are present.
    //

    SeedMaskCounts = &Param->AsSeedMaskCounts;

    if (SeedMaskCounts->NumberOfValues != NumberOfCounts) {
        Result = PH_E_INVALID_NUMBER_OF_SEED_MASK_COUNT_ELEMENTS;
        goto Error;
    }

    //
    // We found the required number of elements.  Fill out the seed number and
    // byte number as per the function parameters, then loop over the provided
    // value array and calculate the total.
    //

    SeedMaskCounts->SeedNumber = SeedNumber;
    SeedMaskCounts->ByteNumber = ByteNumber;
    Total = 0;
    Cumulative = 0;

    for (Index = 0; Index < NumberOfCounts; Index++) {
        Count = SeedMaskCounts->Values[Index];
        Total += Count;
        if (Index == 0) {
            Cumulative = Count;
        } else {
            Cumulative += Count;
        }

        SeedMaskCounts->Cumulative[Index] = Cumulative;
    }

    //
    // Verify total is greater than 0 and less than MAX_ULONG.
    //

    if (Total == 0) {
        Result = PH_E_SEED_MASK_COUNT_TOTAL_IS_ZERO;
        goto Error;
    }

    Large.QuadPart = Total;
    if (Large.HighPart != 0) {
        Result = PH_E_SEED_MASK_COUNT_TOTAL_EXCEEDS_MAX_ULONG;
        goto Error;
    }

    SeedMaskCounts->Total = Large.LowPart;
    ASSERT(SeedMaskCounts->Cumulative[Index-1] == SeedMaskCounts->Total);

    //
    // Copy the input string, which will be comma separated, and convert all
    // the commas to spaces.  This allows us to dump it directly in CSV output
    // without botching up all the columns.
    //

    Buffer = Allocator->Vtbl->Calloc(Allocator,
                                     1,
                                     InputString->MaximumLength);
    if (Buffer == NULL) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    Source = InputString->Buffer;
    Dest = (PWCHAR)Buffer;
    NumberOfInputStringChars = InputString->Length >> 1;

    for (Index = 0; Index < NumberOfInputStringChars; Index++) {
        Wide = Source[Index];
        if (Wide == L',') {
            Wide = L' ';
        }
        Dest[Index] = Wide;
    }

    SeedMaskCounts->CountsString.Length = InputString->Length;
    SeedMaskCounts->CountsString.MaximumLength = InputString->MaximumLength;
    SeedMaskCounts->CountsString.Buffer = Dest;

    //
    // We're done, finish up.
    //

    ASSERT(Result == S_OK);
    goto End;

Error:

    if (Result == S_OK) {
        Result = E_UNEXPECTED;
    }

    //
    // Free the underlying pointers if applicable.
    //

    if (SeedMaskCounts != NULL) {
        Allocator->Vtbl->FreePointer(Allocator,
                                     &SeedMaskCounts->Values);
        Allocator->Vtbl->FreePointer(Allocator,
                                     &SeedMaskCounts->CountsString.Buffer);
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
        if (IsSeedMaskCountParameter(Param->Id)) {
            Allocator->Vtbl->FreePointer(
                Allocator,
                &Param->AsSeedMaskCounts.CountsString.Buffer
            );
        }
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
    LONG64 Value64 = 0;
    PWSTR Source;
    PALLOCATOR Allocator;
    BOOLEAN ValueIsInt64 = FALSE;
    BOOLEAN ValueIsInteger = FALSE;
    BOOLEAN EqualSignFound = FALSE;
    BOOLEAN TableParamFound = FALSE;
    HRESULT Result = S_FALSE;
    PUNICODE_STRING Arg;
    UNICODE_STRING LocalArg = { 0 };
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
    // Wire up LocalArg to Argument, and point the Arg pointer at it.
    // This allows us to fiddle with the length if we see an equal sign.
    //

    LocalArg.Length = Argument->Length;
    LocalArg.MaximumLength = Argument->MaximumLength;
    LocalArg.Buffer = Argument->Buffer;
    Arg = &LocalArg;

    if (EqualSignFound) {
        LocalArg.Length = (USHORT)RtlPointerToOffset(LocalArg.Buffer, Source-1);
        LocalArg.MaximumLength = LocalArg.Length;
        ASSERT(LocalArg.Buffer[LocalArg.Length >> 1] == L'=');
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
    // of the macros (e.g. ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INTEGER()).
    //
    // This routine is only called at the start of the program in order to
    // parse command line arguments, so, eh, we can live with some sloppiness
    // for now.
    //

    //
    // Determine if the incoming argument matches any of our known parameters.
    //

#define EXPAND_AS_IS_TABLE_CREATE_PARAM(Name) \
    if (IS_EQUAL(Name)) {                     \
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

    //
    // If no table parameter was found, return.
    //

    if (!TableParamFound) {
        return S_FALSE;
    }

    //
    // Verify our equal sign is in the right spot based on our
    // pointer arithmetic.
    //

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
    // Attempt to convert the value string into an integer representation.  We
    // use 0 as the base in order to leverage the automatic hex/octal/decimal
    // parsing logic.
    //

    Result = Rtl->RtlUnicodeStringToInteger(ValueString, 0, &Value);
    if (SUCCEEDED(Result)) {
        ValueIsInteger = TRUE;
    }

    Result = Rtl->RtlUnicodeStringToInt64(ValueString, 0, &Value64, NULL);
    if (SUCCEEDED(Result)) {
        ValueIsInt64 = TRUE;
    }

#define SET_PARAM_ID(Name)                         \
    LocalParam.Id = TableCreateParameter##Name##Id

#define ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INTEGER(Name) \
    if (IS_EQUAL(Name) && ValueIsInteger) {           \
        SET_PARAM_ID(Name);                           \
        LocalParam.AsULong = Value;                   \
        goto AddParam;                                \
    }

#define ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INT64(Name) \
    if (IS_EQUAL(Name) && ValueIsInt64) {           \
        SET_PARAM_ID(Name);                         \
        LocalParam.AsLongLong = Value64;            \
        goto AddParam;                              \
    }

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INTEGER(AttemptsBeforeTableResize);

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INTEGER(MaxNumberOfTableResizes);

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INTEGER(MaxNumberOfEqualBestGraphs);

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INTEGER(InitialNumberOfTableResizes);

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INTEGER(BestCoverageAttempts);

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INTEGER(KeySizeInBytes);

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INTEGER(ValueSizeInBytes);

#define IS_VALUE_EQUAL(ValueName) \
    Rtl->RtlEqualUnicodeString(ValueString, &ValueName, TRUE)

#define ADD_PARAM_IF_EQUAL_AND_VALUE_EQUAL(Name, ValueName) \
    if (IS_EQUAL(Name) && IS_VALUE_EQUAL(ValueName)) {      \
        SET_PARAM_ID(Name);                                 \
        LocalParam.AsULongLong = Name##ValueName##Id;       \
        goto AddParam;                                      \
    }

#define EXPAND_AS_ADD_PARAM(Name, Comparison, Comparator) \
    ADD_PARAM_IF_EQUAL_AND_VALUE_EQUAL(BestCoverageType,  \
                                       Comparison##Name);

    BEST_COVERAGE_TYPE_TABLE_ENTRY(EXPAND_AS_ADD_PARAM);

#define ADD_PARAM_IF_EQUAL_AND_VALUE_IS_CSV_OF_ASCENDING_INTEGERS(Name,  \
                                                                  Upper) \
    if (IS_EQUAL(Name)) {                                                \
        Result = TryExtractValueArray(Rtl,                               \
                                      Allocator,                         \
                                      ValueString,                       \
                                      &LocalParam,                       \
                                      TRUE);                             \
                                                                         \
        if (Result == S_OK) {                                            \
            SET_PARAM_ID(Name);                                          \
            goto AddParam;                                               \
        } else {                                                         \
            if (Result == PH_E_NOT_SORTED) {                             \
                Result = PH_E_##Upper##_NOT_SORTED;                      \
            } else if (Result == PH_E_DUPLICATE_DETECTED) {              \
                Result = PH_E_DUPLICATE_VALUE_DETECTED_IN_##Upper;       \
            } else if (Result != E_OUTOFMEMORY) {                        \
                Result = PH_E_INVALID_##Upper;                           \
            }                                                            \
            goto Error;                                                  \
        }                                                                \
    }

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_CSV_OF_ASCENDING_INTEGERS(
        KeysSubset,
        KEYS_SUBSET
    );

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_CSV_OF_ASCENDING_INTEGERS(
        CuDeviceOrdinals,
        CU_DEVICE_ORDINALS
    );

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INTEGER(CuDeviceOrdinal);

#define ADD_PARAM_IF_EQUAL_AND_VALUE_IS_CSV(Name, Upper) \
    if (IS_EQUAL(Name)) {                                \
        Result = TryExtractValueArray(Rtl,               \
                                      Allocator,         \
                                      ValueString,       \
                                      &LocalParam,       \
                                      FALSE);            \
                                                         \
        if (Result == S_OK) {                            \
            SET_PARAM_ID(Name);                          \
            goto AddParam;                               \
        } else {                                         \
            if (Result != E_OUTOFMEMORY) {               \
                Result = PH_E_INVALID_##Upper;           \
            }                                            \
            goto Error;                                  \
        }                                                \
    }

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_CSV(Seeds, SEEDS);

#define ADD_PARAM_IF_EQUAL_AND_VALUE_IS_CSV_SEED_MASK_COUNTS(Name,           \
                                                             Upper,          \
                                                             SeedNumber,     \
                                                             ByteNumber,     \
                                                             NumberOfCounts) \
    if (IS_EQUAL(Name)) {                                                    \
        Result = TryExtractSeedMaskCounts(Rtl,                               \
                                          Allocator,                         \
                                          ValueString,                       \
                                          &LocalParam,                       \
                                          SeedNumber,                        \
                                          ByteNumber,                        \
                                          NumberOfCounts);                   \
                                                                             \
        if (Result == S_OK) {                                                \
            SET_PARAM_ID(Name);                                              \
            TableCreateParameters->Flags.HasSeedMaskCounts = TRUE;           \
            goto AddParam;                                                   \
        } else {                                                             \
            if (Result != E_OUTOFMEMORY) {                                   \
                Result = PH_E_INVALID_##Upper;                               \
            }                                                                \
            goto Error;                                                      \
        }                                                                    \
    }

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_CSV_SEED_MASK_COUNTS(
        Seed3Byte1MaskCounts,
        SEED3_BYTE1_MASK_COUNTS,
        3,
        1,
        32
    );

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_CSV_SEED_MASK_COUNTS(
        Seed3Byte2MaskCounts,
        SEED3_BYTE2_MASK_COUNTS,
        3,
        2,
        32
    );

    if (IS_EQUAL(Rng)) {
        Result = PerfectHashLookupIdForName(Rtl,
                                            PerfectHashRngEnumId,
                                            ValueString,
                                            (PULONG)&LocalParam.AsRngId);
        if (SUCCEEDED(Result)) {
            SET_PARAM_ID(Rng);
            goto AddParam;
        } else {
            Result = PH_E_INVALID_RNG_NAME;
            goto Error;
        }
    }

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INT64(RngSeed);

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INT64(RngSubsequence);

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_INT64(RngOffset);

#define ADD_PARAM_IF_EQUAL_AND_VALUE_IS_TP_PRIORITY(Name, Upper)           \
    if (IS_EQUAL(Name##ThreadpoolPriority)) {                              \
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

    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_TP_PRIORITY(MainWork, MAIN_WORK);
    ADD_PARAM_IF_EQUAL_AND_VALUE_IS_TP_PRIORITY(FileWork, FILE_WORK);

    if (IS_EQUAL(SolutionsFoundRatio)) {
        double Double;
        wchar_t *End = NULL;
        wchar_t *Expected = ValueString->Buffer + (ValueString->Length >> 1);
        Double = wstrtod(ValueString->Buffer, &End);
        if (End == Expected) {
            SET_PARAM_ID(SolutionsFoundRatio);
            LocalParam.AsDouble = Double;
            goto AddParam;
        }
        Result = PH_E_INVALID_SOLUTIONS_FOUND_RATIO;
        goto Error;
    }

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
    // see if we can find a match.  If so, return a duplicate error.
    //

    Param = TableCreateParameters->Params;

    for (Index = 0; Index < NumberOfTableCreateParameters; Index++, Param++) {
        if (Param->Id == LocalParam.Id) {
            return PH_E_DUPLICATE_TABLE_CREATE_PARAMETER_DETECTED;
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

GET_TABLE_CREATE_PARAMETER_FOR_ID GetTableCreateParameterForId;

_Use_decl_annotations_
HRESULT
GetTableCreateParameterForId(
    PPERFECT_HASH_TABLE_CREATE_PARAMETERS TableCreateParameters,
    PERFECT_HASH_TABLE_CREATE_PARAMETER_ID ParameterId,
    PPERFECT_HASH_TABLE_CREATE_PARAMETER *Parameter
    )
/*++

Routine Description:

    Returns the table create parameter for the given ID if one is present.

Arguments:

    TableCreateParameters - Supplies a pointer to the table create parameters
        structure for which the cleanup is to be performed.

Return Value:

    S_OK - Successfully found parametr.

    S_FALSE - No parameter found matching ID.

    E_POINTER - TableCreateParameters was NULL.

    PH_E_INVALID_TABLE_CREATE_PARAMETERS - Invalid table create parameters.

--*/
{
    ULONG Index;
    ULONG Count;
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
        // There are no parameters, nothing to search.
        //

        return S_FALSE;
    }

    if (Params == NULL) {
        return PH_E_INVALID_TABLE_CREATE_PARAMETERS;
    }

    //
    // Argument validation complete.  Try and find the parameter.
    //

    for (Index = 0; Index < Count; Index++, Param++) {
        if (Param->Id == ParameterId) {

            //
            // We found the parameter we were looking for.  Update the caller's
            // pointer and return success.
            //

            *Parameter = Param;
            return S_OK;
        }
    }

    //
    // No parameter found.
    //

    return S_FALSE;

}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
