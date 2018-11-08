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
    UNREFERENCED_PARAMETER(Rtl);
    UNREFERENCED_PARAMETER(Allocator);
    DBG_UNREFERENCED_PARAMETER(Argument);
    DBG_UNREFERENCED_PARAMETER(Flags);

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
    PALLOCATOR Allocator,
    PCUNICODE_STRING Argument,
    PULONG NumberOfTableCreateParametersPointer,
    PPERFECT_HASH_TABLE_CREATE_PARAMETER *TableCreateParametersPointer
    )
{
    USHORT Count;
    USHORT Index;
    ULONG Value = 0;
    PWSTR Source;
    BOOLEAN Found = FALSE;
    BOOLEAN ValueIsInteger = FALSE;
    HRESULT Result = S_FALSE;
    UNICODE_STRING Temp = { 0 };
    PUNICODE_STRING ValueString;
    ULONG NumberOfTableCreateParameters;
    PERFECT_HASH_TABLE_CREATE_PARAMETER LocalParam;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER Param;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER TableCreateParameters;
    PPERFECT_HASH_TABLE_CREATE_PARAMETER NewTableCreateParameters;

    DECL_ARG(AttemptsBeforeTableResize);
    DECL_ARG(MaxNumberOfTableResizes);
    DECL_ARG(BestCoverageNumAttempts);
    DECL_ARG(BestCoverageType);
    DECL_ARG(HighestNumberOfEmptyCacheLines);

    ZeroStructInline(LocalParam);

    NumberOfTableCreateParameters = *NumberOfTableCreateParametersPointer;
    TableCreateParameters = *TableCreateParametersPointer;

    //
    // Invariant check: if number of table create parameters is 0, the array
    // pointer should be null, and vice versa.
    //

    if (NumberOfTableCreateParameters == 0) {
        if (TableCreateParameters != NULL) {
            PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        }
    } else {
        if (TableCreateParameters == NULL) {
            PH_RAISE(PH_E_INVARIANT_CHECK_FAILED);
        }
    }

    //
    // Find where the equals sign occurs, if at all.
    //

    Source = Argument->Buffer;
    Count = Argument->Length >> 1;

    for (Index = 0; Index < Count; Index++) {
        if (*Source == L'=') {
            Source++;
            Found = TRUE;
            break;
        }
        Source++;
    }

    if (!Found) {
        return S_FALSE;
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

    ADD_PARAM_IF_PREFIX_AND_VALUE_IS_INTEGER(BestCoverageNumAttempts);

#define IS_VALUE_EQUAL(ValueName) \
    Rtl->RtlEqualUnicodeString(ValueString, &ValueName, TRUE)

#define ADD_PARAM_IF_PREFIX_AND_VALUE_EQUAL(Name, ValueName) \
    if (IS_PREFIX(Name) && IS_VALUE_EQUAL(ValueName)) {      \
        SET_PARAM_ID(Name);                                  \
        LocalParam.AsULongLong = Name##ValueName##Id;        \
        goto AddParam;                                       \
    }

    ADD_PARAM_IF_PREFIX_AND_VALUE_EQUAL(BestCoverageType,
                                        HighestNumberOfEmptyCacheLines);

    //
    // No more table create parameters to look for, finish up.
    //

    goto End;

AddParam:

    //
    // If there are already table create parameters, we may have already seen
    // a parameter with the same ID.  Loop through all the existing ones and
    // see if we can find a match.  If so, update the value directly and return;
    // avoiding the allocation/reallocation logic altogether.
    //

    Param = TableCreateParameters;
    for (Index = 0; Index < NumberOfTableCreateParameters; Index++) {
        if (Param->Id == LocalParam.Id) {
            Param->AsULongLong = LocalParam.AsULongLong;
            return S_OK;
        }
    }

    if (NumberOfTableCreateParameters == 0) {

        NewTableCreateParameters = (
            Allocator->Vtbl->Calloc(
                Allocator,
                1,
                sizeof(*TableCreateParameters)
            )
        );

    } else {

        NewTableCreateParameters = (
            Allocator->Vtbl->ReCalloc(
                Allocator,
                TableCreateParameters,
                (ULONG_PTR)NumberOfTableCreateParameters + 1,
                sizeof(*TableCreateParameters)
            )
        );

    }

    if (!NewTableCreateParameters) {
        Result = E_OUTOFMEMORY;
        goto Error;
    }

    //
    // Write our new parameter to the end of the array.
    //

    Param = &NewTableCreateParameters[NumberOfTableCreateParameters];
    Param->Id = LocalParam.Id;
    Param->AsULongLong = LocalParam.AsULongLong;

    //
    // Update the caller's pointers and finish up.
    //

    *NumberOfTableCreateParametersPointer = NumberOfTableCreateParameters + 1;
    *TableCreateParametersPointer = NewTableCreateParameters;

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

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
