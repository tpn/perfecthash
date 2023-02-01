/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashNames.c

Abstract:

    This module implements the routines for obtaining string representations
    of the various enum IDs and vice versa.

--*/

#include "stdafx.h"


PERFECT_HASH_LOOKUP_NAME_FOR_ID PerfectHashLookupNameForId;

HRESULT
PerfectHashLookupNameForId(
    PRTL Rtl,
    PERFECT_HASH_ENUM_ID EnumId,
    ULONG Id,
    PCUNICODE_STRING *NamePointer
    )
/*++

Routine Description:

    Looks up a string representation of an integer identifier for a given
    enum type.

Arguments:

    Rtl - Supplies a pointer to an Rtl instance.

    EnumId - Supplies the type of enum for which the Id parameter belongs.

    Id - Supplies the integer identifier for which the name is to be looked up.

    NamePointer - Receives the string representation of the Id.

Return Value:

    S_OK - Success.

    E_POINTER - Rtl or NamePointer were NULL.

    PH_E_INVALID_ENUM_ID - Invalid enum ID.

    PH_E_INVALID_CPU_ARCH_ID - Invalid CPU arch ID.

    PH_E_INVALID_INTERFACE_ID - Invalid interface ID.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_BEST_COVERAGE_ID - Invalid best coverage type ID.

    PH_E_INVALID_TABLE_CREATE_PARAMETER_ID - Invalid table create param ID.

--*/
{
    HRESULT Result;
    PIS_VALID_ID IsValidId;
    PCUNICODE_STRING Name;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(NamePointer)) {
        return E_POINTER;
    }

    if (!IsValidPerfectHashEnumId(EnumId)) {
        return PH_E_INVALID_ENUM_ID;
    }

    //
    // Arguments validated.  Obtain the IsValidId() function pointer for the
    // given enum type and verify the Id value being supplied.  If invalid,
    // return with the appropriate error code for that enum type.
    //

    IsValidId = IsValidIdFunctions[EnumId];
    if (!IsValidId(Id)) {
        Result = InvalidEnumIdHResults[EnumId];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
        _Analysis_assume_(Result < 0);
#pragma clang diagnostic pop
        return Result;
    }

    //
    // The supplied Id is valid.  Obtain the string representation, update the
    // caller's pointer, and return success.
    //

    Name = EnumIdNames[EnumId][Id];
    *NamePointer = Name;

    return S_OK;
}


PERFECT_HASH_LOOKUP_ID_FOR_NAME PerfectHashLookupIdForName;

_Use_decl_annotations_
HRESULT
PerfectHashLookupIdForName(
    PRTL Rtl,
    PERFECT_HASH_ENUM_ID EnumId,
    PCUNICODE_STRING Name,
    PULONG IdPointer
    )
/*++

Routine Description:

    Looks up the enum ID of a string for a given type.

Arguments:

    Rtl - Supplies a pointer to an Rtl instance.

    EnumId - Supplies the type of enum for which the Id parameter belongs.

    Name - Supplies the name to look up.

    IdPointer - Receives the enum ID on success.

Return Value:

    S_OK - Success.

    E_POINTER - Rtl or Name were NULL.

    PH_E_INVALID_ENUM_NAME - Invalid enum name.

    PH_E_INVALID_CPU_ARCH_NAME - Invalid CPU arch name.

    PH_E_INVALID_INTERFACE_NAME - Invalid interface name.

    PH_E_INVALID_ALGORITHM_NAME - Invalid algorithm name.

    PH_E_INVALID_HASH_FUNCTION_NAME - Invalid hash function name.

    PH_E_INVALID_MASK_FUNCTION_NAME - Invalid mask function name.

    PH_E_INVALID_BEST_COVERAGE_NAME - Invalid best coverage type name.

    PH_E_INVALID_TABLE_CREATE_PARAMETER_NAME - Invalid table create param name.

--*/
{
    ULONG Index;
    HRESULT Result;
    PCUNICODE_STRING String;

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(Name)) {
        return E_POINTER;
    }

    if (!IsValidUnicodeString(Name)) {
        return E_INVALIDARG;
    }

    if (!IsValidPerfectHashEnumId(EnumId)) {
        return PH_E_INVALID_ENUM_ID;
    }

    //
    // Arguments validated, proceed.
    //

    Index = EnumIdBoundsTuples[EnumId].NullId + 1;
    ASSERT(Index > 0);

    //
    // Loop through the strings for each ID in the target enum.  If the string
    // matches the name provided by the caller, we've found the one we want;
    // update the caller's pointer with the current index (which will be a
    // valid enum value) and return success.
    //

    while (TRUE) {
        String = EnumIdNames[EnumId][Index];
        if (String == NULL) {
            break;
        }
        if (Rtl->RtlEqualUnicodeString(String, Name, TRUE) != FALSE) {
            *IdPointer = Index;
            return S_OK;
        }
        Index++;
    }

    //
    // If we get here, the name did not match any of the strings for the enum.
    // Return the appropriate error code.
    //

    Result = InvalidEnumNameHResults[EnumId];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
    _Analysis_assume_(Result < 0);
#pragma clang diagnostic pop
    return Result;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
