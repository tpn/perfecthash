/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashNames.c

Abstract:

    This module implements the routines for obtaining string representations
    of the various enum IDs and vice versa.

    N.B. Work-in-progress.

--*/

#include "stdafx.h"

HRESULT
PerfectHashLookupNameForId(
    PRTL Rtl,
    PERFECT_HASH_ENUM_TYPE EnumType,
    ULONG Id,
    PCSZ *NamePointer
    )
/*++

Routine Description:

    Looks up a string representation of an integer identifier for a given
    enum type.

Arguments:

    Rtl - Supplies a pointer to an Rtl instance.

    EnumType - Supplies the type of enum for which the Id parameter belongs.

    Id - Supplies the integer identifier for which the name is to be looked up.

    NamePointer - Receives the string representation of the error code.

Return Value:

    S_OK - Success.

    E_POINTER - Rtl or NamePointer were NULL.

    PH_E_INVALID_ENUM_TYPE - Invalid enum type.

    PH_E_INVALID_CPU_ARCH_ID - Invalid CPU arch ID.

    PH_E_INVALID_INTERFACE_ID - Invalid interface ID.

    PH_E_INVALID_ALGORITHM_ID - Invalid algorithm ID.

    PH_E_INVALID_HASH_FUNCTION_ID - Invalid hash function ID.

    PH_E_INVALID_MASK_FUNCTION_ID - Invalid mask function ID.

    PH_E_INVALID_BEST_COVERAGE_TYPE - Invalid best coverage type.

    PH_E_INVALID_TABLE_CREATE_PARAMETER_ID - Invalid table create param ID.

--*/
{
    DBG_UNREFERENCED_PARAMETER(Id);

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Rtl)) {
        return E_POINTER;
    }

    if (!ARGUMENT_PRESENT(NamePointer)) {
        return E_POINTER;
    }

    if (!IsValidPerfectHashEnumType(EnumType)) {
        return PH_E_INVALID_ENUM_TYPE;
    }

    //
    // Work in progress.
    //

    return PH_E_NOT_IMPLEMENTED;
}


// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
