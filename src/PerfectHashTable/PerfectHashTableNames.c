/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableNames.c

Abstract:

    This module implements the routines for obtaining string representations
    of the algorithm, hash function and masking type enumerations.

--*/

#include "stdafx.h"

GET_PERFECT_HASH_TABLE_ALGORITHM_NAME GetPerfectHashTableAlgorithmName;

_Use_decl_annotations_
BOOLEAN
GetPerfectHashTableAlgorithmName(
    PERFECT_HASH_TABLE_ALGORITHM_ID AlgorithmId,
    PCUNICODE_STRING *Name
    )
/*++

Routine Description:

    Returns the name associated with the given algorithm ID.

Arguments:

    AlgorithmId - Supplies the algorithm ID.

    Name - Receives the UNICODE_STRING representation of the name.  If an
        invalid algorithm ID is supplied, this will be set to NULL.

Return Value:

    TRUE on success, FALSE on failure.  FALSE will be returned if an invalid
    algorithm ID is supplied.

--*/
{

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Name)) {
        return FALSE;
    }

    if (!IsValidPerfectHashTableAlgorithmId(AlgorithmId)) {
        *Name = NULL;
        return FALSE;
    }

    //
    // Update the caller's pointer and return success.
    //

    *Name = AlgorithmNames[AlgorithmId];
    return TRUE;
}

GET_PERFECT_HASH_TABLE_HASH_FUNCTION_NAME GetPerfectHashTableHashFunctionName;

_Use_decl_annotations_
BOOLEAN
GetPerfectHashTableHashFunctionName(
    PERFECT_HASH_TABLE_HASH_FUNCTION_ID HashFunctionId,
    PCUNICODE_STRING *Name
    )
/*++

Routine Description:

    Returns the name associated with the given hash function ID.

Arguments:

    HashFunctionId - Supplies the hash function ID.

    Name - Receives the UNICODE_STRING representation of the name.  If an
        invalid hash function ID is supplied, this will be set to NULL.

Return Value:

    TRUE on success, FALSE on failure.  FALSE will be returned if an invalid
    hash function ID is supplied.

--*/
{

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Name)) {
        return FALSE;
    }

    if (!IsValidPerfectHashTableHashFunctionId(HashFunctionId)) {
        *Name = NULL;
        return FALSE;
    }

    //
    // Update the caller's pointer and return success.
    //

    *Name = HashFunctionNames[HashFunctionId];
    return TRUE;
}

GET_PERFECT_HASH_TABLE_MASK_FUNCTION_NAME GetPerfectHashTableMaskFunctionName;

_Use_decl_annotations_
BOOLEAN
GetPerfectHashTableMaskFunctionName(
    PERFECT_HASH_TABLE_MASK_FUNCTION_ID MaskFunctionId,
    PCUNICODE_STRING *Name
    )
/*++

Routine Description:

    Returns the name associated with the given mask function ID.

Arguments:

    MaskFunctionId - Supplies the mask function ID.

    Name - Receives the UNICODE_STRING representation of the name.  If an
        invalid mask function ID is supplied, this will be set to NULL.

Return Value:

    TRUE on success, FALSE on failure.  FALSE will be returned if an invalid
    mask function ID is supplied.

--*/
{

    //
    // Validate arguments.
    //

    if (!ARGUMENT_PRESENT(Name)) {
        return FALSE;
    }

    if (!IsValidPerfectHashTableMaskFunctionId(MaskFunctionId)) {
        *Name = NULL;
        return FALSE;
    }

    *Name = MaskFunctionNames[MaskFunctionId];
    return TRUE;
}

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
