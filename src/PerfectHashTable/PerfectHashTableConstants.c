/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashTableConstants.c

Abstract:

    This module declares constants used by the PerfectHashTable component.

--*/

#include "stdafx.h"

//
// Declare the array of creation routines.
//

const PCREATE_PERFECT_HASH_TABLE_IMPL CreationRoutines[] = {
    NULL,
    CreatePerfectHashTableImplChm01,
    NULL
};

//
// Define the array of loader routines.
//

const PLOAD_PERFECT_HASH_TABLE_IMPL LoaderRoutines[] = {
    NULL,
    LoadPerfectHashTableImplChm01,
    NULL
};

//
// Define the array of hash routines.
//

const PPERFECT_HASH_TABLE_HASH HashRoutines[] = {
    NULL,
    PerfectHashTableHashCrc32Rotate,
    PerfectHashTableHashJenkins,
    PerfectHashTableHashRotateXor,
    PerfectHashTableHashAddSubXor,
    PerfectHashTableHashXor,
    NULL
};

//
// Define the array of seeded hash routines.
//

const PPERFECT_HASH_TABLE_SEEDED_HASH SeededHashRoutines[] = {
    NULL,
    PerfectHashTableSeededHashCrc32Rotate,
    PerfectHashTableSeededHashJenkins,
    PerfectHashTableSeededHashRotateXor,
    PerfectHashTableSeededHashAddSubXor,
    PerfectHashTableSeededHashXor,
    NULL
};

//
// Define the array of hash mask routines.
//

const PPERFECT_HASH_TABLE_MASK_HASH MaskHashRoutines[] = {
    NULL,
    PerfectHashTableMaskHashModulus,
    PerfectHashTableMaskHashAnd,
    PerfectHashTableMaskHashXorAnd,

    //
    // The PerfectHashTableFoldAutoMaskFunctionId slot is next.  This is a
    // psuedo ID that don't actually match to a mask implementation.  The
    // algorithm is required to detect when this mask function is being used
    // and swap out the Vtbl pointer to one of the following fold methods
    // depending on the table size.  Thus, we use a NULL pointer in this array
    // such that we'll trap on the first attempt to mask if this hasn't been
    // done.
    //

    NULL,

    PerfectHashTableMaskHashFoldOnce,
    PerfectHashTableMaskHashFoldTwice,
    PerfectHashTableMaskHashFoldThrice,
    NULL
};

//
// Define the array of index mask routines.
//

const PPERFECT_HASH_TABLE_MASK_INDEX MaskIndexRoutines[] = {
    NULL,
    PerfectHashTableMaskIndexModulus,
    PerfectHashTableMaskIndexAnd,
    PerfectHashTableMaskIndexXorAnd,

    //
    // See above description regarding the following NULL slot.
    //

    NULL,

    PerfectHashTableMaskIndexFoldOnce,
    PerfectHashTableMaskIndexFoldTwice,
    PerfectHashTableMaskIndexFoldThrice,
    NULL
};

//
// Define the array of index routines.
//

const PPERFECT_HASH_TABLE_INDEX IndexRoutines[] = {
    NULL,
    PerfectHashTableIndexImplChm01,
    NULL
};

//
// Define the array of fast-index routines.
//

const PERFECT_HASH_TABLE_FAST_INDEX_TUPLE FastIndexRoutines[] = {

    {
        PerfectHashTableChm01AlgorithmId,
        PerfectHashTableHashCrc32RotateFunctionId,
        PerfectHashTableAndMaskFunctionId,
        PerfectHashTableFastIndexImplChm01Crc32RotateHashAndMask,
    },

    {
        PerfectHashTableChm01AlgorithmId,
        PerfectHashTableHashJenkinsFunctionId,
        PerfectHashTableAndMaskFunctionId,
        PerfectHashTableFastIndexImplChm01JenkinsHashAndMask,
    },

};

const BYTE NumberOfFastIndexRoutines = ARRAYSIZE(FastIndexRoutines);

//
// Define the array of vtbl sizes.
//

const PGET_VTBL_EX_SIZE GetVtblExSizeRoutines[] = {
    NULL,
    GetVtblExSizeChm01,
    NULL,
};

//
// Define UNICODE_STRING structures for each algorithm name.
//

const UNICODE_STRING PerfectHashTableChm01AlgorithmName =
    RTL_CONSTANT_STRING(L"Chm01");

//
// Define the array of algorithm names.  This is intended to be indexed by the
// PERFECT_HASH_TABLE_ALGORITHM_ID enum.
//

const PCUNICODE_STRING AlgorithmNames[] = {
    NULL,
    &PerfectHashTableChm01AlgorithmName,
    NULL,
};

//
// Define UNICODE_STRING structures for each hash function name.
//

const UNICODE_STRING PerfectHashTableHashCrc32RotateFunctionName =
    RTL_CONSTANT_STRING(L"Crc32Rotate");

const UNICODE_STRING PerfectHashTableHashJenkinsFunctionName =
    RTL_CONSTANT_STRING(L"Jenkins");

const UNICODE_STRING PerfectHashTableHashRotateXorFunctionName =
    RTL_CONSTANT_STRING(L"RotateXor");

const UNICODE_STRING PerfectHashTableHashAddSubXorFunctionName =
    RTL_CONSTANT_STRING(L"AddSubXor");

const UNICODE_STRING PerfectHashTableHashXorFunctionName =
    RTL_CONSTANT_STRING(L"Xor");

//
// Define the array of hash function names.  This is intended to be indexed by
// the PERFECT_HASH_TABLE_HASH_FUNCTION_ID enum.
//

const PCUNICODE_STRING HashFunctionNames[] = {
    NULL,
    &PerfectHashTableHashCrc32RotateFunctionName,
    &PerfectHashTableHashJenkinsFunctionName,
    &PerfectHashTableHashRotateXorFunctionName,
    &PerfectHashTableHashAddSubXorFunctionName,
    &PerfectHashTableHashXorFunctionName,
    NULL,
};

//
// Define UNICODE_STRING structures for each mask function name.
//

const UNICODE_STRING PerfectHashTableModulusMaskFunctionName =
    RTL_CONSTANT_STRING(L"Modulus");

const UNICODE_STRING PerfectHashTableAndMaskFunctionName =
    RTL_CONSTANT_STRING(L"And");

const UNICODE_STRING PerfectHashTableXorAndMaskFunctionName =
    RTL_CONSTANT_STRING(L"XorAnd");

const UNICODE_STRING PerfectHashTableFoldAutoMaskFunctionName =
    RTL_CONSTANT_STRING(L"FoldAuto");

const UNICODE_STRING PerfectHashTableFoldOnceMaskFunctionName =
    RTL_CONSTANT_STRING(L"FoldOnce");

const UNICODE_STRING PerfectHashTableFoldTwiceMaskFunctionName =
    RTL_CONSTANT_STRING(L"FoldTwice");

const UNICODE_STRING PerfectHashTableFoldThriceMaskFunctionName =
    RTL_CONSTANT_STRING(L"FoldThrice");

//
// Define the array of mask function names.  This is intended to be indexed by
// the PERFECT_HASH_TABLE_MASK_FUNCTION_ID enum.
//

const PCUNICODE_STRING MaskFunctionNames[] = {
    NULL,
    &PerfectHashTableModulusMaskFunctionName,
    &PerfectHashTableAndMaskFunctionName,
    &PerfectHashTableXorAndMaskFunctionName,
    &PerfectHashTableFoldAutoMaskFunctionName,
    &PerfectHashTableFoldOnceMaskFunctionName,
    &PerfectHashTableFoldTwiceMaskFunctionName,
    &PerfectHashTableFoldThriceMaskFunctionName,
    NULL,
};

//
// Array of UNICODE_STRING event prefix names used by the runtime context.
//

const UNICODE_STRING ContextShutdownEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashTableContext_ShutdownEvent_");

const UNICODE_STRING ContextSucceededEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashTableContext_SucceededEvent_");

const UNICODE_STRING ContextFailedEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashTableContext_FailedEvent_");

const UNICODE_STRING ContextCompletedEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashTableContext_CompletedEvent_");

const UNICODE_STRING ContextTryLargerTableSizeEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashTableContext_TryLargerTableSizeEvent_");

const UNICODE_STRING ContextPreparedFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashTableContext_PreparedFileEvent_");

const UNICODE_STRING ContextVerifiedEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashTableContext_VerifiedEvent_");

const UNICODE_STRING ContextSavedFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashTableContext_SavedFileEvent_");

const PCUNICODE_STRING ContextObjectPrefixes[] = {
    &ContextShutdownEventPrefix,
    &ContextSucceededEventPrefix,
    &ContextFailedEventPrefix,
    &ContextCompletedEventPrefix,
    &ContextTryLargerTableSizeEventPrefix,
    &ContextPreparedFileEventPrefix,
    &ContextVerifiedEventPrefix,
    &ContextSavedFileEventPrefix,
};

//
// We only have events at the moment so number of event prefixes will equal
// number of object prefixes.
//

const BYTE NumberOfContextEventPrefixes = ARRAYSIZE(ContextObjectPrefixes);
const BYTE NumberOfContextObjectPrefixes = ARRAYSIZE(ContextObjectPrefixes);

//
// Miscellaneous string constants.
//

const UNICODE_STRING No = RTL_CONSTANT_STRING(L"No.\n");
const UNICODE_STRING Yes = RTL_CONSTANT_STRING(L"Yes.\n");
const UNICODE_STRING KeysSuffix = RTL_CONSTANT_STRING(L"keys");
const UNICODE_STRING TableSuffix = RTL_CONSTANT_STRING(L"pht1");
const UNICODE_STRING KeysWildcardSuffix = RTL_CONSTANT_STRING(L"*.keys");

//
// Placeholders for values we patch in the FastIndexEx() instruction streams.
//

const ULONG Seed1Placeholder = 0x11111111;
const ULONG Seed2Placeholder = 0x22222222;
const ULONG Seed3Placeholder = 0x33333333;
const ULONG Seed4Placeholder = 0x44444444;
const ULONG HashMaskPlaceholder = 0xaaaaaaaa;
const ULONG IndexMaskPlaceholder = 0xbbbbbbbb;

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab nowrap                              :
