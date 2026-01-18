/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashConstants.h

Abstract:

    This module declares constants related to the perfect hash library.

--*/

#pragma once

#include "stdafx.h"

//
// Define two magic numbers for the Magic field of the TABLE_INFO_ON_DISK_HEADER
// structure.
//

#define TABLE_INFO_ON_DISK_MAGIC_LOWPART  0x25101981 // My birthday!
#define TABLE_INFO_ON_DISK_MAGIC_HIGHPART 0x17071953 // My mum's birthday!

//
// Define the size, in characters, of the stack-allocated buffer used to
// construct the table suffix in PerfectHashTableCreatePath().
//

#define TABLE_SUFFIX_BUFFER_SIZE_IN_CHARS 512

//
// Define the size, in characters, of the stack-allocated buffer used to
// construct the command line string used to compile the table via msbuild.exe
// in PerfectHashTableCompile().  We make this a multiple of the table suffix
// buffer size (above) to ensure there is ample space to construct the string.
//

#define COMPILE_COMMANDLINE_BUFFER_SIZE_IN_CHARS \
    TABLE_SUFFIX_BUFFER_SIZE_IN_CHARS * 3

//
// Define the size, in characters, of the stack-allocated buffer used to
// construct the table size suffix in PerfectHashKeysTableSizeCreatePath().
//

#define KEYS_TABLE_SIZE_SUFFIX_BUFFER_SIZE_IN_CHARS 512

//
// Declare an array of creation routines.  This is intended to be indexed by
// the PERFECT_HASH_TABLE_ALGORITHM_ID enumeration.
//

extern const PCREATE_PERFECT_HASH_TABLE_IMPL CreationRoutines[];

//
// Declare an array of loader routines.  This is intended to be indexed by
// the PERFECT_HASH_ALGORITHM_ID enumeration.
//

extern const PLOAD_PERFECT_HASH_TABLE_IMPL LoaderRoutines[];

//
// Declare an array of index routines.  This is intended to be indexed by
// the PERFECT_HASH_ALGORITHM_ID enumeration.
//

extern const PPERFECT_HASH_TABLE_INDEX IndexRoutines[];

//
// Declare an array of fast-index routines.  Unlike our other arrays that are
// all indexed by enumeration IDs, this array captures <algorith, hash, mask,
// func> tuples of supporting fast index routines.  The inline method below
// (CompletePerfectHashTableInitialization()) is responsible for walking the
// array and determining if there is an entry present for the requested IDs.
// This approach has been selected over a 3-dimensional array as there will only
// typically be a small number of fast-index routines and maintaining a 3D array
// of mostly NULLs is cumbersome.
//

extern const PERFECT_HASH_TABLE_FAST_INDEX_TUPLE FastIndexRoutines[];
extern const BYTE NumberOfFastIndexRoutines;

//
// As above, but for raw C string representations of Index() routines.
//

extern const PERFECT_HASH_TABLE_INDEX_IMPL_STRING_TUPLE IndexImplStringTuples[];
extern const BYTE NumberOfIndexImplStrings;

//
// Declare an array of hash routines.  This is intended to be indexed by
// the PERFECT_HASH_HASH_FUNCTION_ID enumeration.
//

extern const PPERFECT_HASH_TABLE_HASH HashRoutines[];

//
// Declare an array of numbers representing the number of seeds used by each
// hash function.  Also indexed by PERFECT_HASH_TABLE_HASH_FUNCTION_ID.
//

extern const SHORT HashRoutineNumberOfSeeds[];

//
// Declare an array of hash routine seed masks.  This is intended to be indexed
// by the PERFECT_HASH_TABLE_HASH_FUNCTION_ID enumeration.
//

extern const PCSEED_MASKS HashRoutineSeedMasks[];

//
// Declare an array of hash mask routines.  This is intended to be indexed by
// the PERFECT_HASH_TABLE_HASH_FUNCTION_ID enumeration.
//

extern const PPERFECT_HASH_TABLE_MASK_HASH MaskHashRoutines[];

//
// Declare an array of index mask routines.  This is intended to be indexed by
// the PERFECT_HASH_TABLE_HASH_FUNCTION_ID enumeration.
//

extern const PPERFECT_HASH_TABLE_MASK_INDEX MaskIndexRoutines[];

//
// Declare an array of seeded hash routines.  This is intended to be indexed by the
// PERFECT_HASH_TABLE_HASH_FUNCTION_ID enumeration.
//

extern const PPERFECT_HASH_TABLE_SEEDED_HASH SeededHashRoutines[];

//
// Define hash and seeded hash "Ex" versions of the hash routines.  Both are
// intended to by indexed by PERFECT_HASH_TABLE_HASH_FUNCTION_ID enumeration.
//

extern const PPERFECT_HASH_TABLE_HASH_EX HashExRoutines[];
extern const PPERFECT_HASH_TABLE_SEEDED_HASH_EX SeededHashExRoutines[];
extern const PPERFECT_HASH_TABLE_SEEDED_HASH16_EX SeededHash16ExRoutines[];

//
// Declare an array of STRINGs representing C type names (e.g. 'unsigned short',
// 'int', 'long long', etc); intended to be indexed by the TYPE enum.
//

extern const STRING CTypeNames[];

//
// Declare an array of STRINGs representing NT-style C type names (e.g.
// 'USHORT', 'ULONG', 'ULONGLONG' etc); intended to be indexed by the TYPE enum.
//

extern const STRING NtTypeNames[];

//
// Hacky forward-decl of the 16-bit assigned index impl for Chm01.
//

extern PERFECT_HASH_TABLE_INDEX PerfectHashTableIndex16ImplChm01;

//
// Helper inline routine for initializing the extended vtbl interface and any
// other dynamic values.
//

FORCEINLINE
VOID
CompletePerfectHashTableInitialization(
    _In_ PPERFECT_HASH_TABLE Table
    )
{
    BYTE Index;
    BOOLEAN IsMatch;
    PERFECT_HASH_ALGORITHM_ID AlgorithmId;
    PERFECT_HASH_HASH_FUNCTION_ID HashFunctionId;
    PERFECT_HASH_MASK_FUNCTION_ID MaskFunctionId;
    PCPERFECT_HASH_TABLE_FAST_INDEX_TUPLE FastIndexTuple;
    PCPERFECT_HASH_TABLE_INDEX_IMPL_STRING_TUPLE StringTuple;
    PPERFECT_HASH_TABLE_VTBL Vtbl;

    //
    // Initialize aliases.
    //

    Vtbl = Table->Vtbl;
    AlgorithmId = Table->AlgorithmId;
    HashFunctionId = Table->HashFunctionId;
    MaskFunctionId = Table->MaskFunctionId;

    //
    // Initialize the routines specific to the given hash/mask ID.
    //

    Vtbl->Hash = HashRoutines[HashFunctionId];
    Vtbl->MaskHash = MaskHashRoutines[MaskFunctionId];
    Vtbl->MaskIndex = MaskIndexRoutines[MaskFunctionId];
    Vtbl->SeededHash = SeededHashRoutines[HashFunctionId];
    Vtbl->HashEx = HashExRoutines[HashFunctionId];
    Vtbl->SeededHashEx = SeededHashExRoutines[HashFunctionId];

    //
    // If we're using the Chm01 algorithm as the table indicates it's
    // using 16-bit assigned table data, override the Index routines
    // accordingly.
    //

    if ((Table->State.UsingAssigned16 != FALSE) &&
        (AlgorithmId == PerfectHashChm01AlgorithmId)) {

        Vtbl->Index = PerfectHashTableIndex16ImplChm01;
        Vtbl->SlowIndex = NULL;
        Vtbl->FastIndex = NULL;

    } else {

        //
        // Default the slow index to the normal index routine.
        //

        Vtbl->SlowIndex = IndexRoutines[AlgorithmId];

        //
        // Walk the fast index routine tuples and see if any of the entries match
        // the IDs being requested.  If so, save the routine to Vtbl->FastIndex.
        //

        Vtbl->FastIndex = NULL;

        for (Index = 0; Index < NumberOfFastIndexRoutines; Index++) {

            FastIndexTuple = &FastIndexRoutines[Index];

            IsMatch = (
                AlgorithmId == FastIndexTuple->AlgorithmId &&
                HashFunctionId == FastIndexTuple->HashFunctionId &&
                MaskFunctionId == FastIndexTuple->MaskFunctionId
            );

            if (IsMatch) {
                Vtbl->FastIndex = FastIndexTuple->FastIndex;
                break;
            }

        }

        Vtbl->Index = (Vtbl->FastIndex ? Vtbl->FastIndex : Vtbl->SlowIndex);
    }


    //
    // Walk the C impl string tuples and try find a match.
    //

    Table->IndexImplString = NULL;

    //
    // Chm02 uses all of Chm01's resources, so switch the ID if applicable.
    //

    if (AlgorithmId == PerfectHashChm02AlgorithmId) {
        AlgorithmId = PerfectHashChm01AlgorithmId;
    }

    for (Index = 0; Index < NumberOfIndexImplStrings; Index++) {

        StringTuple = &IndexImplStringTuples[Index];

        IsMatch = (
            AlgorithmId == StringTuple->AlgorithmId &&
            HashFunctionId == StringTuple->HashFunctionId &&
            MaskFunctionId == StringTuple->MaskFunctionId
        );

        if (IsMatch) {
            Table->IndexImplString = StringTuple->RawCString;
            break;
        }
    }
}

//
// Declare the arrays for enum type names.
//

extern const PCUNICODE_STRING RngNames[];
extern const PCUNICODE_STRING CuRngNames[];
extern const PCUNICODE_STRING CpuArchNames[];
extern const PCUNICODE_STRING InterfaceNames[];
extern const PCUNICODE_STRING AlgorithmNames[];
extern const PCUNICODE_STRING HashFunctionNames[];
extern const PCUNICODE_STRING MaskFunctionNames[];
extern const PCUNICODE_STRING BestCoverageTypeNames[];
extern const PCUNICODE_STRING TableCreateParameterNames[];

//
// Declare a generic function pointer for validating enum IDs.
//

typedef
BOOLEAN
(NTAPI IS_VALID_ID)(
    _In_ ULONG Id
    );
typedef IS_VALID_ID *PIS_VALID_ID;

//
// Define an array of these function pointers, indexed by enum type.
//

extern const PIS_VALID_ID IsValidIdFunctions[];

//
// Define an array of invalid enum ID HRESULTs, indexed by enum type.
//

extern const HRESULT InvalidEnumIdHResults[];

//
// Define an array of invalid enum ID and name HRESULTs, indexed by enum type.
//

extern const HRESULT InvalidEnumIdHResults[];
extern const HRESULT InvalidEnumNameHResults[];

//
// Define an array of { NullId, InvalidId } tuples that capture the bounds of
// the given enum type.
//

typedef struct _ENUM_ID_BOUNDS_TUPLE {
    ULONG NullId;
    ULONG InvalidId;
} ENUM_ID_BOUNDS_TUPLE;
typedef ENUM_ID_BOUNDS_TUPLE *PENUM_ID_BOUNDS_TUPLE;

extern const ENUM_ID_BOUNDS_TUPLE EnumIdBoundsTuples[];

//
// Define an array of pointers to array of enum names.
//

extern const PCUNICODE_STRING *EnumIdNames[];

//
// Object (e.g. events, shared memory sections) name prefixes for the runtime
// context.
//

extern const PCUNICODE_STRING ContextObjectPrefixes[];

extern const BYTE NumberOfContextEventPrefixes;
extern const BYTE NumberOfContextObjectPrefixes;

//
// Helper inline function for programmatically determining how many events
// are present in the context based on the FirstEvent and LastEvent addresses.
//

FORCEINLINE
BYTE
GetNumberOfContextEvents(
    _In_ PPERFECT_HASH_CONTEXT Context
    )
{
    BYTE NumberOfEvents;

    //
    // Calculate the number of event handles based on the first and last event
    // indicators in the context structure.  The additional sizeof(HANDLE)
    // accounts for the fact that we're going from 0-based address offsets
    // to 1-based counts.
    //

    NumberOfEvents = (BYTE)(

        (ULONG_PTR)(

            sizeof(HANDLE) +

            RtlOffsetFromPointer(
                &Context->LastSavedEvent,
                &Context->FirstEvent
            )

        ) / (ULONG_PTR)sizeof(HANDLE)
    );

    //
    // Sanity check the number of events matches the number of event prefixes.
    //

    ASSERT(NumberOfEvents == NumberOfContextEventPrefixes);

    return NumberOfEvents;
}

//
// Declare miscellaneous strings.
//

extern const UNICODE_STRING No;
extern const UNICODE_STRING Yes;
extern const UNICODE_STRING CsvSuffix;
extern const UNICODE_STRING CsvExtension;
extern const UNICODE_STRING KeysExtension;
extern const UNICODE_STRING DotKeysSuffix;
extern const UNICODE_STRING DotTableSuffix;
extern const UNICODE_STRING DotHeaderSuffix;
extern const UNICODE_STRING NullUnicodeString;
extern const UNICODE_STRING KeysWildcardSuffix;
extern const UNICODE_STRING TableInfoStreamName;
extern const UNICODE_STRING KeysTableSizeSuffix;
extern const UNICODE_STRING TableValuesSuffix;
extern const UNICODE_STRING TableValuesExtension;
extern const UNICODE_STRING PerfectHashBulkCreateCsvBaseName;
extern const UNICODE_STRING PerfectHashBulkCreateBestCsvBaseName;
extern const UNICODE_STRING PerfectHashTableCreateCsvBaseName;
extern const UNICODE_STRING PerfectHashTableCreateBestCsvBaseName;
extern const UNICODE_STRING CppHeaderFileExtension;
extern const UNICODE_STRING TomlFileExtension;
extern const UNICODE_STRING RustFileExtension;

extern const STRING NullString;
extern const STRING DotExeSuffixA;
extern const STRING DotDllSuffixA;
extern const STRING DotLibSuffixA;
extern const STRING DynamicLibraryConfigurationTypeA;
extern const STRING ApplicationConfigurationTypeA;
extern const STRING FunctionHookCallbackDefaultFunctionNameA;

extern const STRING BestCoverageTypeNamesA[];

//
// Declare VCProject and Makefile related strings.
//

extern const STRING LibTargetPrefix;
extern const STRING TestTargetPrefix;
extern const STRING BenchmarkFullTargetPrefix;
extern const STRING BenchmarkIndexTargetPrefix;

extern const STRING SoFileSuffix;
extern const STRING LibFileSuffix;
extern const STRING DllFileSuffix;
extern const STRING TestFileSuffix;
extern const STRING BenchmarkFullFileSuffix;
extern const STRING BenchmarkIndexFileSuffix;

//
// Arrays indexed by the FILE_WORK_ID enum.
//

extern const UNICODE_STRING FileWorkItemSuffixes[];
extern const UNICODE_STRING FileWorkItemStreamNames[];
extern const UNICODE_STRING FileWorkItemBaseNames[];
extern const PCUNICODE_STRING FileWorkItemExtensions[];
extern const EOF_INIT EofInits[];

//
// Define inline helper functions for obtaining various string pointers.
//

FORCEINLINE
_Must_inspect_result_
_Success_(return != 0)
PCUNICODE_STRING
GetFileWorkItemString(
    _In_ FILE_WORK_ID Id,
    _In_ const UNICODE_STRING *Array
    )
{
    PCUNICODE_STRING String;

    String = &Array[Id];
    return (IsValidUnicodeString(String) ? String : NULL);
}

FORCEINLINE
_Must_inspect_result_
_Success_(return != 0)
PCUNICODE_STRING
GetFileWorkItemSuffix(
    _In_ FILE_WORK_ID Id
    )
{
    PCUNICODE_STRING String;

    String = &FileWorkItemSuffixes[Id];
    return (IsValidUnicodeString(String) ? String : NULL);
}

FORCEINLINE
_Must_inspect_result_
_Success_(return != 0)
PCUNICODE_STRING
GetFileWorkItemStreamName(
    _In_ FILE_WORK_ID Id
    )
{
    PCUNICODE_STRING String;

    String = &FileWorkItemStreamNames[Id];
    return (IsValidUnicodeString(String) ? String : NULL);
}

FORCEINLINE
_Success_(return != 0)
PCUNICODE_STRING
GetFileWorkItemBaseName(
    _In_ FILE_WORK_ID Id
    )
{
    PCUNICODE_STRING String;

    String = &FileWorkItemBaseNames[Id];
    return (IsValidUnicodeString(String) ? String : NULL);
}

FORCEINLINE
_Success_(return != 0)
PCUNICODE_STRING
GetFileWorkItemExtension(
    _In_ FILE_WORK_ID Id
    )
{
    PCUNICODE_STRING String;

    //
    // N.B. Unlike the other methods, we don't use the following construct for
    //      the return statement here:
    //
    //          return (IsValidUnicodeString(String) ? String : NULL);
    //
    //      This is because NullUnicodeString (i.e. a valid UNICODE_STRING
    //      structure with Length == 0, MaximumLength == 2 and Buffer pointing
    //      to nothing) is a valid return type for file extensions, as it is
    //      needed in some circumstances (e.g. for creating the table's Makefile
    //      file (with no table suffix info)).
    //

    String = FileWorkItemExtensions[Id];
    return String;
}

//
// Declare placeholders for values we patch in the FastIndexEx() instruction
// streams.
//

extern const ULONG Seed1Placeholder;
extern const ULONG Seed2Placeholder;
extern const ULONG Seed3Placeholder;
extern const ULONG Seed4Placeholder;
extern const ULONG Seed5Placeholder;
extern const ULONG Seed6Placeholder;
extern const ULONG Seed7Placeholder;
extern const ULONG Seed8Placeholder;
extern const ULONG HashMaskPlaceholder;
extern const ULONG IndexMaskPlaceholder;

//
// COM glue.
//

extern const USHORT ComponentSizes[];
extern const USHORT ComponentInterfaceSizes[];
extern const SHORT ComponentInterfaceOffsets[];
extern const SHORT ComponentInterfaceTlsContextOffsets[];
extern const SHORT GlobalComponentsInterfaceOffsets[];
extern const VOID *ComponentInterfaces[];

extern const PCOMPONENT_INITIALIZE ComponentInitializeRoutines[];
extern const PCOMPONENT_RUNDOWN ComponentRundownRoutines[];

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
