/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashConstants.c

Abstract:

    This module declares constants used by the perfect hash library.

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
    PerfectHashTableHashCrc32RotateXor,
    PerfectHashTableHashScratch,
    NULL
};

//
// Define the array of number of seeds required per hash routine.
//

const SHORT HashRoutineNumberOfSeeds[] = {
    -1,

    2, // Crc32Rotate
    3, // Jenkins
    3, // RotateXor
    2, // AddSub
    2, // Xor
    3, // Crc32RotateXor
    2, // Scratch

    -1,
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
    PerfectHashTableSeededHashCrc32RotateXor,
    PerfectHashTableSeededHashScratch,
    NULL
};

//
// Define the array of hash mask routines.
//

const PPERFECT_HASH_TABLE_MASK_HASH MaskHashRoutines[] = {
    NULL,
    PerfectHashTableMaskHashModulus,
    PerfectHashTableMaskHashAnd,
    NULL
};

//
// Define the array of index mask routines.
//

const PPERFECT_HASH_TABLE_MASK_INDEX MaskIndexRoutines[] = {
    NULL,
    PerfectHashTableMaskIndexModulus,
    PerfectHashTableMaskIndexAnd,
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
        PerfectHashChm01AlgorithmId,
        PerfectHashHashCrc32RotateFunctionId,
        PerfectHashAndMaskFunctionId,
        0,
        PerfectHashTableFastIndexImplChm01Crc32RotateHashAndMask,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashJenkinsFunctionId,
        PerfectHashAndMaskFunctionId,
        0,
        PerfectHashTableFastIndexImplChm01JenkinsHashAndMask,
    },

};

const BYTE NumberOfFastIndexRoutines = ARRAYSIZE(FastIndexRoutines);

//
// Define UNICODE_STRING structures for each algorithm name.
//

const UNICODE_STRING PerfectHashChm01AlgorithmName =
    RTL_CONSTANT_STRING(L"Chm01");

//
// Define the array of algorithm names.  This is intended to be indexed by the
// PERFECT_HASH_ALGORITHM_ID enum.
//

const PCUNICODE_STRING AlgorithmNames[] = {
    NULL,
    &PerfectHashChm01AlgorithmName,
    NULL,
};

//
// Define UNICODE_STRING structures for each hash function name.
//

const UNICODE_STRING PerfectHashHashCrc32RotateFunctionName =
    RTL_CONSTANT_STRING(L"Crc32Rotate");

const UNICODE_STRING PerfectHashHashJenkinsFunctionName =
    RTL_CONSTANT_STRING(L"Jenkins");

const UNICODE_STRING PerfectHashHashRotateXorFunctionName =
    RTL_CONSTANT_STRING(L"RotateXor");

const UNICODE_STRING PerfectHashHashAddSubXorFunctionName =
    RTL_CONSTANT_STRING(L"AddSubXor");

const UNICODE_STRING PerfectHashHashXorFunctionName =
    RTL_CONSTANT_STRING(L"Xor");

const UNICODE_STRING PerfectHashHashCrc32RotateXorFunctionName =
    RTL_CONSTANT_STRING(L"Crc32RotateXor");

const UNICODE_STRING PerfectHashHashScratchFunctionName =
    RTL_CONSTANT_STRING(L"Scratch");

//
// Define the array of hash function names.  This is intended to be indexed by
// the PERFECT_HASH_TABLE_HASH_FUNCTION_ID enum.
//

const PCUNICODE_STRING HashFunctionNames[] = {
    NULL,
    &PerfectHashHashCrc32RotateFunctionName,
    &PerfectHashHashJenkinsFunctionName,
    &PerfectHashHashRotateXorFunctionName,
    &PerfectHashHashAddSubXorFunctionName,
    &PerfectHashHashXorFunctionName,
    &PerfectHashHashCrc32RotateXorFunctionName,
    &PerfectHashHashScratchFunctionName,
    NULL,
};

//
// Define UNICODE_STRING structures for each mask function name.
//

const UNICODE_STRING PerfectHashModulusMaskFunctionName =
    RTL_CONSTANT_STRING(L"Modulus");

const UNICODE_STRING PerfectHashAndMaskFunctionName =
    RTL_CONSTANT_STRING(L"And");

const UNICODE_STRING PerfectHashXorAndMaskFunctionName =
    RTL_CONSTANT_STRING(L"XorAnd");

const UNICODE_STRING PerfectHashFoldAutoHashFunctionName =
    RTL_CONSTANT_STRING(L"FoldAuto");

const UNICODE_STRING PerfectHashFoldOnceHashFunctionName =
    RTL_CONSTANT_STRING(L"FoldOnce");

const UNICODE_STRING PerfectHashFoldTwiceHashFunctionName =
    RTL_CONSTANT_STRING(L"FoldTwice");

const UNICODE_STRING PerfectHashFoldThriceHashFunctionName =
    RTL_CONSTANT_STRING(L"FoldThrice");

//
// Define the array of mask function names.  This is intended to be indexed by
// the PERFECT_HASH_MASK_FUNCTION_ID enum.
//

const PCUNICODE_STRING MaskFunctionNames[] = {
    NULL,
    &PerfectHashModulusMaskFunctionName,
    &PerfectHashAndMaskFunctionName,
    NULL,
};

//
// Array of UNICODE_STRING event prefix names used by the runtime context.
//

const UNICODE_STRING ContextShutdownEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_ShutdownEvent_");

const UNICODE_STRING ContextSucceededEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_SucceededEvent_");

const UNICODE_STRING ContextFailedEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_FailedEvent_");

const UNICODE_STRING ContextCompletedEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_CompletedEvent_");

const UNICODE_STRING ContextTryLargerTableSizeEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_TryLargerTableSizeEvent_");

const UNICODE_STRING ContextVerifiedTableEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_VerifiedTableEvent_");

const UNICODE_STRING ContextPreparedTableFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_PreparedTableFileEvent_");

const UNICODE_STRING ContextSavedTableFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_SavedTableFileEvent_");

const UNICODE_STRING ContextPreparedCSourceTableDataFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_PreparedCSourceTableDataEvent_");

const UNICODE_STRING ContextSavedCSourceTableDataFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_SavedCSourceTableDataFileEvent_");

const UNICODE_STRING ContextPreparedTableInfoStreamEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_PreparedTableInfoStreamEvent_");

const UNICODE_STRING ContextSavedTableInfoStreamEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_SavedTableInfoStreamEvent_");

const UNICODE_STRING ContextPreparedCHeaderFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_PreparedCHeaderFileEvent_");

const UNICODE_STRING ContextSavedCHeaderFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_SavedCHeaderFileEvent_");

const UNICODE_STRING ContextPreparedCSourceFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_PreparedCSourceFileEvent_");

const UNICODE_STRING ContextSavedCSourceFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_SavedCSourceFileEvent_");

const UNICODE_STRING ContextPreparedCSourceKeysFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_PreparedCSourceKeysFileEvent_");

const UNICODE_STRING ContextSavedCSourceKeysFileEventPrefix =
    RTL_CONSTANT_STRING(L"PerfectHashContext_SavedCSourceKeysFileEvent_");

const PCUNICODE_STRING ContextObjectPrefixes[] = {
    &ContextShutdownEventPrefix,
    &ContextSucceededEventPrefix,
    &ContextFailedEventPrefix,
    &ContextCompletedEventPrefix,
    &ContextTryLargerTableSizeEventPrefix,
    &ContextVerifiedTableEventPrefix,
    &ContextPreparedTableFileEventPrefix,
    &ContextSavedTableFileEventPrefix,
    &ContextPreparedCSourceTableDataFileEventPrefix,
    &ContextSavedCSourceTableDataFileEventPrefix,
    &ContextPreparedTableInfoStreamEventPrefix,
    &ContextSavedTableInfoStreamEventPrefix,
    &ContextPreparedCHeaderFileEventPrefix,
    &ContextSavedCHeaderFileEventPrefix,
    &ContextPreparedCSourceFileEventPrefix,
    &ContextSavedCSourceFileEventPrefix,
    &ContextPreparedCSourceKeysFileEventPrefix,
    &ContextSavedCSourceKeysFileEventPrefix,
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
const UNICODE_STRING KeysExtension = RTL_CONSTANT_STRING(L"keys");
const UNICODE_STRING CHeaderExtension = RTL_CONSTANT_STRING(L"h");
const UNICODE_STRING TableExtension = RTL_CONSTANT_STRING(L"pht1");
const UNICODE_STRING TableInfoStreamName = RTL_CONSTANT_STRING(L"Info");
const UNICODE_STRING DotKeysSuffix = RTL_CONSTANT_STRING(L".keys");
const UNICODE_STRING DotTableSuffix = RTL_CONSTANT_STRING(L".pht1");
const UNICODE_STRING DotCHeaderSuffix = RTL_CONSTANT_STRING(L".h");
const UNICODE_STRING DotCSourceSuffix = RTL_CONSTANT_STRING(L".c");
const UNICODE_STRING KeysWildcardSuffix = RTL_CONSTANT_STRING(L"*.keys");
const UNICODE_STRING CSourceExtension = RTL_CONSTANT_STRING(L"c");
const UNICODE_STRING CSourceKeysSuffix = RTL_CONSTANT_STRING(L"Keys");
const UNICODE_STRING CSourceTableDataSuffix =
    RTL_CONSTANT_STRING(L"TableData");


//
// Placeholders for values we patch in the FastIndexEx() instruction streams.
//

const ULONG Seed1Placeholder = 0x11111111;
const ULONG Seed2Placeholder = 0x22222222;
const ULONG Seed3Placeholder = 0x33333333;
const ULONG Seed4Placeholder = 0x44444444;
const ULONG HashMaskPlaceholder = 0xaaaaaaaa;
const ULONG IndexMaskPlaceholder = 0xbbbbbbbb;

//
// COM glue.
//

//
// Bump this every time an interface is added.  This allows us to succeed every
// array declaration (for arrays intended to be indexed via the interface ID
// enum) with a VERIFY_ARRAY_SIZE() static assertion that ensures we've got
// entries for each ID.  The +2 on the EXPECTED_ARRAY_SIZE macro accounts for
// the leading NullInterfaceId and trailing InvalidInterfaceId slots.
//

#define NUMBER_OF_INTERFACES 9
#define EXPECTED_ARRAY_SIZE NUMBER_OF_INTERFACES+2
#define VERIFY_ARRAY_SIZE(Name) C_ASSERT(ARRAYSIZE(Name) == EXPECTED_ARRAY_SIZE)

C_ASSERT(EXPECTED_ARRAY_SIZE == PerfectHashInvalidInterfaceId+1);

const USHORT ComponentSizes[] = {
    0,

    sizeof(IUNKNOWN),
    sizeof(ICLASSFACTORY),
    sizeof(PERFECT_HASH_KEYS),
    sizeof(PERFECT_HASH_CONTEXT),
    sizeof(PERFECT_HASH_TABLE),
    sizeof(RTL),
    sizeof(ALLOCATOR),
    sizeof(PERFECT_HASH_FILE),
    sizeof(PERFECT_HASH_PATH),

    0,
};
VERIFY_ARRAY_SIZE(ComponentSizes);

const USHORT ComponentInterfaceSizes[] = {
    0,

    sizeof(IUNKNOWN_VTBL),
    sizeof(ICLASSFACTORY_VTBL),
    sizeof(PERFECT_HASH_KEYS_VTBL),
    sizeof(PERFECT_HASH_CONTEXT_VTBL),
    sizeof(PERFECT_HASH_TABLE_VTBL),
    sizeof(RTL_VTBL),
    sizeof(ALLOCATOR_VTBL),
    sizeof(PERFECT_HASH_FILE_VTBL),
    sizeof(PERFECT_HASH_PATH_VTBL),

    0,
};
VERIFY_ARRAY_SIZE(ComponentInterfaceSizes);

//
// N.B. We use -1 for invalid offsets instead of 0, as 0 could be a legitimate
// field offset if the member is the first element in the structure.
//

const SHORT ComponentInterfaceOffsets[] = {
    -1,

    (SHORT)FIELD_OFFSET(IUNKNOWN, Interface),
    (SHORT)FIELD_OFFSET(ICLASSFACTORY, Interface),
    (SHORT)FIELD_OFFSET(PERFECT_HASH_KEYS, Interface),
    (SHORT)FIELD_OFFSET(PERFECT_HASH_CONTEXT, Interface),
    (SHORT)FIELD_OFFSET(PERFECT_HASH_TABLE, Interface),
    (SHORT)FIELD_OFFSET(RTL, Interface),
    (SHORT)FIELD_OFFSET(ALLOCATOR, Interface),
    (SHORT)FIELD_OFFSET(PERFECT_HASH_FILE, Interface),
    (SHORT)FIELD_OFFSET(PERFECT_HASH_PATH, Interface),

    -1,
};
VERIFY_ARRAY_SIZE(ComponentInterfaceOffsets);

const SHORT ComponentInterfaceTlsContextOffsets[] = {
    -1,

    -1, // IUnknown
    -1, // IClassFactory
    (SHORT)FIELD_OFFSET(PERFECT_HASH_TLS_CONTEXT, Keys),
    (SHORT)FIELD_OFFSET(PERFECT_HASH_TLS_CONTEXT, Context),
    (SHORT)FIELD_OFFSET(PERFECT_HASH_TLS_CONTEXT, Table),
    (SHORT)FIELD_OFFSET(PERFECT_HASH_TLS_CONTEXT, Rtl),
    (SHORT)FIELD_OFFSET(PERFECT_HASH_TLS_CONTEXT, Allocator),
    (SHORT)FIELD_OFFSET(PERFECT_HASH_TLS_CONTEXT, File),
    (SHORT)FIELD_OFFSET(PERFECT_HASH_TLS_CONTEXT, Path),

    -1,
};
VERIFY_ARRAY_SIZE(ComponentInterfaceTlsContextOffsets);

const SHORT GlobalComponentsInterfaceOffsets[] = {
    -1,

    -1, // IUnknown
    -1, // IClassFactory
    -1, // Keys
    -1, // Context
    -1, // Table
    (SHORT)FIELD_OFFSET(GLOBAL_COMPONENTS, Rtl),
    (SHORT)FIELD_OFFSET(GLOBAL_COMPONENTS, Allocator),
    -1, // File
    -1, // Path

    -1,
};
VERIFY_ARRAY_SIZE(GlobalComponentsInterfaceOffsets);

extern COMPONENT_QUERY_INTERFACE ComponentQueryInterface;
extern COMPONENT_ADD_REF ComponentAddRef;
extern COMPONENT_RELEASE ComponentRelease;
extern COMPONENT_CREATE_INSTANCE ComponentCreateInstance;
extern COMPONENT_LOCK_SERVER ComponentLockServer;

//
// Define a helper macro for catching compile-time errors where additional vtbl
// members have been added to a struct, but the initializer in this module has
// not yet been updated to include the new function pointer.  The 'Count' param
// excludes the 5 members that are present on every interface (the IUnknown +
// IClassFactory).
//

#define VERIFY_VTBL_SIZE(Name, Count) \
    C_ASSERT((sizeof(Name##_VTBL) / sizeof(ULONG_PTR)) == (5 + Count))

//
// IUnknown
//

const IUNKNOWN_VTBL IUnknownInterface = {
    (PIUNKNOWN_QUERY_INTERFACE)&ComponentQueryInterface,
    (PIUNKNOWN_ADD_REF)&ComponentAddRef,
    (PIUNKNOWN_RELEASE)&ComponentRelease,
    (PIUNKNOWN_CREATE_INSTANCE)&ComponentCreateInstance,
    (PIUNKNOWN_LOCK_SERVER)&ComponentLockServer,
};
VERIFY_VTBL_SIZE(IUNKNOWN, 0);

//
// IClassFactory
//

const ICLASSFACTORY_VTBL IClassFactoryInterface = {
    (PICLASSFACTORY_QUERY_INTERFACE)&ComponentQueryInterface,
    (PICLASSFACTORY_ADD_REF)&ComponentAddRef,
    (PICLASSFACTORY_RELEASE)&ComponentRelease,
    (PICLASSFACTORY_CREATE_INSTANCE)&ComponentCreateInstance,
    (PICLASSFACTORY_LOCK_SERVER)&ComponentLockServer,
};
VERIFY_VTBL_SIZE(ICLASSFACTORY, 0);

//
// PerfectHashKeys
//

const PERFECT_HASH_KEYS_VTBL PerfectHashKeysInterface = {
    (PPERFECT_HASH_KEYS_QUERY_INTERFACE)&ComponentQueryInterface,
    (PPERFECT_HASH_KEYS_ADD_REF)&ComponentAddRef,
    (PPERFECT_HASH_KEYS_RELEASE)&ComponentRelease,
    (PPERFECT_HASH_KEYS_CREATE_INSTANCE)&ComponentCreateInstance,
    (PPERFECT_HASH_KEYS_LOCK_SERVER)&ComponentLockServer,
    &PerfectHashKeysLoad,
    &PerfectHashKeysGetFlags,
    &PerfectHashKeysGetAddress,
    &PerfectHashKeysGetBitmap,
    &PerfectHashKeysGetFile,
};
VERIFY_VTBL_SIZE(PERFECT_HASH_KEYS, 5);

//
// PerfectHashContext
//

const PERFECT_HASH_CONTEXT_VTBL PerfectHashContextInterface = {
    (PPERFECT_HASH_CONTEXT_QUERY_INTERFACE)&ComponentQueryInterface,
    (PPERFECT_HASH_CONTEXT_ADD_REF)&ComponentAddRef,
    (PPERFECT_HASH_CONTEXT_RELEASE)&ComponentRelease,
    (PPERFECT_HASH_CONTEXT_CREATE_INSTANCE)&ComponentCreateInstance,
    (PPERFECT_HASH_CONTEXT_LOCK_SERVER)&ComponentLockServer,
    &PerfectHashContextSetMaximumConcurrency,
    &PerfectHashContextGetMaximumConcurrency,
    &PerfectHashContextSelfTest,
    &PerfectHashContextSelfTestArgvW,
    &PerfectHashContextExtractSelfTestArgsFromArgvW,
};
VERIFY_VTBL_SIZE(PERFECT_HASH_CONTEXT, 5);

//
// PerfectHashTable
//

const PERFECT_HASH_TABLE_VTBL PerfectHashTableInterface = {
    (PPERFECT_HASH_TABLE_QUERY_INTERFACE)&ComponentQueryInterface,
    (PPERFECT_HASH_TABLE_ADD_REF)&ComponentAddRef,
    (PPERFECT_HASH_TABLE_RELEASE)&ComponentRelease,
    (PPERFECT_HASH_TABLE_CREATE_INSTANCE)&ComponentCreateInstance,
    (PPERFECT_HASH_TABLE_LOCK_SERVER)&ComponentLockServer,
    &PerfectHashTableCreate,
    &PerfectHashTableLoad,
    &PerfectHashTableGetFlags,
    &PerfectHashTableCompile,
    &PerfectHashTableTest,
    &PerfectHashTableInsert,
    &PerfectHashTableLookup,
    &PerfectHashTableDelete,
    NULL,   // Index
    NULL,   // Hash
    NULL,   // MaskHash
    NULL,   // MaskIndex
    NULL,   // SeededHash
    NULL,   // FastIndex
    NULL,   // SlowIndex
    &PerfectHashTableGetAlgorithmName,
    &PerfectHashTableGetHashFunctionName,
    &PerfectHashTableGetMaskFunctionName,
    &PerfectHashTableGetFile,
};
VERIFY_VTBL_SIZE(PERFECT_HASH_TABLE, 19);

//
// Rtl
//

const RTL_VTBL RtlInterface = {
    (PRTL_QUERY_INTERFACE)&ComponentQueryInterface,
    (PRTL_ADD_REF)&ComponentAddRef,
    (PRTL_RELEASE)&ComponentRelease,
    (PRTL_CREATE_INSTANCE)&ComponentCreateInstance,
    (PRTL_LOCK_SERVER)&ComponentLockServer,
    &RtlGenerateRandomBytes,
    &RtlPrintSysError,
    &RtlCreateBuffer,
    &RtlCreateMultipleBuffers,
    &RtlDestroyBuffer,
    &RtlCopyPages,
    &RtlFillPages,
    &RtlCreateRandomObjectNames,
    &RtlCreateSingleRandomObjectName,
    &RtlTryLargePageVirtualAlloc,
    &RtlTryLargePageVirtualAllocEx,
    &RtlTryLargePageCreateFileMappingW,
};
VERIFY_VTBL_SIZE(RTL, 12);

//
// Allocator
//

const ALLOCATOR_VTBL AllocatorInterface = {
    (PALLOCATOR_QUERY_INTERFACE)&ComponentQueryInterface,
    (PALLOCATOR_ADD_REF)&ComponentAddRef,
    (PALLOCATOR_RELEASE)&ComponentRelease,
    (PALLOCATOR_CREATE_INSTANCE)&ComponentCreateInstance,
    (PALLOCATOR_LOCK_SERVER)&ComponentLockServer,
    &AllocatorMalloc,
    &AllocatorCalloc,
    &AllocatorFree,
    &AllocatorFreePointer,
    &AllocatorFreeStringBuffer,
    &AllocatorFreeUnicodeStringBuffer,
};
VERIFY_VTBL_SIZE(ALLOCATOR, 6);

//
// PerfectHashFile
//

const PERFECT_HASH_FILE_VTBL PerfectHashFileInterface = {
    (PPERFECT_HASH_FILE_QUERY_INTERFACE)&ComponentQueryInterface,
    (PPERFECT_HASH_FILE_ADD_REF)&ComponentAddRef,
    (PPERFECT_HASH_FILE_RELEASE)&ComponentRelease,
    (PPERFECT_HASH_FILE_CREATE_INSTANCE)&ComponentCreateInstance,
    (PPERFECT_HASH_FILE_LOCK_SERVER)&ComponentLockServer,
    &PerfectHashFileLoad,
    &PerfectHashFileCreate,
    &PerfectHashFileGetFlags,
    &PerfectHashFileGetPath,
    &PerfectHashFileGetResources,

    //
    // Begin private methods.
    //

    &PerfectHashFileClose,
    &PerfectHashFileExtend,
    &PerfectHashFileTruncate,
    &PerfectHashFileMap,
    &PerfectHashFileUnmap,
    &PerfectHashFileScheduleRename,
    &PerfectHashFileDoRename,
};
VERIFY_VTBL_SIZE(PERFECT_HASH_FILE, 12);

//
// PerfectHashPath
//

const PERFECT_HASH_PATH_VTBL PerfectHashPathInterface = {
    (PPERFECT_HASH_PATH_QUERY_INTERFACE)&ComponentQueryInterface,
    (PPERFECT_HASH_PATH_ADD_REF)&ComponentAddRef,
    (PPERFECT_HASH_PATH_RELEASE)&ComponentRelease,
    (PPERFECT_HASH_PATH_CREATE_INSTANCE)&ComponentCreateInstance,
    (PPERFECT_HASH_PATH_LOCK_SERVER)&ComponentLockServer,
    &PerfectHashPathCopy,
    &PerfectHashPathCreate,
    &PerfectHashPathReset,
    &PerfectHashPathGetParts,

    //
    // Begin private methods.
    //

    &PerfectHashPathExtractParts,
};
VERIFY_VTBL_SIZE(PERFECT_HASH_PATH, 5);

//
// Interface array.
//

const VOID *ComponentInterfaces[] = {
    NULL,

    &IUnknownInterface,
    &IClassFactoryInterface,
    &PerfectHashKeysInterface,
    &PerfectHashContextInterface,
    &PerfectHashTableInterface,
    &RtlInterface,
    &AllocatorInterface,
    &PerfectHashFileInterface,
    &PerfectHashPathInterface,

    NULL,
};
VERIFY_ARRAY_SIZE(ComponentInterfaces);

const PCOMPONENT_INITIALIZE ComponentInitializeRoutines[] = {
    NULL,

    NULL, // IUnknown
    NULL, // IClassFactory

    (PCOMPONENT_INITIALIZE)&PerfectHashKeysInitialize,
    (PCOMPONENT_INITIALIZE)&PerfectHashContextInitialize,
    (PCOMPONENT_INITIALIZE)&PerfectHashTableInitialize,
    (PCOMPONENT_INITIALIZE)&RtlInitialize,
    (PCOMPONENT_INITIALIZE)&AllocatorInitialize,
    (PCOMPONENT_INITIALIZE)&PerfectHashFileInitialize,
    (PCOMPONENT_INITIALIZE)&PerfectHashPathInitialize,

    NULL,
};
VERIFY_ARRAY_SIZE(ComponentInitializeRoutines);

const PCOMPONENT_RUNDOWN ComponentRundownRoutines[] = {
    NULL,

    NULL, // IUnknown
    NULL, // IClassFactory

    (PCOMPONENT_RUNDOWN)&PerfectHashKeysRundown,
    (PCOMPONENT_RUNDOWN)&PerfectHashContextRundown,
    (PCOMPONENT_RUNDOWN)&PerfectHashTableRundown,
    (PCOMPONENT_RUNDOWN)&RtlRundown,
    (PCOMPONENT_RUNDOWN)&AllocatorRundown,
    (PCOMPONENT_RUNDOWN)&PerfectHashFileRundown,
    (PCOMPONENT_RUNDOWN)&PerfectHashPathRundown,

    NULL,
};
VERIFY_ARRAY_SIZE(ComponentRundownRoutines);

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab nowrap                              :
