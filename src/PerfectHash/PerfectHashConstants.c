/*++

Copyright (c) 2018 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashConstants.c

Abstract:

    This module declares constants used by the perfect hash library.

--*/

#include "stdafx.h"

#define VERIFY_ALGORITHM_ARRAY_SIZE(Name) \
    C_ASSERT(ARRAYSIZE(Name) == PerfectHashInvalidAlgorithmId + 1)

#define VERIFY_HASH_ARRAY_SIZE(Name) \
    C_ASSERT(ARRAYSIZE(Name) == PerfectHashInvalidHashFunctionId + 1)

#define VERIFY_MASK_ARRAY_SIZE(Name) \
    C_ASSERT(ARRAYSIZE(Name) == PerfectHashInvalidMaskFunctionId + 1)

//
// Declare the array of creation routines.
//

const PCREATE_PERFECT_HASH_TABLE_IMPL CreationRoutines[] = {
    NULL,
    CreatePerfectHashTableImplChm01,
    NULL
};
VERIFY_ALGORITHM_ARRAY_SIZE(CreationRoutines);

//
// Define the array of loader routines.
//

const PLOAD_PERFECT_HASH_TABLE_IMPL LoaderRoutines[] = {
    NULL,
    LoadPerfectHashTableImplChm01,
    NULL
};
VERIFY_ALGORITHM_ARRAY_SIZE(LoaderRoutines);

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
    PerfectHashTableHashScratch,
    PerfectHashTableHashCrc32RotateXor,
    PerfectHashTableHashCrc32,
    PerfectHashTableHashDjb,
    PerfectHashTableHashDjbXor,
    PerfectHashTableHashFnv,
    PerfectHashTableHashCrc32Not,
    NULL
};
VERIFY_HASH_ARRAY_SIZE(HashRoutines);

//
// Define the array of number of seeds required per hash routine.
//

const SHORT HashRoutineNumberOfSeeds[] = {
    -1,

    2, // Crc32Rotate
    2, // Jenkins
    3, // RotateXor
    2, // AddSub
    2, // Xor
    4, // Scratch
    3, // Crc32RotateXor
    2, // Crc32
    2, // Djb
    2, // DjbXor
    2, // Fnv
    2, // Crc32Not

    -1,
};
VERIFY_HASH_ARRAY_SIZE(HashRoutineNumberOfSeeds);

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
    PerfectHashTableSeededHashScratch,
    PerfectHashTableSeededHashCrc32RotateXor,
    PerfectHashTableSeededHashCrc32,
    PerfectHashTableSeededHashDjb,
    PerfectHashTableSeededHashDjbXor,
    PerfectHashTableSeededHashFnv,
    PerfectHashTableSeededHashCrc32Not,
    NULL
};
VERIFY_HASH_ARRAY_SIZE(SeededHashRoutines);

//
// Define the array of hash mask routines.
//

const PPERFECT_HASH_TABLE_MASK_HASH MaskHashRoutines[] = {
    NULL,
    PerfectHashTableMaskHashModulus,
    PerfectHashTableMaskHashAnd,
    NULL
};
VERIFY_MASK_ARRAY_SIZE(MaskHashRoutines);

//
// Define the array of index mask routines.
//

const PPERFECT_HASH_TABLE_MASK_INDEX MaskIndexRoutines[] = {
    NULL,
    PerfectHashTableMaskIndexModulus,
    PerfectHashTableMaskIndexAnd,
    NULL
};
VERIFY_MASK_ARRAY_SIZE(MaskIndexRoutines);

//
// Define the array of index routines.
//

const PPERFECT_HASH_TABLE_INDEX IndexRoutines[] = {
    NULL,
    PerfectHashTableIndexImplChm01,
    NULL
};
VERIFY_ALGORITHM_ARRAY_SIZE(IndexRoutines);

//
// Define the array of fast-index routines.
//

const PERFECT_HASH_TABLE_FAST_INDEX_TUPLE FastIndexRoutines[] = {

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashCrc32RotateFunctionId,
        PerfectHashAndMaskFunctionId,
        PerfectHashTableFastIndexImplChm01Crc32RotateHashAndMask,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashJenkinsFunctionId,
        PerfectHashAndMaskFunctionId,
        PerfectHashTableFastIndexImplChm01JenkinsHashAndMask,
    },

};

const BYTE NumberOfFastIndexRoutines = ARRAYSIZE(FastIndexRoutines);

//
// Define the array of raw C string Index() implementations.
//

#include "CompiledPerfectHashTableChm01IndexCrc32RotateAnd_CSource_RawCString.h"
#include "CompiledPerfectHashTableChm01IndexJenkinsAnd_CSource_RawCString.h"
#include "CompiledPerfectHashTableChm01IndexScratchAnd_CSource_RawCString.h"
#include "CompiledPerfectHashTableChm01IndexCrc32RotateXorAnd_CSource_RawCString.h"
#include "CompiledPerfectHashTableChm01IndexCrc32And_CSource_RawCString.h"
#include "CompiledPerfectHashTableChm01IndexDjbAnd_CSource_RawCString.h"
#include "CompiledPerfectHashTableChm01IndexDjbXorAnd_CSource_RawCString.h"
#include "CompiledPerfectHashTableChm01IndexFnvAnd_CSource_RawCString.h"
#include "CompiledPerfectHashTableChm01IndexCrc32NotAnd_CSource_RawCString.h"
#undef RawCString

const PERFECT_HASH_TABLE_INDEX_IMPL_STRING_TUPLE IndexImplStringTuples[] = {

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashCrc32RotateFunctionId,
        PerfectHashAndMaskFunctionId,
        &CompiledPerfectHashTableChm01IndexCrc32RotateAndCSourceRawCString,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashJenkinsFunctionId,
        PerfectHashAndMaskFunctionId,
        &CompiledPerfectHashTableChm01IndexJenkinsAndCSourceRawCString,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashScratchFunctionId,
        PerfectHashAndMaskFunctionId,
        &CompiledPerfectHashTableChm01IndexScratchAndCSourceRawCString,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashCrc32RotateXorFunctionId,
        PerfectHashAndMaskFunctionId,
        &CompiledPerfectHashTableChm01IndexCrc32RotateXorAndCSourceRawCString,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashCrc32FunctionId,
        PerfectHashAndMaskFunctionId,
        &CompiledPerfectHashTableChm01IndexCrc32AndCSourceRawCString,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashDjbFunctionId,
        PerfectHashAndMaskFunctionId,
        &CompiledPerfectHashTableChm01IndexDjbAndCSourceRawCString,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashDjbXorFunctionId,
        PerfectHashAndMaskFunctionId,
        &CompiledPerfectHashTableChm01IndexDjbXorAndCSourceRawCString,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashFnvFunctionId,
        PerfectHashAndMaskFunctionId,
        &CompiledPerfectHashTableChm01IndexFnvAndCSourceRawCString,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashCrc32NotFunctionId,
        PerfectHashAndMaskFunctionId,
        &CompiledPerfectHashTableChm01IndexCrc32NotAndCSourceRawCString,
    },

};

const BYTE NumberOfIndexImplStrings = ARRAYSIZE(IndexImplStringTuples);

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
VERIFY_ALGORITHM_ARRAY_SIZE(AlgorithmNames);

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

const UNICODE_STRING PerfectHashHashCrc32FunctionName =
    RTL_CONSTANT_STRING(L"Crc32");

const UNICODE_STRING PerfectHashHashDjbFunctionName =
    RTL_CONSTANT_STRING(L"Djb");

const UNICODE_STRING PerfectHashHashDjbXorFunctionName =
    RTL_CONSTANT_STRING(L"DjbXor");

const UNICODE_STRING PerfectHashHashFnvFunctionName =
    RTL_CONSTANT_STRING(L"Fnv");

const UNICODE_STRING PerfectHashHashCrc32NotFunctionName =
    RTL_CONSTANT_STRING(L"Crc32Not");

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
    &PerfectHashHashScratchFunctionName,
    &PerfectHashHashCrc32RotateXorFunctionName,
    &PerfectHashHashCrc32FunctionName,
    &PerfectHashHashDjbFunctionName,
    &PerfectHashHashDjbXorFunctionName,
    &PerfectHashHashFnvFunctionName,
    &PerfectHashHashCrc32NotFunctionName,
    NULL,
};
VERIFY_HASH_ARRAY_SIZE(HashFunctionNames);

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
VERIFY_MASK_ARRAY_SIZE(MaskFunctionNames);

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

#define EXPAND_AS_EVENT_NAME(Verb, VUpper, Name, Upper)         \
    const UNICODE_STRING Context##Verb##d##Name##EventPrefix =  \
        RTL_CONSTANT_STRING(L"PerfectHashContext_"              \
                            L#Verb L"d" L#Name L"EventPrefix_");

PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_EVENT_NAME);

SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_EVENT_NAME);

#define EXPAND_AS_EVENT_NAME_ADDRESS(Verb, VUpper, Name, Upper) \
    &Context##Verb##d##Name##EventPrefix,

const PCUNICODE_STRING ContextObjectPrefixes[] = {
    &ContextShutdownEventPrefix,
    &ContextSucceededEventPrefix,
    &ContextFailedEventPrefix,
    &ContextCompletedEventPrefix,
    &ContextTryLargerTableSizeEventPrefix,
    &ContextVerifiedTableEventPrefix,

    PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_EVENT_NAME_ADDRESS)
    SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_EVENT_NAME_ADDRESS)
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

#define RCS RTL_CONSTANT_STRING

const UNICODE_STRING No = RCS(L"No.\n");
const UNICODE_STRING Yes = RCS(L"Yes.\n");
const UNICODE_STRING KeysExtension = RCS(L"keys");
const UNICODE_STRING DotKeysSuffix = RCS(L".keys");
const UNICODE_STRING DotTableSuffix = RCS(L".pht1");
const UNICODE_STRING DotCHeaderSuffix = RCS(L".h");
const UNICODE_STRING DotCSourceSuffix = RCS(L".c");
const UNICODE_STRING KeysWildcardSuffix = RCS(L"*.keys");

const STRING DotExeSuffixA = RCS(".exe");
const STRING DotDllSuffixA = RCS(".dll");
const STRING DotLibSuffixA = RCS(".lib");
const STRING DynamicLibraryConfigurationTypeA = RCS("DynamicLibrary");
const STRING ApplicationConfigurationTypeA = RCS("Application");

//
// Stream names.
//

const UNICODE_STRING TableInfoStreamName = RCS(L"Info");

//
// Extensions.
//

const UNICODE_STRING TextFileExtension = RCS(L"txt");
const UNICODE_STRING CSourceFileExtension = RCS(L"c");
const UNICODE_STRING CHeaderFileExtension = RCS(L"h");
const UNICODE_STRING TableFileExtension = RCS(L"pht1");
const UNICODE_STRING VCPropsExtension = RCS(L"props");
const UNICODE_STRING VCProjectExtension = RCS(L"vcxproj");
const UNICODE_STRING VSSolutionExtension = RCS(L"sln");

#define VERIFY_FILE_WORK_ARRAY_SIZE(Name) \
    C_ASSERT(ARRAYSIZE(Name) == NUMBER_OF_FILES + 2)

#define VERIFY_CONTEXT_FILE_WORK_ARRAY_SIZE(Name) \
    C_ASSERT(ARRAYSIZE(Name) == NUMBER_OF_CONTEXT_FILES + 2)

const PCUNICODE_STRING FileWorkItemExtensions[] = {
    NULL,

    &TableFileExtension,            // TableFile
    &TableFileExtension,            // TableInfoStream
    &CHeaderFileExtension,          // CHeaderFile
    &CSourceFileExtension,          // CSourceFile
    &CHeaderFileExtension,          // CHeaderStdAfxFile
    &CSourceFileExtension,          // CSourceStdAfxFile
    &CSourceFileExtension,          // CSourceKeysFile
    &CSourceFileExtension,          // CSourceTableDataFile
    &CHeaderFileExtension,          // CHeaderSupportFile
    &CSourceFileExtension,          // CSourceSupportFile
    &CSourceFileExtension,          // CSourceTestFile
    &CSourceFileExtension,          // CSourceTestExeFile
    &CSourceFileExtension,          // CSourceBenchmarkFullFile
    &CSourceFileExtension,          // CSourceBenchmarkFullExeFile
    &CSourceFileExtension,          // CSourceBenchmarkIndexFile
    &CSourceFileExtension,          // CSourceBenchmarkIndexExeFile
    &VCProjectExtension,            // VCProjectDllFile
    &VCProjectExtension,            // VCProjectTestExeFile
    &VCProjectExtension,            // VCProjectBenchmarkFullExeFile
    &VCProjectExtension,            // VCProjectBenchmarkIndexExeFile
    &VSSolutionExtension,           // VSSolutionFile
    &CHeaderFileExtension,          // CHeaderCompiledPerfectHashFile
    &CHeaderFileExtension,          // CHeaderCompiledPerfectHashMacroGlueFile
    &VCPropsExtension,              // VCPropsCompiledPerfectHashFile
    &TextFileExtension,             // TableStatsTextFile

    NULL,
};
VERIFY_FILE_WORK_ARRAY_SIZE(FileWorkItemExtensions);

//
// Suffixes.
//

const UNICODE_STRING CHeaderStdAfxFileSuffix = RCS(L"StdAfx");
const UNICODE_STRING CSourceStdAfxFileSuffix = RCS(L"StdAfx");
const UNICODE_STRING CSourceKeysFileSuffix = RCS(L"Keys");
const UNICODE_STRING CHeaderSupportFileSuffix = RCS(L"Support");
const UNICODE_STRING CSourceSupportFileSuffix = RCS(L"Support");
const UNICODE_STRING CSourceTableDataFileSuffix = RCS(L"TableData");
const UNICODE_STRING CSourceTestFileSuffix = RCS(L"Test");
const UNICODE_STRING CSourceTestExeFileSuffix = RCS(L"TestExe");
const UNICODE_STRING CSourceBenchmarkFullFileSuffix = RCS(L"BenchmarkFull");
const UNICODE_STRING CSourceBenchmarkFullExeFileSuffix = RCS(L"BenchmarkFullExe");
const UNICODE_STRING CSourceBenchmarkIndexFileSuffix = RCS(L"BenchmarkIndex");
const UNICODE_STRING CSourceBenchmarkIndexExeFileSuffix = RCS(L"BenchmarkIndexExe");
const UNICODE_STRING VCProjectDllFileSuffix = RCS(L"Dll");
const UNICODE_STRING VCProjectTestExeFileSuffix = RCS(L"TestExe");
const UNICODE_STRING VCProjectBenchmarkFullExeFileSuffix = RCS(L"BenchmarkFullExe");
const UNICODE_STRING VCProjectBenchmarkIndexExeFileSuffix = RCS(L"BenchmarkIndexExe");
const UNICODE_STRING CHeaderCompiledPerfectHashFileSuffix = RCS(L"CompiledPerfectHash");
const UNICODE_STRING VCPropsCompiledPerfectHashFileSuffix = RCS(L"CompiledPerfectHash");
const UNICODE_STRING TableStatsTextFileSuffix = RCS(L"Stats");

const PCUNICODE_STRING FileWorkItemSuffixes[] = {
    NULL,

    NULL,                                   // TableFile
    NULL,                                   // TableInfoStream
    NULL,                                   // CHeaderFile
    NULL,                                   // CSourceFile
    &CHeaderStdAfxFileSuffix,               // CHeaderStdAfxFile
    &CSourceStdAfxFileSuffix,               // CSourceStdAfxFile
    &CSourceKeysFileSuffix,                 // CSourceKeysFile
    &CSourceTableDataFileSuffix,            // CSourceTableDataFile
    &CHeaderSupportFileSuffix,              // CHeaderSupportFile
    &CSourceSupportFileSuffix,              // CSourceSupportFile
    &CSourceTestFileSuffix,                 // CSourceTestFile
    &CSourceTestExeFileSuffix,              // CSourceTestExeFile
    &CSourceBenchmarkFullFileSuffix,        // CSourceBenchmarkFullFile
    &CSourceBenchmarkFullExeFileSuffix,     // CSourceBenchmarkFullExeFile
    &CSourceBenchmarkIndexFileSuffix,       // CSourceBenchmarkIndexFile
    &CSourceBenchmarkIndexExeFileSuffix,    // CSourceBenchmarkIndexExeFile
    &VCProjectDllFileSuffix,                // VCProjectDllFile
    &VCProjectTestExeFileSuffix,            // VCProjectTestExeFile
    &VCProjectBenchmarkFullExeFileSuffix,   // VCProjectBenchmarkFullExeFile
    &VCProjectBenchmarkIndexExeFileSuffix,  // VCProjectBenchmarkIndexExeFile
    NULL,                                   // VSSolutionFile
    NULL,                                   // CHeaderCompiledPerfectHashFile
    NULL,                                   // CHeaderCompiledPerfectHashMacroGlueFile
    NULL,                                   // VCPropsCompiledPerfectHashFile
    &TableStatsTextFileSuffix,              // TableStatsTextFile

    NULL,
};
VERIFY_FILE_WORK_ARRAY_SIZE(FileWorkItemSuffixes);

//
// Stream names.
//

const PCUNICODE_STRING FileWorkItemStreamNames[] = {
    NULL,

    NULL,                           // TableFile
    &TableInfoStreamName,           // TableInfoStream
    NULL,                           // CHeaderFile
    NULL,                           // CSourceFile
    NULL,                           // CHeaderStdAfxFile
    NULL,                           // CSourceStdAfxFile
    NULL,                           // CSourceKeysFile
    NULL,                           // CSourceTableDataFile
    NULL,                           // CHeaderSupportFile
    NULL,                           // CSourceSupportFile
    NULL,                           // CSourceTestFile
    NULL,                           // CSourceTestExeFile
    NULL,                           // CSourceBenchmarkFullFile
    NULL,                           // CSourceBenchmarkFullExeFile
    NULL,                           // CSourceBenchmarkIndexFile
    NULL,                           // CSourceBenchmarkIndexExeFile
    NULL,                           // VCProjectDllFile
    NULL,                           // VCProjectTestExeFile
    NULL,                           // VCProjectBenchmarkFullExeFile
    NULL,                           // VCProjectBenchmarkIndexExeFile
    NULL,                           // VSSolutionFile
    NULL,                           // CHeaderCompiledPerfectHashFile
    NULL,                           // CHeaderCompiledPerfectHashMacroGlueFile
    NULL,                           // VCPropsCompiledPerfectHashFile
    NULL,                           // TableStatsTextFile

    NULL,
};
VERIFY_FILE_WORK_ARRAY_SIZE(FileWorkItemStreamNames);

//
// Base names.
//
// N.B. These are only used for context files (i.e. CONTEXT_FILE_ID).
//

const UNICODE_STRING CHeaderCompiledPerfectHashFileBaseName = RCS(L"CompiledPerfectHash");
const UNICODE_STRING CHeaderCompiledPerfectHashMacroGlueFileBaseName = RCS(L"CompiledPerfectHashMacroGlue");
const UNICODE_STRING VCPropsCompiledPerfectHashFileBaseName = RCS(L"CompiledPerfectHash");

const PCUNICODE_STRING FileWorkItemBaseNames[] = {
    NULL,

    NULL,                           // TableFile
    NULL,                           // TableInfoStream
    NULL,                           // CHeaderFile
    NULL,                           // CSourceFile
    NULL,                           // CHeaderStdAfxFile
    NULL,                           // CSourceStdAfxFile
    NULL,                           // CSourceKeysFile
    NULL,                           // CSourceTableDataFile
    NULL,                           // CHeaderSupportFile
    NULL,                           // CSourceSupportFile
    NULL,                           // CSourceTestFile
    NULL,                           // CSourceTestExeFile
    NULL,                           // CSourceBenchmarkFullFile
    NULL,                           // CSourceBenchmarkFullExeFile
    NULL,                           // CSourceBenchmarkIndexFile
    NULL,                           // CSourceBenchmarkIndexExeFile
    NULL,                           // VCProjectDllFile
    NULL,                           // VCProjectTestExeFile
    NULL,                           // VCProjectBenchmarkFullExeFile
    NULL,                           // VCProjectBenchmarkIndexExeFile
    NULL,                           // VSSolutionFile
    &CHeaderCompiledPerfectHashFileBaseName,
    &CHeaderCompiledPerfectHashMacroGlueFileBaseName,
    &VCPropsCompiledPerfectHashFileBaseName,
    NULL,                           // TableStatsTextFile

    NULL,
};
VERIFY_FILE_WORK_ARRAY_SIZE(FileWorkItemBaseNames);

//
// End-of-file initializers.
//

const EOF_INIT EofInits[] = {
    { EofInitTypeNull, },               // Null
    { EofInitTypeAssignedSize, },       // TableFile
    { EofInitTypeFixed, sizeof(GRAPH_INFO_ON_DISK) },   // TableInfoStream
    { EofInitTypeDefault, },            // CHeaderFile
    { EofInitTypeDefault, },            // CSourceFile
    { EofInitTypeNumberOfPages, 1 },    // CHeaderStdAfxFile
    { EofInitTypeNumberOfPages, 1 },    // CSourceStdAfxFile
    { EofInitTypeNumberOfKeysMultiplier, 16 },          // CSourceKeysFile
    { EofInitTypeNumberOfTableElementsMultiplier, 16 }, // CSourceTableDataFile
    { EofInitTypeDefault, },            // CHeaderSupportFile
    { EofInitTypeDefault, },            // CSourceSupportFile
    { EofInitTypeDefault, },            // CSourceTestFile
    { EofInitTypeDefault, },            // CSourceTestExeFile
    { EofInitTypeDefault, },            // CSourceBenchmarkFullFile
    { EofInitTypeDefault, },            // CSourceBenchmarkFullExeFile
    { EofInitTypeDefault, },            // CSourceBenchmarkIndexFile
    { EofInitTypeDefault, },            // CSourceBenchmarkIndexExeFile
    { EofInitTypeDefault, },            // VCProjectDllFile
    { EofInitTypeDefault, },            // VCProjectTestExeFile
    { EofInitTypeDefault, },            // VCProjectBenchmarkFullExeFile
    { EofInitTypeDefault, },            // VCProjectBenchmarkIndexExeFile
    { EofInitTypeDefault, },            // VSSolutionFile
    { EofInitTypeDefault, },            // CHeaderCompiledPerfectHashFile
    { EofInitTypeDefault, },            // CHeaderCompiledPerfectHashMacroGlueFile
    { EofInitTypeDefault, },            // VCPropsCompiledPerfectHashFile
    { EofInitTypeDefault, },            // TableStatsTextFile
    { EofInitTypeInvalid, },            // Invalid
};
VERIFY_FILE_WORK_ARRAY_SIZE(EofInits);

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

#define NUMBER_OF_INTERFACES 12
#define EXPECTED_ARRAY_SIZE NUMBER_OF_INTERFACES+2
#define VERIFY_ARRAY_SIZE(Name) C_ASSERT(ARRAYSIZE(Name) == EXPECTED_ARRAY_SIZE)

C_ASSERT(EXPECTED_ARRAY_SIZE == PerfectHashInvalidInterfaceId+1);
C_ASSERT(NUMBER_OF_INTERFACES == PerfectHashLastInterfaceId);
C_ASSERT(NUMBER_OF_INTERFACES == PerfectHashInvalidInterfaceId-1);

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
    sizeof(PERFECT_HASH_DIRECTORY),
    sizeof(GUARDED_LIST),
    sizeof(GRAPH),

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
    sizeof(PERFECT_HASH_DIRECTORY_VTBL),
    sizeof(GUARDED_LIST_VTBL),
    sizeof(GRAPH_VTBL),

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
    (SHORT)FIELD_OFFSET(PERFECT_HASH_DIRECTORY, Interface),
    (SHORT)FIELD_OFFSET(GUARDED_LIST, Interface),
    (SHORT)FIELD_OFFSET(GRAPH, Interface),

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
    (SHORT)FIELD_OFFSET(PERFECT_HASH_TLS_CONTEXT, Directory),
    -1, // GuardedList
    (SHORT)FIELD_OFFSET(PERFECT_HASH_TLS_CONTEXT, Graph),

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
    -1, // Directory
    -1, // GuardedList
    -1, // Graph

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
    &PerfectHashContextSetBaseOutputDirectory,
    &PerfectHashContextGetBaseOutputDirectory,
    &PerfectHashContextSelfTest,
    &PerfectHashContextSelfTestArgvW,
    &PerfectHashContextExtractSelfTestArgsFromArgvW,
};
VERIFY_VTBL_SIZE(PERFECT_HASH_CONTEXT, 7);

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
    &AllocatorReAlloc,
    &AllocatorReCalloc,
    &AllocatorFree,
    &AllocatorFreePointer,
    &AllocatorFreeStringBuffer,
    &AllocatorFreeUnicodeStringBuffer,
    &AllocatorAlignedMalloc,
    &AllocatorAlignedCalloc,
    &AllocatorAlignedReAlloc,
    &AllocatorAlignedReCalloc,
    &AllocatorAlignedFree,
    &AllocatorAlignedFreePointer,
    &AllocatorAlignedOffsetMalloc,
    &AllocatorAlignedOffsetCalloc,
    &AllocatorAlignedOffsetReAlloc,
    &AllocatorAlignedOffsetReCalloc,
};
VERIFY_VTBL_SIZE(ALLOCATOR, 18);

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
// PerfectHashDirectory
//

const PERFECT_HASH_DIRECTORY_VTBL PerfectHashDirectoryInterface = {
    (PPERFECT_HASH_DIRECTORY_QUERY_INTERFACE)&ComponentQueryInterface,
    (PPERFECT_HASH_DIRECTORY_ADD_REF)&ComponentAddRef,
    (PPERFECT_HASH_DIRECTORY_RELEASE)&ComponentRelease,
    (PPERFECT_HASH_DIRECTORY_CREATE_INSTANCE)&ComponentCreateInstance,
    (PPERFECT_HASH_DIRECTORY_LOCK_SERVER)&ComponentLockServer,
    &PerfectHashDirectoryOpen,
    &PerfectHashDirectoryCreate,
    &PerfectHashDirectoryGetFlags,
    &PerfectHashDirectoryGetPath,

    //
    // Begin private methods.
    //

    &PerfectHashDirectoryClose,
    &PerfectHashDirectoryScheduleRename,
    &PerfectHashDirectoryDoRename,
    &PerfectHashDirectoryAddFile,
    &PerfectHashDirectoryRemoveFile,
};
VERIFY_VTBL_SIZE(PERFECT_HASH_DIRECTORY, 4 + 5);

//
// GuardedList
//

const GUARDED_LIST_VTBL GuardedListInterface = {
    (PGUARDED_LIST_QUERY_INTERFACE)&ComponentQueryInterface,
    (PGUARDED_LIST_ADD_REF)&ComponentAddRef,
    (PGUARDED_LIST_RELEASE)&ComponentRelease,
    (PGUARDED_LIST_CREATE_INSTANCE)&ComponentCreateInstance,
    (PGUARDED_LIST_LOCK_SERVER)&ComponentLockServer,
    &GuardedListIsEmpty,
    &GuardedListQueryDepth,
    &GuardedListInsertHead,
    &GuardedListInsertTail,
    &GuardedListAppendTail,
    &GuardedListRemoveHead,
    &GuardedListRemoveTail,
    &GuardedListRemoveEntry,
    &GuardedListRemoveHeadEx,
    &GuardedListReset,
};
VERIFY_VTBL_SIZE(GUARDED_LIST, 10);

//
// TSX versions of the GuardedList interface.  See dllmain.c for more info.
//

const GUARDED_LIST_VTBL GuardedListTsxInterface = {
    (PGUARDED_LIST_QUERY_INTERFACE)&ComponentQueryInterface,
    (PGUARDED_LIST_ADD_REF)&ComponentAddRef,
    (PGUARDED_LIST_RELEASE)&ComponentRelease,
    (PGUARDED_LIST_CREATE_INSTANCE)&ComponentCreateInstance,
    (PGUARDED_LIST_LOCK_SERVER)&ComponentLockServer,
    &GuardedListIsEmptyTsx,
    &GuardedListQueryDepthTsx,
    &GuardedListInsertHeadTsx,
    &GuardedListInsertTailTsx,
    &GuardedListAppendTailTsx,
    &GuardedListRemoveHeadTsx,
    &GuardedListRemoveTailTsx,
    &GuardedListRemoveEntryTsx,
    &GuardedListRemoveHeadExTsx,

    //
    // N.B. We don't have a TSX version for Reset() as it's not really
    //      necessary (because it shouldn't be called in contention).
    //

    &GuardedListReset,
};
VERIFY_VTBL_SIZE(GUARDED_LIST, 10);

//
// Graph
//

const GRAPH_VTBL GraphInterface = {
    (PGRAPH_QUERY_INTERFACE)&ComponentQueryInterface,
    (PGRAPH_ADD_REF)&ComponentAddRef,
    (PGRAPH_RELEASE)&ComponentRelease,
    (PGRAPH_CREATE_INSTANCE)&ComponentCreateInstance,
    (PGRAPH_LOCK_SERVER)&ComponentLockServer,
    &GraphSetInfo,
    &GraphEnterSolvingLoop,
    &GraphLoadInfo,
    &GraphReset,
    &GraphLoadNewSeeds,
    &GraphSolve,
    &GraphVerify,
    &GraphCalculateAssignedMemoryCoverage,
    &GraphRegisterSolved,
};
VERIFY_VTBL_SIZE(GRAPH, 9);

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
    &PerfectHashDirectoryInterface,
    &GuardedListInterface,
    &GraphInterface,

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
    (PCOMPONENT_INITIALIZE)&PerfectHashDirectoryInitialize,
    (PCOMPONENT_INITIALIZE)&GuardedListInitialize,
    (PCOMPONENT_INITIALIZE)&GraphInitialize,

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
    (PCOMPONENT_RUNDOWN)&PerfectHashDirectoryRundown,
    (PCOMPONENT_RUNDOWN)&GuardedListRundown,
    (PCOMPONENT_RUNDOWN)&GraphRundown,

    NULL,
};
VERIFY_ARRAY_SIZE(ComponentRundownRoutines);

//
// Include source files for any strings that are referenced in more than one
// compilation unit.
//

#include "CompiledPerfectHashTableRoutinesPre_CSource_RawCString.h"
#include "CompiledPerfectHashTableRoutines_CSource_RawCString.h"
#include "CompiledPerfectHashTableRoutinesPost_CSource_RawCString.h"

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab nowrap                              :
