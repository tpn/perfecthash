/*++

Copyright (c) 2018-2023 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashConstants.c

Abstract:

    This module declares constants used by the perfect hash library.

--*/

#include "stdafx.h"

#define RCS RTL_CONSTANT_STRING
#define NULL_STRING RCS("")
#define NULL_UNICODE_STRING RCS(L"")

#define VERIFY_ALGORITHM_ARRAY_SIZE(Name) \
    C_ASSERT(ARRAYSIZE(Name) == PerfectHashInvalidAlgorithmId + 1)

#define VERIFY_HASH_ARRAY_SIZE(Name) \
    C_ASSERT(ARRAYSIZE(Name) == PerfectHashInvalidHashFunctionId + 1)

#define VERIFY_MASK_ARRAY_SIZE(Name) \
    C_ASSERT(ARRAYSIZE(Name) == PerfectHashInvalidMaskFunctionId + 1)

//
// Implement the IsValidId() routines for each enum type.
//

#define EXPAND_AS_IS_VALID_ID_FUNC(Name, Upper)                             \
    IS_VALID_ID IsValid##Name##Id;                                          \
                                                                            \
    _Use_decl_annotations_                                                  \
    BOOLEAN                                                                 \
    IsValid##Name##Id(                                                      \
        ULONG Id                                                            \
        )                                                                   \
    {                                                                       \
        return IsValidPerfectHash##Name##Id((PERFECT_HASH_##Upper##_ID)Id); \
    }

//
// N.B. We need the following define to use the X-macro with the best coverage
//      type as the enum type name differs slightly (has _TABLE_ in it).
//

#define PERFECT_HASH_BEST_COVERAGE_TYPE_ID \
    PERFECT_HASH_TABLE_BEST_COVERAGE_TYPE_ID

PERFECT_HASH_ENUM_TABLE_ENTRY(EXPAND_AS_IS_VALID_ID_FUNC);

//
// Capture the routines above into the array of IsValidId() functions.
//

#define EXPAND_AS_IS_VALID_ID_FUNC_NAME(Name, Upper) \
    IsValid##Name##Id,

const PIS_VALID_ID IsValidIdFunctions[] = {
    NULL,
    PERFECT_HASH_ENUM_TABLE_ENTRY(EXPAND_AS_IS_VALID_ID_FUNC_NAME)
    NULL,
};

//
// Declare the array of invalid enum ID HRESULT error codes.
//

#define EXPAND_AS_INVALID_ENUM_ID_HRESULT(Name, Upper) \
    PH_E_INVALID_##Upper##_ID,

const HRESULT InvalidEnumIdHResults[] = {
    E_UNEXPECTED,
    PERFECT_HASH_ENUM_TABLE_ENTRY(EXPAND_AS_INVALID_ENUM_ID_HRESULT)
    E_UNEXPECTED,
};

//
// Declare the array of invalid enum name HRESULT error codes.
//

#define EXPAND_AS_INVALID_ENUM_NAME_HRESULT(Name, Upper) \
    PH_E_INVALID_##Upper##_NAME,

const HRESULT InvalidEnumNameHResults[] = {
    E_UNEXPECTED,
    PERFECT_HASH_ENUM_TABLE_ENTRY(EXPAND_AS_INVALID_ENUM_NAME_HRESULT)
    E_UNEXPECTED,
};

//
// Declare the array of enum ID bounds for each type.
//

#define EXPAND_AS_ENUM_ID_BOUNDS(Name, Upper) \
    { PerfectHashNull##Name##Id, PerfectHashInvalid##Name##Id },

const ENUM_ID_BOUNDS_TUPLE EnumIdBoundsTuples[] = {
    { (ULONG)-1, 0 },
    PERFECT_HASH_ENUM_TABLE_ENTRY(EXPAND_AS_ENUM_ID_BOUNDS)
    { (ULONG)-1, 0 },
};

//
// Declare the array of creation routines.
//

#define EXPAND_AS_CREATE_TABLE_IMPL_FUNC(Name, Upper) \
    CreatePerfectHashTableImpl##Name,

const PCREATE_PERFECT_HASH_TABLE_IMPL CreationRoutines[] = {
    NULL,
    PERFECT_HASH_ALGORITHM_TABLE_ENTRY(EXPAND_AS_CREATE_TABLE_IMPL_FUNC)
    NULL
};
VERIFY_ALGORITHM_ARRAY_SIZE(CreationRoutines);

//
// Define the array of loader routines.
//

#define EXPAND_AS_LOAD_TABLE_IMPL_FUNC(Name, Upper) \
    LoadPerfectHashTableImpl##Name,

const PLOAD_PERFECT_HASH_TABLE_IMPL LoaderRoutines[] = {
    NULL,
    PERFECT_HASH_ALGORITHM_TABLE_ENTRY(EXPAND_AS_LOAD_TABLE_IMPL_FUNC)
    NULL
};
VERIFY_ALGORITHM_ARRAY_SIZE(LoaderRoutines);

//
// Define the array of hash routines.
//

#define EXPAND_AS_HASH_ROUTINE(Name, NumberOfSeeds, SeedMasks) \
    PerfectHashTableHash##Name,

const PPERFECT_HASH_TABLE_HASH HashRoutines[] = {
    NULL,
    PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_HASH_ROUTINE)
    NULL
};
VERIFY_HASH_ARRAY_SIZE(HashRoutines);

//
// Define the array of number of seeds required per hash routine.
//

#define EXPAND_AS_HASH_NUMBER_OF_SEEDS(Name, NumberOfSeeds, SeedMasks) \
    NumberOfSeeds,

const SHORT HashRoutineNumberOfSeeds[] = {
    -1,
    PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_HASH_NUMBER_OF_SEEDS)
    -1,
};
VERIFY_HASH_ARRAY_SIZE(HashRoutineNumberOfSeeds);

//
// Define the array of SEED_MASKS for each hash routine.
//

#define EXPAND_AS_HASH_SEED_MASKS(Name, NumberOfSeeds, SeedMasksX) \
    const SEED_MASKS PerfectHashTableHash##Name##SeedMasks = SeedMasksX;

PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_HASH_SEED_MASKS)

#define EXPAND_AS_SEED_MASKS_PTR(Name, NumberOfSeeds, SeedMasksX) \
    &PerfectHashTableHash##Name##SeedMasks,

const PCSEED_MASKS HashRoutineSeedMasks[] = {
    NULL,
    PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_SEED_MASKS_PTR)
    NULL,
};
VERIFY_HASH_ARRAY_SIZE(HashRoutineSeedMasks);

//
// Define the array of seeded hash routines.
//

#define EXPAND_AS_SEEDED_HASH_ROUTINE(Name, NumberOfSeeds, SeedMasks) \
    PerfectHashTableSeededHash##Name,

const PPERFECT_HASH_TABLE_SEEDED_HASH SeededHashRoutines[] = {
    NULL,
    PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_SEEDED_HASH_ROUTINE)
    NULL
};
VERIFY_HASH_ARRAY_SIZE(SeededHashRoutines);

//
// Define the array of hash "Ex" routines.
//

#define EXPAND_AS_HASH_EX_ROUTINE(Name, NumberOfSeeds, SeedMasks) \
    PerfectHashTableHashEx##Name,

const PPERFECT_HASH_TABLE_HASH_EX HashExRoutines[] = {
    NULL,
    PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_HASH_EX_ROUTINE)
    NULL
};
VERIFY_HASH_ARRAY_SIZE(HashExRoutines);

//
// Define the array of seeded hash "Ex" routines.
//

#define EXPAND_AS_SEEDED_HASH_EX_ROUTINE(Name, NumberOfSeeds, SeedMasks) \
    PerfectHashTableSeededHashEx##Name,

const PPERFECT_HASH_TABLE_SEEDED_HASH_EX SeededHashExRoutines[] = {
    NULL,
    PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_SEEDED_HASH_EX_ROUTINE)
    NULL
};
VERIFY_HASH_ARRAY_SIZE(SeededHashExRoutines);

//
// Define the array of seeded hash16 "Ex" routines.
//

#define EXPAND_AS_SEEDED_HASH16_EX_ROUTINE(Name, NumberOfSeeds, SeedMasks) \
    PerfectHashTableSeededHash16Ex##Name,

const PPERFECT_HASH_TABLE_SEEDED_HASH16_EX SeededHash16ExRoutines[] = {
    NULL,
    PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_SEEDED_HASH16_EX_ROUTINE)
    NULL
};
VERIFY_HASH_ARRAY_SIZE(SeededHash16ExRoutines);

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
        PerfectHashHashCrc32Rotate15FunctionId,
        PerfectHashAndMaskFunctionId,
        PerfectHashTableFastIndexImplChm01Crc32Rotate15HashAndMask,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashJenkinsFunctionId,
        PerfectHashAndMaskFunctionId,
        PerfectHashTableFastIndexImplChm01JenkinsHashAndMask,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashCrc32RotateXFunctionId,
        PerfectHashAndMaskFunctionId,
        PerfectHashTableFastIndexImplChm01Crc32RotateXHashAndMask,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashCrc32RotateXYFunctionId,
        PerfectHashAndMaskFunctionId,
        PerfectHashTableFastIndexImplChm01Crc32RotateXYHashAndMask,
    },

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashCrc32RotateWXYZFunctionId,
        PerfectHashAndMaskFunctionId,
        PerfectHashTableFastIndexImplChm01Crc32RotateWXYZHashAndMask,
    },
};

const BYTE NumberOfFastIndexRoutines = ARRAYSIZE(FastIndexRoutines);

//
// Define the array of raw C string Index() implementations.
//

#include "CompiledPerfectHashTableIndexRoutines.h"
#undef RawCString

#define EXPAND_AS_CHM01_AND_INDEX_IMPL_TUPLE(Name, NumberOfSeeds, SeedMasks) \
    {                                                                        \
        PerfectHashChm01AlgorithmId,                                         \
        PerfectHashHash##Name##FunctionId,                                   \
        PerfectHashAndMaskFunctionId,                                        \
        &CompiledPerfectHashTableChm01Index##Name##AndCSourceRawCString,     \
    },


const PERFECT_HASH_TABLE_INDEX_IMPL_STRING_TUPLE IndexImplStringTuples[] = {

    PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(
        EXPAND_AS_CHM01_AND_INDEX_IMPL_TUPLE
    )

    //
    // Manually add our Jenkins modulus masking implementation; we only have
    // one of these as modulus masking just flat out doesn't work (and the
    // only reason this one exists is to try and aid debugging).
    //

    {
        PerfectHashChm01AlgorithmId,
        PerfectHashHashJenkinsFunctionId,
        PerfectHashModulusMaskFunctionId,
        &CompiledPerfectHashTableChm01IndexJenkinsModulusCSourceRawCString,
    },
};

const BYTE NumberOfIndexImplStrings = ARRAYSIZE(IndexImplStringTuples);

//
// The next section defines the UNICODE_STRING representations and supporting
// arrays of enum types.
//

//
// Random Number Generators
//

#define EXPAND_AS_RNG_NAME(Name, Upper) \
    const UNICODE_STRING PerfectHash##Name##RngName = RCS(L"" #Name);

PERFECT_HASH_RNG_TABLE_ENTRY(EXPAND_AS_RNG_NAME);

#define EXPAND_AS_RNG_NAME_PTR(Name, Upper) \
    &PerfectHash##Name##RngName,

const PCUNICODE_STRING RngNames[] = {
    NULL,
    PERFECT_HASH_RNG_TABLE_ENTRY(EXPAND_AS_RNG_NAME_PTR)
    NULL,
};

//
// CPU arch
//

#define EXPAND_AS_CPU_ARCH_NAME(Name, Upper) \
    const UNICODE_STRING PerfectHash##Name##CpuArchName = RCS(L"" #Name);

PERFECT_HASH_CPU_ARCH_TABLE_ENTRY(EXPAND_AS_CPU_ARCH_NAME);

#define EXPAND_AS_CPU_ARCH_NAME_PTR(Name, Upper) \
    &PerfectHash##Name##CpuArchName,

const PCUNICODE_STRING CpuArchNames[] = {
    NULL,
    PERFECT_HASH_CPU_ARCH_TABLE_ENTRY(EXPAND_AS_CPU_ARCH_NAME_PTR)
    NULL,
};

//
// Interface
//

#define EXPAND_AS_INTERFACE_NAME(Name, Upper, Guid) \
    const UNICODE_STRING PerfectHash##Name##InterfaceName = RCS(L"" #Name);

PERFECT_HASH_INTERFACE_TABLE_ENTRY(EXPAND_AS_INTERFACE_NAME);

#define EXPAND_AS_INTERFACE_NAME_PTR(Name, Upper, Guid) \
    &PerfectHash##Name##InterfaceName,

const PCUNICODE_STRING InterfaceNames[] = {
    NULL,
    PERFECT_HASH_INTERFACE_TABLE_ENTRY(EXPAND_AS_INTERFACE_NAME_PTR)
    NULL,
};

//
// Algorithm
//

#define EXPAND_AS_ALGORITHM_NAME(Name, Upper) \
    const UNICODE_STRING PerfectHash##Name##AlgorithmName = RCS(L"" #Name);

PERFECT_HASH_ALGORITHM_TABLE_ENTRY(EXPAND_AS_ALGORITHM_NAME);

#define EXPAND_AS_ALGORITHM_NAME_PTR(Name, Upper) \
    &PerfectHash##Name##AlgorithmName,

const PCUNICODE_STRING AlgorithmNames[] = {
    NULL,
    PERFECT_HASH_ALGORITHM_TABLE_ENTRY(EXPAND_AS_ALGORITHM_NAME_PTR)
    NULL,
};
VERIFY_ALGORITHM_ARRAY_SIZE(AlgorithmNames);

//
// Hash Function
//

#define EXPAND_AS_HASH_FUNCTION_NAME(Name, NumberOfSeeds, SeedMasks) \
    const UNICODE_STRING PerfectHashHash##Name##FunctionName = RCS(L"" #Name);

PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_HASH_FUNCTION_NAME)

#define EXPAND_AS_HASH_FUNCTION_NAME_PTR(Name, NumberOfSeeds, SeedMasks) \
    &PerfectHashHash##Name##FunctionName,

const PCUNICODE_STRING HashFunctionNames[] = {
    NULL,
    PERFECT_HASH_HASH_FUNCTION_TABLE_ENTRY(EXPAND_AS_HASH_FUNCTION_NAME_PTR)
    NULL,
};
VERIFY_HASH_ARRAY_SIZE(HashFunctionNames);

//
// Mask Function
//

#define EXPAND_AS_MASK_FUNCTION_NAME(Name, Upper) \
    const UNICODE_STRING PerfectHash##Name##MaskFunctionName = RCS(L"" #Name);

PERFECT_HASH_MASK_FUNCTION_TABLE_ENTRY(EXPAND_AS_MASK_FUNCTION_NAME);

#define EXPAND_AS_MASK_FUNCTION_NAME_PTR(Name, Upper) \
    &PerfectHash##Name##MaskFunctionName,

const PCUNICODE_STRING MaskFunctionNames[] = {
    NULL,
    PERFECT_HASH_MASK_FUNCTION_TABLE_ENTRY(EXPAND_AS_MASK_FUNCTION_NAME_PTR)
    NULL,
};
VERIFY_MASK_ARRAY_SIZE(MaskFunctionNames);

//
// Best Coverage Type
//

#define EXPAND_AS_BEST_COVERAGE_TYPE_NAME(Name, Comparison, Comparator)    \
    const UNICODE_STRING                                                   \
        PerfectHash##Comparison##Name##BestCoverageTypeName = RCS(L"" #Name);

BEST_COVERAGE_TYPE_TABLE_ENTRY(EXPAND_AS_BEST_COVERAGE_TYPE_NAME);

#define EXPAND_AS_BEST_COVERAGE_TYPE_NAME_PTR(Name, Comparison, Comparator) \
    &PerfectHash##Comparison##Name##BestCoverageTypeName,

const PCUNICODE_STRING BestCoverageTypeNames[] = {
    NULL,
    BEST_COVERAGE_TYPE_TABLE_ENTRY(EXPAND_AS_BEST_COVERAGE_TYPE_NAME_PTR)
    NULL,
};

#define EXPAND_AS_BEST_COVERAGE_TYPE_NAMEA(Name, Comparison, Comparator) \
    RCS(#Comparison#Name),

const STRING BestCoverageTypeNamesA[] = {
    RCS("N/A"),
    BEST_COVERAGE_TYPE_TABLE_ENTRY(EXPAND_AS_BEST_COVERAGE_TYPE_NAMEA)
    RCS("N/A"),
};

//
// Table Create Parameter
//

#define EXPAND_AS_TABLE_CREATE_PARAMETER_NAME(Name)                    \
    const UNICODE_STRING PerfectHash##Name##TableCreateParameterName = \
        RCS(L"" #Name);

TABLE_CREATE_PARAMETER_TABLE_ENTRY(EXPAND_AS_TABLE_CREATE_PARAMETER_NAME);

#define EXPAND_AS_TABLE_CREATE_PARAMETER_NAME_PTR(Name) \
    &PerfectHash##Name##TableCreateParameterName,

const PCUNICODE_STRING TableCreateParameterNames[] = {
    NULL,
    TABLE_CREATE_PARAMETER_TABLE_ENTRY(
        EXPAND_AS_TABLE_CREATE_PARAMETER_NAME_PTR
    )
    NULL,
};

//
// Define an array of enum ID names.
//

#define EXPAND_AS_ENUM_ID_NAMES(Name, Upper) Name##Names,

const PCUNICODE_STRING *EnumIdNames[] = {
    NULL,
    PERFECT_HASH_ENUM_TABLE_ENTRY(EXPAND_AS_ENUM_ID_NAMES)
    NULL,
};

//
// Array of UNICODE_STRING event prefix names used by the runtime context.
//

const UNICODE_STRING ContextShutdownEventPrefix =
    RCS(L"PerfectHashContext_ShutdownEvent_");

const UNICODE_STRING ContextSucceededEventPrefix =
    RCS(L"PerfectHashContext_SucceededEvent_");

const UNICODE_STRING ContextFailedEventPrefix =
    RCS(L"PerfectHashContext_FailedEvent_");

const UNICODE_STRING ContextCompletedEventPrefix =
    RCS(L"PerfectHashContext_CompletedEvent_");

const UNICODE_STRING ContextTryLargerTableSizeEventPrefix =
    RCS(L"PerfectHashContext_TryLargerTableSizeEvent_");

const UNICODE_STRING ContextVerifiedTableEventPrefix =
    RCS(L"PerfectHashContext_VerifiedTableEvent_");

const UNICODE_STRING ContextNewBestGraphEventPrefix =
    RCS(L"PerfectHashContext_NewBestGraphEvent_");

#define EXPAND_AS_EVENT_NAME(                                           \
    Verb, VUpper, Name, Upper,                                          \
    EofType, EofValue,                                                  \
    Suffix, Extension, Stream, Base                                     \
)                                                                       \
    const UNICODE_STRING Context##Verb##d##Name##EventPrefix =          \
        RCS(L"PerfectHashContext_" L"" #Verb L"d" L"" #Name L"EventPrefix_");

PREPARE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_EVENT_NAME);
SAVE_FILE_WORK_TABLE_ENTRY(EXPAND_AS_EVENT_NAME);

#define EXPAND_AS_EVENT_NAME_ADDRESS(     \
    Verb, VUpper, Name, Upper,            \
    EofType, EofValue,                    \
    Suffix, Extension, Stream, Base       \
)                                         \
    &Context##Verb##d##Name##EventPrefix,

const PCUNICODE_STRING ContextObjectPrefixes[] = {
    &ContextShutdownEventPrefix,
    &ContextSucceededEventPrefix,
    &ContextFailedEventPrefix,
    &ContextCompletedEventPrefix,
    &ContextTryLargerTableSizeEventPrefix,
    &ContextVerifiedTableEventPrefix,
    &ContextNewBestGraphEventPrefix,

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

const UNICODE_STRING No = RCS(L"No.\n");
const UNICODE_STRING Yes = RCS(L"Yes.\n");
const UNICODE_STRING CsvSuffix = RCS(L".csv");
const UNICODE_STRING CsvExtension = RCS(L"csv");
const UNICODE_STRING KeysExtension = RCS(L"keys");
const UNICODE_STRING DotKeysSuffix = RCS(L".keys");
const UNICODE_STRING DotTableSuffix = RCS(L".pht1");
const UNICODE_STRING DotCHeaderSuffix = RCS(L".h");
const UNICODE_STRING DotCSourceSuffix = RCS(L".c");
const UNICODE_STRING NullUnicodeString = RCS(L"");
const UNICODE_STRING KeysWildcardSuffix = RCS(L"*.keys");
const UNICODE_STRING KeysTableSizeSuffix = RCS(L".TableSize");
const UNICODE_STRING TableValuesSuffix = RCS(L"_TableValues_");
const UNICODE_STRING TableValuesExtension = RCS(L"bin");

const UNICODE_STRING PerfectHashBulkCreateCsvBaseName =
    RCS(L"PerfectHashBulkCreate");

const UNICODE_STRING PerfectHashBulkCreateBestCsvBaseName =
    RCS(L"PerfectHashBulkCreateBest");

const UNICODE_STRING PerfectHashTableCreateCsvBaseName =
    RCS(L"PerfectHashTableCreate");

const UNICODE_STRING PerfectHashTableCreateBestCsvBaseName =
    RCS(L"PerfectHashTableCreateBest");

const STRING NullString = RCS("");
const STRING DotExeSuffixA = RCS(".exe");
const STRING DotDllSuffixA = RCS(".dll");
const STRING DotLibSuffixA = RCS(".lib");
const STRING DynamicLibraryConfigurationTypeA = RCS("DynamicLibrary");
const STRING ApplicationConfigurationTypeA = RCS("Application");
const STRING FunctionHookCallbackDefaultFunctionNameA =
    RCS("InterlockedIncrement");

//
// VCProject and Makefile string constants.  Consumed by the file work callback
// routines (e.g. Chm01FileWorkVCProjectDllFile.c).
//
// N.B. These are STRING, not UNICODE_STRING, as they're written to ASCII text
//      files (not used as UTF-16 path names).
//

const STRING LibTargetPrefix = RCS("lib");
const STRING TestTargetPrefix = RCS("Test_");
const STRING BenchmarkFullTargetPrefix = RCS("BenchmarkFull_");
const STRING BenchmarkIndexTargetPrefix = RCS("BenchmarkIndex_");

const STRING SoFileSuffix = RCS("So");
const STRING LibFileSuffix = RCS("Lib");
const STRING DllFileSuffix = RCS("Dll");
const STRING TestFileSuffix = RCS("Test");
const STRING TestExeFileSuffix = RCS("TestExe");
const STRING BenchmarkFullFileSuffix = RCS("BenchmarkFullExe");
const STRING BenchmarkIndexFileSuffix = RCS("BenchmarkIndexExe");

//
// Stream names.
//

const UNICODE_STRING TableInfoStreamName = RCS(L"Info");

//
// Extensions.
//

const UNICODE_STRING TextFileExtension = RCS(L"txt");
const UNICODE_STRING BatchFileExtension = RCS(L"bat");
const UNICODE_STRING CSourceFileExtension = RCS(L"c");
const UNICODE_STRING CHeaderFileExtension = RCS(L"h");
const UNICODE_STRING TableFileExtension = RCS(L"pht1");
const UNICODE_STRING ModuleDefFileExtension = RCS(L"def");
const UNICODE_STRING VCPropsFileExtension = RCS(L"props");
const UNICODE_STRING MakefileMkFileExtension = RCS(L"mk");
const UNICODE_STRING VCProjectFileExtension = RCS(L"vcxproj");
const UNICODE_STRING VSSolutionFileExtension = RCS(L"sln");

#define VERIFY_FILE_WORK_ARRAY_SIZE(Name) \
    C_ASSERT(ARRAYSIZE(Name) == NUMBER_OF_FILES + 2)

#define VERIFY_CONTEXT_FILE_WORK_ARRAY_SIZE(Name) \
    C_ASSERT(ARRAYSIZE(Name) == NUMBER_OF_CONTEXT_FILES + 2)

#define EXPAND_AS_FILE_WORK_ITEM_EXTENSION( \
    Verb, VUpper, Name, Upper,              \
    EofType, EofValue,                      \
    Suffix, Extension, Stream, Base         \
)                                           \
    Extension,

const PCUNICODE_STRING FileWorkItemExtensions[] = {
    NULL,
    FILE_WORK_TABLE_ENTRY(EXPAND_AS_FILE_WORK_ITEM_EXTENSION)
    NULL,
};
VERIFY_FILE_WORK_ARRAY_SIZE(FileWorkItemExtensions);

//
// Suffixes.
//

#define EXPAND_AS_FILE_WORK_ITEM_SUFFIX( \
    Verb, VUpper, Name, Upper,           \
    EofType, EofValue,                   \
    Suffix, Extension, Stream, Base      \
)                                        \
    RCS(Suffix),

const UNICODE_STRING FileWorkItemSuffixes[] = {
    NULL_UNICODE_STRING,
    FILE_WORK_TABLE_ENTRY(EXPAND_AS_FILE_WORK_ITEM_SUFFIX)
    NULL_UNICODE_STRING,
};
VERIFY_FILE_WORK_ARRAY_SIZE(FileWorkItemSuffixes);

//
// Stream names.
//

#define EXPAND_AS_FILE_WORK_ITEM_STREAM_NAME( \
    Verb, VUpper, Name, Upper,                \
    EofType, EofValue,                        \
    Suffix, Extension, Stream, Base           \
)                                             \
    RCS(Stream),

const UNICODE_STRING FileWorkItemStreamNames[] = {
    NULL_UNICODE_STRING,
    FILE_WORK_TABLE_ENTRY(EXPAND_AS_FILE_WORK_ITEM_STREAM_NAME)
    NULL_UNICODE_STRING,
};
VERIFY_FILE_WORK_ARRAY_SIZE(FileWorkItemStreamNames);

//
// Base names.
//
// N.B. These are only used for context files (i.e. CONTEXT_FILE_ID).
//

#define EXPAND_AS_FILE_WORK_BASE_NAME( \
    Verb, VUpper, Name, Upper,         \
    EofType, EofValue,                 \
    Suffix, Extension, Stream, Base    \
)                                      \
    RCS(Base),

const UNICODE_STRING FileWorkItemBaseNames[] = {
    NULL_UNICODE_STRING,
    FILE_WORK_TABLE_ENTRY(EXPAND_AS_FILE_WORK_BASE_NAME)
    NULL_UNICODE_STRING,
};
VERIFY_FILE_WORK_ARRAY_SIZE(FileWorkItemBaseNames);

//
// End-of-file initializers.
//

#define EXPAND_AS_FILE_WORK_ITEM_EOF_INIT( \
    Verb, VUpper, Name, Upper,             \
    EofType, EofValue,                     \
    Suffix, Extension, Stream, Base        \
)                                          \
    { EofType, EofValue },


const EOF_INIT EofInits[] = {
    { EofInitTypeNull, },
    FILE_WORK_TABLE_ENTRY(EXPAND_AS_FILE_WORK_ITEM_EOF_INIT)
    { EofInitTypeInvalid, },
};
VERIFY_FILE_WORK_ARRAY_SIZE(EofInits);

//
// C type names corresponding to TYPE enum values.
//

const STRING CTypeNames[] = {
    RCS("unsigned char"),
    RCS("unsigned short"),
    RCS("unsigned int"),
    RCS("unsigned __int64"),
    RCS("__m128i"),
    RCS("__m256i"),
    RCS("__m512i"),
    { 0, },
};

//
// NT-style C type names corresponding to TYPE enum values.
//

const STRING NtTypeNames[] = {
    RCS("BYTE"),
    RCS("USHORT"),
    RCS("ULONG"),
    RCS("ULONGLONG"),
    RCS("XMMWORD"),
    RCS("YMMWORD"),
    RCS("ZMMWORD"),
    { 0, },
};

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

#ifdef PH_WINDOWS
#define NUMBER_OF_INTERFACES 14
#else

//
// Account for no CU.
//

#define NUMBER_OF_INTERFACES 13
#endif

#define EXPECTED_ARRAY_SIZE NUMBER_OF_INTERFACES+2
#define VERIFY_ARRAY_SIZE(Name) C_ASSERT(ARRAYSIZE(Name) == EXPECTED_ARRAY_SIZE)

C_ASSERT(EXPECTED_ARRAY_SIZE == PerfectHashInvalidInterfaceId+1);
C_ASSERT(NUMBER_OF_INTERFACES == PerfectHashLastInterfaceId);
C_ASSERT(NUMBER_OF_INTERFACES == PerfectHashInvalidInterfaceId-1);

//
// Add dummy defines for components whose struct name isn't prefixed with
// PERFECT_HASH_; this allows us to use the X-macro below properly.
//

#define PERFECT_HASH_IUNKNOWN IUNKNOWN
#define PERFECT_HASH_ICLASSFACTORY ICLASSFACTORY
#define PERFECT_HASH_RTL RTL
#define PERFECT_HASH_ALLOCATOR ALLOCATOR
#define PERFECT_HASH_GUARDED_LIST GUARDED_LIST
#define PERFECT_HASH_GRAPH GRAPH
#define PERFECT_HASH_RNG RNG

#define PERFECT_HASH_IUNKNOWN_VTBL IUNKNOWN_VTBL
#define PERFECT_HASH_ICLASSFACTORY_VTBL ICLASSFACTORY_VTBL
#define PERFECT_HASH_RTL_VTBL RTL_VTBL
#define PERFECT_HASH_ALLOCATOR_VTBL ALLOCATOR_VTBL
#define PERFECT_HASH_GUARDED_LIST_VTBL GUARDED_LIST_VTBL
#define PERFECT_HASH_GRAPH_VTBL GRAPH_VTBL
#define PERFECT_HASH_RNG_VTBL RNG_VTBL

#ifdef PH_WINDOWS
#define PERFECT_HASH_CU CU
#define PERFECT_HASH_CU_VTBL CU_VTBL
#endif

#define EXPAND_AS_SIZEOF_COMPONENT(Name, Upper, Guid) \
    sizeof(PERFECT_HASH_##Upper),

const USHORT ComponentSizes[] = {
    0,
    PERFECT_HASH_INTERFACE_TABLE_ENTRY(EXPAND_AS_SIZEOF_COMPONENT)
    0,
};
VERIFY_ARRAY_SIZE(ComponentSizes);

#define EXPAND_AS_SIZEOF_VTBL(Name, Upper, Guid) \
    sizeof(PERFECT_HASH_##Upper##_VTBL),

const USHORT ComponentInterfaceSizes[] = {
    0,
    PERFECT_HASH_INTERFACE_TABLE_ENTRY(EXPAND_AS_SIZEOF_VTBL)
    0,
};
VERIFY_ARRAY_SIZE(ComponentInterfaceSizes);

//
// N.B. We use -1 for invalid offsets instead of 0, as 0 could be a legitimate
// field offset if the member is the first element in the structure.
//

#define EXPAND_AS_COMPONENT_INTERFACE_OFFSET(Name, Upper, Guid) \
    (SHORT)FIELD_OFFSET(PERFECT_HASH_##Upper, Interface),

const SHORT ComponentInterfaceOffsets[] = {
    -1,
    PERFECT_HASH_INTERFACE_TABLE_ENTRY(EXPAND_AS_COMPONENT_INTERFACE_OFFSET)
    -1,
};
VERIFY_ARRAY_SIZE(ComponentInterfaceOffsets);

//
// N.B. Not all components have a TLS context entry, so we don't use an X-macro
//      expansion for the following array definition.
//

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
#ifdef PH_WINDOWS
    (SHORT)FIELD_OFFSET(PERFECT_HASH_TLS_CONTEXT, Cu),
#endif
    -1, // Rng

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
#ifdef PH_WINDOWS
    (SHORT)FIELD_OFFSET(GLOBAL_COMPONENTS, Cu),
#endif
    -1, // Rng

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
    &PerfectHashContextBulkCreate,
    &PerfectHashContextBulkCreateArgvW,
    &PerfectHashContextExtractBulkCreateArgsFromArgvW,
    &PerfectHashContextTableCreate,
    &PerfectHashContextTableCreateArgvW,
    &PerfectHashContextExtractTableCreateArgsFromArgvW,
#ifndef PH_WINDOWS
    &PerfectHashContextTableCreateArgvA,
#endif
};
#ifdef PH_WINDOWS
VERIFY_VTBL_SIZE(PERFECT_HASH_CONTEXT, 13);
#else
VERIFY_VTBL_SIZE(PERFECT_HASH_CONTEXT, 14);
#endif

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
    NULL,   // HashEx
    NULL,   // SeededHashEx
    NULL,   // SeededHash16Ex
};
VERIFY_VTBL_SIZE(PERFECT_HASH_TABLE, 22);

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


#ifdef PH_WINDOWS

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
#endif

//
// Graph
//

GRAPH_VTBL GraphInterface = {
    (PGRAPH_QUERY_INTERFACE)&ComponentQueryInterface,
    (PGRAPH_ADD_REF)&ComponentAddRef,
    (PGRAPH_RELEASE)&ComponentRelease,
    (PGRAPH_CREATE_INSTANCE)&ComponentCreateInstance,
    (PGRAPH_LOCK_SERVER)&ComponentLockServer,
    &GraphSetInfo,
    &GraphEnterSolvingLoop,
    &GraphVerify,
    &GraphLoadInfo,
    &GraphReset,
    &GraphLoadNewSeeds,
    &GraphSolve,
    &GraphIsAcyclic,
    &GraphAssign,
    &GraphCalculateAssignedMemoryCoverage,
    &GraphCalculateAssignedMemoryCoverageForKeysSubset,
    &GraphRegisterSolved,
    &GraphShouldWeContinueTryingToSolve,
    &GraphAddKeys,
    &GraphHashKeys,
};
VERIFY_VTBL_SIZE(GRAPH, 15);

#ifdef PH_WINDOWS

//
// Cu
//

CU_VTBL CuInterface = {
    (PCU_QUERY_INTERFACE)&ComponentQueryInterface,
    (PCU_ADD_REF)&ComponentAddRef,
    (PCU_RELEASE)&ComponentRelease,
    (PCU_CREATE_INSTANCE)&ComponentCreateInstance,
    (PCU_LOCK_SERVER)&ComponentLockServer,
    &LoadCuDeviceAttributes,
};
VERIFY_VTBL_SIZE(CU, 1);

#endif

//
// Rng
//

RNG_VTBL RngInterface = {
    (PRNG_QUERY_INTERFACE)&ComponentQueryInterface,
    (PRNG_ADD_REF)&ComponentAddRef,
    (PRNG_RELEASE)&ComponentRelease,
    (PRNG_CREATE_INSTANCE)&ComponentCreateInstance,
    (PRNG_LOCK_SERVER)&ComponentLockServer,
    &RngInitializePseudo,
    &RngGenerateRandomBytes,
    &RngGetCurrentOffset,
};
VERIFY_VTBL_SIZE(RNG, 3);

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
#ifdef PH_WINDOWS
    &CuInterface,
#endif
    &RngInterface,

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
#ifdef PH_WINDOWS
    (PCOMPONENT_INITIALIZE)&CuInitialize,
#endif
    (PCOMPONENT_INITIALIZE)&RngInitialize,

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
#ifdef PH_WINDOWS
    (PCOMPONENT_RUNDOWN)&CuRundown,
#endif
    (PCOMPONENT_RUNDOWN)&RngRundown,

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
#include "CompiledPerfectHashTableTypesPre_CHeader_RawCString.h"
#include "CompiledPerfectHashTableTypesPost_CHeader_RawCString.h"

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab nowrap                              :
