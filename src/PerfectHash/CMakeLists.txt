set(target PerfectHash)

# ###############################################################################
# Source groups
# ###############################################################################
set(Private_Header_Files
    "BitManipulation.h"
    "BulkCreateBestCsv.h"
    "BulkCreateCsv.h"
    "Chm01.h"
    "Chm01FileWork.h"
    "Chm01Private.h"
    "Chm02Private.h"
    "Chunk.h"
    "Component.h"
    "Cu.h"
    "ExtractArg.h"
    "Graph.h"
    "GraphCounters.h"
    "GraphImpl.h"
    "GraphImpl2.h"
    "GraphImpl3.h"
    "GuardedList.h"
    "Math.h"
    "PerfectHashAllocator.h"
    "PerfectHashConstants.h"
    "PerfectHashContext.h"
    "PerfectHashCu.h"
    "PerfectHashDirectory.h"
    "PerfectHashErrorHandling.h"
    "PerfectHashEventsPrivate.h"
    "PerfectHashFile.h"
    "PerfectHashFileWork.h"
    "PerfectHashKeys.h"
    "PerfectHashNames.h"
    "PerfectHashPath.h"
    "PerfectHashPrimes.h"
    "PerfectHashPrivate.h"
    "PerfectHashTable.h"
    "PerfectHashTimestamp.h"
    "PerfectHashTls.h"
    "Rng.h"
    "Rtl.h"
    "RtlOutput.h"
    "Security.h"
    "stdafx.h"
    "TableCreateBestCsv.h"
    "TableCreateCsv.h"
    "targetver.h"
    "VCProjectFileChunks.h"
    "Cu.cuh"
    "GraphCu.h"
    "PerfectHashCu.h"
    "CuDeviceAttributes.h"
    "Debugger.h"
)
source_group("Private Header Files" FILES ${Private_Header_Files})

set(Private_Header_Files_28Auto-Generated29
    "CompiledPerfectHash_CHeader_RawCString.h"
    "CompiledPerfectHash_VCProps_RawCString.h"
    "CompiledPerfectHashMacroGlue_CHeader_RawCString.h"
    "CompiledPerfectHashTableBenchmarkFull_CSource_RawCString.h"
    "CompiledPerfectHashTableBenchmarkFullExe_CSource_RawCString.h"
    "CompiledPerfectHashTableBenchmarkIndex_CSource_RawCString.h"
    "CompiledPerfectHashTableBenchmarkIndexExe_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexAddSubXorAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexCrc32And_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexCrc32NotAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexCrc32Rotate15And_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexCrc32RotateWXYZAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexCrc32RotateXAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexCrc32RotateXorAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexCrc32RotateXYAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexDjbAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexDjbXorAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexDummyAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexFnvAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexJenkinsAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexJenkinsModAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexJenkinsModulus_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiply643ShiftRAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiply644ShiftRAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiplyAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiplyRotateLRAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiplyRotateR2And_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiplyRotateRAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiplyRotateRMultiplyAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiplyShiftLRAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiplyShiftR2And_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiplyShiftRAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiplyShiftRMultiplyAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiplyShiftRXAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMultiplyXorAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexRotateMultiplyXorRotate2And_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexRotateMultiplyXorRotateAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexRotateRMultiplyAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexRotateRMultiplyRotateRAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexRotateXorAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexScratchAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexShiftMultiplyXorShift2And_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexShiftMultiplyXorShiftAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMulshrolate1RXAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMulshrolate2RXAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMulshrolate3RXAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexMulshrolate4RXAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableChm01IndexXorAnd_CSource_RawCString.h"
    "CompiledPerfectHashTableIndexRoutines.h"
    "CompiledPerfectHashTableRoutines_CSource_RawCString.h"
    "CompiledPerfectHashTableRoutinesPost_CSource_RawCString.h"
    "CompiledPerfectHashTableRoutinesPre_CSource_RawCString.h"
    "CompiledPerfectHashTableSupport_CHeader_RawCString.h"
    "CompiledPerfectHashTableSupport_CSource_RawCString.h"
    "CompiledPerfectHashTableTest_CSource_RawCString.h"
    "CompiledPerfectHashTableTestExe_CSource_RawCString.h"
    "CompiledPerfectHashTableTypesPost_CHeader_RawCString.h"
    "CompiledPerfectHashTableTypesPre_CHeader_RawCString.h"
    "CompiledPerfectHashTableBenchmarkIndexInline_CSource_RawCString.h"
    "Makefile_Makefile_RawCString.h"
    "no_sal2_CHeader_RawCString.h"
)
source_group("Private Header Files %28Auto-Generated%29" FILES ${Private_Header_Files_28Auto-Generated29})

set(Public_Header_Files
    "../../include/CompiledPerfectHash.h"
    "../../include/PerfectHash.h"
    "../../include/PerfectHashErrors.h"
    "../../include/PerfectHashEvents.h"
)
source_group("Public Header Files" FILES ${Public_Header_Files})

set(Resource_Files
    "PerfectHash.def"
    "PerfectHash.rc"
    "PerfectHashEvents.man"
)
source_group("Resource Files" FILES ${Resource_Files})

set(Source_Files
    "BitManipulation.c"
    "Chm01Shared.c"
    "Chm01FileWork.c"
    "Chm01FileWorkBatchBuildSolutionFile.c"
    "Chm01FileWorkCHeaderCompiledPerfectHashFile.c"
    "Chm01FileWorkCHeaderCompiledPerfectHashMacroGlueFile.c"
    "Chm01FileWorkCHeaderFile.c"
    "Chm01FileWorkCHeaderNoSal2File.c"
    "Chm01FileWorkCHeaderStdAfxFile.c"
    "Chm01FileWorkCHeaderSupportFile.c"
    "Chm01FileWorkCHeaderTypesFile.c"
    "Chm01FileWorkCSourceBenchmarkFullExeFile.c"
    "Chm01FileWorkCSourceBenchmarkFullFile.c"
    "Chm01FileWorkCSourceBenchmarkIndexExeFile.c"
    "Chm01FileWorkCSourceBenchmarkIndexFile.c"
    "Chm01FileWorkCSourceDownsizedKeysFile.c"
    "Chm01FileWorkCSourceFile.c"
    "Chm01FileWorkCSourceKeysFile.c"
    "Chm01FileWorkCSourceStdAfxFile.c"
    "Chm01FileWorkCSourceSupportFile.c"
    "Chm01FileWorkCSourceTableDataFile.c"
    "Chm01FileWorkCSourceTableValuesFile.c"
    "Chm01FileWorkCSourceTestExeFile.c"
    "Chm01FileWorkCSourceTestFile.c"
    "Chm01FileWorkCppSourceUnityFile.c"
    "Chm01FileWorkGraphVertexPairsFile.c"
    "Chm01FileWorkGraphVertices3File.c"
    "Chm01FileWorkMakefileBenchmarkFullMkFile.c"
    "Chm01FileWorkMakefileBenchmarkIndexMkFile.c"
    "Chm01FileWorkMakefileFile.c"
    "Chm01FileWorkMakefileLibMkFile.c"
    "Chm01FileWorkMakefileMainMkFile.c"
    "Chm01FileWorkMakefileSoMkFile.c"
    "Chm01FileWorkMakefileTestMkFile.c"
    "Chm01FileWorkModuleDefFile.c"
    "Chm01FileWorkTableFile.c"
    "Chm01FileWorkTableInfoStream.c"
    "Chm01FileWorkVCProjectBenchmarkFullExeFile.c"
    "Chm01FileWorkVCProjectBenchmarkIndexExeFile.c"
    "Chm01FileWorkVCProjectDllFile.c"
    "Chm01FileWorkVCProjectTestExeFile.c"
    "Chm01FileWorkVCPropsCompiledPerfectHashFile.c"
    "Chm01FileWorkVSSolutionFile.c"
    "Chm01FileWorkCMakeListsTextFile.c"
    "Chm01FileWorkCudaSourceFile.c"
    "Chm01Index.c"
    "Chunk.c"
    "Component.c"
    "dllmain.c"
    "Graph.c"
    "GraphImpl1.c"
    "GraphImpl2.c"
    "GraphImpl3.c"
    "GuardedList.c"
    "Math.c"
    "PerfectHashAllocator.c"
    "PerfectHashContext.c"
    "PerfectHashContextTableCreate.c"
    "PerfectHashKeys.c"
    "PerfectHashKeysLoad.c"
    "PerfectHashKeysLoadTableSize.c"
    "PerfectHashNames.c"
    "PerfectHashPath.c"
    "PerfectHashPrimes.c"
    "PerfectHashTable.c"
    "PerfectHashTableCreate.c"
    "PerfectHashTableDelete.c"
    "PerfectHashTableHash.c"
    "PerfectHashTableHash16Ex.c"
    "PerfectHashTableHashEx.c"
    "PerfectHashTableInsert.c"
    "PerfectHashTableLoad.c"
    "PerfectHashTableLookup.c"
    "PerfectHashTableMask.c"
    "PerfectHashTableNames.c"
    "PerfectHashTableTest.c"
    "PerfectHashTls.c"
    "RtlBuffers.c"
    "RtlLargePages.c"
    "PerfectHashFile.c"
    "PerfectHashDirectory.c"
    "PerfectHashErrorHandling.c"
    "PerfectHashContextBulkCreate.c"
    "Rng.c"
    "RngPhilox4x32.c"
    "Rtl.c"
    "RtlMemory.c"
    "RtlOutput.c"
    "stdafx.c"
    "VCProjectFileChunks.c"
    "ExtractArg.c"
    "PerfectHashConstants.c"
    "_pow.c"
    "_dtoa.c"
    "_strtod.c"
    "Cu.c"
    "PerfectHashCu.c"
    "SymbolLoader.c"
)

if(PERFECTHASH_HAS_CUDA)
    list(
        APPEND Source_Files
        "Chm02Shared.c"
        "GraphCu.c"
    )
endif()

if(WIN32)
    list(
        APPEND Source_Files
        "Chm01.c"
        "RtlUuid.c"
        "Security.c"
        "GraphAvx.c"
        "RtlRandom.c"
        "RtlPrivilege.c"
        "PerfectHashCu.c"
        "RtlErrorHandling.c"
        "PerfectHashTableCompile.c"
        "PerfectHashContextBulkCreate.c"
    )

    if(PERFECTHASH_HAS_CUDA)
        list(
            APPEND Source_Files
            "Chm02.c"
        )
    endif()

else()
    list(
        APPEND Source_Files
        "mem.c"
        "bitmap.c"
        "thpool.c"
        "bsthreadpool.cpp"
        "unicode.c"
        "Debugger.c"
        "Chm01Compat.c"
        "PerfectHashCompat.c"
    )

    if(PERFECTHASH_HAS_CUDA)
        list(
            APPEND Source_Files
            "Chm02Compat.c"
        )
    endif()
endif()

source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Private_Header_Files}
    ${Private_Header_Files_28Auto-Generated29}
    ${Public_Header_Files}
    ${Source_Files}
)

if(WIN32)
    list(
        APPEND ALL_FILES
        ${Resource_Files}
    )

    list(
        APPEND ALL_FILES
        ${Assembly_Source_Files}
    )

    set(OUTPUT_FILE "${PERFECTHASH_ROOT_DIR}/include/PerfectHashErrors.h")

    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND mc.exe -v -o -c -b -h "${PERFECTHASH_ROOT_DIR}/include" -x .
                -km PerfectHashEvents.man PerfectHashErrors.mc
        DEPENDS PerfectHashErrors.mc PerfectHashEvents.man
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Compiling PerfectHashErrors.mc and PerfectHashEvents.man..."
    )

    add_custom_target(PerfectHashResources ALL DEPENDS ${OUTPUT_FILE})
endif()

add_library(${target} SHARED ${ALL_FILES})

# ###############################################################################
# Target
# ###############################################################################
perfecthash_apply_common_settings(${target})

# ###############################################################################
# Compile definitions
# ###############################################################################
target_compile_definitions(
    ${target}
    PRIVATE
    "_PERFECT_HASH_INTERNAL_BUILD"
)

# ###############################################################################
# Compile and link options
# ###############################################################################

# ###############################################################################
# Dependencies
# ###############################################################################

# Link with other targets.
if(PERFECTHASH_ENABLE_PENTER)
    target_link_libraries(${target} PUBLIC FunctionHook)
endif()

if(WIN32)
    add_dependencies(${target} PerfectHashAsm)
    set(ADDITIONAL_LIBRARY_DEPENDENCIES rpcrt4 PerfectHashAsm)
    target_link_libraries(${target} PUBLIC "${ADDITIONAL_LIBRARY_DEPENDENCIES}")
endif()

# vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
