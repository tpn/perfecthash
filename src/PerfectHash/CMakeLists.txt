set(target PerfectHash)

if(NOT DEFINED PERFECTHASH_HAS_RAWDOG_JIT)
    set(PERFECTHASH_HAS_RAWDOG_JIT OFF CACHE BOOL "RawDog JIT availability")
endif()

# ###############################################################################
# Source groups
# ###############################################################################
set(Private_Header_Files
    "BitManipulation.h"
    "BulkCreateBestCsv.h"
    "BulkCreateCsv.h"
    "ChmOnline01.h"
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
    "PerfectHashOnline.h"
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
    "ChmOnline01Stub.c"
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
    "Chm01FileWorkPythonFile.c"
    "Chm01FileWorkPythonTestFile.c"
    "Chm01FileWorkCppHeaderOnlyFile.c"
    "Chm01FileWorkCppHeaderOnlyTestFile.c"
    "Chm01FileWorkRustCargoTomlFile.c"
    "Chm01FileWorkRustLibFile.c"
    "Chm01FileWorkRustTestFile.c"
    "Chm01FileWorkRustBenchFile.c"
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
    "PerfectHashOnline.c"
    "PerfectHashOnlineCompile.c"
    "PerfectHashOnlineCreate.c"
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

if((PERFECTHASH_IS_WINDOWS OR PERFECTHASH_IS_LINUX) AND PERFECTHASH_ARCH_X86_64)
    list(
        APPEND Source_Files
        "GraphAvx.c"
    )
endif()

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
        "PerfectHashTableCompileCompat.c"
    )

    if(PERFECTHASH_HAS_CUDA)
        list(
            APPEND Source_Files
            "Chm02Compat.c"
        )
    endif()
endif()

if(PERFECTHASH_IS_LINUX)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    set(PERFECTHASH_MESSAGE_BIN
        "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashErrors_English.bin"
    )
    set(PERFECTHASH_MESSAGE_TABLE_C
        "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashErrors_EnglishBin.c"
    )
    set(PERFECTHASH_MESSAGE_TABLE_H
        "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashErrors_EnglishBin.h"
    )

    add_custom_command(
        OUTPUT ${PERFECTHASH_MESSAGE_TABLE_C} ${PERFECTHASH_MESSAGE_TABLE_H}
        COMMAND "${Python3_EXECUTABLE}"
                "${PERFECTHASH_ROOT_DIR}/scripts/generate_message_table.py"
                --input "${PERFECTHASH_MESSAGE_BIN}"
                --output-c "${PERFECTHASH_MESSAGE_TABLE_C}"
                --output-h "${PERFECTHASH_MESSAGE_TABLE_H}"
        DEPENDS
            "${PERFECTHASH_MESSAGE_BIN}"
            "${PERFECTHASH_ROOT_DIR}/scripts/generate_message_table.py"
        COMMENT "Generating PerfectHash message table for Linux..."
        VERBATIM
    )

    list(APPEND Source_Files "${PERFECTHASH_MESSAGE_TABLE_C}")
    list(APPEND Private_Header_Files "${PERFECTHASH_MESSAGE_TABLE_H}")

    if(PERFECTHASH_ARCH_X86_64)
        find_program(PERFECTHASH_NASM_EXECUTABLE nasm)
        find_program(PERFECTHASH_OBJCOPY_EXECUTABLE llvm-objcopy)
        if(NOT PERFECTHASH_OBJCOPY_EXECUTABLE)
            find_program(PERFECTHASH_OBJCOPY_EXECUTABLE objcopy)
        endif()

        if(PERFECTHASH_NASM_EXECUTABLE AND PERFECTHASH_OBJCOPY_EXECUTABLE)
            set(PERFECTHASH_HAS_RAWDOG_JIT ON CACHE BOOL "RawDog JIT availability" FORCE)

            set(RAWDOG_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftR_x64.nasm"
            )
            set(RAWDOG_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR_x64.o"
            )
            set(RAWDOG_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR_x64.bin"
            )
            set(RAWDOG_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftR_x64.h"
            )
            set(RAWDOG_M1_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate1RX_x64.nasm"
            )
            set(RAWDOG_M1_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX_x64.o"
            )
            set(RAWDOG_M1_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX_x64.bin"
            )
            set(RAWDOG_M1_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate1RX_x64.h"
            )
            set(RAWDOG_MSR16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftR16_x64.nasm"
            )
            set(RAWDOG_MSR16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16_x64.o"
            )
            set(RAWDOG_MSR16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16_x64.bin"
            )
            set(RAWDOG_MSR16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftR16_x64.h"
            )
            set(RAWDOG_M1RX16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate1RX16_x64.nasm"
            )
            set(RAWDOG_M1RX16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX16_x64.o"
            )
            set(RAWDOG_M1RX16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX16_x64.bin"
            )
            set(RAWDOG_M1RX16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate1RX16_x64.h"
            )
            set(RAWDOG_M2_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RX_x64.nasm"
            )
            set(RAWDOG_M2_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX_x64.o"
            )
            set(RAWDOG_M2_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX_x64.bin"
            )
            set(RAWDOG_M2_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RX_x64.h"
            )
            set(RAWDOG_M2RX16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RX16_x64.nasm"
            )
            set(RAWDOG_M2RX16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16_x64.o"
            )
            set(RAWDOG_M2RX16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16_x64.bin"
            )
            set(RAWDOG_M2RX16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RX16_x64.h"
            )
            set(RAWDOG_M2_AVX2_V1_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RXAvx2_v1_x64.nasm"
            )
            set(RAWDOG_M2_AVX2_V1_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXAvx2_v1_x64.o"
            )
            set(RAWDOG_M2_AVX2_V1_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXAvx2_v1_x64.bin"
            )
            set(RAWDOG_M2_AVX2_V1_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RXAvx2_v1_x64.h"
            )
            set(RAWDOG_M2RX16_AVX2_V1_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx2_v1_x64.nasm"
            )
            set(RAWDOG_M2RX16_AVX2_V1_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx2_v1_x64.o"
            )
            set(RAWDOG_M2RX16_AVX2_V1_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx2_v1_x64.bin"
            )
            set(RAWDOG_M2RX16_AVX2_V1_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RX16Avx2_v1_x64.h"
            )
            set(RAWDOG_M2_AVX2_V2_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RXAvx2_v2_x64.nasm"
            )
            set(RAWDOG_M2_AVX2_V2_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXAvx2_v2_x64.o"
            )
            set(RAWDOG_M2_AVX2_V2_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXAvx2_v2_x64.bin"
            )
            set(RAWDOG_M2_AVX2_V2_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RXAvx2_v2_x64.h"
            )
            set(RAWDOG_M2_AVX2_V3_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RXAvx2_v3_x64.nasm"
            )
            set(RAWDOG_M2_AVX2_V3_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXAvx2_v3_x64.o"
            )
            set(RAWDOG_M2_AVX2_V3_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXAvx2_v3_x64.bin"
            )
            set(RAWDOG_M2_AVX2_V3_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RXAvx2_v3_x64.h"
            )
            set(RAWDOG_M2RX16_AVX2_V2_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx2_v2_x64.nasm"
            )
            set(RAWDOG_M2RX16_AVX2_V2_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx2_v2_x64.o"
            )
            set(RAWDOG_M2RX16_AVX2_V2_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx2_v2_x64.bin"
            )
            set(RAWDOG_M2RX16_AVX2_V2_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RX16Avx2_v2_x64.h"
            )
            set(RAWDOG_M2RX16_AVX2_V3_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx2_v3_x64.nasm"
            )
            set(RAWDOG_M2RX16_AVX2_V3_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx2_v3_x64.o"
            )
            set(RAWDOG_M2RX16_AVX2_V3_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx2_v3_x64.bin"
            )
            set(RAWDOG_M2RX16_AVX2_V3_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RX16Avx2_v3_x64.h"
            )
            set(RAWDOG_M2_AVX512_V1_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RXAvx512_v1_x64.nasm"
            )
            set(RAWDOG_M2_AVX512_V1_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXAvx512_v1_x64.o"
            )
            set(RAWDOG_M2_AVX512_V1_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXAvx512_v1_x64.bin"
            )
            set(RAWDOG_M2_AVX512_V1_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RXAvx512_v1_x64.h"
            )
            set(RAWDOG_M2_AVX512_V2_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RXAvx512_v2_x64.nasm"
            )
            set(RAWDOG_M2_AVX512_V2_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXAvx512_v2_x64.o"
            )
            set(RAWDOG_M2_AVX512_V2_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXAvx512_v2_x64.bin"
            )
            set(RAWDOG_M2_AVX512_V2_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RXAvx512_v2_x64.h"
            )
            set(RAWDOG_M2RX16_AVX512_V1_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx512_v1_x64.nasm"
            )
            set(RAWDOG_M2RX16_AVX512_V1_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx512_v1_x64.o"
            )
            set(RAWDOG_M2RX16_AVX512_V1_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx512_v1_x64.bin"
            )
            set(RAWDOG_M2RX16_AVX512_V1_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RX16Avx512_v1_x64.h"
            )
            set(RAWDOG_M2RX16_AVX512_V2_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx512_v2_x64.nasm"
            )
            set(RAWDOG_M2RX16_AVX512_V2_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx512_v2_x64.o"
            )
            set(RAWDOG_M2RX16_AVX512_V2_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Avx512_v2_x64.bin"
            )
            set(RAWDOG_M2RX16_AVX512_V2_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RX16Avx512_v2_x64.h"
            )
            set(RAWDOG_M3_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RX_x64.nasm"
            )
            set(RAWDOG_M3_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX_x64.o"
            )
            set(RAWDOG_M3_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX_x64.bin"
            )
            set(RAWDOG_M3_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RX_x64.h"
            )
            set(RAWDOG_M3RX16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RX16_x64.nasm"
            )
            set(RAWDOG_M3RX16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16_x64.o"
            )
            set(RAWDOG_M3RX16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16_x64.bin"
            )
            set(RAWDOG_M3RX16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RX16_x64.h"
            )
            set(RAWDOG_M3_AVX2_V1_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RXAvx2_v1_x64.nasm"
            )
            set(RAWDOG_M3_AVX2_V1_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXAvx2_v1_x64.o"
            )
            set(RAWDOG_M3_AVX2_V1_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXAvx2_v1_x64.bin"
            )
            set(RAWDOG_M3_AVX2_V1_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RXAvx2_v1_x64.h"
            )
            set(RAWDOG_M3RX16_AVX2_V1_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx2_v1_x64.nasm"
            )
            set(RAWDOG_M3RX16_AVX2_V1_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx2_v1_x64.o"
            )
            set(RAWDOG_M3RX16_AVX2_V1_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx2_v1_x64.bin"
            )
            set(RAWDOG_M3RX16_AVX2_V1_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RX16Avx2_v1_x64.h"
            )
            set(RAWDOG_M3_AVX2_V2_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RXAvx2_v2_x64.nasm"
            )
            set(RAWDOG_M3_AVX2_V2_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXAvx2_v2_x64.o"
            )
            set(RAWDOG_M3_AVX2_V2_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXAvx2_v2_x64.bin"
            )
            set(RAWDOG_M3_AVX2_V2_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RXAvx2_v2_x64.h"
            )
            set(RAWDOG_M3_AVX2_V3_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RXAvx2_v3_x64.nasm"
            )
            set(RAWDOG_M3_AVX2_V3_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXAvx2_v3_x64.o"
            )
            set(RAWDOG_M3_AVX2_V3_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXAvx2_v3_x64.bin"
            )
            set(RAWDOG_M3_AVX2_V3_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RXAvx2_v3_x64.h"
            )
            set(RAWDOG_M3RX16_AVX2_V2_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx2_v2_x64.nasm"
            )
            set(RAWDOG_M3RX16_AVX2_V2_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx2_v2_x64.o"
            )
            set(RAWDOG_M3RX16_AVX2_V2_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx2_v2_x64.bin"
            )
            set(RAWDOG_M3RX16_AVX2_V2_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RX16Avx2_v2_x64.h"
            )
            set(RAWDOG_M3RX16_AVX2_V3_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx2_v3_x64.nasm"
            )
            set(RAWDOG_M3RX16_AVX2_V3_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx2_v3_x64.o"
            )
            set(RAWDOG_M3RX16_AVX2_V3_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx2_v3_x64.bin"
            )
            set(RAWDOG_M3RX16_AVX2_V3_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RX16Avx2_v3_x64.h"
            )
            set(RAWDOG_M3_AVX512_V1_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RXAvx512_v1_x64.nasm"
            )
            set(RAWDOG_M3_AVX512_V1_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXAvx512_v1_x64.o"
            )
            set(RAWDOG_M3_AVX512_V1_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXAvx512_v1_x64.bin"
            )
            set(RAWDOG_M3_AVX512_V1_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RXAvx512_v1_x64.h"
            )
            set(RAWDOG_M3_AVX512_V2_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RXAvx512_v2_x64.nasm"
            )
            set(RAWDOG_M3_AVX512_V2_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXAvx512_v2_x64.o"
            )
            set(RAWDOG_M3_AVX512_V2_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXAvx512_v2_x64.bin"
            )
            set(RAWDOG_M3_AVX512_V2_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RXAvx512_v2_x64.h"
            )
            set(RAWDOG_M3RX16_AVX512_V1_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx512_v1_x64.nasm"
            )
            set(RAWDOG_M3RX16_AVX512_V1_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx512_v1_x64.o"
            )
            set(RAWDOG_M3RX16_AVX512_V1_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx512_v1_x64.bin"
            )
            set(RAWDOG_M3RX16_AVX512_V1_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RX16Avx512_v1_x64.h"
            )
            set(RAWDOG_M3RX16_AVX512_V2_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx512_v2_x64.nasm"
            )
            set(RAWDOG_M3RX16_AVX512_V2_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx512_v2_x64.o"
            )
            set(RAWDOG_M3RX16_AVX512_V2_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Avx512_v2_x64.bin"
            )
            set(RAWDOG_M3RX16_AVX512_V2_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RX16Avx512_v2_x64.h"
            )

            add_custom_command(
                OUTPUT ${RAWDOG_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_OBJ}"
                        "${RAWDOG_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_BIN}
                        "${RAWDOG_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_BIN}"
                        --output "${RAWDOG_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftR_x64
                DEPENDS
                    "${RAWDOG_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M1_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M1_OBJ}"
                        "${RAWDOG_M1_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M1_BIN}
                        "${RAWDOG_M1_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M1_BIN}"
                        --output "${RAWDOG_M1_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate1RX_x64
                DEPENDS
                    "${RAWDOG_M1_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate1RX..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSR16_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_MSR16_OBJ}"
                        "${RAWDOG_MSR16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_MSR16_BIN}
                        "${RAWDOG_MSR16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSR16_BIN}"
                        --output "${RAWDOG_MSR16_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftR16_x64
                DEPENDS
                    "${RAWDOG_MSR16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR16..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M1RX16_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M1RX16_OBJ}"
                        "${RAWDOG_M1RX16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M1RX16_BIN}
                        "${RAWDOG_M1RX16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M1RX16_BIN}"
                        --output "${RAWDOG_M1RX16_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate1RX16_x64
                DEPENDS
                    "${RAWDOG_M1RX16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate1RX16..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2_OBJ}"
                        "${RAWDOG_M2_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2_BIN}
                        "${RAWDOG_M2_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2_BIN}"
                        --output "${RAWDOG_M2_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RX_x64
                DEPENDS
                    "${RAWDOG_M2_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2RX16_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2RX16_OBJ}"
                        "${RAWDOG_M2RX16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2RX16_BIN}
                        "${RAWDOG_M2RX16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2RX16_BIN}"
                        --output "${RAWDOG_M2RX16_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RX16_x64
                DEPENDS
                    "${RAWDOG_M2RX16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX16..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2_AVX2_V1_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2_AVX2_V1_OBJ}"
                        "${RAWDOG_M2_AVX2_V1_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2_AVX2_V1_BIN}
                        "${RAWDOG_M2_AVX2_V1_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2_AVX2_V1_BIN}"
                        --output "${RAWDOG_M2_AVX2_V1_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RXAvx2_v1_x64
                DEPENDS
                    "${RAWDOG_M2_AVX2_V1_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX AVX2 v1..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2_AVX2_V2_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2_AVX2_V2_OBJ}"
                        "${RAWDOG_M2_AVX2_V2_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2_AVX2_V2_BIN}
                        "${RAWDOG_M2_AVX2_V2_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2_AVX2_V2_BIN}"
                        --output "${RAWDOG_M2_AVX2_V2_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RXAvx2_v2_x64
                DEPENDS
                    "${RAWDOG_M2_AVX2_V2_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX AVX2 v2..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2_AVX2_V3_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2_AVX2_V3_OBJ}"
                        "${RAWDOG_M2_AVX2_V3_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2_AVX2_V3_BIN}
                        "${RAWDOG_M2_AVX2_V3_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2_AVX2_V3_BIN}"
                        --output "${RAWDOG_M2_AVX2_V3_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RXAvx2_v3_x64
                DEPENDS
                    "${RAWDOG_M2_AVX2_V3_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX AVX2 v3..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2RX16_AVX2_V1_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2RX16_AVX2_V1_OBJ}"
                        "${RAWDOG_M2RX16_AVX2_V1_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2RX16_AVX2_V1_BIN}
                        "${RAWDOG_M2RX16_AVX2_V1_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2RX16_AVX2_V1_BIN}"
                        --output "${RAWDOG_M2RX16_AVX2_V1_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RX16Avx2_v1_x64
                DEPENDS
                    "${RAWDOG_M2RX16_AVX2_V1_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX16 AVX2 v1..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2RX16_AVX2_V2_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2RX16_AVX2_V2_OBJ}"
                        "${RAWDOG_M2RX16_AVX2_V2_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2RX16_AVX2_V2_BIN}
                        "${RAWDOG_M2RX16_AVX2_V2_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2RX16_AVX2_V2_BIN}"
                        --output "${RAWDOG_M2RX16_AVX2_V2_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RX16Avx2_v2_x64
                DEPENDS
                    "${RAWDOG_M2RX16_AVX2_V2_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX16 AVX2 v2..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2RX16_AVX2_V3_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2RX16_AVX2_V3_OBJ}"
                        "${RAWDOG_M2RX16_AVX2_V3_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2RX16_AVX2_V3_BIN}
                        "${RAWDOG_M2RX16_AVX2_V3_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2RX16_AVX2_V3_BIN}"
                        --output "${RAWDOG_M2RX16_AVX2_V3_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RX16Avx2_v3_x64
                DEPENDS
                    "${RAWDOG_M2RX16_AVX2_V3_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX16 AVX2 v3..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2_AVX512_V1_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2_AVX512_V1_OBJ}"
                        "${RAWDOG_M2_AVX512_V1_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2_AVX512_V1_BIN}
                        "${RAWDOG_M2_AVX512_V1_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2_AVX512_V1_BIN}"
                        --output "${RAWDOG_M2_AVX512_V1_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RXAvx512_v1_x64
                DEPENDS
                    "${RAWDOG_M2_AVX512_V1_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX AVX-512 v1..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2_AVX512_V2_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2_AVX512_V2_OBJ}"
                        "${RAWDOG_M2_AVX512_V2_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2_AVX512_V2_BIN}
                        "${RAWDOG_M2_AVX512_V2_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2_AVX512_V2_BIN}"
                        --output "${RAWDOG_M2_AVX512_V2_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RXAvx512_v2_x64
                DEPENDS
                    "${RAWDOG_M2_AVX512_V2_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX AVX-512 v2..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2RX16_AVX512_V1_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2RX16_AVX512_V1_OBJ}"
                        "${RAWDOG_M2RX16_AVX512_V1_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2RX16_AVX512_V1_BIN}
                        "${RAWDOG_M2RX16_AVX512_V1_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2RX16_AVX512_V1_BIN}"
                        --output "${RAWDOG_M2RX16_AVX512_V1_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RX16Avx512_v1_x64
                DEPENDS
                    "${RAWDOG_M2RX16_AVX512_V1_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX16 AVX-512 v1..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2RX16_AVX512_V2_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M2RX16_AVX512_V2_OBJ}"
                        "${RAWDOG_M2RX16_AVX512_V2_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M2RX16_AVX512_V2_BIN}
                        "${RAWDOG_M2RX16_AVX512_V2_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2RX16_AVX512_V2_BIN}"
                        --output "${RAWDOG_M2RX16_AVX512_V2_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RX16Avx512_v2_x64
                DEPENDS
                    "${RAWDOG_M2RX16_AVX512_V2_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX16 AVX-512 v2..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3_OBJ}"
                        "${RAWDOG_M3_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3_BIN}
                        "${RAWDOG_M3_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3_BIN}"
                        --output "${RAWDOG_M3_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RX_x64
                DEPENDS
                    "${RAWDOG_M3_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3RX16_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3RX16_OBJ}"
                        "${RAWDOG_M3RX16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3RX16_BIN}
                        "${RAWDOG_M3RX16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3RX16_BIN}"
                        --output "${RAWDOG_M3RX16_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RX16_x64
                DEPENDS
                    "${RAWDOG_M3RX16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX16..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3_AVX2_V1_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3_AVX2_V1_OBJ}"
                        "${RAWDOG_M3_AVX2_V1_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3_AVX2_V1_BIN}
                        "${RAWDOG_M3_AVX2_V1_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3_AVX2_V1_BIN}"
                        --output "${RAWDOG_M3_AVX2_V1_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RXAvx2_v1_x64
                DEPENDS
                    "${RAWDOG_M3_AVX2_V1_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX AVX2 v1..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3_AVX2_V2_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3_AVX2_V2_OBJ}"
                        "${RAWDOG_M3_AVX2_V2_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3_AVX2_V2_BIN}
                        "${RAWDOG_M3_AVX2_V2_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3_AVX2_V2_BIN}"
                        --output "${RAWDOG_M3_AVX2_V2_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RXAvx2_v2_x64
                DEPENDS
                    "${RAWDOG_M3_AVX2_V2_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX AVX2 v2..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3_AVX2_V3_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3_AVX2_V3_OBJ}"
                        "${RAWDOG_M3_AVX2_V3_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3_AVX2_V3_BIN}
                        "${RAWDOG_M3_AVX2_V3_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3_AVX2_V3_BIN}"
                        --output "${RAWDOG_M3_AVX2_V3_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RXAvx2_v3_x64
                DEPENDS
                    "${RAWDOG_M3_AVX2_V3_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX AVX2 v3..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3RX16_AVX2_V1_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3RX16_AVX2_V1_OBJ}"
                        "${RAWDOG_M3RX16_AVX2_V1_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3RX16_AVX2_V1_BIN}
                        "${RAWDOG_M3RX16_AVX2_V1_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3RX16_AVX2_V1_BIN}"
                        --output "${RAWDOG_M3RX16_AVX2_V1_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RX16Avx2_v1_x64
                DEPENDS
                    "${RAWDOG_M3RX16_AVX2_V1_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX16 AVX2 v1..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3RX16_AVX2_V2_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3RX16_AVX2_V2_OBJ}"
                        "${RAWDOG_M3RX16_AVX2_V2_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3RX16_AVX2_V2_BIN}
                        "${RAWDOG_M3RX16_AVX2_V2_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3RX16_AVX2_V2_BIN}"
                        --output "${RAWDOG_M3RX16_AVX2_V2_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RX16Avx2_v2_x64
                DEPENDS
                    "${RAWDOG_M3RX16_AVX2_V2_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX16 AVX2 v2..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3RX16_AVX2_V3_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3RX16_AVX2_V3_OBJ}"
                        "${RAWDOG_M3RX16_AVX2_V3_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3RX16_AVX2_V3_BIN}
                        "${RAWDOG_M3RX16_AVX2_V3_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3RX16_AVX2_V3_BIN}"
                        --output "${RAWDOG_M3RX16_AVX2_V3_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RX16Avx2_v3_x64
                DEPENDS
                    "${RAWDOG_M3RX16_AVX2_V3_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX16 AVX2 v3..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3_AVX512_V1_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3_AVX512_V1_OBJ}"
                        "${RAWDOG_M3_AVX512_V1_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3_AVX512_V1_BIN}
                        "${RAWDOG_M3_AVX512_V1_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3_AVX512_V1_BIN}"
                        --output "${RAWDOG_M3_AVX512_V1_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RXAvx512_v1_x64
                DEPENDS
                    "${RAWDOG_M3_AVX512_V1_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX AVX-512 v1..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3_AVX512_V2_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3_AVX512_V2_OBJ}"
                        "${RAWDOG_M3_AVX512_V2_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3_AVX512_V2_BIN}
                        "${RAWDOG_M3_AVX512_V2_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3_AVX512_V2_BIN}"
                        --output "${RAWDOG_M3_AVX512_V2_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RXAvx512_v2_x64
                DEPENDS
                    "${RAWDOG_M3_AVX512_V2_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX AVX-512 v2..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3RX16_AVX512_V1_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3RX16_AVX512_V1_OBJ}"
                        "${RAWDOG_M3RX16_AVX512_V1_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3RX16_AVX512_V1_BIN}
                        "${RAWDOG_M3RX16_AVX512_V1_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3RX16_AVX512_V1_BIN}"
                        --output "${RAWDOG_M3RX16_AVX512_V1_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RX16Avx512_v1_x64
                DEPENDS
                    "${RAWDOG_M3RX16_AVX512_V1_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX16 AVX-512 v1..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3RX16_AVX512_V2_HDR}
                COMMAND ${PERFECTHASH_NASM_EXECUTABLE}
                        -f elf64
                        -o "${RAWDOG_M3RX16_AVX512_V2_OBJ}"
                        "${RAWDOG_M3RX16_AVX512_V2_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text=${RAWDOG_M3RX16_AVX512_V2_BIN}
                        "${RAWDOG_M3RX16_AVX512_V2_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3RX16_AVX512_V2_BIN}"
                        --output "${RAWDOG_M3RX16_AVX512_V2_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RX16Avx512_v2_x64
                DEPENDS
                    "${RAWDOG_M3RX16_AVX512_V2_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX16 AVX-512 v2..."
                VERBATIM
            )

            list(APPEND Private_Header_Files
                 "${RAWDOG_HDR}"
                 "${RAWDOG_M1_HDR}"
                 "${RAWDOG_MSR16_HDR}"
                 "${RAWDOG_M1RX16_HDR}"
                 "${RAWDOG_M2_HDR}"
                 "${RAWDOG_M2RX16_HDR}"
                 "${RAWDOG_M2_AVX2_V1_HDR}"
                 "${RAWDOG_M2RX16_AVX2_V1_HDR}"
                 "${RAWDOG_M2_AVX2_V2_HDR}"
                 "${RAWDOG_M2RX16_AVX2_V2_HDR}"
                 "${RAWDOG_M2_AVX2_V3_HDR}"
                 "${RAWDOG_M2RX16_AVX2_V3_HDR}"
                 "${RAWDOG_M2_AVX512_V1_HDR}"
                 "${RAWDOG_M2_AVX512_V2_HDR}"
                 "${RAWDOG_M2RX16_AVX512_V1_HDR}"
                 "${RAWDOG_M2RX16_AVX512_V2_HDR}"
                 "${RAWDOG_M3_HDR}"
                 "${RAWDOG_M3RX16_HDR}"
                 "${RAWDOG_M3_AVX2_V1_HDR}"
                 "${RAWDOG_M3RX16_AVX2_V1_HDR}"
                 "${RAWDOG_M3_AVX2_V2_HDR}"
                 "${RAWDOG_M3RX16_AVX2_V2_HDR}"
                 "${RAWDOG_M3_AVX2_V3_HDR}"
                 "${RAWDOG_M3RX16_AVX2_V3_HDR}"
                 "${RAWDOG_M3_AVX512_V1_HDR}"
                 "${RAWDOG_M3_AVX512_V2_HDR}"
                 "${RAWDOG_M3RX16_AVX512_V1_HDR}"
                 "${RAWDOG_M3RX16_AVX512_V2_HDR}")
            list(APPEND Source_Files "ChmOnline01RawDog.c")
        else()
            message(WARNING "RawDog JIT disabled (missing nasm/objcopy).")
            set(PERFECTHASH_HAS_RAWDOG_JIT OFF CACHE BOOL "RawDog JIT availability" FORCE)
        endif()
    endif()
endif()

if(PERFECTHASH_IS_WINDOWS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    if(PERFECTHASH_ARCH_X86_64)
        find_program(PERFECTHASH_ML64_EXECUTABLE ml64)
        find_program(PERFECTHASH_OBJCOPY_EXECUTABLE llvm-objcopy)

        if(PERFECTHASH_ML64_EXECUTABLE AND PERFECTHASH_OBJCOPY_EXECUTABLE)
            set(PERFECTHASH_HAS_RAWDOG_JIT ON CACHE BOOL "RawDog JIT availability" FORCE)

            set(RAWDOG_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftR_x64.asm"
            )
            set(RAWDOG_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR_x64.obj"
            )
            set(RAWDOG_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR_x64.bin"
            )
            set(RAWDOG_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftR_x64.h"
            )
            set(RAWDOG_MSR16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftR16_x64.asm"
            )
            set(RAWDOG_MSR16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16_x64.obj"
            )
            set(RAWDOG_MSR16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16_x64.bin"
            )
            set(RAWDOG_MSR16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftR16_x64.h"
            )
            set(RAWDOG_MSR_INDEX32X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x8_x64.asm"
            )
            set(RAWDOG_MSR_INDEX32X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x8_x64.obj"
            )
            set(RAWDOG_MSR_INDEX32X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x8_x64.bin"
            )
            set(RAWDOG_MSR_INDEX32X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftRIndex32x8_x64.h"
            )
            set(RAWDOG_MSR16_INDEX32X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x8_x64.asm"
            )
            set(RAWDOG_MSR16_INDEX32X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x8_x64.obj"
            )
            set(RAWDOG_MSR16_INDEX32X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x8_x64.bin"
            )
            set(RAWDOG_MSR16_INDEX32X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftR16Index32x8_x64.h"
            )
            set(RAWDOG_MSR_INDEX32X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x16_x64.asm"
            )
            set(RAWDOG_MSR_INDEX32X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x16_x64.obj"
            )
            set(RAWDOG_MSR_INDEX32X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x16_x64.bin"
            )
            set(RAWDOG_MSR_INDEX32X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftRIndex32x16_x64.h"
            )
            set(RAWDOG_MSR16_INDEX32X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x16_x64.asm"
            )
            set(RAWDOG_MSR16_INDEX32X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x16_x64.obj"
            )
            set(RAWDOG_MSR16_INDEX32X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x16_x64.bin"
            )
            set(RAWDOG_MSR16_INDEX32X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftR16Index32x16_x64.h"
            )

            add_custom_command(
                OUTPUT ${RAWDOG_HDR}
                COMMAND ${PERFECTHASH_ML64_EXECUTABLE}
                        /nologo
                        /c
                        /Fo "${RAWDOG_OBJ}"
                        /I "${PERFECTHASH_ROOT_DIR}/include"
                        "${RAWDOG_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text$mn$00=${RAWDOG_BIN}
                        "${RAWDOG_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_BIN}"
                        --output "${RAWDOG_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftR_x64
                DEPENDS
                    "${RAWDOG_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR (MASM)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSR16_HDR}
                COMMAND ${PERFECTHASH_ML64_EXECUTABLE}
                        /nologo
                        /c
                        /Fo "${RAWDOG_MSR16_OBJ}"
                        /I "${PERFECTHASH_ROOT_DIR}/include"
                        "${RAWDOG_MSR16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text$mn$00=${RAWDOG_MSR16_BIN}
                        "${RAWDOG_MSR16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSR16_BIN}"
                        --output "${RAWDOG_MSR16_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftR16_x64
                DEPENDS
                    "${RAWDOG_MSR16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR16 (MASM)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSR_INDEX32X8_HDR}
                COMMAND ${PERFECTHASH_ML64_EXECUTABLE}
                        /nologo
                        /c
                        /Fo "${RAWDOG_MSR_INDEX32X8_OBJ}"
                        /I "${PERFECTHASH_ROOT_DIR}/include"
                        "${RAWDOG_MSR_INDEX32X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text$mn$00=${RAWDOG_MSR_INDEX32X8_BIN}
                        "${RAWDOG_MSR_INDEX32X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSR_INDEX32X8_BIN}"
                        --output "${RAWDOG_MSR_INDEX32X8_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftRIndex32x8_x64
                DEPENDS
                    "${RAWDOG_MSR_INDEX32X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR Index32x8 (MASM)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSR16_INDEX32X8_HDR}
                COMMAND ${PERFECTHASH_ML64_EXECUTABLE}
                        /nologo
                        /c
                        /Fo "${RAWDOG_MSR16_INDEX32X8_OBJ}"
                        /I "${PERFECTHASH_ROOT_DIR}/include"
                        "${RAWDOG_MSR16_INDEX32X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text$mn$00=${RAWDOG_MSR16_INDEX32X8_BIN}
                        "${RAWDOG_MSR16_INDEX32X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSR16_INDEX32X8_BIN}"
                        --output "${RAWDOG_MSR16_INDEX32X8_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftR16Index32x8_x64
                DEPENDS
                    "${RAWDOG_MSR16_INDEX32X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR16 Index32x8 (MASM)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSR_INDEX32X16_HDR}
                COMMAND ${PERFECTHASH_ML64_EXECUTABLE}
                        /nologo
                        /c
                        /Fo "${RAWDOG_MSR_INDEX32X16_OBJ}"
                        /I "${PERFECTHASH_ROOT_DIR}/include"
                        "${RAWDOG_MSR_INDEX32X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text$mn$00=${RAWDOG_MSR_INDEX32X16_BIN}
                        "${RAWDOG_MSR_INDEX32X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSR_INDEX32X16_BIN}"
                        --output "${RAWDOG_MSR_INDEX32X16_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftRIndex32x16_x64
                DEPENDS
                    "${RAWDOG_MSR_INDEX32X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR Index32x16 (MASM)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSR16_INDEX32X16_HDR}
                COMMAND ${PERFECTHASH_ML64_EXECUTABLE}
                        /nologo
                        /c
                        /Fo "${RAWDOG_MSR16_INDEX32X16_OBJ}"
                        /I "${PERFECTHASH_ROOT_DIR}/include"
                        "${RAWDOG_MSR16_INDEX32X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=.text$mn$00=${RAWDOG_MSR16_INDEX32X16_BIN}
                        "${RAWDOG_MSR16_INDEX32X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSR16_INDEX32X16_BIN}"
                        --output "${RAWDOG_MSR16_INDEX32X16_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftR16Index32x16_x64
                DEPENDS
                    "${RAWDOG_MSR16_INDEX32X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR16 Index32x16 (MASM)..."
                VERBATIM
            )

            list(APPEND Private_Header_Files
                 "${RAWDOG_HDR}"
                 "${RAWDOG_MSR16_HDR}"
                 "${RAWDOG_MSR_INDEX32X8_HDR}"
                 "${RAWDOG_MSR16_INDEX32X8_HDR}"
                 "${RAWDOG_MSR_INDEX32X16_HDR}"
                 "${RAWDOG_MSR16_INDEX32X16_HDR}")
            list(APPEND Source_Files "ChmOnline01RawDog.c")
        else()
            message(WARNING "RawDog JIT disabled (missing ml64/llvm-objcopy).")
            set(PERFECTHASH_HAS_RAWDOG_JIT OFF CACHE BOOL "RawDog JIT availability" FORCE)
        endif()
    endif()
endif()

if(PERFECTHASH_IS_MAC)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    if(PERFECTHASH_ARCH_ARM64)
        find_program(PERFECTHASH_CLANG_EXECUTABLE clang)
        find_program(PERFECTHASH_OBJCOPY_EXECUTABLE llvm-objcopy)

        if(PERFECTHASH_CLANG_EXECUTABLE AND PERFECTHASH_OBJCOPY_EXECUTABLE)
            set(PERFECTHASH_HAS_RAWDOG_JIT ON CACHE BOOL "RawDog JIT availability" FORCE)

            set(RAWDOG_ARM64_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftR_arm64.s"
            )
            set(RAWDOG_ARM64_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR_arm64.o"
            )
            set(RAWDOG_ARM64_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR_arm64.bin"
            )
            set(RAWDOG_ARM64_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftR_arm64.h"
            )
            set(RAWDOG_ARM64_16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftR16_arm64.s"
            )
            set(RAWDOG_ARM64_16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16_arm64.o"
            )
            set(RAWDOG_ARM64_16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16_arm64.bin"
            )
            set(RAWDOG_ARM64_16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftR16_arm64.h"
            )
            set(RAWDOG_MSRX_ARM64_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftRX_arm64.s"
            )
            set(RAWDOG_MSRX_ARM64_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRX_arm64.o"
            )
            set(RAWDOG_MSRX_ARM64_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRX_arm64.bin"
            )
            set(RAWDOG_MSRX_ARM64_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftRX_arm64.h"
            )
            set(RAWDOG_MSRX_ARM64_16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftRX16_arm64.s"
            )
            set(RAWDOG_MSRX_ARM64_16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRX16_arm64.o"
            )
            set(RAWDOG_MSRX_ARM64_16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRX16_arm64.bin"
            )
            set(RAWDOG_MSRX_ARM64_16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftRX16_arm64.h"
            )
            set(RAWDOG_M1_ARM64_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate1RX_arm64.s"
            )
            set(RAWDOG_M1_ARM64_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX_arm64.o"
            )
            set(RAWDOG_M1_ARM64_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX_arm64.bin"
            )
            set(RAWDOG_M1_ARM64_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate1RX_arm64.h"
            )
            set(RAWDOG_M1RX16_ARM64_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate1RX16_arm64.s"
            )
            set(RAWDOG_M1RX16_ARM64_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX16_arm64.o"
            )
            set(RAWDOG_M1RX16_ARM64_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX16_arm64.bin"
            )
            set(RAWDOG_M1RX16_ARM64_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate1RX16_arm64.h"
            )
            set(RAWDOG_M2_ARM64_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RX_arm64.s"
            )
            set(RAWDOG_M2_ARM64_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX_arm64.o"
            )
            set(RAWDOG_M2_ARM64_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX_arm64.bin"
            )
            set(RAWDOG_M2_ARM64_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RX_arm64.h"
            )
            set(RAWDOG_M2RX16_ARM64_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RX16_arm64.s"
            )
            set(RAWDOG_M2RX16_ARM64_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16_arm64.o"
            )
            set(RAWDOG_M2RX16_ARM64_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16_arm64.bin"
            )
            set(RAWDOG_M2RX16_ARM64_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RX16_arm64.h"
            )
            set(RAWDOG_M3_ARM64_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RX_arm64.s"
            )
            set(RAWDOG_M3_ARM64_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX_arm64.o"
            )
            set(RAWDOG_M3_ARM64_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX_arm64.bin"
            )
            set(RAWDOG_M3_ARM64_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RX_arm64.h"
            )
            set(RAWDOG_M3RX16_ARM64_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RX16_arm64.s"
            )
            set(RAWDOG_M3RX16_ARM64_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16_arm64.o"
            )
            set(RAWDOG_M3RX16_ARM64_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16_arm64.bin"
            )
            set(RAWDOG_M3RX16_ARM64_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RX16_arm64.h"
            )
            set(RAWDOG_MSR_ARM64_X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x8_arm64.s"
            )
            set(RAWDOG_MSR_ARM64_X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x8_arm64.o"
            )
            set(RAWDOG_MSR_ARM64_X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x8_arm64.bin"
            )
            set(RAWDOG_MSR_ARM64_X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftRIndex32x8_arm64.h"
            )
            set(RAWDOG_MSR16_ARM64_X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x8_arm64.s"
            )
            set(RAWDOG_MSR16_ARM64_X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x8_arm64.o"
            )
            set(RAWDOG_MSR16_ARM64_X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x8_arm64.bin"
            )
            set(RAWDOG_MSR16_ARM64_X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftR16Index32x8_arm64.h"
            )
            set(RAWDOG_MSR_ARM64_X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x16_arm64.s"
            )
            set(RAWDOG_MSR_ARM64_X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x16_arm64.o"
            )
            set(RAWDOG_MSR_ARM64_X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRIndex32x16_arm64.bin"
            )
            set(RAWDOG_MSR_ARM64_X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftRIndex32x16_arm64.h"
            )
            set(RAWDOG_MSR16_ARM64_X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x16_arm64.s"
            )
            set(RAWDOG_MSR16_ARM64_X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x16_arm64.o"
            )
            set(RAWDOG_MSR16_ARM64_X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftR16Index32x16_arm64.bin"
            )
            set(RAWDOG_MSR16_ARM64_X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftR16Index32x16_arm64.h"
            )
            set(RAWDOG_MSRX_ARM64_X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftRXIndex32x8_arm64.s"
            )
            set(RAWDOG_MSRX_ARM64_X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRXIndex32x8_arm64.o"
            )
            set(RAWDOG_MSRX_ARM64_X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRXIndex32x8_arm64.bin"
            )
            set(RAWDOG_MSRX_ARM64_X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftRXIndex32x8_arm64.h"
            )
            set(RAWDOG_MSRX16_ARM64_X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftRX16Index32x8_arm64.s"
            )
            set(RAWDOG_MSRX16_ARM64_X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRX16Index32x8_arm64.o"
            )
            set(RAWDOG_MSRX16_ARM64_X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRX16Index32x8_arm64.bin"
            )
            set(RAWDOG_MSRX16_ARM64_X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftRX16Index32x8_arm64.h"
            )
            set(RAWDOG_MSRX_ARM64_X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftRXIndex32x16_arm64.s"
            )
            set(RAWDOG_MSRX_ARM64_X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRXIndex32x16_arm64.o"
            )
            set(RAWDOG_MSRX_ARM64_X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRXIndex32x16_arm64.bin"
            )
            set(RAWDOG_MSRX_ARM64_X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftRXIndex32x16_arm64.h"
            )
            set(RAWDOG_MSRX16_ARM64_X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMultiplyShiftRX16Index32x16_arm64.s"
            )
            set(RAWDOG_MSRX16_ARM64_X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRX16Index32x16_arm64.o"
            )
            set(RAWDOG_MSRX16_ARM64_X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMultiplyShiftRX16Index32x16_arm64.bin"
            )
            set(RAWDOG_MSRX16_ARM64_X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMultiplyShiftRX16Index32x16_arm64.h"
            )
            set(RAWDOG_M1_ARM64_X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate1RXIndex32x8_arm64.s"
            )
            set(RAWDOG_M1_ARM64_X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RXIndex32x8_arm64.o"
            )
            set(RAWDOG_M1_ARM64_X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RXIndex32x8_arm64.bin"
            )
            set(RAWDOG_M1_ARM64_X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate1RXIndex32x8_arm64.h"
            )
            set(RAWDOG_M1RX16_ARM64_X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate1RX16Index32x8_arm64.s"
            )
            set(RAWDOG_M1RX16_ARM64_X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX16Index32x8_arm64.o"
            )
            set(RAWDOG_M1RX16_ARM64_X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX16Index32x8_arm64.bin"
            )
            set(RAWDOG_M1RX16_ARM64_X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate1RX16Index32x8_arm64.h"
            )
            set(RAWDOG_M1_ARM64_X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate1RXIndex32x16_arm64.s"
            )
            set(RAWDOG_M1_ARM64_X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RXIndex32x16_arm64.o"
            )
            set(RAWDOG_M1_ARM64_X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RXIndex32x16_arm64.bin"
            )
            set(RAWDOG_M1_ARM64_X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate1RXIndex32x16_arm64.h"
            )
            set(RAWDOG_M1RX16_ARM64_X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate1RX16Index32x16_arm64.s"
            )
            set(RAWDOG_M1RX16_ARM64_X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX16Index32x16_arm64.o"
            )
            set(RAWDOG_M1RX16_ARM64_X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate1RX16Index32x16_arm64.bin"
            )
            set(RAWDOG_M1RX16_ARM64_X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate1RX16Index32x16_arm64.h"
            )
            set(RAWDOG_M2_ARM64_X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RXIndex32x8_arm64.s"
            )
            set(RAWDOG_M2_ARM64_X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXIndex32x8_arm64.o"
            )
            set(RAWDOG_M2_ARM64_X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXIndex32x8_arm64.bin"
            )
            set(RAWDOG_M2_ARM64_X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RXIndex32x8_arm64.h"
            )
            set(RAWDOG_M2RX16_ARM64_X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RX16Index32x8_arm64.s"
            )
            set(RAWDOG_M2RX16_ARM64_X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Index32x8_arm64.o"
            )
            set(RAWDOG_M2RX16_ARM64_X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Index32x8_arm64.bin"
            )
            set(RAWDOG_M2RX16_ARM64_X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RX16Index32x8_arm64.h"
            )
            set(RAWDOG_M2_ARM64_X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RXIndex32x16_arm64.s"
            )
            set(RAWDOG_M2_ARM64_X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXIndex32x16_arm64.o"
            )
            set(RAWDOG_M2_ARM64_X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RXIndex32x16_arm64.bin"
            )
            set(RAWDOG_M2_ARM64_X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RXIndex32x16_arm64.h"
            )
            set(RAWDOG_M2RX16_ARM64_X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate2RX16Index32x16_arm64.s"
            )
            set(RAWDOG_M2RX16_ARM64_X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Index32x16_arm64.o"
            )
            set(RAWDOG_M2RX16_ARM64_X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate2RX16Index32x16_arm64.bin"
            )
            set(RAWDOG_M2RX16_ARM64_X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate2RX16Index32x16_arm64.h"
            )
            set(RAWDOG_M3_ARM64_X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RXIndex32x8_arm64.s"
            )
            set(RAWDOG_M3_ARM64_X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXIndex32x8_arm64.o"
            )
            set(RAWDOG_M3_ARM64_X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXIndex32x8_arm64.bin"
            )
            set(RAWDOG_M3_ARM64_X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RXIndex32x8_arm64.h"
            )
            set(RAWDOG_M3RX16_ARM64_X8_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RX16Index32x8_arm64.s"
            )
            set(RAWDOG_M3RX16_ARM64_X8_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Index32x8_arm64.o"
            )
            set(RAWDOG_M3RX16_ARM64_X8_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Index32x8_arm64.bin"
            )
            set(RAWDOG_M3RX16_ARM64_X8_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RX16Index32x8_arm64.h"
            )
            set(RAWDOG_M3_ARM64_X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RXIndex32x16_arm64.s"
            )
            set(RAWDOG_M3_ARM64_X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXIndex32x16_arm64.o"
            )
            set(RAWDOG_M3_ARM64_X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RXIndex32x16_arm64.bin"
            )
            set(RAWDOG_M3_ARM64_X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RXIndex32x16_arm64.h"
            )
            set(RAWDOG_M3RX16_ARM64_X16_ASM
                "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashJitIndexMulshrolate3RX16Index32x16_arm64.s"
            )
            set(RAWDOG_M3RX16_ARM64_X16_OBJ
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Index32x16_arm64.o"
            )
            set(RAWDOG_M3RX16_ARM64_X16_BIN
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitIndexMulshrolate3RX16Index32x16_arm64.bin"
            )
            set(RAWDOG_M3RX16_ARM64_X16_HDR
                "${CMAKE_CURRENT_BINARY_DIR}/PerfectHashJitRawDogMulshrolate3RX16Index32x16_arm64.h"
            )

            add_custom_command(
                OUTPUT ${RAWDOG_ARM64_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_ARM64_OBJ}"
                        "${RAWDOG_ARM64_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_ARM64_BIN}
                        "${RAWDOG_ARM64_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_ARM64_BIN}"
                        --output "${RAWDOG_ARM64_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftR_arm64
                DEPENDS
                    "${RAWDOG_ARM64_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_ARM64_16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_ARM64_16_OBJ}"
                        "${RAWDOG_ARM64_16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_ARM64_16_BIN}
                        "${RAWDOG_ARM64_16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_ARM64_16_BIN}"
                        --output "${RAWDOG_ARM64_16_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftR16_arm64
                DEPENDS
                    "${RAWDOG_ARM64_16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSRX_ARM64_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_MSRX_ARM64_OBJ}"
                        "${RAWDOG_MSRX_ARM64_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_MSRX_ARM64_BIN}
                        "${RAWDOG_MSRX_ARM64_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSRX_ARM64_BIN}"
                        --output "${RAWDOG_MSRX_ARM64_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftRX_arm64
                DEPENDS
                    "${RAWDOG_MSRX_ARM64_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftRX (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSRX_ARM64_16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_MSRX_ARM64_16_OBJ}"
                        "${RAWDOG_MSRX_ARM64_16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_MSRX_ARM64_16_BIN}
                        "${RAWDOG_MSRX_ARM64_16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSRX_ARM64_16_BIN}"
                        --output "${RAWDOG_MSRX_ARM64_16_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftRX16_arm64
                DEPENDS
                    "${RAWDOG_MSRX_ARM64_16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftRX16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M1_ARM64_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M1_ARM64_OBJ}"
                        "${RAWDOG_M1_ARM64_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M1_ARM64_BIN}
                        "${RAWDOG_M1_ARM64_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M1_ARM64_BIN}"
                        --output "${RAWDOG_M1_ARM64_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate1RX_arm64
                DEPENDS
                    "${RAWDOG_M1_ARM64_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate1RX (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSR_ARM64_X8_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_MSR_ARM64_X8_OBJ}"
                        "${RAWDOG_MSR_ARM64_X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_MSR_ARM64_X8_BIN}
                        "${RAWDOG_MSR_ARM64_X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSR_ARM64_X8_BIN}"
                        --output "${RAWDOG_MSR_ARM64_X8_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftRIndex32x8_arm64
                DEPENDS
                    "${RAWDOG_MSR_ARM64_X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR Index32x8 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSR16_ARM64_X8_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_MSR16_ARM64_X8_OBJ}"
                        "${RAWDOG_MSR16_ARM64_X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_MSR16_ARM64_X8_BIN}
                        "${RAWDOG_MSR16_ARM64_X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSR16_ARM64_X8_BIN}"
                        --output "${RAWDOG_MSR16_ARM64_X8_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftR16Index32x8_arm64
                DEPENDS
                    "${RAWDOG_MSR16_ARM64_X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR16 Index32x8 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSR_ARM64_X16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_MSR_ARM64_X16_OBJ}"
                        "${RAWDOG_MSR_ARM64_X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_MSR_ARM64_X16_BIN}
                        "${RAWDOG_MSR_ARM64_X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSR_ARM64_X16_BIN}"
                        --output "${RAWDOG_MSR_ARM64_X16_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftRIndex32x16_arm64
                DEPENDS
                    "${RAWDOG_MSR_ARM64_X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR Index32x16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSR16_ARM64_X16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_MSR16_ARM64_X16_OBJ}"
                        "${RAWDOG_MSR16_ARM64_X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_MSR16_ARM64_X16_BIN}
                        "${RAWDOG_MSR16_ARM64_X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSR16_ARM64_X16_BIN}"
                        --output "${RAWDOG_MSR16_ARM64_X16_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftR16Index32x16_arm64
                DEPENDS
                    "${RAWDOG_MSR16_ARM64_X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftR16 Index32x16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSRX_ARM64_X8_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_MSRX_ARM64_X8_OBJ}"
                        "${RAWDOG_MSRX_ARM64_X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_MSRX_ARM64_X8_BIN}
                        "${RAWDOG_MSRX_ARM64_X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSRX_ARM64_X8_BIN}"
                        --output "${RAWDOG_MSRX_ARM64_X8_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftRXIndex32x8_arm64
                DEPENDS
                    "${RAWDOG_MSRX_ARM64_X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftRX Index32x8 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSRX16_ARM64_X8_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_MSRX16_ARM64_X8_OBJ}"
                        "${RAWDOG_MSRX16_ARM64_X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_MSRX16_ARM64_X8_BIN}
                        "${RAWDOG_MSRX16_ARM64_X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSRX16_ARM64_X8_BIN}"
                        --output "${RAWDOG_MSRX16_ARM64_X8_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftRX16Index32x8_arm64
                DEPENDS
                    "${RAWDOG_MSRX16_ARM64_X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftRX16 Index32x8 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSRX_ARM64_X16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_MSRX_ARM64_X16_OBJ}"
                        "${RAWDOG_MSRX_ARM64_X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_MSRX_ARM64_X16_BIN}
                        "${RAWDOG_MSRX_ARM64_X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSRX_ARM64_X16_BIN}"
                        --output "${RAWDOG_MSRX_ARM64_X16_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftRXIndex32x16_arm64
                DEPENDS
                    "${RAWDOG_MSRX_ARM64_X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftRX Index32x16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_MSRX16_ARM64_X16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_MSRX16_ARM64_X16_OBJ}"
                        "${RAWDOG_MSRX16_ARM64_X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_MSRX16_ARM64_X16_BIN}
                        "${RAWDOG_MSRX16_ARM64_X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_MSRX16_ARM64_X16_BIN}"
                        --output "${RAWDOG_MSRX16_ARM64_X16_HDR}"
                        --symbol PerfectHashJitRawDogMultiplyShiftRX16Index32x16_arm64
                DEPENDS
                    "${RAWDOG_MSRX16_ARM64_X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for MultiplyShiftRX16 Index32x16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M1RX16_ARM64_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M1RX16_ARM64_OBJ}"
                        "${RAWDOG_M1RX16_ARM64_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M1RX16_ARM64_BIN}
                        "${RAWDOG_M1RX16_ARM64_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M1RX16_ARM64_BIN}"
                        --output "${RAWDOG_M1RX16_ARM64_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate1RX16_arm64
                DEPENDS
                    "${RAWDOG_M1RX16_ARM64_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate1RX16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M1_ARM64_X8_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M1_ARM64_X8_OBJ}"
                        "${RAWDOG_M1_ARM64_X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M1_ARM64_X8_BIN}
                        "${RAWDOG_M1_ARM64_X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M1_ARM64_X8_BIN}"
                        --output "${RAWDOG_M1_ARM64_X8_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate1RXIndex32x8_arm64
                DEPENDS
                    "${RAWDOG_M1_ARM64_X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate1RX Index32x8 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M1RX16_ARM64_X8_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M1RX16_ARM64_X8_OBJ}"
                        "${RAWDOG_M1RX16_ARM64_X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M1RX16_ARM64_X8_BIN}
                        "${RAWDOG_M1RX16_ARM64_X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M1RX16_ARM64_X8_BIN}"
                        --output "${RAWDOG_M1RX16_ARM64_X8_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate1RX16Index32x8_arm64
                DEPENDS
                    "${RAWDOG_M1RX16_ARM64_X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate1RX16 Index32x8 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M1_ARM64_X16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M1_ARM64_X16_OBJ}"
                        "${RAWDOG_M1_ARM64_X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M1_ARM64_X16_BIN}
                        "${RAWDOG_M1_ARM64_X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M1_ARM64_X16_BIN}"
                        --output "${RAWDOG_M1_ARM64_X16_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate1RXIndex32x16_arm64
                DEPENDS
                    "${RAWDOG_M1_ARM64_X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate1RX Index32x16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M1RX16_ARM64_X16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M1RX16_ARM64_X16_OBJ}"
                        "${RAWDOG_M1RX16_ARM64_X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M1RX16_ARM64_X16_BIN}
                        "${RAWDOG_M1RX16_ARM64_X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M1RX16_ARM64_X16_BIN}"
                        --output "${RAWDOG_M1RX16_ARM64_X16_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate1RX16Index32x16_arm64
                DEPENDS
                    "${RAWDOG_M1RX16_ARM64_X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate1RX16 Index32x16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2_ARM64_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M2_ARM64_OBJ}"
                        "${RAWDOG_M2_ARM64_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M2_ARM64_BIN}
                        "${RAWDOG_M2_ARM64_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2_ARM64_BIN}"
                        --output "${RAWDOG_M2_ARM64_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RX_arm64
                DEPENDS
                    "${RAWDOG_M2_ARM64_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2RX16_ARM64_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M2RX16_ARM64_OBJ}"
                        "${RAWDOG_M2RX16_ARM64_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M2RX16_ARM64_BIN}
                        "${RAWDOG_M2RX16_ARM64_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2RX16_ARM64_BIN}"
                        --output "${RAWDOG_M2RX16_ARM64_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RX16_arm64
                DEPENDS
                    "${RAWDOG_M2RX16_ARM64_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3_ARM64_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M3_ARM64_OBJ}"
                        "${RAWDOG_M3_ARM64_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M3_ARM64_BIN}
                        "${RAWDOG_M3_ARM64_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3_ARM64_BIN}"
                        --output "${RAWDOG_M3_ARM64_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RX_arm64
                DEPENDS
                    "${RAWDOG_M3_ARM64_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3RX16_ARM64_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M3RX16_ARM64_OBJ}"
                        "${RAWDOG_M3RX16_ARM64_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M3RX16_ARM64_BIN}
                        "${RAWDOG_M3RX16_ARM64_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3RX16_ARM64_BIN}"
                        --output "${RAWDOG_M3RX16_ARM64_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RX16_arm64
                DEPENDS
                    "${RAWDOG_M3RX16_ARM64_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2_ARM64_X8_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M2_ARM64_X8_OBJ}"
                        "${RAWDOG_M2_ARM64_X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M2_ARM64_X8_BIN}
                        "${RAWDOG_M2_ARM64_X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2_ARM64_X8_BIN}"
                        --output "${RAWDOG_M2_ARM64_X8_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RXIndex32x8_arm64
                DEPENDS
                    "${RAWDOG_M2_ARM64_X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX Index32x8 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2RX16_ARM64_X8_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M2RX16_ARM64_X8_OBJ}"
                        "${RAWDOG_M2RX16_ARM64_X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M2RX16_ARM64_X8_BIN}
                        "${RAWDOG_M2RX16_ARM64_X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2RX16_ARM64_X8_BIN}"
                        --output "${RAWDOG_M2RX16_ARM64_X8_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RX16Index32x8_arm64
                DEPENDS
                    "${RAWDOG_M2RX16_ARM64_X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX16 Index32x8 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2_ARM64_X16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M2_ARM64_X16_OBJ}"
                        "${RAWDOG_M2_ARM64_X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M2_ARM64_X16_BIN}
                        "${RAWDOG_M2_ARM64_X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2_ARM64_X16_BIN}"
                        --output "${RAWDOG_M2_ARM64_X16_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RXIndex32x16_arm64
                DEPENDS
                    "${RAWDOG_M2_ARM64_X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX Index32x16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M2RX16_ARM64_X16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M2RX16_ARM64_X16_OBJ}"
                        "${RAWDOG_M2RX16_ARM64_X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M2RX16_ARM64_X16_BIN}
                        "${RAWDOG_M2RX16_ARM64_X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M2RX16_ARM64_X16_BIN}"
                        --output "${RAWDOG_M2RX16_ARM64_X16_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate2RX16Index32x16_arm64
                DEPENDS
                    "${RAWDOG_M2RX16_ARM64_X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate2RX16 Index32x16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3_ARM64_X8_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M3_ARM64_X8_OBJ}"
                        "${RAWDOG_M3_ARM64_X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M3_ARM64_X8_BIN}
                        "${RAWDOG_M3_ARM64_X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3_ARM64_X8_BIN}"
                        --output "${RAWDOG_M3_ARM64_X8_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RXIndex32x8_arm64
                DEPENDS
                    "${RAWDOG_M3_ARM64_X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX Index32x8 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3RX16_ARM64_X8_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M3RX16_ARM64_X8_OBJ}"
                        "${RAWDOG_M3RX16_ARM64_X8_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M3RX16_ARM64_X8_BIN}
                        "${RAWDOG_M3RX16_ARM64_X8_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3RX16_ARM64_X8_BIN}"
                        --output "${RAWDOG_M3RX16_ARM64_X8_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RX16Index32x8_arm64
                DEPENDS
                    "${RAWDOG_M3RX16_ARM64_X8_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX16 Index32x8 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3_ARM64_X16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M3_ARM64_X16_OBJ}"
                        "${RAWDOG_M3_ARM64_X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M3_ARM64_X16_BIN}
                        "${RAWDOG_M3_ARM64_X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3_ARM64_X16_BIN}"
                        --output "${RAWDOG_M3_ARM64_X16_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RXIndex32x16_arm64
                DEPENDS
                    "${RAWDOG_M3_ARM64_X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX Index32x16 (arm64)..."
                VERBATIM
            )

            add_custom_command(
                OUTPUT ${RAWDOG_M3RX16_ARM64_X16_HDR}
                COMMAND ${PERFECTHASH_CLANG_EXECUTABLE}
                        -c
                        -o "${RAWDOG_M3RX16_ARM64_X16_OBJ}"
                        "${RAWDOG_M3RX16_ARM64_X16_ASM}"
                COMMAND ${PERFECTHASH_OBJCOPY_EXECUTABLE}
                        --dump-section=__TEXT,__text=${RAWDOG_M3RX16_ARM64_X16_BIN}
                        "${RAWDOG_M3RX16_ARM64_X16_OBJ}"
                COMMAND "${Python3_EXECUTABLE}"
                        "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                        --input "${RAWDOG_M3RX16_ARM64_X16_BIN}"
                        --output "${RAWDOG_M3RX16_ARM64_X16_HDR}"
                        --symbol PerfectHashJitRawDogMulshrolate3RX16Index32x16_arm64
                DEPENDS
                    "${RAWDOG_M3RX16_ARM64_X16_ASM}"
                    "${PERFECTHASH_ROOT_DIR}/scripts/generate_rawdog_jit_header.py"
                COMMENT "Generating RawDog JIT blob for Mulshrolate3RX16 Index32x16 (arm64)..."
                VERBATIM
            )

            list(APPEND Private_Header_Files
                 "${RAWDOG_ARM64_HDR}"
                 "${RAWDOG_ARM64_16_HDR}"
                 "${RAWDOG_MSRX_ARM64_HDR}"
                 "${RAWDOG_MSRX_ARM64_16_HDR}"
                 "${RAWDOG_MSR_ARM64_X8_HDR}"
                 "${RAWDOG_MSR_ARM64_X16_HDR}"
                 "${RAWDOG_MSR16_ARM64_X8_HDR}"
                 "${RAWDOG_MSR16_ARM64_X16_HDR}"
                 "${RAWDOG_MSRX_ARM64_X8_HDR}"
                 "${RAWDOG_MSRX_ARM64_X16_HDR}"
                 "${RAWDOG_MSRX16_ARM64_X8_HDR}"
                 "${RAWDOG_MSRX16_ARM64_X16_HDR}"
                 "${RAWDOG_M1_ARM64_HDR}"
                 "${RAWDOG_M1RX16_ARM64_HDR}"
                 "${RAWDOG_M1_ARM64_X8_HDR}"
                 "${RAWDOG_M1RX16_ARM64_X8_HDR}"
                 "${RAWDOG_M1_ARM64_X16_HDR}"
                 "${RAWDOG_M1RX16_ARM64_X16_HDR}"
                 "${RAWDOG_M2_ARM64_HDR}"
                 "${RAWDOG_M2RX16_ARM64_HDR}"
                 "${RAWDOG_M3_ARM64_HDR}"
                 "${RAWDOG_M3RX16_ARM64_HDR}"
                 "${RAWDOG_M2_ARM64_X8_HDR}"
                 "${RAWDOG_M2RX16_ARM64_X8_HDR}"
                 "${RAWDOG_M2_ARM64_X16_HDR}"
                 "${RAWDOG_M2RX16_ARM64_X16_HDR}"
                 "${RAWDOG_M3_ARM64_X8_HDR}"
                 "${RAWDOG_M3RX16_ARM64_X8_HDR}"
                 "${RAWDOG_M3_ARM64_X16_HDR}"
                 "${RAWDOG_M3RX16_ARM64_X16_HDR}")
            list(APPEND Source_Files "ChmOnline01RawDog.c")
        else()
            message(WARNING "RawDog JIT disabled (missing clang/llvm-objcopy).")
            set(PERFECTHASH_HAS_RAWDOG_JIT OFF CACHE BOOL "RawDog JIT availability" FORCE)
        endif()
    endif()
endif()

source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Private_Header_Files}
    ${Private_Header_Files_28Auto-Generated29}
    ${Public_Header_Files}
    ${Source_Files}
)

set(ONLINE_SOURCE_FILES ${Source_Files})
list(FILTER ONLINE_SOURCE_FILES EXCLUDE REGEX "Chm01FileWork.*\\.c$")
list(FILTER ONLINE_SOURCE_FILES EXCLUDE REGEX "PerfectHash(File|Path|Directory)\\.c$")
list(FILTER ONLINE_SOURCE_FILES EXCLUDE REGEX "PerfectHashContext(BulkCreate|TableCreate)\\.c$")
list(FILTER ONLINE_SOURCE_FILES EXCLUDE REGEX "VCProjectFileChunks\\.c$")
list(APPEND ONLINE_SOURCE_FILES "Chm01FileWorkStub.c")
list(APPEND ONLINE_SOURCE_FILES "PerfectHashContextCreateStub.c")

set(ONLINE_CORE_SOURCE_FILES ${ONLINE_SOURCE_FILES})
list(FILTER ONLINE_CORE_SOURCE_FILES EXCLUDE REGEX "ChmOnline01\\.c$")

set(ONLINE_ALL_FILES
    ${Private_Header_Files}
    ${Private_Header_Files_28Auto-Generated29}
    ${Public_Header_Files}
    ${ONLINE_SOURCE_FILES}
)

set(ONLINE_CORE_ALL_FILES
    ${Private_Header_Files}
    ${Private_Header_Files_28Auto-Generated29}
    ${Public_Header_Files}
    ${ONLINE_CORE_SOURCE_FILES}
)

set(LLVM_SOURCE_FILES
    "ChmOnline01.c"
)
if(WIN32)
    list(APPEND LLVM_SOURCE_FILES "PerfectHashLLVM.def")
endif()

if(PERFECTHASH_IS_LINUX AND PERFECTHASH_ARCH_X86_64)
    set_source_files_properties(
        "GraphAvx.c"
        PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512cd"
    )
endif()

if(WIN32)
    list(
        APPEND ALL_FILES
        ${Resource_Files}
    )

    list(
        APPEND ALL_FILES
        ${Assembly_Source_Files}
    )

    set(PERFECTHASH_ERRORS_HEADER
        "${PERFECTHASH_ROOT_DIR}/include/PerfectHashErrors.h"
    )
    set(PERFECTHASH_ERRORS_BIN
        "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashErrors_English.bin"
    )
    set(PERFECTHASH_ERRORS_DBG
        "${CMAKE_CURRENT_SOURCE_DIR}/PerfectHashErrors.dbg"
    )

    add_custom_command(
        OUTPUT ${PERFECTHASH_ERRORS_HEADER}
        COMMAND mc.exe -v -o -c -b -h "${PERFECTHASH_ROOT_DIR}/include" -x .
                -km PerfectHashEvents.man PerfectHashErrors.mc
        BYPRODUCTS ${PERFECTHASH_ERRORS_BIN} ${PERFECTHASH_ERRORS_DBG}
        DEPENDS PerfectHashErrors.mc PerfectHashEvents.man
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Compiling PerfectHashErrors.mc and PerfectHashEvents.man..."
    )

    add_custom_target(PerfectHashResources ALL DEPENDS ${PERFECTHASH_ERRORS_HEADER})
    set_property(
        TARGET PerfectHashResources
        PROPERTY SOURCES
        "${PERFECTHASH_ERRORS_HEADER}"
        "PerfectHashErrors.mc"
        "PerfectHashEvents.man"
    )
endif()

add_library(${target} SHARED ${ALL_FILES})

# Generated Linux message table header and RawDog blobs live in the build dir.
if(PERFECTHASH_IS_LINUX OR PERFECTHASH_HAS_RAWDOG_JIT)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# ###############################################################################
# Target
# ###############################################################################
set_property(TARGET ${target} PROPERTY PERFECTHASH_SKIP_RUNTIME_LIBS TRUE)
perfecthash_apply_common_settings(${target})

# ###############################################################################
# Compile definitions
# ###############################################################################
target_compile_definitions(
    ${target}
    PRIVATE
    "_PERFECT_HASH_INTERNAL_BUILD"
    "PH_USE_LLVMLIB"
)

if(PERFECTHASH_HAS_RAWDOG_JIT)
    target_compile_definitions(
        ${target}
        PRIVATE
        "PH_HAS_RAWDOG_JIT"
    )
    if(PERFECTHASH_IS_WINDOWS)
        target_compile_definitions(
            ${target}
            PRIVATE
            "PH_RAWDOG_MSR_ONLY"
        )
    endif()
    if(PERFECTHASH_ARCH_X86_64)
        target_compile_definitions(
            ${target}
            PRIVATE
            "PH_RAWDOG_X64"
        )
    elseif(PERFECTHASH_ARCH_ARM64)
        target_compile_definitions(
            ${target}
            PRIVATE
            "PH_RAWDOG_ARM64"
        )
    endif()
endif()

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


if(TARGET PerfectHashAsm)
    add_dependencies(${target} PerfectHashAsm)
    set(ADDITIONAL_LIBRARY_DEPENDENCIES PerfectHashAsm)
    if(WIN32)
        list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES rpcrt4)
    endif()
    target_link_libraries(${target} PUBLIC "${ADDITIONAL_LIBRARY_DEPENDENCIES}")
endif()

if(MSVC)
    target_link_libraries(
        ${target}
        PRIVATE
        $<$<CONFIG:Debug>:vcruntimed>
        $<$<NOT:$<CONFIG:Debug>>:vcruntime>
        $<$<CONFIG:Debug>:ucrtd>
        $<$<NOT:$<CONFIG:Debug>>:ucrt>
    )
endif()

if(WIN32)
    add_dependencies(${target} PerfectHashResources)
endif()

set(online_target PerfectHashOnline)
set(online_static_target PerfectHashOnlineStatic)
set(online_core_target PerfectHashOnlineCore)
set(online_core_static_target PerfectHashOnlineCoreStatic)

add_library(${online_target} SHARED ${ONLINE_ALL_FILES})
add_library(${online_static_target} STATIC ${ONLINE_ALL_FILES})
set_target_properties(
    ${online_static_target}
    PROPERTIES
    OUTPUT_NAME PerfectHashOnline
)

add_library(${online_core_target} SHARED ${ONLINE_CORE_ALL_FILES})
add_library(${online_core_static_target} STATIC ${ONLINE_CORE_ALL_FILES})
set_target_properties(
    ${online_core_static_target}
    PROPERTIES
    OUTPUT_NAME PerfectHashOnlineCore
)

set(llvm_target PerfectHashLLVM)

if(PERFECTHASH_HAS_LLVM)
    add_library(${llvm_target} SHARED ${LLVM_SOURCE_FILES})
endif()

if(WIN32)
    add_dependencies(${online_target} PerfectHashResources)
    add_dependencies(${online_static_target} PerfectHashResources)
    add_dependencies(${online_core_target} PerfectHashResources)
    add_dependencies(${online_core_static_target} PerfectHashResources)
    if(TARGET ${llvm_target})
        add_dependencies(${llvm_target} PerfectHashResources)
    endif()
endif()

# Generated Linux message table header and RawDog blobs live in the build dir.
if(PERFECTHASH_IS_LINUX OR PERFECTHASH_HAS_RAWDOG_JIT)
    target_include_directories(${online_target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_include_directories(${online_static_target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_include_directories(${online_core_target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_include_directories(${online_core_static_target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    if(TARGET ${llvm_target})
        target_include_directories(${llvm_target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
endif()

set_property(TARGET ${online_target} PROPERTY PERFECTHASH_SKIP_RUNTIME_LIBS TRUE)
set_property(TARGET ${online_static_target} PROPERTY PERFECTHASH_SKIP_RUNTIME_LIBS TRUE)
set_property(TARGET ${online_core_target} PROPERTY PERFECTHASH_SKIP_RUNTIME_LIBS TRUE)
set_property(TARGET ${online_core_static_target} PROPERTY PERFECTHASH_SKIP_RUNTIME_LIBS TRUE)
perfecthash_apply_common_settings(${online_target})
perfecthash_apply_common_settings(${online_static_target})
perfecthash_apply_common_settings(${online_core_target})
perfecthash_apply_common_settings(${online_core_static_target})

if(TARGET ${llvm_target})
    set_property(TARGET ${llvm_target} PROPERTY PERFECTHASH_SKIP_RUNTIME_LIBS TRUE)
    perfecthash_apply_common_settings(${llvm_target})
endif()

foreach(ph_online_target ${online_target} ${online_static_target})
    target_compile_definitions(
        ${ph_online_target}
        PRIVATE
        "_PERFECT_HASH_INTERNAL_BUILD"
        "PH_ONLINE_ONLY"
        "PH_USE_LLVMLIB"
    )
    if(PERFECTHASH_HAS_RAWDOG_JIT)
        target_compile_definitions(
            ${ph_online_target}
            PRIVATE
            "PH_HAS_RAWDOG_JIT"
        )
        if(PERFECTHASH_IS_WINDOWS)
            target_compile_definitions(
                ${ph_online_target}
                PRIVATE
                "PH_RAWDOG_MSR_ONLY"
            )
        endif()
        if(PERFECTHASH_ARCH_X86_64)
            target_compile_definitions(
                ${ph_online_target}
                PRIVATE
                "PH_RAWDOG_X64"
            )
        elseif(PERFECTHASH_ARCH_ARM64)
            target_compile_definitions(
                ${ph_online_target}
                PRIVATE
                "PH_RAWDOG_ARM64"
            )
        endif()
    endif()
endforeach()

foreach(ph_online_target ${online_core_target} ${online_core_static_target})
    target_compile_definitions(
        ${ph_online_target}
        PRIVATE
        "_PERFECT_HASH_INTERNAL_BUILD"
        "PH_ONLINE_ONLY"
        "PH_ONLINE_CORE_ONLY"
    )
endforeach()

if(TARGET ${llvm_target})
    target_compile_definitions(
        ${llvm_target}
        PRIVATE
        "_PERFECT_HASH_INTERNAL_BUILD"
    )
endif()

if(MSVC)
    foreach(ph_online_target ${online_static_target} ${online_core_static_target})
        set_target_properties(
            ${ph_online_target}
            PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/static"
        )
    endforeach()
    if(CMAKE_CONFIGURATION_TYPES)
        foreach(config ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER "${config}" config_upper)
            foreach(ph_online_target ${online_static_target} ${online_core_static_target})
                set_target_properties(
                    ${ph_online_target}
                    PROPERTIES
                    ARCHIVE_OUTPUT_DIRECTORY_${config_upper}
                    "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper}}/static"
                )
            endforeach()
        endforeach()
    endif()
endif()

foreach(ph_online_target ${online_target} ${online_static_target})
    if(PERFECTHASH_ENABLE_PENTER)
        target_link_libraries(${ph_online_target} PUBLIC FunctionHook)
    endif()

    if(TARGET PerfectHashAsm)
        add_dependencies(${ph_online_target} PerfectHashAsm)
        set(ADDITIONAL_LIBRARY_DEPENDENCIES PerfectHashAsm)
        if(WIN32)
            list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES rpcrt4)
        endif()
        target_link_libraries(
            ${ph_online_target}
            PUBLIC
            "${ADDITIONAL_LIBRARY_DEPENDENCIES}"
        )
    endif()

    if(MSVC)
        target_link_libraries(
            ${ph_online_target}
            PRIVATE
            $<$<CONFIG:Debug>:vcruntimed>
            $<$<NOT:$<CONFIG:Debug>>:vcruntime>
            $<$<CONFIG:Debug>:ucrtd>
            $<$<NOT:$<CONFIG:Debug>>:ucrt>
        )
    endif()
endforeach()

foreach(ph_online_target ${online_core_target} ${online_core_static_target})
    if(PERFECTHASH_ENABLE_PENTER)
        target_link_libraries(${ph_online_target} PUBLIC FunctionHook)
    endif()

    if(TARGET PerfectHashAsm)
        add_dependencies(${ph_online_target} PerfectHashAsm)
        set(ADDITIONAL_LIBRARY_DEPENDENCIES PerfectHashAsm)
        if(WIN32)
            list(APPEND ADDITIONAL_LIBRARY_DEPENDENCIES rpcrt4)
        endif()
        target_link_libraries(
            ${ph_online_target}
            PUBLIC
            "${ADDITIONAL_LIBRARY_DEPENDENCIES}"
        )
    endif()

    if(MSVC)
        target_link_libraries(
            ${ph_online_target}
            PRIVATE
            $<$<CONFIG:Debug>:vcruntimed>
            $<$<NOT:$<CONFIG:Debug>>:vcruntime>
            $<$<CONFIG:Debug>:ucrtd>
            $<$<NOT:$<CONFIG:Debug>>:ucrt>
        )
    endif()
endforeach()

if(TARGET ${llvm_target})
    target_link_libraries(${llvm_target} PRIVATE ${target})

    target_include_directories(${llvm_target} SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
    if(LLVM_DEFINITIONS)
        separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
        target_compile_options(${llvm_target} PRIVATE ${LLVM_DEFINITIONS_LIST})
    endif()
    if(MSVC)
        find_library(LLVM_C_LIB NAMES LLVM-C)
    endif()
    if(MSVC AND LLVM_C_LIB)
        target_link_libraries(${llvm_target} PRIVATE ${LLVM_C_LIB})
    endif()
    if(LLVM_LINK_LLVM_DYLIB AND TARGET LLVM)
        target_link_libraries(${llvm_target} PRIVATE LLVM)
    else()
        llvm_map_components_to_libnames(
            LLVM_LIBS
            core
            executionengine
            mcjit
            native
            target
            support
        )
        target_link_libraries(${llvm_target} PRIVATE ${LLVM_LIBS})
        if(MSVC)
            target_link_libraries(
                ${llvm_target}
                PRIVATE
                $<$<CONFIG:Debug>:msvcprtd>
                $<$<NOT:$<CONFIG:Debug>>:msvcprt>
                $<$<CONFIG:Debug>:msvcrtd>
                $<$<NOT:$<CONFIG:Debug>>:msvcrt>
                oldnames
            )
        endif()
    endif()

    if(MSVC)
        target_link_libraries(
            ${llvm_target}
            PRIVATE
            $<$<CONFIG:Debug>:vcruntimed>
            $<$<NOT:$<CONFIG:Debug>>:vcruntime>
            $<$<CONFIG:Debug>:ucrtd>
            $<$<NOT:$<CONFIG:Debug>>:ucrt>
        )
    endif()
endif()

# vim:set ts=8 sw=4 sts=4 tw=80 expandtab syntax=cmake                        :
