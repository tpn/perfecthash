/*++

Copyright (c) 2018-2026 Trent Nelson <trent@trent.me>

Module Name:

    PerfectHashFileWork.h

Abstract:

    This is the private header file for the file work component of the perfect
    hash library.  It defines the FILE_WORK_ID enumeration and various X-macro
    helper macro definitions.

--*/

#pragma once

#include "stdafx.h"

//
// Define an "X-Macro"-style macro for capturing the ordered definition of file
// work items.
//
// Invariants regarding order:
//
//      - NTFS streams must be preceeded by their containing file.
//
//          Streams will wait on their containing file's prepare event to be
//          set before they continue with creating the file.  In order to get
//          the handle for this event, they rely on being able to look at the
//          event that proceeds them, e.g. (in FileWorkCallbackChm01()):
//
//              DependentEvent = *(
//                  &Context->FirstPreparedEvent +
//                  (EventIndex - 1)
//              );
//
//      - All VC Projects must be contiguous.
//      - All context files must be contigous.
//
//          This ensures the corresponding enum's first and last members are
//          contiguous (e.g. VCPROJECT_FILE_ID), which is required in order
//          for the "is valid ID" inline functions to work properly, e.g.:
//
//              FORCEINLINE
//              BOOLEAN
//              IsValidVCProjectFileId(
//                  _In_ VCPROJECT_FILE_ID VCProjectFileId
//                  )
//              {
//                  return (
//                      VCProjectFileId >= VCProjectFileFirstId &&
//                      VCProjectFileId <= VCProjectFileLastId
//                  );
//              }
//
// Unlike the array size variants we explicitly verify in the Constants.c file
// (i.e. by trailing each array with VERIFY_ARRAY_SIZE(...)), these invariants
// aren't verified with compile-time (or run-time) assertions, so extra care
// needs to be paid to ensure they're not violated when adding new members.
//

#define NO_EOF_VALUE 0
#define NO_SUFFIX L""
#define NO_BASE_NAME L""
#define NO_EXTENSION &NullUnicodeString
#define NO_STREAM_NAME L""

#define SUFFIX(N)       (L""#N)
#define BASE_NAME(N)    (L""#N)
#define STREAM_NAME(N)  (L""#N)

#define EXPAND_AS_EXAMPLE(          \
    Verb, VUpper, Name, Upper,      \
    EofType, EofValue,              \
    Suffix, Extension, Stream, Base \
)

#define VERB_FILE_WORK_TABLE(Verb, VUpper, FIRST_ENTRY, ENTRY, LAST_ENTRY) \
                                                                           \
    FIRST_ENTRY(                                                           \
        Verb,                                                              \
        VUpper,                                                            \
        TableFile,                                                         \
        TABLE_FILE,                                                        \
        EofInitTypeAssignedSize,                                           \
        NO_EOF_VALUE,                                                      \
        NO_SUFFIX,                                                         \
        &TableFileExtension,                                               \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        TableInfoStream,                                                   \
        TABLE_INFO_STREAM,                                                 \
        EofInitTypeFixed,                                                  \
        sizeof(GRAPH_INFO_ON_DISK),                                        \
        NO_SUFFIX,                                                         \
        &TableFileExtension,                                               \
        STREAM_NAME(Info),                                                 \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CHeaderFile,                                                       \
        C_HEADER_FILE,                                                     \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        NO_SUFFIX,                                                         \
        &CHeaderFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CHeaderTypesFile,                                                  \
        C_HEADER_TYPES_FILE,                                               \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(Types),                                                     \
        &CHeaderFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceFile,                                                       \
        C_SOURCE_FILE,                                                     \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        NO_SUFFIX,                                                         \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CudaSourceFile,                                                    \
        CUDA_SOURCE_FILE,                                                  \
        EofInitTypeNumberOfTableElementsMultiplier,                        \
        256,                                                               \
        NO_SUFFIX,                                                         \
        &CudaSourceFileExtension,                                          \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CHeaderStdAfxFile,                                                 \
        C_HEADER_STDAFX_FILE,                                              \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(StdAfx),                                                    \
        &CHeaderFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceStdAfxFile,                                                 \
        C_SOURCE_STDAFX_FILE,                                              \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(StdAfx),                                                    \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceKeysFile,                                                   \
        C_SOURCE_KEYS_FILE,                                                \
        EofInitTypeNumberOfKeysMultiplier,                                 \
        32,                                                                \
        SUFFIX(Keys),                                                      \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceDownsizedKeysFile,                                          \
        C_SOURCE_DOWNSIZED_KEYS_FILE,                                      \
        EofInitTypeNumberOfKeysMultiplier,                                 \
        16,                                                                \
        SUFFIX(DownsizedKeys),                                             \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceTableDataFile,                                              \
        C_SOURCE_TABLE_DATA_FILE,                                          \
        EofInitTypeNumberOfTableElementsMultiplier,                        \
        128,                                                               \
        SUFFIX(TableData),                                                 \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceTableValuesFile,                                            \
        C_SOURCE_TABLE_VALUES_FILE,                                        \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(TableValues),                                               \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        PythonFile,                                                        \
        PYTHON_FILE,                                                       \
        EofInitTypeNumberOfTableElementsMultiplier,                        \
        256,                                                               \
        SUFFIX(Python),                                                    \
        &PythonFileExtension,                                              \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        PythonTestFile,                                                    \
        PYTHON_TEST_FILE,                                                  \
        EofInitTypeNumberOfKeysMultiplier,                                 \
        8,                                                                 \
        SUFFIX(python),                                                    \
        &PythonFileExtension,                                              \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CppHeaderOnlyFile,                                                 \
        CPP_HEADER_ONLY_FILE,                                              \
        EofInitTypeNumberOfTableElementsMultiplier,                        \
        256,                                                               \
        SUFFIX(CppHeaderOnly),                                             \
        &CppHeaderFileExtension,                                           \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CppHeaderOnlyTestFile,                                             \
        CPP_HEADER_ONLY_TEST_FILE,                                         \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(CppHeaderOnlyTest),                                         \
        &CppSourceFileExtension,                                           \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        RustCargoTomlFile,                                                 \
        RUST_CARGO_TOML_FILE,                                              \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        NO_SUFFIX,                                                         \
        &TomlFileExtension,                                                \
        NO_STREAM_NAME,                                                    \
        BASE_NAME(Cargo)                                                   \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        RustLibFile,                                                       \
        RUST_LIB_FILE,                                                     \
        EofInitTypeNumberOfTableElementsMultiplier,                        \
        256,                                                               \
        SUFFIX(RustLib),                                                   \
        &RustFileExtension,                                                \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        RustTestFile,                                                      \
        RUST_TEST_FILE,                                                    \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(RustTest),                                                  \
        &RustFileExtension,                                                \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        RustBenchFile,                                                     \
        RUST_BENCH_FILE,                                                   \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(RustBench),                                                 \
        &RustFileExtension,                                                \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CHeaderSupportFile,                                                \
        C_HEADER_SUPPORT_FILE,                                             \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(Support),                                                   \
        &CHeaderFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceSupportFile,                                                \
        C_SOURCE_SUPPORT_FILE,                                             \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(Support),                                                   \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceTestFile,                                                   \
        C_SOURCE_TEST_FILE,                                                \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(Test),                                                      \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceTestExeFile,                                                \
        C_SOURCE_TEST_EXE_FILE,                                            \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(TestExe),                                                   \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceBenchmarkFullFile,                                          \
        C_SOURCE_BENCHMARK_FULL_FILE,                                      \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(BenchmarkFull),                                             \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceBenchmarkFullExeFile,                                       \
        C_SOURCE_BENCHMARK_FULL_EXE_FILE,                                  \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(BenchmarkFullExe),                                          \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceBenchmarkIndexFile,                                         \
        C_SOURCE_BENCHMARK_INDEX_FILE,                                     \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(BenchmarkIndex),                                            \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CSourceBenchmarkIndexExeFile,                                      \
        C_SOURCE_BENCHMARK_INDEX_EXE_FILE,                                 \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(BenchmarkIndexExe),                                         \
        &CSourceFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        BatchBuildSolutionFile,                                            \
        BUILD_SOLUTION_BATCH_FILE,                                         \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(Build),                                                     \
        &BatchFileExtension,                                               \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        MakefileFile,                                                      \
        MAKEFILE_FILE,                                                     \
        EofInitTypeNumberOfPages,                                          \
        4,                                                                 \
        NO_SUFFIX,                                                         \
        NO_EXTENSION,                                                      \
        NO_STREAM_NAME,                                                    \
        BASE_NAME(Makefile)                                                \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        MakefileSoMkFile,                                                  \
        MAKEFILE_SO_MK_FILE,                                               \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(So),                                                        \
        &MakefileMkFileExtension,                                          \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        MakefileLibMkFile,                                                 \
        MAKEFILE_LIB_MK_FILE,                                              \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(Lib),                                                       \
        &MakefileMkFileExtension,                                          \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        MakefileTestMkFile,                                                \
        MAKEFILE_TEST_MK_FILE,                                             \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(Test),                                                      \
        &MakefileMkFileExtension,                                          \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        MakefileBenchmarkFullMkFile,                                       \
        MAKEFILE_BENCHMARK_FULL_MK_FILE,                                   \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(BenchmarkFull),                                             \
        &MakefileMkFileExtension,                                          \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        MakefileBenchmarkIndexMkFile,                                      \
        MAKEFILE_BENCHMARK_INDEX_MK_FILE,                                  \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        SUFFIX(BenchmarkIndex),                                            \
        &MakefileMkFileExtension,                                          \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        MakefileMainMkFile,                                                \
        MAKEFILE_MAIN_MK_FILE,                                             \
        EofInitTypeNumberOfPages,                                          \
        1,                                                                 \
        NO_SUFFIX,                                                         \
        &MakefileMkFileExtension,                                          \
        NO_STREAM_NAME,                                                    \
        BASE_NAME(main)                                                    \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        ModuleDefFile,                                                     \
        MODULE_DEF_FILE,                                                   \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        NO_SUFFIX,                                                         \
        &ModuleDefFileExtension,                                           \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CMakeListsTextFile,                                                \
        CMAKE_LISTS_TEXT_FILE,                                             \
        EofInitTypeNumberOfPages,                                          \
        4,                                                                 \
        NO_SUFFIX,                                                         \
        &TextFileExtension,                                                \
        NO_STREAM_NAME,                                                    \
        BASE_NAME(CMakeLists)                                              \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CppSourceUnityFile,                                                \
        CPP_SOURCE_UNITY_FILE,                                             \
        EofInitTypeNumberOfTableElementsMultiplier,                        \
        16384,                                                             \
        NO_SUFFIX,                                                         \
        &CppSourceFileExtension,                                           \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        VCProjectDllFile,                                                  \
        VCPROJECT_DLL_FILE,                                                \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(Dll),                                                       \
        &VCProjectFileExtension,                                           \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        VCProjectTestExeFile,                                              \
        VCPROJECT_TEST_EXE_FILE,                                           \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(TestExe),                                                   \
        &VCProjectFileExtension,                                           \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        VCProjectBenchmarkFullExeFile,                                     \
        VCPROJECT_BENCHMARK_FULL_EXE_FILE,                                 \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(BenchmarkFullExe),                                          \
        &VCProjectFileExtension,                                           \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        VCProjectBenchmarkIndexExeFile,                                    \
        VCPROJECT_BENCHMARK_INDEX_EXE_FILE,                                \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        SUFFIX(BenchmarkIndexExe),                                         \
        &VCProjectFileExtension,                                           \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        VSSolutionFile,                                                    \
        VSSOLUTION_FILE,                                                   \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        NO_SUFFIX,                                                         \
        &VSSolutionFileExtension,                                          \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CHeaderCompiledPerfectHashFile,                                    \
        C_HEADER_COMPILED_PERFECT_HASH_FILE,                               \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        NO_SUFFIX,                                                         \
        &CHeaderFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        BASE_NAME(CompiledPerfectHash)                                     \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CHeaderCompiledPerfectHashMacroGlueFile,                           \
        C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE,                    \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        NO_SUFFIX,                                                         \
        &CHeaderFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        BASE_NAME(CompiledPerfectHashMacroGlue)                            \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        CHeaderNoSal2File,                                                 \
        C_HEADER_NO_SAL2_FILE,                                             \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        NO_SUFFIX,                                                         \
        &CHeaderFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        BASE_NAME(no_sal2)                                                 \
    )                                                                      \
                                                                           \
    LAST_ENTRY(                                                            \
        Verb,                                                              \
        VUpper,                                                            \
        VCPropsCompiledPerfectHashFile,                                    \
        VCPROPS_COMPILED_PERFECT_HASH_FILE,                                \
        EofInitTypeDefault,                                                \
        NO_EOF_VALUE,                                                      \
        NO_SUFFIX,                                                         \
        &VCPropsFileExtension,                                             \
        NO_STREAM_NAME,                                                    \
        BASE_NAME(CompiledPerfectHash)                                     \
    )

#if 0

//
// These files are intended for debugging.
//

    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        GraphVertexPairsFile,                                              \
        GRAPH_VERTEX_PAIRS_FILE,                                           \
        EofInitTypeNumberOfKeysMultiplier,                                 \
        8,                                                                 \
        SUFFIX(GraphVertexPairs),                                          \
        &BinFileExtension,                                                 \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )                                                                      \
                                                                           \
    ENTRY(                                                                 \
        Verb,                                                              \
        VUpper,                                                            \
        GraphVertices3File,                                                \
        GRAPH_VERTICES3_FILE,                                              \
        EofInitTypeNumberOfTableElementsMultiplier,                        \
        4,                                                                 \
        SUFFIX(GraphVertices3),                                            \
        &BinFileExtension,                                                 \
        NO_STREAM_NAME,                                                    \
        NO_BASE_NAME                                                       \
    )

#endif

#define PREPARE_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_FILE_WORK_TABLE(Prepare, PREPARE, FIRST_ENTRY, ENTRY, LAST_ENTRY)

#define SAVE_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_FILE_WORK_TABLE(Save, SAVE, FIRST_ENTRY, ENTRY, LAST_ENTRY)

#define CLOSE_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_FILE_WORK_TABLE(Close, CLOSE, FIRST_ENTRY, ENTRY, LAST_ENTRY)

#define PREPARE_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_FILE_WORK_TABLE(Prepare, PREPARE, ENTRY, ENTRY, ENTRY)

#define SAVE_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_FILE_WORK_TABLE(Save, SAVE, ENTRY, ENTRY, ENTRY)

#define CLOSE_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_FILE_WORK_TABLE(Close, CLOSE, ENTRY, ENTRY, ENTRY)

#define FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_FILE_WORK_TABLE(Nothing, NOTHING, FIRST_ENTRY, ENTRY, LAST_ENTRY)

#define FILE_WORK_TABLE_ENTRY(ENTRY) FILE_WORK_TABLE(ENTRY, ENTRY, ENTRY)

//
// Some files are only generated once per context lifetime (and are written
// to the base output directory (owned by the context), instead of the output
// directory (owned by the table)).  We refer to this as CONTEXT_FILE_WORK.
// As above, define another X-macro for this type of file work.  We don't need
// the full set of parameters for each macro invocation (i.e. eof, suffix etc),
// just the verbs and names.
//

#define VERB_CONTEXT_FILE_WORK_TABLE(Verb,             \
                                     VUpper,           \
                                     FIRST_ENTRY,      \
                                     ENTRY,            \
                                     LAST_ENTRY)       \
                                                       \
    FIRST_ENTRY(                                       \
        Verb,                                          \
        VUpper,                                        \
        CHeaderCompiledPerfectHashFile,                \
        C_HEADER_COMPILED_PERFECT_HASH_FILE            \
    )                                                  \
                                                       \
    ENTRY(                                             \
        Verb,                                          \
        VUpper,                                        \
        CHeaderCompiledPerfectHashMacroGlueFile,       \
        C_HEADER_COMPILED_PERFECT_HASH_MACRO_GLUE_FILE \
    )                                                  \
                                                       \
    ENTRY(                                             \
        Verb,                                          \
        VUpper,                                        \
        CHeaderNoSal2File,                             \
        C_HEADER_NO_SAL2_FILE                          \
    )                                                  \
                                                       \
    LAST_ENTRY(                                        \
        Verb,                                          \
        VUpper,                                        \
        VCPropsCompiledPerfectHashFile,                \
        VCPROPS_COMPILED_PERFECT_HASH_FILE             \
    )


#define PREPARE_CONTEXT_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_CONTEXT_FILE_WORK_TABLE(Prepare,                               \
                                 PREPARE,                               \
                                 FIRST_ENTRY,                           \
                                 ENTRY,                                 \
                                 LAST_ENTRY)


#define SAVE_CONTEXT_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_CONTEXT_FILE_WORK_TABLE(Save,                               \
                                 SAVE,                               \
                                 FIRST_ENTRY,                        \
                                 ENTRY,                              \
                                 LAST_ENTRY)


#define CLOSE_CONTEXT_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_CONTEXT_FILE_WORK_TABLE(Close,                               \
                                 CLOSE,                               \
                                 FIRST_ENTRY,                         \
                                 ENTRY,                               \
                                 LAST_ENTRY)


#define PREPARE_CONTEXT_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_CONTEXT_FILE_WORK_TABLE(Prepare,            \
                                 PREPARE,            \
                                 ENTRY,              \
                                 ENTRY,              \
                                 ENTRY)


#define SAVE_CONTEXT_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_CONTEXT_FILE_WORK_TABLE(Save,            \
                                 SAVE,            \
                                 ENTRY,           \
                                 ENTRY,           \
                                 ENTRY)


#define CLOSE_CONTEXT_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_CONTEXT_FILE_WORK_TABLE(Close,            \
                                 CLOSE,            \
                                 ENTRY,            \
                                 ENTRY,            \
                                 ENTRY)


#define CONTEXT_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_CONTEXT_FILE_WORK_TABLE(Nothing,                       \
                                 NOTHING,                       \
                                 FIRST_ENTRY,                   \
                                 ENTRY,                         \
                                 LAST_ENTRY)


#define CONTEXT_FILE_WORK_TABLE_ENTRY(ENTRY) \
    CONTEXT_FILE_WORK_TABLE(ENTRY,           \
                            ENTRY,           \
                            ENTRY)

//
// Define an X-macro for VC project files.
//

#define VERB_VCPROJECT_FILE_WORK_TABLE(Verb,        \
                                       VUpper,      \
                                       FIRST_ENTRY, \
                                       ENTRY,       \
                                       LAST_ENTRY)  \
                                                    \
    FIRST_ENTRY(                                    \
        Verb,                                       \
        VUpper,                                     \
        VCProjectDllFile,                           \
        VC_PROJECT_DLL_FILE                         \
    )                                               \
                                                    \
    ENTRY(                                          \
        Verb,                                       \
        VUpper,                                     \
        VCProjectTestExeFile,                       \
        VC_PROJECT_TEST_EXE_FILE                    \
    )                                               \
                                                    \
    ENTRY(                                          \
        Verb,                                       \
        VUpper,                                     \
        VCProjectBenchmarkFullExeFile,              \
        VC_PROJECT_BENCHMARK_FULL_EXE_FILE          \
    )                                               \
                                                    \
    LAST_ENTRY(                                     \
        Verb,                                       \
        VUpper,                                     \
        VCProjectBenchmarkIndexExeFile,             \
        VC_PROJECT_TEST_EXE_FILE                    \
    )


#define PREPARE_VCPROJECT_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_VCPROJECT_FILE_WORK_TABLE(Prepare,                               \
                                   PREPARE,                               \
                                   FIRST_ENTRY,                           \
                                   ENTRY,                                 \
                                   LAST_ENTRY)


#define SAVE_VCPROJECT_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_VCPROJECT_FILE_WORK_TABLE(Save,                               \
                                   SAVE,                               \
                                   FIRST_ENTRY,                        \
                                   ENTRY,                              \
                                   LAST_ENTRY)


#define CLOSE_VCPROJECT_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_VCPROJECT_FILE_WORK_TABLE(Close,                               \
                                   CLOSE,                               \
                                   FIRST_ENTRY,                         \
                                   ENTRY,                               \
                                   LAST_ENTRY)


#define PREPARE_VCPROJECT_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_VCPROJECT_FILE_WORK_TABLE(Prepare,            \
                                   PREPARE,            \
                                   ENTRY,              \
                                   ENTRY,              \
                                   ENTRY)


#define SAVE_VCPROJECT_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_VCPROJECT_FILE_WORK_TABLE(Save,            \
                                   SAVE,            \
                                   ENTRY,           \
                                   ENTRY,           \
                                   ENTRY)


#define CLOSE_VCPROJECT_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_VCPROJECT_FILE_WORK_TABLE(Close,            \
                                   CLOSE,            \
                                   ENTRY,            \
                                   ENTRY,            \
                                   ENTRY)


#define VCPROJECT_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_VCPROJECT_FILE_WORK_TABLE(Nothing,                       \
                                   NOTHING,                       \
                                   FIRST_ENTRY,                   \
                                   ENTRY,                         \
                                   LAST_ENTRY)


#define VCPROJECT_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VCPROJECT_FILE_WORK_TABLE(ENTRY,           \
                              ENTRY,           \
                              ENTRY)


//
// Define an X-macro for sub-Makefile files (e.g. those ending in .mk).
//

#define VERB_SUBMAKEFILE_FILE_WORK_TABLE(Verb,        \
                                         VUpper,      \
                                         FIRST_ENTRY, \
                                         ENTRY,       \
                                         LAST_ENTRY)  \
                                                      \
    FIRST_ENTRY(                                      \
        Verb,                                         \
        VUpper,                                       \
        MakefileSoMkFile,                             \
        MAKEFILE_SO_MK_FILE                           \
    )                                                 \
                                                      \
    ENTRY(                                            \
        Verb,                                         \
        VUpper,                                       \
        MakefileLibMkFile,                            \
        MAKEFILE_LIB_MK_FILE                          \
    )                                                 \
                                                      \
    ENTRY(                                            \
        Verb,                                         \
        VUpper,                                       \
        MakefileTestMkFile,                           \
        MAKEFILE_TEST_MK_FILE                         \
    )                                                 \
                                                      \
    ENTRY(                                            \
        Verb,                                         \
        VUpper,                                       \
        MakefileBenchmarkFullMkFile,                  \
        MAKEFILE_BENCHMARK_FULL_MK_FILE               \
    )                                                 \
                                                      \
    LAST_ENTRY(                                       \
        Verb,                                         \
        VUpper,                                       \
        MakefileBenchmarkIndexMkFile,                 \
        MAKEFILE_BENCHMARK_INDEX_MK_FILE              \
    )


#define PREPARE_SUBMAKEFILE_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_SUBMAKEFILE_FILE_WORK_TABLE(Prepare,                               \
                                     PREPARE,                               \
                                     FIRST_ENTRY,                           \
                                     ENTRY,                                 \
                                     LAST_ENTRY)


#define SAVE_SUBMAKEFILE_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_SUBMAKEFILE_FILE_WORK_TABLE(Save,                               \
                                     SAVE,                               \
                                     FIRST_ENTRY,                        \
                                     ENTRY,                              \
                                     LAST_ENTRY)


#define CLOSE_SUBMAKEFILE_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_SUBMAKEFILE_FILE_WORK_TABLE(Close,                               \
                                     CLOSE,                               \
                                     FIRST_ENTRY,                         \
                                     ENTRY,                               \
                                     LAST_ENTRY)


#define PREPARE_SUBMAKEFILE_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_SUBMAKEFILE_FILE_WORK_TABLE(Prepare,            \
                                     PREPARE,            \
                                     ENTRY,              \
                                     ENTRY,              \
                                     ENTRY)


#define SAVE_SUBMAKEFILE_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_SUBMAKEFILE_FILE_WORK_TABLE(Save,            \
                                     SAVE,            \
                                     ENTRY,           \
                                     ENTRY,           \
                                     ENTRY)


#define CLOSE_SUBMAKEFILE_FILE_WORK_TABLE_ENTRY(ENTRY) \
    VERB_SUBMAKEFILE_FILE_WORK_TABLE(Close,            \
                                     CLOSE,            \
                                     ENTRY,            \
                                     ENTRY,            \
                                     ENTRY)


#define SUBMAKEFILE_FILE_WORK_TABLE(FIRST_ENTRY, ENTRY, LAST_ENTRY) \
    VERB_SUBMAKEFILE_FILE_WORK_TABLE(Nothing,                       \
                                     NOTHING,                       \
                                     FIRST_ENTRY,                   \
                                     ENTRY,                         \
                                     LAST_ENTRY)


#define SUBMAKEFILE_FILE_WORK_TABLE_ENTRY(ENTRY) \
    SUBMAKEFILE_FILE_WORK_TABLE(ENTRY,           \
                                ENTRY,           \
                                ENTRY)


//
// Define an enum for individual file IDs.
//

typedef enum _FILE_ID {

    //
    // Null ID.
    //

    FileNullId = 0,

#define EXPAND_AS_FILE_ENUM(        \
    Verb, VUpper, Name, Upper,      \
    EofType, EofValue,              \
    Suffix, Extension, Stream, Base \
)                                   \
    File##Name##Id,

#define EXPAND_AS_FIRST_FILE_ENUM(  \
    Verb, VUpper, Name, Upper,      \
    EofType, EofValue,              \
    Suffix, Extension, Stream, Base \
)                                   \
    File##Name##Id,                 \
    FileFirstId = File##Name##Id,

#define EXPAND_AS_LAST_FILE_ENUM(   \
    Verb, VUpper, Name, Upper,      \
    EofType, EofValue,              \
    Suffix, Extension, Stream, Base \
)                                   \
    File##Name##Id,                 \
    FileLastId = File##Name##Id,

    FILE_WORK_TABLE(EXPAND_AS_FIRST_FILE_ENUM,
                    EXPAND_AS_FILE_ENUM,
                    EXPAND_AS_LAST_FILE_ENUM)

    //
    // Invalid ID, this must come last.
    //

    FileInvalidId,

} FILE_ID;
typedef FILE_ID *PFILE_ID;

FORCEINLINE
BOOLEAN
IsValidFileId(
    _In_ FILE_ID FileId
    )
{
    return (
        FileId >= FileFirstId &&
        FileId <= FileLastId
    );
}

#define NUMBER_OF_FILES ((FileLastId - FileFirstId) + 1)

//
// Define an enum for individual context file IDs.
//

typedef enum _CONTEXT_FILE_ID {

    //
    // Null ID.
    //

    ContextFileNullId = 0,

#define EXPAND_AS_CONTEXT_FILE_ENUM(Verb, VUpper, Name, Upper) \
    ContextFile##Name##Id = File##Name##Id,

#define EXPAND_AS_FIRST_CONTEXT_FILE_ENUM(Verb, VUpper, Name, Upper) \
    ContextFile##Name##Id = File##Name##Id,                          \
    ContextFileFirstId = ContextFile##Name##Id,

#define EXPAND_AS_LAST_CONTEXT_FILE_ENUM(Verb, VUpper, Name, Upper) \
    ContextFile##Name##Id = File##Name##Id,                         \
    ContextFileLastId = ContextFile##Name##Id,

    CONTEXT_FILE_WORK_TABLE(EXPAND_AS_FIRST_CONTEXT_FILE_ENUM,
                            EXPAND_AS_CONTEXT_FILE_ENUM,
                            EXPAND_AS_LAST_CONTEXT_FILE_ENUM)

    //
    // Invalid ID, this must come last.
    //

    ContextFileInvalidId,

} CONTEXT_FILE_ID;
typedef CONTEXT_FILE_ID *PCONTEXT_FILE_ID;

#define NUMBER_OF_CONTEXT_FILES ((ContextFileLastId - ContextFileFirstId) + 1)

FORCEINLINE
BOOLEAN
IsValidContextFileId(
    _In_ CONTEXT_FILE_ID ContextFileId
    )
{
    return (
        ContextFileId >= ContextFileFirstId &&
        ContextFileId <= ContextFileLastId
    );
}

//
// Define an enum for individual VC Project file IDs.
//

typedef enum _VCPROJECT_FILE_ID {

    //
    // Null ID.
    //

    VCProjectFileNullId = 0,

#define EXPAND_AS_VCPROJECT_FILE_ENUMS(Verb, VUpper, Name, Upper) \
    VCProjectFile##Name##Id = File##Name##Id,

#define EXPAND_AS_FIRST_VCPROJECT_FILE_ENUM(Verb, VUpper, Name, Upper) \
    VCProjectFile##Name##Id = File##Name##Id,                          \
    VCProjectFileFirstId = VCProjectFile##Name##Id,

#define EXPAND_AS_LAST_VCPROJECT_FILE_ENUM(Verb, VUpper, Name, Upper) \
    VCProjectFile##Name##Id = File##Name##Id,                         \
    VCProjectFileLastId = VCProjectFile##Name##Id,

    VCPROJECT_FILE_WORK_TABLE(EXPAND_AS_FIRST_VCPROJECT_FILE_ENUM,
                              EXPAND_AS_VCPROJECT_FILE_ENUMS,
                              EXPAND_AS_LAST_VCPROJECT_FILE_ENUM)

    //
    // Invalid ID, this must come last.
    //

    VCProjectFileInvalidId,

} VCPROJECT_FILE_ID;
typedef VCPROJECT_FILE_ID *PVCPROJECT_FILE_ID;

#define NUMBER_OF_VCPROJECT_FILES (                  \
    (VCProjectFileLastId - VCProjectFileFirstId) + 1 \
)

FORCEINLINE
BOOLEAN
IsValidVCProjectFileId(
    _In_ VCPROJECT_FILE_ID VCProjectFileId
    )
{
    return (
        VCProjectFileId >= VCProjectFileFirstId &&
        VCProjectFileId <= VCProjectFileLastId
    );
}

FORCEINLINE
BOOLEAN
FileRequiresUuid(
    _In_ FILE_ID Id
    )
{
    return (
        IsValidVCProjectFileId((VCPROJECT_FILE_ID)Id) ||
        Id == FileVSSolutionFileId
    );
}

//
// Define an enum for individual sub-Makefile file IDs.
//

typedef enum _SUBMAKEFILE_FILE_ID {

    //
    // Null ID.
    //

    SubMakefileFileNullId = 0,

#define EXPAND_AS_SUBMAKEFILE_FILE_ENUMS(Verb, VUpper, Name, Upper) \
    SubMakefileFile##Name##Id = File##Name##Id,

#define EXPAND_AS_FIRST_SUBMAKEFILE_FILE_ENUM(Verb, VUpper, Name, Upper) \
    SubMakefileFile##Name##Id = File##Name##Id,                          \
    SubMakefileFileFirstId = SubMakefileFile##Name##Id,

#define EXPAND_AS_LAST_SUBMAKEFILE_FILE_ENUM(Verb, VUpper, Name, Upper) \
    SubMakefileFile##Name##Id = File##Name##Id,                         \
    SubMakefileFileLastId = SubMakefileFile##Name##Id,

    SUBMAKEFILE_FILE_WORK_TABLE(EXPAND_AS_FIRST_SUBMAKEFILE_FILE_ENUM,
                                EXPAND_AS_SUBMAKEFILE_FILE_ENUMS,
                                EXPAND_AS_LAST_SUBMAKEFILE_FILE_ENUM)

    //
    // Invalid ID, this must come last.
    //

    SubMakefileFileInvalidId,

} SUBMAKEFILE_FILE_ID;
typedef SUBMAKEFILE_FILE_ID *PSUBMAKEFILE_FILE_ID;

#define NUMBER_OF_SUBMAKEFILE_FILES (                    \
    (SubMakefileFileLastId - SubMakefileFileFirstId) + 1 \
)

FORCEINLINE
BOOLEAN
IsValidSubMakefileFileId(
    _In_ SUBMAKEFILE_FILE_ID SubMakefileFileId
    )
{
    return (
        SubMakefileFileId >= SubMakefileFileFirstId &&
        SubMakefileFileId <= SubMakefileFileLastId
    );
}

//
// Define an enumeration to capture the type of file work operations we want
// to be able to dispatch to the file work threadpool callback.
//

typedef enum _FILE_WORK_ID {

    //
    // Null ID.
    //

    FileWorkNullId = 0,

#define EXPAND_AS_FILE_WORK_ENUM(   \
    Verb, VUpper, Name, Upper,      \
    EofType, EofValue,              \
    Suffix, Extension, Stream, Base \
)                                   \
    FileWork##Verb##Name##Id,

#define EXPAND_AS_FIRST_FILE_WORK_ENUM(                 \
    Verb, VUpper, Name, Upper,                          \
    EofType, EofValue,                                  \
    Suffix, Extension, Stream, Base                     \
)                                                       \
    FileWork##Verb##Name##Id,                           \
    FileWork##Verb##FirstId = FileWork##Verb##Name##Id,

#define EXPAND_AS_LAST_FILE_WORK_ENUM(                 \
    Verb, VUpper, Name, Upper,                         \
    EofType, EofValue,                                 \
    Suffix, Extension, Stream, Base                    \
)                                                      \
    FileWork##Verb##Name##Id,                          \
    FileWork##Verb##LastId = FileWork##Verb##Name##Id,

    PREPARE_FILE_WORK_TABLE(EXPAND_AS_FIRST_FILE_WORK_ENUM,
                            EXPAND_AS_FILE_WORK_ENUM,
                            EXPAND_AS_LAST_FILE_WORK_ENUM)

    SAVE_FILE_WORK_TABLE(EXPAND_AS_FIRST_FILE_WORK_ENUM,
                         EXPAND_AS_FILE_WORK_ENUM,
                         EXPAND_AS_LAST_FILE_WORK_ENUM)

    CLOSE_FILE_WORK_TABLE(EXPAND_AS_FIRST_FILE_WORK_ENUM,
                          EXPAND_AS_FILE_WORK_ENUM,
                          EXPAND_AS_LAST_FILE_WORK_ENUM)

    //
    // Invalid ID, this must come last.
    //

    FileWorkInvalidId,

} FILE_WORK_ID;
typedef FILE_WORK_ID *PFILE_WORK_ID;

#define NUMBER_OF_PREPARE_FILE_EVENTS (                  \
    (FileWorkPrepareLastId - FileWorkPrepareFirstId) + 1 \
)
C_ASSERT(NUMBER_OF_PREPARE_FILE_EVENTS == NUMBER_OF_FILES);

#define NUMBER_OF_SAVE_FILE_EVENTS (               \
    (FileWorkSaveLastId - FileWorkSaveFirstId) + 1 \
)
C_ASSERT(NUMBER_OF_SAVE_FILE_EVENTS == NUMBER_OF_FILES);

//
// N.B. Close file work type does not use events.
//

#define TOTAL_NUMBER_OF_FILE_EVENTS ( \
    NUMBER_OF_PREPARE_FILE_EVENTS +   \
    NUMBER_OF_SAVE_FILE_EVENTS        \
)

FORCEINLINE
BOOLEAN
IsValidFileWorkId(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    return (
        FileWorkId > FileWorkNullId &&
        FileWorkId < FileWorkInvalidId
    );
}

FORCEINLINE
BOOLEAN
IsPrepareFileWorkId(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    return (
        FileWorkId >= FileWorkPrepareFirstId &&
        FileWorkId <= FileWorkPrepareLastId
    );
}

FORCEINLINE
BOOLEAN
IsSaveFileWorkId(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    return (
        FileWorkId >= FileWorkSaveFirstId &&
        FileWorkId <= FileWorkSaveLastId
    );
}

FORCEINLINE
BOOLEAN
IsCloseFileWorkId(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    return (
        FileWorkId >= FileWorkCloseFirstId &&
        FileWorkId <= FileWorkCloseLastId
    );
}

FORCEINLINE
FILE_ID
FileWorkIdToFileId(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    LONG Id;

    Id = (LONG)FileWorkId;

    if (IsSaveFileWorkId(FileWorkId)) {
        Id -= ((LONG)(FileWorkSaveFirstId - 1));
    } else if (IsCloseFileWorkId(FileWorkId)) {
        Id -= ((LONG)(FileWorkCloseFirstId - 1));
    }

    ASSERT(IsValidFileId((FILE_ID)Id));

    return (FILE_ID)Id;
}

FORCEINLINE
ULONG
FileWorkIdToFileIndex(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    LONG Index;

    Index = FileWorkId - 1;

    if (IsSaveFileWorkId(FileWorkId)) {
        Index -= (FileWorkSaveFirstId - 1);
    } else if (IsCloseFileWorkId(FileWorkId)) {
        Index -= (FileWorkCloseFirstId - 1);
    }

    ASSERT(Index >= 0);

    return Index;
}

FORCEINLINE
ULONG
ContextFileIdToContextFileIndex(
    _In_ CONTEXT_FILE_ID ContextFileId
    )
{
    LONG Index;

    Index = (LONG)ContextFileId - (LONG)ContextFileFirstId;

    ASSERT(Index >= 0 && Index <= NUMBER_OF_CONTEXT_FILES-1);

    return Index;
}

FORCEINLINE
ULONG
FileWorkIdToEventIndex(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    return FileWorkId - 1;
}

FORCEINLINE
FILE_WORK_ID
FileWorkIdToDependentId(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    LONG Id = 0;

    if (IsPrepareFileWorkId(FileWorkId)) {
        Id = FileWorkId + FileWorkSaveFirstId;
    } else if (IsSaveFileWorkId(FileWorkId)) {
        Id = FileWorkId - (FileWorkSaveFirstId - 1);
    } else if (IsCloseFileWorkId(FileWorkId)) {
        Id = FileWorkId - (FileWorkCloseFirstId - 1);
    } else {
        PH_RAISE(PH_E_UNREACHABLE_CODE);
    }

    ASSERT(Id >= FileWorkPrepareFirstId && Id <= FileWorkCloseLastId);

    return (FILE_WORK_ID)Id;
}

FORCEINLINE
ULONG
FileWorkIdToDependentEventIndex(
    _In_ FILE_WORK_ID FileWorkId
    )
{
    return FileWorkIdToEventIndex(FileWorkIdToDependentId(FileWorkId));
}

typedef enum _EOF_INIT_TYPE {
    EofInitTypeNull = 0,
    EofInitTypeDefault,
    EofInitTypeAssignedSize,
    EofInitTypeFixed,
    EofInitTypeNumberOfPages,
    EofInitTypeNumberOfKeysMultiplier,
    EofInitTypeNumberOfTableElementsMultiplier,
    EofInitTypeInvalid
} EOF_INIT_TYPE;

FORCEINLINE
BOOLEAN
IsValidEofInitType(
    _In_ EOF_INIT_TYPE EofInitType
    )
{
    return EofInitType < EofInitTypeInvalid;
}

typedef struct _EOF_INIT {
    EOF_INIT_TYPE Type;
    union {
        ULONG FixedValue;
        ULONG Multiplier;
        ULONG NumberOfPages;
    };
} EOF_INIT;
typedef EOF_INIT *PEOF_INIT;
typedef const EOF_INIT *PCEOF_INIT;

typedef union _FILE_WORK_ITEM_FLAGS {
    struct _Struct_size_bytes_(sizeof(ULONG)) {

        //
        // When set, indicates the file can only be prepared once, e.g. for
        // streams and context files.  Checked against the underlying file
        // pointer (*FilePointer); ensures prepare routines aren't ever called
        // when an existing file instance exists.
        //

        ULONG PrepareOnce:1;

        //
        // When set, indicates this is a context file.
        //

        ULONG IsContextFile:1;

        //
        // Unused bits.
        //

        ULONG Unused:30;
    };

    LONG AsLong;
    ULONG AsULong;
} FILE_WORK_ITEM_FLAGS;
C_ASSERT(sizeof(FILE_WORK_ITEM_FLAGS) == sizeof(ULONG));
typedef FILE_WORK_ITEM_FLAGS *PFILE_WORK_ITEM_FLAGS;

//
// Define a file work item structure that will be pushed to the context's
// file work list head.
//

typedef struct _FILE_WORK_ITEM {

    //
    // Entry used to add this structure onto a guarded list.
    //

    LIST_ENTRY ListEntry;

    FILE_WORK_ITEM_FLAGS Flags;

    union {
        FILE_ID FileId;
        CONTEXT_FILE_ID ContextFileId;
        VCPROJECT_FILE_ID VCProjectFileId;
        SUBMAKEFILE_FILE_ID SubMakefileFileId;
    };

    FILE_WORK_ID FileWorkId;

    volatile LONG NumberOfErrors;
    volatile LONG LastError;

    volatile HRESULT LastResult;

    STRING Uuid;

    PLARGE_INTEGER EndOfFile;

    struct _PERFECT_HASH_FILE **FilePointer;

    struct _PERFECT_HASH_CONTEXT *Context;

    PTP_CALLBACK_INSTANCE Instance;

} FILE_WORK_ITEM;
typedef FILE_WORK_ITEM *PFILE_WORK_ITEM;

FORCEINLINE
BOOLEAN
IsContextFileWorkItem(
    _In_ PFILE_WORK_ITEM Item
    )
{
    return Item->Flags.IsContextFile == TRUE;
}

//
// Define specific file work functions.
//

typedef
_Must_inspect_result_
_Success_(return >= 0)
HRESULT
(NTAPI FILE_WORK_CALLBACK_IMPL)(
    _In_ struct _PERFECT_HASH_CONTEXT *Context,
    _In_ PFILE_WORK_ITEM Item
    );
typedef FILE_WORK_CALLBACK_IMPL *PFILE_WORK_CALLBACK_IMPL;

//
// Define a helper macro for writing the '#include "<tablename>_StdAfx.h"' line
// to the output file.  Assumes Output and Item variables are in scope.  For an
// example, see Chm01FileWorkCSourceTestFile.c.
//

#define OUTPUT_INCLUDE_STDAFX_H()                                      \
    OUTPUT_RAW("#include \"");                                         \
    OUTPUT_STRING(&(GetActivePath((*Item->FilePointer))->TableNameA)); \
    OUTPUT_RAW("_StdAfx.h\"\n\n");

//
// As above, but for _Support.h.
//

#define OUTPUT_INCLUDE_SUPPORT_H()                                     \
    OUTPUT_RAW("#include \"");                                         \
    OUTPUT_STRING(&(GetActivePath((*Item->FilePointer))->TableNameA)); \
    OUTPUT_RAW("_Support.h\"\n\n");

#define OUTPUT_INCLUDE_STDAFX_AND_SUPPORT_H() \
    OUTPUT_INCLUDE_STDAFX_H();                \
    OUTPUT_INCLUDE_SUPPORT_H()                \

//
// Helper macros for common constructs written from multiple places.
//

#define OUTPUT_OPEN_EXTERN_C_SCOPE()  \
    OUTPUT_RAW("#ifdef __cplusplus\n" \
               "extern \"C\" {\n"     \
               "#endif\n\n")

#define OUTPUT_CLOSE_EXTERN_C_SCOPE() \
    OUTPUT_RAW("#ifdef __cplusplus\n" \
               "} // extern C\n"      \
               "#endif\n\n")

#define OUTPUT_PRAGMA_WARNING_DISABLE_UNREFERENCED_INLINE()              \
    OUTPUT_RAW("#ifdef _WIN32\n\n"                                       \
               "//\n// Disable \"unreferenced inline function has been " \
                      "removed\" warning.\n//\n\n"                       \
               "#pragma warning(push)\n"                                 \
               "#pragma warning(disable: 4514)\n\n"                      \
               "#endif\n\n")

#define OUTPUT_PRAGMA_WARNING_POP()   \
    OUTPUT_RAW("\n#ifdef _WIN32\n"    \
               "#pragma warning(pop)" \
               "\n#endif\n")

#define OUTPUT_PRAGMA_WARNING_DISABLE_FUNC_SELECTED_FOR_INLINE_EXP_WARNING() \
    OUTPUT_RAW("#ifdef _WIN32\n\n"                                           \
               "//\n"                                                        \
               "// Disable \"function ... selected for "                     \
                   "inline expansion\" warning.\n"                           \
               "//\n\n"                                                      \
               "#pragma warning(push)\n"                                     \
               "#pragma warning(disable: 4711)\n\n"                          \
               "#endif\n\n")

#define OUTPUT_DOT_MK() \
    *Output++ = '.';    \
    *Output++ = 'm';    \
    *Output++ = 'k'

#define OUTPUT_DOT_C() \
    *Output++ = '.';   \
    *Output++ = 'c'

#define OUTPUT_SPACE_SLASH_NEWLINE_TAB() \
    *Output++ = ' ';                     \
    *Output++ = '\\';                    \
    *Output++ = '\n';                    \
    *Output++ = '\t'

#define MAYBE_OUTPUT_INCLUDE_TABLE_VALUES_DOT_C() \
    if (!IsIndexOnly(Context->Table)) {           \
        OUTPUT_STRING(Name);                      \
        OUTPUT_RAW("_TableValues.c");             \
        OUTPUT_SPACE_SLASH_NEWLINE_TAB();         \
    }

#define MAYBE_OUTPUT_INCLUDE_KEYS_DOT_C()                                     \
    if (Context->Table->TableCreateFlags.IncludeKeysInCompiledDll != FALSE) { \
        OUTPUT_STRING(Name);                                                  \
        OUTPUT_RAW("_Keys.c");                                                \
        OUTPUT_SPACE_SLASH_NEWLINE_TAB();                                     \
    }

#define OUTPUT_MAKEFILE_SPLASH_COMMENT(N)             \
    OUTPUT_RAW("# Compiled Perfect Hash Table " # N); \
    OUTPUT_RAW(" Makefile.\n"                         \
               "# Auto-generated.\n\n");

#define OUTPUT_MAKEFILE_TARGET(Pfx)    \
    OUTPUT_RAW("TARGET := ");          \
    OUTPUT_STRING(&Pfx##TargetPrefix); \
    OUTPUT_STRING(Name);               \
    OUTPUT_RAW("\n\n")

#define OUTPUT_MAKEFILE_TGT_VARS()                \
    OUTPUT_RAW("TGT_LDFLAGS := -L${TARGET_DIR}\n" \
               "TGT_LDLIBS := -l");               \
    OUTPUT_STRING(Name);                          \
    OUTPUT_RAW("\nTGT_PREREQS := lib");           \
    OUTPUT_STRING(Name);                          \
    OUTPUT_RAW(".so\n\n")

#define OUTPUT_MAKEFILE_SOURCES(Suffix)            \
    OUTPUT_RAW("SOURCES := \\\n\t");               \
    OUTPUT_STRING(Name);                           \
    OUTPUT_RAW("_Support.c \\\n\t");               \
    OUTPUT_STRING(Name);                           \
    OUTPUT_RAW("_Keys.c \\\n\t");                  \
    if (KeysWereDownsized(Context->Table->Keys)) { \
        OUTPUT_STRING(Name);                       \
        OUTPUT_RAW("_DownsizedKeys.c \\\n\t");     \
    }                                              \
    OUTPUT_STRING(Name);                           \
    OUTPUT_RAW("_" # Suffix);                      \
    OUTPUT_RAW(".c \\\n\t");                       \
    OUTPUT_STRING(Name);                           \
    OUTPUT_RAW("_" # Suffix);                      \
    OUTPUT_RAW("Exe.c\n")

// vim:set ts=8 sw=4 sts=4 tw=80 expandtab                                     :
