include(FetchContent)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
if(MSVC AND POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "" FORCE)
endif()

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
FetchContent_MakeAvailable(googletest)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(CARGO_EXECUTABLE cargo)
if(NOT CARGO_EXECUTABLE)
    set(CARGO_EXECUTABLE "")
endif()

add_executable(perfecthash_unit_tests
    PerfectHashArgvTests.cpp
    PerfectHashOnlineJitTests.cpp
    PerfectHashOnlineTests.cpp
)

target_link_libraries(perfecthash_unit_tests
    PRIVATE
        PerfectHash
        GTest::gtest_main
)

perfecthash_apply_common_definitions(perfecthash_unit_tests)
target_compile_definitions(perfecthash_unit_tests
    PRIVATE PERFECTHASH_TEST_ROOT_DIR="${PERFECTHASH_ROOT_DIR}"
    PRIVATE PERFECTHASH_TEST_BUILD_LIB_DIR="$<TARGET_FILE_DIR:PerfectHash>"
)

set_target_properties(perfecthash_unit_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

if(PERFECTHASH_HAS_RAWDOG_JIT)
    target_compile_definitions(perfecthash_unit_tests PRIVATE PH_HAS_RAWDOG_JIT)
endif()
if(TARGET PerfectHashLLVM)
    add_dependencies(perfecthash_unit_tests PerfectHashLLVM)
endif()

add_executable(perfecthash_online_jit_invalid_env
    PerfectHashOnlineJitInvalidEnv.cpp
)

target_link_libraries(perfecthash_online_jit_invalid_env
    PRIVATE
        PerfectHash
)

perfecthash_apply_common_definitions(perfecthash_online_jit_invalid_env)

set_target_properties(perfecthash_online_jit_invalid_env PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

add_executable(perfecthash_benchmarks
    PerfectHashBenchmarks.cpp
)

target_link_libraries(perfecthash_benchmarks
    PRIVATE
        PerfectHash
)

perfecthash_apply_common_definitions(perfecthash_benchmarks)

set_target_properties(perfecthash_benchmarks PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

if(PERFECTHASH_HAS_RAWDOG_JIT)
    target_compile_definitions(perfecthash_benchmarks PRIVATE PH_HAS_RAWDOG_JIT)
endif()

include(GoogleTest)
if(PERFECTHASH_ARCH_ARM64)
    add_executable(perfecthash_neon_collision_test GraphNeonCollisionTest.c)
    target_link_libraries(perfecthash_neon_collision_test PRIVATE PerfectHashOnlineCoreStatic)
    target_compile_definitions(perfecthash_neon_collision_test PRIVATE _PERFECT_HASH_INTERNAL_BUILD)
    target_include_directories(perfecthash_neon_collision_test PRIVATE "${PERFECTHASH_SRC_DIR}/PerfectHash")
    add_test(NAME perfecthash_unit_tests.Arm64NeonCollisionPrefix COMMAND perfecthash_neon_collision_test)
    set_tests_properties(perfecthash_unit_tests.Arm64NeonCollisionPrefix PROPERTIES SKIP_RETURN_CODE 77)
endif()

if((WIN32 OR PERFECTHASH_IS_LINUX) AND PERFECTHASH_ARCH_ARM64 AND PERFECTHASH_HAS_RAWDOG_JIT AND TARGET PerfectHashLLVM)
    # Exercise the slim API through the shared library used by installed Core clients.
    add_executable(perfecthash_arm64_online_core_tests PerfectHashOnlineJitTests.cpp)
    target_link_libraries(perfecthash_arm64_online_core_tests PRIVATE PerfectHashOnlineCore GTest::gtest_main)
    target_compile_features(perfecthash_arm64_online_core_tests PRIVATE cxx_std_17)
    perfecthash_apply_common_definitions(perfecthash_arm64_online_core_tests)
    target_compile_definitions(perfecthash_arm64_online_core_tests PRIVATE PH_HAS_RAWDOG_JIT)
    add_dependencies(perfecthash_arm64_online_core_tests PerfectHashLLVM)
    add_test(NAME perfecthash_unit_tests.Arm64OnlineCoreJit COMMAND perfecthash_arm64_online_core_tests)
endif()

if(TARGET PerfectHashLLVM)
    # A static client loads PerfectHashLLVM from its own directory on Windows
    # and through its rpath elsewhere. Give this one a plugin that lacks the
    # hash JIT entry points, in a directory of its own.
    add_library(perfecthash_stale_llvm_plugin SHARED StaleLlvmPlugin.c)
    add_executable(perfecthash_stale_llvm_plugin_client StaleLlvmPluginClient.cpp)
    set(ph_stale_dir "${CMAKE_CURRENT_BINARY_DIR}/stale-llvm-plugin")
    set(ph_stale_dirs)
    foreach(ph_kind RUNTIME LIBRARY ARCHIVE)
        list(APPEND ph_stale_dirs ${ph_kind}_OUTPUT_DIRECTORY "${ph_stale_dir}")
        foreach(config ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER "${config}" config_upper)
            list(APPEND ph_stale_dirs
                ${ph_kind}_OUTPUT_DIRECTORY_${config_upper} "${ph_stale_dir}/${config}")
        endforeach()
    endforeach()
    set_target_properties(
        perfecthash_stale_llvm_plugin perfecthash_stale_llvm_plugin_client
        PROPERTIES ${ph_stale_dirs}
    )
    set_target_properties(perfecthash_stale_llvm_plugin PROPERTIES OUTPUT_NAME PerfectHashLLVM)
    target_link_libraries(perfecthash_stale_llvm_plugin_client PRIVATE PerfectHashOnlineCoreStatic)
    target_compile_features(perfecthash_stale_llvm_plugin_client PRIVATE cxx_std_17)
    add_dependencies(perfecthash_stale_llvm_plugin_client perfecthash_stale_llvm_plugin)
    if(NOT WIN32)
        # A target's own link options precede the archive's rpath to the real plugin.
        target_link_options(perfecthash_stale_llvm_plugin_client PRIVATE
            "LINKER:-rpath,$<TARGET_FILE_DIR:perfecthash_stale_llvm_plugin>")
    endif()
    add_test(
        NAME perfecthash.llvm.stale_plugin
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_stale_llvm_plugin.py
            $<TARGET_FILE:perfecthash_stale_llvm_plugin_client>
    )
endif()

gtest_discover_tests(perfecthash_unit_tests
    TEST_PREFIX "perfecthash_unit_tests."
)

add_test(
    NAME perfecthash_unit_tests.PerfectHashOnlineJitInvalidEnvRejected
    COMMAND ${CMAKE_COMMAND} -E env
        LD_LIBRARY_PATH=$<TARGET_FILE_DIR:PerfectHash>
        PERFECT_HASH_ONLINE_MAX_CONCURRENCY=abc
        $<TARGET_FILE:perfecthash_online_jit_invalid_env>
)

add_test(
    NAME perfecthash.fast.unit
    COMMAND
        $<TARGET_FILE:perfecthash_unit_tests>
        "--gtest_filter=-PerfectHashOnlineTests.Assigned16Boundary32767Vs32768Keys:PerfectHashOnlineTests.Assigned16RequiresGraphImpl3AndOptIn"
)
set_tests_properties(perfecthash.fast.unit PROPERTIES LABELS fast)

if(PERFECTHASH_HAS_CUDA)
    add_executable(perfecthash_cuda_best_graph_test GraphCuBestGraphTest.c)
    target_link_libraries(perfecthash_cuda_best_graph_test PRIVATE PerfectHashOnlineCoreStatic)
    target_compile_definitions(perfecthash_cuda_best_graph_test PRIVATE _PERFECT_HASH_INTERNAL_BUILD)
    target_include_directories(perfecthash_cuda_best_graph_test PRIVATE "${PERFECTHASH_SRC_DIR}/PerfectHash")
    add_test(NAME perfecthash.cuda.best-graph-ownership COMMAND perfecthash_cuda_best_graph_test)
    set_tests_properties(perfecthash.cuda.best-graph-ownership PROPERTIES LABELS "cuda")

    add_executable(perfecthash_cuda_threshold2_bitset_tests
        PerfectHashCudaThreshold2BitsetTests.cu
    )

    perfecthash_apply_cuda_settings(perfecthash_cuda_threshold2_bitset_tests)

    if(CUDAToolkit_FOUND)
        target_link_libraries(
            perfecthash_cuda_threshold2_bitset_tests
            PRIVATE
                CUDA::cudart
        )
    endif()

    set_target_properties(perfecthash_cuda_threshold2_bitset_tests PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
    )

    add_test(
        NAME perfecthash.cuda.threshold2_bitset
        COMMAND $<TARGET_FILE:perfecthash_cuda_threshold2_bitset_tests>
    )

    set_tests_properties(
        perfecthash.cuda.threshold2_bitset
        PROPERTIES
            LABELS "cuda"
    )
endif()

set(TEST_KEYS_FILE "${PERFECTHASH_ROOT_DIR}/keys/HologramWorld-31016.keys")
set(TEST_KEYS_DIR "${PERFECTHASH_ROOT_DIR}/keys")
set(TEST_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/cli-out")

if(TARGET PerfectHashCreateExe AND TARGET PerfectHashBulkCreateExe)
    add_test(
        NAME perfecthash.cli.exit_codes
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_cli_exit_codes.py
            $<TARGET_FILE:PerfectHashCreateExe> $<TARGET_FILE:PerfectHashBulkCreateExe>
    )

    add_test(
        NAME perfecthash.cli.usage-in-sync
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_usage_in_sync.py
            $<TARGET_FILE:PerfectHashCreateExe> ${PERFECTHASH_ROOT_DIR}/USAGE.txt
    )

    if(TARGET PerfectHashLLVM)
        # Parallel solves hash through the LLVM hash JIT, one set of seeds per graph.
        add_test(
            NAME perfecthash.cli.hash_jit
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_cli_hash_jit.py
                $<TARGET_FILE:PerfectHashCreateExe> ${TEST_KEYS_FILE}
        )
    endif()

    add_test(
        NAME perfecthash.cli.create
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
            -DTEST_KEYS=${TEST_KEYS_FILE}
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/create
            -DTEST_ARGS=Chm01|MultiplyShiftR|And|1
            -DTEST_FLAGS=--NoFileIo
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_test.cmake
    )

    add_test(
        NAME perfecthash.cli.create.assigned16-boundary
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/assigned16-boundary
            -DTEST_PYTHON=${Python3_EXECUTABLE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_assigned16_boundary_test.cmake
    )

    add_test(
        NAME perfecthash.cli.bulk
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashBulkCreateExe>
            -DTEST_KEYS=${TEST_KEYS_DIR}
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/bulk
            -DTEST_ARGS=Chm01|MultiplyShiftR|And|1
            -DTEST_FLAGS=--Quiet|--NoFileIo
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_test.cmake
    )

    add_test(
        NAME perfecthash.cli.codegen
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
            -DTEST_KEYS=${TEST_KEYS_FILE}
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/codegen
            -DTEST_ARGS=Chm01|MultiplyShiftR|And|1
            -DTEST_FLAGS=--Quiet
            -DTEST_PYTHON=${Python3_EXECUTABLE}
            -DTEST_CARGO=${CARGO_EXECUTABLE}
            -DTEST_BUILD_CONFIG=$<CONFIG>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_codegen_test.cmake
    )

    add_test(
        NAME perfecthash.cli.codegen.mulshrolate1rx
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
            -DTEST_KEYS=${TEST_KEYS_FILE}
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/codegen-mulshrolate1rx
            -DTEST_ARGS=Chm01|Mulshrolate1RX|And|1
            -DTEST_EXPECT_RESIZE=ON
            -DTEST_FLAGS=--Quiet|--AttemptsBeforeTableResize=1000
            -DTEST_PYTHON=${Python3_EXECUTABLE}
            -DTEST_CARGO=${CARGO_EXECUTABLE}
            -DTEST_BUILD_CONFIG=$<CONFIG>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_codegen_test.cmake
    )

    add_test(
        NAME perfecthash.cli.codegen.mulshrolate2rx
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
            -DTEST_KEYS=${TEST_KEYS_FILE}
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/codegen-mulshrolate2rx
            -DTEST_ARGS=Chm01|Mulshrolate2RX|And|1
            -DTEST_FLAGS=--Quiet
            -DTEST_PYTHON=${Python3_EXECUTABLE}
            -DTEST_CARGO=${CARGO_EXECUTABLE}
            -DTEST_BUILD_CONFIG=$<CONFIG>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_codegen_test.cmake
    )

    add_test(
        NAME perfecthash.cli.codegen.mulshrolate3rx
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
            -DTEST_KEYS=${TEST_KEYS_FILE}
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/codegen-mulshrolate3rx
            -DTEST_ARGS=Chm01|Mulshrolate3RX|And|1
            -DTEST_FLAGS=--Quiet
            -DTEST_PYTHON=${Python3_EXECUTABLE}
            -DTEST_CARGO=${CARGO_EXECUTABLE}
            -DTEST_BUILD_CONFIG=$<CONFIG>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_codegen_test.cmake
    )

    add_test(
        NAME perfecthash.cli.codegen.mulshrolate4rx
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
            -DTEST_KEYS=${TEST_KEYS_FILE}
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/codegen-mulshrolate4rx
            -DTEST_ARGS=Chm01|Mulshrolate4RX|And|1
            -DTEST_FLAGS=--Quiet
            -DTEST_PYTHON=${Python3_EXECUTABLE}
            -DTEST_CARGO=${CARGO_EXECUTABLE}
            -DTEST_BUILD_CONFIG=$<CONFIG>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_codegen_test.cmake
    )

    add_test(
        NAME perfecthash.cli.codegen.multiplyshiftrx
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
            -DTEST_KEYS=${TEST_KEYS_FILE}
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/codegen-multiplyshiftrx
            -DTEST_ARGS=Chm01|MultiplyShiftRX|And|1
            -DTEST_EXPECT_RESIZE=ON
            -DTEST_FLAGS=--Quiet|--AttemptsBeforeTableResize=1000
            -DTEST_PYTHON=${Python3_EXECUTABLE}
            -DTEST_CARGO=${CARGO_EXECUTABLE}
            -DTEST_BUILD_CONFIG=$<CONFIG>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_codegen_test.cmake
    )

    add_test(
        NAME perfecthash.fast.cli.create
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
            -DTEST_KEYS=${TEST_KEYS_FILE}
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/fast-create
            -DTEST_ARGS=Chm01|Mulshrolate4RX|And|1
            -DTEST_FLAGS=--NoFileIo
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_test.cmake
    )

    add_test(
        NAME perfecthash.fast.cli.bulk
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashBulkCreateExe>
            -DTEST_KEYS=${TEST_KEYS_DIR}
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/fast-bulk
            -DTEST_ARGS=Chm01|Mulshrolate4RX|And|1
            -DTEST_FLAGS=--Quiet|--NoFileIo
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_test.cmake
    )

    add_test(
        NAME perfecthash.fast.cli.codegen
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
            -DTEST_KEYS=${TEST_KEYS_FILE}
            -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/fast-codegen
            -DTEST_ARGS=Chm01|Mulshrolate4RX|And|1
            -DTEST_FLAGS=--Quiet
            -DTEST_PYTHON=${Python3_EXECUTABLE}
            -DTEST_CARGO=${CARGO_EXECUTABLE}
            -DTEST_BUILD_CONFIG=$<CONFIG>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_codegen_test.cmake
    )

    set_tests_properties(
        perfecthash.fast.cli.create
        perfecthash.fast.cli.bulk
        perfecthash.fast.cli.codegen
        PROPERTIES
            LABELS fast
    )

    if(PERFECTHASH_HAS_CUDA)
        add_test(
            NAME perfecthash.cuda.chm02.hologram.nofileio
            COMMAND ${CMAKE_COMMAND}
                -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
                -DTEST_KEYS=${TEST_KEYS_FILE}
                -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/cuda-chm02-hologram-nofileio
                "-DTEST_FLAGS=--NoFileIo;--DisableCsvOutputFile"
                -DREQUIRE_GPU_ASSIGN=1
                -DREQUIRE_GPU_ORDER_VALID=1
                -DREQUIRE_GPU_VERIFY=1
                -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_chm02_cuda_known_seed_test.cmake
        )

        add_test(
            NAME perfecthash.cuda.chm02.hologram.fileio
            COMMAND ${CMAKE_COMMAND}
                -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
                -DTEST_KEYS=${TEST_KEYS_FILE}
                -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/cuda-chm02-hologram-fileio
                -DREQUIRE_GPU_ASSIGN=1
                -DREQUIRE_GPU_ORDER_VALID=1
                -DREQUIRE_GPU_VERIFY=1
                -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_chm02_cuda_known_seed_test.cmake
        )

        add_test(
            NAME perfecthash.cuda.chm02.hologram.nondebug
            COMMAND ${CMAKE_COMMAND}
                -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
                -DTEST_KEYS=${TEST_KEYS_FILE}
                -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/cuda-chm02-hologram-nondebug
                "-DTEST_FLAGS=--NoFileIo"
                -DTEST_ENABLE_DEBUG=0
                -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_chm02_cuda_known_seed_test.cmake
        )

        add_test(
            NAME perfecthash.cuda.chm02.generated33000.nofileio
            COMMAND ${CMAKE_COMMAND}
                -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
                -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/cuda-chm02-generated33000-nofileio
                -DTEST_PYTHON=${Python3_EXECUTABLE}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_chm02_cuda_generated_33000_test.cmake
        )

        foreach(width IN ITEMS 16 32)
            add_test(
                NAME perfecthash.cuda.chm02.retained-graph.assigned${width}
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_cuda_retained_graph.py
                    $<TARGET_FILE:PerfectHashCreateExe> ${TEST_KEYS_FILE} ${width}
            )
            add_test(
                NAME perfecthash.cuda.chm02.retained-graph.mixed.assigned${width}
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_cuda_retained_graph.py
                    $<TARGET_FILE:PerfectHashCreateExe> ${TEST_KEYS_FILE} ${width} --mixed
            )
            set_tests_properties(perfecthash.cuda.chm02.retained-graph.mixed.assigned${width}
                PROPERTIES SKIP_RETURN_CODE 77)
        endforeach()

        add_test(
            NAME perfecthash.cuda.chm02.perf-surface
            COMMAND ${CMAKE_COMMAND}
                -DTEST_EXE=$<TARGET_FILE:PerfectHashCreateExe>
                -DTEST_KEYS=${TEST_KEYS_FILE}
                -DTEST_OUTPUT=${TEST_OUTPUT_DIR}/cuda-chm02-perf-surface
                -DTEST_PYTHON=${Python3_EXECUTABLE}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_chm02_cuda_perf_benchmark.cmake
        )

        set_tests_properties(
            perfecthash.cuda.chm02.hologram.nofileio
            perfecthash.cuda.chm02.hologram.fileio
            perfecthash.cuda.chm02.hologram.nondebug
            perfecthash.cuda.chm02.generated33000.nofileio
            perfecthash.cuda.chm02.retained-graph.assigned16
            perfecthash.cuda.chm02.retained-graph.assigned32
            perfecthash.cuda.chm02.retained-graph.mixed.assigned16
            perfecthash.cuda.chm02.retained-graph.mixed.assigned32
            perfecthash.cuda.chm02.perf-surface
            PROPERTIES
                LABELS "cuda"
        )
    endif()
else()
    message(STATUS "Skipping PerfectHash CLI tests for profile '${PERFECTHASH_BUILD_PROFILE_NORMALIZED}' (CLI executable targets are not available).")
endif()

# Each online JIT example carries its own copy of these modules so that it can
# be copied out of the tree. Fail when the two copies drift apart.
set(ph_console_cmake "${PERFECTHASH_ROOT_DIR}/examples/cpp-console-online-jit/cmake")
set(ph_sqlite_cmake "${PERFECTHASH_ROOT_DIR}/examples/sqlite-online-jit/cmake")
if(IS_DIRECTORY "${ph_console_cmake}" AND IS_DIRECTORY "${ph_sqlite_cmake}")
    foreach(ph_module FindPerfectHashOnlineJit PerfectHashOnlineJitRoots StagePerfectHashRuntime)
        add_test(
            NAME perfecthash.examples.online-jit-cmake-in-sync.${ph_module}
            COMMAND ${CMAKE_COMMAND} -E compare_files
                "${ph_console_cmake}/${ph_module}.cmake"
                "${ph_sqlite_cmake}/${ph_module}.cmake"
        )
    endforeach()
endif()
