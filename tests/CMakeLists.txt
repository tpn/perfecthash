include(FetchContent)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

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
)

target_link_libraries(perfecthash_unit_tests
    PRIVATE
        PerfectHash
        GTest::gtest_main
)

perfecthash_apply_common_definitions(perfecthash_unit_tests)

set_target_properties(perfecthash_unit_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

include(GoogleTest)
gtest_discover_tests(perfecthash_unit_tests)

set(TEST_KEYS_FILE "${PERFECTHASH_ROOT_DIR}/keys/HologramWorld-31016.keys")
set(TEST_KEYS_DIR "${PERFECTHASH_ROOT_DIR}/keys")
set(TEST_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/cli-out")

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
        -DTEST_FLAGS=--Quiet
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
        -DTEST_FLAGS=--Quiet
        -DTEST_PYTHON=${Python3_EXECUTABLE}
        -DTEST_CARGO=${CARGO_EXECUTABLE}
        -DTEST_BUILD_CONFIG=$<CONFIG>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_cli_codegen_test.cmake
)
