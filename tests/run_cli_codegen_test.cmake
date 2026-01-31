if(NOT DEFINED TEST_EXE)
  message(FATAL_ERROR "TEST_EXE is required")
endif()
if(NOT DEFINED TEST_KEYS)
  message(FATAL_ERROR "TEST_KEYS is required")
endif()
if(NOT DEFINED TEST_OUTPUT)
  message(FATAL_ERROR "TEST_OUTPUT is required")
endif()
if(NOT DEFINED TEST_PYTHON)
  message(FATAL_ERROR "TEST_PYTHON is required")
endif()

if(NOT DEFINED TEST_CARGO)
  set(TEST_CARGO "")
endif()

function(require_file path)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Expected file not found: ${path}")
  endif()
endfunction()

function(require_glob pattern label)
  file(GLOB matches "${pattern}")
  list(LENGTH matches match_count)
  if(match_count LESS 1)
    message(FATAL_ERROR "Expected ${label} matching ${pattern} not found")
  endif()
endfunction()

file(TO_NATIVE_PATH "${TEST_EXE}" test_exe_native)
file(TO_NATIVE_PATH "${TEST_KEYS}" test_keys_native)
file(TO_NATIVE_PATH "${TEST_OUTPUT}" test_output_native)

set(args_list "")
set(flags_list "")

if(DEFINED TEST_ARGS)
  string(REPLACE "|" ";" args_list "${TEST_ARGS}")
endif()
if(DEFINED TEST_FLAGS)
  string(REPLACE "|" ";" flags_list "${TEST_FLAGS}")
endif()

file(REMOVE_RECURSE "${test_output_native}")
file(MAKE_DIRECTORY "${test_output_native}")

execute_process(
  COMMAND "${test_exe_native}" "${test_keys_native}" "${test_output_native}" ${args_list} ${flags_list}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(NOT result EQUAL 0)
  message(STATUS "stdout: ${stdout}")
  message(STATUS "stderr: ${stderr}")
  message(FATAL_ERROR "Command failed with exit code ${result}")
endif()

file(GLOB cmake_lists "${test_output_native}/*/CMakeLists.txt")
list(LENGTH cmake_lists cmake_lists_count)
if(cmake_lists_count LESS 1)
  message(FATAL_ERROR "No generated CMakeLists.txt under ${TEST_OUTPUT}")
endif()

list(GET cmake_lists 0 gen_cmake)
get_filename_component(gen_dir "${gen_cmake}" DIRECTORY)

set(build_dir "${test_output_native}/_build")
file(MAKE_DIRECTORY "${build_dir}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${gen_dir}" -B "${build_dir}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(NOT result EQUAL 0)
  message(STATUS "stdout: ${stdout}")
  message(STATUS "stderr: ${stderr}")
  message(FATAL_ERROR "CMake configure failed with exit code ${result}")
endif()

set(build_command "${CMAKE_COMMAND}" --build "${build_dir}")
if(DEFINED TEST_BUILD_CONFIG AND NOT TEST_BUILD_CONFIG STREQUAL "")
  list(APPEND build_command --config "${TEST_BUILD_CONFIG}")
endif()

execute_process(
  COMMAND ${build_command}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(NOT result EQUAL 0)
  message(STATUS "stdout: ${stdout}")
  message(STATUS "stderr: ${stderr}")
  message(FATAL_ERROR "CMake build failed with exit code ${result}")
endif()

file(READ "${gen_cmake}" cmake_text)
string(REGEX MATCH "project\\(([^)]+)\\)" _project_match "${cmake_text}")
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "Failed to find project() name in ${gen_cmake}")
endif()

set(table_name "${CMAKE_MATCH_1}")

require_file("${test_output_native}/CompiledPerfectHash.h")
require_file("${test_output_native}/CompiledPerfectHashMacroGlue.h")
require_file("${test_output_native}/CompiledPerfectHash.props")
require_file("${test_output_native}/no_sal2.h")
require_glob("${test_output_native}/PerfectHashTableCreate*.csv"
             "PerfectHash CSV output")

require_file("${gen_dir}/${table_name}.c")
require_file("${gen_dir}/${table_name}.h")
require_file("${gen_dir}/${table_name}.cpp")
require_file("${gen_dir}/${table_name}.def")
require_file("${gen_dir}/${table_name}.pht1")
require_file("${gen_dir}/${table_name}.sln")
require_file("${gen_dir}/${table_name}_StdAfx.c")
require_file("${gen_dir}/${table_name}_StdAfx.h")
require_file("${gen_dir}/${table_name}_Support.c")
require_file("${gen_dir}/${table_name}_Support.h")
require_file("${gen_dir}/${table_name}_TableData.c")
require_file("${gen_dir}/${table_name}_TableValues.c")
require_file("${gen_dir}/${table_name}_Keys.c")
require_file("${gen_dir}/${table_name}_Types.h")
require_file("${gen_dir}/${table_name}_Test.c")
require_file("${gen_dir}/${table_name}_TestExe.c")
require_file("${gen_dir}/${table_name}_BenchmarkIndex.c")
require_file("${gen_dir}/${table_name}_BenchmarkIndexExe.c")
require_file("${gen_dir}/${table_name}_BenchmarkFull.c")
require_file("${gen_dir}/${table_name}_BenchmarkFullExe.c")
require_file("${gen_dir}/${table_name}_CppHeaderOnlyTest.cpp")
require_file("${gen_dir}/${table_name}_CppHeaderOnly.hpp")
require_file("${gen_dir}/${table_name}_Python.py")
require_file("${gen_dir}/${table_name}_RustLib.rs")
require_file("${gen_dir}/${table_name}_RustTest.rs")
require_file("${gen_dir}/${table_name}_RustBench.rs")
require_file("${gen_dir}/Makefile")
require_file("${gen_dir}/main.mk")
require_file("${gen_dir}/Cargo.toml")
require_glob("${gen_dir}/*.vcxproj" "Visual Studio projects")
require_glob("${gen_dir}/*_Lib.mk" "Lib makefile fragment")
require_glob("${gen_dir}/*_So.mk" "Shared object makefile fragment")
require_glob("${gen_dir}/*_Test.mk" "Test makefile fragment")
require_glob("${gen_dir}/*_BenchmarkFull.mk" "BenchmarkFull makefile fragment")
require_glob("${gen_dir}/*_BenchmarkIndex.mk" "BenchmarkIndex makefile fragment")

if(NOT CMAKE_HOST_WIN32)
  set(make_build_dir "${test_output_native}/_build_make")
  file(MAKE_DIRECTORY "${make_build_dir}")

  set(make_config_args "")
  if(DEFINED TEST_BUILD_CONFIG AND NOT TEST_BUILD_CONFIG STREQUAL "")
    list(APPEND make_config_args "-DCMAKE_BUILD_TYPE=${TEST_BUILD_CONFIG}")
  endif()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${gen_dir}" -B "${make_build_dir}"
            -G "Unix Makefiles" ${make_config_args}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )

  if(NOT result EQUAL 0)
    message(STATUS "stdout: ${stdout}")
    message(STATUS "stderr: ${stderr}")
    message(FATAL_ERROR "Unix Makefiles configure failed with exit code ${result}")
  endif()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${make_build_dir}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )

  if(NOT result EQUAL 0)
    message(STATUS "stdout: ${stdout}")
    message(STATUS "stderr: ${stderr}")
    message(FATAL_ERROR "Unix Makefiles build failed with exit code ${result}")
  endif()
endif()

set(test_exe_name "Test_${table_name}")
set(cpp_header_test_name "CppHeaderOnlyTest_${table_name}")
set(cuda_test_name "CudaTest_${table_name}")

set(candidate_paths
  "${build_dir}/${test_exe_name}"
  "${build_dir}/${test_exe_name}.exe"
)

if(DEFINED TEST_BUILD_CONFIG AND NOT TEST_BUILD_CONFIG STREQUAL "")
  list(APPEND candidate_paths
    "${build_dir}/${TEST_BUILD_CONFIG}/${test_exe_name}"
    "${build_dir}/${TEST_BUILD_CONFIG}/${test_exe_name}.exe"
  )
endif()

set(test_exe_path "")
foreach(candidate IN LISTS candidate_paths)
  if(EXISTS "${candidate}")
    set(test_exe_path "${candidate}")
    break()
  endif()
endforeach()

if(test_exe_path STREQUAL "")
  message(FATAL_ERROR "Failed to locate test executable for ${test_exe_name}")
endif()

execute_process(
  COMMAND "${test_exe_path}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(NOT result EQUAL 0)
  message(STATUS "stdout: ${stdout}")
  message(STATUS "stderr: ${stderr}")
  message(FATAL_ERROR "Generated test executable failed with exit code ${result}")
endif()

set(cpp_header_test_path "")
set(cpp_candidate_paths
  "${build_dir}/${cpp_header_test_name}"
  "${build_dir}/${cpp_header_test_name}.exe"
)

if(DEFINED TEST_BUILD_CONFIG AND NOT TEST_BUILD_CONFIG STREQUAL "")
  list(APPEND cpp_candidate_paths
    "${build_dir}/${TEST_BUILD_CONFIG}/${cpp_header_test_name}"
    "${build_dir}/${TEST_BUILD_CONFIG}/${cpp_header_test_name}.exe"
  )
endif()

foreach(candidate IN LISTS cpp_candidate_paths)
  if(EXISTS "${candidate}")
    set(cpp_header_test_path "${candidate}")
    break()
  endif()
endforeach()

if(cpp_header_test_path STREQUAL "")
  message(FATAL_ERROR "Failed to locate C++ header-only test executable for ${cpp_header_test_name}")
endif()

execute_process(
  COMMAND "${cpp_header_test_path}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(NOT result EQUAL 0)
  message(STATUS "stdout: ${stdout}")
  message(STATUS "stderr: ${stderr}")
  message(FATAL_ERROR "C++ header-only test failed with exit code ${result}")
endif()

set(cuda_test_path "")
set(cuda_candidate_paths
  "${build_dir}/${cuda_test_name}"
  "${build_dir}/${cuda_test_name}.exe"
)

if(DEFINED TEST_BUILD_CONFIG AND NOT TEST_BUILD_CONFIG STREQUAL "")
  list(APPEND cuda_candidate_paths
    "${build_dir}/${TEST_BUILD_CONFIG}/${cuda_test_name}"
    "${build_dir}/${TEST_BUILD_CONFIG}/${cuda_test_name}.exe"
  )
endif()

foreach(candidate IN LISTS cuda_candidate_paths)
  if(EXISTS "${candidate}")
    set(cuda_test_path "${candidate}")
    break()
  endif()
endforeach()

if(NOT cuda_test_path STREQUAL "")
  execute_process(
    COMMAND "${cuda_test_path}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )

  if(NOT result EQUAL 0)
    message(STATUS "stdout: ${stdout}")
    message(STATUS "stderr: ${stderr}")
    message(FATAL_ERROR "CUDA test failed with exit code ${result}")
  endif()
endif()

file(GLOB test_python "${gen_dir}/test_*.py")
list(LENGTH test_python test_python_count)
if(test_python_count LESS 1)
  message(FATAL_ERROR "No generated test_*.py files under ${gen_dir}")
endif()

foreach(test_file IN LISTS test_python)
  execute_process(
    COMMAND "${TEST_PYTHON}" "${test_file}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )
  if(NOT result EQUAL 0)
    message(STATUS "stdout: ${stdout}")
    message(STATUS "stderr: ${stderr}")
    message(FATAL_ERROR "Python test failed with exit code ${result}")
  endif()
endforeach()

if(EXISTS "${gen_dir}/Cargo.toml")
  if(TEST_CARGO STREQUAL "")
    message(FATAL_ERROR "Cargo.toml present but TEST_CARGO is empty")
  endif()

  execute_process(
    COMMAND "${TEST_CARGO}" test --quiet
    WORKING_DIRECTORY "${gen_dir}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )
  if(NOT result EQUAL 0)
    message(STATUS "stdout: ${stdout}")
    message(STATUS "stderr: ${stderr}")
    message(FATAL_ERROR "Cargo test failed with exit code ${result}")
  endif()

  execute_process(
    COMMAND "${TEST_CARGO}" run --quiet --bin rust_test
    WORKING_DIRECTORY "${gen_dir}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )
  if(NOT result EQUAL 0)
    message(STATUS "stdout: ${stdout}")
    message(STATUS "stderr: ${stderr}")
    message(FATAL_ERROR "Cargo rust_test failed with exit code ${result}")
  endif()

  execute_process(
    COMMAND "${TEST_CARGO}" run --quiet --bin rust_bench --release
    WORKING_DIRECTORY "${gen_dir}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )
  if(NOT result EQUAL 0)
    message(STATUS "stdout: ${stdout}")
    message(STATUS "stderr: ${stderr}")
    message(FATAL_ERROR "Cargo rust_bench failed with exit code ${result}")
  endif()
endif()