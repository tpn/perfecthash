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

set(args_list "")
set(flags_list "")

if(DEFINED TEST_ARGS)
  string(REPLACE "|" ";" args_list "${TEST_ARGS}")
endif()
if(DEFINED TEST_FLAGS)
  string(REPLACE "|" ";" flags_list "${TEST_FLAGS}")
endif()

file(REMOVE_RECURSE "${TEST_OUTPUT}")
file(MAKE_DIRECTORY "${TEST_OUTPUT}")

execute_process(
  COMMAND "${TEST_EXE}" "${TEST_KEYS}" "${TEST_OUTPUT}" ${args_list} ${flags_list}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(NOT result EQUAL 0)
  message(STATUS "stdout: ${stdout}")
  message(STATUS "stderr: ${stderr}")
  message(FATAL_ERROR "Command failed with exit code ${result}")
endif()

file(GLOB cmake_lists "${TEST_OUTPUT}/*/CMakeLists.txt")
list(LENGTH cmake_lists cmake_lists_count)
if(cmake_lists_count LESS 1)
  message(FATAL_ERROR "No generated CMakeLists.txt under ${TEST_OUTPUT}")
endif()

list(GET cmake_lists 0 gen_cmake)
get_filename_component(gen_dir "${gen_cmake}" DIRECTORY)

set(build_dir "${gen_dir}/_build")
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
string(REGEX MATCH "project\\(([A-Za-z0-9_]+)\\)" _project_match "${cmake_text}")
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "Failed to find project() name in ${gen_cmake}")
endif()

set(table_name "${CMAKE_MATCH_1}")
set(test_exe_name "Test_${table_name}")
set(cpp_header_test_name "CppHeaderOnlyTest_${table_name}")

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

if(NOT TEST_CARGO STREQUAL "")
  if(EXISTS "${gen_dir}/Cargo.toml")
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
endif()
