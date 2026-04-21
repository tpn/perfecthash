if(NOT DEFINED TEST_EXE)
  message(FATAL_ERROR "TEST_EXE is required")
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

get_filename_component(test_output_parent "${TEST_OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${test_output_parent}")

set(graphimpl4_keys "${test_output_parent}/graphimpl4.keys")
set(generator_script "${test_output_parent}/generate_graphimpl4_keys.py")
file(WRITE "${generator_script}" "import struct\n")
file(APPEND "${generator_script}" "from pathlib import Path\n")
file(APPEND "${generator_script}" "path = Path(r'''${graphimpl4_keys}''')\n")
file(APPEND "${generator_script}" "keys = [0x2468ACE1,0x2468ACE3,0x2468ACE5,0x2468ACE7,0x2468ACEB,0x2468ACED,0x2468ACF1,0x2468ACF3]\n")
file(APPEND "${generator_script}" "with path.open('wb') as f:\n")
file(APPEND "${generator_script}" "    for key in keys:\n")
file(APPEND "${generator_script}" "        f.write(struct.pack('<I', key))\n")

execute_process(
  COMMAND "${TEST_PYTHON}" "${generator_script}"
  RESULT_VARIABLE generate_result
  OUTPUT_VARIABLE generate_stdout
  ERROR_VARIABLE generate_stderr
)

if(NOT generate_result EQUAL 0)
  message(STATUS "stdout: ${generate_stdout}")
  message(STATUS "stderr: ${generate_stderr}")
  message(FATAL_ERROR "Failed to generate GraphImpl4 key file")
endif()

set(TEST_KEYS "${graphimpl4_keys}")
set(TEST_ARGS "Chm01|MultiplyShiftR|And|1")
set(TEST_FLAGS "--Quiet|--GraphImpl=4|--DoNotHashAllKeysFirst|--MaxSolveTimeInSeconds=5")

file(TO_NATIVE_PATH "${TEST_EXE}" test_exe_native)
file(TO_NATIVE_PATH "${TEST_KEYS}" test_keys_native)
file(TO_NATIVE_PATH "${TEST_OUTPUT}" test_output_native)

string(REPLACE "|" ";" args_list "${TEST_ARGS}")
string(REPLACE "|" ";" flags_list "${TEST_FLAGS}")

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
file(READ "${gen_cmake}" cmake_text)
string(REGEX MATCH "project\\(([^)]+)\\)" _project_match "${cmake_text}")
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "Failed to find project() name in ${gen_cmake}")
endif()
set(table_name "${CMAKE_MATCH_1}")

require_file("${test_output_native}/CompiledPerfectHash.h")
require_file("${test_output_native}/CompiledPerfectHashMacroGlue.h")
require_file("${test_output_native}/CompiledPerfectHash.props")
require_file("${gen_dir}/${table_name}.pht1")
require_file("${gen_dir}/${table_name}.pht1:Info")
require_file("${gen_dir}/${table_name}_TableData.c")
require_file("${gen_dir}/${table_name}_CppHeaderOnly.hpp")
require_file("${gen_dir}/${table_name}_RustLib.rs")
require_file("${gen_dir}/${table_name}.cu")

file(READ "${gen_dir}/${table_name}.h" generated_header)
string(FIND "${generated_header}" "extern const BYTE ${table_name}_TableData[]" has_byte_table)
if(has_byte_table LESS 0)
  message(FATAL_ERROR "Expected BYTE table data declaration in generated header")
endif()

file(READ "${gen_dir}/${table_name}_TableData.c" generated_table_data)
string(FIND "${generated_table_data}" "const BYTE ${table_name}_TableData" has_byte_table_data)
if(has_byte_table_data LESS 0)
  message(FATAL_ERROR "Expected BYTE table data definition in generated C source")
endif()

file(READ "${gen_dir}/${table_name}_RustLib.rs" generated_rust)
string(FIND "${generated_rust}" "pub type TableDataType = u8;" has_rust_u8)
if(has_rust_u8 LESS 0)
  message(FATAL_ERROR "Expected u8 table data type in generated Rust file")
endif()

file(READ "${gen_dir}/${table_name}.cu" generated_cuda)
string(FIND "${generated_cuda}" "using table_data_type = std::uint8_t;" has_cuda_u8)
if(has_cuda_u8 LESS 0)
  message(FATAL_ERROR "Expected uint8_t table data type in generated CUDA file")
endif()
