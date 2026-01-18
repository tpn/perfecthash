if(NOT DEFINED TEST_EXE)
  message(FATAL_ERROR "TEST_EXE is required")
endif()
if(NOT DEFINED TEST_KEYS)
  message(FATAL_ERROR "TEST_KEYS is required")
endif()
if(NOT DEFINED TEST_OUTPUT)
  message(FATAL_ERROR "TEST_OUTPUT is required")
endif()

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
