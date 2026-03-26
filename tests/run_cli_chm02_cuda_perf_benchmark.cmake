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

file(TO_NATIVE_PATH "${TEST_EXE}" test_exe_native)
file(TO_NATIVE_PATH "${TEST_KEYS}" test_keys_native)
file(TO_NATIVE_PATH "${TEST_OUTPUT}" test_output_native)
file(REMOVE_RECURSE "${test_output_native}")
file(MAKE_DIRECTORY "${test_output_native}")

set(test_args
  Chm02
  Mulshrolate3RX
  And
  1
  --CuConcurrency=1
  --FixedAttempts=2
  --Seeds=0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847
  --NoFileIo
  --SkipTestAfterCreate
)

execute_process(
  COMMAND "${test_exe_native}" "${test_keys_native}" "${test_output_native}" ${test_args}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(NOT result EQUAL 0)
  message(STATUS "stdout: ${stdout}")
  message(STATUS "stderr: ${stderr}")
  message(FATAL_ERROR "Command failed with exit code ${result}")
endif()

file(GLOB csv_files "${test_output_native}/*.csv")
list(LENGTH csv_files csv_count)
if(NOT csv_count EQUAL 1)
  message(FATAL_ERROR "Expected exactly one CSV output file, found ${csv_count}")
endif()

set(csv_path "${csv_files}")

execute_process(
  COMMAND "${TEST_PYTHON}" -c
          "import csv, sys; \
path = sys.argv[1]; \
required = ['CuAddKeysMicroseconds', 'CuIsAcyclicMicroseconds', 'CuAssignMicroseconds', 'CuVerifyMicroseconds']; \
row = next(csv.DictReader(open(path, newline=''))); \
missing = [name for name in required if name not in row]; \
assert not missing, f'Missing timing field(s): {missing}'; \
negative = [name for name in required if int(row[name]) < 0]; \
assert not negative, f'Negative timing field(s): {negative}'" "${csv_path}"
  RESULT_VARIABLE parse_result
  OUTPUT_VARIABLE parse_stdout
  ERROR_VARIABLE parse_stderr
)

if(NOT parse_result EQUAL 0)
  message(STATUS "stdout: ${parse_stdout}")
  message(STATUS "stderr: ${parse_stderr}")
  message(FATAL_ERROR "CSV timing validation failed (exit ${parse_result})")
endif()
