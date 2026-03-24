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

file(MAKE_DIRECTORY "${test_output_native}")

set(args
  "Chm02"
  "Mulshrolate3RX"
  "And"
  "1"
  "--CuConcurrency=1"
  "--FixedAttempts=2"
  "--Seeds=0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847"
  "--NoFileIo"
  "--DisableCsvOutputFile"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E env PH_DEBUG_CUDA_CHM02=1
          "${test_exe_native}" "${test_keys_native}" "${test_output_native}" ${args}
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

message(STATUS "stdout: ${stdout}")
message(STATUS "stderr: ${stderr}")

if(NOT result EQUAL 0)
  message(FATAL_ERROR "Command failed with exit code ${result}")
endif()

string(FIND "${stderr}" "PerfectHashTableCreate failed" failure_index)
if(NOT failure_index EQUAL -1)
  message(FATAL_ERROR "Expected Chm02 CUDA known-seed run to succeed, but stderr reported failure.")
endif()

if(DEFINED REQUIRE_GPU_ASSIGN AND REQUIRE_GPU_ASSIGN)
  string(FIND "${stderr}" "[GraphCuAssign] GpuAssignResult=0x00000000" gpu_assign_index)
  if(gpu_assign_index EQUAL -1)
    message(FATAL_ERROR "Expected GPU assignment success log, but it was not present.")
  endif()

  string(FIND "${stderr}" "[GraphCuAssign] CpuAssignResult=" cpu_assign_index)
  if(NOT cpu_assign_index EQUAL -1)
    message(FATAL_ERROR "Expected GPU assignment path without CPU assign fallback, but CPU assign log was present.")
  endif()
endif()
