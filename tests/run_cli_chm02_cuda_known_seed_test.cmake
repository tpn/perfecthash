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

set(test_algorithm "Chm02")
if(DEFINED TEST_ALGORITHM)
  set(test_algorithm "${TEST_ALGORITHM}")
endif()

set(test_hash "Mulshrolate3RX")
if(DEFINED TEST_HASH)
  set(test_hash "${TEST_HASH}")
endif()

set(test_mask "And")
if(DEFINED TEST_MASK)
  set(test_mask "${TEST_MASK}")
endif()

set(test_concurrency "1")
if(DEFINED TEST_CONCURRENCY)
  set(test_concurrency "${TEST_CONCURRENCY}")
endif()

set(test_cu_concurrency "--CuConcurrency=1")
if(DEFINED TEST_CU_CONCURRENCY)
  set(test_cu_concurrency "${TEST_CU_CONCURRENCY}")
endif()

set(test_fixed_attempts "--FixedAttempts=2")
if(DEFINED TEST_FIXED_ATTEMPTS)
  set(test_fixed_attempts "${TEST_FIXED_ATTEMPTS}")
endif()

set(test_seeds "--Seeds=0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847")
if(DEFINED TEST_SEEDS)
  set(test_seeds "${TEST_SEEDS}")
endif()

set(args
  "${test_algorithm}"
  "${test_hash}"
  "${test_mask}"
  "${test_concurrency}"
  "${test_cu_concurrency}"
  "${test_fixed_attempts}"
  "${test_seeds}"
)

if(DEFINED TEST_FLAGS)
  foreach(flag IN LISTS TEST_FLAGS)
    list(APPEND args "${flag}")
  endforeach()
else()
  list(APPEND args "--NoFileIo" "--DisableCsvOutputFile")
endif()

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
  string(FIND "${stderr}" "PH_CHM02_CUDA_ASSIGN_OK" gpu_assign_index)
  if(gpu_assign_index EQUAL -1)
    message(FATAL_ERROR "Expected stable GPU assignment success token, but it was not present.")
  endif()
endif()

if(DEFINED REQUIRE_GPU_ORDER_VALID AND REQUIRE_GPU_ORDER_VALID)
  string(FIND "${stderr}" "PH_CHM02_CUDA_ORDER_OK" gpu_order_valid_index)
  if(gpu_order_valid_index EQUAL -1)
    message(FATAL_ERROR "Expected stable GPU order-validation success token, but it was not present.")
  endif()
endif()

if(DEFINED REQUIRE_GPU_VERIFY AND REQUIRE_GPU_VERIFY)
  string(FIND "${stderr}" "PH_CHM02_CUDA_VERIFY_OK" gpu_verify_index)
  if(gpu_verify_index EQUAL -1)
    message(FATAL_ERROR "Expected stable GPU verify success token, but it was not present.")
  endif()
endif()
