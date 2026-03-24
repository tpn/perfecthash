if(NOT DEFINED TEST_EXE)
  message(FATAL_ERROR "TEST_EXE is required")
endif()
if(NOT DEFINED TEST_OUTPUT)
  message(FATAL_ERROR "TEST_OUTPUT is required")
endif()
if(NOT DEFINED TEST_PYTHON)
  message(FATAL_ERROR "TEST_PYTHON is required")
endif()

set(test_output "${TEST_OUTPUT}")
file(MAKE_DIRECTORY "${test_output}")

set(generator_script "${test_output}/generate_cuda_33000_keys.py")
file(WRITE "${generator_script}" "import struct\n")
file(APPEND "${generator_script}" "from pathlib import Path\n")
file(APPEND "${generator_script}" "out = Path(r'''${test_output}''')\n")
file(APPEND "${generator_script}" "out.mkdir(parents=True, exist_ok=True)\n")
file(APPEND "${generator_script}" "count = 33000\n")
file(APPEND "${generator_script}" "salt = 0x13579BDF\n")
file(APPEND "${generator_script}" "multiplier = 2654435761\n")
file(APPEND "${generator_script}" "keys = []\n")
file(APPEND "${generator_script}" "i = 1\n")
file(APPEND "${generator_script}" "while len(keys) < count:\n")
file(APPEND "${generator_script}" "    value = ((i * multiplier) ^ salt) & 0xffffffff\n")
file(APPEND "${generator_script}" "    if value != 0 and value != 0xffffffff:\n")
file(APPEND "${generator_script}" "        keys.append(value)\n")
file(APPEND "${generator_script}" "    i += 1\n")
file(APPEND "${generator_script}" "keys.sort()\n")
file(APPEND "${generator_script}" "(out / 'generated-33000.keys').write_bytes(b''.join(struct.pack('<I', v) for v in keys))\n")

execute_process(
  COMMAND "${TEST_PYTHON}" "${generator_script}"
  RESULT_VARIABLE generate_result
  OUTPUT_VARIABLE generate_stdout
  ERROR_VARIABLE generate_stderr
)

if(NOT generate_result EQUAL 0)
  message(STATUS "stdout: ${generate_stdout}")
  message(STATUS "stderr: ${generate_stderr}")
  message(FATAL_ERROR "Failed to generate 33000-key fixture (exit ${generate_result})")
endif()

set(generated_keys "${test_output}/generated-33000.keys")
set(run_output "${test_output}/run")

execute_process(
  COMMAND ${CMAKE_COMMAND}
    -DTEST_EXE=${TEST_EXE}
    -DTEST_KEYS=${generated_keys}
    -DTEST_OUTPUT=${run_output}
    -DTEST_SEEDS=--Seeds=0x4FB8349A,0x64E0E8DD,0x1700010F,0xF7F80AD9
    -DREQUIRE_GPU_ASSIGN=1
    -DREQUIRE_GPU_ORDER_VALID=1
    -DREQUIRE_GPU_VERIFY=1
    -P ${CMAKE_CURRENT_LIST_DIR}/run_cli_chm02_cuda_known_seed_test.cmake
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_stdout
  ERROR_VARIABLE run_stderr
)

if(NOT run_result EQUAL 0)
  message(STATUS "stdout: ${run_stdout}")
  message(STATUS "stderr: ${run_stderr}")
  message(FATAL_ERROR "Generated 33000-key CUDA regression failed (exit ${run_result})")
endif()
