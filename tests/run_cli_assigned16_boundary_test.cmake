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

set(generator_script "${test_output}/generate_boundary_keys.py")
file(WRITE "${generator_script}" "import struct\n")
file(APPEND "${generator_script}" "from pathlib import Path\n")
file(APPEND "${generator_script}" "out = Path(r'''${test_output}''')\n")
file(APPEND "${generator_script}" "out.mkdir(parents=True, exist_ok=True)\n")
file(APPEND "${generator_script}" "def make_keys(count, salt):\n")
file(APPEND "${generator_script}" "    keys = []\n")
file(APPEND "${generator_script}" "    i = 1\n")
file(APPEND "${generator_script}" "    multiplier = 2654435761\n")
file(APPEND "${generator_script}" "    while len(keys) < count:\n")
file(APPEND "${generator_script}" "        value = ((i * multiplier) ^ salt) & 0xffffffff\n")
file(APPEND "${generator_script}" "        if value != 0 and value != 0xffffffff:\n")
file(APPEND "${generator_script}" "            keys.append(value)\n")
file(APPEND "${generator_script}" "        i += 1\n")
file(APPEND "${generator_script}" "    keys.sort()\n")
file(APPEND "${generator_script}" "    return keys\n")
file(APPEND "${generator_script}" "(out / 'keys-32768.keys').write_bytes(b''.join(struct.pack('<I', v) for v in make_keys(32768, 0x5a5a5a5a)))\n")
file(APPEND "${generator_script}" "(out / 'keys-32767.keys').write_bytes(b''.join(struct.pack('<I', v) for v in make_keys(32767, 0xa5a5a5a5)))\n")

execute_process(
  COMMAND "${TEST_PYTHON}" "${generator_script}"
  RESULT_VARIABLE generate_result
  OUTPUT_VARIABLE generate_stdout
  ERROR_VARIABLE generate_stderr
)

if(NOT generate_result EQUAL 0)
  message(STATUS "stdout: ${generate_stdout}")
  message(STATUS "stderr: ${generate_stderr}")
  message(FATAL_ERROR "Failed to generate boundary key files (exit ${generate_result})")
endif()

file(TO_NATIVE_PATH "${TEST_EXE}" test_exe_native)

function(run_create_case key_file output_dir)
  file(MAKE_DIRECTORY "${output_dir}")
  file(TO_NATIVE_PATH "${key_file}" key_native)
  file(TO_NATIVE_PATH "${output_dir}" output_native)

  execute_process(
    COMMAND
      "${test_exe_native}"
      "${key_native}"
      "${output_native}"
      Chm01
      MultiplyShiftR
      And
      0
      --DoNotHashAllKeysFirst
      --NoFileIo
      --Silent
      --MaxNumberOfTableResizes=0
      --MaxSolveTimeInSeconds=20
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )

  if(NOT result EQUAL 0)
    message(STATUS "stdout: ${stdout}")
    message(STATUS "stderr: ${stderr}")
    message(FATAL_ERROR
      "PerfectHashCreate failed for ${key_file} with exit code ${result}")
  endif()
endfunction()

run_create_case("${test_output}/keys-32768.keys" "${test_output}/out-32768")
run_create_case("${test_output}/keys-32767.keys" "${test_output}/out-32767")
