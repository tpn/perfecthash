if(NOT DEFINED TEST_EXE)
  message(FATAL_ERROR "TEST_EXE is required")
endif()
if(NOT DEFINED TEST_OUTPUT)
  message(FATAL_ERROR "TEST_OUTPUT is required")
endif()
if(NOT DEFINED TEST_PYTHON)
  message(FATAL_ERROR "TEST_PYTHON is required")
endif()

set(test_exe "${TEST_EXE}")
set(test_output_dir "${TEST_OUTPUT}")

file(TO_NATIVE_PATH "${test_exe}" test_exe_native)
file(TO_NATIVE_PATH "${test_output_dir}" test_output_native)

file(MAKE_DIRECTORY "${test_output_native}")

set(stdout_path "${test_output_native}/stdout.json")
set(stderr_path "${test_output_native}/stderr.txt")

execute_process(
  COMMAND "${test_exe_native}"
          --edges 16
          --batch 1
          --threads 64
          --solve-mode device-serial
          --assign-geometry warp
          --device-serial-peel-geometry warp
          --output-format json
  RESULT_VARIABLE result
  OUTPUT_FILE "${stdout_path}"
  ERROR_FILE "${stderr_path}"
)

if(NOT result EQUAL 0)
  file(READ "${stderr_path}" stderr_contents)
  message(STATUS "stderr: ${stderr_contents}")
  message(FATAL_ERROR "gpu-poc geometry smoke command failed with exit code ${result}")
endif()

execute_process(
  COMMAND "${TEST_PYTHON}" -c
          "import json, pathlib, sys; data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')); assert data['assign_geometry'] == 'warp'; assert data['device_serial_peel_geometry'] == 'warp'; assert data['solve_mode'] == 'device-serial'; all_attempts = data['cpu_stage_timings_ms_all_attempts']; solved_only = data['cpu_stage_timings_ms_solved_only']; assert all(k in all_attempts for k in ('add_build', 'peel', 'assign', 'verify')); assert all(k in solved_only for k in ('add_build', 'peel', 'assign', 'verify'))"
          "${stdout_path}"
  RESULT_VARIABLE verify_result
  OUTPUT_VARIABLE verify_stdout
  ERROR_VARIABLE verify_stderr
)

if(NOT verify_result EQUAL 0)
  message(STATUS "stdout verification output: ${verify_stdout}")
  message(STATUS "stderr verification output: ${verify_stderr}")
  message(FATAL_ERROR "gpu-poc geometry JSON verification failed with exit code ${verify_result}")
endif()
