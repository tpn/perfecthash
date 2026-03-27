import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "benchmark_gpu_solver.py"


class BenchmarkGpuSolverSmokeTests(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

    def make_stub_executable(self, temp_path: Path, name: str, log_path: Path):
        stub_path = temp_path / name
        stub_path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "import json",
                    "import os",
                    "import sys",
                    "from pathlib import Path",
                    "",
                    "log_path = Path(os.environ['STUB_LOG_PATH'])",
                    "log_path.write_text(json.dumps({'argv': sys.argv[1:]}, indent=2), encoding='utf-8')",
                    "print('stub stdout')",
                    "print('stub stderr', file=sys.stderr)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stub_path.chmod(0o755)
        return stub_path, {"STUB_LOG_PATH": str(log_path)}

    def test_missing_required_config_sections_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text("{}\n", encoding="utf-8")

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "local",
                "--output",
                str(output_path),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("datasets", result.stderr)
            self.assertIn("variants", result.stderr)
            self.assertIn("output_options", result.stderr)

    def test_missing_config_file_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "missing.json"
            output_path = temp_path / "out.json"

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "local",
                "--output",
                str(output_path),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not exist", result.stderr)

    def test_non_mapping_top_level_config_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text("42\n", encoding="utf-8")

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "local",
                "--output",
                str(output_path),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("top-level", result.stderr)

    def test_dataset_name_with_path_separator_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"name": "../bad", "kind": "repo", "path": "keys/HologramWorld-31016.keys"}
                        ],
                        "variants": [],
                        "output_options": {},
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "local",
                "--output",
                str(output_path),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("safe identifier", result.stderr)

    def test_dry_run_with_empty_matrix_writes_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [],
                        "variants": [],
                        "output_options": {},
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "local",
                "--output",
                str(output_path),
                "--dry-run",
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["machine_label"], "local")
            self.assertEqual(payload["runs"], [])
            self.assertTrue(payload["dry_run"])

    def test_non_dry_run_with_empty_matrix_writes_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [],
                        "variants": [],
                        "output_options": {},
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "local",
                "--output",
                str(output_path),
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["dry_run"])
            self.assertEqual(payload["runs"], [])

    def test_dry_run_expands_datasets_and_variants_into_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"name": "hologram31016", "kind": "repo", "path": "keys/HologramWorld-31016.keys"},
                            {"name": "generated33000", "kind": "generated", "count": 33000, "salt": 324508639},
                        ],
                        "variants": [
                            {"name": "cuda-chm02", "solver_family": "cuda-chm02", "algorithm": "Chm02", "hash_function": "Mulshrolate3RX", "mask_function": "And", "fixed_attempts": 128, "allocation_mode": "explicit-device"},
                            {"name": "gpu-poc", "solver_family": "gpu-poc", "solve_mode": "device-serial", "hash_function": "Mulshrolate3RX", "batch": 128, "threads": 128, "storage_bits": "auto", "allocation_mode": "managed-default"},
                        ],
                        "output_options": {"format": "json"},
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "gb10",
                "--output",
                str(output_path),
                "--dry-run",
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_count"], 4)
            self.assertEqual(len(payload["runs"]), 4)
            self.assertEqual(payload["runs"][0]["machine_label"], "gb10")
            self.assertEqual(payload["runs"][0]["dataset"]["name"], "hologram31016")
            self.assertEqual(payload["runs"][0]["variant"]["name"], "cuda-chm02")
            self.assertEqual(payload["output_options"]["format"], "json")

    def test_filtered_dry_run_limits_to_one_safe_case(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"name": "hologram31016", "kind": "repo", "path": "keys/HologramWorld-31016.keys"},
                            {"name": "generated8193", "kind": "generated", "count": 8193, "salt": 2779096485},
                        ],
                        "variants": [
                            {"name": "cpu-cli-chm01-single", "solver_family": "cpu-cli", "algorithm": "Chm01", "hash_function": "Mulshrolate3RX", "mask_function": "And", "fixed_attempts": 128},
                            {"name": "cuda-chm02-single", "solver_family": "cuda-chm02", "algorithm": "Chm02", "hash_function": "Mulshrolate3RX", "mask_function": "And", "fixed_attempts": 128, "allocation_mode": "explicit-device"},
                            {"name": "gpu-poc-device-serial", "solver_family": "gpu-poc", "solve_mode": "device-serial", "hash_function": "Mulshrolate3RX", "batch": 128, "threads": 128, "storage_bits": "auto", "allocation_mode": "managed-default"},
                        ],
                        "output_options": {"format": "json"},
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "gb10",
                "--output",
                str(output_path),
                "--dry-run",
                "--dataset",
                "hologram31016",
                "--variant",
                "cpu-cli-chm01-single",
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_count"], 1)
            self.assertEqual(payload["runs"][0]["dataset"]["name"], "hologram31016")
            self.assertEqual(payload["runs"][0]["variant"]["name"], "cpu-cli-chm01-single")
            self.assertTrue(payload["dry_run"])

    def test_dry_run_max_runs_caps_the_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"name": "hologram31016", "kind": "repo", "path": "keys/HologramWorld-31016.keys"},
                            {"name": "generated33000", "kind": "generated", "count": 33000, "salt": 324508639},
                        ],
                        "variants": [
                            {"name": "cuda-chm02", "solver_family": "cuda-chm02", "algorithm": "Chm02", "hash_function": "Mulshrolate3RX", "mask_function": "And", "fixed_attempts": 128},
                            {"name": "gpu-poc", "solver_family": "gpu-poc", "solve_mode": "device-serial", "hash_function": "Mulshrolate3RX", "batch": 128, "threads": 128, "storage_bits": "auto", "allocation_mode": "managed-default"},
                        ],
                        "output_options": {"format": "json"},
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "gb10",
                "--output",
                str(output_path),
                "--dry-run",
                "--max-runs",
                "1",
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_count"], 1)
            self.assertEqual(len(payload["runs"]), 1)

    def test_non_dry_run_without_filters_fails_safely(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"name": "hologram31016", "kind": "repo", "path": "keys/HologramWorld-31016.keys"}
                        ],
                        "variants": [
                            {"name": "cuda-chm02-single", "solver_family": "cuda-chm02", "algorithm": "Chm02", "hash_function": "Mulshrolate3RX", "mask_function": "And", "fixed_attempts": 128}
                        ],
                        "output_options": {},
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "gb10",
                "--output",
                str(output_path),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("explicit dataset and variant filters", result.stderr)
            self.assertFalse(output_path.exists())

    def test_non_dry_run_rejects_incompatible_dataset_and_variant_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"name": "generated8193", "kind": "generated", "count": 8193, "salt": 2779096485}
                        ],
                        "variants": [
                            {"name": "cpu-cli-chm01-single", "solver_family": "cpu-cli", "algorithm": "Chm01", "hash_function": "Mulshrolate3RX", "mask_function": "And", "fixed_attempts": 2}
                        ],
                        "output_options": {},
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "gb10",
                "--output",
                str(output_path),
                "--dataset",
                "generated8193",
                "--variant",
                "cpu-cli-chm01-single",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsupported", result.stderr.lower())
            self.assertFalse(output_path.exists())

    def test_non_dry_run_rejects_repo_dataset_outside_repo_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"name": "hologram31016", "kind": "repo", "path": "/etc/hosts"}
                        ],
                        "variants": [
                            {"name": "cpu-cli-chm01-single", "solver_family": "cpu-cli", "algorithm": "Chm01", "hash_function": "Mulshrolate3RX", "mask_function": "And", "fixed_attempts": 2, "known_seeds": {"hologram31016": "0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847"}}
                        ],
                        "output_options": {},
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "gb10",
                "--output",
                str(output_path),
                "--dataset",
                "hologram31016",
                "--variant",
                "cpu-cli-chm01-single",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside repo root", result.stderr)
            self.assertFalse(output_path.exists())

    def test_non_dry_run_rejects_unsafe_gpu_poc_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"name": "generated8193", "kind": "generated", "count": 8193, "salt": 2779096485}
                        ],
                        "variants": [
                            {"name": "gpu-poc-device-serial", "solver_family": "gpu-poc", "solve_mode": "device-serial", "hash_function": "Mulshrolate3RX", "batch": 999999, "threads": 999999, "storage_bits": "auto", "allocation_mode": "managed-default"}
                        ],
                        "output_options": {},
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "gb10",
                "--output",
                str(output_path),
                "--dataset",
                "generated8193",
                "--variant",
                "gpu-poc-device-serial",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsafe", result.stderr.lower())
            self.assertFalse(output_path.exists())

    def test_cpu_cli_safe_execution_runs_exactly_one_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            log_path = temp_path / "cpu-cli-log.json"
            stub_path, stub_env = self.make_stub_executable(temp_path, "perfect-hash-create-stub.py", log_path)
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"name": "hologram31016", "kind": "repo", "path": "keys/HologramWorld-31016.keys"}
                        ],
                        "variants": [
                            {"name": "cpu-cli-chm01-single", "solver_family": "cpu-cli", "algorithm": "Chm01", "hash_function": "Mulshrolate3RX", "mask_function": "And", "fixed_attempts": 2, "known_seeds": {"hologram31016": "0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847"}}
                        ],
                        "output_options": {},
                    }
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(stub_env)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--config",
                    str(config_path),
                    "--machine-label",
                    "gb10",
                    "--output",
                    str(output_path),
                    "--dataset",
                    "hologram31016",
                    "--variant",
                    "cpu-cli-chm01-single",
                    "--perfect-hash-create-exe",
                    str(stub_path),
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=env,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_count"], 1)
            self.assertEqual(payload["runs"][0]["status"], "executed")
            self.assertEqual(payload["runs"][0]["returncode"], 0)
            stub_log = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertIn("Chm01", stub_log["argv"])
            self.assertIn("Mulshrolate3RX", stub_log["argv"])
            self.assertIn("--Seeds=0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847", stub_log["argv"])
            self.assertEqual(payload["runs"][0]["executed_command"][1:], stub_log["argv"])

    def test_cuda_chm02_safe_execution_runs_exactly_one_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            log_path = temp_path / "cuda-log.json"
            stub_path, stub_env = self.make_stub_executable(temp_path, "perfect-hash-create-stub.py", log_path)
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"name": "hologram31016", "kind": "repo", "path": "keys/HologramWorld-31016.keys"}
                        ],
                        "variants": [
                            {"name": "cuda-chm02-single", "solver_family": "cuda-chm02", "algorithm": "Chm02", "hash_function": "Mulshrolate3RX", "mask_function": "And", "fixed_attempts": 2, "allocation_mode": "explicit-device", "known_seeds": {"hologram31016": "0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847"}}
                        ],
                        "output_options": {},
                    }
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(stub_env)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--config",
                    str(config_path),
                    "--machine-label",
                    "gb10",
                    "--output",
                    str(output_path),
                    "--dataset",
                    "hologram31016",
                    "--variant",
                    "cuda-chm02-single",
                    "--perfect-hash-create-exe",
                    str(stub_path),
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=env,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_count"], 1)
            self.assertEqual(payload["runs"][0]["status"], "executed")
            self.assertEqual(payload["runs"][0]["returncode"], 0)
            stub_log = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertIn("Chm02", stub_log["argv"])
            self.assertIn("--CuConcurrency=1", stub_log["argv"])
            self.assertIn("--Seeds=0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847", stub_log["argv"])
            self.assertNotIn("--DisableCsvOutputFile", stub_log["argv"])
            self.assertEqual(payload["runs"][0]["executed_command"][1:], stub_log["argv"])

    def test_gpu_poc_safe_execution_runs_exactly_one_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            log_path = temp_path / "poc-log.json"
            stub_path, stub_env = self.make_stub_executable(temp_path, "gpu-poc-stub.py", log_path)
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"name": "generated8193", "kind": "generated", "count": 8193, "salt": 2779096485}
                        ],
                        "variants": [
                            {"name": "gpu-poc-device-serial", "solver_family": "gpu-poc", "solve_mode": "device-serial", "hash_function": "Mulshrolate3RX", "batch": 128, "threads": 128, "storage_bits": "auto", "allocation_mode": "managed-default"}
                        ],
                        "output_options": {},
                    }
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(stub_env)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--config",
                    str(config_path),
                    "--machine-label",
                    "gb10",
                    "--output",
                    str(output_path),
                    "--dataset",
                    "generated8193",
                    "--variant",
                    "gpu-poc-device-serial",
                    "--gpu-poc-exe",
                    str(stub_path),
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=env,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_count"], 1)
            self.assertEqual(payload["runs"][0]["status"], "executed")
            self.assertEqual(payload["runs"][0]["returncode"], 0)
            stub_log = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertIn("--solve-mode", stub_log["argv"])
            self.assertIn("device-serial", stub_log["argv"])
            self.assertEqual(payload["runs"][0]["executed_command"][1:], stub_log["argv"])

    def test_non_dry_run_with_planned_runs_fails_without_writing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [{"name": "hologram31016", "kind": "repo", "path": "keys/HologramWorld-31016.keys"}],
                        "variants": [{"name": "cuda-chm02", "solver_family": "cuda-chm02", "algorithm": "Chm02", "hash_function": "Mulshrolate3RX", "mask_function": "And", "fixed_attempts": 128}],
                        "output_options": {},
                    }
                ),
                encoding="utf-8",
            )

            result = self.run_script(
                "--config",
                str(config_path),
                "--machine-label",
                "gb10",
                "--output",
                str(output_path),
            )

            self.assertEqual(result.returncode, 3)
            self.assertIn("explicit dataset and variant filters", result.stderr)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
