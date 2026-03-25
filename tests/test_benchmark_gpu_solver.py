import json
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
                            {"name": "cuda-chm02", "solver_family": "cuda-chm02", "allocation_mode": "explicit-device"},
                            {"name": "gpu-poc", "solver_family": "gpu-poc", "allocation_mode": "managed-default"},
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

    def test_non_dry_run_with_planned_runs_fails_without_writing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            output_path = temp_path / "out.json"
            config_path.write_text(
                json.dumps(
                    {
                        "datasets": [{"name": "hologram31016", "kind": "repo", "path": "keys/HologramWorld-31016.keys"}],
                        "variants": [{"name": "cuda-chm02", "solver_family": "cuda-chm02"}],
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
            self.assertIn("not implemented", result.stderr)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
