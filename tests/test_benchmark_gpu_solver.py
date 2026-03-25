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


if __name__ == "__main__":
    unittest.main()
