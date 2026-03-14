#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE_DISTRIBUTION_NAME = "tpn-perfecthash"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def build_env(*, version: str, native_root: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PERFECTHASH_PYTHON_VERSION"] = version
    if native_root is None:
        env.pop("PERFECTHASH_PYTHON_NATIVE_ROOT", None)
    else:
        env["PERFECTHASH_PYTHON_NATIVE_ROOT"] = str(native_root)
    return env


def artifact_prefixes(version: str) -> tuple[str, ...]:
    normalized_name = re.sub(r"[-_.]+", "-", PACKAGE_DISTRIBUTION_NAME).lower()
    wheel_name = normalized_name.replace("-", "_")
    return (
        f"{normalized_name}-{version}",
        f"{wheel_name}-{version}",
    )


def find_built_artifact(dist_dir: Path, suffix: str, version: str) -> Path:
    prefixes = artifact_prefixes(version)
    matches = sorted(
        path
        for path in dist_dir.iterdir()
        if path.name.startswith(prefixes) and path.name.endswith(suffix)
    )
    if not matches:
        raise FileNotFoundError(
            f"Unable to find built artifact with suffix {suffix!r} in {dist_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Expected exactly one artifact matching *{suffix}, found: {matches}")
    return matches[0]


def smoke_test_wheel(*, wheel: Path, version: str, root_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="perfecthash-wheel-smoke-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        venv_dir = temp_dir / "venv"
        python_exe = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        ph_exe = venv_dir / ("Scripts/ph.exe" if sys.platform == "win32" else "bin/ph")

        run([sys.executable, "-m", "venv", str(venv_dir)])
        run([str(python_exe), "-m", "pip", "install", str(wheel)])

        smoke_env = os.environ.copy()
        smoke_env.pop("PYTHONPATH", None)

        run(
            [
                str(python_exe),
                "-c",
                (
                    "from perfecthash import __version__;"
                    f" assert __version__ == {version!r}, __version__"
                ),
            ],
            env=smoke_env,
        )
        run([str(ph_exe), "--version"], env=smoke_env)
        run(
            [
                str(ph_exe),
                "create",
                str(root_dir / "keys" / "HologramWorld-31016.keys"),
                "out",
                "--hash-function",
                "MultiplyShiftR",
                "--dry-run",
            ],
            env=smoke_env,
        )
        run(
            [
                str(python_exe),
                "-c",
                "\n".join(
                    [
                        "from perfecthash import build_table",
                        "keys = [1, 3, 5, 7, 11, 13, 17, 19]",
                        "with build_table(keys, hash_function='MultiplyShiftR') as table:",
                        "    assert len(table.index_many(keys)) == len(keys)",
                    ]
                ),
            ],
            env=smoke_env,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist-dir", required=True, type=Path)
    parser.add_argument("--native-root", type=Path)
    parser.add_argument(
        "--build-sdist",
        action="store_true",
        help="Also build a source distribution.",
    )
    parser.add_argument(
        "--smoke-test-wheel",
        action="store_true",
        help="Install the built wheel in a clean venv and smoke-test it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    version = args.version.strip().removeprefix("v")
    if not version:
        raise SystemExit("error: --version must not be empty")
    dist_dir = args.dist_dir.resolve()
    dist_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(version=version, native_root=args.native_root)

    run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(dist_dir),
        ],
        env=env,
    )

    if args.build_sdist:
        run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--outdir",
                str(dist_dir),
            ],
            env=env,
        )

    wheel = find_built_artifact(dist_dir, ".whl", version)
    print(f"Built wheel: {wheel}")

    if args.smoke_test_wheel:
        smoke_test_wheel(wheel=wheel, version=version, root_dir=root_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
