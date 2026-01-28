#!/usr/bin/env python3
"""
Generate LLVM-MCA-ready assembly inputs and run llvm-mca on selected functions.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
JIT_DUMPS = REPO_ROOT / "build" / "jit-dumps"

PRESETS = {
    "msr-avx-x4": {
        "input": JIT_DUMPS / "msr_avx_x4.intel.s",
        "function": "PerfectHashJitIndex32x4Vector",
        "mattr": "+sse4.1",
        "mcpu": "haswell",
    },
    "msr-avx2-x8": {
        "input": JIT_DUMPS / "msr_avx2_x8.intel.s",
        "function": "PerfectHashJitIndex32x8Vector",
        "mattr": "+avx2",
        "mcpu": "haswell",
    },
    "msr-avx512-x16": {
        "input": JIT_DUMPS / "msr_avx512_x16.intel.s",
        "function": "PerfectHashJitIndex32x16Vector",
        "mattr": "+avx512f,+avx512bw",
        "mcpu": "skylake-avx512",
    },
}


def find_llvm_mca(llvm_bin: str | None) -> pathlib.Path:
    if llvm_bin:
        candidate = pathlib.Path(llvm_bin) / "llvm-mca.exe"
        if candidate.exists():
            return candidate
    env_bin = os.environ.get("LLVM_BIN")
    if env_bin:
        candidate = pathlib.Path(env_bin) / "llvm-mca.exe"
        if candidate.exists():
            return candidate
    default = pathlib.Path(r"C:\Program Files\LLVM\bin\llvm-mca.exe")
    if default.exists():
        return default
    raise FileNotFoundError("llvm-mca.exe not found; pass --llvm-bin or set LLVM_BIN.")


def insert_mca_markers(lines: list[str], function: str) -> list[str]:
    label = f"{function}:"
    label_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(label):
            label_idx = i
            break
    if label_idx is None:
        raise ValueError(f"Function label not found: {function}")

    end_idx = None
    for i in range(label_idx + 1, len(lines)):
        if "# -- End function" in lines[i]:
            end_idx = i
            break
    if end_idx is None:
        raise ValueError(f"End marker not found for function: {function}")

    out = list(lines)
    begin = "\t# LLVM-MCA-BEGIN\n"
    end = "\t# LLVM-MCA-END\n"

    out.insert(label_idx + 1, begin)
    end_idx += 1
    out.insert(end_idx, end)
    return out


def generate_mca_input(input_path: pathlib.Path, function: str, output_path: pathlib.Path) -> None:
    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    out_lines = insert_mca_markers(lines, function)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(out_lines), encoding="utf-8")


def run_llvm_mca(
    llvm_mca: pathlib.Path,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    mtriple: str | None,
    mattr: str | None,
    mcpu: str | None,
    iterations: int,
    extra_args: list[str],
) -> None:
    cmd = [str(llvm_mca)]
    if mtriple:
        cmd.append(f"--mtriple={mtriple}")
    if mattr:
        cmd.append(f"--mattr={mattr}")
    if mcpu:
        cmd.append(f"--mcpu={mcpu}")
    cmd.append(f"--iterations={iterations}")
    cmd.extend(extra_args)
    cmd.append(str(input_path))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"llvm-mca failed (exit {result.returncode}).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run llvm-mca on selected JIT functions.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()))
    parser.add_argument("--input", type=pathlib.Path, help="Intel-syntax assembly input")
    parser.add_argument("--function", help="Function label to analyze")
    parser.add_argument("--llvm-bin", help="Path to LLVM bin directory")
    parser.add_argument("--mtriple", default="x86_64-pc-windows-msvc")
    parser.add_argument("--mattr", help="Target features, e.g. +avx2")
    parser.add_argument("--mcpu", help="Target CPU, e.g. haswell or skylake-avx512")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=pathlib.Path, default=JIT_DUMPS / "mca")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra llvm-mca args (repeatable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.preset:
        preset = PRESETS[args.preset]
        input_path = preset["input"]
        function = preset["function"]
        mattr = args.mattr or preset.get("mattr")
        mcpu = args.mcpu or preset.get("mcpu")
    else:
        if not args.input or not args.function:
            raise ValueError("Provide --preset or both --input and --function.")
        input_path = args.input
        function = args.function
        mattr = args.mattr
        mcpu = args.mcpu

    llvm_mca = find_llvm_mca(args.llvm_bin)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    stem = f"{function}.mca"
    mca_input = args.output_dir / f"{stem}.s"
    mca_output = args.output_dir / f"{stem}.txt"

    generate_mca_input(input_path, function, mca_input)
    run_llvm_mca(
        llvm_mca=llvm_mca,
        input_path=mca_input,
        output_path=mca_output,
        mtriple=args.mtriple,
        mattr=mattr,
        mcpu=mcpu,
        iterations=args.iterations,
        extra_args=args.extra,
    )

    print(f"MCA input: {mca_input}")
    print(f"MCA output: {mca_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
