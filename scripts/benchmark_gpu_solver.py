#!/usr/bin/env python3

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_TOP_LEVEL_SECTIONS = ("datasets", "variants", "output_options")
SUPPORTED_DATASET_KINDS = {"repo", "generated"}
SUPPORTED_SOLVER_FAMILIES = {"cpu-cli", "cuda-chm02", "gpu-poc"}
DEFAULT_SAFE_FIXED_ATTEMPTS = 2
DEFAULT_SAFE_GPU_POC_BATCH = 128
DEFAULT_SAFE_GPU_POC_THREADS = 128
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run GPU solver benchmark matrices. "
            "Execution is intentionally narrow and requires explicit filters."
        )
    )
    parser.add_argument("--config", required=True, help="Path to benchmark JSON config")
    parser.add_argument("--machine-label", required=True, help="Logical machine label, e.g. gb10 or nv1")
    parser.add_argument("--output", required=True, help="Path to write JSON results/plan")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Exact dataset name to include; repeat to select multiple datasets",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Exact variant name to include; repeat to select multiple variants",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of planned runs in dry-run mode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit the planned benchmark runs without executing them",
    )
    parser.add_argument(
        "--perfect-hash-create-exe",
        default=os.environ.get("PERFECTHASH_CREATE_EXE"),
        help=(
            "Path to PerfectHashCreate for cpu-cli and cuda-chm02 execution. "
            "Defaults to the PERFECTHASH_CREATE_EXE environment variable if set."
        ),
    )
    parser.add_argument(
        "--gpu-poc-exe",
        default=os.environ.get("GPU_BATTED_PEELING_POC_EXE"),
        help=(
            "Path to gpu_batched_peeling_poc for gpu-poc execution. "
            "Defaults to the GPU_BATTED_PEELING_POC_EXE environment variable if set."
        ),
    )
    return parser.parse_args()


def load_config(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Config file does not exist: {path}") from None
    except OSError as exc:
        raise ValueError(f"Unable to read config file {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {path}: {exc}") from None

    if not isinstance(payload, dict):
        raise ValueError("Config file top-level value must be an object/mapping")

    missing = [name for name in REQUIRED_TOP_LEVEL_SECTIONS if name not in payload]
    if missing:
        raise ValueError(
            "Config file is missing required section(s): " + ", ".join(missing)
        )

    if not isinstance(payload["datasets"], list):
        raise ValueError("Config section 'datasets' must be a list")
    if not isinstance(payload["variants"], list):
        raise ValueError("Config section 'variants' must be a list")
    if not isinstance(payload["output_options"], dict):
        raise ValueError("Config section 'output_options' must be an object/mapping")

    _validate_named_mappings(payload["datasets"], "dataset")
    _validate_named_mappings(payload["variants"], "variant")

    return payload


def _validate_named_mappings(items, item_kind):
    if not isinstance(items, list):
        raise ValueError(f"Config section '{item_kind}s' must be a list")

    seen_names = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Config {item_kind} entry at index {index} must be an object/mapping")

        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Config {item_kind} entry at index {index} must have a non-empty string 'name'")
        if name in seen_names:
            raise ValueError(f"Config {item_kind} names must be unique: {name}")
        seen_names.add(name)

        if item_kind == "dataset":
            kind = item.get("kind")
            if kind not in SUPPORTED_DATASET_KINDS:
                raise ValueError(
                    f"Dataset '{name}' has unsupported kind {kind!r}; "
                    f"supported kinds are: {', '.join(sorted(SUPPORTED_DATASET_KINDS))}"
                )
            if kind == "repo":
                path = item.get("path")
                if not isinstance(path, str) or not path:
                    raise ValueError(f"Repo dataset '{name}' must define a non-empty string 'path'")
            elif kind == "generated":
                count = item.get("count")
                salt = item.get("salt")
                if not isinstance(count, int) or count <= 0:
                    raise ValueError(f"Generated dataset '{name}' must define a positive integer 'count'")
                if not isinstance(salt, int) or salt < 0:
                    raise ValueError(f"Generated dataset '{name}' must define a non-negative integer 'salt'")
        else:
            solver_family = item.get("solver_family")
            if solver_family not in SUPPORTED_SOLVER_FAMILIES:
                raise ValueError(
                    f"Variant '{name}' has unsupported solver_family {solver_family!r}; "
                    f"supported families are: {', '.join(sorted(SUPPORTED_SOLVER_FAMILIES))}"
                )
            if solver_family in {"cpu-cli", "cuda-chm02"}:
                for field in ("algorithm", "hash_function", "mask_function", "fixed_attempts"):
                    if field not in item:
                        raise ValueError(f"Variant '{name}' must define '{field}'")
                if not isinstance(item["fixed_attempts"], int) or item["fixed_attempts"] <= 0:
                    raise ValueError(f"Variant '{name}' must define a positive integer 'fixed_attempts'")
            elif solver_family == "gpu-poc":
                for field in ("solve_mode", "hash_function", "batch", "threads", "storage_bits", "allocation_mode"):
                    if field not in item:
                        raise ValueError(f"Variant '{name}' must define '{field}'")
                if not isinstance(item["batch"], int) or item["batch"] <= 0:
                    raise ValueError(f"Variant '{name}' must define a positive integer 'batch'")
                if not isinstance(item["threads"], int) or item["threads"] <= 0:
                    raise ValueError(f"Variant '{name}' must define a positive integer 'threads'")


def select_named_items(items, selected_names, item_kind):
    if not selected_names:
        return list(items)

    by_name = {item["name"]: item for item in items}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise ValueError(
            f"Unknown {item_kind} name(s): " + ", ".join(sorted(set(missing)))
        )
    return [by_name[name] for name in selected_names]


def build_run_plan(config: dict, machine_label: str, dataset_filters=None, variant_filters=None):
    dataset_filters = dataset_filters or []
    variant_filters = variant_filters or []
    selected_datasets = select_named_items(config["datasets"], dataset_filters, "dataset")
    selected_variants = select_named_items(config["variants"], variant_filters, "variant")
    runs = []
    for dataset, variant in itertools.product(selected_datasets, selected_variants):
        runs.append(
            {
                "machine_label": machine_label,
                "dataset": dataset,
                "variant": variant,
            }
        )
    return runs


def cap_runs_for_dry_run(runs, max_runs):
    if max_runs is None:
        return runs
    if max_runs <= 0:
        raise ValueError("--max-runs must be a positive integer")
    return runs[:max_runs]


def resolve_repo_path(relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def resolve_executable(explicit_path, env_var_name, fallbacks):
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate
        raise ValueError(f"Executable does not exist: {candidate}")

    env_value = os.environ.get(env_var_name)
    if env_value:
        candidate = Path(env_value)
        if candidate.exists():
            return candidate
        raise ValueError(f"Executable from ${env_var_name} does not exist: {candidate}")

    for fallback in fallbacks:
        candidate = Path(fallback)
        if candidate.exists():
            return candidate

    raise ValueError(
        f"Unable to locate executable for {env_var_name}; provide --{env_var_name.lower().replace('_', '-')} or set ${env_var_name}"
    )


def build_perfect_hash_command(executable: Path, dataset: dict, variant: dict, run_output_dir: Path):
    dataset_path = resolve_repo_path(dataset["path"])
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    fixed_attempts = min(int(variant.get("fixed_attempts", DEFAULT_SAFE_FIXED_ATTEMPTS)), DEFAULT_SAFE_FIXED_ATTEMPTS)
    if fixed_attempts <= 0:
        fixed_attempts = DEFAULT_SAFE_FIXED_ATTEMPTS

    command = [
        str(executable),
        str(dataset_path),
        str(run_output_dir),
        variant["algorithm"],
        variant["hash_function"],
        variant["mask_function"],
        "1",
    ]

    if variant["solver_family"] == "cuda-chm02":
        command.extend(["--CuConcurrency=1", "--DisableCsvOutputFile"])

    command.extend(
        [
            f"--FixedAttempts={fixed_attempts}",
            "--NoFileIo",
            "--SkipTestAfterCreate",
        ]
    )

    return command


def build_gpu_poc_command(executable: Path, dataset: dict, variant: dict):
    command = [
        str(executable),
    ]

    if dataset["kind"] == "repo":
        dataset_path = resolve_repo_path(dataset["path"])
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        command.extend(["--keys-file", str(dataset_path)])
    elif dataset["kind"] == "generated":
        command.extend(["--edges", str(dataset["count"]), "--key-seed", str(dataset["salt"])])
    else:
        raise ValueError(f"Unsupported dataset kind for gpu-poc: {dataset['kind']}")

    batch = min(int(variant["batch"]), DEFAULT_SAFE_GPU_POC_BATCH)
    threads = min(int(variant["threads"]), DEFAULT_SAFE_GPU_POC_THREADS)

    command.extend(
        [
            "--batch",
            str(batch),
            "--threads",
            str(threads),
            "--solve-mode",
            variant["solve_mode"],
            "--storage-bits",
            str(variant["storage_bits"]),
            "--hash-function",
            variant["hash_function"],
            "--allocation-mode",
            variant["allocation_mode"],
            "--output-format",
            "json",
        ]
    )
    return command


def build_execution_command(args, run, run_output_dir):
    variant = run["variant"]
    dataset = run["dataset"]

    if variant["solver_family"] in {"cpu-cli", "cuda-chm02"}:
        executable = resolve_executable(
            args.perfect_hash_create_exe,
            "PERFECTHASH_CREATE_EXE",
            (
                REPO_ROOT / "build-cuda" / "bin" / "PerfectHashCreate",
                REPO_ROOT / "build" / "bin" / "PerfectHashCreate",
            ),
        )
        return build_perfect_hash_command(executable, dataset, variant, run_output_dir)

    if variant["solver_family"] == "gpu-poc":
        executable = resolve_executable(
            args.gpu_poc_exe,
            "GPU_BATCHED_PEELING_POC_EXE",
            (
                REPO_ROOT / "build" / "gpu-batched-peeling-poc" / "gpu_batched_peeling_poc",
            ),
        )
        return build_gpu_poc_command(executable, dataset, variant)

    raise ValueError(f"Unsupported solver family: {variant['solver_family']}")


def execute_runs(args, runs):
    results = []
    for index, run in enumerate(runs, start=1):
        run_output_dir = Path(args.output).with_suffix("")
        run_output_dir = run_output_dir.parent / f"{run_output_dir.name}.{index:02d}-{run['dataset']['name']}-{run['variant']['name']}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        command = build_execution_command(args, run, run_output_dir)

        env = os.environ.copy()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
        )
        results.append(
            {
                **run,
                "status": "executed",
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "execution_output_dir": str(run_output_dir),
            }
        )
        if completed.returncode != 0:
            return results, completed.returncode

    return results, 0


def main():
    args = parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output)

    try:
        config = load_config(config_path)
        runs = build_run_plan(
            config,
            args.machine_label,
            dataset_filters=args.dataset,
            variant_filters=args.variant,
        )
        if args.dry_run:
            runs = cap_runs_for_dry_run(runs, args.max_runs)
        elif runs and (not args.dataset or not args.variant):
            print(
                "Execution requires explicit dataset and variant filters so the full matrix cannot run accidentally.",
                file=sys.stderr,
            )
            return 3

        if not args.dry_run and args.max_runs is not None and args.max_runs <= 0:
            print("--max-runs must be a positive integer", file=sys.stderr)
            return 2

        payload = {
            "machine_label": args.machine_label,
            "config_path": str(config_path),
            "dry_run": bool(args.dry_run),
            "dataset_filters": args.dataset,
            "variant_filters": args.variant,
            "max_runs": args.max_runs,
            "output_options": config["output_options"],
            "run_count": len(runs),
            "runs": runs,
        }

        if args.dry_run or not runs:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return 0

        if len(runs) != 1:
            print(
                "Execution is limited to exactly one filtered run at a time; "
                "use --dataset and --variant to narrow the matrix to one selected run.",
                file=sys.stderr,
            )
            return 3

        executed_runs, returncode = execute_runs(args, runs)
        payload["runs"] = executed_runs
        payload["run_count"] = len(executed_runs)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return returncode
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
