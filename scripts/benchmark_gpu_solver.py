#!/usr/bin/env python3

import argparse
import itertools
import json
import sys
from pathlib import Path


REQUIRED_TOP_LEVEL_SECTIONS = ("datasets", "variants")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run GPU solver benchmark matrices. "
            "The initial scaffold supports dry-run planning only."
        )
    )
    parser.add_argument("--config", required=True, help="Path to benchmark JSON config")
    parser.add_argument("--machine-label", required=True, help="Logical machine label, e.g. gb10 or nv1")
    parser.add_argument("--output", required=True, help="Path to write JSON results/plan")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit the planned benchmark runs without executing them",
    )
    return parser.parse_args()


def load_config(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Config file does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {path}: {exc}") from None

    missing = [name for name in REQUIRED_TOP_LEVEL_SECTIONS if name not in payload]
    if missing:
        raise ValueError(
            "Config file is missing required section(s): " + ", ".join(missing)
        )

    if not isinstance(payload["datasets"], list):
        raise ValueError("Config section 'datasets' must be a list")
    if not isinstance(payload["variants"], list):
        raise ValueError("Config section 'variants' must be a list")

    return payload


def build_run_plan(config: dict, machine_label: str):
    runs = []
    for dataset, variant in itertools.product(config["datasets"], config["variants"]):
        runs.append(
            {
                "machine_label": machine_label,
                "dataset": dataset,
                "variant": variant,
            }
        )
    return runs


def main():
    args = parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output)

    try:
        config = load_config(config_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    runs = build_run_plan(config, args.machine_label)

    payload = {
        "machine_label": args.machine_label,
        "config_path": str(config_path),
        "dry_run": bool(args.dry_run),
        "run_count": len(runs),
        "runs": runs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dry_run or not runs:
        return 0

    print(
        "Benchmark execution is not implemented yet; rerun with --dry-run or an empty matrix.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
