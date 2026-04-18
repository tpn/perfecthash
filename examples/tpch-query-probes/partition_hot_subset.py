#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def read_u64_file(path: Path):
    with path.open("rb") as f:
        data = f.read()
    if len(data) % 8 != 0:
        raise ValueError(f"{path} size is not a multiple of 8")
    return [value[0] for value in struct.iter_unpack("<Q", data)]


def write_u64_file(path: Path, values):
    count = 0
    with path.open("wb") as f:
        for value in values:
            f.write(struct.pack("<Q", int(value)))
            count += 1
    return count


def partition_probe_stream(probe_path: Path, hot_keys: set[int], hot_out: Path, cold_out: Path):
    hot_count = 0
    cold_count = 0
    with probe_path.open("rb") as src, hot_out.open("wb") as hot_f, cold_out.open("wb") as cold_f:
        while True:
            chunk = src.read(8 * 100_000)
            if not chunk:
                break
            for (value,) in struct.iter_unpack("<Q", chunk):
                if value in hot_keys:
                    hot_f.write(struct.pack("<Q", value))
                    hot_count += 1
                else:
                    cold_f.write(struct.pack("<Q", value))
                    cold_count += 1
    return hot_count, cold_count


def summarize_build_key_frequencies(probe_path: Path, build_keys: set[int]) -> dict[int, int]:
    counts = {key: 0 for key in build_keys}
    with probe_path.open("rb") as src:
        while True:
            chunk = src.read(8 * 100_000)
            if not chunk:
                break
            for (value,) in struct.iter_unpack("<Q", chunk):
                if value in counts:
                    counts[value] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    summary_path = candidate_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json in {candidate_dir}")

    with summary_path.open() as f:
        summary = json.load(f)

    def resolve_summary_path(path_text: str) -> Path:
        path = Path(path_text)
        if path.is_absolute():
            return path
        return candidate_dir / path

    build_keys = set(read_u64_file(resolve_summary_path(summary["build_keys"]["path"])))
    probe_count = summary["probe_stream"]["count"]
    probe_path = resolve_summary_path(summary["probe_stream"]["path"])

    for subset_name in ["top1pct_distinct_keys", "top10pct_distinct_keys"]:
        subset_info = summary[subset_name]
        probe_hot_distinct = read_u64_file(resolve_summary_path(subset_info["path"]))
        build_hot_distinct = sorted(set(probe_hot_distinct).intersection(build_keys))

        tag = "top10pct" if subset_name.startswith("top10") else "top1pct"
        build_hot_path = candidate_dir / f"build_hot_{tag}-64.keys"
        hot_probe_path = candidate_dir / f"probe_hot_{tag}-64.bin"
        cold_probe_path = candidate_dir / f"probe_cold_{tag}-64.bin"

        build_hot_count = write_u64_file(build_hot_path, build_hot_distinct)
        hot_probe_count, cold_probe_count = partition_probe_stream(
            probe_path, set(build_hot_distinct), hot_probe_path, cold_probe_path
        )

        derived = {
            "tag": tag,
            "build_hot_keys": {
              "path": str(build_hot_path),
              "count": build_hot_count,
              "bytes": build_hot_count * 8,
            },
            "probe_hot_stream": {
              "path": str(hot_probe_path),
              "count": hot_probe_count,
              "bytes": hot_probe_count * 8,
            },
            "probe_cold_stream": {
              "path": str(cold_probe_path),
              "count": cold_probe_count,
              "bytes": cold_probe_count * 8,
            },
            "hit_mass": (hot_probe_count / probe_count) if probe_count else 0.0,
            "miss_mass": (cold_probe_count / probe_count) if probe_count else 0.0,
            "build_hot_fraction_of_build": (build_hot_count / summary["build_rows"])
              if summary["build_rows"] else 0.0,
        }

        out_path = candidate_dir / f"{tag}_partition_summary.json"
        with out_path.open("w") as f:
            json.dump(derived, f, indent=2, sort_keys=True)

        print(candidate_dir.name, tag, build_hot_count, hot_probe_count, cold_probe_count,
              f"hit_mass={derived['hit_mass']:.4f}",
              f"build_hot_frac={derived['build_hot_fraction_of_build']:.4f}")

    counts = summarize_build_key_frequencies(probe_path, build_keys)
    ranked_build_hits = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    build_count = summary["build_rows"]
    probe_count = summary["probe_stream"]["count"]

    for fraction, tag in [(0.01, "buildhit_top1pct"), (0.10, "buildhit_top10pct")]:
        keep = max(1, math.ceil(build_count * fraction))
        hot_keys = sorted(key for key, _ in ranked_build_hits[:keep] if _ > 0)

        build_hot_path = candidate_dir / f"build_hot_{tag}-64.keys"
        hot_probe_path = candidate_dir / f"probe_hot_{tag}-64.bin"
        cold_probe_path = candidate_dir / f"probe_cold_{tag}-64.bin"

        build_hot_count = write_u64_file(build_hot_path, hot_keys)
        hot_probe_count, cold_probe_count = partition_probe_stream(
            probe_path, set(hot_keys), hot_probe_path, cold_probe_path
        )

        derived = {
            "tag": tag,
            "fraction_of_build": fraction,
            "build_hot_keys": {
                "path": str(build_hot_path),
                "count": build_hot_count,
                "bytes": build_hot_count * 8,
            },
            "probe_hot_stream": {
                "path": str(hot_probe_path),
                "count": hot_probe_count,
                "bytes": hot_probe_count * 8,
            },
            "probe_cold_stream": {
                "path": str(cold_probe_path),
                "count": cold_probe_count,
                "bytes": cold_probe_count * 8,
            },
            "hit_mass": (hot_probe_count / probe_count) if probe_count else 0.0,
            "miss_mass": (cold_probe_count / probe_count) if probe_count else 0.0,
            "build_hot_fraction_of_build": (build_hot_count / build_count) if build_count else 0.0,
            "min_hot_frequency": min((counts[k] for k in hot_keys), default=0),
            "max_hot_frequency": max((counts[k] for k in hot_keys), default=0),
        }

        out_path = candidate_dir / f"{tag}_partition_summary.json"
        with out_path.open("w") as f:
            json.dump(derived, f, indent=2, sort_keys=True)

        print(candidate_dir.name, tag, build_hot_count, hot_probe_count, cold_probe_count,
              f"hit_mass={derived['hit_mass']:.4f}",
              f"build_hot_frac={derived['build_hot_fraction_of_build']:.4f}",
              f"min_hot_freq={derived['min_hot_frequency']}",
              f"max_hot_freq={derived['max_hot_frequency']}")


if __name__ == "__main__":
    main()
