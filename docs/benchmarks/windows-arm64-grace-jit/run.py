"""Run the Grace lookup and solver benchmarks from a native developer shell."""

import argparse
import csv
import ctypes
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
KEYS = ROOT / "keys/HologramWorld-31016.keys"


def measure_lookup(exe, bits, backend, lanes):
    args = ["--preset=hologramworld-msr", "--fixed-attempts=1",
            "--iterations=5", "--loops=50", "--no-std-map-baselines"]
    if bits == 16:
        args.append("--allow-assigned16")
    if backend == "native":
        args.append("--no-jit")
    else:
        args.append("--jit-backend=" + backend)
        if backend == "llvm":
            args.append("--jit-max-isa=neon")
        if lanes > 1:
            args.append(f"--index32x{lanes}")
            if backend == "llvm" and lanes in (4, 8):
                args.append(f"--jit-vector-index32x{lanes}")

    output = subprocess.check_output([exe, *args], cwd=ROOT, text=True, timeout=120)
    jit = re.search(r"^JIT compile time:\s+([\d.]+) ms", output, re.M)
    lookup = re.search(r"Lookup avg \(PerfectHash\):\s+([\d.]+) ms", output)
    if (not lookup or bool(jit) != (backend != "native") or
            "falling back" in output or "skipped" in output or
            not re.search(rf"Assigned element bits:\s+{bits}\b", output)):
        raise RuntimeError(output)
    if jit:
        compiled = re.search(r"JIT compiled widths:([^\n]*)", output)
        vector = re.search(r"JIT vector widths:([^\n]*)", output)
        if (f"Active JIT backend: {backend}\n" not in output or
                not compiled or str(lanes) not in compiled[1].split() or
                not vector or (str(lanes) in vector[1].split()) !=
                (backend == "llvm" and lanes > 1)):
            raise RuntimeError(output)
    return dict(backend=backend, bits=bits, lanes=lanes, lookup_ms=float(lookup[1]),
                compile_ms=float(jit[1]) if jit else None)


def measure_solve(exe, bits, name, neon, frequency):
    seeds = (0x41AD0EDF, 0x5F8D672D, 0xD09) if name == "MultiplyShiftR" else (
             0xC2E3C0B7, 0xE8D9CDF9, 0x1B10)
    with tempfile.TemporaryDirectory() as directory:
        args = [exe, KEYS, directory, "Chm01", name, "And", "1", "--GraphImpl=3",
                "--HashAllKeysFirst", "--NoFileIo", "--FixedAttempts=1",
                "--Seeds=" + ",".join(map(hex, seeds)), "--Silent"]
        if bits == 32:
            args.append("--DoNotTryUseHash16Impl")
        if not neon:
            args.append("--DoNotTryUseNeonHashFunction")
        subprocess.run(args, cwd=ROOT, check=True, timeout=30)
        csv_path, = Path(directory).glob("PerfectHashTableCreate_*.csv")
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            row, = csv.DictReader(stream)
    ticks = int(row["HashKeysElapsedCycles"])  # This graph counter uses QPC.
    usec = int(row["HashKeysElapsedMicroseconds"])
    if (row["TableCreateResult"] != "S_OK" or row["FirstAttemptSolved"] != "1" or
            row["UsedNeonHashFunction"] != ("Y" if neon else "N") or
            row["UsedAssigned16"] != ("Y" if bits == 16 else "N") or
            tuple(int(row[f"Seed{i}"]) for i in (1, 2, 3)) != seeds or
            int(row["NumberOfVertices"]) != 65536 or
            usec != ticks * 1_000_000 // frequency):
        raise RuntimeError(f"Unexpected solve or counters in {csv_path}: {row}")
    return dict(hash=name, bits=bits, neon=neon, hash_qpc_ticks=ticks,
                hash_us=usec, solve_us=int(row["SolveMicroseconds"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--affinity", type=lambda x: int(x, 0), default=4)
    args = parser.parse_args()
    build = args.build.resolve()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), args.affinity):
        raise ctypes.WinError(ctypes.get_last_error())
    freq = ctypes.c_longlong()
    if not kernel.QueryPerformanceFrequency(ctypes.byref(freq)):
        raise ctypes.WinError(ctypes.get_last_error())
    os.environ["PERFECT_HASH_ONLINE_MAX_CONCURRENCY"] = "1"
    output = dict(affinity=args.affinity, qpc_hz=freq.value, key_count=31016,
                  loops=50, iterations=5, lookup=[], solve=[])
    rng = random.Random(42)
    lookup_cases = [(bits, backend, lanes) for bits in (16, 32)
                    for backend in ("native", "rawdog", "llvm")
                    for lanes in ((1,) if backend == "native" else (1, 4, 8, 16))]
    for repeat in range(1, 4):
        rng.shuffle(lookup_cases)
        for case in lookup_cases:
            row = measure_lookup(build / "bin/Release/perfecthash_benchmarks.exe", *case)
            output["lookup"].append(dict(repeat=repeat, **row))
            print(repeat, row, flush=True)
    for repeat in range(1, 46):
        for name in ("MultiplyShiftR", "MultiplyShiftRX"):
            for bits in (16, 32):
                for neon in (bool(repeat % 2), not bool(repeat % 2)):
                    row = measure_solve(build / "bin/Release/PerfectHashCreate.exe",
                                        bits, name, neon, freq.value)
                    output["solve"].append(dict(repeat=repeat, **row))
    dest = build / "grace-jit-results.json"
    with dest.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(dest)


if __name__ == "__main__":
    main()
