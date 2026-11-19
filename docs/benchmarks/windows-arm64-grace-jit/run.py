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
MULTIPLY_SHIFT = ("MultiplyShiftR", "MultiplyShiftRX")
MULSHROLATE = tuple(f"Mulshrolate{i}RX" for i in range(1, 5))
SEEDS = {
    "MultiplyShiftR": (0x41AD0EDF, 0x5F8D672D, 0xD09),
    "MultiplyShiftRX": (0xC2E3C0B7, 0xE8D9CDF9, 0x1B10),
    # The CLI treats the shipped seed's zero rotation byte as unspecified.
    # Halving its even multiplier and rotating left once (ROR 31) preserves
    # the upper 16 product bits and exercises a nonzero rotation.
    "Mulshrolate1RX": (0x39D252C1, 0x6A37DC9E, 0x0C041F10),
    "Mulshrolate2RX": (0x36B166E3, 0x059700B5, 0x1C080110),
    "Mulshrolate3RX": (0xF0192B55, 0xD9C83970, 0x0C1E0D10, 0xD11A5847),
    "Mulshrolate4RX": (0xED1BAA28, 0x5570F5A6, 0x0E050810, 0x76EA4FF2, 0xBA628C02),
}


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
    seeds = SEEDS[name]
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
            row["HashFunction"] != name or int(row["NumberOfKeys"]) != 31016 or
            int(row["NumberOfSeeds"]) != len(seeds) or
            tuple(int(row[f"Seed{i + 1}"]) for i in range(len(seeds))) != seeds or
            int(row["NumberOfVertices"]) != 65536 or
            ticks <= 0 or int(row["SolveMicroseconds"]) <= 0 or
            usec != ticks * 1_000_000 // frequency):
        raise RuntimeError(f"Unexpected solve or counters in {csv_path}: {row}")
    return dict(hash=name, bits=bits, neon=neon, hash_qpc_ticks=ticks,
                hash_us=usec, solve_us=int(row["SolveMicroseconds"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--affinity", type=lambda x: int(x, 0), default=4)
    parser.add_argument("--mulshrolate-only", action="store_true",
                        help="run only the four Mulshrolate solver comparisons")
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
    if not args.mulshrolate_only:
        for repeat in range(1, 4):
            rng.shuffle(lookup_cases)
            for case in lookup_cases:
                row = measure_lookup(build / "bin/Release/perfecthash_benchmarks.exe", *case)
                output["lookup"].append(dict(repeat=repeat, **row))
                print(repeat, row, flush=True)
    hashes = MULSHROLATE if args.mulshrolate_only else MULTIPLY_SHIFT
    for repeat in range(1, 46):
        for name in hashes:
            for bits in (16, 32):
                for neon in (bool(repeat % 2), not bool(repeat % 2)):
                    row = measure_solve(build / "bin/Release/PerfectHashCreate.exe",
                                        bits, name, neon, freq.value)
                    output["solve"].append(dict(repeat=repeat, **row))
    filename = "mulshrolate-solver-results" if args.mulshrolate_only else "grace-jit-results"
    dest = build / f"{filename}.json"
    with dest.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(dest)


if __name__ == "__main__":
    main()
