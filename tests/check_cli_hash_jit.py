"""Solve with --TryUseJitHashFunction and require that the hash JIT was used.

Every solver graph hashes with the table's JIT-compiled routines, each
supplying its own seeds. HologramWorld needs thousands of attempts with
MultiplyShiftR, so its eight workers hash concurrently for the whole solve.
"""
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WIDTHS = {16: (), 32: ("--DoNotTryUseHash16Impl",)}


def solve(exe, keys, output, hash_name, concurrency, *flags):
    output.mkdir()
    args = [exe, keys, output, "Chm01", hash_name, "And", str(concurrency),
            "--HashAllKeysFirst", "--NoFileIo", "--Quiet", *flags]
    result = subprocess.run(args, timeout=600)
    if result.returncode:
        raise AssertionError(f"{args} returned {result.returncode:#x}")
    report, = output.glob("*.csv")
    with report.open(newline="") as stream:
        row, = csv.DictReader(stream)
    if row["TableCreateResult"] != "S_OK":
        raise AssertionError(f"{args} reported {row['TableCreateResult']}")
    return row


def solve_with_jit(exe, keys, output, hash_name, concurrency, width):
    row = solve(exe, keys, output, hash_name, concurrency,
                "--TryUseJitHashFunction", *WIDTHS[width])
    name = f"{hash_name}, {concurrency} workers, {width}-bit vertices"
    if row["UsedJitHashFunction"] != "Y":
        raise AssertionError(f"{name}: the winning graph did not use the hash JIT")
    if row["UsedAvx2HashFunction"] != "N" or row["UsedAvx512HashFunction"] != "N":
        raise AssertionError(f"{name}: an AVX routine was reported beside the JIT")
    return row


create, source = sys.argv[1:]
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    # The CLI writes a table-size cache beside its input.
    keys = Path(shutil.copy(source, root))
    for width in WIDTHS:
        for hash_name, concurrency in (("MultiplyShiftR", 8),
                                       ("Mulshrolate3RX", 4),
                                       ("Mulshrolate4RX", 4)):
            solve_with_jit(create, keys, root / f"{hash_name}-{width}",
                           hash_name, concurrency, width)

        # One worker draws the same seeds either way, so equal attempt counts
        # show that the JIT hashed every attempt exactly as the scalar code did.
        jit = solve_with_jit(create, keys, root / f"single-jit-{width}",
                             "Mulshrolate3RX", 1, width)
        scalar = solve(create, keys, root / f"single-scalar-{width}",
                       "Mulshrolate3RX", 1, *WIDTHS[width])
        if scalar["UsedJitHashFunction"] != "N":
            raise AssertionError("The hash JIT was used without being requested")
        if jit["Attempts"] != scalar["Attempts"]:
            raise AssertionError(
                f"{width}-bit vertices: the JIT solved after {jit['Attempts']} "
                f"attempts, the scalar routine after {scalar['Attempts']}"
            )
