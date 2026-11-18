"""Verify a retained Chm02 winner after later attempts reuse the other GPU graph."""

import argparse
import csv
import os
import subprocess
import tempfile
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("exe")
parser.add_argument("source", type=Path)
parser.add_argument("width", type=int, choices=(16, 32), nargs="?", default=32)
parser.add_argument("--mixed", action="store_true", help="run one CPU and one CUDA worker")
args = parser.parse_args()
width_flags = ["--DoNotTryUseHash16Impl"] if args.width == 32 else []
# The CPU can consume the short CUDA-only budget while the GPU initializes.
attempts = 8192 if args.mixed else 128
workers = "2" if args.mixed else "1"
with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp)
    keys = output / "Hologram8193.keys"
    with args.source.open("rb") as fixture:
        keys.write_bytes(fixture.read(8193 * 4))

    # Philox selects new seeds each attempt. This prefix produces several
    # successive winners and the best graph predates the final attempt.
    result = subprocess.run(
        [args.exe, keys, output, "Chm02", "Mulshrolate3RX", "And", workers,
         "--CuConcurrency=1", "--FindBestGraph",
         "--BestCoverageType=HighestNumberOfEmptyCacheLines",
         f"--FixedAttempts={attempts}", *width_flags,
         "--Rng=Philox43210", "--RngSeed=12345", "--NoFileIo"],
        env={**os.environ, "PH_DEBUG_CUDA_CHM02": "1"},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120,
    )
    # In a mixed run the winning graph may be verified by the CPU.
    if result.returncode or (not args.mixed and b"PH_CHM02_CUDA_VERIFY_OK" not in result.stderr):
        raise AssertionError(
            f"Retained graph verification failed (exit {result.returncode:#x})\n"
            + result.stderr[-3000:].decode(errors="replace")
        )

    report, = output.glob("*.csv")
    with report.open(newline="") as stream:
        row, = csv.DictReader(stream)
    if row["TableCreateResult"] != "S_OK":
        raise AssertionError(f"Table creation returned {row['TableCreateResult']}")
    if row["UsedAssigned16"] != ("Y" if args.width == 16 else "N"):
        raise AssertionError(f"Expected {args.width}-bit data, got UsedAssigned16={row['UsedAssigned16']}")
    # The shared attempt counter can include the final reset that observes
    # the fixed limit. It is also advanced by the CPU oracle's reset.
    reported = int(row["Attempts"])
    if int(row["FixedAttempts"]) != attempts or reported < attempts:
        raise AssertionError(f"Expected to reach the {attempts}-attempt limit, got {reported}")
    winners = int(row["NewBestGraphCount"])
    if winners < 2:
        raise AssertionError(f"Expected multiple best graphs, got {winners}")
    winning_attempt = int(row[f"BestGraph{winners}_Attempt"])
    if int(row["AttemptThatFoundBestGraph"]) != winning_attempt:
        raise AssertionError(f"Expected the summary and winner to agree on attempt {winning_attempt}")
    if not 0 < winning_attempt < attempts - 1:
        raise AssertionError(f"Expected an earlier retained winner, got attempt {winning_attempt}")
    if args.mixed:
        gpu_solutions = int(row["GpuAndCpuIsAcyclicSuccess"])
        solutions = int(row["NumberOfSolutionsFound"])
        if gpu_solutions and b"PH_CHM02_CUDA_ASSIGN_OK" not in result.stderr:
            raise AssertionError("GPU solutions reported without a successful CUDA assignment")
        if not 0 <= gpu_solutions <= solutions:
            raise AssertionError(f"Invalid GPU solution count {gpu_solutions} of {solutions}")
        if gpu_solutions in (0, solutions):
            print(f"Only one worker solved before the shared limit ({gpu_solutions}/{solutions} GPU)")
            raise SystemExit(77)
