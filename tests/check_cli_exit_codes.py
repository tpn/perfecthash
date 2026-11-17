import csv
import signal
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PH_E_SYSTEM_CALL_FAILED = 0xE0040224
PH_E_NO_KEYS_FOUND_IN_DIRECTORY = 0xE0040344
PH_E_DUPLICATE_KEYS_DETECTED = 0xE0040211
PH_E_CTRL_C_PRESSED = 0xE0040352
PH_I_SOLVE_TIMEOUT_EXPIRED = 0x60040088


def command(exe, source, output, *flags):
    output.mkdir()
    return [exe, source, output, "Chm01", "MultiplyShiftR", "And", "1",
            "--NoFileIo", "--Quiet", *flags]


def check_status(result, expected_hresult):
    expected = expected_hresult if sys.platform == "win32" else expected_hresult & 0xff
    if result.returncode != expected:
        raise AssertionError(
            f"{result.args} returned {result.returncode:#x}, expected {expected:#x}"
        )


def check(exe, source, output, expected_hresult, *flags):
    result = subprocess.run(command(exe, source, output, *flags), timeout=30)
    check_status(result, expected_hresult)


def check_rows(output, expected):
    report, = output.glob("*.csv")
    with report.open(newline="") as stream:
        rows = [(row["KeysName"], row["TableCreateResult"])
                for row in csv.DictReader(stream)]
    if rows != expected:
        raise AssertionError(f"{report} contained {rows!r}, expected {expected!r}")


create, bulk_create = sys.argv[1:]
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    keys = root / "keys"
    keys.mkdir()
    path = keys / "Test.keys"
    check(bulk_create, root / "no-such-directory", root / "missing-dir", PH_E_SYSTEM_CALL_FAILED)
    check(bulk_create, keys, root / "empty-dir", PH_E_NO_KEYS_FOUND_IN_DIRECTORY)
    check(create, path, root / "missing", PH_E_SYSTEM_CALL_FAILED)
    path.write_bytes(struct.pack("<4I", 1, 1, 3, 5))
    check(bulk_create, keys, root / "duplicate", PH_E_DUPLICATE_KEYS_DETECTED)
    if any((root / "duplicate").glob("*.csv")):
        raise AssertionError("An invalid first input left an incomplete bulk report")
    path.write_bytes(struct.pack("<4I", 1, 3, 5, 7))
    check(create, path, root / "create", 0)
    check(bulk_create, keys, root / "bulk", 0)

    # Equal multipliers and shifts select identical vertices on every attempt.
    bad_seeds = ("--DoNotTryUseHash16Impl", "--Seeds=1,1,0x1010")
    for exe, source, name in ((create, path, "single"), (bulk_create, keys, "batch")):
        fixed = root / f"{name}-fixed"
        check(exe, source, fixed, 0, *bad_seeds, "--FixedAttempts=1")
        check_rows(fixed, [("Test", "PH_I_CREATE_TABLE_ROUTINE_FAILED_TO_FIND_SOLUTION")])
        timed = root / f"{name}-timed"
        check(exe, source, timed, PH_I_SOLVE_TIMEOUT_EXPIRED, *bad_seeds,
              "--MaxSolveTimeInSeconds=1")
        check_rows(timed, [("Test", "PH_I_SOLVE_TIMEOUT_EXPIRED")])
        silent = root / f"{name}-no-report"
        check(exe, source, silent, PH_I_SOLVE_TIMEOUT_EXPIRED, *bad_seeds,
              "--MaxSolveTimeInSeconds=1", "--DisableCsvOutputFile")
        if any(silent.glob("*.csv")):
            raise AssertionError(f"{exe} produced a report with CSV output disabled")

        if sys.platform != "win32":
            # Opening the report confirms the signal handler is installed. With
            # fixed colliding seeds and no timeout the solver waits for SIGINT.
            interrupted = root / f"{name}-interrupt"
            args = command(exe, source, interrupted, *bad_seeds)
            with subprocess.Popen(args) as proc:
                try:
                    deadline = time.monotonic() + 30
                    while not any(interrupted.glob("*.csv")):
                        if proc.poll() is not None or time.monotonic() >= deadline:
                            raise AssertionError(f"{exe} did not open its report")
                        time.sleep(0.01)
                    proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=15)
                    check_status(proc, PH_E_CTRL_C_PRESSED)
                finally:
                    if proc.poll() is None:
                        proc.kill()

    (keys / "Other.keys").touch()
    # Native directory traversal order can differ by filesystem.
    first, second = keys.iterdir()
    first.write_bytes(struct.pack("<4I", 1, 3, 5, 7))
    second.write_bytes(struct.pack("<4I", 1, 1, 3, 5))
    mixed = root / "mixed"
    check(bulk_create, keys, mixed, PH_E_DUPLICATE_KEYS_DETECTED)
    check_rows(mixed, [(first.stem, "S_OK")])
    timed_mixed = root / "mixed-after-timeout"
    check(bulk_create, keys, timed_mixed, PH_E_DUPLICATE_KEYS_DETECTED,
          *bad_seeds, "--MaxSolveTimeInSeconds=1")
    check_rows(timed_mixed, [(first.stem, "PH_I_SOLVE_TIMEOUT_EXPIRED")])
