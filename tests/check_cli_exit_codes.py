import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def check(exe, source, output, expect_success):
    output.mkdir()
    result = subprocess.run(
        [exe, source, output, "Chm01", "MultiplyShiftR", "And", "1",
         "--NoFileIo", "--Quiet"],
        capture_output=True, text=True, timeout=30,
    )
    if (result.returncode == 0) != expect_success:
        raise AssertionError(
            f"{exe} {source} returned {result.returncode}\n"
            f"{result.stdout}\n{result.stderr}"
        )


create, bulk_create = sys.argv[1:]
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    keys = root / "keys"
    keys.mkdir()
    path = keys / "Test.keys"
    check(create, path, root / "missing", False)
    path.write_bytes(struct.pack("<4I", 1, 1, 3, 5))
    check(bulk_create, keys, root / "duplicate", False)
    path.write_bytes(struct.pack("<4I", 1, 3, 5, 7))
    check(create, path, root / "create", True)
    check(bulk_create, keys, root / "bulk", True)
