"""Check Linux CUDA driver lookup independently of the host's installed driver."""

import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile


exe, source_keys, library_without_cuda_symbols = map(Path, sys.argv[1:])
names = ("libcuda.so.1", "libcuda.so")
with tempfile.TemporaryDirectory() as tmp:
    keys = Path(tmp) / "Test.keys"
    shutil.copyfile(source_keys, keys)
    for usable in ((names[0],), (names[1],), (), names):
        directory = Path(tmp) / ("-".join(usable) or "neither")
        directory.mkdir()
        for name in names:
            path = directory / name
            if name in usable:
                path.symlink_to(library_without_cuda_symbols.resolve())
            else:
                # A bad ELF header prevents dlopen from using a system copy.
                path.write_bytes(b"invalid")

        search = [str(directory)]
        if os.environ.get("LD_LIBRARY_PATH"):
            search.append(os.environ["LD_LIBRARY_PATH"])
        env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(search)}
        trace_lookup = usable == names and platform.libc_ver()[0] == "glibc"
        if trace_lookup:
            env["LD_DEBUG"] = "libs"
        result = subprocess.run(
            [exe, keys, directory, "Chm02", "Mulshrolate3RX", "And", "1",
             "--NoFileIo", "--DisableCsvOutputFile", "--Quiet"],
            env=env,
            capture_output=True, text=True, timeout=30,
        )
        # A loadable fixture must reach symbol resolution. With neither
        # name usable, retain the driver-load error and both diagnostics.
        expected = 0xE0040397 if usable else 0xE0040396
        if result.returncode != (expected & 0xff) or f"0x{expected:08x}" not in result.stderr:
            raise AssertionError(f"{usable=}, exit={result.returncode}: {result.stderr}")
        reported = {name for name in names if f"{directory / name}:" in result.stderr}
        if reported != (set() if usable else set(names)):
            raise AssertionError(f"{usable=}, unexpected loader diagnostics: {result.stderr}")
        if trace_lookup:
            lookups = re.findall(r"find library=(libcuda\.so(?:\.1)?) ", result.stderr)
            if lookups != [names[0]]:
                raise AssertionError(f"Expected only the versioned driver lookup, got {lookups}")
