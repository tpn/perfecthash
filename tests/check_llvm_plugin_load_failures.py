"""Run a static client where PerfectHashLLVM.dll cannot be loaded.

A static Windows client loads the plugin from its own directory. Copy the
client to a directory with no plugin, then to one whose plugin cannot find
LLVM-C.dll. Each must name its cause once and report every LLVM compile as a
missing backend.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PH_E_LLVM_BACKEND_NOT_FOUND = "0xe004041c"
ABSENT = "PerfectHashLLVM.dll was not found."
NO_RUNTIME = ("PerfectHashLLVM.dll could not be loaded (error 126). LLVM-C.dll "
              "must be beside it or in a directory that PATH names by "
              "absolute path.")


def check(directory, files, expected, env=None):
    directory.mkdir()
    client, *_ = [shutil.copy(file, directory) for file in files]
    result = subprocess.run([client], capture_output=True, text=True,
                            timeout=60, env=env)
    if result.returncode:
        raise AssertionError(
            f"{client} returned {result.returncode}: {result.stdout}{result.stderr}"
        )
    if result.stdout.split() != [PH_E_LLVM_BACKEND_NOT_FOUND] * 2:
        raise AssertionError(f"{client}: LLVM compiles returned {result.stdout!r}")
    report = result.stderr.splitlines()
    if (len(report) != 1 or
            not report[0].startswith("LLVM JIT support is unavailable: ") or
            not report[0].endswith(expected)):
        raise AssertionError(
            f"{client}: expected one report ending {expected!r}, got {result.stderr!r}"
        )


client, plugin, plugin_imports_llvm_c = sys.argv[1:]
with tempfile.TemporaryDirectory() as tmp:
    check(Path(tmp) / "absent", [client], ABSENT)
    if plugin_imports_llvm_c == "1":
        # The loader looks for LLVM-C.dll in the absolute directories of PATH.
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join(
            entry for entry in env.get("PATH", "").split(os.pathsep)
            if entry and not (Path(entry.strip('"')) / "LLVM-C.dll").exists()
        )
        check(Path(tmp) / "no-runtime", [client, plugin], NO_RUNTIME, env)
