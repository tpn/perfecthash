"""Run a client whose PerfectHashLLVM plugin lacks required entry points.

The loader must refuse the plugin, say why once however many compiles follow,
and report every LLVM compile as a missing backend.
"""
import os
from pathlib import Path
import subprocess
import sys

PH_E_LLVM_BACKEND_NOT_FOUND = "0xe004041c"
missing = ["PerfectHashTableCompileHashJitV1", "PerfectHashTableHashJitRundownV1"]
if sys.argv[2] == "unversioned":
    missing[:0] = ["PerfectHashTableCompileJitV1", "PerfectHashTableJitRundownV1"]
report = ("LLVM JIT support is unavailable: the PerfectHashLLVM library lacks "
          f"these entry points: {', '.join(missing)}. "
          "It was built from a different PerfectHash version.")

env = os.environ.copy()
if os.name != "nt":
    variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    directory = str(Path(sys.argv[1]).resolve().parent)
    env[variable] = os.pathsep.join(filter(None, [directory, env.get(variable)]))
result = subprocess.run([sys.argv[1]], capture_output=True, text=True, env=env, timeout=60)
if result.returncode:
    raise AssertionError(
        f"{sys.argv[1]} returned {result.returncode}: {result.stdout}{result.stderr}"
    )
if result.stdout.split() != [PH_E_LLVM_BACKEND_NOT_FOUND] * 2:
    raise AssertionError(f"LLVM compiles returned {result.stdout!r}")
if result.stderr.splitlines() != [report]:
    raise AssertionError(f"Expected only {report!r} on stderr, got {result.stderr!r}")
