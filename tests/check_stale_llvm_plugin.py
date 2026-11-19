"""Run a client whose PerfectHashLLVM plugin lacks the hash JIT entry points.

The loader must refuse the plugin, say why once however many compiles follow,
and report every LLVM compile as a missing backend.
"""
import subprocess
import sys

PH_E_LLVM_BACKEND_NOT_FOUND = "0xe004041c"
REPORT = ("LLVM JIT support is unavailable: the PerfectHashLLVM library lacks "
          "these entry points: PerfectHashTableCompileHashJit, "
          "PerfectHashTableHashJitRundown. It was built from a different "
          "PerfectHash version.")

result = subprocess.run(sys.argv[1:], capture_output=True, text=True, timeout=60)
if result.returncode:
    raise AssertionError(
        f"{sys.argv[1]} returned {result.returncode}: {result.stdout}{result.stderr}"
    )
if result.stdout.split() != [PH_E_LLVM_BACKEND_NOT_FOUND] * 2:
    raise AssertionError(f"LLVM compiles returned {result.stdout!r}")
if result.stderr.splitlines() != [REPORT]:
    raise AssertionError(f"Expected only {REPORT!r} on stderr, got {result.stderr!r}")
