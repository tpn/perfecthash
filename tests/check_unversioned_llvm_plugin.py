"""A module with all legacy symbols must be rejected before any JIT call."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

client, module = map(Path, sys.argv[1:3])
with tempfile.TemporaryDirectory() as directory:
    executable = shutil.copy2(client, directory)
    shutil.copy2(module, Path(directory, sys.argv[3]))
    env = os.environ.copy()
    if os.name != "nt":
        name = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        env[name] = os.pathsep.join(filter(None, [directory, env.get(name)]))
    result = subprocess.run([executable], capture_output=True, text=True, env=env,
                            timeout=60)

if result.returncode or result.stdout.split() != ["0xe004041c"] * 2:
    raise AssertionError(f"Expected two missing-backend results: {result}")
expected = ("LLVM JIT support is unavailable: the PerfectHashLLVM library lacks "
            "these entry points: PerfectHashTableCompileJitV1, PerfectHashTableJitRundownV1, "
            "PerfectHashTableCompileHashJitV1, PerfectHashTableHashJitRundownV1. "
            "It was built from a different PerfectHash version.")
if result.stderr.splitlines() != [expected]:
    raise AssertionError(f"Expected one diagnostic naming the four versioned entries: {result.stderr}")
