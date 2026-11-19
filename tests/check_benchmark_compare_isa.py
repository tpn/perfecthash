"""Check that the x64 Mulshrolate4RX comparison reports only compiled widths."""
import os
import re
import subprocess
import sys


command = [*sys.argv[1:], "--hash=Mulshrolate4RX", "--keys=64",
           "--iterations=1", "--loops=1", "--no-std-map-baselines"]


def run(*arguments):
    return subprocess.run([*command, *arguments], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=60,
                          env=dict(os.environ, PERFECT_HASH_ONLINE_MAX_CONCURRENCY="1"))


comparison = run("--compare-isa")
if comparison.returncode:
    raise AssertionError(comparison.stdout)

rows = set(re.findall(r"^(llvm|rawdog)\s+\w+\s+\w+\s+(x\d+)\s+\d",
                      comparison.stdout, re.MULTILINE))
skipped = set(re.findall(r"^Skipping unsupported (llvm|rawdog) \w+ (x\d+)$",
                         comparison.stdout, re.MULTILINE))
for backend in ("llvm", "rawdog"):
    if (backend, "x1") not in rows:
        raise AssertionError(f"Missing {backend} scalar baseline\n{comparison.stdout}")
    for width, isa in ((4, "avx"), (8, "avx2"), (16, "avx512")):
        entry = (backend, f"x{width}")
        # Compare to an independent single-width run. The ordinary benchmark
        # may fall back to scalar, so read the actual compiled widths.
        flags = [f"--jit-backend={backend}", f"--jit-max-isa={isa}", f"--index32x{width}"]
        if width in (4, 8):
            flags.append(f"--jit-vector-index32x{width}")
        probe = run(*flags)
        if probe.returncode:
            if "CompileTable failed: 0xe0040230" not in probe.stdout:  # PH_E_NOT_IMPLEMENTED
                raise AssertionError(probe.stdout)
            available = False
        else:
            compiled = re.search(r"JIT compiled widths:([^\n]*)", probe.stdout)
            if compiled is None:
                raise AssertionError(probe.stdout)
            available = str(width) in compiled[1].split()
        if (entry in rows) != available or (entry in skipped) == available:
            raise AssertionError(f"{entry} availability was {available}\n{comparison.stdout}")
