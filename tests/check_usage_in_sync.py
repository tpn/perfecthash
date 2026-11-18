"""Compare USAGE.txt with the usage text the CLI prints when run bare.

USAGE.txt is a saved copy of that text. Regenerate it from the CLI's output
whenever the usage message in PerfectHashErrors.mc changes.
"""
import difflib
import subprocess
import sys
from pathlib import Path

exe, usage = sys.argv[1:]
result = subprocess.run([exe], capture_output=True, timeout=60)
printed = (result.stdout + result.stderr).decode("ascii").replace("\r\n", "\n")
saved = Path(usage).read_text(encoding="ascii")
if printed != saved:
    diff = difflib.unified_diff(saved.splitlines(), printed.splitlines(),
                                usage, exe, lineterm="", n=1)
    raise AssertionError("USAGE.txt is out of date:\n" + "\n".join(list(diff)[:40]))
