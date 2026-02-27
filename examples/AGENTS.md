# Examples Agent Notes

## Scope
- Default working scope is `examples/` unless a task explicitly requires touching root-level CMake or source files.
- Keep this subtree optimized for two audiences: humans reading integration guidance and LLMs ingesting concise, structured context.

## Required Workflow
- Update `examples/TODO.md` before and after meaningful work (mark planned/in-progress/done).
- Append progress entries to `examples/LOG.md` only; do not rewrite or reorder prior entries.
- Keep design decisions and assumptions in `examples/NOTES.md`.
- Commit in small, reviewable increments with messages that mention `examples`.

## CMake/Integration Guardrails
- Prefer a `find_package(...)`-based consumer flow over `add_subdirectory(...)`.
- Keep the example self-contained and portable across Windows (MSVC, clang-cl), Linux (GCC, Clang), and macOS (AppleClang, Clang).
- Keep architecture routing explicit (`x86_64` vs `arm64`) and avoid cross-arch blobs in a single build artifact.
- Favor the smallest runtime target that still supports online table creation + RawDog JIT.
- For dual-backend samples, use explicit user-facing backend names:
  `rawdog-jit`, `llvm-jit`, `auto`.

## SQLite Example Guardrails
- First sqlite integration pass should prefer an extension or virtual table
  seam before considering sqlite core planner modifications.
- Keep A/B benchmarking first-class:
  baseline sqlite behavior vs PerfectHash-enabled behavior must be easy to run.
- Keep key-path assumptions explicit:
  primary target is 32-bit keys; if 64-bit inputs are used, define and validate
  the downsize/coercion strategy in docs and code.
- Keep sqlite vendoring explicit and pinned:
  record exact upstream version and download source in `sqlite/VERSION.txt`.

## Documentation Style
- Use concrete file paths and exact target names.
- Keep sections short and scannable.
- Include copy/paste command examples where possible.
- Call out assumptions explicitly, then record validation status once verified.
