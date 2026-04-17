# PerfectHash Agent Notes

## How to add new generated files
- Add a new entry to `src/PerfectHash/PerfectHashFileWork.h` in the `VERB_FILE_WORK_TABLE` macro with the right suffix/extension/EOF size; keep invariants (streams follow their owning files, VC project entries stay contiguous, context file entries stay contiguous).
- Implement the prepare/save callbacks in `src/PerfectHash/Chm01FileWork*.c`; if a stage is intentionally unused, add a `#define PrepareXxxChm01 NULL` or `#define SaveXxxChm01 NULL` in `src/PerfectHash/Chm01FileWork.h`.
- If you need a new file extension, add a `UNICODE_STRING` in `src/PerfectHash/PerfectHashConstants.c` and reference it in the file work table entry.
- Register new source files in `src/PerfectHash/CMakeLists.txt`; for Visual Studio builds, mirror additions in `src/PerfectHash/PerfectHash.vcxproj` and `src/PerfectHash/PerfectHash.vcxproj.filters`.
- If the filename needs custom path behavior (special base name, directory, or stream rules), update `src/PerfectHash/Chm01FileWork.c` where paths are constructed.
- Update tests or CMakeLists generators if the new file needs compilation or validation coverage.
- See `skills/add-new-file/SKILL.md` for the full step-by-step checklist.

## Main hash functions
- Use `PERFECT_HASH_GOOD_HASH_FUNCTION_TABLE_ENTRY` and `IsGoodPerfectHashHashFunctionId()` in `include/PerfectHash.h` to identify the curated hash set for downstream outputs.
- Current good set: MultiplyShiftR, MultiplyShiftRX, Mulshrolate1RX, Mulshrolate2RX, Mulshrolate3RX, Mulshrolate4RX.

## File enum capacity
- File enums are tracked via a 64-bit bitmap. When we run out of bits, add a new enum group with a `_2`, `_3`, etc. suffix and keep the existing enum ordering rules intact.

## Packaging
- Use `skills/perfecthash-conda-packaging/SKILL.md` for PerfectHash conda build/smoke-test/upload flows, Anaconda.org org-channel publishing, and conda-forge bootstrap/feedstock workflow updates.

## Release Engineering
- Use `skills/perfecthash-release-engineering/SKILL.md` for tag-first release cutting, release workflow validation, release-notes/doc synchronization, and release automation audits.
- Release versioning rule: reserve patch (`X.Y.Z`) bumps for release-oriented non-functional fixes only. Use a minor or major bump for functional changes, new features, or user-visible behavior changes.

## Roborev Workflow
- Repo-local Roborev defaults live in `.roborev.toml`; review scope and operating notes live in `ROBOREV-CONTEXT.md`, `ROBOREV-LOCAL.md`, and `ROBOREV-PLAYBOOK.md`.
- For new machines or fresh clones, use `ROBOREV-BOOTSTRAP.md` and `scripts/bootstrap-roborev-local.sh` instead of `roborev init`; this repo already tracks its Roborev config and hook scripts.
- After every local commit in this repo, wait for the automatic Roborev review to finish with `roborev wait`, inspect it with `roborev show HEAD`, and address actionable findings before considering the work complete.
- Use `scripts/roborev-matrix-review.sh HEAD^..HEAD` for an explicit two-agent commit-level gate, and `scripts/roborev-matrix-review.sh` for a branch-level gate before push or PR updates.
- Read `.roborev/last-review.md` after the matrix script finishes and fix any concrete correctness, regression, build, packaging, or docs issues it identifies in the touched scope.
