---
name: add-new-file
description: Add or update PerfectHash generated file outputs (file work entries, prepare/save callbacks, CMake/VCXProj registration, and tests) when new output files or artifacts need to be added.
---

# Add New Generated File

## Workflow
- Add a new entry to `src/PerfectHash/PerfectHashFileWork.h` in `VERB_FILE_WORK_TABLE` with the right suffix/extension/EOF init; keep ordering invariants (streams follow owning files, VC project entries stay contiguous, context files stay contiguous).
- Implement prepare/save callbacks in `src/PerfectHash/Chm01FileWork*.c`. If a stage is intentionally unused, set `PrepareXxxChm01` or `SaveXxxChm01` to `NULL` in `src/PerfectHash/Chm01FileWork.h`.
- If a new file extension is needed, add a `UNICODE_STRING` in `src/PerfectHash/PerfectHashConstants.c` and wire it into the file work entry.
- Register new sources in `src/PerfectHash/CMakeLists.txt`; mirror additions in `src/PerfectHash/PerfectHash.vcxproj` and `src/PerfectHash/PerfectHash.vcxproj.filters`.
- Update `src/PerfectHash/Chm01FileWork.c` path logic when a file needs custom base names, directories, or stream rules.
- Update generators/tests (`src/PerfectHash/Chm01FileWorkCMakeListsTextFile.c`, `tests/run_cli_codegen_test.cmake`, etc.) if the new file needs build/test coverage.
- Add or expand tests to write the new files and compile/run them.

## File Enum Capacity
- When the 64-bit bitmap for file enums is exhausted, add a new enum group with `_2`, `_3`, etc. suffix and preserve existing ordering rules.
