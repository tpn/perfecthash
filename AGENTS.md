# PerfectHash Agent Notes

## How to add new generated files
- Add a new entry to `src/PerfectHash/PerfectHashFileWork.h` in the `VERB_FILE_WORK_TABLE` macro with the right suffix/extension/EOF size; keep invariants (streams follow their owning files, VC project entries stay contiguous, context file entries stay contiguous).
- Implement the prepare/save callbacks in `src/PerfectHash/Chm01FileWork*.c`; if a stage is intentionally unused, add a `#define PrepareXxxChm01 NULL` or `#define SaveXxxChm01 NULL` in `src/PerfectHash/Chm01FileWork.h`.
- If you need a new file extension, add a `UNICODE_STRING` in `src/PerfectHash/PerfectHashConstants.c` and reference it in the file work table entry.
- Register new source files in `src/PerfectHash/CMakeLists.txt`; for Visual Studio builds, mirror additions in `src/PerfectHash/PerfectHash.vcxproj` and `src/PerfectHash/PerfectHash.vcxproj.filters`.
- If the filename needs custom path behavior (special base name, directory, or stream rules), update `src/PerfectHash/Chm01FileWork.c` where paths are constructed.
- Update tests or CMakeLists generators if the new file needs compilation or validation coverage.

## File enum capacity
- File enums are tracked via a 64-bit bitmap. When we run out of bits, add a new enum group with a `_2`, `_3`, etc. suffix and keep the existing enum ordering rules intact.
