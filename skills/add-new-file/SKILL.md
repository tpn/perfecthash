# Add New Generated File

Use this when adding a new auto-generated file to PerfectHash output.

## Steps
1) Add a new entry to `src/PerfectHash/PerfectHashFileWork.h` in `VERB_FILE_WORK_TABLE` with suffix/extension/EOF sizing. Respect ordering invariants (streams after owning file; VC project and context file groups contiguous).
2) Implement the prepare/save callback(s) in `src/PerfectHash/Chm01FileWork*.c`. If a stage is unused, add a NULL define in `src/PerfectHash/Chm01FileWork.h`.
3) If the extension is new, define it in `src/PerfectHash/PerfectHashConstants.c` and reference it from the file work table.
4) Add new `.c` sources to `src/PerfectHash/CMakeLists.txt`; mirror in `src/PerfectHash/PerfectHash.vcxproj` and `src/PerfectHash/PerfectHash.vcxproj.filters` if you need VS support.
5) If the filename requires custom path handling (special base name/dir/stream), update `src/PerfectHash/Chm01FileWork.c`.
6) Extend tests or generators to cover the new output (e.g., CMakeLists, CLI tests).

## Notes
- File enums are tracked via a 64-bit bitmap. When it fills, add a new enum group with `_2`, `_3`, etc. and keep ordering rules intact.
