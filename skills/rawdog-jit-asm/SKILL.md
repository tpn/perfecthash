---
name: rawdog-jit-asm
description: Author and wire RawDog JIT NASM blobs for PerfectHash online (x64), including sentinel patching, CMake header generation, and validation against LLVM/offline outputs. Use when adding or modifying RawDog JIT assembly for a hash function or when comparing compiler output for reference.
---

# RawDog JIT Assembly

## Overview

Author x64 NASM RawDog JIT Index() blobs, wire them into the RawDog backend, and validate correctness against SlowIndex and offline compiler output.

## Workflow

1. Identify the exact index algorithm
   - Inspect `src/PerfectHash/PerfectHashTableHashEx.c` for the hash math.
   - Inspect `src/PerfectHash/ChmOnline01.c` (BuildChm01Index* switch) for the actual index logic (masking, table loads, index mask).

2. Create the NASM blob
   - Copy style and conventions from `src/PerfectHash/PerfectHashTableFastIndexEx_x64_01.asm` and existing RawDog blobs.
   - Use the RawDog sentinel values for `Assigned`, `Seed1`, `Seed2`, `Seed3Byte1`, `Seed3Byte2`, `HashMask`, and `IndexMask` as needed.
   - Ensure each sentinel appears exactly once in the data block; load once into a register and reuse if needed.

3. Generate the embedded header
   - Add a CMake `add_custom_command()` to run `nasm` -> `objcopy` -> `scripts/generate_rawdog_jit_header.py`.
   - Append the generated header to `Private_Header_Files` in `src/PerfectHash/CMakeLists.txt`.

4. Wire the backend
   - Update `src/PerfectHash/ChmOnline01RawDog.c` to select the new blob and patch sentinels.
   - Keep RawDog constrained to supported flags (scalar Index32 only for now).

5. Validate
   - Add or update a RawDog unit test in `tests/PerfectHashOnlineTests.cpp`.
   - Run the offline compiler path and disassemble the resulting `.so` for reference (see `references/offline-compile-and-disasm.md`).

## References

- Offline build + disassembly workflow: `references/offline-compile-and-disasm.md`
