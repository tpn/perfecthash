# Offline Vectorized Index Log

## 2026-03-01
- Initialized tracking files:
  - `agents/OFFLINE-VECTORIZED-NOTES.md`
  - `agents/OFFLINE-VECTORIZED-TODO.md`
  - `agents/OFFLINE-VECTORIZED-LOG.md`
- Audited current offline generation path:
  - Verified `src/CompiledPerfectHashTable/CompiledPerfectHashTableChm01IndexMulshrolate1RXAnd.c` is scalar-only.
  - Verified generated offline C/C++ output is stitched from RawCString payloads in `src/PerfectHash/*_RawCString.h`.
  - Confirmed RawDog assembly references for `Mulshrolate1RX` x8/x16 and AVX2 x4.
- Chosen implementation strategy for first step:
  - Vectorize arithmetic in AVX2/AVX-512.
  - Keep lane-wise scalar table lookup to preserve correctness across varying `TABLE_DATA` element widths.
- Implemented `Mulshrolate1RX` offline vectorized routines and wiring:
  - Added `Index32x8` and `Index32x16` routine-name macros in
    `src/CompiledPerfectHashTable/CompiledPerfectHashTableChm01IndexMulshrolate1RXAnd.c`.
  - Added AVX2 x8 and AVX-512 x16 arithmetic helpers via x86 intrinsics.
  - Added scalar vertex fallback helpers and runtime CPU feature checks for
    GCC/Clang target-attribute builds.
  - Added generated test coverage in
    `src/CompiledPerfectHashTable/CompiledPerfectHashTableTest.c` to compare
    vector routine outputs against scalar `INDEX_ROUTINE` for first 8/16 keys.
- Synced RawCString payloads after template changes:
  - `src/PerfectHash/CompiledPerfectHashTableChm01IndexMulshrolate1RXAnd_CSource_RawCString.h`
  - `src/PerfectHash/CompiledPerfectHashTableTest_CSource_RawCString.h`
- Fixed build regression discovered by codegen test:
  - Generated C compilation failed with duplicate `static` (`static FORCEINLINE`).
  - Removed redundant `static` on helper declarations in template and RawCString.
- Rebuilt codegen producer (`build-tests/bin/PerfectHashCreate`) and validated:
  - `ctest --test-dir build-tests -R perfecthash.cli.codegen.mulshrolate1rx`
    now passes.
- Generated a dedicated offline `Mulshrolate1RX` table and validated generated
  project builds/runs with both compilers:
  - GCC (`mamba run -n perfecthash-linux`, `CC=gcc CXX=g++`): generated
    `Test_*` returned `0`.
  - Clang (`mamba run -n dev-linux_os-linux_arch-x86_64_py-314_cuda-none_compiler-llvm`,
    `CC=clang CXX=clang++`): generated `Test_*` returned `0`.
- Collected benchmark data:
  - Generated `BenchmarkIndex_*` executable:
    - GCC run output: `13866`
    - Clang run output: `19352`
  - Temporary vector benchmark harness (`vector_index_bench.c`) in generated
    output measured scalar vs x8/x16 wrappers over 2,000,000 iterations
    (16 keys/iteration):
    - GCC: scalar `2.427 ns/key`, x8 `3.094 ns/key`, x16 `2.115 ns/key`
    - Clang: scalar `2.304 ns/key`, x8 `2.590 ns/key`, x16 `2.465 ns/key`
