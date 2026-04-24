# GraphImpl4 Notes

## Branch And Commit Stack

- Branch: `codex/graphimpl4-cpu-backend`
- Commit stack on top of `main`:
  - `d47165c` Add experimental GraphImpl4 CPU solver backend
  - `33c90f0` Add GraphImpl4 file-backed codegen support
  - `672c172` Add solver-only GraphImpl4 benchmark support
  - `97c1244` Enable GraphImpl4 JIT for assigned16/32 tables
  - `a37e259` Align mamba envs with LLVM-capable builds

## Current State

- `GraphImpl4` exists as an experimental C++ CPU backend in
  `src/PerfectHash/GraphImpl4.cpp` with template specialization over:
  - assigned/table element width: `uint8_t`, `uint16_t`, `uint32_t`
  - effective key width: `uint8_t`, `uint16_t`, `uint32_t`
- Solver/index path is working for:
  - 32-bit keys
  - downsized 64-bit keys that compact to 32 bits or less
  - curated good hash set only
- File-backed create/self-test/codegen path is working for the current smoke:
  - the old `--SkipTestAfterCreate` workaround is removed
  - generated-project configure/build is part of the smoke now
- GraphImpl4 JIT is now enabled for:
  - assigned32 tables
  - assigned16 tables
- GraphImpl4 JIT remains intentionally disabled for:
  - assigned8 tables, which return `PH_E_NOT_IMPLEMENTED`
- Benchmark harness no longer force-disables GraphImpl4 JIT.
- LLVM-capable env/bootstrap surfaces were updated so LLVM support is explicit:
  - `dependencies.yaml`
  - generated `conda/environments/*compiler-llvm.yaml`
  - `scripts/install-deps/linux-mamba-env.yaml`
  - `scripts/install-deps/windows-mamba-env.yaml`
  - `ui/src/pages/developer/Environment.jsx`
- `tests/CMakeLists.txt` now makes `perfecthash_unit_tests` and
  `perfecthash_benchmarks` depend on `PerfectHashLLVM` when that target exists,
  so LLVM-specific test paths do not silently skip just because the runtime
  library was never built.

## Verified

- Focused GraphImpl4 online tests:

  ```bash
  ./build-graphimpl4/bin/perfecthash_unit_tests --gtest_filter='PerfectHashOnlineTests.GraphImpl4Assigned8RequiresOptIn:PerfectHashOnlineTests.GraphImpl4SupportsDownsized64BitInputs:PerfectHashOnlineTests.GraphImpl4RejectsNonGoodHashes:PerfectHashOnlineTests.GraphImpl4Assigned8JitRejected:PerfectHashOnlineTests.GraphImpl4RawDogJitMatchesIndexAssigned32:PerfectHashOnlineTests.GraphImpl4RawDogIndex32x4MatchesIndexAssigned16:PerfectHashOnlineTests.GraphImpl4LlvmJitMatchesIndexAssigned32:PerfectHashOnlineTests.GraphImpl4LlvmJitMulshrolate3RXMatchesIndexAssigned32:PerfectHashOnlineTests.GraphImpl4LlvmIndex32x4MatchesIndexAssigned16:PerfectHashOnlineTests.GraphImpl4LlvmIndex64x4MatchesDownsizedIndex'
  ```

- Current quick sanity slice:

  ```bash
  ./build-graphimpl4/bin/perfecthash_unit_tests --gtest_filter='PerfectHashOnlineTests.GraphImpl4Assigned8JitRejected:PerfectHashOnlineTests.GraphImpl4RawDogJitMatchesIndexAssigned32:PerfectHashOnlineTests.GraphImpl4LlvmJitMatchesIndexAssigned32'
  ```

- GraphImpl4 ctest slice:

  ```bash
  ctest --test-dir build-graphimpl4 --output-on-failure -R 'graphimpl4'
  ```

- Manual benchmark checks:

  ```bash
  ./build-graphimpl4/bin/perfecthash_benchmarks --keys=256 --iterations=1 --loops=1 --graph-impl=4 --jit-backend=rawdog --no-std-map-baselines
  ./build-graphimpl4/bin/perfecthash_benchmarks --keys=256 --iterations=1 --loops=1 --graph-impl=4 --jit-backend=rawdog --allow-assigned16 --no-std-map-baselines
  ```

  Observed locally:
  - default path reported `Assigned element bits: 32`
  - `--allow-assigned16` path reported `Assigned element bits: 16`

## Important Caveats

- Persisted/reloaded GraphImpl4 tables are still the main unfinished area.
  The safe assumption remains: GraphImpl4 is **create-only** for now.
- `LoadPerfectHashTableImplChm01()` does not reconstruct GraphImpl4 compact-key
  metadata. That matters especially for downsized 64-bit tables.
- Assigned8 JIT was deferred by design. It is not a guard-only problem; it
  will require real LLVM/RawDog backend work.
- The current file-backed GraphImpl4 codegen smoke is still small and 32-bit.
  There is no dedicated file-backed smoke yet for downsized 64-bit GraphImpl4.
- GraphImpl4 compare-backend / compare-isa benchmark modes are no longer
  blanket-blocked, but they were not exhaustively validated in this tranche.

## Suggested Next Pickup Points

1. Decide whether GraphImpl4 should stay create-only or whether to extend
   `TABLE_INFO_ON_DISK` so compact-key metadata is persisted and reloadable.

2. If persisted reload support is desired:
   - add the necessary GraphImpl4 metadata to the on-disk header
   - reconstruct it during `LoadPerfectHashTableImplChm01()`
   - add a file-backed reload smoke for downsized 64-bit GraphImpl4 tables

3. Add a dedicated CLI/codegen smoke for downsized 64-bit GraphImpl4 so the
   compacted-key path is exercised through generated outputs, not just online
   mode and table JIT.

4. Expand GraphImpl4 benchmark coverage for:
   - `--compare-backends`
   - `--compare-isa`
   - explicit vector-width combinations on assigned16/assigned32 tables

5. Only if worthwhile, implement assigned8 JIT support.
   Expect changes in both:
   - LLVM table element typing / load paths
   - RawDog code asset selection / patching

## Practical Notes For A New Session

- If LLVM JIT behavior looks missing, check whether `PerfectHashLLVM` was
  actually built in the current tree. The env packages alone are not enough.
- LLVM-capable profiles are:
  - `full`
  - `online-rawdog-and-llvm-jit`
  - `online-llvm-jit`
- For local work where LLVM matters, prefer one of the above profiles over an
  ad hoc default build.
