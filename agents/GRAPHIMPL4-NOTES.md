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
- Persisted/reloaded GraphImpl4 tables now restore the CHM01 runtime metadata
  needed by compact-key paths:
  - graph implementation version
  - final downsize bitmap metadata
  - GraphImpl4 effective-key/inner compaction metadata
- File-backed reload coverage now includes:
  - sparse 32-bit GraphImpl4 tables compiled through LLVM `JitIndex32`
  - downsized 64-bit GraphImpl4 tables compiled through LLVM `JitIndex64`
  - keyless non-GraphImpl downsized 64-bit tables
- Dedicated CLI/codegen coverage now exercises downsized 64-bit GraphImpl4
  output and validates the generated C/C++ header, Rust, CUDA, and Python paths.
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
  ./build-graphimpl4/bin/perfecthash_unit_tests --gtest_filter='PerfectHashOnlineJitTests.CreateTable64AndIndex64:PerfectHashOnlineJitTests.Index64OnNonGraphImpl64BitTableUsesMetadata:GraphImpl4BitUtils.ContiguousBitmapDetection:GraphImpl4BitUtils.ComposedDownsizeMetadataUsesComposedBitmap:GraphImpl4BitUtils.ComposedDownsizeMetadataRejectsInnerContiguityMismatch:GraphImpl4BitUtils.ComposedExtractionMatchesTwoStepExtraction:GraphImpl4BitUtils.Downsized64OuterBitmapProducesIdentityInnerBitmap:PerfectHashOnlineTests.GraphImpl4Assigned8RequiresOptIn:PerfectHashOnlineTests.GraphImpl4SupportsDownsized64BitInputs:PerfectHashOnlineTests.GraphImpl4RejectsNonGoodHashes:PerfectHashOnlineTests.GraphImpl4Assigned8JitRejected:PerfectHashOnlineTests.GraphImpl4RawDogJitMatchesIndexAssigned32:PerfectHashOnlineTests.GraphImpl4RawDogJitMatchesIndexSparse32:PerfectHashOnlineTests.GraphImpl4RawDogIndex32x4MatchesIndexAssigned16:PerfectHashOnlineTests.GraphImpl4LlvmJitMatchesIndexAssigned32:PerfectHashOnlineTests.GraphImpl4LlvmJitMatchesIndexSparse32:PerfectHashOnlineTests.GraphImpl4LlvmJitMulshrolate3RXMatchesIndexAssigned32:PerfectHashOnlineTests.GraphImpl4LlvmIndex32x4MatchesIndexAssigned16:PerfectHashOnlineTests.GraphImpl4LlvmIndex64x4MatchesDownsizedIndex:PerfectHashOnlineTests.GraphImpl4FileBackedReloadPreservesSparse32Compaction:PerfectHashOnlineTests.GraphImpl4FileBackedReloadPreservesDownsized64Compaction:PerfectHashOnlineTests.NonGraphImplFileBackedReloadPreservesDownsized64Metadata:PerfectHashOnlineTests.GraphImpl4FileBackedReloadSparse32JitIndex32:PerfectHashOnlineTests.GraphImpl4FileBackedReloadDownsized64JitIndex64'
  ```

- Focused persisted reload tests:

  ```bash
  ./build-graphimpl4/bin/perfecthash_unit_tests --gtest_filter='PerfectHashOnlineTests.GraphImpl4FileBackedReloadPreservesSparse32Compaction:PerfectHashOnlineTests.GraphImpl4FileBackedReloadPreservesDownsized64Compaction:PerfectHashOnlineTests.NonGraphImplFileBackedReloadPreservesDownsized64Metadata:PerfectHashOnlineTests.GraphImpl4FileBackedReloadSparse32JitIndex32:PerfectHashOnlineTests.GraphImpl4FileBackedReloadDownsized64JitIndex64'
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

- Persisted/reloaded GraphImpl4 tables are supported for newly-created CHM01
  files that include the extended graph info metadata.
- Older experimental GraphImpl4 table files that predate the extended graph
  info metadata cannot reliably reconstruct compact-key state and cannot be
  distinguished from legacy pre-metadata CHM01 files on disk.
- Regenerate any experimental GraphImpl4 files produced before the persisted
  metadata and downsize-contiguity fixes in this branch. They may not carry the
  final composed bitmap/contiguity state needed by loaded-table JIT.
- Loaded-table JIT support is intentionally limited to the JIT compile path.
  Non-JIT table compilation still requires a created table, and RawDog loaded
  table JIT remains out of scope.
- Assigned8 JIT was deferred by design. It is not a guard-only problem; it
  will require real LLVM/RawDog backend work.
- GraphImpl4 compare-backend / compare-isa benchmark modes are no longer
  blanket-blocked, but they were not exhaustively validated in this tranche.

## Suggested Next Pickup Points

1. Expand GraphImpl4 benchmark coverage for:
   - `--compare-backends`
   - `--compare-isa`
   - explicit vector-width combinations on assigned16/assigned32 tables

2. Only if worthwhile, implement assigned8 JIT support.
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
