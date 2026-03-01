# Offline Vectorized Index TODO

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Completed

## Planning / Tracking
- [x] Create NOTES/LOG/TODO tracking files for offline vectorization effort.
- [~] Keep TODO synchronized with implementation and validation progress.

## Mulshrolate1RX: AVX2 + AVX-512
- [x] Add `Mulshrolate1RX` vector routine declarations/macros in offline index template.
- [x] Implement `Index32x8` AVX2 arithmetic path + scalar fallback.
- [x] Implement `Index32x16` AVX-512 arithmetic path + scalar fallback.
- [x] Ensure `CPH_INLINE_ROUTINES` handling is correct (no non-inline symbol emission in header-only inline section).

## Generated Test + Benchmark Coverage
- [x] Add correctness validation for new vector routines in generated test path.
- [ ] Add benchmark path for vector routines in generated benchmark index path.
- [x] Confirm scalar and vector outputs match for representative keys.

## RawCString Synchronization
- [x] Regenerate/update RawCString header for modified Mulshrolate1RX index template.
- [x] Regenerate/update RawCString headers for any modified test/benchmark templates.

## Validation
- [x] Rebuild project artifacts used for codegen after source updates.
- [x] Run offline codegen test for Mulshrolate1RX and build generated CMake project.
- [x] Build and validate generated output with GCC.
- [x] Build and validate generated output with Clang/LLVM.
- [x] Run correctness executables and collect benchmark outputs.

## Commits
- [x] Commit planning docs.
- [ ] Commit Mulshrolate1RX vector implementation + test/RawCString sync.
- [ ] Commit tracking/log updates for validation + benchmark results.
