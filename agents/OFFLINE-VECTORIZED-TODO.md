# Offline Vectorized Index TODO

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Completed

## Planning / Tracking
- [x] Create NOTES/LOG/TODO tracking files for offline vectorization effort.
- [~] Keep TODO synchronized with implementation and validation progress.

## Mulshrolate1RX: AVX2 + AVX-512
- [ ] Add `Mulshrolate1RX` vector routine declarations/macros in offline index template.
- [ ] Implement `Index32x8` AVX2 arithmetic path + scalar fallback.
- [ ] Implement `Index32x16` AVX-512 arithmetic path + scalar fallback.
- [ ] Ensure `CPH_INLINE_ROUTINES` handling is correct (no non-inline symbol emission in header-only inline section).

## Generated Test + Benchmark Coverage
- [ ] Add correctness validation for new vector routines in generated test path.
- [ ] Add benchmark path for vector routines in generated benchmark index path.
- [ ] Confirm scalar and vector outputs match for representative keys.

## RawCString Synchronization
- [ ] Regenerate/update RawCString header for modified Mulshrolate1RX index template.
- [ ] Regenerate/update RawCString headers for any modified test/benchmark templates.

## Validation
- [ ] Rebuild project artifacts used for codegen after source updates.
- [ ] Run offline codegen test for Mulshrolate1RX and build generated CMake project.
- [ ] Build and validate generated output with GCC.
- [ ] Build and validate generated output with Clang/LLVM.
- [ ] Run correctness executables and collect benchmark outputs.

## Commits
- [ ] Commit planning docs.
- [ ] Commit Mulshrolate1RX vector implementation.
- [ ] Commit test/benchmark + RawCString sync + validation adjustments.
