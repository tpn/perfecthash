# Offline Vectorized Index Notes

## Mission
Port online-style vectorized index routines into offline-generated C/C++ outputs, starting with `Mulshrolate1RX` using AVX2 (`x8`) and AVX-512 (`x16`) paths while preserving portable scalar fallback behavior.

## Scope (Current Session)
- Hash family focus:
  - `Mulshrolate1RX` (first implementation target)
- Architectures / widths in-scope now:
  - AVX2: `Index32x8`
  - AVX-512: `Index32x16`
- Required constraints:
  - Keep generated offline projects building with GCC and Clang/LLVM.
  - Avoid assumptions that `TABLE_DATA` is 32-bit (it can be 8/16/32-bit); keep table reads type-safe.
  - Preserve scalar behavior and correctness for all targets.

## Technical Direction
- Implement new vectorized entrypoints in `src/CompiledPerfectHashTable/CompiledPerfectHashTableChm01IndexMulshrolate1RXAnd.c`.
- Vectorize arithmetic stages only (multiply/rotate/shift), then perform lane-wise scalar table lookups and index combines:
  - Mirrors existing RawDog assembly behavior.
  - Avoids unsafe 32-bit gathers when `TABLE_DATA` is narrower than 32-bit.
- Add scalar fallback implementations when AVX2/AVX-512 are unavailable.
- Add correctness checks and benchmark hooks so generated offline artifacts validate and measure the new routines.

## Expected Follow-On
After `Mulshrolate1RX` is complete and validated, apply the same approach to:
- `MultiplyShiftR`
- `MultiplyShiftRX`
- `Mulshrolate2RX`
- `Mulshrolate3RX`
- `Mulshrolate4RX`

## Risks / Watch Items
- Generated code path uses embedded RawCString headers in `src/PerfectHash/`; template changes require synchronized RawCString updates.
- AVX feature availability/flags differ across compilers and generated build systems.
- Inline-header generation mode (`CPH_INLINE_ROUTINES`) must not emit non-inline exported symbols.
