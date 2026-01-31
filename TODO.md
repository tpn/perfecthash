# AVX/AVX2/AVX-512 Porting Progress

- [x] Inventory MASM assembly and AVX intrinsics usage.
- [x] Add NASM include and convert MASM routines (RtlCopyPages, RtlFillPages, memset, FastIndexEx).
- [x] Update CMake to build NASM assembly and link PerfectHashAsm cross-platform.
- [x] Enable AVX intrinsics on Linux builds (GraphAvx + AVX selection in Graph/Rtl/TableCreate).
- [x] Build and smoke-test on Linux.
- [ ] Verify Windows build (NASM or MASM fallback).

# JIT/Vector Follow-ups

- [ ] Add masked vector index variants to handle tail keys without scalar fallback (AVX2/AVX-512 masks).
- [ ] Benchmark tail handling with non-multiple key counts (e.g., 31,021) and consider x4/x2 fallback.
