# AVX/AVX2/AVX-512 Porting Progress

- [x] Inventory MASM assembly and AVX intrinsics usage.
- [x] Add NASM include and convert MASM routines (RtlCopyPages, RtlFillPages, memset, FastIndexEx).
- [x] Update CMake to build NASM assembly and link PerfectHashAsm cross-platform.
- [x] Enable AVX intrinsics on Linux builds (GraphAvx + AVX selection in Graph/Rtl/TableCreate).
- [x] Build and smoke-test on Linux.
- [ ] Verify Windows build (NASM or MASM fallback).
