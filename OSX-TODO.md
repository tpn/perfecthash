# OSX Bring-up TODO

- [x] Run initial CMake configure/build on macOS arm64 and capture failures.
- [x] Fix macOS-specific compat gaps (sysinfo/get_nprocs, mmap flags, file mapping, SRW locks, VirtualAlloc sizing).
- [x] Add macOS build notes and dependency guidance to docs.
- [x] Run unit/integration tests via CTest and record results.
- [ ] (Optional) Run `rapids-dependency-file-generator` to regenerate all env files from `dependencies.yaml`.
