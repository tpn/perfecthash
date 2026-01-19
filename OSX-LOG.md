# OSX Port Log

- 2026-01-19: Started macOS Apple Silicon bring-up; reviewed existing CMake + compat layer and prepared for first build attempt.
- 2026-01-19: Created `perfecthash-macos` mamba env; added macOS env recipes under `conda/environments/` and updated `dependencies.yaml`.
- 2026-01-19: Added macOS compile definitions and page-size detection in CMake; fixed compat layer for macOS (pthread barriers, SRW lock lazy init, file mapping/move/info/handle closures, TLS typing, VirtualAlloc/Protect/Free sizing, MAP_* fallbacks).
- 2026-01-19: Built `Release` via CMake/Ninja and ran full CTest suite successfully (14/14 passing).
