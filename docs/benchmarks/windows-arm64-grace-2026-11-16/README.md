# Windows ARM64 Grace benchmark baseline (2026-11-16)

Native Windows ARM64 Debug and Release builds on an NVIDIA Grace A02P (72 cores, 495 GiB RAM), using Visual Studio 2026 / MSVC 19.51, CMake 4.4.4, Ninja 1.13.2, and Rust 1.100.0. GPU correctness checks use CUDA 13.5; these are CPU performance measurements.

Base revision: `ab4bc9d10f58e9c0728d0b01fe0fa95b10d0a203`, with the fixes in this PR. Unmodified main did not build on this machine, so this is a baseline, not a before/after speedup claim.

## Method

This follows the existing `perfecthash_benchmarks` harness and the README generated `BenchmarkIndex` / `BenchmarkFull` precedent. Release builds, no JIT, affinity mask 4 (one logical CPU), and `PERFECT_HASH_ONLINE_MAX_CONCURRENCY=1`. Three independent process runs; each runs the harness warmup and five timed iterations. Each iteration uses 50 passes over the keys (five passes for one million keys). Table construction is measured once per process. Runs occurred in a normal desktop session, not an isolated performance lab.

HologramWorld contains 31,016 keys in file order. Synthetic inputs use the existing seed `0x4b1d5d73` and random lookup order. All lookups hit known keys. PerfectHash computes an index; the standard containers perform membership/value lookup, so the timings do not establish equivalent dictionary semantics or miss performance.

## Online scalar lookup

Medians of three process averages, with PerfectHash min–max in parentheses. Times are ns per lookup; construction is ms. Synthetic cases disable 16-bit assigned data.

| Case | PerfectHash ns | unordered_map ns | map ns | Construction ms |
|---|---:|---:|---:|---:|
| hologram-msr32 | 7.1 (7.0–7.1) | 10.3 | 44.0 | 17.86 |
| hologram-msr16 | 4.5 (4.4–4.5) | 10.3 | 43.8 | 17.61 |
| random100k-MultiplyShiftR | 10.3 (10.3–13.3) | 13.9 | 227.8 | 252.58 |
| random100k-MultiplyShiftRX | 11.7 (11.7–11.9) | 14.0 | 227.1 | 41.55 |
| random100k-Mulshrolate1RX | 11.8 (11.8–12.4) | 13.9 | 227.3 | 36.03 |
| random100k-Mulshrolate2RX | 12.4 (11.3–12.6) | 13.9 | 227.0 | 41.40 |
| random100k-Mulshrolate3RX | 12.3 (12.0–12.8) | 13.9 | 226.4 | 36.68 |
| random100k-Mulshrolate4RX | 13.1 (12.3–13.9) | 13.9 | 226.5 | 36.48 |
| random1m-Mulshrolate4RX | 18.7 (18.4–19.9) | 26.2 | 521.1 | 1222.05 |

MultiplyShiftR with one million synthetic keys did not solve within the 30-second limit. This exposed an online API bug that published an incomplete table after an informational timeout status. The fix returns a failure and no table handle; timeout runs are excluded from lookup statistics.

## Generated C benchmark precedent

Generated ARM64 benchmark routines run on a dedicated pinned worker, preserving the caller's complete scheduling eligibility. Worker setup is outside the timed region; worker creation or pinning failure returns an unavailable sentinel that the runner rejects. Seven process runs per executable, each returning its minimum measured CPU cycle count as the process exit code. `BenchmarkIndex` measures 1,000 calls on the same first key, best of 1,000 batches. `BenchmarkFull` takes the best of 100 attempts at ten insert/lookup/delete passes over all 31,016 keys. These hot-key measurements are distinct from the online full-key sweep above. The pinned generated executables read PMCCNTR_EL0, matching the Windows SDK cycle-counter helper. Internal context intervals may cross worker cores, so their raw cycle fields are zero (unavailable) on Windows ARM64; their QueryPerformanceCounter elapsed times remain valid.

| Hash | Median index cycles/call | Median full cycles/op |
|---|---:|---:|
| MultiplyShiftR | 4.010 | 4.695 |
| MultiplyShiftRX | 4.010 | 5.108 |
| Mulshrolate1RX | 4.016 | 5.395 |
| Mulshrolate2RX | 4.010 | 5.084 |
| Mulshrolate3RX | 4.379 | 5.614 |
| Mulshrolate4RX | 5.010 | 6.289 |

MultiplyShiftRX and Mulshrolate1RX resized to 131,072 vertices with 32-bit assigned data; the other generated tables use 65,536 vertices and 16-bit assigned data. All generated CPU validators pass before timing.

The tables above use the non-inline `BenchmarkIndex` routine. The inline variant emitted into the generated C++ unity source now reads its key and stores its result through `volatile` locals on every platform, so the optimizer cannot hoist or drop the lookup. That adds a load and a store per call; inline results generated before this change are not comparable.

## Reproduction and validation

Follow the native ARM64 setup in the root README, then run:

```powershell
$env:CMAKE_GENERATOR = "Ninja Multi-Config"
ctest --test-dir build -C Debug --output-on-failure --timeout 180 -j 4
ctest --test-dir build -C Release --output-on-failure --timeout 180 -j 4
python docs/benchmarks/windows-arm64-grace-2026-11-16/run.py
```

Run the Release tests last: both configurations share generated-output directories. The runner saves raw output and JSON in `build/benchmarks-grace`. Its default affinity mask assumes this machine; adjust it on machines with fewer than three logical CPUs.

Debug and Release each pass 32 tests, with zero failures; seven LLVM/JIT tests are skipped because those backends are not installed. The six generated CUDA validators also pass on GPU 1 (GB300); the full suites use GPU 0 (RTX PRO 4000). Coverage includes generated C/C++, Rust, Python, and CUDA validation, graph storage resizing, rejection of unsolved online tables, and file-backed retries on the same table/context, including tiny generated tables and preservation of shared files. This does not validate the optional native CUDA solver build, JIT performance, or the Linux/macOS compatibility changes.

Raw output, `results.json`, `generated.json`, and `environment.json` accompany this report. Performance values are rounded by the existing harness; no confidence intervals are inferred from three runs.
