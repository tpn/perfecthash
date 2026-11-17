# JIT lookup and NEON solver performance on Windows ARM64

On volt (NVIDIA Grace A02P, 72 cores), LLVM's eight-key lookup was the fastest
configuration measured. RawDog compiled faster and also improved lookup
throughput. The four-lane native NEON solver reduced the time spent hashing
keys. The effect on the complete solve varied by hash and assigned-data width.

These measurements use the Windows ARM64 changes based on `f2d5be6`, Windows
11 Enterprise build 28000, MSVC 19.51, LLVM 23.1.6, and a Release build.
Windows reported SVE and SVE2. A native `rdvl` probe returned 16 bytes (128
bits) on the calling thread. LLVM measurements below use a NEON cap. The
16-key LLVM kernel emitted vector multiplies and shifts on four 32-bit lanes
per instruction. RawDog's multi-key ARM64 routines batch scalar operations.

## Online lookup

Median nanoseconds per key over three independent processes. Each cell uses
the same HologramWorld file, MultiplyShiftR hash, fixed successful seeds,
65,536 vertices and file-order lookups. The assigned-data width is selected
explicitly. Each process warms up once, then measures five iterations of
50 passes over all 31,016 keys with the repository's `perfecthash_benchmarks`.
The runner checks the active backend, compiled width and vector flags.

| Backend and assigned data | 1 key | 4 keys | 8 keys | 16 keys |
|---|---:|---:|---:|---:|
| Native, 16-bit | 4.50 | | | |
| RawDog, 16-bit | 3.63 | 2.24 | 2.10 | 2.85 |
| LLVM, 16-bit | 2.92 | 1.87 | **1.68** | 1.94 |
| Native, 32-bit | 7.18 | | | |
| RawDog, 32-bit | 3.78 | 2.38 | 2.19 | 3.31 |
| LLVM, 32-bit | 3.01 | 1.97 | **1.78** | 1.97 |

At width eight, LLVM handled **2.7×** as many keys per second as native scalar
lookup with 16-bit assigned data and **4.0×** with 32-bit data. The harness
calls the COM table's `Index` method for individual keys and the JIT
interface's `Index32xN` methods for batches. The timed region includes
writing and consuming the indices. A batch amortizes interface calls over multiple keys, so the
ratios measure the complete lookup path. The native results reproduce the
earlier 4.5 and 7.1 ns/key measurements. [Earlier Grace baseline](../windows-arm64-grace-2026-11-16/README.md)

| Width-eight compile time, median per fresh process | 16-bit assigned | 32-bit assigned |
|---|---:|---:|
| RawDog | 0.115 ms | 0.072 ms |
| LLVM | 14.25 ms | 14.16 ms |

Compile time covers the `CompileTable` call, including any first-use backend
initialization, and is outside the lookup timings. This workload reuses the
table for every pass.

## Solver key hashing

Each comparison uses 45 pairs of fresh `PerfectHashCreate` processes.
Pairs alternate which implementation runs first. The runner requests one
worker, the shipped successful seeds for each hash, and a single attempt.
It checks the winning implementation, assigned-data width, vertex count and
seeds in each CSV. The hash column uses elapsed QPC ticks divided by the
measured frequency (10 MHz). The solve column uses `SolveMicroseconds`.

The time columns show the median for each implementation. For speedup, divide
scalar time by NEON time within each adjacent pair, then take the median of the
45 ratios. A value above 1 favors NEON.

| Hash | Assigned | Scalar hash (µs) | NEON hash (µs) | Paired hash speedup | Pairs favoring NEON | Paired whole-solve speedup |
|---|---|---:|---:|---:|---:|---:|
| MultiplyShiftR | 16-bit | 189.4 | 11.4 | 4.42× | 39/45 | 1.04× |
| MultiplyShiftR | 32-bit | 38.7 | 12.3 | 3.17× | 35/45 | 1.00× |
| MultiplyShiftRX | 16-bit | 34.8 | 11.7 | 3.09× | 35/45 | 1.03× |
| MultiplyShiftRX | 32-bit | 32.5 | 71.8 | 2.82× | 32/45 | 0.98× |

NEON reduced the hash phase in **141 of 180 pairs**. The complete solve showed
little consistent benefit. Both implementations had distinct clusters of fast
and slow runs in this desktop session. For 32-bit MultiplyShiftRX, for
example, NEON hash times ranged from 10.2 µs at the 10th percentile to
176.6 µs at the 90th. This spread explains how NEON could win most pairs
yet have a higher standalone median. All runs are included. Larger inputs
and solves that need several attempts may spend a different share of their
time hashing.

## Reproduce

Build with both JITs using the native Windows ARM64 instructions in the root
README, keep the LLVM runtime on `PATH`, then run:

```powershell
python docs/benchmarks/windows-arm64-grace-jit/run.py
```

The runner pins itself and its children using affinity mask 4 and sets online
solver concurrency to one. Lookup cases run in a reproducibly shuffled order.
Use `--build PATH` to select another Ninja Multi-Config build and
`--affinity MASK` to change the processor. The script writes
`grace-jit-results.json` in the build directory. [Per-process measurements](results.json)
