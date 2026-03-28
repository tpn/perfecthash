# GPU POC Nsight Profile Notes

**Date:** 2026-03-28

## Environment

Profilers already present:

- `/usr/local/cuda-13.1/bin/nsys`
- `/usr/local/cuda-13.1/bin/ncu`

Driver state:

- `nvidia-smi`: `NVIDIA GB10`, driver `580.126.09`
- `/proc/driver/nvidia/params`: `RmProfilingAdminOnly: 1`

Implication:

- `nsys` works as the normal user
- `ncu` requires root on this machine
- passwordless `sudo` was available, so `ncu` profiling was run via `sudo -n`

## Profile Targets

Two main profiling targets were used:

1. generated `8193`, `batch=128`, `fixed_attempts=20000`, `device-serial`,
   `block` peel, `assignment_backend=cpu`
2. same input, but `assignment_backend=gpu`

These isolate:

- the best current peel-only hybrid path
- the current scalar GPU assignment bottleneck

## Nsight Systems: Hybrid CPU Backend

Command shape:

```bash
nsys profile \
  --trace=cuda,osrt \
  --sample=none \
  --cpuctxsw=none \
  -o profiling/nsight/generated8193_block_cpu \
  ./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
    --edges 8193 \
    --batch 128 \
    --fixed-attempts 20000 \
    --threads 128 \
    --solve-mode device-serial \
    --device-serial-peel-geometry block \
    --assignment-backend cpu \
    --output-format json
```

`nsys` kernel summary:

- `PeelGraphsDeviceSerialBlockKernel<unsigned short>`
  - `79.6%`
  - `375,949,408 ns`
  - `157` launches
  - average `2.394 ms`
- `BuildGraphsKernel<MultiplyShiftR, unsigned short>`
  - `20.4%`
  - `96,462,752 ns`
  - `157` launches
  - average `0.614 ms`

Takeaway:

- On the current best hybrid path, GPU time is overwhelmingly peel.
- Assignment/verify have already been pushed off the device enough that they are
  no longer visible as meaningful GPU kernels.

## Nsight Systems: GPU Assignment Backend

Same run shape, but `assignment_backend=gpu`.

`nsys` kernel summary:

- `AssignGraphsKernel<unsigned short>`
  - `50.2%`
  - `544,361,184 ns`
  - `157` launches
  - average `3.467 ms`
- `PeelGraphsDeviceSerialBlockKernel<unsigned short>`
  - `34.8%`
  - `377,809,696 ns`
  - `157` launches
  - average `2.406 ms`
- `BuildGraphsKernel<MultiplyShiftR, unsigned short>`
  - `8.9%`
  - `96,498,144 ns`
  - average `0.615 ms`
- `VerifyGraphsKernel<unsigned short>`
  - `6.1%`
  - `66,029,632 ns`
  - average `0.421 ms`

Takeaway:

- The current scalar GPU assignment kernel is the single biggest GPU bottleneck
  in the non-hybrid path.
- This directly explains why the hybrid CPU assignment backend wins so clearly.

## Nsight Compute: Peel Kernel

Target:

- `PeelGraphsDeviceSerialBlockKernel<unsigned short>`
- hybrid CPU backend
- one launch
- `ncu --set basic`

Most relevant metrics:

- block size: `128`
- grid size: `128`
- memory throughput: `7.30%`
- compute throughput: `1.40%`
- achieved occupancy: `14.41%`
- theoretical occupancy: `100%`
- waves per SM: `0.22`
- registers per thread: `40`

Profiler guidance:

- grid is too small to fill the device
- kernel is not compute-bound
- kernel is not saturating memory bandwidth either

Interpretation:

- The kernel is dominated by coordination overhead, atomics, and per-round
  block synchronization with a relatively small amount of actual useful work.
- This points toward improving the internal block algorithm, not just trying to
  “use more math”.

## Nsight Compute: Assign Kernel

Target:

- `AssignGraphsKernel<unsigned short>`
- GPU assignment backend
- one launch
- `ncu --set basic`

Most relevant metrics:

- block size: `1`
- grid size: `128`
- memory throughput: `0.37%`
- compute throughput: `0.33%`
- achieved occupancy: `3.11%`
- theoretical occupancy: `50%`
- waves per SM: `0.11`

Profiler guidance:

- launch configuration warning:
  - single-thread blocks waste almost an entire warp
  - reported speedup potential from better launch configuration is very large

Interpretation:

- The profiler is correctly identifying the launch inefficiency.
- But the underlying ordered dependency chain means we should not “fix” this by
  just widening the launch shape.
- This validates the current decision to keep assignment on CPU for the hybrid
  path until peel-layer metadata exists.

## Batch-Size Sweep

Hybrid path:

- generated `8193`
- `fixed_attempts=20000`
- `device-serial`
- `block` peel
- `assignment_backend=cpu`

Results:

| Batch | Actual Attempts | GPU ms | CPU ms | Approx GPU us/attempt |
|-------|------------------|--------|--------|-----------------------|
| `128` | `20096` | `516.610` | `47.698` | `25.7` |
| `256` | `20224` | `599.459` | `54.483` | `29.6` |
| `512` | `20480` | `605.346` | `53.468` | `29.6` |

Takeaway:

- Simply increasing batch size did **not** help.
- `batch=128` was better than `256` and `512` on this workload.
- So the next win is unlikely to come from just “make the batch bigger”.

## Follow-On Real-Key Control Runs

I also ran a small control matrix after profiling to check whether the current
parallel peel paths stay correct on higher-yield real-key cases.

`Mulshrolate4RX`, `fixed_attempts=2048`, `batch=128`, `assignment_backend=cpu`:

- `HologramWorld-31016.keys`
  - `thread` peel:
    - solved `36`
    - `cpu_success = 36`
    - `mismatches = 0`
  - `warp` peel:
    - solved `36`
    - `cpu_success = 36`
    - `mismatches = 0`
  - `block` peel:
    - solved `36`
    - `cpu_success = 16`
    - `mismatches = 20`

- `Hydrogen-40147.keys`
  - `thread` peel:
    - solved `258`
    - `cpu_success = 258`
    - `mismatches = 0`
  - `warp` peel:
    - solved `258`
    - `cpu_success = 258`
    - `mismatches = 0`
  - `block` peel:
    - solved `258`
    - `cpu_success = 193`
    - `mismatches = 65`

Interpretation:

- on the current binary, the plain `block` peel path is correctness-aligned on
  these real-key controls
- `warp` remains the simpler correctness reference

## Experimental Staged Block Peel

A new `block-staged` peel geometry was then added as an experiment that keeps
the existing `block` kernel untouched and uses a stricter staged round
collection/processing model.

Results:

- generated `8193`, `fixed_attempts=20000`, `assignment_backend=cpu`
  - `block-staged`: GPU `3811.779 ms`
  - current `block`: GPU `516.610 ms`
- `HologramWorld-31016.keys`, `Mulshrolate4RX`, `fixed_attempts=2048`
  - `block-staged`: solved `36`, CPU success `36`, mismatches `0`
- `Hydrogen-40147.keys`, `Mulshrolate4RX`, `fixed_attempts=2048`
  - `block-staged`: solved `258`, CPU success `258`, mismatches `0`

Interpretation:

- `block-staged` fixes the correctness failures seen in the current `block`
  path
- but it is far too slow to be the final performance kernel
- this makes it useful as a correctness oracle / reference path while we debug
  the faster `block` kernel

## Experimental Block-Shared Peel

A second experimental block kernel, `block-shared`, was added after
`block-staged`. It keeps the fast block shape and preserves the current
whole-vertex round scan, but moves two coordination points into block-local
shared state:

- frontier counting during collection
- one round-wide `PeeledCount` reservation instead of one atomic increment per
  peeled edge

Results:

`Mulshrolate4RX`, `fixed_attempts=2048`, `batch=128`, `assignment_backend=cpu`:

- `HologramWorld-31016.keys`
  - `block`: GPU `306.200 ms`
  - `block-shared`: GPU `301.214 ms`
  - both correct
- `Hydrogen-40147.keys`
  - `block`: GPU `1035.545 ms`
  - `block-shared`: GPU `562.849 ms`
  - both correct

`Mulshrolate3RX`, `fixed_attempts=2048`, `batch=128`, `assignment_backend=cpu`:

- `HologramWorld-31016.keys`
  - `block`: GPU `305.070 ms`
  - `block-shared`: GPU `396.659 ms`
  - both correct
- `Hydrogen-40147.keys`
  - `block`: GPU `694.154 ms`
  - `block-shared`: GPU `563.361 ms`
  - both correct

generated `8193`, `fixed_attempts=20000`, `assignment_backend=cpu`:

- `block`: GPU `516.610 ms`
- `block-shared`: GPU `720.727 ms`

Interpretation:

- `block-shared` is now the strongest real-key block peel candidate
- it materially improves the Hydrogen cases
- it is not universally better, because it loses on generated `8193` and on
  Hologram `Mulshrolate3RX`

## Current Recommendation

The next best GPU algorithm work is:

1. Keep hybrid CPU assignment as the default experimental path.
2. Focus on the internals of the block-family peel kernels.
3. Specifically target:
   - `block-shared` as the primary real-key block path
   - continued block-local/shared-memory coordination
   - using CUB/CCCL block primitives where they simplify local compaction or
     prefix work
4. Keep `block-staged` as a correctness oracle / reference path.
5. Keep plain `block` around for comparison on synthetic and lighter runs.
6. Do **not** spend time trying to widen the current ordered GPU assignment
   kernel without adding peel-layer metadata first.

## Relevant Files

- `profiling/nsight/generated8193_block_cpu.nsys-rep`
- `profiling/nsight/generated8193_block_gpu.nsys-rep`
- `profiling/nsight/ncu_generated8193_block_cpu_peel.csv`
- `profiling/nsight/ncu_generated8193_block_gpu_assign.csv`
