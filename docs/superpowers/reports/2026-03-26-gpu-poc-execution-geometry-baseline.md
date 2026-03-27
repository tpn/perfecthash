# GPU POC Execution Geometry Baseline

**Date:** 2026-03-26

## Scope

This report captures the first bounded local baseline for stage-1 GPU POC
execution geometry work.

What is included:

- Task 1: geometry option surface
- Task 3: real `device-serial` peel geometry variants
- Task 4: runner/test/smoke coverage for geometry flags

What is intentionally *not* included:

- real assignment geometry variants
- shared-memory staging
- CUB block primitives
- `ITEMS_PER_THREAD` tuning

## Important Caveat

Assignment geometry is still deferred.

The attempted Task 2 assignment split was reverted because ordered assignment is
currently a reverse-peel dependency chain:

- each assignment step reads `Assigned[other]`
- then writes `Assigned[owner]`
- so the current algorithm is loop-carried and not safely parallelized by
  simply giving a warp or block to a graph

Current branch state reflects that:

- `assign_geometry` is still surfaced in CLI/JSON for future work
- assignment execution remains scalar/thread-only
- `device_serial_peel_geometry` is the first geometry knob that now changes real
  execution behavior

## Commands

All runs below were executed locally on GB10 with sequential single-command
invocation to avoid oversubscription.

Generated baseline:

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --edges 8193 \
  --batch 128 \
  --threads 128 \
  --solve-mode device-serial \
  --assign-geometry thread \
  --device-serial-peel-geometry {thread|warp|block} \
  --output-format json
```

Real-key baseline:

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --keys-file keys/HologramWorld-31016.keys \
  --batch 16 \
  --threads 128 \
  --solve-mode device-serial \
  --assign-geometry thread \
  --device-serial-peel-geometry {thread|warp|block} \
  --output-format json
```

Verification that remained green during the work:

```bash
python -m unittest discover -s tests -p 'test_benchmark_gpu_solver.py' -v
ctest --test-dir build-cuda --output-on-failure -R 'perfecthash\.gpu\.poc\.geometry\.smoke|perfecthash\.cuda\.chm02'
```

## Results

### Generated 8193 (`batch=128`, `threads=128`)

| Peel geometry | GPU ms | Peel ms | Assign ms | Solved | Mismatches | Peel rounds |
|---------------|--------|---------|-----------|--------|------------|-------------|
| `thread` | `38.688` | `33.154` | `3.790` | `15` | `0` | `16` |
| `warp`   | `13.007` | `7.494`  | `3.804` | `15` | `0` | `29` |
| `block`  | `7.790`  | `2.245`  | `3.815` | `15` | `0` | `29` |

Observations:

- `block` is materially faster than `warp`, and both are much faster than the
  scalar `thread` peel.
- After peel improves, assignment becomes a much larger fraction of total GPU
  time.
- Correctness stayed intact across all three modes.

### HologramWorld-31016 (`batch=16`, `threads=128`)

| Peel geometry | GPU ms | Peel ms | Assign ms | Solved | Mismatches | Peel rounds |
|---------------|--------|---------|-----------|--------|------------|-------------|
| `thread` | `76.849` | `75.735` | `0.047` | `0` | `0` | `16` |
| `warp`   | `16.704` | `15.576` | `0.056` | `0` | `0` | `29` |
| `block`  | `5.156`  | `4.041`  | `0.059` | `0` | `0` | `29` |

Observations:

- Even on this harder real key set, `block` peel geometry is decisively better
  than `warp`, and both beat the scalar thread path by a large margin.
- The zero-solution result here does not reduce the value of the measurement:
  the comparison is still useful because all three modes processed the same
  attempts and stayed correctness-aligned with CPU.

## Interpretation

The first useful stage-1 conclusion is:

- **one-block-per-graph is the best current target for follow-on work**

Reasoning:

- It was the fastest mode on both the easy generated case and the harder
  HologramWorld case.
- The improvement came specifically from the peel phase, which is where the POC
  now has real geometry freedom.
- Once peel got cheaper, assignment became the next obvious bottleneck.

## Recommendation for Next Work

1. Keep assignment scalar for now.
2. Treat `block` peel as the current winning geometry.
3. Next optimization pass should target `block` peel with:
   - shared-memory control/state where useful
   - CUB/CCCL block primitives where they simplify local frontier handling
4. Revisit cooperative assignment only after the solver records enough metadata
   to support reverse-layer assignment safely.

## Relevant Commits

- `6efeffa` `Add GPU POC execution geometry options`
- `7e153a8` `Clarify GPU POC geometry wording`
- `a7e0be6` `Revert "Add GPU POC assignment geometry variants"`
- `a7377c9` `Add GPU POC device-serial peel geometry variants`
- `263b7e3` `Add GPU POC geometry benchmark coverage`
