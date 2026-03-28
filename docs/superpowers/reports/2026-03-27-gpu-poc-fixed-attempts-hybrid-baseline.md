# GPU POC Fixed-Attempts And Hybrid Assignment Baseline

**Date:** 2026-03-27

## Scope

This report captures the first bounded results for two new POC features:

- controller-level fixed-attempt semantics
- hybrid GPU-peel / CPU-assign execution

## What Changed

The POC now supports:

- `--fixed-attempts <n>`
- `--first-solution-wins`
- `--assignment-backend <gpu|cpu>`

Key semantics:

- `batch` is still concurrent attempts
- `fixed_attempts` is now a total attempt budget across batches
- stopping happens at batch boundaries
- `actual_attempts_tried` may exceed `requested_fixed_attempts`
- `assignment_backend=cpu` means:
  - GPU build/peel stays active
  - only solved survivors are assigned/verified on CPU
  - GPU assign/verify are skipped

## Verification Commands

The following were rerun after the final backend fix:

```bash
cmake --build build/gpu-batched-peeling-poc -j2
ctest --test-dir build-cuda --output-on-failure -R 'perfecthash\.gpu\.poc\.geometry\.smoke'
```

Both passed.

## Fixed-Attempts Smoke

Tiny smoke:

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --edges 16 \
  --batch 3 \
  --fixed-attempts 5 \
  --output-format json
```

Result:

- `requested_fixed_attempts: 5`
- `actual_attempts_tried: 6`
- `batches_run: 2`
- `first_solution_wins: false`
- `first_solved_attempt: 2`

First-solution smoke:

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --edges 16 \
  --batch 3 \
  --fixed-attempts 100 \
  --first-solution-wins \
  --output-format json
```

Result:

- `requested_fixed_attempts: 100`
- `actual_attempts_tried: 3`
- `batches_run: 1`
- `first_solution_wins: true`
- `first_solved_attempt: 2`

So the controller semantics are behaving as intended.

## Hybrid Assignment Comparison

### Generated 8193, one batch

Command shape:

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --edges 8193 \
  --batch 128 \
  --threads 128 \
  --solve-mode device-serial \
  --device-serial-peel-geometry block \
  --assignment-backend {gpu|cpu} \
  --output-format json
```

Results:

| Backend | GPU ms | CPU ms | Solved | Mismatches | GPU peel ms | GPU assign ms | GPU verify ms | CPU assign ms | CPU verify ms |
|---------|--------|--------|--------|------------|-------------|---------------|---------------|---------------|---------------|
| `gpu` | `10.431` | `8.809` | `15` | `0` | `2.278` | `6.345` | `0.461` | `0.110` | `0.067` |
| `cpu` | `3.945`  | `0.265` | `15` | `0` | `2.517` | `0.000` | `0.000` | `0.169` | `0.064` |

Interpretation:

- Solve counts stayed identical.
- The CPU backend removed almost all of the post-peel device cost.
- On this generated solved case, hybrid is already much cheaper than the
  current scalar GPU assignment path.

### Generated 8193, fixed-attempt controller

Command shape:

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --edges 8193 \
  --batch 128 \
  --fixed-attempts 200 \
  --threads 128 \
  --solve-mode device-serial \
  --device-serial-peel-geometry block \
  --assignment-backend {gpu|cpu} \
  --output-format json
```

Results:

| Backend | Requested | Actual | Batches | Solved | GPU ms | CPU ms | GPU peel ms | GPU assign ms | GPU verify ms | CPU assign ms | CPU verify ms |
|---------|-----------|--------|---------|--------|--------|--------|-------------|---------------|---------------|---------------|---------------|
| `gpu` | `200` | `256` | `2` | `38` | `20.232` | `19.164` | `6.288` | `9.082` | `0.917` | `0.276` | `0.168` |
| `cpu` | `200` | `256` | `2` | `38` | `10.081` | `0.629` | `7.843` | `0.000` | `0.000` | `0.406` | `0.163` |

Interpretation:

- The batch-boundary overshoot logic worked as intended: `200 -> 256`.
- Solve counts stayed identical.
- With the CPU backend, the total non-peel work dropped sharply.
- The hybrid path is already materially better than the current scalar GPU
  assignment path on this case.

### HologramWorld-31016.keys, fixed-attempt controller

Command shape:

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --keys-file keys/HologramWorld-31016.keys \
  --hash-function Mulshrolate3RX \
  --batch 128 \
  --fixed-attempts 2048 \
  --threads 128 \
  --solve-mode device-serial \
  --device-serial-peel-geometry block \
  --assignment-backend {gpu|cpu} \
  --output-format json
```

Results:

| Backend | Requested | Actual | Batches | Solved | GPU ms | CPU ms | GPU peel ms | GPU assign ms | GPU verify ms | CPU assign ms | CPU verify ms |
|---------|-----------|--------|---------|--------|--------|--------|-------------|---------------|---------------|---------------|---------------|
| `gpu` | `2048` | `2048` | `16` | `1` | `485.840` | `659.399` | `375.614` | `17.937` | `29.753` | `0.048` | `0.020` |
| `cpu` | `2048` | `2048` | `16` | `1` | `421.490` | `0.157` | `378.038` | `0.000` | `0.000` | `0.084` | `0.020` |

Interpretation:

- Solve counts stayed identical.
- The hybrid path is still clearly better even though solve yield is only
  `1/2048`.
- The scalar GPU assignment/verify tail is already material even at that low
  yield.

### Hydrogen-40147.keys, fixed-attempt controller

Command shape:

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc \
  --keys-file /home/trent/src/perfecthash-keys/hard/Hydrogen-40147.keys \
  --hash-function Mulshrolate3RX \
  --batch 128 \
  --fixed-attempts 2048 \
  --threads 128 \
  --solve-mode device-serial \
  --device-serial-peel-geometry block \
  --assignment-backend {gpu|cpu} \
  --output-format json
```

Results:

| Backend | Requested | Actual | Batches | Solved | GPU ms | CPU ms | GPU peel ms | GPU assign ms | GPU verify ms | CPU assign ms | CPU verify ms |
|---------|-----------|--------|---------|--------|--------|--------|-------------|---------------|---------------|---------------|---------------|
| `gpu` | `2048` | `2048` | `16` | `26` | `1095.872` | `1034.561` | `539.958` | `412.507` | `45.871` | `3.652` | `0.916` |
| `cpu` | `2048` | `2048` | `16` | `26` | `694.154` | `6.627` | `591.964` | `0.000` | `0.000` | `4.356` | `0.935` |

Interpretation:

- Solve counts stayed identical.
- This is the clearest result so far:
  - the current scalar GPU assignment path is a major problem on a real solved
    case
  - hybrid GPU peel + CPU assignment is decisively better
- The remaining dominant cost in the hybrid path is still peel.

## Current Conclusion

The next work item should still target `block` peel with shared-memory/CUB
optimization, but the hybrid CPU assignment idea is now strongly supported.

The updated working recommendation is:

1. keep GPU build/peel batched
2. compact solved survivor ids
3. use CPU assignment/verify for survivors on GB10
4. only revisit GPU assignment after peel-layer metadata exists
5. use `Mulshrolate3RX` / `Mulshrolate4RX` for real-key hybrid exploration

## Relevant Commits

- `90b2785` `Add fixed-attempt control plane to GPU POC`
- `e77fed5` `Add hybrid CPU assignment backend to GPU POC`
- `27515f0` `Skip GPU assign verify in CPU backend`
