# GPU Batched Peeling POC

This is a standalone proof-of-concept for the idea that matters most for
`PerfectHashCuda`: batch many graph construction attempts together, peel them on
the GPU in bulk-synchronous rounds, then assign and verify on the GPU per graph.

It does not try to reuse the existing PerfectHash CLI or `GraphCu` path.
Instead, it minimizes friction so the core execution model can be tested in
isolation.

## What It Models

- One fixed key set shared by all attempts.
- Many seed pairs, one per graph attempt.
- A 2-part graph with:
  - `edge_u[]`, `edge_v[]`
  - per-vertex `degree`
  - per-vertex XOR of incident edge ids
- GPU peel rounds:
  - collect degree-1 frontier vertices
  - peel edges once with edge-level CAS
  - update endpoint degree/XOR state atomically
- GPU assignment:
  - one block per graph
  - reverse peel order
  - assign owner vertex from the already-fixed opposite endpoint
- GPU verification:
  - check `(assigned[u] + assigned[v]) & edge_mask == edge_id`
- CPU reference:
  - sequential queue-based peel
  - reverse-order assignment
  - used to cross-check GPU success/failure behavior

## Build

```bash
cmake -S experiments/gpu_batched_peeling_poc -B build/gpu-batched-peeling-poc
cmake --build build/gpu-batched-peeling-poc -j
```

## Run

```bash
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc --edges 2048 --batch 256
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc --edges 4096 --batch 512 --threads 256
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc --keys-file keys/HologramWorld-31016.keys --batch 128
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc --keys-file /home/trent/src/perfecthash-keys/sys32/Hydrogen-40147.keys --batch 128
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc --keys-file keys/HologramWorld-31016.keys --batch 128 --storage-bits 16
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc --keys-file keys/HologramWorld-31016.keys --batch 128 --storage-bits 32
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc --keys-file keys/HologramWorld-31016.keys --batch 128 --storage-bits 16 --hash-function MultiplyShiftR
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc --keys-file keys/HologramWorld-31016.keys --seeds-file keys/HologramWorld-31016.MultiplyShiftR.seeds --batch 1 --storage-bits 16 --hash-function MultiplyShiftR
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc --keys-file keys/HologramWorld-31016.keys --batch 16 --output-format json --allocation-mode managed-prefetch-gpu
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc --edges 16 --batch 1 --threads 64 --solve-mode device-serial --assign-geometry warp --device-serial-peel-geometry warp --output-format json
./build/gpu-batched-peeling-poc/gpu_batched_peeling_poc --edges 16 --batch 8 --fixed-attempts 10000 --first-solution-wins --output-format json
```

## Notes

- `--edges` controls the number of generated logical keys.
- Edge capacity is rounded up to the next power of two from the logical key
  count; vertex count is then the next power of two above edge capacity, which
  for the current settings means `vertices = 2 * edge_capacity`.
- `--storage-bits` selects the templated storage variant:
  - `auto`: choose the smallest supported storage type for edges/vertices
  - `16`: force 16-bit edge/vertex/owner/order/assigned storage if supported
  - `32`: force the original 32-bit storage path
- `--hash-function` selects the templated hash family:
  - `SplitMix`
  - `MultiplyShiftR`
  - `MultiplyShiftRX`
  - `Mulshrolate1RX`
  - `Mulshrolate2RX`
  - `Mulshrolate3RX`
  - `Mulshrolate4RX`
- `--allocation-mode` selects how graph state is placed:
  - `explicit-device`: current CUDA device allocations with explicit host/device copies
  - `managed-default`: managed memory with default migration behavior
  - `managed-prefetch-gpu`: managed memory plus a prefetch pass for the hot graph-state buffers before solve
- `--fixed-attempts` and `--first-solution-wins` add a batch controller around the existing solve path:
  - `--fixed-attempts <n>` reruns full batches until at least `n` attempts have been tried
  - `--first-solution-wins` stops at the next batch boundary after any batch with one or more solved attempts
  - batch-boundary overshoot is expected and reported
- `--assign-geometry` is a stage-1 configuration/reporting field only; assignment still runs the scalar thread path for now:
  - default: `thread`
  - allowed values: `thread`, `warp`, `block`
- `--device-serial-peel-geometry` selects the real `device-serial` peel execution geometry:
  - default: `thread`
  - allowed values: `thread`, `warp`, `block`
- `--output-format json` emits a single machine-readable JSON object and suppresses the human summary on stdout.
  The JSON includes at least:
  - `dataset`
  - `batch`
  - `solve_mode`
  - `threads`
  - `assign_geometry`
  - `device_serial_peel_geometry`
  - `storage_bits`
  - `hash_function`
  - `allocation_mode`
  - `requested_fixed_attempts`
  - `actual_attempts_tried`
  - `batches_run`
  - `first_solution_wins`
  - `first_solved_attempt`
  - `gpu_ms`
  - `cpu_ms`
  - `cpu_stage_timings_ms_all_attempts.add_build`
  - `cpu_stage_timings_ms_all_attempts.peel`
  - `cpu_stage_timings_ms_all_attempts.assign`
  - `cpu_stage_timings_ms_all_attempts.verify`
  - `cpu_stage_timings_ms_solved_only.add_build`
  - `cpu_stage_timings_ms_solved_only.peel`
  - `cpu_stage_timings_ms_solved_only.assign`
  - `cpu_stage_timings_ms_solved_only.verify`
  - `solved`
  - `peel_rounds`
  - `stage_timings_ms.add_build`
  - `stage_timings_ms.peel`
  - `stage_timings_ms.assign`
  - `stage_timings_ms.verify`
- `cpu_ms` is the end-to-end wall time for the CPU reference batch.
- `cpu_stage_timings_ms_all_attempts` accumulates the CPU add/build, peel, assign, and verify stage timings across every graph attempt.
- `cpu_stage_timings_ms_solved_only` accumulates the same CPU stage timings, but only for graphs where the CPU reference both solved and verified the graph.
- `gpu_ms` is intended to be an end-to-end solve wall time for the GPU portion of the run.
  It includes mode-specific overhead such as managed-memory prefetch and host-roundtrip frontier synchronization.
- `stage_timings_ms` are narrower phase timings and may not sum exactly to `gpu_ms`.
- The CPU stage timings are additive aggregates, not wall-clock spans, so they may exceed `cpu_ms`.
- `first_solved_attempt` is a global attempt id, not a per-batch index.
- `device-serial` now reports a real four-way stage split by launching separate peel, assign, and verify stages.
- `--device-serial-peel-geometry` validation:
  - `warp` requires `--threads` to be a multiple of 32
  - `block` requires `--threads` to be at least 32
  - non-default values only apply when `--solve-mode device-serial` is selected
- Stage 1 assignment geometry remains reporting-only.
- Stage 1 `device-serial` peel geometry now changes actual kernel execution, but it still uses the existing global-memory degree/XOR model.
- `--seeds-file` lets you validate the port against a known-good seed set
  generated by the real PerfectHash solver.
- The POC uses simple 64-bit mixing for seeded hashing; it is not wired to the
  full PerfectHash hash-function table yet, but the main curated hash families
  above now use formulas and seed counts aligned to the CPU counterparts.
- The point here is to validate the execution model, not the exact CLI/plumbing.
- `mapped-pinned` / zero-copy style placement is not implemented in this first pass.
  The three supported allocation modes above are the practical ones on this machine.
