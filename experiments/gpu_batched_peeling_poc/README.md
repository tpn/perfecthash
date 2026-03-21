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
```

## Notes

- `--edges` is rounded up to a power of two.
- Vertex count is set to the next power of two above edge count, which for
  power-of-two edge counts means `vertices = 2 * edges`.
- The POC uses simple 64-bit mixing for seeded hashing; it is not wired to the
  PerfectHash hash-function table yet.
- The point here is to validate the execution model, not the exact CLI/plumbing.
