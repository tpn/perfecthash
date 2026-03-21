# PerfectHash GPU Solving TODO

## Done
- Established project ledgers for this investigation.
- Located the main CPU and CUDA graph-solving source files.
- Mapped the active CPU solve path (`GraphImpl1/2/3`) and the active CUDA backend (`GraphCu.c` + `PerfectHashCuda/Graph.cu`).
- Confirmed the current CUDA backend is hybrid and falls back to CPU for `Order[]`, assignment, and verification.
- Confirmed the older dynamic-parallel CUDA path is incomplete / not production-usable.
- Surveyed recent external work relevant to GPU MPHF / AMQ construction and lookup.
- Created branch `gpu-batched-peeling-poc`.
- Implemented a standalone batched GPU peeling/assignment proof-of-concept under `experiments/gpu_batched_peeling_poc/`.
- Built and ran the prototype successfully on the local GB10 system.
- Verified exact GPU/CPU agreement on solve success across multiple size/batch configurations.
- Added real `.keys` file loading to the prototype.
- Ran the prototype on `keys/HologramWorld-31016.keys`.
- Ran the prototype on `/home/trent/src/perfecthash-keys/sys32/Hydrogen-40147.keys`.
- Committed the initial POC checkpoint.
- Synced the branch to `nv1`, built there, and collected cross-machine Hologram/Hydrogen results.
- Refactored the POC to use C++ templates for 16-bit and 32-bit storage variants.
- Benchmarked serial 16-bit vs 32-bit HologramWorld runs locally and on `nv1`.

## In Progress
- Decide whether the next cut should stay standalone or start reusing PerfectHash hash routines and seed semantics.

## Next
- Decide whether to push the 16-bit idea further:
  - keep only the current light-touch downsizing, or
  - template more of the peel/update state, including `XorEdge`
- Replace the surrogate hash with one or more real PerfectHash hash functions:
  - `MultiplyShiftR`
  - `MultiplyShiftRX`
  - good `Mulshrolate*RX` family members
- Measure real-key throughput and solve rate on Hydrogen/HologramWorld across batch sizes.
- Measure solve throughput as a function of:
  - edges
  - batch size
  - threads per block
  - vertex/edge ratio
- Replace the host-driven peel-round loop with a more GPU-native synchronization strategy.
- Evaluate one-warp-per-graph and one-CTA-per-graph assignment kernels.
- Add richer instrumentation:
  - per-graph peel rounds
  - frontier sizes by round
  - unified-memory vs explicit device memory behavior
- Decide if the next integration target should be:
  - a new experimental PerfectHash component, or
  - continued iteration in `experiments/`

## Later
- Prototype with CCCL/CUB or `cuda.coop` primitives before investing in Tile / DKG kernel generation.
- Inspect remote `nv1` DKG repos later if a DSL-based kernel authoring path becomes attractive.
- Evaluate whether a PHOBIC/PTHash-style construction path would be a better long-term GPU target than CHM.
