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
- Wired the main curated hash families into the POC with compile-time selection.
- Validated the hash-family port against known-good offline `.seeds` files.
- Replaced SplitMix-derived candidate seeds with a Philox-based generator closer to solver precedent.
- Mirrored the `HashShift` override for non-AND-mask (`RX`) families.
- Matched CPU fixed-attempt yield for representative real-key cases using Philox and real hash families.
- Verified additional known-good HologramWorld CPU seed files in the POC.
- Verified a CPU-found Hydrogen `Mulshrolate3RX` seed set in the POC.
- Added solved-seed dumping to the POC for mismatch analysis.
- Confirmed an easy-case random-yield discrepancy on `CoreUIComponents-8193.keys`.
- Resolved the CoreUI discrepancy as a CPU concurrency-semantic issue; with `MaximumConcurrency=1`, CPU and POC fixed-attempt yield match.
- Added a device-resident per-graph solve mode and benchmarked it against the host-roundtrip baseline.
- Added CUDA memory estimation, headroom enforcement, and auto batch scaling.
- Added a `Chm02` CUDA repro harness and identified that the default Linux builds were not actually exercising CUDA.
- Created a CUDA-enabled build tree and pushed the single-graph `Graph.cu` path through add-keys, GPU acyclic detection, and CPU assignment oracle.
- Fixed the `NoFileIo()` table-data copy bug in `Chm02.c`.
- Fixed the no-spare-graph trap in `GraphRegisterSolved16NoBestCoverage()` for the narrow single-GPU fixed-attempt bring-up path.
- Converted the `Chm02` CUDA harness from failing repro to passing regression for the no-file-I/O bring-up case.
- Replaced the `GraphCuAssign()` CPU fallback with a real GPU assignment stage for the known-good HologramWorld `Chm02` regression.
- Added a regression mode that requires a GPU assignment log and fails if the old CPU assign fallback is still being used.

## In Progress
- Continue the `Graph.cu` / `Graph.cuh` single-graph correctness bring-up in the CUDA-enabled build.

## Next
- Validate the new GPU assignment path on a 32-bit / non-`Assigned16` known-good seed case.
- Stabilize the `build-cuda` `PerfectHashCreate ... Chm02 ...` path without relying on `--NoFileIo --DisableCsvOutputFile`.
- Determine why the GPU peel order diverges from the CPU oracle order on the known-good HologramWorld `Mulshrolate3RX` seed.
- Decide whether the order divergence is acceptable (valid topological order) or indicative of a remaining bug in the serial peel kernel.
- Decide whether CPU `Verify()` should remain the long-lived oracle for this legacy path or whether a minimal GPU verify stage is worth adding.
- Decide whether to push the 16-bit idea further:
  - keep only the current light-touch downsizing, or
  - template more of the peel/update state, including `XorEdge`
- Improve random attempt quality for real hash families:
  - add weighted seed-mask count behavior analogous to `GraphApplyWeightedSeedMasks()`
  - compare Philox-only yield against actual CLI / library solve yield
  - consider a mutation step around known-good seed shapes instead of full random generation
- Consider measuring actual `PerfectHashCreate` attempt yield for one representative case so the POC has a concrete target.
- Compare winning subsequence / seed material against CPU CSV for a case with non-zero solutions.
- Compare POC random-yield vs CPU random-yield for Hydrogen `Mulshrolate3RX`, where CPU definitely finds multiple solutions under Philox.
- Replay dumped POC-success seeds against the real CLI until the first true mismatch is identified.
- Sync the latest equivalence findings to `nv1` if cross-machine confirmation is still useful.
- Measure real-key throughput and solve rate on Hydrogen/HologramWorld across batch sizes.
- Compare `host-roundtrip` vs `device-serial` on hard cases with clean serial measurements.
- Measure solve throughput as a function of:
  - edges
  - batch size
  - threads per block
  - vertex/edge ratio
- Revisit the deferred direction:
  - cooperative-groups / global frontier device-side convergence
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
