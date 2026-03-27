# PerfectHash GPU Solving Notes

## Session Context
- Focus: GPU solving of perfect hash graphs, especially whether batching many solve requests is the viable path versus parallelizing a single branchy graph solve.
- Relevant local areas:
  - `src/PerfectHash/GraphImpl1.c`
  - `src/PerfectHash/GraphImpl2.c`
  - `src/PerfectHash/GraphImpl3.c`
  - `src/PerfectHashCuda/Graph.cu`
  - `src/PerfectHashCuda/GraphThrust.cu`
  - `src/PerfectHashCuda/Graph4.cu`
- Related local references mentioned by user:
  - `~/src/tile-interop`
  - `~/src/dkg`
  - `~/src/*dkg*`

## Initial Findings
- `GraphImpl1` exposes the classic API surface: `GraphAddEdge()`, `IsGraphAcyclic()`, `GraphAssign()`.
- `GraphImpl3` appears to be the more compact/optimized CPU representation with direct degree manipulation and a serial peeling/assignment style.
- CUDA implementation is split between:
  - `Graph.cu`: lower-level kernels and graph primitives.
  - `GraphThrust.cu`: higher-level solve loop, seed management, and host/device orchestration.
  - `Graph4.cu`: thin entry/registration layer.
- The user’s intuition about batching is plausible and should be evaluated against the current single-graph execution model.

## Current Implementation Understanding
- The main CHM/PerfectHash solve path is still fundamentally a 2-part graph, not a 3-uniform hypergraph.
- `GraphImpl1` is explicitly hostile to SIMT:
  - adjacency is stored via `First[]`, `Next[]`, `Edges[]`
  - acyclicity is found by recursive/chasing deletion from degree-1 vertices
  - assignment is recursive DFS over neighbors
- `GraphImpl3` is the important CPU baseline for any GPU work:
  - graph state is compressed into `VERTEX3 { Degree, Edges }`, where `Edges` is the XOR of incident edge IDs
  - each edge stores only the two endpoint vertices (`EDGE3`)
  - acyclicity is a two-phase peel over `Order[]`
  - assignment is a reverse traversal of `Order[]`, choosing the first unvisited endpoint
- The active CUDA backend is a hybrid, not an end-to-end GPU solver:
  - `src/PerfectHash/GraphCu.c` is the host-facing backend wired into the main library
  - GPU side performs hashing, builds the `Vertices3`/`Edges3` style graph, and runs an acyclicity test
  - if the GPU reports success, the code rebuilds the same graph on `Graph->CpuGraph`, reruns CPU `AddKeys()` and CPU `IsAcyclic()`, copies `Order[]`, then executes CPU `Assign()` and CPU `Verify()`
  - consequence: current code never solves assignment on device, and it does not preserve a GPU-generated peel order
- The older dynamic-parallel path under `src/PerfectHashCuda/GraphThrust.cu` and `Graph4.cu` is incomplete / effectively dead:
  - `GraphCuReset()` returns early before most reset logic
  - `GraphCuIsAcyclic()` is stubbed
  - `GraphCuAssignKernel()` is empty
  - `GraphCuSolve()` has an early `goto End` that bypasses the disabled downstream work
  - `Graph4.cu` disables the actual solve body with `#if 0`

## CUDA Backend Bottlenecks
- Current device-side graph mutation uses lock-and-retry behavior:
  - per-vertex locks via `CuVertexLocks`
  - per-edge locks via `CuEdgeLocks`
  - `TryLock()` loops in both add and remove paths
- The GPU acyclicity test repeatedly launches the same peel kernel, copies back `OrderIndex`, synchronizes the stream, checks progress on the host, and relaunches if necessary.
- The current GPU peel path does not materialize a usable peel order per edge; it only decrements a shared `OrderIndex`.
- Managed memory is used for the core solver arrays (`Order`, `Assigned`, `Vertices3`, `VertexPairs`, lock arrays), while host-side CPU fallback also touches related state. On Grace/Blackwell unified memory this is less catastrophic than on PCIe boxes, but the mixed CPU/GPU ownership still encourages synchronization and migration costs.
- `GraphCuHashKeysKernel()` only checks duplicate vertex-pair collisions within a coalesced group / warp; impossible graphs are still mainly rejected later by acyclicity logic.

## External Research Findings
- PHOBIC (ESA 2024) is the most relevant recent “GPU-construction” result in the MPHF space:
  - GPU construction parallelizes over partitions, seeds, and keys
  - it becomes worthwhile only for sufficiently large input size and sufficiently large average bucket size
  - this supports the user’s batching intuition: the GPU wants many independent work items, not a single tiny branchy graph
- GPU-accelerated BDZ / xor-filter work is directly relevant to the peeling problem:
  - a correct parallel peeling scheme uses subtables / partitions so each peeled edge is “owned” by exactly one partition in a subround
  - assignment can then be made parallel by processing peel layers in reverse and storing the unique degree-1 vertex that marked each edge
  - profiling in that work showed assignment and synchronization overhead becoming dominant, especially when the number of peel layers is high
- Recent GPU AMQ work (for example Cuckoo-GPU, 2026) suggests a useful design principle:
  - random memory access is acceptable on modern GPUs if divergence is bounded and sequential critical sections are short
  - lock-free CAS plus bounded search / eviction beats lock-heavy fine-grained retry loops
- GPH (SIGMOD 2025) is more about lookup than construction, but it is relevant to the long-term “compete with cuCollections” goal:
  - it argues CPU perfect-hash tables do not map cleanly to GPUs
  - it uses GPU-specific lookup grouping and bucket-request scheduling to maximize active warps and memory efficiency
- cuCollections is a useful reference point for build strategy:
  - the project centers on bulk APIs and GPU-native concurrent structures
  - `cuco::bloom_filter` uses a blocked Bloom filter design, which is exactly the kind of locality-aware reformulation that tends to work well on GPUs

## Design Direction
- Short-term viable direction if CHM01 semantics must be preserved:
  - batch many independent graph attempts together
  - keep the `GraphImpl3` XOR+degree representation
  - replace lock-based peel with bulk-synchronous frontier processing
  - explicitly store per-edge peel layer and “owner endpoint” metadata so assignment can also stay on device
- Higher-upside direction if format / construction algorithm can change:
  - move away from CHM-style single-table peeling toward partitioned bucket+pilot methods such as PHOBIC/PTHash-style families
  - these algorithms already align much better with GPU parallelism over partitions/buckets/seeds
  - this would also make it easier to design a GPU-native solved-table representation instead of retrofitting CHM

## Recommended Next Prototype
1. Treat the unit of parallelism as “many attempts at once”, not “many threads solving one graph”.
2. Keep one graph attempt per warp or CTA for small/medium tables; use batches large enough to saturate the GPU.
3. Replace per-vertex/per-edge locks with layer/frontier kernels:
   - kernel A: hash keys for all graphs in the batch
   - kernel B: build degrees / XOR edge accumulators
   - kernel C: identify degree-1 frontier and emit `(edge, owner_vertex, layer)` tuples
   - kernel D: peel/update neighboring degrees
   - repeat C/D until no progress
   - kernel E: reverse-layer assignment using stored owner vertices
4. Use CUB / CCCL primitives for compaction, scans, histograms, radix sort, and per-layer grouping instead of custom pointer-chasing control flow.
5. Defer Tile / DKG / DSL work until the algorithm is stabilized; the current bottleneck is execution model, not syntax for writing kernels.

## Prototype Status
- Working branch: `gpu-batched-peeling-poc`
- Standalone POC location:
  - `experiments/gpu_batched_peeling_poc/CMakeLists.txt`
  - `experiments/gpu_batched_peeling_poc/main.cu`
  - `experiments/gpu_batched_peeling_poc/README.md`
- Prototype scope:
  - independent of existing PerfectHash CLI and `GraphCu`
  - same generated key set shared across all attempts
  - now also supports loading real 32-bit `.keys` files directly
  - one seeded graph attempt per batch element
  - GPU build, peel, assign, and verify
  - CPU queue-based reference in the same binary for correctness comparison
- Prototype representation:
  - flattened `edge_u[]` / `edge_v[]`
  - per-vertex `degree[]`
  - per-vertex XOR accumulator of incident edges
  - `owner_vertex[edge]`
  - `peel_order[graph][position]`
- Prototype execution model:
  - build all graphs in parallel on GPU
  - host-driven bulk-synchronous peel rounds:
    - collect degree-1 frontier vertices
    - peel via edge-level CAS
    - record peel order and owner vertex
  - GPU assignment runs one block per graph, with one thread walking reverse peel order
  - GPU verification checks `(assigned[u] + assigned[v]) & edge_mask == edge_id`
  - storage model is now templated for 16-bit vs 32-bit variants

## Prototype Results
- Build command:
  - `cmake -S experiments/gpu_batched_peeling_poc -B build/gpu-batched-peeling-poc`
  - `cmake --build build/gpu-batched-peeling-poc -j`
- Runs completed successfully on the local CUDA 13.1 / NVIDIA GB10 system.
- Observed results:
  - `--edges 2048 --batch 256`
    - peel rounds: 47
    - GPU success: `42/256`
    - CPU success: `42/256`
    - success mismatches: `0`
    - GPU time: `3.080 ms`
    - CPU time: `5.145 ms`
  - `--edges 4096 --batch 512`
    - peel rounds: 81
    - GPU success: `63/512`
    - CPU success: `63/512`
    - success mismatches: `0`
    - GPU time: `13.089 ms`
    - CPU time: `20.644 ms`
  - `--edges 1024 --batch 1024`
    - peel rounds: 66
    - GPU success: `177/1024`
    - CPU success: `177/1024`
    - success mismatches: `0`
    - GPU time: `4.879 ms`
    - CPU time: `10.104 ms`
  - `--edges 8192 --batch 256`
    - peel rounds: 85
    - GPU success: `34/256`
    - CPU success: `34/256`
    - success mismatches: `0`
    - GPU time: `14.247 ms`
    - CPU time: `22.058 ms`
  - `--keys-file keys/HologramWorld-31016.keys --batch 128`
    - actual keys: `31,016`
    - edge capacity: `32,768`
    - vertices: `65,536`
    - peel rounds: `96`
    - GPU success: `26/128`
    - CPU success: `26/128`
    - success mismatches: `0`
    - GPU time: `55.471 ms`
    - CPU time: `47.300 ms`
  - `--keys-file /home/trent/src/perfecthash-keys/sys32/Hydrogen-40147.keys --batch 64`
    - actual keys: `40,147`
    - edge capacity: `65,536`
    - vertices: `131,072`
    - peel rounds: `13`
    - GPU success: `35/64`
    - CPU success: `35/64`
    - success mismatches: `0`
    - GPU time: `48.020 ms`
    - CPU time: `45.235 ms`
  - `--keys-file /home/trent/src/perfecthash-keys/sys32/Hydrogen-40147.keys --batch 128`
    - actual keys: `40,147`
    - edge capacity: `65,536`
    - vertices: `131,072`
    - peel rounds: `13`
    - GPU success: `78/128`
    - CPU success: `78/128`
    - success mismatches: `0`
    - GPU time: `66.623 ms`
    - CPU time: `95.456 ms`

## Cross-Machine Results
- Local machine (`NVIDIA GB10`, CUDA 13.1):
  - `HologramWorld-31016.keys`, batch `128`: GPU `55.471 ms`, CPU `47.300 ms`
  - `Hydrogen-40147.keys`, batch `128`: GPU `68.038 ms`, CPU `92.163 ms`
- `nv1` (`NVIDIA RTX PRO 6000 Blackwell Workstation Edition`, CUDA 13.1):
  - `HologramWorld-31016.keys`, batch `128`: GPU `20.761 ms`, CPU `58.231 ms`
  - `Hydrogen-40147.keys`, batch `128`: GPU `78.354 ms`, CPU `122.843 ms`
  - `Hydrogen-40147.keys`, batch `256`: GPU `93.918 ms`, CPU `240.248 ms`
- Current inference:
  - the SplitMix-based POC scales to `nv1` cleanly with exact GPU/CPU agreement
  - HologramWorld benefits much more from the workstation GPU/CPU combination than the current local box
  - Hydrogen remains GPU-favorable on both machines once batch size is large enough
  - the remaining dominant cost is still the host-driven peel-round loop, not correctness

## What The POC Proves
- The batched execution model is viable.
- We do not need per-vertex/per-edge lock objects plus retry loops to peel many attempts in parallel.
- Capturing explicit peel order and owner-vertex metadata is enough to support assignment after peeling.
- Even with a conservative host-driven peel-round loop and sequential-per-graph assignment, batching across many graphs already yields a useful GPU advantage.
- We are already close to real PerfectHash datasets:
  - `HologramWorld-31016.keys` runs today
  - `Hydrogen-40147.keys` from the external `perfecthash-keys` repo runs today
  - the remaining distance to “try Hydrogen” is basically operational, not architectural

## 16-bit Templating Results
- The standalone POC now uses C++ templates to select 16-bit or 32-bit storage for:
  - `EdgeU`
  - `EdgeV`
  - `OwnerVertex`
  - `PeelOrder`
  - `Assigned`
- The atomic-heavy arrays are still 32-bit for now:
  - `Degree`
  - `XorEdge`
  - `EdgePeeled`
  - per-graph counters / verification counters
- This is intentionally the lowest-risk analogue of the library’s `assigned16` idea:
  - reduce memory footprint where it is easy and safe
  - avoid immediately rewriting the peel/update atomics around packed 16-bit state
- Serial benchmark results for `HologramWorld-31016.keys`, batch `128`:
  - local machine:
    - 32-bit storage: GPU `38.958 ms`, CPU `47.473 ms`
    - 16-bit storage: GPU `35.812 ms`, CPU `46.817 ms`
    - approximate GPU improvement from templated 16-bit storage: `~8.1%`
  - `nv1`:
    - 32-bit storage: GPU `20.218 ms`, CPU `56.878 ms`
    - 16-bit storage: GPU `18.913 ms`, CPU `54.885 ms`
    - approximate GPU improvement from templated 16-bit storage: `~6.5%`
- Interpretation:
  - downsizing helps, but not dramatically yet
  - this is consistent with the current bottleneck still being the host-driven peel-round loop rather than raw assigned-array traffic alone
  - a larger benefit likely requires either:
    - more state downsizing, including peel/update structures, or
    - removing the host round-trip from the inner solve loop

## Real Hash-Family Wiring
- The POC now supports templated hash-family selection via `--hash-function`.
- Implemented families:
  - `SplitMix`
  - `MultiplyShiftR`
  - `MultiplyShiftRX`
  - `Mulshrolate1RX`
  - `Mulshrolate2RX`
  - `Mulshrolate3RX`
  - `Mulshrolate4RX`
- The intent is to stay close to the CPU counterparts:
  - same seed counts as the PerfectHash hash-function table
  - same `Seed3` masks for shift-byte fields
  - same formula structure as `PerfectHashTableHashEx.c` / `PerfectHashTableHashExCpp.hpp`
  - host-side selection, compile-time kernel instantiation, no runtime dispatch in the hot path
- Added `--seeds-file` support so the POC can validate against known-good offline seed sets generated by the real solver.

## Hash Port Validation
- Known-good offline seed files validate cleanly in the POC:
  - `keys/HologramWorld-31016.MultiplyShiftR.seeds` with `MultiplyShiftR`, batch `1`, storage `16`
    - GPU success `1/1`
    - CPU success `1/1`
    - mismatches `0`
  - `keys/HologramWorld-31016.Mulshrolate3RX.seeds` with `Mulshrolate3RX`, batch `1`, storage `16`
    - GPU success `1/1`
    - CPU success `1/1`
    - mismatches `0`
  - `tests/data/HologramWorld-31016.Mulshrolate1RX.seeds` with `Mulshrolate1RX`, batch `1`, storage `32`
    - GPU success `1/1`
    - CPU success `1/1`
    - mismatches `0`
- Current inference:
  - the formula port is correct for the tested families
  - the earlier `0/128` random-attempt runs for some real hash families are therefore most likely a seed-generation / seed-yield issue, not a bad hash implementation
  - this suggests the next fidelity step is to improve how the POC generates or mutates candidate seed sets, not to rework the formulas again

## Philox Seed Generation
- The POC now uses a Philox 4x32 10-round seed generator for candidate graph seeds, aligned with repo precedent:
  - non-zero 32-bit words are generated from Philox output
  - per-graph subsequence is derived from a base subsequence plus the graph index
  - a base offset is supported
- The POC also now mirrors the non-AND-mask behavior for `RX` families:
  - `Seed3.Byte1` is forced to `HashShift`
  - then seed masks are applied / preserved
- This is closer to the real solver than the earlier SplitMix-derived seed words.

## Philox Yield Results
- Even after switching to Philox and `HashShift` override, random yield for real hash families remains very low in the POC:
  - `HologramWorld-31016.keys`, batch `128`, `MultiplyShiftR`, storage `16`
    - GPU success `0/128`
    - CPU success `0/128`
  - `HologramWorld-31016.keys`, batch `128`, `Mulshrolate3RX`, storage `16`
    - GPU success `0/128`
    - CPU success `0/128`
  - `Hydrogen-40147.keys`, batch `128`, `MultiplyShiftR`, storage `32`
    - GPU success `0/128`
    - CPU success `0/128`
  - `HologramWorld-31016.keys`, batch `2048`, `MultiplyShiftR`, storage `16`
    - GPU success `0/2048`
    - CPU success `0/2048`
  - `HologramWorld-31016.keys`, batch `2048`, `Mulshrolate3RX`, storage `16`
    - GPU success `1/2048`
    - CPU success `1/2048`
- Current interpretation:
  - the ported hashes are correct
  - Philox alone was not enough to reproduce the practical seed yield of the full solver
  - the next likely missing piece is seed-shaping behavior:
    - weighted seed-mask counts
    - better constrained seed-byte mutation
    - or additional solver heuristics around candidate seed selection
- Cross-machine confirmation:
  - `nv1`, `HologramWorld-31016.keys`, batch `2048`, `MultiplyShiftR`, storage `16`
    - GPU success `0/2048`
    - CPU success `0/2048`
    - GPU time `113.163 ms`
    - CPU time `455.704 ms`
  - This matches the local yield result and reinforces that the remaining issue is algorithmic seed shaping, not machine-specific behavior.

## CPU-Equivalent Yield Checks
- The strongest current equivalence signal is no longer just “known-good seeds solve”, but that fixed-attempt yield matches the real CPU solver on tested cases:
  - `HologramWorld-31016.keys`, `MultiplyShiftR`, Philox, fixed/batch `2048`
    - CPU solver: `0/2048`
    - POC: `0/2048`
  - `HologramWorld-31016.keys`, `Mulshrolate3RX`, Philox, fixed/batch `2048`
    - CPU solver: `1/2048`
    - POC: `1/2048`
  - `Hydrogen-40147.keys`, `MultiplyShiftR`, Philox, fixed/batch `128`
    - CPU solver: `0/128`
    - POC: `0/128`
- This does not prove full equivalence yet, but it does show the standalone GPU+CPU reference path is now matching the library’s fixed-attempt yield behavior for representative real-key cases and real hash families.

## Known-Good Seed Validation
- The confusion about “not finding a solution” is now resolved:
  - the POC does find solutions when given known-good CPU seed sets
  - the earlier failures were about random seed yield, not inability to solve a valid graph
- Confirmed known-good CPU seed sets that solve in the POC:
  - `HologramWorld-31016.MultiplyShiftR.seeds`
  - `HologramWorld-31016.MultiplyShiftRX.seeds`
  - `HologramWorld-31016.Mulshrolate3RX.seeds`
  - `HologramWorld-31016.Mulshrolate4RX.seeds`
  - `tests/data/HologramWorld-31016.Mulshrolate1RX.seeds`
- Confirmed Hydrogen CPU-found seed set:
  - extracted a winning `Mulshrolate3RX` seed set from `PerfectHashCreate` CSV for `/home/trent/src/perfecthash-keys/sys32/Hydrogen-40147.keys`
  - the POC solved it successfully with GPU `1/1`, CPU `1/1`, mismatches `0`
- This is the strongest current evidence that the standalone graph build/peel/assign/verify path is functionally equivalent to CPU for tested seed sets.

## Easier-Key Findings
- Testing easier fixtures was helpful.
- Example: `/home/trent/src/perfecthash-keys/sys32/CoreUIComponents-8193.keys`
  - key-to-edge-capacity ratio: `8193 / 16384 ~= 0.50006`
  - `MultiplyShiftR`, Philox, batch/fixed `2048`, 16-bit storage in the POC:
    - GPU success `164/2048`
    - internal CPU reference success `164/2048`
  - `PerfectHashCreate` CSV for the same case:
    - `NumberOfSolutionsFound=162`
    - `SolutionsFoundRatio=0.07906`
- Interpretation:
  - easier fixtures do raise real-hash yield substantially, as expected
  - the original `164 vs 162` discrepancy on CoreUI turned out to be a concurrency-accounting issue in the comparison, not a solver mismatch
  - when `PerfectHashCreate` is forced to `MaximumConcurrency=1`, the CPU result becomes `164/2048`, matching the POC exactly

## Concurrency Semantics
- `PerfectHashCreate` with default CPU concurrency can yield slightly different aggregate fixed-attempt solution counts than a simple “one batch slot == one attempt” model, even under Philox.
- When comparing the standalone POC against the CPU solver, the cleanest apples-to-apples baseline is:
  - `MaximumConcurrency=1`
  - same Philox seed / subsequence / offset
  - same fixed-attempt budget
- With that baseline:
  - `CoreUIComponents-8193.keys`, `MultiplyShiftR`, Philox, fixed/batch `2048`
    - CPU (`MaximumConcurrency=1`): `164/2048`
    - POC: `164/2048`
  - `HologramWorld-31016.keys`, `MultiplyShiftR`, Philox, fixed/batch `2048`
    - CPU: `0/2048`
    - POC: `0/2048`
  - `Hydrogen-40147.keys`, `Mulshrolate3RX`, Philox, fixed/batch `2048`
    - CPU (`MaximumConcurrency=1`): `26/2048`
    - POC: `26/2048`
- This is currently the strongest statement of equivalence:
  - known-good CPU seeds solve correctly
  - fixed-attempt yield matches CPU for tested cases when concurrency semantics are aligned

## Chosen Direction
- Chosen now:
  - device-resident per-graph solve loop (`device-serial`)
  - one block per graph
  - local convergence on device
- Deferred for later:
  - cooperative-groups / global frontier device-side convergence across the full batch
  - I am not taking this path yet so it remains easy to revisit later

## Device-Resident Solve Mode
- The POC now supports two solve modes:
  - `host-roundtrip`
  - `device-serial`
- `device-serial` keeps the same graph state and verification semantics, but performs the peel/assign/verify loop entirely inside a device kernel, one block per graph.
- Clean local comparison on the easy case:
  - `CoreUIComponents-8193.keys`
  - `MultiplyShiftR`, Philox, batch `2048`, storage `16`
  - `host-roundtrip`
    - GPU success `164/2048`
    - CPU success `164/2048`
    - GPU time `242.481 ms`
  - `device-serial`
    - GPU success `164/2048`
    - CPU success `164/2048`
    - GPU time `217.927 ms`
- Interpretation:
  - the chosen direction is functionally correct
  - it removes the host frontier-count round trip
  - it is only a stepping stone, not the final high-performance design, because each graph’s inner convergence loop is still serialized to one thread/block

## Memory Guarding
- The POC now estimates required device bytes before large allocations.
- Inputs to the estimate:
  - logical key count
  - vertex count
  - batch size
  - storage width (`16` vs `32`)
  - solve mode (`host-roundtrip` vs `device-serial`)
- It queries CUDA free/total memory and enforces configurable headroom:
  - `--memory-headroom-pct` (default `10`)
- It also detects unified-like device characteristics from the CUDA runtime/device attributes and prints a summary before the run.
- If the requested batch does not fit with headroom:
  - default behavior is to auto-scale batch down
  - `--disable-auto-batch-scale` converts that into a hard failure
- The estimator is solve-mode aware:
  - `host-roundtrip` includes the large frontier buffer
  - `device-serial` omits that buffer
- Example summary on the local GB10:
  - free: `113.68 GiB`
  - total: `121.69 GiB`
  - unified-like: `Y`
  - estimated bytes for `CoreUIComponents-8193`, batch `2048`, `device-serial`, storage `16`: `832.16 MiB`

## PerfectHashCuda Review
- Why the historical `PerfectHashCuda` work struggled with parity:
  - the active `GraphCu` path was always hybrid:
    - GPU acyclicity result was followed by CPU `AddKeys()`, CPU `IsAcyclic()`, CPU `Assign()`, and CPU `Verify()`
    - this meant the GPU path never actually owned the full solve/assign/verify pipeline
  - the older standalone `PerfectHashCuda/GraphThrust.cu` path was incomplete:
    - reset returns early
    - acyclicity is stubbed
    - assignment kernel is empty
    - solve body exits before the disabled downstream stages
  - the lock-based concurrent peel in `PerfectHashCuda/Graph.cu` had multiple correctness hazards:
    - `GraphCuRemoveVertex()` reads `Degree`/`Edges` before acquiring locks, then never revalidates that the chosen edge is still current
    - the code decrements `OrderIndex` but does not record the peeled edge into `Order[]`
    - one add-edge helper increments degree twice
- In other words:
  - parity trouble was not just “GPU graphs are hard”
  - some of the historical GPU code paths were structurally incomplete
  - and parts of the concurrency logic that did exist were not robust enough to preserve the exact CPU semantics

## Graph.cu / Chm02 Bring-Up
- Important discovery:
  - the existing Linux builds had `PERFECTHASH_USE_CUDA=OFF`
  - `Chm02` therefore hit the `E_NOTIMPL` stub in `PerfectHashConstants.c` instead of the real CUDA path
- A separate CUDA-enabled build tree (`build-cuda/`) is required to exercise `Graph.cu` through `PerfectHashCreate`.
- With that build:
  - `Chm02` + `Mulshrolate3RX` + known-good seeds + `--CuConcurrency=1` now reaches:
    - `LoadInfo()`
    - `Reset()`
    - `LoadNewSeeds()`
    - `GraphCuAddKeys()`
    - `GraphCuIsAcyclic()`
  - the path no longer dies at the old `E_NOTIMPL` stub
- Immediate fixes applied:
  - fixed the double degree increment in `GraphCuAddEdge1()`
  - re-enabled GPU `Order[]` capture in `GraphCuRemoveVertex()`
  - changed `GraphCuIsAcyclic()` to use a serial device-side peel kernel for correctness bring-up
  - changed `GraphCuIsAcyclic()` / `GraphCuAssign()` bring-up path to feed GPU-generated `Order[]` into the CPU assignment oracle instead of rerunning CPU `IsAcyclic()`
- Current observed state for the known-good HologramWorld `Mulshrolate3RX` seed:
  - GPU add-keys succeeds
  - GPU acyclic stage reports success and now sets host `DeletedEdgeCount` / `OrderIndex`
  - CPU assignment oracle is entered and returns success
  - the GPU `Order[]` still differs from the CPU oracle order at index 0, but appears to be a valid peel order for assignment
- Remaining blocker:
  - the `PerfectHashCreate` CLI path is still not clean end-to-end in the CUDA-enabled build when normal CSV / file-output behavior is enabled
  - however, the narrow bring-up target now works cleanly with:
    - `--NoFileIo`
    - `--DisableCsvOutputFile`
    - `MaximumConcurrency=1`
    - `--CuConcurrency=1`
    - known-good seed set
- Chosen direction:
  - continue debugging/fixing the single-graph `Graph.cu` / `Graph.cuh` path through `build-cuda`
- Deferred direction:
  - batching / POC work inside `Graph.cu`
  - cooperative-groups global frontier rewrite
  - full performance tuning

## Current Working Regression
- CUDA-enabled build tree: `build-cuda/`
- Stable single-graph CUDA regression:
  - `PerfectHashCreate`
  - `Chm02`
  - `Mulshrolate3RX`
  - `And`
  - maximum concurrency `1`
  - `--CuConcurrency=1`
  - `--FixedAttempts=2`
  - known-good HologramWorld seed set
  - `--NoFileIo --DisableCsvOutputFile`
- Local harness:
  - `tests/run_cli_chm02_cuda_known_seed_test.cmake`
- Current scope of success:
  - single-graph CUDA add-keys + GPU acyclic detection + GPU assignment path succeeds
  - CPU `Verify()` still succeeds after consuming the GPU-produced `Assigned[]`
  - this is a correctness bring-up checkpoint, not a full production-ready CUDA integration

## Current Single-Graph CUDA Assignment Status
- The narrow `Graph.cu` / `Chm02` bring-up path now owns assignment on the GPU as well as peel/order.
- The implemented GPU assignment is intentionally conservative:
  - one serial CUDA kernel
  - one graph
  - no batching
  - no attempt yet at performance
- It mirrors the CPU `GraphImpl3` logic directly:
  - iterate `Order[]` from `OrderIndex`
  - choose the unvisited endpoint as the owner vertex
  - compute the owner assignment modulo `NumberOfEdges`
  - mark both vertices visited in `VisitedVerticesBitmap`
- Ordered-index semantics are still what is being validated:
  - GPU produces `Assigned[]`
  - `GraphCuAssign()` copies that `Assigned[]` into `CpuGraph`
  - CPU `Verify()` confirms the resulting table is order-preserving
- Chosen direction:
  - keep CPU `Verify()` as the oracle while the single-graph GPU assignment path stabilizes
- Deferred direction:
  - porting verification itself onto the GPU
  - revisiting batching/performance work inside `Graph.cu`
  - cooperative-groups/global-frontier variants for this legacy path

## Linux Compat Findings
- On Linux, the relevant file-I/O path for `Chm02` is `Chm02Compat.c`, not the Windows-style `Chm02.c` threadpool/file-work path.
- The full file-I/O crash was not a solver bug:
  - GPU add-keys, GPU peel/order, GPU assignment, and CPU verify were already succeeding
  - the crash happened afterwards on a BS threadpool file-work callback
- Root cause:
  - `Chm02Compat.c` submitted `FILE_WORK_ITEM`s directly to `FileWorkItemCallbackChm01`
  - unlike `Chm01Compat.c`, it did not populate `Item->Context`
  - `FileWorkItemCallbackChm01()` immediately dereferenced `Item->Context`, causing a null-context segfault
- Minimal durable fix:
  - set `Verb##Name.Context = Context` in both compat file-work submission macros
  - keep a non-Windows `FileWorkCallbackChm01()` wrapper in `Chm01FileWork.c` / `Chm01FileWorkStub.c`

## 32-bit Coverage
- The single-graph CUDA path is now validated on both sides of the `Assigned16` boundary:
  - `HologramWorld-31016.keys` with known-good `Mulshrolate3RX` seed
  - `random-33000.keys` with CPU-generated known-good `Mulshrolate3RX` seed `0x7EFEA947,0xC649CF69,0x0C03170F,0xF87EDD5E`
- In both cases:
  - GPU acyclic detection succeeded
  - GPU assignment succeeded
  - CPU verify succeeded

## Peel-Order Divergence
- The GPU `Order[]` does diverge from the CPU oracle order on the tested seeds.
- That divergence is now understood and validated:
  - `GraphImpl3` stores `Order[]` in reverse peel order
  - the first peeled edge lands at the end of the array
  - the last peeled edge lands at index `0`
- A debug-only CPU-side oracle in `GraphCu.c` now replays the GPU `Order[]` against a scratch copy of the CPU-built graph in the correct direction.
- Result on the tested cases:
  - HologramWorld `Assigned16`: valid
  - `random-33000` non-`Assigned16`: valid
  - HologramWorld full file-I/O run: valid
- So the current interpretation is:
  - GPU peel order mismatch does not imply incorrectness
  - the GPU path is producing a different but valid reverse-peel order

## Assignment / Order Semantics
- The POC does perform assignment.
- It explicitly verifies order-preserving indexing for actual keys:
  - for each edge/key id `e`, it checks `(assigned[u] + assigned[v]) & edge_mask == e`
  - this means the resulting index reconstructs the original key ordinal in the input file / array
- This is stricter than merely asking for a collision-free index into `[0, n)`.
- Exploring unordered indexing does make sense:
  - it relaxes the verification target and may open up more GPU-friendly formulations
  - it could reduce assignment constraints if the downstream consumer only needs a stable collision-free placement plus a values array
  - however, it is a semantic change relative to the existing PerfectHash model and compiled-table expectations
- it is probably best treated as a parallel design branch, not as a silent change to CHM behavior

## Minimal GPU Verify
- The legacy `Graph.cu` path now has a minimal GPU `Verify()` stage as well.
- Scope is deliberately narrow:
  - one thread per key
  - recompute the hash on device
  - load `Assigned[u]` and `Assigned[v]`
  - compute the final index
  - fail if the computed index is not equal to the key's ordinal
- This is sufficient for the current ordered-index bring-up because:
  - it directly checks the semantic we care about
  - if every key maps to its own ordinal, uniqueness is implied
- It is simpler than the current CPU verify implementation:
  - no `Values[]` scratch array
  - no assigned bitmap / collision replay logic
  - no host copy-back
- Current validation:
  - HologramWorld `Assigned16`: pass
  - `random-33000` non-`Assigned16`: pass
  - HologramWorld full file-I/O: pass

## First-Class Test Coverage
- The parameterized CUDA known-seed harness is now promoted into `tests/CMakeLists.txt` for CUDA-enabled builds.
- Current first-class `ctest` coverage is intentionally self-contained to this repo:
  - `perfecthash.cuda.chm02.hologram.nofileio`
  - `perfecthash.cuda.chm02.hologram.fileio`
  - `perfecthash.cuda.chm02.generated33000.nofileio`
- The non-`Assigned16` `ctest` case is now self-contained:
  - generated deterministically at test time
  - no dependency on the external `perfecthash-keys` checkout
- Chosen now:
  - keep first-class CUDA tests repo-local and deterministic

## Performance Exploration Bootstrap
- The performance-plan execution has now started, but still in a deliberately safe mode:
  - no broad benchmark matrices launched yet
  - no large batch experiments launched yet
  - only dry-run planning and tiny POC verification runs so far
- The benchmark-runner scaffold now exists:
  - `scripts/benchmark_gpu_solver.py`
  - `scripts/benchmark_gpu_solver_config.json`
  - `tests/test_benchmark_gpu_solver.py`
- Current runner behavior:
  - validates `datasets`, `variants`, and `output_options`
  - emits a machine-readable dry-run plan
  - refuses to pretend real execution is implemented when non-empty runs are requested without `--dry-run`
- The batched POC now exposes the minimum data needed for fair GPU-vs-CPU performance exploration:
  - `--output-format json`
  - `--allocation-mode explicit-device|managed-default|managed-prefetch-gpu`
  - per-stage timings:
    - `stage_timings_ms.add_build`
    - `stage_timings_ms.peel`
    - `stage_timings_ms.assign`
    - `stage_timings_ms.verify`
- This puts the branch in a good place to start the actual benchmark runner work without risking oversubscription on GB10.

## Legacy `Chm02` Perf Surface
- The initial CLI probe showed the existing CSV timing fields were not sufficient for fair comparison:
  - `SolveMicroseconds` and `VerifyMicroseconds` existed
  - there was no explicit breakdown for CUDA add-keys / acyclic / assign / verify
  - compat timing units were wrong because `QueryPerformanceCounter()` returned nanoseconds while `QueryPerformanceFrequency()` reported `1000`
- That compat timing bug is now fixed, which normalizes all context/graph microsecond fields on Linux.
- The legacy CUDA path now emits explicit benchmark fields:
  - `CuAddKeysMicroseconds`
  - `CuIsAcyclicMicroseconds`
  - `CuAssignMicroseconds`
  - `CuVerifyMicroseconds`
- These are captured at the `GraphCu*()` wrapper level, which means they reflect the current implementation as it actually runs today, including any host-side work that remains inside those stages.
- A dedicated CUDA perf-surface regression now exists:
  - `tests/run_cli_chm02_cuda_perf_benchmark.cmake`
  - `perfecthash.cuda.chm02.perf-surface`

## First Actual Safe Runner Output
- The benchmark runner now executes a tiny safe subset instead of only dry-run planning.
- Current execution scope is intentionally tiny and explicit:
  - `hologram31016` + `cpu-cli-chm01-single`
  - `hologram31016` + `cuda-chm02-single`
  - `generated8193` + `gpu-poc-device-serial`
- The runner now records the exact executed command and refuses broader execution shapes.
- First GB10 results:
  - CPU CLI HologramWorld known-good single run:
    - `SolveMicroseconds=1779`
    - `VerifyMicroseconds=176`
  - CUDA `Chm02` HologramWorld known-good single run:
    - `SolveMicroseconds=208808`
    - `VerifyMicroseconds=245088`
    - `CuAddKeysMicroseconds=23997`
    - `CuIsAcyclicMicroseconds=165989`
    - `CuAssignMicroseconds=15124`
  - GPU POC generated8193 tiny run:
    - `gpu_ms=100.256`
    - `cpu_ms=15.180`
    - `solved=81`
- The current large CPU-vs-GPU discrepancy is therefore no longer just theoretical:
  - for single-table known-good latency, legacy `Chm02` is still roughly two orders of magnitude slower than CPU on HologramWorld
  - the dominant term in the current `Chm02` path is `CuIsAcyclic`
  - this strongly supports the hypothesis that the legacy single-graph execution model is the main problem, not hashing or verify

## Immediate Follow-On Improvements
- Replace the host round loop with cooperative launch / grid-synchronous or device-side work queues.
- Add `GraphImpl3`-compatible hashing routines so the prototype can use real PerfectHash hash functions and seeds.
- Explore one-warp-per-graph and one-CTA-per-graph assignment variants.
- Add frontier compaction via CUB / CCCL instead of a simple global atomic append.
- Integrate coverage/performance instrumentation:
  - rounds per graph
  - frontier size histogram per round
  - solved-graph throughput vs batch size
  - unified-memory vs explicit device allocation comparison
- If this path continues to look good, then decide whether to:
  - graft it into PerfectHash as a new experimental component, or
  - keep iterating standalone until the kernel stack stabilizes

## Open Questions
- Which exact CUDA failure mode matters most today: correctness, performance, scaling, or maintenance complexity?
- Can graph peeling/assignment be reformulated as bulk-synchronous frontier processing across many graphs?
- Is the best near-term path a custom kernel stack, cooperative groups, or reuse/inspiration from cuCollections/cuDF-style build pipelines?

## References
- PHOBIC paper: https://drops.dagstuhl.de/storage/00lipics/lipics-vol308-esa2024/LIPIcs.ESA.2024.69/LIPIcs.ESA.2024.69.pdf
- GPU-accelerated BDZ thesis / xor-filter construction: https://dash.harvard.edu/server/api/core/bitstreams/7fb2a614-f61a-4bba-9239-880a79b15520/content
- PtrHash paper: https://arxiv.org/abs/2502.15539
- Cuckoo-GPU paper: https://arxiv.org/abs/2603.15486
- GPH SIGMOD 2025 paper: https://www4.comp.polyu.edu.hk/~csmlyiu/conf/SIGMOD25_GPH.pdf
- cuCollections repository: https://github.com/NVIDIA/cuCollections
- `cuda.coop` API docs: https://nvidia.github.io/cccl/python/coop_api.html

## Batched Create Integration Review
- Reviewed how invasive it would be to support true batched GPU creation attempts inside the current `PerfectHashCreate` / `Chm02` infrastructure.
- Conclusion:
  - adding CLI parameters is easy
  - adding batched-attempt semantics to the current create path is not
- Main reasons:
  - `PerfectHashContextTableCreateArgvW()` and `PerfectHashOnlineCreateTableFromKeys()` are table-centric, not batched-attempt-centric
  - `GraphEnterSolvingLoop()` is explicitly one-graph, one-attempt-at-a-time
  - solved-graph promotion is built around one `BestGraph` plus an optional `SpareGraph`
  - `Chm02Shared.c` allocates one host/device graph pair per solve context, not bulk attempt buffers
  - `Chm02.c` / `Chm02Compat.c` completion, events, finished-work, and file-work all assume a single winning graph and a single table artifact set
  - CSV/reporting are likewise one-create-one-row today
- Existing bulk-create does not solve this problem:
  - it batches many key files / tables, not many attempts for one key set
- Recommendation:
  - keep iterating in the standalone batched POC
  - if the POC needs a more official surface, add a new experimental batched component or CLI before trying to integrate with `PerfectHashCreate`
  - defer any `Chm03`-inside-`PerfectHashCreate` work until the batched solver contract has stabilized
- Full write-up:
  - `docs/superpowers/reports/2026-03-26-gpu-batched-create-integration-review.md`

## Execution Geometry Stage 1
- Stage 1 geometry option surface is now real:
  - `--assign-geometry`
  - `--device-serial-peel-geometry`
  - JSON fields:
    - `assign_geometry`
    - `device_serial_peel_geometry`
- Important correction:
  - `assign_geometry` remains configuration/reporting only
  - real assignment warp/block execution is blocked by the current reverse-peel dependency chain
  - the first attempt to fake assignment geometry via launch-shape wrappers was reverted
- `device-serial` peel geometry now has real implementations for:
  - `thread`
  - `warp`
  - `block`
- The first bounded local baseline shows:
  - `block` peel is clearly better than `warp`
  - both beat the scalar `thread` peel path
  - once peel improves, assignment becomes a much larger share of total GPU time
- Local GB10 results:
  - generated `8193`, `batch=128`, `threads=128`:
    - `thread`: GPU `38.688 ms`, peel `33.154 ms`
    - `warp`: GPU `13.007 ms`, peel `7.494 ms`
    - `block`: GPU `7.790 ms`, peel `2.245 ms`
  - `HologramWorld-31016.keys`, `batch=16`, `threads=128`:
    - `thread`: GPU `76.849 ms`, peel `75.735 ms`
    - `warp`: GPU `16.704 ms`, peel `15.576 ms`
    - `block`: GPU `5.156 ms`, peel `4.041 ms`
- Recommendation:
  - keep assignment scalar for now
  - target one-block-per-graph peel for the next shared-memory / CUB pass
  - revisit cooperative assignment only after peel-layer metadata exists

## CPU Step Equivalents
- Added CPU stage timing instrumentation to the POC:
  - `cpu_stage_timings_ms_all_attempts`
  - `cpu_stage_timings_ms_solved_only`
  - each includes:
    - `add_build`
    - `peel`
    - `assign`
    - `verify`
- On generated `8193`, `batch=128`, `threads=128`, best current GPU mode (`block` peel):
  - GPU:
    - add/build `0.983 ms`
    - peel `2.245 ms`
    - assign `3.815 ms`
    - verify `0.457 ms`
  - CPU all-attempt equivalents:
    - add/build `1.717 ms`
    - peel `6.312 ms`
    - assign `0.112 ms`
    - verify `0.061 ms`
- On this solved generated case:
  - GPU peel is now clearly ahead
  - CPU assignment/verify are still dramatically cheaper
- On `HologramWorld-31016.keys`, `batch=16`, `threads=128`, best current GPU mode (`block` peel):
  - GPU:
    - add/build `0.733 ms`
    - peel `4.041 ms`
  - CPU all-attempt equivalents:
    - add/build `1.036 ms`
    - peel `2.759 ms`
- This strengthens the hybrid idea on GB10:
  - GPU build/peel across many attempts
  - compact solved graphs
  - CPU assignment/verify for the survivors
