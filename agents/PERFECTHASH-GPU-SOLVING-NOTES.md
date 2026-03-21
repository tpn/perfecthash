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

## What The POC Proves
- The batched execution model is viable.
- We do not need per-vertex/per-edge lock objects plus retry loops to peel many attempts in parallel.
- Capturing explicit peel order and owner-vertex metadata is enough to support assignment after peeling.
- Even with a conservative host-driven peel-round loop and sequential-per-graph assignment, batching across many graphs already yields a useful GPU advantage.
- We are already close to real PerfectHash datasets:
  - `HologramWorld-31016.keys` runs today
  - `Hydrogen-40147.keys` from the external `perfecthash-keys` repo runs today
  - the remaining distance to “try Hydrogen” is basically operational, not architectural

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
