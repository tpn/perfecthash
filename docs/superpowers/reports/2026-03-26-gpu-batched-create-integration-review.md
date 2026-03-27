# GPU Batched Create Integration Review

**Date:** 2026-03-26

## Bottom Line

Supporting a true batched-attempt GPU creation path inside the current
`PerfectHashCreate` / `Chm02` infrastructure is a large architectural change,
not a small extension.

The CLI is not the hard part. The hard part is that the existing stack is built
around:

- one table-create session per invocation
- one `PERFECT_HASH_CONTEXT`
- one winning graph promoted into one table artifact set
- one `GRAPH` object per active solver lane
- one attempt at a time per graph

That is fundamentally different from the batched GPU POC, which is built
around:

- one key set
- many seed tuples for that key set
- one batched solve request
- bulk build/peel/assign/verify
- compact result summaries

## Main Findings

### CLI and parameter plumbing are the easy part

The table-create CLI already uses an extensible parameter table and centralized
argument extraction. Adding parameters such as batch size, allocation mode, or
solve mode would be mechanically straightforward.

Relevant files:

- `include/PerfectHash/PerfectHash.h`
- `src/PerfectHash/ExtractArg.c`
- `src/PerfectHash/PerfectHashContextTableCreate.c`

This means a new CLI is not required because parsing is difficult. A new CLI
would only be justified to avoid inheriting the deeper single-run assumptions.

### The create pipeline is table-centric, not batched-attempt-centric

`PerfectHashContextTableCreateArgvW()` extracts one key path, one output
directory, one algorithm/hash/mask tuple, one concurrency value, then calls
`Context->Vtbl->TableCreate()`.

The online path does the same thing in memory:

- `PerfectHashOnlineCreateTableFromKeys()` creates one keys object, one context,
  one table, then calls `Table->Create()`

Relevant files:

- `src/PerfectHash/PerfectHashContextTableCreate.c`
- `src/PerfectHash/PerfectHashOnlineCreate.c`

There is no natural notion at this layer of "run 4096 attempts for this one key
set as a single device batch, then select a winner".

### Graph solving is explicitly one-attempt-at-a-time

`GraphEnterSolvingLoop()` is the clearest signal. It does:

1. `LoadInfo()`
2. loop:
   - `Reset()`
   - `LoadNewSeeds()`
   - `Solve()`
   - possibly continue with a replacement graph

Relevant files:

- `src/PerfectHash/Graph.c`

So the current contract is:

- one graph object represents one solver lane
- each lane owns one mutable attempt state
- attempts advance one at a time
- stop conditions are context-global:
  - `FixedAttempts`
  - `TargetNumberOfSolutions`
  - `FirstSolvedGraphWins`
  - `FindBestGraph`

That is a poor match for a GPU execution model where the natural unit is "many
attempts at once".

### Winner promotion assumes one solved graph object

`GraphRegisterSolved()`, `GraphRegisterSolvedNoBestCoverage()`, and the
`Assigned16` variants revolve around a single promoted graph plus an optional
spare graph used to continue solving.

Relevant files:

- `src/PerfectHash/Graph.c`
- `src/PerfectHash/PerfectHashContext.h`

The context stores singular state such as:

- `BestGraph`
- `SpareGraph`
- `BestGraphInfo[]`
- `SolvedContext`
- `FinishedCount`
- `MostRecentSolvedAttempt`
- `RunningSolutionsFoundRatio`

That works when every meaningful candidate is already a `GRAPH` object. It is
not the shape you want for a batched GPU solver that evaluates many seeds in
bulk and only materializes a small winner set.

### Chm02 CUDA infrastructure is one host/device graph pair per solve context

`InitializeCudaAndGraphsChm02()` allocates:

- one `PH_CU_SOLVE_CONTEXT` per CUDA concurrency slot
- one host graph and one device graph per solve context
- optional spare host/device graphs in best-graph mode

Relevant files:

- `src/PerfectHash/Chm02Shared.c`
- `src/PerfectHash/PerfectHashCu.h`
- `src/PerfectHash/GraphCu.c`

`PH_CU_SOLVE_CONTEXT` literally stores:

- `HostGraph`
- `DeviceGraph`
- `HostSpareGraph`
- `DeviceSpareGraph`

That is the opposite of the POC design, which wants bulk attempt buffers, not
one graph object per solver lane.

### Completion, events, file work, and persistence assume one final artifact set

`Chm02.c` / `Chm02Compat.c` coordinate:

- `SucceededEvent`
- `FailedEvent`
- `CompletedEvent`
- `ShutdownEvent`
- `MainWorkList`
- `FinishedWorkList`
- file preparation and save work items
- final promotion of one graph into the table

Relevant files:

- `src/PerfectHash/Chm02.c`
- `src/PerfectHash/Chm02Compat.c`
- `src/PerfectHash/PerfectHashContext.h`
- `src/PerfectHash/PerfectHashTable.c`
- `src/PerfectHash/Chm01FileWork.c`

This machinery is useful for persisted-table creation, but it is tightly
coupled to:

- one winning graph
- one output directory / one file set
- one `:Info` stream and one table-data artifact set

If batched attempts were introduced under the current CLI, batch identity would
need to flow through file naming, output directories, and saved metadata.

### CSV/reporting are also one-create-one-row

The normal table-create CSV path assumes one row per create invocation, and the
best-graph CSV path flattens best-graph snapshots into columns on that same row.

Relevant files:

- `src/PerfectHash/PerfectHashContextTableCreate.c`
- `src/PerfectHash/TableCreateCsv.h`
- `src/PerfectHash/TableCreateBestCsv.h`

A batched-attempt create mode would need either:

- a different row model
- a new CSV/report format
- or a separate reporting surface

### Existing bulk-create is not the same problem, but it is a useful clue

`PerfectHashContextBulkCreateArgvW()` batches many key files / many tables, not
many attempts for one table.

Relevant files:

- `src/PerfectHash/PerfectHashContextBulkCreate.c`

So it is not a shortcut to batched GPU solving. But it does suggest a safer
architecture: keep `PerfectHashCreate` as a single-create primitive and add any
batch orchestration above it rather than distorting its semantics.

### The batched POC is valuable because it avoids all of this

The POC does not try to reuse `PerfectHashCreate`, `Graph`, or file-work
plumbing. It models the execution unit that actually matters:

- one key set
- many attempts
- bulk GPU build/peel/assign/verify
- compact timing/result output

Relevant files:

- `experiments/gpu_batched_peeling_poc/README.md`
- `experiments/gpu_batched_peeling_poc/main.cu`

That lower-friction boundary is what has made iteration possible.

## What Would Need to Change for a True Mainline Batched Path

### Small-to-moderate change

- Add new CLI / table-create parameters for batch size, memory placement, or
  explicit seed bundles.
- Add a new algorithm id such as `Chm03` if we want to keep the existing CLI
  entrypoint.
- Add batched reporting fields and perf schema.

### Large change

- New solver contract that is not one `GRAPH` object per active attempt.
- New CUDA runtime-context layout that owns batched buffers directly instead of
  host/device graph pairs per solve context.
- New winner-selection path that selects from batch results without requiring
  every attempt to be a full `GRAPH` object.
- New integration for assignment / verify artifacts that only materializes
  winners.
- Reworked stopping semantics for:
  - fixed attempts
  - target number of solutions
  - best-graph / best-coverage selection
- Careful decision about whether table resize participates at all.

### Very large change if forced into the current Graph model

- Reinterpreting `GraphEnterSolvingLoop()` to represent batches, not scalar
  attempts.
- Reworking `GraphReset()`, `LoadNewSeeds()`, and `Solve()` contracts.
- Retrofitting `GraphRegisterSolved*()` and spare-graph logic for batch
  candidate sets.
- Preserving current file-work, CSV, ETW, and test behavior while doing so.

That path looks expensive and likely to slow down learning.

## Integration Options

### Option A: Keep iterating in `experiments/`

Pros:

- lowest friction
- fastest iteration
- no legacy create-path coupling

Cons:

- less reuse of existing reporting/persistence infrastructure
- integration debt accumulates

### Option B: Add a new experimental batched component or CLI

Examples:

- `PerfectHashCreateBatched`
- a new online API specialized for batched solve requests
- a new internal component with request/result structs and a thin dedicated CLI

Pros:

- preserves the POC execution model
- allows shared build/test/benchmark infrastructure
- avoids destabilizing current `PerfectHashCreate`

Cons:

- another API/CLI to maintain

### Option C: Add `Chm03` under `PerfectHashCreate` now

Pros:

- preserves the familiar CLI entrypoint

Cons:

- high risk of inheriting the current single-graph lifecycle too early
- likely forces premature decisions about file-work, persistence, coverage, and
  table-resize semantics

## Recommendation

Near-term:

- Keep iterating on the batched POC.
- If it needs a more official boundary, add a new experimental batched
  component or CLI first.
- Do not try to force it into current `Chm02` / `Graph` plumbing yet.

Medium-term:

- If the POC continues to win on correctness and throughput, introduce a new
  mainline experimental boundary first.
- Consider `Chm03` inside `PerfectHashCreate` only after the batched solver
  request/result contract has stabilized.

## Practical Translation

If the question is "how much of the existing infrastructure would need to
change?", the answer is:

- CLI parsing: small
- parameter plumbing: small-to-moderate
- context / graph / winner-selection model: large
- CUDA solve-context model: large
- file-work / persistence / CSV integration: moderate-to-large

So yes, changing `Chm02` or adding `Chm03` to properly leverage a batched GPU
solver is a big undertaking if attempted inside the current mainline path.

That is exactly why continuing the proof-of-concept is the right move for now.
