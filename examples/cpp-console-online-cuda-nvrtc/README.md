# cpp-console-online-cuda-nvrtc

Minimal C++ example and benchmark driver that:

1. builds a perfect hash table on CPU via `PerfectHashOnlineJit`,
2. requests the fragment-only generated CUDA lookup source,
3. uploads the assigned table payload separately by default,
4. appends its own templated `ITEMS_PER_THREAD` probe kernel,
5. compiles the combined translation unit with NVRTC for the active GPU, and
6. launches the kernel via the CUDA driver API.

This is the intended integration shape for downstream GPU consumers that want
call-site specialization around the lookup routine, rather than relying on a
prepackaged kernel shape.

By default, the generated CUDA fragment omits the embedded `table_data[]` array
and expects the consumer to provide the assigned table payload as a runtime
kernel argument. This keeps the NVRTC translation unit small even for large
TPC-H domains. Use `--embed-table-data` to restore the previous fully embedded
mode for comparison.

It can run against either:

- the built-in synthetic 64-bit sample domain, or
- a raw little-endian `uint64_t` `.keys` file such as the extracted TPC-H
  domains under `/mnt/data/tpch/.../perfecthash-keys/*.keys`

## Build

### Linux/macOS

```bash
cmake -S examples/cpp-console-online-cuda-nvrtc \
      -B build/examples/cpp-console-online-cuda-nvrtc \
      -DPERFECTHASH_BUILD_PROFILE=online-rawdog-jit

cmake --build build/examples/cpp-console-online-cuda-nvrtc -j
```

### Local Tree Fallback

```bash
cmake -S examples/cpp-console-online-cuda-nvrtc \
      -B build/examples/cpp-console-online-cuda-nvrtc \
      -DPH_ONLINE_CUDA_NVRTC_USE_FETCHCONTENT=OFF \
      -DPERFECTHASH_ROOT=/path/to/perfecthash/build-verify-full
```

## Run

```bash
./build/examples/cpp-console-online-cuda-nvrtc/cpp-console-online-cuda-nvrtc \
  --hash mulshrolate3rx \
  --items-per-thread 8
```

Optional arguments:

- `--keys-file <path>`
- `--max-keys <N>`
- `--hash <name>`
- `--items-per-thread <N>`
- `--threads <N>`
- `--iterations <N>`
- `--warmup <N>`
- `--device <ordinal>`
- `--compile-mode <ptx|lto>`
- `--lookup-mode <direct|split|warpcache|blocksort>`
- `--table-load-mode <generic|readonly>`
- `--cpu-backend <none|rawdog-jit|llvm-jit|auto>`
- `--cpu-vector-width <1|2|4|8|16>`
- `--cpu-strict-vector-width <0|1>`
- `--analyze-slot-reuse`
- `--analysis-only`
- `--dump-fragment`
- `--source-out <path>`
- `--embed-table-data`
- `--csv`
- `--csv-header`
- `--no-verify`

`--dump-fragment` prints the fragment-only generated CUDA source.

`--source-out` writes the full combined NVRTC translation unit to disk, which is
useful for debugging or for experimenting with PTX versus LTO-IR compilation
strategies.

`--embed-table-data` forces the generated fragment to inline the assigned table
payload into the CUDA source instead of uploading it separately at runtime.

`--lookup-mode direct` runs the current fused path where each item computes its
two assigned-table offsets and immediately performs the two table reads.

`--lookup-mode split` breaks the work into two kernels:

- `compute_slots_kernel`: compute the two assigned-table offsets per key
- `gather_kernel`: perform the assigned-table reads and final index add

This is intended for backend analysis. It does not reorder or coalesce the
requests yet; it simply separates address generation from the random gathers.
Like the other non-`direct` modes, it is currently intended for self-probe or
known-member workloads, not external build/probe-stream validation.

`--lookup-mode warpcache` keeps the fused structure but uses warp-local
duplicate detection for slot loads, so identical slot requests within a warp
can be serviced once and broadcast with warp intrinsics. It is currently
intended for self-probe or known-member workloads, not external build/probe
stream validation.

`--lookup-mode blocksort` is a heavier experimental path that sorts all slot
requests for a block in shared memory before performing the assigned-table
loads, then scatters the gathered values back to the original outputs. It is a
clustering experiment, not a production path, and is currently intended for
self-probe or known-member workloads.

`--table-load-mode generic` uses normal global loads for the assigned table.

`--table-load-mode readonly` routes assigned-table reads through an explicit
read-only load helper in the generated consumer TU.

`--cpu-backend` enables a CPU bulk-index baseline in the same run. The benchmark
derives exact downsized 32-bit keys from the 64-bit source domain using the
table's downsize bitmap and then runs the requested JIT backend/vector width via
the public `Index32`/`Index32xN` APIs.

`--cpu-vector-width` selects the requested CPU bulk width. The CSV output
records both requested and effective width, so fallback behavior is visible.

`--cpu-strict-vector-width` forwards
`PH_ONLINE_JIT_COMPILE_FLAG_STRICT_VECTOR_WIDTH` to the CPU JIT compile path.

`--analyze-slot-reuse` computes exact host-side slot-stream reuse statistics for
the current `ITEMS_PER_THREAD` and block size using the table's actual seeds,
masks, and downsize bitmap.

`--analysis-only` runs the slot-reuse analysis and skips the CPU/GPU benchmark
phases.

`--csv` prints a single machine-readable result row. `--csv-header` prepends the
header line. Both CPU and GPU lookup throughput are reported as normalized
`ns/key` figures; GPU values can legitimately be sub-nanosecond because they are
aggregate throughput numbers over many concurrent threads.

`--no-verify` skips the post-kernel uniqueness check on returned indices.

## TPC-H Example

```bash
./build/examples/cpp-console-online-cuda-nvrtc/cpp-console-online-cuda-nvrtc \
  --keys-file /mnt/data/tpch/scale-10/perfecthash-keys/c_custkey_q03_ph-64.keys \
  --hash mulshrolate3rx \
  --compile-mode lto \
  --lookup-mode direct \
  --items-per-thread 8 \
  --threads 128 \
  --warmup 2 \
  --iterations 10
```
