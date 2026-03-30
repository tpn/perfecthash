# cpp-console-cuco-static-map-bench

Standalone `cuCollections` baseline for the same raw `uint64_t` `.keys` domains
used by the PerfectHash GPU experiments.

This benchmark:

1. loads a `.keys` file as `uint64_t` keys,
2. builds a `cuco::static_map<uint64_t, uint32_t>` mapping `key -> ordinal`,
3. runs `find_async()` over the same key stream,
4. reports build and lookup timings, including normalized `ns/key`.

This is the closest simple `cuCollections` baseline to the current PerfectHash
GPU key-to-index lookup path.

## Build

```bash
cmake -S examples/cpp-console-cuco-static-map-bench \
      -B build/examples/cpp-console-cuco-static-map-bench \
      -DCUCO_ROOT=/home/trentn/src/cucollections

cmake --build build/examples/cpp-console-cuco-static-map-bench -j
```

## Run

```bash
./build/examples/cpp-console-cuco-static-map-bench/cpp-console-cuco-static-map-bench \
  --keys-file /mnt/data/tpch/scale-10/perfecthash-keys/c_custkey_q03_ph-64.keys \
  --load-factor 0.5
```

Optional arguments:

- `--keys-file <path>`
- `--max-keys <N>`
- `--load-factor <f>`
- `--device <ordinal>`
- `--warmup <N>`
- `--iterations <N>`
- `--csv`
- `--csv-header`
- `--no-verify`
