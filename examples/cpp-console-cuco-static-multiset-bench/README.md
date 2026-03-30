# cpp-console-cuco-static-multiset-bench

Standalone `cuCollections` `static_multiset` baseline for the same raw `uint64_t`
`.keys` domains used by the PerfectHash GPU experiments.

This benchmark:

1. loads a `.keys` file as `uint64_t` keys,
2. builds a `cuco::static_multiset<uint64_t>`,
3. runs `find_async()` over the same key stream,
4. reports build and lookup timings, including normalized `ns/key`.

This is a closer `cuco` baseline to the join-like duplicate-preserving substrate
used in `cudf` than the `static_map` benchmark.

## Build

```bash
cmake -S examples/cpp-console-cuco-static-multiset-bench \
      -B build/examples/cpp-console-cuco-static-multiset-bench \
      -DCUCO_ROOT=/home/trentn/src/cucollections

cmake --build build/examples/cpp-console-cuco-static-multiset-bench -j
```

## Run

```bash
./build/examples/cpp-console-cuco-static-multiset-bench/cpp-console-cuco-static-multiset-bench \
  --keys-file /mnt/data/tpch/scale-10/perfecthash-keys/c_custkey_q03_ph-64.keys \
  --load-factor 0.5
```
