# tpch-query-probes

Query-driven extractor for TPC-H SF10 parquet data that materializes:

1. a unique build-side `.keys` domain for a candidate join,
2. a duplicate-preserving probe stream file for the corresponding fact-side key
   accesses, and
3. summary JSON with probe cardinality and skew statistics.

Current query/join targets:

- `q8_part_lineitem`
- `q21_supplier_lineitem`

The output is intended for downstream PerfectHash and cuCollections lookup
benchmarks on realistic probe streams instead of the current unique-key
microbenchmarks.

## Usage

```bash
PYTHONPATH=/tmp/tpch_duckdb python examples/tpch-query-probes/extract_tpch_query_probes.py \
  --dataset-root /mnt/data/tpch/scale-10 \
  --output-root /mnt/data/tpch/scale-10/query-probes
```

If `duckdb` is installed elsewhere, the script will use the normal import path.
