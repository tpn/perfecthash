#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import sys
from pathlib import Path


def import_duckdb():
    try:
        import duckdb  # type: ignore

        return duckdb
    except ModuleNotFoundError:
        fallback = "/tmp/tpch_duckdb"
        if os.path.isdir(fallback):
            sys.path.insert(0, fallback)
            import duckdb  # type: ignore

            return duckdb
        raise


def write_u64_file(path: Path, values) -> int:
    count = 0
    with path.open("wb") as f:
        for value in values:
            f.write(struct.pack("<Q", int(value)))
            count += 1
    return count


def materialize_build_keys(con, output_path: Path, sql: str) -> dict:
    rows = con.execute(sql).fetchall()
    values = [row[0] for row in rows]
    count = write_u64_file(output_path, values)
    return {
        "path": str(output_path),
        "count": count,
        "bytes": count * 8,
    }


def materialize_probe_stream(con, output_path: Path, sql: str) -> dict:
    count = 0
    with output_path.open("wb") as f:
        cur = con.execute(sql)
        while True:
            rows = cur.fetchmany(100_000)
            if not rows:
                break
            for (value,) in rows:
                f.write(struct.pack("<Q", int(value)))
                count += 1
    return {
        "path": str(output_path),
        "count": count,
        "bytes": count * 8,
    }


def summarize_probe_freq(con, sql: str) -> dict:
    summary_sql = f"""
    with freq as (
      {sql}
    ), ranked as (
      select
        k,
        cnt,
        row_number() over(order by cnt desc, k) as rn,
        count(*) over() as dk,
        sum(cnt) over(order by cnt desc, k rows between unbounded preceding and current row) as prefix,
        sum(cnt) over() as total_rows,
        max(cnt) over() as max_cnt
      from freq
    )
    select
      max(total_rows) as total_rows,
      max(dk) as distinct_keys,
      max(max_cnt) as max_cnt,
      max(prefix) filter (where rn <= ceil(dk * 0.01)) as top1_rows,
      max(prefix) filter (where rn <= ceil(dk * 0.10)) as top10_rows
    from ranked
    """
    total_rows, distinct_keys, max_cnt, top1_rows, top10_rows = con.execute(summary_sql).fetchone()
    return {
        "probe_rows": int(total_rows),
        "probe_distinct_keys": int(distinct_keys),
        "max_frequency": int(max_cnt),
        "avg_multiplicity": float(total_rows) / float(distinct_keys),
        "top1_mass": float(top1_rows) / float(total_rows),
        "top10_mass": float(top10_rows) / float(total_rows),
    }


def write_top_distinct_subset(con, output_path: Path, sql: str, fraction: float) -> dict:
    subset_sql = f"""
    with freq as (
      {sql}
    ), ranked as (
      select
        k,
        cnt,
        row_number() over(order by cnt desc, k) as rn,
        count(*) over() as dk
      from freq
    )
    select k
    from ranked
    where rn <= ceil(dk * {fraction})
    order by k
    """
    rows = con.execute(subset_sql).fetchall()
    values = [row[0] for row in rows]
    count = write_u64_file(output_path, values)
    return {
        "path": str(output_path),
        "count": count,
        "bytes": count * 8,
        "fraction_of_distinct": fraction,
    }


def register_views(con, dataset_root: Path, tables):
    for table in tables:
        path = dataset_root / table / "*.parquet"
        con.execute(f"create or replace view {table} as select * from parquet_scan('{path}')")


def extract_q8(con, output_root: Path):
    name = "q8_part_lineitem"
    build_sql = """
        select p_partkey as k
        from part
        where p_type = 'ECONOMY ANODIZED STEEL'
        order by k
    """
    probe_stream_sql = """
        with orders_region as (
            select o_orderkey
            from orders
            join customer on o_custkey = c_custkey
            join nation on c_nationkey = n_nationkey
            join region on n_regionkey = r_regionkey
            where r_name = 'AMERICA'
              and o_orderdate between date '1995-01-01' and date '1996-12-31'
        )
        select l_partkey as k
        from lineitem
        join orders_region on l_orderkey = o_orderkey
    """
    probe_freq_sql = """
        with orders_region as (
            select o_orderkey
            from orders
            join customer on o_custkey = c_custkey
            join nation on c_nationkey = n_nationkey
            join region on n_regionkey = r_regionkey
            where r_name = 'AMERICA'
              and o_orderdate between date '1995-01-01' and date '1996-12-31'
        )
        select l_partkey as k, count(*) as cnt
        from lineitem
        join orders_region on l_orderkey = o_orderkey
        group by l_partkey
    """

    target_dir = output_root / name
    target_dir.mkdir(parents=True, exist_ok=True)

    build_info = materialize_build_keys(con, target_dir / "build_p_partkey-64.keys", build_sql)
    probe_info = materialize_probe_stream(con, target_dir / "probe_l_partkey-64.bin", probe_stream_sql)
    summary = summarize_probe_freq(con, probe_freq_sql)
    summary["candidate"] = name
    summary["build_rows"] = build_info["count"]
    summary["probe_to_build_ratio"] = summary["probe_rows"] / build_info["count"]
    summary["build_keys"] = build_info
    summary["probe_stream"] = probe_info
    summary["top1pct_distinct_keys"] = write_top_distinct_subset(
        con, target_dir / "top1pct_distinct_l_partkey-64.keys", probe_freq_sql, 0.01
    )
    summary["top10pct_distinct_keys"] = write_top_distinct_subset(
        con, target_dir / "top10pct_distinct_l_partkey-64.keys", probe_freq_sql, 0.10
    )
    with (target_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def extract_q21(con, output_root: Path):
    name = "q21_supplier_lineitem"
    build_sql = """
        select s_suppkey as k
        from supplier
        join nation on s_nationkey = n_nationkey
        where n_name = 'SAUDI ARABIA'
        order by k
    """
    probe_stream_sql = """
        with q1 as (
            select l_orderkey, count(l_suppkey) as n_supp_by_order
            from (
                select l_orderkey, l_suppkey
                from lineitem
                where l_receiptdate > l_commitdate
                  and l_orderkey in (
                    select l_orderkey
                    from lineitem
                    group by l_orderkey
                    having count(l_suppkey) > 1
                  )
            ) t
            group by l_orderkey
        ), q1_expanded as (
            select q1.n_supp_by_order, li.l_orderkey, li.l_suppkey
            from q1
            join lineitem li
              on q1.l_orderkey = li.l_orderkey
             and li.l_receiptdate > li.l_commitdate
            join orders o on li.l_orderkey = o.o_orderkey
            where q1.n_supp_by_order = 1
              and o.o_orderstatus = 'F'
        )
        select l_suppkey as k
        from q1_expanded
    """
    probe_freq_sql = """
        with q1 as (
            select l_orderkey, count(l_suppkey) as n_supp_by_order
            from (
                select l_orderkey, l_suppkey
                from lineitem
                where l_receiptdate > l_commitdate
                  and l_orderkey in (
                    select l_orderkey
                    from lineitem
                    group by l_orderkey
                    having count(l_suppkey) > 1
                  )
            ) t
            group by l_orderkey
        ), q1_expanded as (
            select q1.n_supp_by_order, li.l_orderkey, li.l_suppkey
            from q1
            join lineitem li
              on q1.l_orderkey = li.l_orderkey
             and li.l_receiptdate > li.l_commitdate
            join orders o on li.l_orderkey = o.o_orderkey
            where q1.n_supp_by_order = 1
              and o.o_orderstatus = 'F'
        )
        select l_suppkey as k, count(*) as cnt
        from q1_expanded
        group by l_suppkey
    """

    target_dir = output_root / name
    target_dir.mkdir(parents=True, exist_ok=True)

    build_info = materialize_build_keys(con, target_dir / "build_s_suppkey-64.keys", build_sql)
    probe_info = materialize_probe_stream(con, target_dir / "probe_l_suppkey-64.bin", probe_stream_sql)
    summary = summarize_probe_freq(con, probe_freq_sql)
    summary["candidate"] = name
    summary["build_rows"] = build_info["count"]
    summary["probe_to_build_ratio"] = summary["probe_rows"] / build_info["count"]
    summary["build_keys"] = build_info
    summary["probe_stream"] = probe_info
    summary["top1pct_distinct_keys"] = write_top_distinct_subset(
        con, target_dir / "top1pct_distinct_l_suppkey-64.keys", probe_freq_sql, 0.01
    )
    summary["top10pct_distinct_keys"] = write_top_distinct_subset(
        con, target_dir / "top10pct_distinct_l_suppkey-64.keys", probe_freq_sql, 0.10
    )
    with (target_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def extract_q5(con, output_root: Path):
    name = "q5_supplier_lineitem"
    build_sql = """
        select s_suppkey as k
        from supplier
        join nation on s_nationkey = n_nationkey
        join region on n_regionkey = r_regionkey
        where r_name = 'ASIA'
        order by k
    """
    probe_stream_sql = """
        with asia_customers as (
            select c_custkey
            from customer
            join nation on c_nationkey = n_nationkey
            join region on n_regionkey = r_regionkey
            where r_name = 'ASIA'
        ), asia_orders as (
            select o_orderkey
            from orders
            join asia_customers on o_custkey = c_custkey
            where o_orderdate >= date '1994-01-01'
              and o_orderdate < date '1995-01-01'
        )
        select l_suppkey as k
        from lineitem
        join asia_orders on l_orderkey = o_orderkey
    """
    probe_freq_sql = """
        with asia_customers as (
            select c_custkey
            from customer
            join nation on c_nationkey = n_nationkey
            join region on n_regionkey = r_regionkey
            where r_name = 'ASIA'
        ), asia_orders as (
            select o_orderkey
            from orders
            join asia_customers on o_custkey = c_custkey
            where o_orderdate >= date '1994-01-01'
              and o_orderdate < date '1995-01-01'
        )
        select l_suppkey as k, count(*) as cnt
        from lineitem
        join asia_orders on l_orderkey = o_orderkey
        group by l_suppkey
    """

    target_dir = output_root / name
    target_dir.mkdir(parents=True, exist_ok=True)

    build_info = materialize_build_keys(con, target_dir / "build_s_suppkey-64.keys", build_sql)
    probe_info = materialize_probe_stream(con, target_dir / "probe_l_suppkey-64.bin", probe_stream_sql)
    summary = summarize_probe_freq(con, probe_freq_sql)
    summary["candidate"] = name
    summary["build_rows"] = build_info["count"]
    summary["probe_to_build_ratio"] = summary["probe_rows"] / build_info["count"]
    summary["build_keys"] = build_info
    summary["probe_stream"] = probe_info
    summary["top1pct_distinct_keys"] = write_top_distinct_subset(
        con, target_dir / "top1pct_distinct_l_suppkey-64.keys", probe_freq_sql, 0.01
    )
    summary["top10pct_distinct_keys"] = write_top_distinct_subset(
        con, target_dir / "top10pct_distinct_l_suppkey-64.keys", probe_freq_sql, 0.10
    )
    with (target_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    duckdb = import_duckdb()
    con = duckdb.connect()
    register_views(
        con,
        Path(args.dataset_root),
        ["customer", "orders", "lineitem", "part", "supplier", "nation", "region"],
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = [
        extract_q5(con, output_root),
        extract_q8(con, output_root),
        extract_q21(con, output_root),
    ]

    table_path = output_root / "summary.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "build_rows",
                "probe_rows",
                "probe_distinct_keys",
                "max_frequency",
                "avg_multiplicity",
                "top1_mass",
                "top10_mass",
                "probe_to_build_ratio",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    print(table_path)
    for row in results:
        print(
            row["candidate"],
            row["build_rows"],
            row["probe_rows"],
            row["probe_distinct_keys"],
            f"avg_mult={row['avg_multiplicity']:.3f}",
            f"top1={row['top1_mass']:.3f}",
            f"top10={row['top10_mass']:.3f}",
            f"probe/build={row['probe_to_build_ratio']:.3f}",
        )


if __name__ == "__main__":
    main()
