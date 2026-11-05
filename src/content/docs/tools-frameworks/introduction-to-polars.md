---
title: Introduction to Polars - Fast DataFrames for ML Data Preparation
description: Learn how Polars' Rust-based engine, expression API, and lazy evaluation speed up feature engineering and dataset preparation compared with traditional pandas workflows.
---

Polars is a DataFrame library written in Rust with Python bindings. It is built on Apache Arrow's columnar memory layout, uses all CPU cores by default, and offers both eager and lazy execution, which makes it well suited to large-scale feature engineering and dataset preparation.

## Expressions

Polars operations are built from composable *expressions* that the engine can parallelize and optimize:

```python
import polars as pl

df = pl.read_csv("events.csv")

features = df.group_by("user_id").agg(
    pl.col("amount").sum().alias("total_spend"),
    pl.col("amount").mean().alias("avg_spend"),
    pl.col("event_time").max().alias("last_seen"),
)
```

## Lazy Evaluation

With `scan_*` and `.lazy()`, Polars builds a query plan and optimizes it before executing:

```python
result = (
    pl.scan_parquet("logs/*.parquet")
    .filter(pl.col("status") == "ok")
    .select(["user_id", "latency_ms"])
    .group_by("user_id")
    .agg(pl.col("latency_ms").quantile(0.95))
    .collect()
)
```

The optimizer applies predicate pushdown (filtering while reading) and projection pushdown (reading only needed columns), which can dramatically cut I/O and memory.

## Comparison with pandas

| Aspect | pandas | Polars |
|--------|--------|--------|
| Execution | Mostly single-threaded | Multi-threaded |
| Evaluation | Eager | Eager and lazy |
| Memory format | NumPy-based | Arrow columnar |
| Larger-than-memory | Limited | Streaming mode |

## Practical Notes

- Convert to NumPy or pandas at the boundary with `.to_numpy()` or `.to_pandas()` when feeding model libraries.
- Prefer expressions over row-wise Python functions, which bypass the engine's optimizations.
- Use Parquet as the on-disk format to benefit from column pruning.
