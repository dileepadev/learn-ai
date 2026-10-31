---
title: Introduction to DuckDB - In-Process Analytical SQL for Data Science
description: Learn how DuckDB lets you run fast analytical SQL directly over Parquet, CSV, and DataFrames inside your Python process, without managing a database server.
---

DuckDB is an embedded, columnar analytical database. It runs inside your process like SQLite but is optimized for analytical queries (aggregations, joins, window functions) over large datasets, which makes it a favorite tool for data exploration and ML feature preparation.

## Query Files Directly

```python
import duckdb

duckdb.sql("""
    SELECT label, COUNT(*) AS n, AVG(score) AS avg_score
    FROM 'predictions/*.parquet'
    GROUP BY label
    ORDER BY n DESC
""").show()
```

No import step or schema definition is needed; DuckDB reads Parquet and CSV in place and pushes filters and projections into the scan.

## Working with DataFrames

DuckDB can query pandas or Polars DataFrames in memory by name and return results as DataFrames:

```python
import pandas as pd
df = pd.read_csv("users.csv")

out = duckdb.sql("SELECT country, COUNT(*) c FROM df GROUP BY country").df()
```

## Why It Fits ML Workflows

- **Larger-than-memory analysis**: spills to disk when needed
- **Fast joins and window functions** for feature construction
- **Vector similarity**: array types and the VSS extension support nearest-neighbor search on embeddings
- **Zero infrastructure**: one library, one optional database file

## When Not to Use It

DuckDB is built for analytics, not high-concurrency transactional workloads. For many simultaneous writers or serving low-latency point lookups, use a client-server database instead.

## Tips

- Store intermediate datasets as Parquet and query them with DuckDB rather than keeping everything in memory.
- Use `EXPLAIN` to confirm that filters are being pushed down to the file scan.
