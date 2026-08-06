# Create Dataset Timing Report

Generated: `2026-08-04T05:16:47.046510+00:00`

**Scope:** Create Dataset / master_build (raw ticks → features → master SQLite)

## Numba (production path)

- Numba enabled: **YES**
- `ARUNEO_FEATURE_NUMBA` env: `unset`
- Numba available: `True`
- Numba kernel calls: `962,434`
- Python fallback calls: `0`
- Python fallback active: `False`

## Phase timings

| Phase | Seconds | Share |
|-------|--------:|------:|
| Loading ticks | 86.796 | 22.5% |
| Feature computation | 248.110 | 64.4% |
| Prediction targets | 0.000 | 0.0% |
| DuckDB/SQLite insert | 14.796 | 3.8% |
| Polars / DuckDB frame IO | 0.000 | 0.0% |
| Writing output | 0.000 | 0.0% |
| Create Dataset wall | 385.161 | — |

**Feature engine likely bottleneck:** `True`

## Notes

- Create Dataset (master_build) builds feature rows + prediction target columns (horizons) in one day path. Exporting a train-ready parquet from an existing master DB is a separate 'Create Dataset from Master' flow.
- Per-row feature path is dict + controllers (not Polars). DuckDB/Polars appear mainly in dataset export / frame IO, not in build_feature_raw_for_row.
- chain_replay_ml.performance.benchmark exercises controllers directly; this report is for the production Create Dataset entry points.

## Meta

```json
{
  "job_id": "1e34b081-7d6e-46d9-a9e7-2ae91144c696",
  "market": "NIFTY",
  "rows": 2584218,
  "feature_count": 206,
  "target_count": 3,
  "sources": 1
}
```
