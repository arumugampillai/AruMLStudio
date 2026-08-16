# Create Dataset Timing Report

Generated: `2026-08-16T14:29:48.167782+00:00`

**Scope:** Create Dataset / master_build (raw ticks → features → master SQLite)

## Numba (production path)

- Numba enabled: **YES**
- `ARUNEO_FEATURE_NUMBA` env: `unset`
- Numba available: `True`
- Numba kernel calls: `21,511`
- Python fallback calls: `0`
- Python fallback active: `False`

## Phase timings

| Phase | Seconds | Share |
|-------|--------:|------:|
| Loading ticks | 52.481 | 52.6% |
| Feature computation | 40.790 | 40.9% |
| Prediction targets | 0.000 | 0.0% |
| DuckDB/SQLite insert | 0.343 | 0.3% |
| Polars / DuckDB frame IO | 0.000 | 0.0% |
| Writing output | 0.000 | 0.0% |
| Create Dataset wall | 99.830 | — |

**Feature engine likely bottleneck:** `True`

## Notes

- Create Dataset (master_build) builds feature rows + prediction target columns (horizons) in one day path. Exporting a train-ready parquet from an existing master DB is a separate 'Create Dataset from Master' flow.
- Per-row feature path is dict + controllers (not Polars). DuckDB/Polars appear mainly in dataset export / frame IO, not in build_feature_raw_for_row.
- chain_replay_ml.performance.benchmark exercises controllers directly; this report is for the production Create Dataset entry points.

## Meta

```json
{
  "job_id": "f4b3f239-a99b-42f8-b31f-c5a03005e33e",
  "market": "NIFTY",
  "rows": 65370,
  "feature_count": 203,
  "target_count": 2,
  "sources": 1
}
```
