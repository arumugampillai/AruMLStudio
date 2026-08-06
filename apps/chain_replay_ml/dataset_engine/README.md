# Dataset Engine — implementation

**ADR (frozen):** [`docs/architecture/DATASET_ENGINE.md`](../../../../docs/project-main/architecture/DATASET_ENGINE.md)

This package implements the Dataset Engine. **Edit this README and the Python modules** as the design is proven in code. Do not amend the ADR unless implementation reveals an architectural flaw.

## Layout

```text
dataset_engine/
  README.md                 ← this file (implementation notes)
  __init__.py               ← public exports
  api.py                    ← query_dataset / stream_dataset
  types.py                  ← SampleSpec, ExecutionStats, QueryResult, …
  planner.py                ← query planning + partition selection
  backends/
    __init__.py
    base.py                 ← Backend protocol
    duckdb_backend.py       ← first backend
```

## Consumer contract

```python
from chain_replay_ml.dataset_engine import query_dataset

result = query_dataset(
    dataset_id,
    columns=[...],
    filters={...},
    sample=...,
)
result.table   # pyarrow.Table
result.stats   # ExecutionStats
```

Consumers must not import backends directly.

## Roadmap (from ADR)

1. Skeleton: `query_dataset` → `{table, stats}` + DuckDB + Arrow + validate + thin planner  
2. Model Builder: Premium / DTE / ATM / Session  
3. Analysis Lab: Corr / MI / HCA / Discovery  
4. Prediction joins  
5. Optional Polars (measured only)  
6. Later: `stream_dataset`, richer planner, schema-version checks  

## Principle

Read-optimized only. No Master feature generation or mutation.

## Model Builder (first consumer)

Create Model load path: `training/dataset_loader.load_training_xy`.

Default is **Phase 1**: `ARUNEO_DATASET_ENGINE=auto` with Pandas fallback on any Engine failure.
See **Rollout (phased)** below.

UI and XGBoost training are unchanged. Premium Selection is pushed into Engine filters when enabled.

Load metrics land on training metadata / package `metadata.json`:

```text
metadata["dataset_load"] = {
  backend, load_time_sec, peak_rss_mb,
  rows_returned, columns_returned,
  partitions_scanned, partitions_pruned,
  engine_fallback, engine_fallback_reason, ...
}
```

A/B helper (no train) — compares metrics **and** training matrices:

```python
from chain_replay_ml.training.dataset_loader import compare_training_load_backends
report = compare_training_load_backends(data_dir, config)
# report["rows_match"], report["matrices_equal"]
# report["pandas"] / report["dataset_engine"]:
#   load_time_sec, peak_rss_mb, rows_returned, columns_returned, partitions_scanned
```

Both load paths stabilize row order (`trading_day`, `timestamp`, `token`) before
`select_xy`, so Engine vs Pandas matrices can be byte-compared for parity.

Objective metrics:

| Metric | Notes |
|--------|--------|
| Dataset load time | `load_time_sec` |
| Peak RAM | `peak_rss_mb` (process RSS span) |
| Rows returned | After premium filter |
| Columns returned | After column prune |
| Partitions scanned | Planner/backend (`partitions_scanned`) |
| Training matrices | `matrices_equal` (X and y) |
| Training time (ex-load) | Measure separately; expect unchanged |
| Engine fallback | `engine_fallback` + `engine_fallback_reason` (Phase 2 watch) |

### Rollout (phased) — status

| Phase | Status | Notes |
|-------|--------|-------|
| **1** Default `auto` + Pandas fallback + telemetry | ✅ Complete | Model Builder production posture |
| **2** Usage-based Create Model observation | ✅ Complete | Multi-dataset (`025644`, `094409`), no unexpected fallbacks |
| **3A** Correlation → `query_dataset()` | ✅ Complete | Full-frame Engine/Pandas matrix parity |
| **3B** Engine rollout gate (broader coverage) | ✅ Complete | MI / HCA / Discovery may adopt Engine later; not blocking research work |
| **4** Higher-level ML research capabilities | ▶ Current | Build on the stable engine (below) |

Pandas fallback remains until the **pre-removal stress gate** passes (see below). Do **not** flip `auto` → `on` / remove fallback yet.

#### Phase 4 — research capabilities (current focus)

**Dataset Engine:** Qualified for ML Research Platform (`auto` + Pandas fallback).

**Phase 4.1 — Feature Importance Studio** (next implementation):

Spec: [`docs/project-main/feature-studio/FEATURE_STUDIO_ARCHITECTURE.md`](../../../../docs/project-main/feature-studio/FEATURE_STUDIO_ARCHITECTURE.md)

```text
Model package → Native (Gain/Weight/Cover) → Permutation (holdout)
             → TreeSHAP (holdout) → Comparison table → UI
```

No retraining; reuse package + holdout via Dataset Engine.

Scaffold: `chain_replay_ml/feature_importance_studio/`

Then: Distribution → Drift → Multi-model Comparison → Diagnostics.

#### Dataset Engine Qualification Test (pre-fallback-removal)

Before declaring the Engine production-ready *without* Pandas fallback:

```bash
cd angelone/chart
set PYTHONPATH=.

python -m chain_replay_ml.dataset_engine.stress_test
python -m chain_replay_ml.dataset_engine.stress_test --include-5m
```

| Metric | Why |
|--------|-----|
| Wall time | Overall scalability |
| Peak RSS | Memory growth |
| Rows/sec | Throughput |
| Columns/sec | Wide-table behavior |
| Partitions scanned/pruned | Planner effectiveness |
| Fallback count | Should remain zero |
| Engine exceptions | Should remain zero |

Maturity bar: healthy metrics + parity on the **1M × 500 feature** case (and 5M suite when run). Cases include missing values, constants, high-cardinality, and mixed numeric dtypes.
### Env reference

| Env `ARUNEO_DATASET_ENGINE` | Behaviour |
|-----------------------------|-----------|
| `auto` (**default**, Phase 1) | Engine when `duckdb` importable; else pandas. Runtime Engine errors → pandas fallback |
| `on` | Prefer Engine; same runtime pandas fallback |
| `off` | Force pandas (reference / debug) |

### Decision gate history (Create Model pair)

Already satisfied before Phase 1:

1. Parity tests pass  
2. Training matrices identical (off vs on)  
3. Load time improved or equal  
4. Peak RSS improved (separate-process bench)  
5. Holdout + WF metrics identical on Create Model pair  

Evidence: `data/models/create_model_engine_ab_20260731_015113.json`.

### Fair Peak RSS (separate processes)

Do **not** compare Peak RSS from in-process A/B (`compare_training_load_backends`).
The second path inherits allocations from the first.

Use two independent processes:

```bash
cd angelone/chart
set PYTHONPATH=.

python -m chain_replay_ml.dataset_engine.bench_training_load_ab ^
  --data-dir data ^
  --dataset analysis_206r_193p_3s_20260730_094409 ^
  --premium-min 15 --premium-max 100
```

Process A = `ARUNEO_DATASET_ENGINE=off`, Process B = `on`. Summary prints wall load time, peak RSS, and checksum match on X/y.

### Create Model pair checklist (reference)

What to compare for transparent replacement:

| Category | Expected outcome |
|----------|------------------|
| Dataset checksum / shape | Identical (`schema_hash.txt`, `validation_hash.txt`) |
| Feature count | Identical |
| Training rows | Identical |
| Holdout metrics (MAE/RMSE/etc.) | Identical (within floating-point tolerance) |
| Walk-forward metrics | Identical (within floating-point tolerance) |
| Selected features (if deterministic) | Identical, or explain any intentional nondeterminism |
| Dataset load metrics | Improved or equal (`load_time_sec`, `peak_rss_mb`; `rows_returned` / `columns_returned` identical; `partitions_*` are Engine-only) |

```bash
cd angelone/chart
set PYTHONPATH=.

python -m chain_replay_ml.dataset_engine.compare_create_model_runs ^
  --off data/models/YourModel_OFF ^
  --on  data/models/YourModel_ON
```

```bash
python -m chain_replay_ml.dataset_engine.run_create_model_engine_ab
```

## Tests

```text
tests/dataset_engine/
  test_filter_parity.py
  test_sampling_parity.py
  test_partition_pruning.py
  test_schema_validation.py
tests/test_training_dataset_engine_load.py
```

```bash
cd angelone/chart
python -m unittest discover -s chain_replay_ml/tests/dataset_engine -v
python -m unittest chain_replay_ml.tests.test_training_dataset_engine_load -v
python -m unittest chain_replay_ml.tests.dataset_engine.test_compare_create_model_runs -v
```

Install `duckdb` (and `pyarrow`) for Engine integration tests. Without DuckDB, planner + schema KeyError + pandas training-load tests still run.

## Status

| Phase | Status |
|-------|--------|
| 1–3B Dataset Engine rollout | ✅ Complete (`auto` + fallback retained) |
| 4 Higher-level ML research capabilities | ▶ Current — start with Feature Importance Studio |
| Dataset Engine Qualification Test | Default suite ✅ (incl. 1M×500); 5M×50 ✅ (chunked); keep fallback until ops sign-off |