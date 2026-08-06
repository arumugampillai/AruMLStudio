# Apps restructure report

**Date:** 2026-08-07  
**Scope:** `C:\Users\admin\PycharmProjects\AruMLStudio`  
**Phases:** (1) ML packages → `apps/`; (2) `static/` + `tick_data_paths.py` → `apps/`, remove empty `angelone/chart/`.

---

## Phase 2 — files / folders moved

| Source | Destination |
|--------|-------------|
| `angelone/chart/static/` | `apps/static/` (includes `ml_schema_registry.json`) |
| `angelone/chart/tick_data_paths.py` | `apps/tick_data_paths.py` |

**Removed after validation:** `angelone/chart/` (empty after moves; no remaining tree under `angelone/` in repo).

### Phase 1 — packages (unchanged from prior pass)

| Package | New location |
|---------|----------------|
| `chain_replay_ml/` | `apps/chain_replay_ml/` |
| `feature_intelligence/` | `apps/feature_intelligence/` |
| `live_inference/` | `apps/live_inference/` |
| `master_dataset_tk/` | `apps/master_dataset_tk/` |
| `ml_phase1/` | `apps/ml_phase1/` |
| `storage/` | `apps/storage/` |

---

## Imports and path references updated

### Launcher (`master_dataset_manager.py`)

- `sys.path`: **`apps/` only** (no `angelone/chart/`).
- Help text: default bundled project dir is **`apps/`**.

### `apps/path_config.py`

- `CHART_DATA_ROOT` = `APPS_DIR` (`apps/`).
- `STATIC_DIR` = `apps/static/`.
- `ensure_ml_studio_paths()` adds `apps/` + repo root only.

### Schema / static

- `apps/chain_replay_ml/dataset_builder/schema_registry.py` — `_SCHEMA_PATH` via `STATIC_DIR`.
- `apps/chain_replay_ml/dataset_builder/scripts/generate_schema_registry.py` — writes under `STATIC_DIR`.

### Project resolution

- `apps/master_dataset_tk/project_config.py` — repo root resolves to `apps/` (legacy `angelone/chart` still accepted).
- `apps/master_dataset_tk/ormp_service.py` — `_repo_root_from_chart_dir()` treats `apps/` and legacy `angelone/chart`.
- `apps/master_dataset_tk/feature_intelligence_studio_panel.py` — user message references `apps/data`.

### ORMP / research

- `ormp/config.py` — `default_candle_db_path()` → `apps/data/angel_historic_bars.db`.
- `research/atm_band_ml/*.py` — `_CHART_DIR` / `chart_dir` → repo `apps/`.

### Tests

- `tests/test_pyi_scipy_distn_fix.py`, `tests/test_pyi_chain_replay_ml_data_fix.py` — `sys.path`: `apps/` only.
- `apps/chain_replay_ml/tests/test_project_config.py` — added `test_resolve_chart_dir_repo_root_apps`; legacy `angelone/chart` test retained.

### Package import style (unchanged)

With `apps/` on `sys.path`:

- `from tick_data_paths import ...`
- `from chain_replay_ml...`, `from master_dataset_tk...`, etc.

No `angelone.chart.static` or `angelone.chart.tick_data_paths` imports remain in the codebase (verified by search).

### Phase 1 bulk rewrites (representative)

| Area | Change |
|------|--------|
| `_CHART_DIR` dirname hacks (~30 modules) | `from path_config import CHART_DATA_ROOT as _CHART_DIR` |
| Script `sys.path` bootstraps (~14 modules) | `ensure_ml_studio_paths()` |
| `storage/chain_replay_export.py` | cache under `CHART_DATA_ROOT/data/cache` |

Bulk helpers: `migration/_fix_chart_paths.py`, `migration/_fix_sys_path.py`.

---

## Validation results

| Check | Result |
|-------|--------|
| `python -m compileall -q apps ormp tests master_dataset_manager.py research/atm_band_ml` | **Pass** |
| `python -m unittest discover -s tests -p "test_*.py"` | **Pass** (45 tests) |
| `MLResearchStudioApp()` smoke (800ms Tk lifecycle) | **Pass** (`gui_smoke_ok`) |
| Grep `angelone.chart.static` / `angelone.chart.tick_data_paths` | **No matches** |
| `angelone/chart/` directory | **Removed** (post-validation) |

**Note:** `python -m unittest discover -s apps/chain_replay_ml/tests` was not part of the requested gate; it reports pre-existing failures/errors unrelated to this move.

---

## Notes for developers

- **Code:** put `apps/` on `PYTHONPATH` (launcher does this automatically).
- **Data / static:** default bundled layout is `apps/data/`, `apps/static/`.
- **CLI:** `python master_dataset_manager.py` from repo root.
- **Schema:** `python apps/chain_replay_ml/dataset_builder/scripts/generate_schema_registry.py` (with `apps` on path).

---

## Broker code

Broker modules under `angelone/` were not modified for this restructure. The former `angelone/chart/` ML tree is fully relocated under `apps/`.
