# Web cleanup report — AruMLStudio

**Scope:** `C:\Users\admin\PycharmProjects\AruMLStudio` only  
**Date:** 2026-08-07  
**Status:** Phase 1 complete. **Phase 2 complete (2026-08-07)** — see `migration/browser_dependency_report.md`.

**AruNeo / Kotak-AruNeo:** not accessed.

---

## Executive summary

AruMLStudio was already extracted as a **standalone Tkinter ML Research Studio**. This tree does **not** contain a legacy web application shell (no `main.py` / `chart_server_controller.py`, no Flask/FastAPI app instance, no HTML/JS/Vue frontend, no `uvicorn` entrypoint).

What remains of the **legacy web stack** is:

1. **One orphaned FastAPI WebSocket adapter** (`model_across_days_ws.py`) with **no in-repo caller** for `register_model_across_days_ws_routes`.
2. **Desktop → browser bridges** (`ui_util.py`, `registry_panel.py`, `model_registry_detail.py`) that open `http://127.0.0.1:8000/ml/...` if a chart server were still running.
3. **HTTP client** for an external Angel chart server (`research/atm_band_ml/chart_tick_client.py`) — not a web server, but a **client** of the old chart HTTP API.
4. **Metadata URL strings** (`/api/ml/...`, `/ml/create-dataset#...`) in training/registry code — inert without a web host.

**Recommended deletion after approval (Category A):** **1 file** — `angelone/chart/chain_replay_ml/model_across_days_ws.py`.

**Do not delete** `model_day_comparison.py`, `recompute_2_1_ratio.py`, or `registry_backtest.py` as part of web cleanup — they are core ML/backtest libraries (Category C), not web server code.

---

## Step 1 — Legacy web entry points

| Entry / surface | Path | Framework | Registered in repo? | Notes |
|-----------------|------|-----------|---------------------|-------|
| Model across-days WebSocket | `angelone/chart/chain_replay_ml/model_across_days_ws.py` | FastAPI (`WebSocket`, `@app.websocket("/ws/model-across-days")`) | **No** — `register_model_across_days_ws_routes(app, ...)` is never imported | Only web-server transport left in tree |
| *(absent)* Chart / ML HTTP app | — | Flask / FastAPI / Bottle / Dash / Tornado | N/A | Not present in AruMLStudio |
| Desktop launcher | `master_dataset_manager.py` → `master_dataset_tk.app` | Tkinter | Yes | Primary product entry |
| FIC CLI | `angelone/chart/feature_intelligence/__main__.py` | CLI | Yes | Desktop-adjacent tooling |
| Chain replay CLI | `angelone/chart/chain_replay_ml/__main__.py` | CLI | Yes | Export pipeline |
| PyInstaller build | `build.py` | — | Yes | Desktop packaging |

**Searched (no matches as app entry):** `chart_server_controller.py`, `flask`, `uvicorn`, `gunicorn`, `socketio`, `*.html`, `*.js` / `*.vue` frontend bundles.

---

## Step 2 — AST dependency graph (method)

- Parsed **1,205** Python files under AruMLStudio (excluding `.venv`, `__pycache__`, `build/`, `dist/`).
- Built import edges via `ast` (`import` / `from ... import`).
- **Desktop closure:** BFS from seeds `master_dataset_manager`, `build`, `feature_intelligence.__main__`, `chain_replay_ml.__main__` → **~598** modules resolved on `sys.path` conventions (`angelone/chart` + `research` + repo root).
- **Web closure:** BFS from seed `chain_replay_ml.model_across_days_ws` → **~200** modules (includes large backtest subgraph).

**Modules reachable from web seed but not from desktop seeds** (static import graph only):

| Module | Role |
|--------|------|
| `chain_replay_ml.model_across_days_ws` | FastAPI WebSocket routes |
| `chain_replay_ml.model_day_comparison` | Across-days compare logic (only static importer: `model_across_days_ws`) |
| `chain_replay_ml.recompute_2_1_ratio` | Also imported by `model_analysis`, `registry_backtest`, `replay_live_sim` (not desktop Tk) |
| `chain_replay_ml.registry_backtest` | Registry scoring helpers |

**Important:** `recompute_2_1_ratio` / `registry_backtest` are **not** web-only modules — they are shared ML infrastructure with **no** FastAPI dependency. The graph marks them “web-only reachable” only because desktop Tk does not currently import that subgraph; they must **remain** (Category C).

**FastAPI mention in source (not all are servers):**

| File | Role |
|------|------|
| `chain_replay_ml/model_across_days_ws.py` | **Web server transport** |
| `master_dataset_tk/build_service.py` | Docstring: in-process, **no** FastAPI |
| `storage/chain_replay_export.py` | Docstring: **no** FastAPI; uses `requests` to Angel public scrip master URL |

---

## Step 3 — Classification

### Category A — WEB ONLY (safe to remove after approval)

| File | Reason | Imported by (in-repo) | Safe to remove? | Risk |
|------|--------|------------------------|-----------------|------|
| `angelone/chart/chain_replay_ml/model_across_days_ws.py` | FastAPI WebSocket route registration for `/ws/model-across-days`; legacy web UI transport | **None** (no imports of `register_model_across_days_ws_routes` or module) | **Yes** | **Low** — desktop does not reference module; removes optional `fastapi` import site |

### Category B — SHARED (desktop + legacy web references; keep files)

| File | Reason | Imported by | Safe to remove? | Risk |
|------|--------|-------------|-----------------|------|
| `angelone/chart/master_dataset_tk/ui_util.py` | Tk helpers; **also** `open_web_model_builder`, `open_model_builder_lifecycle`, `open_web_model_registry` | `model_registry_detail.py`, registry flows | **No** (file) — refactor web helpers later | **High** if whole file deleted |
| `angelone/chart/master_dataset_tk/registry_panel.py` | Dataset Registry UI; optional `webbrowser.open` to `:8000/ml/model-builder` | `app.py` | **No** | **High** |
| `angelone/chart/master_dataset_tk/model_registry_detail.py` | Model Registry UI; buttons call `open_web_*` | Model registry panel | **No** | **High** |
| `angelone/chart/chain_replay_ml/training/registry.py` | Model registry data; `report_url` paths for old web reports | Training / registry panels | **No** | **Medium** — URLs unused in Tk |
| `angelone/chart/chain_replay_ml/training/orchestrator.py` | Same `report_url` metadata | Model builder runner | **No** | **Medium** |
| `angelone/chart/chain_replay_ml/dataset_builder/master_registry_export.py` | Export metadata `registry_url: /ml/create-dataset#registry` | Master export / registry | **No** | **Low** |
| `angelone/chart/chain_replay_ml/dataset_builder/progress.py` | Progress event shape (doc references `/ws/ml-dataset`) | Dataset build (Tk + library) | **No** | **High** if deleted |
| `angelone/chart/chain_replay_ml/training/progress.py` | WS-oriented docstring; used in-process by Tk runner | `model_builder/runner.py` | **No** | **High** |
| `angelone/chart/chain_replay_ml/training/wf_progress.py` | WS-oriented docstring | Walk-forward / HPO | **No** | **Medium** |
| `angelone/chart/chain_replay_ml/dataset_builder/audit_progress.py` | WS-oriented docstring | Dataset audit | **No** | **Medium** |
| `research/atm_band_ml/chart_tick_client.py` | **HTTP client** to legacy chart server (`/replay/{token}/ticks`) | `band_evaluator.py` → `atm_band_ml` package | **No** | **High** — breaks live tick path unless refactored to local data |
| `research/atm_band_ml/tick_timeline.py` | Documents chart server tick JSON | `atm_band_ml`, `replay_live_sim` | **No** | **Medium** |
| `angelone/chart/storage/chain_replay_export.py` | Broker HTTP fetch for scrip master (not local chart server) | CLI `chain_replay_ml.__main__` | **No** | **High** |

### Category C — DESKTOP / ML STUDIO (keep)

| Area | Approx. files | Notes |
|------|----------------|-------|
| `angelone/chart/master_dataset_tk/` | ~150+ | Tk shell, panels, in-process services (`*_service.py` explicitly no chart server) |
| `angelone/chart/chain_replay_ml/` | ~500+ | Dataset builder, training, model lab, registries (minus Category A file) |
| `angelone/chart/feature_intelligence/` | ~80+ | FIC registry / CLI |
| `angelone/chart/storage/`, `ml_phase1/` | — | Data / export |
| `research/` | ~30+ | Strategy math, ATM band ML (includes chart **client**, not server) |
| `ormp/`, `tests/`, `build.py`, `pyi_hooks/` | — | Support / packaging / tests |
| `angelone/chart/static/ml_schema_registry.json` | 1 | Generated schema artifact for Feature Registry |

**Orphan library note (Category C):** `chain_replay_ml/model_day_comparison.py` had been referenced only by the removed WebSocket module. It remains available for future Tk “across days” UI; not part of Phase 1.

---

## Phase 1 — Web server removal (done)

- **Deleted:** `angelone/chart/chain_replay_ml/model_across_days_ws.py`
- **Validation:** see `migration/web_cleanup_validation.md`

---

## Phase 2 — Web reference cleanup (planned; do not delete whole files)

Goal: **100% desktop-native** ML Studio — no dormant browser integration or localhost chart URLs.

### Scope rules

- **Do not delete** `ui_util.py`, `registry_panel.py`, `model_registry_detail.py`, or `chart_tick_client.py`.
- Refactor in place: remove web-only code paths; keep Tk UI behavior (often already native via `app.py` callbacks).

### Checklist

| Action | Locations (initial inventory) |
|--------|-------------------------------|
| Remove all `webbrowser.open(...)` | `registry_panel.py`, `ui_util.py` |
| Remove `http://127.0.0.1:8000/...` literals | `ui_util.py`, `registry_panel.py` |
| Remove `/ml/...` URL builders and dead helpers | `ui_util.py`: `model_builder_url`, `open_web_model_builder`, `open_model_builder_lifecycle`, `open_web_model_registry` |
| Wire model lifecycle to Tk only | `model_registry_detail.py` → use same pattern as `app._open_model_builder_lifecycle` (Create Model tab + `load_lifecycle_preset`) instead of `ui_util.open_model_builder_lifecycle` |
| Dataset Registry “Train” fallback | `registry_panel.py` — when `_on_open_builder` is set (it is in `app.py`), browser branch is dead; remove fallback + chart-server copy |
| Neutralize or drop web `report_url` / `registry_url` metadata | `training/registry.py`, `training/orchestrator.py`, `dataset_builder/master_registry_export.py` — use empty string, Tk route id, or remove from export JSON |
| Chart tick source (optional / later) | `research/atm_band_ml/chart_tick_client.py` — replace HTTP chart server with local replay DB / parquet; update `band_evaluator.py` call sites |

### Out of scope for Phase 2 (docstrings only)

- `model_builder/__init__.py`, `training_panel.py` — comments referencing web parity; update text when convenient.

### Success criteria (Phase 2)

- `rg` / search finds **no** `webbrowser.open`, **no** `127.0.0.1:8000`, **no** `open_web_` in `master_dataset_tk/`.
- All former browser actions navigate within Tk (existing panels).
- Unit tests and manual smoke: Dataset Registry, Model Registry, Create Model, Model Lab.

---

## Validation (Phase 1)

See `migration/web_cleanup_validation.md`.
