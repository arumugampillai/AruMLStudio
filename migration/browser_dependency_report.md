# Browser dependency report — Phase 2 complete

**Date:** 2026-08-07  
**Scope:** `C:\Users\admin\PycharmProjects\AruMLStudio` (Python sources; migration docs excluded)

---

## Goal

Remove **browser dependencies** from ML Research Studio (not “delete web servers” — Phase 1 already removed the FastAPI WebSocket module).

---

## Completion criteria (Python `*.py`)

| Pattern | Matches in `*.py` | Status |
|---------|-------------------|--------|
| `webbrowser.open(` | **0** | Pass |
| `http://127.0.0.1:8000` | **0** | Pass |
| `open_web_` helper functions | **0** | Pass |
| `/ml/` desktop navigation URLs | **0** | Pass |
| `/api/ml/` report URLs | **0** | Pass |

Verification command (PowerShell, repo root):

```powershell
rg "webbrowser\.open" --glob "*.py"
rg "127\.0\.0\.1:8000" --glob "*.py"
rg "open_web_" --glob "*.py"
rg "/ml/" --glob "*.py"
rg "/api/ml/" --glob "*.py"
```

All return no matches (exit code 1 from `rg` = no hits).

---

## Files modified (Phase 2)

| File | Change |
|------|--------|
| `angelone/chart/master_dataset_tk/ui_util.py` | Removed `model_builder_url`, `open_web_*`, `open_model_builder_lifecycle`; kept `open_path` |
| `angelone/chart/master_dataset_tk/registry_panel.py` | Train action uses `_on_open_builder` only; removed `webbrowser` fallback |
| `angelone/chart/master_dataset_tk/model_registry_detail.py` | Retrain / View Trials use `on_lifecycle` → Tk Create Model (`complete_optimization` for trials) |
| `angelone/chart/chain_replay_ml/training/registry.py` | `report_url` cleared (no web paths) |
| `angelone/chart/chain_replay_ml/training/orchestrator.py` | `report_url` cleared in training results |
| `angelone/chart/chain_replay_ml/dataset_builder/master_registry_export.py` | `registry_url` cleared |
| `angelone/chart/master_dataset_tk/model_builder/__init__.py` | Docstring (no `/ml/` reference) |
| `angelone/chart/master_dataset_tk/model_builder/training_panel.py` | Docstring (no `/ml/` reference) |

**Files deleted:** none

---

## Behavior preserved

- Dataset Registry **Train** → `app._open_model_builder_dataset` (Create Model with dataset).
- Model Registry **Retrain / Optimization / Feature Optimization** → `app._open_model_builder_lifecycle`.
- **View Trials** → `on_lifecycle(model, "complete_optimization")` (Create Model optimization flow).
- UI layout and button labels unchanged.

---

## Validation run

| Step | Result |
|------|--------|
| `py_compile` on each modified file | Pass |
| `python -m unittest discover tests` | **45 tests OK** |
| Tk smoke (`MLResearchStudioApp` create + `after(800, destroy)`) | **gui_smoke_ok** |
| `python master_dataset_manager.py --help` | Pass |

---

## Out of scope (intentional)

| Item | Notes |
|------|--------|
| `research/atm_band_ml/chart_tick_client.py` | HTTP **client** to chart tick API — not `webbrowser`; separate data-source migration |
| Docstrings mentioning `/ws/ml-dataset` in progress modules | WebSocket **event shape** docs for in-process progress, not browser navigation |
| `migration/*.md` | Historical Phase 1/2 planning text may still mention old URLs |

---

## Summary

ML Research Studio no longer opens a browser or encodes localhost chart-server navigation URLs in Python code. Model Builder actions are routed through the existing Tk **Create Model** workflow.
