# Web cleanup validation — Phase 1

**Date:** 2026-08-07  
**Scope:** `C:\Users\admin\PycharmProjects\AruMLStudio`  
**Phase:** 1 — remove legacy FastAPI WebSocket transport only

---

## Files removed

| File | Reason |
|------|--------|
| `angelone/chart/chain_replay_ml/model_across_days_ws.py` | Orphan FastAPI `/ws/model-across-days` transport; no in-repo imports |

**Total:** 1 file

---

## Remaining shared modules (unchanged)

These were **not** deleted (Phase 2 will refactor web references in place):

| Module | Role |
|--------|------|
| `angelone/chart/master_dataset_tk/ui_util.py` | Tk helpers + legacy `open_web_*` / `webbrowser` (Phase 2) |
| `angelone/chart/master_dataset_tk/registry_panel.py` | Dataset Registry; browser fallback when no callback (Phase 2) |
| `angelone/chart/master_dataset_tk/model_registry_detail.py` | Model Registry detail; some actions still call `open_web_*` (Phase 2) |
| `research/atm_band_ml/chart_tick_client.py` | HTTP client to chart server ticks API (Phase 2+ / local data) |
| `chain_replay_ml/model_day_comparison.py` | Across-days compare library (no longer imported by deleted WS module) |
| `chain_replay_ml/training/registry.py`, `orchestrator.py` | `report_url` web path metadata (Phase 2) |
| `chain_replay_ml/dataset_builder/master_registry_export.py` | `registry_url` metadata (Phase 2) |

---

## Launcher verification

| Check | Result |
|-------|--------|
| `master_dataset_manager.py` → import `MLResearchStudioApp` | **Pass** (`import_ok MLResearchStudioApp`) |
| Feature Registry catalog load | **Pass** (206 features via `build_feature_registry_catalog`) |
| `registry_service` / `feature_registry_service` import | **Pass** |
| Full GUI `mainloop()` | **Not automated** — run manually: `python master_dataset_manager.py` |

**Note:** Primary shell already routes model lifecycle via Tk (`app._open_model_builder_lifecycle` → Create Model tab). Dataset Registry in `app.py` sets `_on_open_builder`, so the browser branch in `registry_panel.py` should not run in normal use; Phase 2 will remove that dead path.

---

## Unit test results

```text
cd C:\Users\admin\PycharmProjects\AruMLStudio
python -m unittest discover tests -q
```

| Metric | Value |
|--------|--------|
| Tests run | 45 |
| Result | **OK** |
| Time | ~8.7s |

No `ImportError` / `ModuleNotFoundError` related to the deleted module.

---

## Desktop verification (automated subset)

| Area | Verification |
|------|----------------|
| ML Studio app module | Import OK |
| Dataset / Feature Registry services | Import + catalog build OK |
| Model Lab / panels | Not individually exercised in CI; covered by app import graph |
| Research panels | No change in Phase 1 |

---

## FastAPI / web server surface

After Phase 1, **no** Python module in AruMLStudio imports `fastapi` for route registration.

Remaining web **references** (browser URLs, `/ml/...` strings) are documented in **Phase 2** in `migration/web_cleanup_report.md`.

---

## Next step

Implement **Phase 2: Web Reference Cleanup** per `migration/web_cleanup_report.md` when ready.
