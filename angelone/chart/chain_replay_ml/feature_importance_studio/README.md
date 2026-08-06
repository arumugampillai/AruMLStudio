# Feature Importance Studio — implementation notes (Phase 4.1)

**Spec:** [`docs/project-main/feature-studio/FEATURE_IMPORTANCE_STUDIO.md`](../../../../docs/project-main/feature-studio/FEATURE_IMPORTANCE_STUDIO.md)

## Status

Milestones **1–4 complete** (compute + artifacts + UI). Feature Studio shell hosts this tab.

## Layout

```text
feature_importance_studio/
  api.py            ← run_feature_importance_studio
  compute.py        ← orchestrate load → native → perm → shap → write
  native.py         ← gain / weight / cover
  permutation.py    ← holdout permutation (mean / std / rank)
  shap.py           ← TreeSHAP mean |SHAP|
  comparison.py     ← joined UI contract rows
  writer.py         ← JSON under package/feature_importance_studio/
  types.py
  __main__.py       ← CLI
```

## CLI

```bash
cd angelone/chart
set PYTHONPATH=.
python -m chain_replay_ml.feature_importance_studio ^
  --data-dir data ^
  --model YourModelName ^
  --holdout-max-rows 20000 ^
  --permutation-repeats 5
```

## Artifact contract

```text
models/<Model>/feature_importance_studio/
  native_xgb.json
  permutation.json
  shap.json
  comparison.json      ← Feature | gain | weight | cover | perm | shap | ranks
  run_meta.json        ← model, dataset, holdout rows, engine backend, timings
```

No retraining. Holdout via Dataset Engine (`load_training_xy` / `auto` fallback).
