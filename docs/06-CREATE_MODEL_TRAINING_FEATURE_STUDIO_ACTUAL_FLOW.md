# AruMLStudio Actual Flow — Create Model &rarr; Training &rarr; Artifacts &rarr; Feature Studio &rarr; Model Registry &rarr; Production Validation

---

## 1. Executive Summary & Purpose

This document provides a strictly factual, source-verified trace of the entire end-to-end training and diagnostics pipeline in **AruMLStudio**:

```
                                  END-TO-END EXECUTION FLOW
                                  
Create Model UI ──► Training Orchestrator ──► Artifact Generation ──► Post-Training Studios ──► Model Registry ──► Feature Studio ──► Production Validation
 (ModelBuilder)       (train_model)           (save_model_pkg)       (Imp/Dist/Drift)         (models/<name>/)   (Load Artifacts)    (Unseen Days Replay)
```

Every section, class name, function call, metadata field, and file path in this document was verified directly against the current codebase.

---

## 2. Create Model UI: Sections & Controls

The **Create Model** workbench is implemented in [`apps/master_dataset_tk/model_builder/panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_builder/panel.py) via `ModelBuilderPanel` and backed by `ModelBuilderState` in [`state.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_builder/state.py).

### 2.1. Panel Structure & Section Inventory

| Section # | Section Title | UI Container / Widget | State / Configuration Produced |
|---|---|---|---|
| **Top Banners** | Feature Preset / Lifecycle | `_preset_banner`, `_lifecycle_banner` | Preset label, active lifecycle mode (`"train"`, `"retrain"`, `"complete_optimization"`). |
| **Section 1** | **Dataset** | `ds_frame` (`ttk.Combobox`, `_compat_frame`) | `state.dataset`: Selected Parquet analysis dataset name (e.g. `"analysis_nifty_6s_exp005"`). |
| **Section 2** | **Target** | `tgt_frame` (`reg_panel`, `clf_panel`, `strat_panel`) | `state.target` (e.g. `"future_ltp_5m"`), `state.prediction_type` (`"regression"` or `"classification"`). |
| **Section 3** | **Prediction Type** | `pred_frame` (Horizon radio buttons) | `state.prediction_horizon_sec` (e.g. `300`). |
| **Section 4** | **Algorithm** | `algo_frame` (Algorithm radio buttons) | `state.algorithm` (`"xgboost"`, `"lightgbm"`, `"catboost"`, `"random_forest"`). |
| **Section 5** | **Feature Selection** | `feat_frame` (`_feat_tree_nb`, 3 tabs + post-training frame) | `state.features`: Selected list of feature column names.<br>• **Tab 1**: Feature Registry<br>• **Tab 2**: Base Pipeline<br>• **Tab 3**: Selected Experimental<br>• **Post-Training Frame**: Checkboxes for `importance` (True), `distribution` (True), `drift` (True). |
| **Section 6** | **Data Split** | `split_frame` (`_wf_panel`, `_wf_preview_panel`) | `state.split`: Walk-forward split configuration dictionary (`strategy="walk_forward"`, `n_folds=5`, `train_window_size=3000`, `validation_window_size=1000`, `test=15`). |
| **Section 7** | **Algorithm Parameters**| `self._params_frame` (`_lifecycle_hpo_panel`) | `state.parameters`: Algorithm hyperparameters (e.g. `max_depth=6`, `learning_rate=0.05`, `n_estimators=1000`, `early_stopping_rounds=50`), plus HPO config (`n_trials=25`). |
| **Section 8** | **Model Information** | `info_frame` (`ttk.Entry`) | `state.model_name`: User-specified or auto-suggested name (`Future_LTP_5m_WF_1168f_XGB_2243_14`). |
| **Section 9** | **Feature Policy Preview**| `preview_frame` (`ScrolledText`) | Read-only summary of active feature constraints and selection rules. |
| **Section 10**| **Configuration Summary**| `sum_frame` (`_train_btn`) | Read-only configuration summary card + **"Train Model"** action button. |

---

## 3. Train Model Execution Flow & Call Chain

When the user clicks **"Train Model"** (`self._train_btn`):

```
ModelBuilderPanel._start_training()
    │
    ▼
service.validate_config(data_dir, cfg)
    │
    ▼
ModelBuilderPanel._enrich_training_config(cfg)
    │
    ▼
ModelTrainingPanel (shows dashboard) ──► ModelTrainingRunner.start() (daemon thread)
                                              │
                                              ▼
                             orchestrator.train_model(data_dir, raw_config)
                                              │
                                              ▼
                             orchestrator._train_walk_forward()
```

### 3.1. Detailed Function Call Chain

1. **`ModelBuilderPanel._start_training()`** ([`apps/master_dataset_tk/model_builder/panel.py#L3437`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_builder/panel.py#L3437)):
   - Synchronizes Tk state: `cfg = self.state.build_training_config()`.
   - Validates configuration: `service.validate_config(self._data_dir, cfg)`.
   - Enriches config with dataset build snapshot and metadata: `self._enrich_training_config(final_cfg)`.
   - Switches view to `ModelTrainingPanel`: `self._show_training(final_cfg)`.
2. **`ModelTrainingRunner.start()`** ([`apps/master_dataset_tk/model_builder/runner.py#L31`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_builder/runner.py#L31)):
   - Spawns background worker thread named `"tk-model-train"`.
   - Invokes `train_model(data_dir, raw_config, on_progress, cancel_check)`.
3. **`orchestrator.train_model()`** ([`apps/chain_replay_ml/training/orchestrator.py#L334`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/training/orchestrator.py#L334)):
   - Executes `validate_training_config(data_dir, raw_config)`.
   - Normalizes config: `config = normalize_training_config(validation["config"])`.
   - If `config.split["strategy"] == "walk_forward"`, calls `_train_walk_forward()`.

---

## 4. Dataset Loading & Feature Matrix Preparation

Inside `_train_walk_forward()` ([`apps/chain_replay_ml/training/orchestrator.py#L827`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/training/orchestrator.py#L827)):

### 4.1. Step 1: Loading Dataset
- **Function**: `load_training_xy(data_dir, config)` in [`apps/chain_replay_ml/training/dataset_loader.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/training/dataset_loader.py).
- **Execution**:
  - Resolves Parquet path: `datasets/analysis_datasets/<dataset_name>.parquet`.
  - Reads Parquet schema with PyArrow/Polars, loading only requested feature columns (`config.features`), target column (`config.target`), and identity columns (`trading_day`, `timestamp`, `token`, `strike`).
  - Returns `X` (DataFrame of features), `y` (target Series), `features` (list of column names), `metadata` (dataset metadata JSON), and `context_df` (identity context).

### 4.2. Step 2: Preparing Feature Matrix
- **Functions**: In [`apps/chain_replay_ml/training/feature_matrix.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/training/feature_matrix.py):
  - `drop_invalid_rows(X, y, context_df)`: Drops NaNs in targets or infinite features.
  - `sanitize_training_features(X)`: Casts numeric columns to `float32` / `float64`.
  - `validate_feature_matrix(X, y)`: Checks feature counts and non-zero row counts.

---

## 5. Walk-Forward Training, Optimization & Holdout Evaluation

```
                    FULL DATASET TIMELINE (X, y)
┌────────────────────────────────────────────────────────┬─────────────────────┐
│           Walk-Forward Folds (Training + Val)          │ Holdout Test Slice  │
│  Fold 1: [Train ──► Val]                               │ (test_holdout_pct,  │
│  Fold 2:      [Train ──► Val]                          │  e.g. Last 15% of   │
│  Fold 3:           [Train ──► Val]                     │  chronological days)│
│  Fold 4:                [Train ──► Val]                │                     │
│  Fold 5:                     [Train ──► Val]           │                     │
└────────────────────────────────────────────────────────┴─────────────────────┘
```

### 5.1. Walk-Forward Execution (`run_walk_forward_validation`)
- **Module**: [`apps/chain_replay_ml/training/walk_forward_runner.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/training/walk_forward_runner.py).
- **Splits**: Slices dataset chronologically into $N$ expanding or rolling folds.
- **Fold Training**: Trains fold models and computes fold validation RMSE, MAE, R², and Directional Accuracy.
- **Aggregated Stats**: Computes mean and standard deviation of validation metrics across all folds.
- **Holdout Test Set**: Separates the final holdout test slice (`test_sl`, default 15% of total dataset).

### 5.2. Hyperparameter Optimization (`optimize_xgboost_hyperparameters`)
- If enabled, runs $N$ Optuna trials across walk-forward folds to find optimal `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.

### 5.3. Champion Training & Production Re-Evaluation
- Trains **Baseline Candidate** on all walk-forward training rows using initial hyperparameters.
- Trains **Tuned Candidate** on all walk-forward training rows using HPO best parameters.
- Re-evaluates both models across all walk-forward folds (`evaluate_hyperparameters_on_walk_forward`).
- Determines winner: `optimization_result["winner"]` (`"tuned"` or `"baseline"`).

---

## 6. Model Package Artifact Generation

The orchestrator calls `save_model_package()` in [`apps/chain_replay_ml/training/artifacts.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/training/artifacts.py), writing the following artifacts into `<chart_data_dir>/models/<model_name>/`:

| Artifact File | Format | Source / Writer Function | Contents & Purpose |
|---|---|---|---|
| `config.json` | JSON | `save_model_package` | Complete model configuration, target, features list (`features: list[str]`), dataset lineage (`feature_project_id`, `pipeline_id`, `pipeline_snapshot_id`), and dataset metadata snapshot. |
| `training_config.json` | JSON | `save_model_package` | Training hyperparameters, fold split definitions, and loss settings. |
| `metrics.json` | JSON | `evaluator.py` | Out-of-fold validation metrics, holdout test metrics, and composite objective scores. |
| `model.ubj` / `model.json` | Binary/JSON | `model.save_model` | Serialized champion XGBoost/CatBoost model weights and tree structures. |
| `baseline_model.ubj` | Binary | `baseline_model.save_model`| Serialized baseline model weights (for comparison). |
| `tuned_model.ubj` | Binary | `tuned_model.save_model` | Serialized tuned model weights (if HPO was enabled). |
| `feature_importance.csv` | CSV | `feature_importance_df` | Native tree split, gain, and weight importances. |
| `training_summary.json` | JSON | `build_training_summary_doc` | High-level summary of rows, feature count, training duration, and device information. |
| `training_metadata.json` | JSON | `_build_and_save_training_metadata` | Hardware telemetry (CPU/GPU load, peak memory), git commit hash, and training timestamps. |
| `training_monitor.csv` | CSV | `TrainingMonitor` | 5-second interval CPU, RAM, and GPU memory usage logs. |
| `training_log.txt` | Text | `TrainingLog` | Sequential plain-text console execution log. |
| `training_report.html` | HTML | `build_training_report_html` | Interactive offline HTML report of metrics, loss curves, and importances. |
| `dataset_build_snapshot.json`| JSON | `dataset_loader.py` | Cryptographic content hash of the training Parquet dataset. |
| `walk_forward/champion_aggregate.json`| JSON | `orchestrator.py` | Fold-by-fold results and aggregate scores for the champion model. |
| `walk_forward/folds.json`| JSON | `walk_forward_runner.py` | Exact start/end indices for each walk-forward fold. |

---

## 7. Post-Training Feature Studio Pipeline

Immediately after package saving, `_attach_post_training()` calls `post_training.orchestrator.run()` ([`apps/chain_replay_ml/post_training/orchestrator.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/post_training/orchestrator.py)):

```
save_model_package() ──► _attach_post_training()
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       POST-TRAINING FEATURE STUDIOS          │
        │                                              │
        │ 1. Feature Importance Studio                 │
        │    ──► package_dir/feature_importance_studio/│
        │                                              │
        │ 2. Feature Distribution Studio               │
        │    ──► package_dir/feature_distribution_studio/
        │                                              │
        │ 3. Feature Drift Studio                      │
        │    ──► package_dir/feature_drift_studio/     │
        │                                              │
        │ 4. Diagnostics Studio (Summary & Narrative)  │
        │    ──► package_dir/diagnostics_studio/       │
        │                                              │
        │ 5. Writes feature_studio_status.json         │
        └──────────────────────────────────────────────┘
```

### 7.1. Stage 1: Feature Importance Studio
- **Module**: [`apps/chain_replay_ml/feature_importance_studio/`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/feature_importance_studio/).
- **Calculations**: Native gain/split, Holdout Permutation Importance (metric drop on shuffle), and Holdout Tree SHAP attributions.
- **Files Written** in `<package_dir>/feature_importance_studio/`:
  - `native_xgb.json`
  - `permutation.json`
  - `shap.json`
  - `comparison.json`
  - `run_meta.json`

### 7.2. Stage 2: Feature Distribution Studio
- **Module**: [`apps/chain_replay_ml/feature_distribution_studio/`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/feature_distribution_studio/).
- **Calculations**: Univariate mean, standard deviation, skewness, kurtosis, missingness, and percentile distributions on Holdout data.
- **Files Written** in `<package_dir>/feature_distribution_studio/`:
  - `holdout_stats.json`
  - `comparison.json`
  - `run_meta.json`

### 7.3. Stage 3: Feature Drift Studio
- **Module**: [`apps/chain_replay_ml/feature_drift_studio/`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/feature_drift_studio/).
- **Calculations**: Distribution shift between Walk-Forward training rows and Holdout rows (Normalized Mean Shift, Kolmogorov-Smirnov test $p$-values, Wasserstein distance).
- **Files Written** in `<package_dir>/feature_drift_studio/`:
  - `drift_rows.json`
  - `comparison.json`
  - `run_meta.json`

### 7.4. Stage 4: Diagnostics Studio
- **Module**: [`apps/chain_replay_ml/diagnostics_studio/`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/diagnostics_studio/).
- **Calculations**: Synthesizes results from Importance, Distribution, and Drift to generate rule-based root-cause diagnostic summaries and narrative bullets.
- **Files Written** in `<package_dir>/diagnostics_studio/`:
  - `summary.json`
  - `narrative.json`
  - `comparison.json`
  - `run_meta.json`

### 7.5. Status Telemetry
- Writes `<package_dir>/feature_studio_status.json` containing timing, completion status (`"completed"`, `"partial"`, or `"failed"`), and stage execution telemetry.

---

## 8. Feature Studio Loading Flow

When the user opens **Feature Studio** ([`apps/master_dataset_tk/feature_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_studio_panel.py)) and selects a model:

```
FeatureStudioPanel._on_shared_model_changed()
    │
    ▼
feature_studio_pipeline.run_load_pipeline(package_dir)
    │
    ├── _load_importance()   ──► Reads feature_importance_studio/comparison.json
    ├── _load_distribution() ──► Reads feature_distribution_studio/comparison.json
    ├── _load_drift()        ──► Reads feature_drift_studio/comparison.json
    ├── _load_diagnostics()  ──► Reads diagnostics_studio/summary.json
    └── _load_planner()      ──► Reads recommendation_engine/comparison.json
    │
    ▼
Updates UI TreeViews & Plots across all tabs instantly (zero re-computation)
```

- **Instant Decoupled Rendering**: Because all artifacts are pre-computed on disk during post-training, switching models in Feature Studio loads JSON files in milliseconds without touching the raw dataset.

---

## 9. Production Validation & Unseen Dataset Lineage

When the user opens the **Production Validation** tab ([`apps/master_dataset_tk/production_validation_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/production_validation_panel.py)):

### 9.1. Lineage Extraction
Production Validation inspects the selected model's `config.json` and `dataset_build_snapshot.json` to extract:
1. `feature_project_id` (e.g. `"all"`, `"chart"`)
2. `pipeline_id` (e.g. `"PL_0005"`)
3. `pipeline_snapshot_id` (e.g. `"ca5945f58f8b1a2c"`)
4. `features`: Exact list of model features
5. `training_days`: Days used during training (seen days)

### 9.2. Unseen Dataset Resolution
- **Module**: [`apps/chain_replay_ml/production_validation/unseen_dataset.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/unseen_dataset.py).
- **Identity Hash**:
  $$\text{hash} = \text{SHA256}\Big(\text{master\_db} + \text{unseen\_days} + \text{feature\_project\_id} + \text{pipeline\_id} + \text{pipeline\_snapshot\_id} + \text{flags}\Big)[:8]$$
- **Resolution**:
  - Checks if `datasets/analysis_datasets/unseen_<slug>_<hash>.parquet` exists.
  - If found and valid, loads it immediately.
  - If missing, executes `create_analysis_dataset()` on unseen days using the parent model's exact feature pipeline configuration.

### 9.3. Forward Metrics & Evidence Recommendation
- **Module**: [`apps/chain_replay_ml/production_validation/rules.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/rules.py).
- Evaluates Holdout Rank vs. Unseen Rank ($\Delta R$), Relative Importance Drop ($\text{RelDrop}$), and Drift Severity.
- Assigns **`KEEP`**, **`WATCH`**, or **`REMOVE`**.
- Clicking **"Update Registry Recommendations"** stores results in `feature_recommendation_history.json`.

---

## 10. Real Model Package Structure (Actual Verified Artifacts)

Below is the verified on-disk directory structure of a real trained model package in AruMLStudio:

```
models/Future_LTP_5m_WF_1168f_XGB_2243_14/
├── config.json
├── training_config.json
├── metrics.json
├── model.ubj
├── baseline_model.ubj
├── tuned_model.ubj
├── feature_importance.csv
├── training_summary.json
├── training_metadata.json
├── training_monitor.csv
├── training_log.txt
├── training_report.html
├── dataset_build_snapshot.json
├── feature_studio_status.json
├── walk_forward/
│   ├── champion_aggregate.json
│   ├── folds.json
│   └── feature_selection.json
├── feature_importance_studio/
│   ├── native_xgb.json
│   ├── permutation.json
│   ├── shap.json
│   ├── comparison.json
│   └── run_meta.json
├── feature_distribution_studio/
│   ├── holdout_stats.json
│   ├── comparison.json
│   └── run_meta.json
├── feature_drift_studio/
│   ├── drift_rows.json
│   ├── comparison.json
│   └── run_meta.json
└── diagnostics_studio/
    ├── summary.json
    ├── narrative.json
    ├── comparison.json
    └── run_meta.json
```

---

## 11. Current Implementation vs. Not Implemented vs. Open Questions

### 11.1. CURRENT IMPLEMENTATION (What the Code Actually Does)
1. **Model Builder**: Has 10 distinct sections; displays 3 feature-source tabs (**Feature Registry**, **Base Pipeline**, **Selected Experimental**); validates feature existence in dataset Parquet before training.
2. **Orchestrator**: Executes chronological Walk-Forward validation, optional Optuna HPO, trains baseline & tuned final candidates, re-evaluates both on walk-forward folds, and selects champion.
3. **Artifact Persistence**: Generates 14+ standardized JSON, CSV, binary, and HTML artifacts in `<package_dir>`.
4. **Post-Training Pipeline**: Automatically runs Importance &rarr; Distribution &rarr; Drift &rarr; Diagnostics studios sequentially without blocking package saving if any studio fails.
5. **Feature Studio**: Loads pre-computed JSON artifacts in milliseconds via `feature_studio_pipeline.py`.
6. **Production Validation**: Preserves parent model lineage (`feature_project_id`, `pipeline_id`, `pipeline_snapshot_id`) and hashes them into an 8-character identity hash to resolve or generate the exact matching `unseen_*` Parquet dataset.
7. **Feature Recommendation & Governance (Phases 1–3D)**: Closed-loop Evidence DB, Recommendation-to-Training Decision Engine, and Governed Feature Promotion/Graduation into Registry and Base Pipeline (PL_0001).
8. **Model Taxonomy Foundation (Phase 4C.1)**: Canonical 4-dimensional taxonomy (`apps/chain_replay_ml/model_taxonomy/`) separating Task Type, Market Regime, Population Tier, and Lifecycle Status with full backward compatibility.

---

### 11.2. PLANNED SUBSYSTEMS (Future Phases)
1. **Model Registry SQLite Extension & Regime Registry (Phases 4C.2–4C.3)**: Additive SQLite columns and `regime_registry_store.json`.
2. **Model Research Lab Population Awareness & Faceted UI (Phase 4C.4)**: Multi-faceted UI filtering by Task, Regime, Population, and Status.
3. **Persistent Multi-Model Benchmarking (Phase 4D)**: Research memory and cross-dataset leaderboards in `analysis.db`.
4. **Autonomous Research Factory (Phase 5)**: Overnight automated feature, pipeline, model, and regime discovery loop.
