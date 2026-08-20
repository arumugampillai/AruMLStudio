# AruMLStudio Model Research Lab — Actual Implementation & Workflow Audit

---

## 1. Executive Summary & Purpose

This document provides a source-code verified audit of the **Research Lab** (also implemented as **Model Lab**) in **AruMLStudio**, specifically documenting:
1. The exact invocation path from the **Open Research** button in **Model Registry**.
2. The complete UI structure, database schema, and operational workflow of the Research Lab.
3. The underlying prediction dataset generation engine and trade outcome metrics.
4. Its precise relationship (and lack of direct coupling) with **Feature Studio**, **Create Model**, and **Production Validation**.

---

## 2. Invocation from Model Registry & Call Chain

### 2.1. UI Entry Point: The "Open Research" Action
In the **Model Registry Panel** ([`apps/master_dataset_tk/model_registry_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_registry_panel.py)), each trained model entry in the TreeView displays a dedicated `"research"` column cell.

- **Trigger Handler**: `ModelRegistryPanel._on_tree_click(event)` ([`model_registry_panel.py#L1150-L1167`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_registry_panel.py#L1150-L1167))
- **Detection**: `if col_id == "research": self.after_idle(lambda n=row: self._open_research_lab(n))`

### 2.2. Exact Function Call Chain

```
ModelRegistryPanel._on_tree_click()
    │
    ▼ (col_id == "research")
ModelRegistryPanel._open_research_lab(model_name)
    │
    │ Arguments: chart_dir, model_name, detail_doc=row, ensure_lab=True, initial_tab="prediction"
    ▼
open_model_lab_window()  [apps/master_dataset_tk/model_lab_window.py#L8566]
    │
    ▼
ModelLabWindow.__init__() ──► ModelLabWindow._begin_async_open(ensure_lab=True)
                                   │
                                   ▼
                    ModelLabStore.open() / ensure_schema()
              (Connects to <data_dir>/model_research/model_lab_<name>_v1.db)
```

---

## 3. Model-Specific Research Workspace Identification

Every model opened in Research Lab is assigned a dedicated, isolated SQLite database file:

- **Database Path**: `<data_dir>/model_research/model_lab_{safe_model_name(model_name)}_v{version}.db`
- **Location Resolution**: Resolved dynamically via `resolve_model_research_dir(data_dir)` in [`apps/chain_replay_ml/model_lab/paths.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/model_lab/paths.py), pointing to `<data_dir>/model_research/` within the active workspace.
- **Version Management**: `next_lab_version()` increments `_v1.db`, `_v2.db` if multiple research labs are created for the same parent model.

### 3.1. Metadata Read by Research Lab at Startup

When `ensure_lab` initializes the workspace, it calls `create_model_lab()` ([`apps/chain_replay_ml/model_lab/store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/model_lab/store.py)), capturing a point-in-time snapshot into table `model_lab_info`:

| Metadata Entity | Source Location | Read By Model Lab | Stored in `model_lab_info`? |
|---|---|---|---|
| `model_name` | `models/<model_name>/config.json` | `ModelLabInfo.parent_model_name` | Yes (`parent_model_name`) |
| `target` | `models/<model_name>/config.json` | Target resolver | Yes (`target`) |
| `algorithm` | `models/<model_name>/config.json` | Inference engine | Yes (`algorithm`) |
| `features` (selected list) | `models/<model_name>/config.json` | `selected_features_snapshot_json` | Yes (JSON string of selected feature names) |
| `dataset_name` | `models/<model_name>/config.json` | `dataset_snapshot_json` | Yes (Snapshot of dataset config) |
| `metrics` | `models/<model_name>/metrics.json` | `metrics_snapshot_json` | Yes (Training/Holdout/WF metrics) |
| `feature_ranking` | `models/<model_name>/walk_forward/feature_selection.json` | `feature_ranking_snapshot_json` | Yes (RFE ranking snapshot) |
| `feature_project_id` | `models/<model_name>/config.json` | Available in config doc | Stored inside `dataset_snapshot_json` |
| `pipeline_id` & `snapshot_id`| `models/<model_name>/config.json` | Available in config doc | Stored inside `dataset_snapshot_json` |

---

## 4. Complete Research Lab UI Structure

The UI is implemented in `ModelLabWindow` ([`apps/master_dataset_tk/model_lab_window.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_lab_window.py)) containing 10 tabbed workspaces:

```
                                  MODEL LAB WINDOW
                                  (ModelLabWindow)
                                         │
 ┌──────────────┬──────────────┬─────────┴────┬──────────────┬──────────────┐
 ▼              ▼              ▼              ▼              ▼              ▼
[Overview] [Prediction Dataset] [Research] [Strategy Sim] [Strike Dash] [Feature Research]
               │         [Programs] [Improvement] [Confidence] [RR Validation]
               ▼
   [Parallel Generation Engine]
```

### 4.1. Tab Breakdown

1. **Overview Tab (`_overview_tab`)**:
   - **Displays**: Model summary card, target definition, algorithm, selected feature count, training rows, and dataset name.
   - **Source**: Reads `model_lab_info` table from SQLite.
2. **Prediction Dataset Tab (`_prediction_tab`)**:
   - **Displays**: Build controls (date picker, parallel workers, batch size), per-day generation progress, total prediction row count, and per-day completion status (`completed`, `partial`, `failed`).
   - **Actions**: "Generate / Build Prediction Dataset", "Pause", "Resume", "Export Parquet".
3. **Research Dashboard Tab (`_research_tab`)**:
   - **Displays**: Prediction error distribution, actual vs. predicted scatter plots, directional accuracy %, profit factor, and time-to-target histograms.
   - **Source**: `research_dashboard.py` querying `prediction_dataset` SQLite table.
4. **Strategy Simulator Tab (`_strategy_sim_tab`)**:
   - **Displays**: Trade entry/exit rule simulator, hurdle thresholds, capital curve, win/loss ratio, maximum drawdown, and Sharpe ratio.
   - **Module**: [`apps/master_dataset_tk/model_lab_strategy_sim_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_lab_strategy_sim_panel.py).
5. **Strike Dashboard Tab (`_strike_dashboard_tab`)**:
   - **Displays**: Accuracy and error breakdown partitioned across option strikes (Deep ITM, ITM, ATM, OTM, Deep OTM).
   - **Module**: [`apps/master_dataset_tk/strike_prediction_dashboard_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/strike_prediction_dashboard_panel.py).
6. **Feature Research Tab (`_feature_research_tab`)**:
   - **Displays**: Per-feature tertile analysis (low/mid/high feature values vs. trade win rate and prediction error).
   - **Module**: [`apps/chain_replay_ml/model_lab/feature_research.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/model_lab/feature_research.py).
7. **Research Programs Tab (`_research_programs_tab`)**:
   - **Displays**: Systematic multi-hypothesis experiment tracking.
8. **Model Improvement Tab (`_model_improvement_tab`)**:
   - **Displays**: Residual analysis and error concentration alerts.
9. **Confidence Tab (`_confidence_tab`)**:
   - **Displays**: Secondary meta-classifier training to predict primary model trade success probability.
   - **Module**: [`apps/master_dataset_tk/model_lab_confidence_labels_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_lab_confidence_labels_panel.py).
10. **RR Validation Tab (`_rr_validation_tab`)**:
    - **Displays**: Risk-to-Reward ratio matrix ($1:1, 1:2, 1:3, 1:4$) hit rates across forward prediction horizons.

---

## 5. Prediction Dataset Generation Engine

```
[User clicks "Build Prediction Dataset"]
                │
                ▼
prediction_builder.build_prediction_dataset()
                │
                ▼
prediction_parallel.run_parallel_prediction_job()
  ├── Worker 1 (Day 2026-06-01) ──► Loads Model & Parquet ──► Computes Predictions & MFE/MAE
  ├── Worker 2 (Day 2026-06-02) ──► Loads Model & Parquet ──► Computes Predictions & MFE/MAE
  └── Worker N (Day 2026-06-0N) ──► Loads Model & Parquet ──► Computes Predictions & MFE/MAE
                │
                ▼
prediction_io.write_prediction_rows_chunk()
                │
                ▼
Persists to SQLite table: prediction_dataset
```

### 5.1. Execution Flow & Backend Classes

1. **Job Initialization**: `build_prediction_dataset()` in [`apps/chain_replay_ml/model_lab/prediction_builder.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/model_lab/prediction_builder.py).
2. **Parallel Process Dispatch**: `run_parallel_prediction_job()` in [`apps/chain_replay_ml/model_lab/prediction_parallel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/model_lab/prediction_parallel.py) forks $N$ worker processes via `multiprocessing`.
3. **Inference Worker**: `prediction_worker.py` loads:
   - Serialized model weights (`model.ubj` or `model.json`).
   - Feature columns specified in `config.json["features"]`.
   - Feature matrix rows for the assigned trading day from the Master Dataset / Parquet.
4. **Tick-by-Tick Forward Path Evaluation**:
   - Evaluates `predicted_future_ltp = model.predict(X)`.
   - Traverses future ticks within the horizon window to track Maximum Favorable Excursion (**MFE**) and Maximum Adverse Excursion (**MAE**).
5. **Database Storage**: Writes rows in batches into SQLite table `prediction_dataset`.

---

## 6. Prediction Metrics & Table Schema

The fixed schema for table `prediction_dataset` is defined in [`apps/chain_replay_ml/model_lab/prediction_schema.py#L64-L117`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/model_lab/prediction_schema.py#L64-L117):

| Column Name | SQL Type | Mathematical Definition / Meaning | Verified in Schema? |
|---|---|---|---|
| `trading_day` | `TEXT` | Date of the trading day (`"YYYY-MM-DD"`) | **Yes** |
| `timestamp` | `REAL` | Epoch or intra-day seconds timestamp | **Yes** |
| `token` / `strike` | `TEXT` / `REAL` | Option strike price and instrument token | **Yes** |
| `option_type` | `TEXT` | `"CE"` or `"PE"` | **Yes** |
| `current_spot` | `REAL` | Underlying Nifty spot price at prediction time | **Yes** |
| `current_ltp` | `REAL` | Option LTP at prediction time ($P_0$) | **Yes** |
| `predicted_future_ltp` | `REAL` | Model predicted LTP at horizon ($P_{\text{pred}}$) | **Yes** |
| `actual_future_ltp` | `REAL` | Actual recorded LTP at horizon ($P_{\text{actual}}$) | **Yes** |
| `expected_move` | `REAL` | $P_{\text{pred}} - P_0$ | **Yes** |
| `actual_move` | `REAL` | $P_{\text{actual}} - P_0$ | **Yes** |
| `predicted_trend` | `TEXT` | `"UP"`, `"DOWN"`, or `"FLAT"` | **Yes** |
| `actual_trend` | `TEXT` | `"UP"`, `"DOWN"`, or `"FLAT"` | **Yes** |
| `absolute_error` | `REAL` | $\|P_{\text{actual}} - P_{\text{pred}}\|$ | **Yes** |
| `prediction_error` | `REAL` | $P_{\text{actual}} - P_{\text{pred}}$ | **Yes** |
| `premium_error_pct` | `REAL` | $\frac{\|P_{\text{actual}} - P_{\text{pred}}\|}{P_0} \times 100$ | **Yes** |
| `direction_correct` | `INTEGER` | $1$ if $\text{sign}(\text{expected\_move}) == \text{sign}(\text{actual\_move})$, else $0$ | **Yes** |
| `maximum_profit` (**MFE**) | `REAL` | Highest favorable price excursion during horizon window | **Yes** |
| `maximum_drawdown` (**MAE**)| `REAL` | Highest adverse price drop during horizon window | **Yes** |
| `dd_before_target` | `REAL` | Maximum drawdown experienced before target price was reached | **Yes** |
| `time_to_max_profit` | `REAL` | Seconds from entry to peak profit | **Yes** |
| `time_to_max_drawdown`| `REAL` | Seconds from entry to peak drawdown | **Yes** |
| `time_to_target` | `REAL` | Seconds until expected price target was hit | **Yes** |
| `target_reached` | `INTEGER` | $1$ if actual price reached or exceeded predicted target, else $0$ | **Yes** |
| `rr_1_1_hit` &rarr; `rr_1_4_hit`| `INTEGER` | Whether Risk-to-Reward thresholds ($1:1, 1:2, 1:3, 1:4$) were touched | **Yes** |
| `master_row_id` | `INTEGER` | Foreign key pointer to Master Dataset tick table | **Yes** |

---

## 7. Relationship with Feature Studio & Other Studios

### 7.1. Direct Connections
- **NO DIRECT CODE IMPORTS**: Research Lab / Model Lab does **NOT** import, invoke, or depend on any module from `feature_importance_studio`, `feature_distribution_studio`, `feature_drift_studio`, `diagnostics_studio`, or `feature_studio_panel.py`.
- Feature Studio does **NOT** import or call `model_lab` or `research_lab`.

### 7.2. Indirect Shared Entities
- **Shared Parent Model Package**: Both Feature Studio and Research Lab read from the same underlying model directory (`models/<model_name>/`).
  - Both read `config.json` to get `features: list[str]`, `target`, and `dataset_name`.
  - Both read `metrics.json` for validation scores.
- **Shared Feature Store Concept**: Both resolve feature columns from the same underlying Master Dataset / Analysis Parquet file.

---

## 8. Feature Population Awareness in Research Lab

In the **CURRENT IMPLEMENTATION**:

> [!IMPORTANT]
> The Research Lab does **NOT** partition or classify features into the three distinct populations (**Feature Registry**, **Base Pipeline**, **Selected Experimental**).

- In `ModelLabStore` and `FeatureResearch`, the model's features are snapshotted as a **flat list of string names** (`selected_features_snapshot_json`).
- In the **Feature Research Tab** (`apps/chain_replay_ml/model_lab/feature_research.py`), tertile distributions and correlations are computed uniformly for every feature name in the list without identifying whether a feature came from the Feature Registry or an Experimental Pipeline.

---

## 9. Comprehensive Comparison Across All Studios

| Studio Component | Primary Purpose | Primary Input Data | Primary Output Artifact | Relationship with Research Lab |
|---|---|---|---|---|
| **Model Registry** | Model indexing, active model management, and deletion protection. | `models/<model_name>/` packages | `.active_model.json`, registry tables | **Parent Launcher**: Spawns Research Lab via the "Open Research" button. |
| **Create Model** | Model architecture, feature selection (3 tabs), split config, and training. | Analysis Dataset Parquet | Trained model package (`models/<name>/`) | **Upstream Producer**: Creates the model package that Research Lab consumes. |
| **Research Lab (Model Lab)**| Tick-by-tick prediction generation, trade simulation, MFE/MAE analysis, and confidence modeling. | Model weights (`model.ubj`) + Master Dataset tick feeds | SQLite database (`model_lab_<name>_v1.db`), `prediction_dataset` rows | **Independent Workspace**: Evaluates simulated trading performance row-by-row. |
| **Feature Studio** | Post-training statistical diagnostics (Importance, Distribution, Drift, Diagnostics). | Pre-computed JSON artifacts in `models/<name>/` | Interactive GUI charts and tables | **Parallel Post-Training Inspector**: Reads static tree importances and moments; does not run tick simulations. |
| **Diagnostics Studio** | Automated rule-based root cause diagnosis of feature instability. | Importance, Distribution, and Drift comparisons | `diagnostics_studio/summary.json` | **Diagnostic Advisor**: Analyzes in-sample holdout degradation; does not simulate trade paths. |
| **Production Validation** | Out-of-sample forward testing against true unseen trading days. | Unseen Parquet dataset (`unseen_*.parquet`) | `feature_recommendation_history.json` (KEEP/WATCH/REMOVE) | **Forward Generalization Verifier**: Evaluates rank stability on live forward days. |

---

## 10. Complete End-to-End Execution Flow

```
Model Registry (ModelRegistryPanel)
     │
     ▼ (User clicks "Open Research" cell in TreeView)
ModelRegistryPanel._open_research_lab(model_name)
     │
     ▼
open_model_lab_window(model_name, ensure_lab=True)  [model_lab_window.py]
     │
     ▼
ModelLabStore.write_info()  [store.py]
     │ (Snapshots config.json, metrics.json, selected features to model_lab_info)
     ▼
Research Lab UI renders 10 Tabs (Overview, Prediction Dataset, Simulator, etc.)
     │
     ▼ (User clicks "Generate / Build Prediction Dataset")
prediction_builder.build_prediction_dataset()  [prediction_builder.py]
     │
     ▼
prediction_parallel.run_parallel_prediction_job()  [prediction_parallel.py]
     │
     ▼ (Multiprocessing workers execute prediction_worker.py)
Inference Engine loads model.ubj + slices selected features from Parquet
     │
     ▼
Calculates predicted_future_ltp, actual_future_ltp, MFE, MAE, time_to_target
     │
     ▼
prediction_io.write_prediction_rows_chunk()  [prediction_io.py]
     │
     ▼
Inserts rows into SQLite table: prediction_dataset (<data_dir>/model_research/model_lab_<name>_v1.db)
     │
     ▼
Research Dashboard & Strategy Simulator query prediction_dataset to display live trade curves
```

---

## 11. Current Implementation vs. Future Roadmap

### 11.1. CURRENTLY IMPLEMENTED (Verified from Source)
1. **Open Research Button**: Wired in `ModelRegistryPanel` to invoke `open_model_lab_window()` with `ensure_lab=True`.
2. **Model Lab SQLite Workspace**: Automatically creates and maintains `<data_dir>/model_research/model_lab_<name>_v1.db`.
3. **Prediction Generation**: Parallel multi-worker batch inference generating tick-level predictions, `maximum_profit` (MFE), `maximum_drawdown` (MAE), `time_to_target`, `direction_correct`, and `rr_1_1_hit` through `rr_1_4_hit`.
4. **Strategy Simulator & Confidence Builder**: Full interactive trade simulation, equity curve visualization, strike analysis, and secondary confidence classifier training.
5. **Model Taxonomy Foundation (Phase 4C.1 — IMPLEMENTED)**: Canonical 4-dimensional taxonomy (`apps/chain_replay_ml/model_taxonomy/`) providing Task Type, Regime, Population Tier, and Lifecycle Status data contracts.

### 11.2. FEATURE STUDIO CONNECTIONS (Verified Findings)
- **Direct Code Connection**: **NONE**. Neither subsystem imports or calls the other.
- **Indirect Connection**: Both subsystems are consumers of the same parent model package (`models/<model_name>/`), reading `config.json` and selected feature column names.

### 11.3. PLANNED SUBSYSTEMS (Future Phases)
1. **Faceted Filtering & Population Awareness (Phase 4C.4 — PLANNED)**: Multi-dimensional filtering by Task Type, Regime, Population Tier, and Status in Model Registry and Research Lab.
2. **Persistent Multi-Model Benchmarking (Phase 4D — PLANNED)**: Historical cross-model evaluations in `analysis.db`.
3. **Autonomous Model Discovery & Fine-Tuning Controller (Phase 4F — IMPLEMENTED & VERIFIED)**: Standardized strategy evaluation harness, automated candidate generation, composite ranking, fine-tuning mutation controller, overnight campaign orchestrator, and `▶ Start Autonomous Research` UI integration.

### 11.4. STORAGE MODE
1. **Storage Mode**: The codebase supports both `FEATURE_STORAGE_REFERENCED` (`master_row_id` joined to Master DB) and legacy `FEATURE_STORAGE_EMBEDDED` (`sf_*` columns stored in `prediction_dataset`). Future dataset migrations may fully deprecate embedded mode to minimize disk usage.
