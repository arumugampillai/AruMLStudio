# AruMLStudio Feature Analysis Lab — Technical & Functional Architecture

---

## 1. Purpose of the Analysis Tab

The **Feature Analysis Lab** (the "Analysis" tab in Feature Transformations, implemented in [`apps/master_dataset_tk/feature_analysis_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_analysis_panel.py)) is the dedicated pre-training statistical research, collinearity diagnostics, and feature down-selection workbench in **AruMLStudio**.

### 1.1. Role in the Feature Lifecycle

```
Feature Registry ──► Master Dataset ──► Experimental Pipeline ──► Analysis Dataset (.parquet)
                                                                            │
                                                                            ▼
                                                                ┌───────────────────────┐
                                                                │ FEATURE ANALYSIS LAB  │
                                                                │ • Collinearity Pairs  │
                                                                │ • HCA Feature Families│
                                                                │ • Mutual Information  │
                                                                │ • Permutation Scoring │
                                                                │ • Fast Tree SHAP      │
                                                                │ • Feature Selection   │
                                                                └───────────┬───────────┘
                                                                            │
                                                                            ▼
Production Validation ◄── Feature Studio ◄── Model Package ◄── Model Builder Selection
(Unseen Forward Days)   (Post-Model Diag)   (Trained XGB)     (Tabs 1, 2, 3 Partition)
```

### 1.2. Key Architectural Distinctions:
- **Research & Down-Selection Only**: The Analysis Lab evaluates relationships, redundancies, and non-linear predictive utility *within a specific historical analysis dataset*.
- **No In-Sample Proof of Forward Alpha**: Analysis results indicate which features have statistical merit and low collinearity on historical data; **they do NOT prove that a feature will generalize to unseen live production trading regimes**. (That is the exclusive role of Production Validation).
- **Read-Only over Parquet**: The Analysis Lab does **not** generate new feature data; it loads materialized Parquet datasets and stores research artifacts in SQLite (`analysis.db`).

---

## 2. Input Data & Dataset Ingestion

### 2.1. Supported Dataset Types
The Analysis Lab operates exclusively on **Analysis Datasets** (`datasets/analysis_datasets/analysis_<name>.parquet`) and their companion metadata JSON (`analysis_<name>.json`).

### 2.2. Ingestion & Column Discovery
1. **Metadata Ingestion**: Reads companion JSON to discover dataset properties:
   - `feature_project_id` (e.g. `"all"`, `"chart"`)
   - `registry_export_features` (list of canonical features)
   - `base_pipeline_export_features` (list of base pipeline features)
   - `pipeline_id` & `pipeline_snapshot_id` (experimental pipeline lineage)
   - `pipeline_provenance` (snapshot metadata & candidate features)
   - `include_registry` & `include_pipeline` (inclusion flags)
2. **Fingerprinting & Parquet Ingestion**: [`apps/chain_replay_ml/dataset_builder/analysis_lab_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_lab_store.py) inspects the Parquet schema via `pyarrow.parquet.read_schema()` without loading all rows into RAM:
   - Evaluates row count, column count, and column names.
   - Computes `columns_hash` and `dataset_hash` (SHA256 of columns + size + mtime) to detect stale cache entries.
3. **Execution-Time Streaming**: During compute runs, Polars / PyArrow loads only the required numerical feature columns and target variables into memory.

---

## 3. Treatment of the Three Feature Populations

In the **CURRENT IMPLEMENTATION**:

> [!IMPORTANT]
> The Analysis Lab treats all loaded feature columns as a **single unified, flat feature matrix**. It does **not** visually partition features into separate Registry, Base Pipeline, and Experimental tabs within its research modules.

- **Unified Feature Pool**: Whether a feature originated from Feature Registry (`spot`, `atm_iv_ce`), Base Pipeline (`spot_lag1`), or an Experimental Pipeline (`spot_roll_mean_60`), the Analysis Lab analyzes all columns uniformly in correlation matrices, HCA clusters, and permutation scoring.
- **Downstream Categorization**: Feature source separation occurs downstream when features are passed to **Model Builder**, where `feature_sources_catalog.py` splits the selected feature bundle into **Tab 1: Feature Registry**, **Tab 2: Base Pipeline**, and **Tab 3: Selected Experimental**.

---

## 4. Analysis Lab UI Structure & Hierarchy

```
Feature Analysis Lab (FeatureAnalysisPanel)
├── Dataset Selector Toolbar
│   ├── Dataset Dropdown (analysis_*.parquet)
│   ├── Stats Header (Rows, Features, Created, Hash, Status)
│   └── Refresh & Scan Controls
├── Main Notebook Tabs
│   ├── Tab 1: Feature Selection (Down-Selection Engine)
│   │   ├── Strategy Selector (HCA+Corr+Perm, Corr+Perm, Perm Only, Corr Only)
│   │   ├── Representative Policy (Top 1, Top 2, Top 3, Top N)
│   │   ├── Threshold Inputs (Correlation Thr = 0.95, Permutation Thr = 0.001)
│   │   ├── Action Buttons (Preview Final Dataset, View Features, ► Create Model Builder)
│   │   └── Dataset Card Preview (Consolas text display)
│   ├── Tab 2: Module Run Status (Execution Checklist)
│   │   ├── Checkboxes for 7 Modules (Correlation, HCA, VIF, MI, Perm, Scorecard, SHAP)
│   │   ├── Status Indicators (not_run, running, completed, failed, stale)
│   │   └── Run Selected / Run All Buttons
│   ├── Tab 3: Correlation Views
│   │   ├── Backend Selector (Auto / Polars / PyTorch GPU / NumPy)
│   │   ├── Summary Card (Max Corr, High-Collinearity Pair Count)
│   │   ├── Top Correlated Pairs Table (|r| >= threshold)
│   │   ├── Full Correlation Matrix Viewer & Heatmap
│   │   └── Cluster Heatmap View
│   ├── Tab 4: HCA / Feature Families
│   │   ├── Family Cluster Listbox
│   │   ├── Family Member TreeView
│   │   ├── Representative Selector & Review Status
│   │   └── Manual Representative Override Dialog
│   ├── Tab 5: Mutual Information
│   │   ├── Target Variable Dropdown (e.g. future_ltp_3s, fwd_ret_5m)
│   │   ├── Run MI Button
│   │   └── Feature Ranking Table (Score, Rank, Percentile, Interpretation)
│   ├── Tab 6: Permutation Importance
│   │   ├── Model Type Selector & Target Selector
│   │   ├── Permutation Progress Bar
│   │   └── Importance Table (Baseline Metric, Permuted Metric, Delta, Rank)
│   ├── Tab 7: SHAP Model Validation
│   │   ├── Trained Model Selector Dropdown
│   │   ├── Sample Size Configuration
│   │   └── Mean Absolute SHAP Attribution Table & Bar Charts
│   └── Tab 8: Feature Profile & Roles
│       ├── Feature Search / Selector
│       └── Univariate Profile (Mean, Std, Skew, Kurtosis, Missingness, Min, Max, Quantiles)
```

---

## 5. Correlation Analysis Module

Located in [`apps/chain_replay_ml/dataset_builder/analysis_correlation.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_correlation.py):

### 5.1. Computation & Acceleration Backends
- **Supported Methods**: Pearson ($r$) and Spearman rank correlation ($\rho$).
- **Multi-Backend Engine**:
  - `gpu` (PyTorch CUDA tensor dot product for fast matrix calculation)
  - `polars` (Vectorized multithreaded CPU computation)
  - `numpy` (Standard BLAS CPU fallback)
- **Missing Value Handling**: Pairwise null drop / zero-variance filtering ($\sigma < 10^{-8}$).

### 5.2. Mathematical Formula
$$r_{xy} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}$$

### 5.3. High Collinearity Filtering
- Identifies feature pairs with $|r| \ge \text{threshold}$ (default $0.95$).
- Flags severe multicollinearity to prevent unstable tree splits and variance inflation in downstream linear/logistic models.
- **Persistence**: Pairwise correlations are written to the `correlation` table in `analysis.db`.

---

## 6. Hierarchical Cluster Analysis (HCA / Feature Families)

Located in [`apps/chain_replay_ml/dataset_builder/analysis_hca.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_hca.py):

### 6.1. Clustering Methodology
1. **Distance Metric**:
   $$D(x, y) = 1 - |r_{xy}|$$
2. **Linkage Method**: `scipy.cluster.hierarchy.linkage` with `method="complete"` (or `"average"`).
3. **Cluster Threshold**: $t = 1 - \text{correlation\_threshold}$ (e.g. $1 - 0.95 = 0.05$).
4. **Cluster Formation**: Groups features with high mutual collinearity into orthogonal **Feature Families** (e.g. `Family_01`, `Family_02`).

### 6.2. Representative (Centroid) Selection
For each family cluster, the algorithm selects a **Representative Feature**:
- Feature with the highest average correlation to other cluster members (centroid).
- If Permutation scores are available, selects the member with the highest individual predictive power.
- **Persistence**: Stored in the `clusters` table in `analysis.db` (`representative=1`).

---

## 7. Mutual Information Engine

Located in [`apps/chain_replay_ml/dataset_builder/analysis_mutual_information.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_mutual_information.py):

### 7.1. Purpose & Methodology
- Uses `sklearn.feature_selection.mutual_info_regression` based on k-nearest neighbors entropy estimation (Kraskov et al.).
- **Detects Non-Linear Relationships**: Uncovers complex dependencies (e.g. U-shaped volatility smiles) invisible to linear correlation ($r \approx 0$).

### 7.2. Mathematical Formulation
$$I(X; Y) = \iint p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \, dx \, dy$$

- **Output**: Generates `score`, `rank`, and `percentile`.
- **Persistence**: Stored in `mutual_information` and `mi_runs` tables in `analysis.db`.

---

## 8. Permutation Importance Module

Located in [`apps/chain_replay_ml/dataset_builder/analysis_permutation.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_permutation.py):

### 8.1. Purpose & Methodology
1. Trains a fast baseline estimator (e.g. LightGBM / Fast XGBoost) on historical training rows.
2. Evaluates baseline error metric (RMSE, MAE, or LogLoss).
3. For each feature, shuffles (permutes) column values across $N$ repeats and measures metric degradation:
   $$\text{Importance}(f) = \text{Metric}_{\text{permuted}}(f) - \text{Metric}_{\text{baseline}}$$
4. **What it Proves**: Measures feature utility *within this historical dataset*.
5. **What it DOES NOT Prove**: Does not prove resistance to future out-of-sample temporal drift.
- **Persistence**: Stored in `permutation_importance` and `permutation_runs` in `analysis.db`.

---

## 9. Model Explanation Module (Fast Tree SHAP)

Located in [`apps/chain_replay_ml/dataset_builder/analysis_shap.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_shap.py):

### 9.1. Purpose & Methodology
- Executes **TreeSHAP** (`shap.TreeExplainer`) over trained tree ensembles.
- Subsamples $100\dots 500$ representative rows to provide fast interactive attribution.
- Computes mean absolute SHAP value per feature:
  $$\text{MeanSHAP}(f) = \frac{1}{N} \sum_{i=1}^N |\phi_i(f)|$$
- **Persistence**: Stored in `shap` and `shap_runs` tables in `analysis.db`.

---

## 10. Feature Selection Engine

Located in [`apps/chain_replay_ml/dataset_builder/analysis_feature_selection.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_feature_selection.py):

### 10.1. Available Selection Strategies

```
                            INPUT: Full Feature Dataset
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
[hca_corr_perm]                     [corr_perm]                      [perm_only]
HCA + Corr + Perm                Corr + Perm Only               Permutation Only
        │                                │                                │
1. High-Corr Filtering           1. Correlation Filter            1. Rank by Permutation
2. HCA Family Clustering         2. Rank by Permutation           2. Keep >= Threshold
3. Policy Selection              3. Keep >= Threshold                     │
   (Top 1 / Top 2 / Top N)               │                                │
4. Permutation Filter                    │                                │
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         │
                                         ▼
                        OUTPUT: Final Feature Selection Bundle
```

| Strategy | Workflow & Decision Rules | Primary Use Case |
|---|---|---|
| **`hca_corr_perm`** | Correlation &rarr; HCA Families &rarr; Pick Top 1/N Representatives per Family &rarr; Filter $\text{Permutation} \ge \text{threshold}$. | Standard multi-collinearity reduction & robust signal distillation. |
| **`corr_perm`** | Drop highly correlated pairs ($|r| \ge 0.95$) &rarr; Filter $\text{Permutation} \ge 0.001$. | Simpler baseline filtering without hierarchical clustering. |
| **`perm_only`** | Rank all features by permutation importance &rarr; Filter $\text{Permutation} \ge \text{threshold}$. | Unconstrained predictive ranking for tree ensembles. |
| **`corr_only`** | Drop collinear duplicates without performance filtering. | Pure redundancy elimination. |

---

## 11. Schema of `analysis.db`

Located at `<chart_data_dir>/analysis.db`:

| Table Name | Purpose | Key Columns |
|---|---|---|
| `datasets` | Registered analysis dataset fingerprints | `dataset_id`, `name`, `path`, `rows`, `features`, `columns_hash`, `dataset_hash`, `registered_at` |
| `analysis_runs` | Parent run container for a dataset | `run_id`, `dataset_id`, `dataset_hash`, `status`, `created_at` |
| `module_runs` | Status and timing for each module | `run_id`, `module_id`, `status`, `started_at`, `finished_at`, `elapsed_sec`, `message` |
| `correlation` | Pairwise correlation values | `run_id`, `feature_a`, `feature_b`, `correlation` |
| `clusters` | HCA feature family memberships | `run_id`, `feature`, `cluster`, `representative` |
| `vif` | Variance Inflation Factors | `run_id`, `feature`, `cluster`, `vif` |
| `mutual_information` | MI scores per target | `run_id`, `target`, `feature`, `score`, `rank`, `percentile` |
| `mi_runs` | Metadata for MI executions | `run_id`, `target`, `n_features`, `n_samples`, `created_at` |
| `permutation_importance`| Full metric degradation stats | `run_id`, `model_id`, `target`, `feature_name`, `importance`, `importance_rank`, `delta_rmse` |
| `permutation_runs` | Metadata for permutation runs | `run_id`, `dataset_id`, `target`, `task_type`, `n_features`, `elapsed_sec` |
| `shap` | Tree SHAP feature importances | `run_id`, `model_name`, `feature`, `importance`, `rank`, `percentile` |
| `shap_runs` | Metadata for SHAP executions | `run_id`, `model_name`, `n_features`, `n_samples`, `elapsed_sec` |
| `feature_scorecard` | Combined discovery rating notes | `run_id`, `feature`, `recommendation`, `notes` |
| `lab_prefs` | UI state persistence across sessions| `key`, `value`, `updated_at` |

---

## 12. Analysis Run Lifecycle

```
[1] Select Dataset in Dropdown (analysis_*.parquet)
         │
         ▼
[2] Parquet Fingerprint & Schema Verification (columns_hash, dataset_hash)
         │
         ▼
[3] Execute Research Modules (Correlation → HCA → Mutual Info → Permutation)
         │
         ▼
[4] Persist Module Artifacts into SQLite (analysis.db)
         │
         ▼
[5] Review Feature Families & Interactive Representative Overrides
         │
         ▼
[6] Execute Feature Selection Engine (Strategy: hca_corr_perm, Policy: Top 1)
         │
         ▼
[7] Generate Final Feature Bundle Preview Card
         │
         ▼
[8] Press "► Create Model Builder" (Transfers feature list & lineage preset)
```

---

## 13. Feature Selection Output & Model Builder Handoff

When the user clicks **"► Create Model Builder"**:
1. **Bundle Packaging**: `build_final_feature_dataset()` packages the selected features and selection metadata (`strategy`, `thresholds`, `representative_policy`, `dataset_name`).
2. **Preset Persistence**: Saves the selection preset to `build_config_prefs.json`.
3. **Callback Invocation**: Invokes `_on_open_model_builder(features=features, dataset=dataset_name, analysis_feature_selection=lineage)`.
4. **Model Builder Handoff**:
   - Model Builder loads the target dataset.
   - Automatically checks the active checkboxes for the selected features across its three feature trees (**Tab 1: Feature Registry**, **Tab 2: Base Pipeline**, **Tab 3: Selected Experimental**).

---

## 14. Current Feature Lineage Support

| Lineage Dimension | Current Implementation Status |
|---|---|
| `feature_project_id` | **PRESERVED** in Dataset Metadata; inherited by Model Builder. |
| `pipeline_id` & `pipeline_snapshot_id` | **PRESERVED** in Dataset Metadata; inherited by Model Builder. |
| `registry_export_features` | **PRESERVED** in Dataset Metadata. |
| `base_pipeline_export_features` | **PRESERVED** in Dataset Metadata. |
| In-Analysis 3-Source Tabs | **NOT CURRENTLY IMPLEMENTED** (Analysis operates on a flat feature matrix). |
| Feature Source Preserved in Selection Bundle | **PRESERVED AS NAMES** (Model Builder resolves source classification on handoff). |

---

## 15. Relationship Matrix Across Subsystems

```
┌───────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Subsystem                 │ Role & Lifecycle Boundary                                              │
├───────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ **Feature Analysis Lab**  │ Pre-training statistical discovery, collinearity reduction, down-selection.│
│ **Model Builder**         │ Training configuration, model architecture, 3-source tree selection.  │
│ **Feature Studio**        │ Post-training model diagnostics (Importance, Drift, Distribution).     │
│ **Production Validation** │ Forward-testing against true unseen trading days (KEEP/WATCH/REMOVE).  │
└───────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## 16. Current Limitations

1. **Flat Feature Representation**: The Analysis Lab does not visually partition features into the three feature populations inside its module tabs.
2. **Single Dataset Scope**: An analysis run analyzes one dataset at a time; cross-dataset differential analysis is handled in Feature Studio Compare.
3. **In-Sample Boundary**: High permutation scores in Analysis Lab do not guarantee stability against unseen forward market drift.

---

## 17. Future Architecture Roadmap (Not Currently Implemented)

> [!NOTE]
> The following items represent conceptual future enhancements and are **not implemented in the current codebase**:

1. **Three-Source Analysis Tabs**: Adding visual sub-filters for Feature Registry, Base Pipeline, and Selected Experimental inside the Correlation and Permutation tables.
2. **Source-Aware Feature Scoring**: Introducing weighted composite scoring based on feature generation complexity.
3. **Cross-Run Historical Accumulation**: Tracking how often a feature is chosen by the Feature Selection Engine across multiple research campaigns.
4. **Automated Promotion Pipeline**: Direct linkage between Analysis Lab discovery ratings and Base Pipeline promotion candidates.

---

## 18. Complete End-to-End System Diagram

```
Feature Registry (206 Canonical Features)
         │
         ▼
Master Dataset Build (master_nifty_6s.db)
         │
         ▼
Experimental Pipeline Candidates (PL_0002+)
         │
         ▼
Analysis Dataset Export (analysis_*.parquet & metadata)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│                  FEATURE ANALYSIS LAB                  │
│  ├── Correlation Matrix & Pairwise Collinearity        │
│  ├── Hierarchical Clustering (HCA Feature Families)    │
│  ├── Mutual Information Non-Linear Scoring             │
│  ├── Permutation Importance Baseline Scoring           │
│  ├── Fast TreeSHAP Attribution                         │
│  └── Feature Selection Engine (hca_corr_perm)          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
               Selected Feature Bundle List
                            │
                            ▼
                      Model Builder
        (Tabs 1, 2, 3 Partition & XGBoost Train)
                            │
                            ▼
                  Trained Model Package
                 (models/<model_name>/)
                            │
                            ▼
                     Feature Studio
       (Importance, Distribution, Drift, Diagnostics)
                            │
                            ▼
                  Production Validation
              (True Unseen Days Resolution)
                            │
                            ▼
              KEEP / WATCH / REMOVE Audit History
```

---

## 19. Source Code Map

| Component | Source File | Key Class / Functions | Reads | Writes |
|---|---|---|---|---|
| **Analysis UI Panel** | [`apps/master_dataset_tk/feature_analysis_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_analysis_panel.py) | `FeatureAnalysisPanel` | Parquet datasets | `analysis.db`, UI State |
| **SQLite Store** | [`apps/chain_replay_ml/dataset_builder/analysis_lab_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_lab_store.py) | `_AnalysisDb`, `fingerprint_parquet`, `ensure_analysis_schema` | Parquet Metadata | `analysis.db` tables |
| **Correlation** | [`apps/chain_replay_ml/dataset_builder/analysis_correlation.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_correlation.py) | `compute_correlation_matrix`, `top_correlated_pairs` | Parquet columns | `correlation` table |
| **HCA Clustering** | [`apps/chain_replay_ml/dataset_builder/analysis_hca.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_hca.py) | `compute_hca_clusters`, `select_cluster_representatives` | Correlation matrix | `clusters` table |
| **Mutual Info** | [`apps/chain_replay_ml/dataset_builder/analysis_mutual_information.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_mutual_information.py) | `compute_mutual_information` | Features & Targets | `mutual_information` |
| **Permutation** | [`apps/chain_replay_ml/dataset_builder/analysis_permutation.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_permutation.py) | `run_permutation_importance` | Features & Targets | `permutation_importance` |
| **Tree SHAP** | [`apps/chain_replay_ml/dataset_builder/analysis_shap.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_shap.py) | `compute_shap_attribution` | Model & Parquet | `shap` table |
| **Feature Selection**| [`apps/chain_replay_ml/dataset_builder/analysis_feature_selection.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_feature_selection.py) | `select_features_by_strategy`, `build_final_feature_dataset` | Module results | Selection bundles |

---

## 20. Glossary

- **Feature Analysis Lab**: Phase 2 research workspace providing multi-module statistical and machine learning evaluation over Analysis Datasets.
- **Analysis Dataset**: A materialized Parquet feature matrix containing Feature Registry and Pipeline features.
- **`analysis.db`**: Local SQLite database storing all research run results, correlation pairs, clusters, and ratings.
- **Hierarchical Cluster Analysis (HCA)**: Tree-based clustering algorithm grouping collinear features into orthogonal feature families.
- **Representative Feature**: The primary centroid or highest-performing feature selected to represent a feature family.
- **Mutual Information**: Information-theoretic metric capturing linear and non-linear dependencies with the target.
- **Permutation Importance**: Feature scoring metric based on performance degradation when column values are shuffled.
- **Feature Selection Bundle**: The final list of down-selected feature names and selection metadata passed to Model Builder.
