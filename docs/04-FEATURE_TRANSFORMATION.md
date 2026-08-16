# AruMLStudio Feature Transformation & Pipeline Architecture

---

## 1. Scope & Purpose

This document provides complete technical and functional documentation for the **Feature Transformation and Pipeline Subsystem** in **AruMLStudio**.

In algorithmic options trading, raw domain features (such as underlying spot prices, option implied volatilities, order book metrics, and strike statistics) must be transformed into higher-order predictive signals across multiple time horizons. The Feature Transformation subsystem provides:
- **Deterministic Base Transformations**: Standard lags, differences, percentage returns, and cross-feature interactions.
- **Experimental Pipeline Discovery**: Manual interactive builder and combinatorial Auto Candidate generation.
- **Persistent Pipeline Registry**: Immutable pipeline records (`PL_0001`, `PL_0002+`), content-hashed snapshots, and candidate memberships.
- **Analysis Dataset Materialization**: Unified parquet export combining Feature Registry data with on-the-fly pipeline generators.
- **Feature Analysis Lab**: Multi-module statistical and machine learning evaluation (Correlation, HCA clustering, Mutual Information, Permutation, SHAP, Feature Selection).
- **Lineage Integrity**: Strict provenance propagation from Dataset &rarr; Model &rarr; Feature Studio &rarr; Unseen Production Validation.

---

## 2. Architecture Overview

```
                      ┌─────────────────────────────────────────┐
                      │             MASTER DATASET              │
                      │         (master_nifty_6s.db)            │
                      │    Materialized Feature Registry        │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │         FEATURE TRANSFORMATIONS         │
                      │  ┌─────────────────┐ ┌────────────────┐ │
                      │  │  Base Pipeline  │ │  Experimental  │ │
                      │  │    (PL_0001)    │ │   (PL_0002+)   │ │
                      │  └────────┬────────┘ └────────┬───────┘ │
                      └───────────┼───────────────────┼─────────┘
                                  │                   │
                                  └─────────┬─────────┘
                                            │
                                            ▼
                      ┌─────────────────────────────────────────┐
                      │            ANALYSIS DATASET             │
                      │         (analysis_*.parquet)            │
                      │  Registry ∪ Base ∪ Experimental Pipeline│
                      └─────────────────────┬───────────────────┘
                                            │
                      ┌─────────────────────┼─────────────────────┐
                      ▼                                           ▼
       ┌───────────────────────────────┐           ┌───────────────────────────────┐
       │     FEATURE ANALYSIS LAB      │           │         MODEL BUILDER         │
       │    (Research & Discovery)     │           │     (Create Model Sec. 5)     │
       │ • Correlation Matrix & Pairs  │           │ • Tab 1: Feature Registry     │
       │ • HCA (Feature Families)      │           │ • Tab 2: Base Pipeline        │
       │ • Mutual Information & SHAP   │           │ • Tab 3: Experimental Pipeline│
       │ • Feature Selection Engine    │           │ • Down-selects Model Features │
       └───────────────────────────────┘           └──────────────┬────────────────┘
                                                                  │
                                                                  ▼
                                                   ┌───────────────────────────────┐
                                                   │        TRAINED MODEL          │
                                                   │   models/<model_name>/        │
                                                   │ • config.json (lineage meta)  │
                                                   │ • pipeline_id & snapshot_id   │
                                                   └──────────────┬────────────────┘
                                                                  │
                                           ┌──────────────────────┴──────────────────────┐
                                           ▼                                             ▼
                            ┌─────────────────────────────┐               ┌─────────────────────────────┐
                            │       FEATURE STUDIO        │               │    PRODUCTION VALIDATION    │
                            │ • Importance (Gain, Perm)   │               │ • unseen_* Dataset Build    │
                            │ • Distribution (Moments)    │               │ • Exact Lineage Matching    │
                            │ • Drift (Holdout vs. WF)    │               │ • 3-Source Radio Partition  │
                            │ • Diagnostics (3-Source)    │               │ • KEEP / WATCH / REMOVE     │
                            └─────────────────────────────┘               └─────────────────────────────┘
```

---

## 3. Three Pipeline-Related Feature Populations

AruMLStudio maintains a strict architectural boundary separating three distinct feature populations:

```
                                  ALL MODEL FEATURES
                                     (N features)
                                          │
                 ┌────────────────────────┼────────────────────────┐
                 ▼                        ▼                        ▼
       ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
       │ Feature Registry  │    │   Base Pipeline   │    │Selected Experim.  │
       │ (Population 1)    │    │   (Population 2)  │    │  (Population 3)   │
       └───────────────────┘    └───────────────────┘    └───────────────────┘
```

### 3.1. Population 1: Feature Registry Features
- **Nature**: Canonical, domain-specific market features (prices, IVs, Greeks, straddle metrics, OI, volume, ratios).
- **Source of Truth**: Canonical Registry (`feature_registry_store.py`, `feature_domains.py`). 206 active features across 11 financial domains.
- **Scoping**: Organized and filtered by `feature_project_id` (e.g. `"all"`, `"chart"`).
- **Materialization**: Computed during the Master Dataset build and persisted directly into SQLite (`master_dataset_*.db`).

### 3.2. Population 2: Base Pipeline Features
- **Nature**: Deterministic, baseline mathematical transformations applied to active registry features.
- **Source of Truth**: `PIPELINE_OWNED_FEATURES` in `feature_migration.py` and `PL_0001` in `pipeline_registry_store.py`.
- **Generators**: Fixed families including `lag`, `diff`, `ret`, and `interact` (e.g. `spot_lag1`, `atm_iv_ce_diff5`, `spot_ret5`).
- **Materialization**: Generated dynamically when exporting an Analysis Dataset with `include_pipeline=True`. Never written to Master SQLite.
- **Metadata**: Recorded in dataset JSON as `base_pipeline_export_features` and `base_pipeline_export_count`.

### 3.3. Population 3: Selected Experimental Pipeline Features
- **Nature**: Novel candidate transformations generated through interactive manual configuration or automated combinatorial search.
- **Source of Truth**: Experimental Pipeline records in `pipeline_registry_store.json` (`PL_0002+`, `pipeline_type="manual"` or `"auto"`).
- **Lineage**: Identified by unique `pipeline_id` (e.g. `PL_0005`), `pipeline_name`, and immutable cryptographic `pipeline_snapshot_id`.
- **Selection**: Filtered down during Model Builder (Create Model Section 5.3) from the pipeline's `candidate_features`.
- **Metadata**: Recorded in dataset JSON as `pipeline_provenance` (containing `candidate_features`, `transformation_config`, and `pipeline_snapshot_id`).

### Architectural Boundary Summary
- **Master Dataset** $\equiv$ Feature Registry features only.
- **Analysis Dataset** $\equiv$ Feature Registry $\cup$ Base Pipeline $\cup$ Selected Experimental Pipeline $\cup$ Targets & Metadata.
- **Model Selected Features** $\equiv$ Subset of Analysis Dataset columns chosen during Model Builder.

---

## 4. Pipeline Registry (`pipeline_registry_store.py`)

The Pipeline Registry ([`apps/chain_replay_ml/dataset_builder/pipeline_registry_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/pipeline_registry_store.py)) manages persistent pipeline identities and candidate memberships.

### 4.1. File Location & Format
- **Path**: `<chart_data_dir>/pipeline_registry_store.json`
- **Format**: JSON document with sequential ID generators, pipeline dictionaries, and append-only audit history.

### 4.2. Schema Structure
```json
{
  "registry_version": "1.0",
  "created_on": "2026-08-16T10:00:00+00:00",
  "next_pipeline_id_seq": 6,
  "next_display_seq": 6,
  "pipelines": {
    "PL_0001": {
      "pipeline_id": "PL_0001",
      "name": "Pipeline_001 — Base",
      "type": "base",
      "status": "ready",
      "registry_feature_ids": [],
      "candidate_features": ["spot_lag1", "spot_diff1", "atm_iv_ce_ret5", "..."],
      "transformation_config": null,
      "created_at": "2026-08-16T10:00:00+00:00",
      "updated_at": "2026-08-16T10:00:00+00:00"
    },
    "PL_0005": {
      "pipeline_id": "PL_0005",
      "name": "Pipeline_005 (Auto Momentum & Volatility)",
      "type": "auto",
      "status": "ready",
      "registry_feature_ids": ["FR0001", "FR0012"],
      "candidate_features": ["spot_roll_mean_60", "atm_iv_ce_zscore_120", "..."],
      "transformation_config": {
        "transformation_pipeline_version": 1,
        "transformations": [...]
      },
      "created_at": "2026-08-16T12:30:00+00:00",
      "updated_at": "2026-08-16T12:35:00+00:00"
    }
  },
  "history": [
    {"ts": "2026-08-16T10:00:00+00:00", "action": "create", "pipeline_id": "PL_0001", "name": "Pipeline_001 — Base"},
    {"ts": "2026-08-16T12:30:00+00:00", "action": "create", "pipeline_id": "PL_0005", "name": "Pipeline_005"}
  ]
}
```

### 4.3. Base vs. Experimental Pipeline Rules
| Property | `PL_0001` (Base Pipeline) | `PL_0002+` (Experimental Pipelines) |
|---|---|---|
| **Pipeline Type** | `"base"` | `"manual"` or `"auto"` |
| **Creation** | Seeded automatically via `ensure_default_existing_pipeline()` | Created via Manual UI or Auto Candidate Generation |
| **Deletion** | **Protected**: Cannot be deleted (`ValueError`) | Can be deleted by user |
| **Export Role** | Automatically included in all Analysis Datasets | Selected explicitly as the experimental source |
| **Features** | Seeded from `PIPELINE_OWNED_FEATURES` | Dynamically generated from transformation configs |

---

## 5. Base Pipeline Engine

The Base Pipeline provides essential temporal and relational transforms across market features:
1. **Seeding**: `ensure_default_existing_pipeline(data_dir)` queries `PIPELINE_OWNED_FEATURES` from `feature_migration.py` and filters out user-excluded features (`active_pipeline_feature_names`).
2. **Generators**: Built via `build_pipeline_features_transformation_config(sample_interval_sec, exclude_features)` in [`apps/chain_replay_ml/dataset_builder/pipeline_features_config.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/pipeline_features_config.py).
3. **Execution**: During dataset export, base pipeline generators execute day-at-a-time over partitioned market ticks (`trading_day`, `token`), computing lags and rolling metrics.

---

## 6. Transformation Engine & Generator Families

The transformation engine in [`apps/chain_replay_ml/dataset_builder/transformations/`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/transformations/) implements modular, vectorized feature generators with Polars acceleration and Pandas fallbacks:

```
                            TRANSFORMATION ENGINE
                                      │
     ┌─────────────┬─────────────┬────┴────────┬─────────────┬─────────────┐
     ▼             ▼             ▼             ▼             ▼             ▼
  [Lag]       [Difference]    [Return]     [Rolling]     [Exp Rolling] [Interaction]
```

### 6.1. Generator Families Detail

| Family | Module | Operation | Parameters | Naming Convention |
|---|---|---|---|---|
| **Lag** | `lag.py` | Shift values backwards in time | `lag_seconds` ($6, 12, 30, 60, 120, 300\text{s}$) | `<feature>_lag<N>` (e.g. `spot_lag1`) |
| **Difference** | `difference.py` | Absolute change: $X_t - X_{t-k}$ | `lag_seconds` | `<feature>_diff<N>` (e.g. `futures_basis_diff5`) |
| **Return** | `return_transform.py` | Percentage change: $\frac{X_t - X_{t-k}}{X_{t-k}} \times 100$ | `lag_seconds` | `<feature>_ret<N>` (e.g. `spot_ret5`) |
| **Rolling Stats** | `rolling.py` | Windowed stats: `mean`, `std`, `min`, `max` | `window_seconds` ($\ge 30\text{s}$) | `<feature>_roll_<op>_<window>` (e.g. `spot_roll_mean_60`) |
| **Exp Rolling** | `exponential_rolling.py`| Exponential moving average: EMA, EWMA std | `period_seconds`, $\alpha$ | `<feature>_ema_<period>` (e.g. `atm_iv_ce_ema_120`) |
| **Normalization** | `normalization.py` | Rolling Z-score: $\frac{X_t - \mu_w}{\sigma_w}$ | `window_seconds` | `<feature>_zscore_<window>` (e.g. `pcr_oi_zscore_300`) |
| **Regime / Bucket**| `regime.py` | Discretize into quantile bins or thresholds | `n_bins`, `windows` | `<feature>_bucket_<bins>` (e.g. `spot_bucket_5`) |
| **Math** | `math_transform.py` | Elementwise: `abs`, `log`, `sign`, `sqrt` | `operation` | `<feature>_<op>` (e.g. `delta_spread_abs`) |
| **Interaction** | `interaction.py` | Cross-feature arithmetic: $+$, $-$, $\times$, $/$, $|\Delta|$ | `operand_a`, `operand_b`, `op` | `<featA>_x_<featB>`, `<featA>_div_<featB>` |

---

## 7. Manual Pipeline Workflow

The **Manual Pipeline** interface in `master_data_panel.py` and `pipeline_registry_panel.py` enables targeted candidate creation:

```
   1. Select Target Pipeline (PL_0002+) in Pipeline Registry Panel
         │
         ▼
   2. Select Source Feature(s) from Canonical Registry
         │
         ▼
   3. Configure Transformation Parameters (Family, Horizons, Operations)
         │
         ▼
   4. Preview Generated Feature Columns & Sample Calculations
         │
         ▼
   5. Save Transformation & Append Candidate Features to Pipeline Record
```

---

## 8. Automatic Pipeline & Auto Candidate Generation

The Auto Candidate subsystem ([`apps/master_dataset_tk/auto_candidate_generation.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/auto_candidate_generation.py)) performs high-throughput combinatorial feature synthesis:

### 8.1. Generation Flow
1. **Source Feature Resolution**: Extracts active Feature Registry features or existing pipeline candidates.
2. **Combinatorial Synthesis**: Expands selected transformations across defined time horizons:
   $$\text{Combinations} \approx N_{\text{sources}} \times \left( H_{\text{lag}} + H_{\text{diff}} + H_{\text{ret}} + 4 H_{\text{roll}} + 2 H_{\text{exp}} + H_{\text{norm}} \right) + N_{\text{interactions}}$$
3. **Policy Rejection & De-duplication**: Filters out prohibited metadata columns (`META_SKIP_COLUMNS`), retired features, identity transforms, and pre-existing candidates.
4. **Pipeline Record Creation**: Writes the combined `transformation_config` and `candidate_features` list into a newly allocated `PL_000X` record (`pipeline_type="auto"`).

---

## 9. Feature Analysis Lab (Analysis Tab)

The **Analysis Tab** ([`apps/master_dataset_tk/feature_analysis_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_analysis_panel.py)) provides a read-only scientific evaluation environment operating directly over materialized **Analysis Datasets** (`.parquet`).

### 9.1. Analysis Lab Modules
1. **Correlation Engine** (`analysis_correlation.py`):
   - Computes full Spearman and Pearson correlation matrices.
   - Identifies collinear feature pairs ($|r| \ge 0.95$) to prevent multicollinearity in downstream models.
2. **Hierarchical Cluster Analysis (HCA / Feature Families)** (`analysis_hca.py`, `analysis_family_review.py`):
   - Clusters high-dimensional feature spaces into orthogonal feature families.
   - Identifies the most representative centroid feature for each cluster.
3. **Mutual Information Engine** (`analysis_mutual_information.py`):
   - Evaluates non-linear dependencies between candidate features and target returns.
4. **Permutation Importance & Fast SHAP** (`analysis_permutation.py`, `analysis_shap.py`):
   - Rapidly benchmarks standalone feature predictive power prior to full model training.
5. **Feature Selection Engine** (`analysis_feature_selection.py`):
   - Applies multi-stage down-selection strategies (`hca_corr_perm`, `hca_only`, `top_1`, `top_n`) to package filtered candidate feature bundles for Model Builder.
- **Persistence**: All research runs, correlation pairs, clusters, and ratings are stored in SQLite at `<chart_data_dir>/analysis.db`.

---

## 10. Manual vs. Auto Pipeline Comparison

| Dimension | Manual Pipeline | Auto Candidate Generation |
|---|---|---|
| **Initiation** | Interactive UI configuration | Combinatorial batch generator |
| **Source Scope** | Explicitly picked source features | Broad registry/pipeline domain sets |
| **Transformations** | Selected per individual generator | Batch-applied across all active families |
| **Horizons** | Custom user-specified seconds | Standard grids ($6, 12, 30, 60, 120, 300\text{s}$) |
| **Pipeline Type** | `"manual"` | `"auto"` |
| **Output Volume** | Small ($1\dots 50$ features) | Large ($100\dots 1,000+$ candidates) |
| **Use Case** | Specific hypothesis testing | Broad exploratory feature discovery |

---

## 11. Pipeline Feature Naming Conventions

To guarantee deterministic reconstruction, generated feature names encode their full transformation genealogy:

```
  [Source Feature]  _  [Transformation]  _  [Operation / Window]
      spot                  lag                  1
      spot                  diff                 5
      spot                  ret                  5
      spot                  roll_mean            60
   atm_iv_ce                zscore               120
     spot         _x_    atm_iv_ce
```

- **Collision Prevention**: `_policy_reject_names` checks new candidate names against canonical registry names, existing candidate lists, and forbidden column names.

---

## 12. Pipeline Provenance & Snapshot Integrity

Reproducibility is enforced through immutable snapshots:

### 12.1. Snapshot Generation (`build_pipeline_snapshot`)
When an Analysis Dataset or Model is built, the system captures an immutable snapshot of the active pipeline record:
```json
{
  "pipeline_id": "PL_0005",
  "pipeline_name": "Pipeline_005 (Auto Momentum)",
  "pipeline_type": "auto",
  "registry_feature_ids": ["FR0001", "FR0012"],
  "candidate_features": ["spot_roll_mean_60", "atm_iv_ce_zscore_120"],
  "transformation_config": { ... },
  "pipeline_updated_at": "2026-08-16T12:35:00+00:00",
  "snapshot_at": "2026-08-16T12:40:00+00:00",
  "pipeline_snapshot_id": "ca5945f58f8b1a2c"
}
```

### 12.2. Cryptographic Hash (`compute_pipeline_snapshot_id`)
$$\text{pipeline\_snapshot\_id} = \text{SHA256}\Big(\text{canonical\_json}(\text{pipeline\_id}, \text{name}, \text{type}, \text{candidates}, \text{config})\Big)[:16]$$
- If a user modifies `PL_0005` later, the dataset's `pipeline_snapshot_id` remains unchanged, preserving exact historical lineage.

---

## 13. Analysis Dataset Creation (`create_analysis_dataset`)

Located in [`apps/chain_replay_ml/dataset_builder/analysis_dataset_export.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_dataset_export.py):

### 13.1. Staged Execution Pipeline
1. **Stage 1 (`registry`)**: Reads selected Master Dataset trading days from SQLite; materializes `registry_export_features` scoped by `feature_project_id`.
2. **Stage 2 (`pipeline`)**: Executes transformation generators day-at-a-time for `base_pipeline_export_features` $\cup$ experimental `candidate_features`.
3. **Stage 3 (`no_null`)**: (Optional) Drops rows with NaN values across all feature columns.
4. **Stage 4 (`premium`)**: (Optional) Filters rows based on LTP option premium boundaries.
5. **Stage 5 (`finalize`)**: Writes Parquet dataset (`analysis_*.parquet`) and companion JSON metadata (`analysis_*.json`).

---

## 14. Dataset Metadata Schema

The companion JSON metadata file (`analysis_*.json`) stores the complete lineage:

| Field | Type | Origin / Producer | Consumer |
|---|---|---|---|
| `feature_project_id` | `str` | Master Dataset / Config | Model Builder, Unseen Verifier |
| `registry_export_features` | `list[str]` | Master Registry Export | Feature Classification Engine |
| `registry_export_count` | `int` | Master Registry Export | UI Header & Stats |
| `base_pipeline_export_features` | `list[str]` | Base Pipeline Generator | Feature Sources Catalog |
| `base_pipeline_export_count` | `int` | Base Pipeline Generator | UI Header & Stats |
| `pipeline_id` | `str` | Selected Experimental Pipeline | Training Engine, Lineage Verifier |
| `pipeline_name` | `str` | Pipeline Registry | Display Headers |
| `pipeline_snapshot_id` | `str` | `build_pipeline_snapshot` | Unseen Dataset Generator |
| `pipeline_provenance` | `dict` | `build_pipeline_snapshot` | Model Package, Diagnostics |
| `include_registry` | `bool` | Dataset Export Options | Feature Loader |
| `include_pipeline` | `bool` | Dataset Export Options | Feature Loader |

---

## 15. Feature Source Classification Engine

Located in [`apps/chain_replay_ml/dataset_builder/feature_sources_catalog.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/feature_sources_catalog.py):

### Precedence Rules in `classify_dataset_feature_source`:
```
                     Input Feature Name
                             │
                             ▼
              Is in Registry Export List? ────────► DATASET_SOURCE_FEATURE_REGISTRY
                             │ No
                             ▼
            Is in Base Pipeline Export List? ────► DATASET_SOURCE_BASE_PIPELINE
                             │ No
                             ▼
            Is in Experimental Candidates / ────► DATASET_SOURCE_OTHER_PIPELINE
            Matches Pipeline Prefix?
                             │ No
                             ▼
                     UNCLASSIFIED (Error)
```

---

## 16. Model Builder Integration

During **Create Model (Section 5)** in [`apps/master_dataset_tk/model_builder/panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_builder/panel.py):

```
                        Loaded Analysis Dataset
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Tab 1: Registry  │      │ Tab 2: Base Pipe │      │ Tab 3: Experim.  │
│ Scoped by        │      │ Filtered by Base │      │ Scoped by active │
│ feature_project  │      │ Generators       │      │ pipeline_id      │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
                   Combined Model Feature Selection
                     (self.state.features list)
                                   │
                                   ▼
                        Model Package config.json
               (records feature_project_id & pipeline_id)
```

---

## 17. Feature Studio & Production Validation Lineage

### 17.1. Feature Studio Diagnostics
- Benchmarks **Walk-Forward vs. Holdout**.
- In `DiagnosticsStudioPanel`, features are partitioned into 3 tabs (`Feature Registry`, `Base Pipeline`, `Selected Experimental`) using `partition_diagnostic_rows`.

### 17.2. Production Validation Unseen Dataset Resolution
- Benchmarks **Holdout vs. Unseen Forward Days**.
- `unseen_dataset.py` extracts `pipeline_id`, `pipeline_snapshot_id`, `feature_project_id`, `include_pipeline`, and `include_registry` directly from the parent dataset metadata.
- Computes `unseen_dataset_identity_hash`:
  $$\text{hash} = \text{SHA256}\Big(\text{master\_db} + \text{unseen\_days} + \text{feature\_project\_id} + \text{pipeline\_id} + \text{pipeline\_snapshot\_id}\Big)[:8]$$
- Generates or reuses `unseen_<slug>_<hash>.parquet`, ensuring 100% feature parity with zero manual pipeline re-entry.

---

## 18. End-to-End Pipeline Lifecycle Diagram

```
[1] Feature Registry (206 Canonical Features)
         │
         ▼
[2] Feature Project Scoping (feature_project_id = 'all' / 'chart')
         │
         ▼
[3] Master Dataset SQLite Build (master_nifty_6s.db)
         │
         ▼
[4] Transformation Specification (Manual UI or Auto Candidate Batch)
         │
         ▼
[5] Pipeline Registry Persistence (PL_0001 Base, PL_0005 Experimental)
         │
         ▼
[6] Pipeline Snapshot Generation (pipeline_snapshot_id = 'ca5945f58f8b1a2c')
         │
         ▼
[7] Analysis Dataset Materialization (analysis_*.parquet & metadata)
         │
         ▼
[8] Feature Analysis Lab (Correlation, HCA Clusters, Permutation, Selection)
         │
         ▼
[9] Model Builder Feature Selection (Registry + Base Pipeline + Experimental)
         │
         ▼
[10] Model Training & Packaging (models/<model_name>/)
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
[11] Feature Studio Profiling             [12] Production Validation
     • Importance (Native/Perm/SHAP)           • Unseen Dataset Hash Resolution
     • Distribution Moments                    • Holdout vs. Unseen Verification
     • Drift Analysis                          • 3-Source Radio Buttons
     • 3-Source Diagnostic Tabs                • KEEP / WATCH / REMOVE Audit
```

---

## 19. Storage Map

| Artifact | Typical Path | Format | Created By | Read By | Purpose |
|---|---|---|---|---|---|
| **Pipeline Registry** | `<chart_data_dir>/pipeline_registry_store.json` | JSON | `pipeline_registry_store` | Pipeline Panels, Export Engine | Persistent catalog of all pipelines |
| **Analysis Dataset** | `datasets/analysis_datasets/analysis_<name>.parquet` | Parquet | `analysis_dataset_export` | Model Builder, Analysis Lab | Materialized feature matrix |
| **Dataset Metadata** | `datasets/analysis_datasets/analysis_<name>.json` | JSON | `analysis_dataset_export` | Model Builder, Prod Validation | Complete lineage and provenance |
| **Analysis Lab DB** | `<chart_data_dir>/analysis.db` | SQLite | `analysis_lab_store` | Feature Analysis Lab | Correlation, HCA, & MI run cache |
| **Pipeline Prefs** | `<chart_data_dir>/pipeline_features_prefs.json` | JSON | `pipeline_features_prefs` | Export Engine, UI Panels | Excluded/retired pipeline features |
| **Model Config** | `models/<model_name>/config.json` | JSON | `model_builder` | Feature Studio, Prod Validation | Model parameters & lineage IDs |
| **Unseen Dataset** | `datasets/analysis_datasets/unseen_<slug>_<hash>.parquet` | Parquet | `unseen_dataset` | Production Validation | Unseen out-of-sample feature data |

---

## 20. Source Code Map

| Component | Source File | Key Class / Functions | Reads | Writes |
|---|---|---|---|---|
| **Pipeline Store** | [`apps/chain_replay_ml/dataset_builder/pipeline_registry_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/pipeline_registry_store.py) | `create_pipeline`, `build_pipeline_snapshot`, `ensure_default_existing_pipeline` | Store JSON | `pipeline_registry_store.json` |
| **Base Config** | [`apps/chain_replay_ml/dataset_builder/pipeline_features_config.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/pipeline_features_config.py) | `build_pipeline_features_transformation_config` | Preferences | Generator configs |
| **Auto Candidates** | [`apps/master_dataset_tk/auto_candidate_generation.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/auto_candidate_generation.py) | `build_auto_candidate_transformation_config`, `_policy_reject_names` | Registry catalog | Candidate lists |
| **Analysis Export** | [`apps/chain_replay_ml/dataset_builder/analysis_dataset_export.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_dataset_export.py) | `create_analysis_dataset` | Master SQLite, Pipeline Store | `analysis_*.parquet`, `analysis_*.json` |
| **Sources Catalog** | [`apps/chain_replay_ml/dataset_builder/feature_sources_catalog.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/feature_sources_catalog.py) | `classify_dataset_feature_source`, `dataset_registry_export_feature_names` | Dataset metadata | Source categories |
| **Feature Analysis**| [`apps/master_dataset_tk/feature_analysis_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_analysis_panel.py) | `FeatureAnalysisPanel` | Parquet datasets | `analysis.db` |
| **Model Builder** | [`apps/master_dataset_tk/model_builder/panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_builder/panel.py) | `ModelBuilderPanel` | Dataset JSON/Parquet | `models/<model>/config.json` |
| **Unseen Lineage** | [`apps/chain_replay_ml/production_validation/unseen_dataset.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/unseen_dataset.py) | `resolve_or_create_unseen_dataset`, `unseen_dataset_identity_hash` | Parent dataset metadata | `unseen_*.parquet` |

---

## 21. Architectural Invariants

1. **Master Dataset Exclusivity**: Master Dataset stores canonical Feature Registry data only. Pipeline features are never materialized into Master SQLite.
2. **Analysis Dataset Coexistence**: Analysis Dataset is the sole storage layer where Feature Registry, Base Pipeline, and Experimental Pipeline features coexist.
3. **Immutability of Snapshots**: Once an Analysis Dataset is materialized, its `pipeline_snapshot_id` is immutable. Subsequent edits to pipeline definitions create new snapshot IDs and do not corrupt historical datasets.
4. **Deterministic Feature Classification**: Feature classification is strictly deterministic: Registry $\prec$ Base Pipeline $\prec$ Experimental Pipeline.
5. **Zero-Leakage Unseen Resolution**: Production Validation must regenerate unseen features using the parent model's exact `feature_project_id`, `pipeline_id`, and `pipeline_snapshot_id`.

---

## 22. Troubleshooting

| Symptom | Cause | Diagnostic Approach | Resolution |
|---|---|---|---|
| `"Select an experimental pipeline when Pipeline Features is enabled"` | Pipeline Features was toggled on without selecting a target experimental pipeline ID. | Check Analysis Dataset export parameters. | Select an experimental pipeline (`PL_0002+`) or disable Pipeline Features. |
| `"The Base pipeline cannot be used as an experimental pipeline source"` | `PL_0001` was passed as `pipeline_id`. | Inspect pipeline dropdown selection. | Select an Auto/Manual experimental pipeline; Base features are included automatically. |
| `Pipeline feature missing from Analysis Dataset` | Feature name was rejected by policy or excluded in preferences. | Check `pipeline_features_prefs.json` and No-Null drop report. | Remove feature from exclusion list or relax No-Null threshold. |
| `Unseen dataset requests manual pipeline selection` | Parent dataset metadata lacked `pipeline_id` propagation. | Check parent dataset `config.json` metadata. | Ensure `pipeline_id` and `pipeline_snapshot_id` are saved in parent dataset JSON. |
| `Duplicate candidate feature generated` | Combinatorial generator produced an existing name. | Check `CandidateGenerationReport.duplicate_names`. | Automatically handled: duplicates are deduplicated during generation. |

---

## 23. Known Limitations

### Current Implementation Limitations:
- **Combinatorial Explosion in Auto Mode**: Generating interactions across $\ge 50$ features creates large pairwise sets; the generator currently caps interaction candidates to the first 40 source features.
- **Single Experimental Pipeline per Dataset**: An Analysis Dataset currently binds to exactly one experimental pipeline (`PL_000X`) in addition to the Base Pipeline (`PL_0001`).

### Intentional Architecture:
- **Day-at-a-Time Pipeline Execution**: Transformations compute one trading day at a time to prevent multi-gigabyte RAM spikes across high-frequency tick data.
- **Immutable Provenance Hashes**: Pipelines use cryptographic content hashes rather than timestamps to guarantee identical feature generation across distributed runs.

---

## 24. Glossary

- **Feature Registry**: Canonical catalog of 206 domain-specific financial features.
- **Feature Project**: Organizational grouping (`feature_project_id`) scoping a subset of Feature Registry features.
- **Base Pipeline (`PL_0001`)**: Deterministic baseline transformations (`lag`, `diff`, `ret`, `interact`) included in all pipeline-enabled datasets.
- **Experimental Pipeline (`PL_0002+`)**: Novel transformation pipelines generated manually or automatically.
- **Candidate Feature**: A newly transformed feature column generated by a pipeline definition.
- **Pipeline Snapshot (`pipeline_snapshot_id`)**: A cryptographic content hash of a pipeline's definition guaranteeing immutable historical replay.
- **Analysis Dataset**: A materialized Parquet feature matrix containing Feature Registry and Pipeline features used for research and model training.
- **Master Dataset**: The core SQLite database storing materialized Feature Registry ticks for a market.
- **`pipeline_provenance`**: The metadata dictionary recording pipeline ID, snapshot ID, candidate features, and generator parameters.
