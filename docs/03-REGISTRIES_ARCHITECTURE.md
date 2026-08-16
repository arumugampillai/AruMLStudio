# AruMLStudio Registries Architecture — Feature, Dataset & Model Registries

---

## 1. Registry Architecture & Data Lifecycle Overview

AruMLStudio is an algorithmic machine learning platform for options trading built upon a strict, three-tiered registry hierarchy and an explicit, non-overlapping data lifecycle:

```
                                      ARUMLSTUDIO DATA LIFECYCLE
                                      
  Raw Tick Data ──► Master Dataset ──► Feature Transformation ──► Analysis Dataset ──► Create Model ──► Model Registry ──► Production Validation
   (Market Feeds)    (SQLite .db)      (Pipeline Engine)          (Parquet Matrix)     (3-Source Pick)   (Trained Package)   (Unseen Days Replay)
```

```
                            ┌────────────────────────────────────────────────────────┐
                            │                    FEATURE REGISTRY                    │
                            │  "What features exist, who owns them, how are they     │
                            │   grouped, and which Feature Project contains them?"   │
                            └───────────────────────────┬────────────────────────────┘
                                                        │
                                                        ▼
                            ┌────────────────────────────────────────────────────────┐
                            │                    DATASET REGISTRY                    │
                            │  "What datasets exist, how were they created, what     │
                            │   features do they contain, and what is their lineage?"│
                            └───────────────────────────┬────────────────────────────┘
                                                        │
                                                        ▼
                            ┌────────────────────────────────────────────────────────┐
                            │                     MODEL REGISTRY                     │
                            │  "What models exist, what dataset/features created     │
                            │   them, what hyperparams were used, and how did they   │
                            │   perform?"                                            │
                            └────────────────────────────────────────────────────────┘
```

### 1.1. Core Lifecycle Stages & Clear Architectural Boundaries

1. **Master Dataset (Foundation)**:
   - Built directly from **raw market tick data**.
   - Contains **Feature Registry features only** (scoped by `feature_project_id`).
   - Does **NOT** contain Base Pipeline or Selected Experimental Pipeline features.
   - Designed to be a stable foundational dataset; routine feature experimentation **does not rebuild the Master Dataset**.

2. **Feature Transformation (Pipeline & Enrichment)**:
   - Takes the stable **Master Dataset as its input**.
   - Creates transformed and experimental pipeline features via manual or auto candidate generation.
   - Materializes the **Analysis Dataset** containing the appropriate combination of Registry, Base Pipeline, and Selected Experimental features.

3. **Create Model (Model Builder)**:
   - Consumes the **Analysis Dataset**, not raw tick data.
   - Reads Analysis Dataset metadata and feature lineage, presenting the three feature populations in distinct tree tabs.
   - User selects the final model features; training executes on exactly that final selected subset.

4. **Model Registry (Trained Package & Lineage)**:
   - Stores the trained model package, tree weights, and training configuration.
   - Preserves complete dataset and pipeline lineage (`feature_project_id`, `pipeline_id`, `pipeline_snapshot_id`, `selected_features`).

5. **Production Validation (Out-of-Sample Verification)**:
   - Uses the Model Registry lineage to resolve or generate the appropriate **Unseen Dataset**.
   - The Unseen Dataset reproduces the exact feature population required by the model without manual re-entry.

---

## 2. Feature Registry

### 2.1. Purpose & Source of Truth
The **Feature Registry** provides a curated, domain-specific taxonomy of 206 canonical market features (underlying spot, options chain quotes, Greeks, implied volatilities, order flow, and open interest).

- **Primary Store**: `<chart_data_dir>/feature_registry_store.json` ([`apps/chain_replay_ml/dataset_builder/feature_registry_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/feature_registry_store.py))
- **Schema Definitions**: [`apps/chain_replay_ml/dataset_builder/schema_feature_meta.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/schema_feature_meta.py) and [`feature_domains.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/feature_domains.py)

### 2.2. Feature Identity Schema
A canonical feature is uniquely identified by its snake_case `name` and permanent `feature_id` (e.g. `FR0012` &rarr; `atm_iv_ce`).

| Field Name | Type | Required | Meaning | Example | Source File |
|---|---|---|---|---|---|
| `feature_id` | `str` | Yes | Stable alphanumeric canonical identifier | `"FR0012"` | `feature_registry_store.py` |
| `name` | `str` | Yes | Unique canonical snake_case name | `"atm_iv_ce"` | `feature_domains.py` |
| `primary_domain`| `str` | Yes | 1 of 11 financial taxonomy domains | `"implied_volatility"`| `feature_domains.py` |
| `display_name` | `str` | Yes | Human-readable label for UI | `"ATM Implied Volatility (CE)"`| `schema_feature_meta.py` |
| `description` | `str` | Yes | Technical and economic explanation | `"Black-Scholes IV for ATM CE"`| `schema_feature_meta.py` |
| `data_type` | `str` | Yes | Value classification | `"Volatility"`, `"Price"`, `"Ratio"`| `feature_domains.py` |
| `ownership` | `str` | Yes | Extraction tier | `"base"`, `"computed_base"` | `feature_ownership.py` |
| `depends_on` | `list[str]`| Yes | Raw feed input dependencies | `["ltp", "spot", "strike"]` | `schema_feature_meta.py` |
| `can_apply_lag`| `bool` | Yes | Transformation compatibility flag | `True` | `feature_domains.py` |
| `status` | `str` | Yes | Lifecycle status | `"production"`, `"deprecated"` | `feature_registry_store.py` |

---

### 2.3. Canonical Financial Domains
All 206 canonical features belong to exactly one of **11 Primary Domains** (`DOMAIN_ORDER` in `feature_domains.py`):
1. `price_premium` ("Price & Premium") — 24 features
2. `spot_futures` ("Spot & Futures") — 28 features
3. `greeks` ("Greeks") — 32 features
4. `implied_volatility` ("Implied Volatility") — 22 features
5. `open_interest` ("Open Interest") — 18 features
6. `volume_liquidity` ("Volume & Liquidity") — 16 features
7. `chain_analytics` ("Chain Analytics") — 26 features
8. `historical_context` ("Historical Context") — 18 features
9. `market_structure` ("Market Structure") — 14 features
10. `time_session` ("Time & Session") — 12 features
11. `metadata` ("Metadata") — Token & partition columns

---

### 2.4. Feature Projects (`feature_project_id`)

A **Feature Project** defines a project-scoped subset of Feature Registry features and custom display groups.

```
                              FEATURE REGISTRY STORE
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     Project: "all" (Reserved)                       Project: "chart" (Custom)
     • All 206 active registry features              • Selected subset (e.g. 45 features)
     • 11 Canonical Domain Groups                    • Custom Groups ("Chart Core", "EMA")
```

#### Schema of a Feature Project (`feature_registry_projects.json`):
```json
{
  "label": "chart",
  "description": "Core spot and IV features for chart analysis models.",
  "group_ids": ["chart_core", "volatility"],
  "feature_names": ["spot", "futures_ltp", "atm_iv_ce", "atm_iv_pe", "spot_ema20"],
  "project_groups": [
    {"id": "chart_core", "label": "Chart Core"}
  ],
  "feature_group_map": {
    "spot": "chart_core",
    "futures_ltp": "chart_core"
  },
  "reserved": false,
  "created_at": "2026-08-16T10:00:00Z"
}
```

- **UI Tree Construction**: `project_registry_feature_source(project_id)` loads the project document, filters active features, and renders the hierarchical tree in Master Dataset Build and Model Builder.

---

## 3. Dataset Registry

### 3.1. Purpose & Storage
The **Dataset Registry** manages all materialized dataset matrices and their lineage.
- **Master Dataset Store**: SQLite database (`master_dataset_<market>_<interval>.db`) managed via [`apps/chain_replay_ml/dataset_builder/master_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/master_store.py).
- **Analysis & Unseen Datasets**: Parquet feature matrices stored under `datasets/analysis_datasets/` with companion JSON metadata files.

---

### 3.2. Dataset Types Comparison

| Dataset Type | Purpose | Input Source | Feature Composition | Used By |
|---|---|---|---|---|
| **Master Dataset** (`.db`) | Historical tick database for a market. | Raw normalized tick feeds. | **Feature Registry features only** (scoped by `feature_project_id`). | Dataset Exporters, Analysis Builders. |
| **Analysis Dataset** (`.parquet`) | Training and research feature matrix. | Master Dataset + Pipeline Snapshot. | **Registry $\cup$ Base Pipeline $\cup$ Selected Experimental**. | Analysis Lab, Model Builder. |
| **Unseen Dataset** (`.parquet`) | Out-of-sample forward testing matrix. | Master Dataset unseen days + Parent Model Lineage. | **Exact Parent Lineage Match** (Registry $\cup$ Pipeline). | Production Validation. |

---

### 3.3. Complete Dataset Metadata Schema (`analysis_*.json`)

| Metadata Field | Type | Created By | Read By | Purpose & Meaning | Immutable? |
|---|---|---|---|---|---|
| `dataset_name` | `str` | `analysis_dataset_export` | UI, Model Builder | Unique dataset name identifier | Yes |
| `dataset_kind` | `str` | `analysis_dataset_export` | Model Builder | `"analysis"` or `"unseen"` | Yes |
| `feature_project_id` | `str` | Dataset Config | Model Builder, Prod Validation | Scoping Feature Project ID (`"all"`, `"chart"`) | Yes |
| `registry_export_features`| `list[str]`| Master Export | Feature Sources Catalog | List of materialized Feature Registry columns | Yes |
| `registry_export_count` | `int` | Master Export | Header & Stats | Count of Feature Registry columns | Yes |
| `base_pipeline_export_features`| `list[str]`| Base Pipeline | Feature Sources Catalog | List of exported Base Pipeline columns | Yes |
| `base_pipeline_export_count` | `int` | Base Pipeline | Header & Stats | Count of Base Pipeline columns | Yes |
| `pipeline_id` | `str` | Pipeline Store | Model Package, Lineage Replay | Bound experimental pipeline ID (`"PL_0005"`) | Yes |
| `pipeline_name` | `str` | Pipeline Store | Display UI | Bound pipeline display label | Yes |
| `pipeline_snapshot_id`| `str` | `pipeline_registry_store`| Unseen Verifier | Cryptographic SHA256 content hash | **Yes** |
| `pipeline_provenance` | `dict` | `pipeline_registry_store`| Model Builder, Diagnostics | Snapshot recording candidate feature list & config | Yes |
| `include_registry` | `bool` | Dataset Config | Export Engine | Whether registry features were included | Yes |
| `include_pipeline` | `bool` | Dataset Config | Export Engine | Whether pipeline features were included | Yes |
| `row_count` | `int` | Parquet Writer | UI Panels | Total number of sample rows | Yes |
| `feature_count` | `int` | Parquet Writer | UI Panels | Total number of feature columns | Yes |

---

### 3.4. Unseen Dataset Resolution & Identity Hash
Located in [`apps/chain_replay_ml/production_validation/unseen_dataset.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/unseen_dataset.py):

$$\text{unseen\_dataset\_identity\_hash} = \text{SHA256}\Big(\text{master\_db} + \text{unseen\_days} + \text{feature\_project\_id} + \text{pipeline\_id} + \text{pipeline\_snapshot\_id} + \text{flags}\Big)[:8]$$

- `_existing_unseen_valid()` checks if `unseen_<slug>_<hash>.parquet` already exists with matching metadata. If valid, it reuses the file; otherwise, it triggers a clean generation to prevent data leakage or stale evaluations.

---

### 3.5. Master Dataset (Foundational Boundary)
- **Built from Raw Tick Data**: The Master Dataset is built directly from raw options and spot tick feeds and persisted in SQLite.
- **Contains Feature Registry Only**: It contains only canonical Feature Registry features scoped by `feature_project_id`.
- **Exclusivity**: It does **NOT** contain Base Pipeline features or Selected Experimental Pipeline features.
- **Stability**: Routine downstream feature experimentation and pipeline builds **never rebuild the Master Dataset**.

---

### 3.6. Analysis Dataset (Enrichment Boundary)
- **Takes Master Dataset as Input**: Analysis Datasets are materialized by reading the stable Master Dataset and executing pipeline transformations.
- **Three-Source Coexistence**: It is the designated boundary where Feature Registry, Base Pipeline, and Selected Experimental Pipeline features coexist in a single Parquet matrix.
- **Input to Create Model**: Model Builder consumes the Analysis Dataset, **never raw tick data**.

---

### 3.7. Unseen Dataset (Production Validation Boundary)
- **Lineage Replay**: Dynamically reproduces the parent model's exact feature set (`feature_project_id`, `pipeline_id`, `pipeline_snapshot_id`) on unseen trading days.
- **Zero-Leakage Guarantee**: Prevents evaluating models on stale or mismatched feature definitions.

---

## 4. Model Registry

### 4.1. Purpose & Structure
The **Model Registry** ([`apps/chain_replay_ml/training/registry.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/training/registry.py)) indexes, manages, protects, and evaluates trained model packages stored under `<chart_data_dir>/models/<model_name>/`.

### 4.2. Model Package Artifact Files
Every registered model package contains a standardized directory structure:

| Artifact File | Format | Created By | Read By | Purpose & Contents |
|---|---|---|---|---|
| `config.json` | JSON | `orchestrator.py` | Feature Studio, Prod Validation | Complete model config, dataset lineage, and selected features list |
| `training_config.json` | JSON | `orchestrator.py` | Model Registry UI | Hyperparameters, loss functions, walk-forward fold settings |
| `metrics.json` | JSON | `evaluator.py` | Model Explorer | Out-of-fold and holdout performance metrics (MAE, RMSE, R²) |
| `model.ubj` / `model.json`| Binary/JSON| Trainer (XGBoost/CatBoost)| Inference Engine | Serialized trained model tree weights |
| `feature_importance.csv` | CSV | `evaluator.py` | Importance Studio | Native gain, split, and permutation feature importances |
| `training_metadata.json` | JSON | `orchestrator.py` | Model Details | Execution timings, sample counts, hardware devices |
| `dataset_build_snapshot.json`| JSON | `dataset_loader.py` | Lineage Verifier | Hash of parent dataset at training time |

---

### 4.3. Complete Model Metadata Schema (`config.json`)

```json
{
  "model_name": "Future_LTP_5m_WF_1168f_XGB_2243_14",
  "algorithm": "xgboost",
  "target": "future_ltp_5m",
  "prediction_horizon_sec": 300,
  "dataset_name": "analysis_nifty_6s_exp005",
  "dataset_path": "datasets/analysis_datasets/analysis_nifty_6s_exp005.parquet",
  "feature_project_id": "all",
  "features": [
    "spot",
    "atm_iv_ce",
    "spot_lag1",
    "spot_roll_mean_60"
  ],
  "feature_count": 4,
  "pipeline_id": "PL_0005",
  "pipeline_name": "Pipeline_005 (Auto Momentum)",
  "pipeline_type": "auto",
  "pipeline_snapshot_id": "ca5945f58f8b1a2c",
  "pipeline_provenance": {
    "pipeline_id": "PL_0005",
    "candidate_features": ["spot_roll_mean_60", "atm_iv_ce_zscore_120"],
    "snapshot_at": "2026-08-16T12:00:00Z"
  },
  "training_days": ["2026-06-01", "2026-06-02", "2026-06-03"],
  "holdout_days": ["2026-06-04"],
  "status": "ready",
  "created_at": "2026-08-16T14:00:00Z"
}
```

---

### 4.4. Model &rarr; Feature Subset Mapping
An Analysis Dataset may contain **1,186 feature columns**, but a trained model may select only **583 features** (via Model Builder Section 5 or Feature Selection Lab).
- The model stores its exact subset in `config.json` under `features: list[str]`.
- During inference, Feature Studio profiling, and Production Validation, the runtime slices the dataset columns to match `config.json["features"]` exactly.

---

## 5. Three Feature Populations Across the Registries

```
Population                 Feature Registry            Dataset Registry             Model Registry
------------------------------------------------------------------------------------------------------------------
1. Feature Registry        Canonical Schema & FR ID    Materialized in Master DB    Selected in Model Tab 1
2. Base Pipeline           (Future Promotion State)    Exported in Analysis DB      Selected in Model Tab 2
3. Selected Experimental   PL_0002+ in Pipeline Store  Exported in Analysis DB      Selected in Model Tab 3
```

- **Base Pipeline Status**: Base Pipeline represents already accepted, core pipeline features. (It is not a separate generator family; automated promotion from Experimental &rarr; Base is not yet implemented).
- **Selected Experimental Pipeline**: Candidate features tied to `pipeline_id` and immutable `pipeline_snapshot_id`.

---

## 6. Registry Relationship & Lineage Flow

```
┌────────────────────────────────────────┐
│            FEATURE REGISTRY            │
│  (schema_feature_meta + domains)       │
└───────────────────┬────────────────────┘
                    │
                    │ Scoped by feature_project_id
                    ▼
┌────────────────────────────────────────┐
│            DATASET REGISTRY            │
│  ├── Master Dataset (.db)              │
│  └── Analysis Dataset (.parquet)       │
│      ├── registry_export_features      │
│      ├── base_pipeline_export_features │
│      └── pipeline_provenance (PL_XXXX) │
└───────────────────┬────────────────────┘
                    │
                    │ Slices features subset
                    ▼
┌────────────────────────────────────────┐
│             MODEL REGISTRY             │
│  (models/<model_name>/config.json)     │
│  ├── Inherits feature_project_id       │
│  ├── Inherits pipeline_id & snapshot   │
│  └── Stores selected features list     │
└───────────────────┬────────────────────┘
                    │
                    │ Evaluates unseen days
                    ▼
┌────────────────────────────────────────┐
│         PRODUCTION VALIDATION          │
│  (unseen_<slug>_<hash>.parquet)        │
│  └── Replays exact parent lineage      │
└────────────────────────────────────────┘
```

---

## 7. Metadata Ownership Matrix

| Metadata Property | Feature Registry | Dataset Registry | Model Registry | Source of Truth Authority |
|---|---|---|---|---|
| `feature_id` (`FRXXXX`) | **Owner** | Referenced | Referenced | Feature Registry Store |
| `primary_domain` | **Owner** | Referenced | Referenced | `feature_domains.py` |
| `feature_project_id` | **Owner** | Bound at Build | Inherited | Feature Project Manager |
| `registry_export_features`| Computed | **Owner** | Referenced | Master Dataset Export |
| `base_pipeline_export_features`| None | **Owner** | Referenced | Analysis Dataset Export |
| `pipeline_id` & `snapshot_id`| Referenced | **Owner** | Inherited | Pipeline Registry Store |
| `pipeline_provenance` | Referenced | **Owner** | Inherited | Analysis Dataset JSON |
| `selected_features` (`features`)| None | None | **Owner** | Model `config.json` |
| `hyperparameters` | None | None | **Owner** | Model `training_config.json` |
| `validation_metrics` | None | None | **Owner** | Model `metrics.json` |
| `unseen_identity_hash` | None | Stored on Unseen | Evaluated | Production Validation |

---

## 8. Immutability & Snapshot Rules

1. **Feature Registry is Dynamic**: New canonical features can be registered or retired over time.
2. **Feature Projects are Configurable**: Custom projects can update their membership list.
3. **Pipeline Snapshots are Cryptographically Immutable**: When an Analysis Dataset is exported, `pipeline_snapshot_id` captures a SHA256 hash of the pipeline's configuration. Subsequent changes to `pipeline_registry_store.json` do **not** alter the historical snapshot ID.
4. **Dataset Parquet Files are Immutable**: Once written, dataset matrices are never overwritten; new settings produce new files.
5. **Model Packages are Immutable**: Model weights and `config.json` are frozen upon training completion.

---

## 9. Registry User Interfaces

### 9.1. Feature Registry UI ([`feature_registry_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_registry_panel.py))
- **Reads**: `feature_registry_store.json`, `feature_domains.py`.
- **Actions**: View catalog, inspect feature formulas/dependencies, launch Feature Project Manager.
- **Writes**: Disables/retires custom features.

### 9.2. Dataset Registry UI ([`dataset_metadata_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/dataset_metadata_panel.py))
- **Reads**: `analysis_*.json`, Parquet schemas, SQLite master meta.
- **Actions**: Inspect column counts, feature sources, No-Null drop reports, and pipeline provenance.
- **Writes**: Registers new datasets in SQLite.

### 9.3. Model Registry UI ([`model_registry_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_registry_panel.py))
- **Reads**: `models/<model_name>/` package files (`config.json`, `metrics.json`).
- **Actions**: Inspect out-of-fold metrics, feature importances, training monitor logs, set active model, launch Feature Studio.
- **Writes**: Deletes unprotected models, sets active model in `.active_model.json`.

---

## 10. Current Implementation vs. Future Roadmap

| Capability | Current Implementation Status | Future Roadmap Status |
|---|---|---|
| **Feature Registry Taxonomy** | **IMPLEMENTED** (206 Canonical Features, 11 Domains) | Expansion of higher-order Greek surfaces |
| **Feature Projects** | **IMPLEMENTED** (`feature_project_id`, `"all"`, `"chart"`) | Automated project recommendation |
| **Dataset Lineage Tracking** | **IMPLEMENTED** (`pipeline_snapshot_id`, `pipeline_provenance`) | Cloud dataset synchronization |
| **Model Registry Protection** | **IMPLEMENTED** (`deployed`, `production` protection) | Automated A/B deployment routing |
| **Experimental &rarr; Base Promotion**| **NOT IMPLEMENTED YET** (Manual / Placeholder) | Automated promotion engine based on scoring |
| **Cross-Registry Audit Sync** | **IMPLEMENTED** (Lineage hashes verified on load) | Continuous background integrity daemon |

---

## 11. Complete End-to-End Lineage Example

```
1. Feature Registry:
   • Canonical Feature: atm_iv_ce (FR0012)
   • Feature Project: feature_project_id = "all"

2. Pipeline Registry:
   • Pipeline ID: PL_0005
   • Candidate Feature: spot_roll_mean_60
   • Pipeline Snapshot ID: ca5945f58f8b1a2c

3. Dataset Registry:
   • Analysis Dataset: analysis_nifty_6s_exp005.parquet
   • Metadata: analysis_nifty_6s_exp005.json
   • Feature Matrix: 1,186 columns (Registry ∪ Base ∪ PL_0005)

4. Model Registry:
   • Model Name: Future_LTP_5m_WF_1168f_XGB_2243_14
   • Config: models/Future_LTP_5m_WF_1168f_XGB_2243_14/config.json
   • Selected Features: 583 features
   • Preserved Lineage: feature_project_id="all", pipeline_id="PL_0005", snapshot="ca5945f58f8b1a2c"

5. Production Validation:
   • Unseen Days: ["2026-06-05", "2026-06-06"]
   • Identity Hash: SHA256(master_db + unseen_days + "all" + "PL_0005" + "ca5945f58f8b1a2c")[:8] = "9b1deb4d"
   • Unseen Dataset: unseen_Future_LTP_5m_WF_1168f_XGB_2243_14_9b1deb4d.parquet
```

---

## 12. Source Code Map

| Registry | Source File | Key Class / Functions | Primary Responsibility |
|---|---|---|---|
| **Feature Registry** | [`apps/chain_replay_ml/dataset_builder/feature_registry_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/feature_registry_store.py) | `load_store`, `save_store`, `format_feature_id` | Persistent JSON catalog of canonical feature IDs & statuses |
| **Feature Domains** | [`apps/chain_replay_ml/dataset_builder/feature_domains.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/feature_domains.py) | `primary_domain_of`, `DOMAIN_ORDER` | 11 financial domain taxomony definitions |
| **Feature Projects** | [`apps/chain_replay_ml/dataset_builder/feature_project_organization.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/feature_project_organization.py) | `project_registry_feature_source`, `build_default_all_project_doc` | Project-scoped feature trees and custom group mappings |
| **Master Store** | [`apps/chain_replay_ml/dataset_builder/master_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/master_store.py) | `MasterDatasetStore`, `ensure_master_feature_project_id` | SQLite tick database storage for Feature Registry columns |
| **Analysis Export** | [`apps/chain_replay_ml/dataset_builder/analysis_dataset_export.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_dataset_export.py) | `create_analysis_dataset` | Materializes Analysis Parquet feature matrix & JSON metadata |
| **Unseen Dataset** | [`apps/chain_replay_ml/production_validation/unseen_dataset.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/unseen_dataset.py) | `unseen_dataset_identity_hash`, `resolve_or_create_unseen_dataset` | Replays exact parent model lineage for out-of-sample forward testing |
| **Model Registry** | [`apps/chain_replay_ml/training/registry.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/training/registry.py) | `list_model_packages`, `load_model_package`, `delete_model_package` | Manages trained model packages, metrics, and active model state |
| **Model Config** | [`apps/chain_replay_ml/training/config.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/training/config.py) | `TrainingConfig`, `normalize_training_config` | Validates training hyperparameters and dataset references |

---

## 13. Troubleshooting Common Registry Issues

| Symptom | Cause | Where to Inspect | Resolution |
|---|---|---|---|
| `Feature Project tree shows unexpected features` | The active `feature_project_id` in Build Config does not match expectations. | Check Master Dataset Build Configuration dropdown. | Select the appropriate Feature Project (e.g. `"all"` or `"chart"`). |
| `Pipeline feature missing from Analysis Dataset` | Pipeline was not enabled or feature was excluded in preferences. | Check `analysis_<name>.json` under `pipeline_provenance`. | Ensure `include_pipeline=True` and candidate is in `candidate_features`. |
| `Stale unseen dataset reused during Production Validation` | Model was retrained with different features without updating `pipeline_snapshot_id`. | Check `unseen_dataset_identity_hash` logging. | System automatically recalculates hash; delete corrupted parquet cache if forced. |
| `Model delete blocked error` | Model is currently marked as `"deployed"`, `"production"`, or active. | Check `.active_model.json` in models directory. | Deactivate or undeploy model before deletion. |
| `Selected model feature missing during prediction` | Parquet dataset used for inference was built with a different `feature_project_id`. | Compare `config.json["features"]` with Parquet schema. | Rebuild dataset using the parent model's exact `feature_project_id`. |

---

## 14. Glossary

- **Feature Registry**: Canonical central catalog of 206 domain-specific financial features.
- **Feature Project (`feature_project_id`)**: Organizational grouping scoping a subset of Feature Registry features.
- **Master Dataset**: SQLite database storing raw materialized Feature Registry features for a market.
- **Analysis Dataset**: Materialized Parquet feature matrix containing Feature Registry and Pipeline features.
- **Unseen Dataset**: Out-of-sample Parquet feature matrix used exclusively for live Production Validation.
- **Pipeline Snapshot (`pipeline_snapshot_id`)**: Immutable cryptographic SHA256 content hash of a pipeline definition.
- **Model Registry**: Management catalog indexing trained model packages, weights, configs, and performance metrics.
- **Selected Features**: The exact subset of dataset feature columns chosen for a trained model (`config.json["features"]`).
- **Production Validation**: Forward validation benchmarking Holdout performance against true unseen trading days.
- **Lineage Integrity**: The end-to-end cryptographic link preserving exact feature configurations from Dataset &rarr; Model &rarr; Validation.
