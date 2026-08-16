# AruMLStudio Feature Studio — Technical & Functional Architecture

---

## 1. Purpose

The **Feature Studio** is the dedicated analytical suite within **AruMLStudio** for inspecting, profiling, diagnosing, and validating model features across their lifecycle. 

Machine learning models for options trading combine features from multiple distinct generative populations (canonical domain features, deterministic mathematical transformations, and experimental statistical pipelines). Feature Studio provides visibility into:
- Feature importance rankings (Native tree metrics, Permutation degradation, SHAP attribution).
- Univariate distribution properties and data anomalies (null percentage, skewness, percentile profiles).
- Temporal feature drift between training (Walk-Forward) and out-of-sample (Holdout) regimes.
- Multi-model delta comparisons across paired training architectures.
- Automated root-cause diagnostics (identifying overfitting, severe drift, volatility regimes, or distribution shifts).
- Production validation over unseen forward trading days to generate auditable feature lifecycle recommendations (**KEEP**, **WATCH**, **REMOVE**).

---

## 2. Scope

This document provides end-to-end technical and functional documentation for:
- **`FeatureStudioPanel`** (`apps/master_dataset_tk/feature_studio_panel.py`) and its 7 viewer tabs.
- **Compute and artifact pipelines** in `apps/chain_replay_ml/`:
  - Feature Importance (`chain_replay_ml.feature_importance_studio`)
  - Feature Distribution (`chain_replay_ml.feature_distribution_studio`)
  - Feature Drift (`chain_replay_ml.feature_drift_studio`)
  - Studio Compare / Multi-Model (`chain_replay_ml.multi_model_studio`)
  - Diagnostics Studio (`chain_replay_ml.diagnostics_studio`)
  - Production Validation (`chain_replay_ml.production_validation`)
  - Experiment Planner / Recommendation Engine (`chain_replay_ml.recommendation_engine`)
- **Three-Population Feature Classification Engine** (`chain_replay_ml.dataset_builder.feature_sources_catalog`).
- **Feature Project Isolation & Lineage** (`chain_replay_ml.dataset_builder.feature_project_organization`, `master_feature_project`).

---

## 3. Architecture Overview

Feature Studio follows a decoupled **Compute &rarr; Persist &rarr; Load &rarr; Populate** execution model. UI panels act as stateless visualizers over on-disk JSON/Parquet artifacts stored directly inside each trained model's package directory (`models/<model_name>/`).

```
                              ┌────────────────────────────────────────────────────────┐
                              │                 FeatureStudioPanel                     │
                              │   (Shared Toolbar: Model Selector, Filter, Top-N)      │
                              └──────────────────────────┬─────────────────────────────┘
                                                         │
                        ┌────────────────────────────────┴─────────────────────────────┐
                        ▼                                                              ▼
               [Load Artifacts]                                                    [Compute]
                        │                                                              │
         run_load_pipeline()                                                run_compute_pipeline()
                        │                                                              │
   Reads on-disk artifacts sequentially                             Runs compute sequentially:
   (Importance → Dist → Drift → Diag → Planner)                     (Importance → Dist → Drift → Diag → Planner)
                        │                                                              │
                        ▼                                                              ▼
   ┌───────────────────────────────────────────────┐                  ┌─────────────────────────────────────────┐
   │ Populates UI Tab Payloads                     │                  │ Writes JSON artifacts to model package  │
   │ (importance, distribution, drift, diagnostics)│                  │ models/<model_name>/<studio_dir>/       │
   └───────────────────────────────────────────────┘                  └─────────────────────────────────────────┘
```

### Core Design Invariants:
1. **Controller-Owned Pipeline**: `FeatureStudioPanel` manages model selection, shared filtering, and background compute threading. Tabs do not trigger ad-hoc recomputations upon tab-switching.
2. **Deterministic Artifact Caching**: Loaded artifacts are cached per model in `_cache: PipelineResult`. Switching tabs reuses cached in-memory structures without disk re-reads.
3. **Strict Separation of Data Regimes**:
   - **Diagnostics Studio** strictly evaluates **Walk-Forward (WF) Training Data vs. Holdout Data**.
   - **Production Validation** strictly evaluates **Holdout Data vs. Unseen Forward Days**.

---

## 4. Feature Types

The current architecture partitions all model features into **three disjoint, exhaustive feature populations**:

```
                       ┌───────────────────────────────────────────────┐
                       │          Model Selected Features              │
                       │               (N features)                    │
                       └───────────────────────┬───────────────────────┘
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             ▼                                 ▼                                 ▼
   ┌───────────────────┐             ┌───────────────────┐             ┌───────────────────┐
   │ Feature Registry  │             │   Base Pipeline   │             │Selected Experim.  │
   │ (Canonical 206)   │             │ (Deterministic)   │             │   (Pipeline)      │
   └─────────┬─────────┘             └─────────┬─────────┘             └─────────┬─────────┘
             │                                 │                                 │
   • Domain-based                    • Mathematical ops                • Dynamic discovery
   • Filtered by Project             • lag, diff, ret, interact        • pipeline_id & snapshot
   • Stored in Master DB             • Derived in Analysis DB          • Derived in Analysis DB
```

### 4.1. Population 1: Feature Registry Features
- **Source of Truth**: Canonical Feature Registry (`chain_replay_ml.dataset_builder.feature_registry_store` and `feature_domains.py`). 206 active canonical features across 11 core financial domains (Price & Premium, Spot & Futures, IV, Greeks, Straddle, OI, Volume, PCR, Ratios, Spreads, Microstructure).
- **Project Scoping**: Filtered and organized by `feature_project_id` (e.g. `"all"`, `"chart"`).
- **Materialization**: Pre-computed and stored directly inside the SQLite **Master Dataset** (`master_dataset_*.db`).
- **Metadata Fields**:
  - `feature_project_id`: Project identifier.
  - `registry_export_features`: Array of selected registry feature names.
  - `registry_export_count`: Total count of materialized registry features.

### 4.2. Population 2: Base Pipeline Features
- **Source of Truth**: Fixed, deterministic mathematical feature generators configured in `chain_replay_ml/pipeline.py` and `pipeline_templates.py`.
- **Generation Families**:
  - `lag`: Historical time-lagged values (e.g., `spot_lag1`, `atm_iv_ce_lag5`).
  - `diff`: Absolute first differences (e.g., `spot_diff1`, `futures_basis_diff5`).
  - `ret` / `pct_change`: Percentage returns (e.g., `spot_ret5`, `atm_straddle_ret1`).
  - `interact` / `ratio`: Product or quotient interactions (e.g., `spot_x_atm_iv_ce`).
- **Materialization**: Computed on-the-fly when building an **Analysis Dataset** (`create_analysis_dataset(..., include_pipeline=True)`). Never written to the Master Dataset SQLite table.
- **Metadata Fields**:
  - `base_pipeline_export_features`: Array of exported base pipeline feature names.
  - `base_pipeline_export_count`: Count of generated base pipeline features.

### 4.3. Population 3: Selected Experimental Pipeline Features
- **Source of Truth**: Experimental Pipeline Registry (`pipeline_id`, e.g., `PL_0005`), immutable snapshot (`pipeline_snapshot_id`, e.g., `ca5945f58f8`), and candidate feature catalog (`pipeline_provenance["candidate_features"]`).
- **Generation**: Created during Pipeline Exploration via advanced statistical, volatility surface, or cross-strike transformations.
- **Selection**: Down-selected during **Create Model (Section 5.3)** from candidate features generated by the experimental pipeline.
- **Metadata Fields**:
  - `pipeline_id`: Unique pipeline identifier.
  - `pipeline_name`: Display label of the pipeline.
  - `pipeline_type`: Transformation engine architecture.
  - `pipeline_snapshot_id`: Immutable content hash of the pipeline definition.
  - `pipeline_provenance`: Dictionary tracking `candidate_features`, `input_features`, and transform parameters.
  - `experimental_pipeline_export_count`: Number of experimental features in the model.

### 4.4. Classification Engine
The engine in [`chain_replay_ml.dataset_builder.feature_sources_catalog`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/feature_sources_catalog.py) classifies any feature name deterministically using the following precedence:
1. If present in `registry_export_features` or canonical registry &rarr; **`DATASET_SOURCE_FEATURE_REGISTRY`**.
2. If present in `base_pipeline_export_features` or matches base generator templates &rarr; **`DATASET_SOURCE_BASE_PIPELINE`**.
3. If present in experimental pipeline candidate list or has pipeline prefix &rarr; **`DATASET_SOURCE_OTHER_PIPELINE`**.

---

## 5. Feature Lifecycle

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. DEFINITION & ORGANIZATION                                                           │
│    Feature Projects organize canonical registry features into custom hierarchies.      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. MATERIALIZATION (Master Dataset Builder)                                            │
│    Materializes Registry features selected by feature_project_id into master_*.db.     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. TRANSFORMATION & ENRICHMENT (Analysis Dataset Engine)                               │
│    Applies Base Pipeline + Experimental Pipeline generators to create training matrix. │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. MODEL TRAINING (Model Builder)                                                      │
│    Trains XGBoost/Classifier on Walk-Forward splits and validates on Holdout slice.    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. FEATURE STUDIO ANALYSIS                                                             │
│    Computes Native/Permutation/SHAP importance, univariate distributions, and drift.  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 6. AUTOMATED DIAGNOSTICS                                                               │
│    Synthesizes Holdout vs. WF drift to detect overfitting and assign risk scores.      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 7. PRODUCTION VALIDATION & RECOMMENDATIONS                                             │
│    Builds identical lineage unseen_* dataset, validates against Holdout, records       │
│    KEEP / WATCH / REMOVE recommendations into feature_recommendation_history.json.     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Feature Project Integration

`feature_project_id` represents the organizational boundary for Feature Registry features:
1. **Canonical Source**: Managed via Feature Project Manager (`apps/master_dataset_tk/feature_project_manager_panel.py`). Projects define a subset of canonical registry features and custom display groups.
2. **Master Dataset Builder**:
   - `BuildConfigPanel` features a **Feature Project** dropdown.
   - `FeatureSelectionPicker` dynamically loads `project_registry_feature_source(data_dir, feature_project_id)`.
   - Materialized SQLite database records `feature_project_id` and `registry_export_features` in `master_config` and `master_dataset_meta_json`.
3. **Analysis & Unseen Datasets**:
   - Analysis dataset builder extracts `feature_project_id` from master database and propagates it into dataset metadata.
   - Unseen dataset generation validates that the target `feature_project_id` matches the parent training dataset exactly before accepting dataset reuse.

---

## 7. Feature Studio Tabs

### 7.1. Feature Importance Studio
- **UI Class**: `FeatureImportanceStudioPanel` ([`apps/master_dataset_tk/feature_importance_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_importance_studio_panel.py))
- **Compute Package**: `chain_replay_ml.feature_importance_studio` ([`apps/chain_replay_ml/feature_importance_studio/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/feature_importance_studio/compute.py))
- **Purpose**: Computes multi-method feature importance to identify which features drive predictive power and detect spurious signals.
- **Inputs**: Model binary (`model.xgb` / `model.json`), training config (`config.json`), Holdout slice ($X_{\text{ho}}, y_{\text{ho}}$).
- **Calculations**:
  1. **Native XGBoost Importance**: Extracts `weight`, `gain`, `cover`, `total_gain`, and `total_cover` via booster `get_score()`.
  2. **Permutation Importance**: Evaluates score degradation on Holdout slice across $N=3$ repeats (RMSE metric for regression, log-loss for classification).
  3. **Tree SHAP**: Calculates mean absolute SHAP values across a representative holdout sample ($N=400$) using `shap.TreeExplainer`.
  4. **Consensus Ranking**: Calculates rank deltas between Native Gain, Permutation, and SHAP.
- **Outputs / Artifacts**:
  - `models/<model>/feature_importance/native.json`
  - `models/<model>/feature_importance/permutation.json`
  - `models/<model>/feature_importance/shap.json`
  - `models/<model>/feature_importance/comparison.json`
  - `models/<model>/feature_importance/meta.json`

---

### 7.2. Feature Distribution Studio
- **UI Class**: `FeatureDistributionStudioPanel` ([`apps/master_dataset_tk/feature_distribution_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_distribution_studio_panel.py))
- **Compute Package**: `chain_replay_ml.feature_distribution_studio` ([`apps/chain_replay_ml/feature_distribution_studio/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/feature_distribution_studio/compute.py))
- **Purpose**: Profiles holdout feature column statistics to catch data anomalies, extreme values, and high missingness.
- **Inputs**: Holdout feature matrix ($X_{\text{ho}}$) up to 20,000 rows, joined with Importance comparison rows.
- **Calculations**:
  - Univariate metrics: `count`, `n_finite`, `null_count`, `null_pct`, `n_unique`.
  - Moments: `mean`, `std`, `skew`.
  - Percentiles: `min`, `p1`, `p5`, `p25`, `p50`, `p75`, `p95`, `p99`, `max`.
  - Accelerated execution via Polars (`feature_distribution_rows_via_polars`) with automatic fallback to Pandas.
- **Outputs / Artifacts**:
  - `models/<model>/feature_distribution/holdout.json`
  - `models/<model>/feature_distribution/comparison.json`
  - `models/<model>/feature_distribution/meta.json`

---

### 7.3. Feature Drift Studio
- **UI Class**: `FeatureDriftStudioPanel` ([`apps/master_dataset_tk/feature_drift_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_drift_studio_panel.py))
- **Compute Package**: `chain_replay_ml.feature_drift_studio` ([`apps/chain_replay_ml/feature_drift_studio/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/feature_drift_studio/compute.py))
- **Purpose**: Measures statistical divergence between the training regime (Walk-Forward) and validation regime (Holdout).
- **Inputs**: Walk-Forward slice ($X_{\text{wf}}$ up to 50,000 rows) and Holdout slice ($X_{\text{ho}}$ up to 20,000 rows).
- **Calculations & Formulas**:
  1. **Normalized Mean Shift**:
     $$\text{shift} = \frac{\mu_{\text{ho}} - \mu_{\text{wf}}}{\sigma_{\text{pooled}}}, \quad \sigma_{\text{pooled}} = \sqrt{\frac{\sigma_{\text{wf}}^2 + \sigma_{\text{ho}}^2}{2}}$$
  2. **Feature Drift Score ($0.0 \dots 1.0$)**:
     $$\text{drift} = \text{mean}\left( \text{clamp}_{[0,1]}\left(\frac{|\text{shift}|}{1.5}\right), \, \text{clamp}_{[0,1]}\left(\frac{|\sigma_{\text{ho}}/\sigma_{\text{wf}} - 1.0|}{0.5}\right) \right)$$
     > [!NOTE]
     > **Drift = 1.0000** indicates complete saturation where the mean shifted by $\ge 1.5$ pooled standard deviations and standard deviation changed by $\ge 50\%$.
  3. **Distribution Shape & Missingness**:
     - Two-sample Kolmogorov-Smirnov test (`ks_statistic`, `ks_pvalue`).
     - 1D Wasserstein distance (`wasserstein_distance`, `wasserstein_normalized`).
     - Missingness divergence: $\text{null\_drift\_pp} = \text{null\%}_{\text{ho}} - \text{null\%}_{\text{wf}}$.
  4. **Composite Risk Score ($0 \dots 100$)**:
     Combines mean drift, KS statistic, Wasserstein metric, null shift, and feature importance weight into four risk buckets: `low` ($<30$), `medium` ($30\dots 55$), `high` ($55\dots 75$), `critical` ($\ge 75$).
- **Outputs / Artifacts**:
  - `models/<model>/feature_drift/ranking.json`
  - `models/<model>/feature_drift/comparison.json`
  - `models/<model>/feature_drift/meta.json`

---

### 7.4. Studio Compare
- **UI Class**: `MultiModelStudioPanel` ([`apps/master_dataset_tk/multi_model_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/multi_model_studio_panel.py))
- **Compute Package**: `chain_replay_ml.multi_model_studio` ([`apps/chain_replay_ml/multi_model_studio/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/multi_model_studio/compute.py))
- **Purpose**: Performs side-by-side artifact delta joins between two trained models (Model A vs. Model B) without recomputation.
- **Inputs**: Pre-computed Importance, Distribution, and Drift artifacts for Model A and Model B.
- **Outputs**:
  - Delta metrics: `in_a`, `in_b`, `rank_gain_a`, `rank_gain_b`, $\Delta\text{Rank}$, `risk_a`, `risk_b`, $\Delta\text{Risk Score}$, `drift_a`, `drift_b`, $\text{null\%}_a$, $\text{null\%}_b$.
  - Saved to `models/_pairs/<model_a>__vs__<model_b>/feature_studio_compare/comparison.json`.

---

### 7.5. Diagnostics Studio
- **UI Class**: `DiagnosticsStudioPanel` ([`apps/master_dataset_tk/diagnostics_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/diagnostics_studio_panel.py))
- **Compute Package**: `chain_replay_ml.diagnostics_studio` ([`apps/chain_replay_ml/diagnostics_studio/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/diagnostics_studio/compute.py))
- **Purpose**: Automated root-cause performance diagnosis across Holdout vs. Walk-Forward data splits.
- **Headline Diagnostics**:
  - **Primary Cause**: `Overfitting`, `Severe Drift`, `Feature Shift`, `Target Shift`, `Premium Shift`, `Volatility Shift`, or `Stable/Good`.
  - **Confidence**: Diagnostic confidence percentage ($0\dots 100\%$).
  - **MAE $\Delta$**: Out-of-sample error degradation percentage.
  - **Similarity Score**: Regime similarity metric ($100 - \text{composite drift}$).
  - **Overall Feature Drift**: Average drift across Top-10 features.
- **Three-Source Partitioning**:
  Features are partitioned into 3 isolated sub-tabs using `partition_diagnostic_rows`:
  1. `Feature Registry`: Features belonging to `feature_project_id` & `registry_export_features`.
  2. `Base Pipeline`: Features matching base pipeline generators (`lag`, `diff`, `ret`, `interact`).
  3. `Selected Experimental`: Features from experimental pipeline snapshots.
- **Action Invariants**:
  $$\text{registry\_count} + \text{base\_pipeline\_count} + \text{experimental\_count} \equiv \text{total\_model\_features}$$
  Every row displays recommended actions: `KEEP`, `WATCH`, or `REMOVE`.

---

### 7.6. Production Validation
- **UI Class**: `ProductionValidationPanel` ([`apps/master_dataset_tk/production_validation_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/production_validation_panel.py))
- **Compute Package**: `chain_replay_ml.production_validation` ([`apps/chain_replay_ml/production_validation/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/compute.py))
- **Purpose**: Validates the model against forward out-of-sample **unseen trading days** (days in Master Dataset not present in training).
- **Lineage-Preserving Unseen Dataset Resolution**:
  - Extracts `feature_project_id`, `pipeline_id`, `pipeline_snapshot_id`, `include_registry`, and `include_pipeline` from the model's parent dataset metadata.
  - Generates or reuses an `unseen_*` analysis dataset with exact matching lineage hash (`unseen_dataset_identity_hash`).
  - Verifies that 100% of the model's selected features exist in the unseen matrix.
- **Feature Validation Radio Filter**:
  Allows filtering the Holdout vs. Unseen comparison table across the three feature sources:
  1. `Selected Experimental` (Default)
  2. `Base Pipeline`
  3. `Feature Registry`
- **Output Artifacts**:
  - `models/<model>/production_validation/unseen_metrics.json`
  - `models/<model>/production_validation/feature_comparison.json`
  - `data/feature_recommendation_history.json` (Cumulative history store)

---

## 8. Diagnostics Architecture

Diagnostics Studio synthesizes signals across metrics, importance, distribution, and drift:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DIAGNOSTICS ENGINE                              │
├────────────────────────┬───────────────────────────────────────────────┤
│ Metric Inputs          │ WF MAE, Holdout MAE, RMSE, Premium RMSE       │
│ Distribution Inputs    │ WF vs. Holdout Target / Volatility Moments    │
│ Drift Inputs           │ Feature Drift Scores, KS stats, Wasserstein   │
│ Importance Inputs      │ Native Gain, Permutation, SHAP consensus      │
├────────────────────────┴───────────────────────────────────────────────┤
│                           HEURISTIC RULES                              │
├────────────────────────┬───────────────────────────────────────────────┤
│ Overfitting            │ Holdout MAE > WF MAE + 15% AND                │
│                        │ Target/Vol Drift < 25% AND Feature Drift Low  │
├────────────────────────┼───────────────────────────────────────────────┤
│ Severe Drift           │ Overall Feature Drift > 40% OR                │
│                        │ Similarity Score < 60%                        │
├────────────────────────┼───────────────────────────────────────────────┤
│ Target / Vol Regime    │ Target Drift > 30% OR Volatility Drift > 30%  │
└────────────────────────┴───────────────────────────────────────────────┘
```

---

## 9. Production Validation Architecture

Production Validation runs an un-biased forward evaluation:

```
   Selected Model
         │
         ▼
   Parent Dataset Metadata ────────► Extract: feature_project_id, pipeline_id,
         │                                    pipeline_snapshot_id, include_pipeline
         ▼
   Identify Unseen Trading Days (Master Days \ Training Days)
         │
         ▼
   Compute / Resolve Unseen Dataset (unseen_<slug>_<hash>)
   • Exact lineage match: feature_project_id + pipeline_snapshot_id
   • Materializes Feature Registry + Base Pipeline + Experimental Pipeline
         │
         ▼
   Evaluate Model on Unseen Days
   • Production MAE, RMSE, Directional Accuracy, PnL simulation
         │
         ▼
   Holdout vs. Unseen Feature Comparison
   • Drift score on unseen regime
   • Null percentage on unseen regime
   • Action assignment: KEEP / WATCH / REMOVE
```

---

## 10. Feature Validation

Inside Production Validation, the **Feature Validation** section evaluates individual feature behavior in production:
1. **Radio Selectors**:
   - `Selected Experimental` (Default): Focuses validation on new candidate features generated by experimental pipelines.
   - `Base Pipeline`: Evaluates deterministic mathematical transforms.
   - `Feature Registry`: Evaluates canonical registry domain features.
2. **Metrics Displayed**:
   - Holdout Importance Rank & Weight.
   - Holdout Mean & Std vs. Unseen Mean & Std.
   - Unseen Drift Score & Unseen Null %.
   - Production Recommendation (`KEEP`, `WATCH`, `REMOVE`).

---

## 11. KEEP / WATCH / REMOVE Rules

| Recommendation | Criteria | Operational Meaning |
|---|---|---|
| **`KEEP`** | • Drift $< 0.30$<br>• Null $\% < 1.0\%$<br>• Importance Rank stable | Feature demonstrates high generalization and stability in production. Retained in future models. |
| **`WATCH`** | • Drift $0.30 \dots 0.65$<br>• OR Null $\% 1.0\% \dots 5.0\%$<br>• OR Moderate rank loss | Feature exhibits mild instability or regime sensitivity. Flagged for monitoring. |
| **`REMOVE`** | • Drift $\ge 0.65$<br>• OR Null $\% \ge 5.0\%$<br>• OR Permutation rank $\le 0$ | Feature suffers from severe distribution shift or spurious correlation. Excluded from future model runs. |

### Recommendation Persistence:
- Clicking **"Update Registry Recommendations"** appends records to [`data/feature_recommendation_history.json`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/data/feature_recommendation_history.json).
- **Critical Invariant**: Recommendations **never delete or alter canonical feature registry definitions or pipeline transformations**. They serve as advisory metadata for Model Builder and Feature Project filtering.

---

## 12. Model Builder Integration

Feature Studio directly feeds into **Create Model (Section 5)**:
1. **Registry Tab**: Populated by the selected dataset's `feature_project_id`. Users can exclude features flagged as `REMOVE`.
2. **Base Pipeline Tab**: Populated by `base_pipeline_export_features`. Users enable/disable standard generator families.
3. **Selected Experimental Tab**: Populated by `pipeline_provenance["candidate_features"]` for the active `pipeline_id` and `pipeline_snapshot_id`.
4. **Final Model Feature Vector**:
   $$\mathcal{F}_{\text{model}} = \mathcal{F}_{\text{registry}} \cup \mathcal{F}_{\text{base\_pipeline}} \cup \mathcal{F}_{\text{experimental}}$$

---

## 13. Storage & Artifacts Map

| Artifact | Typical Path | Format | Producer | Consumer |
|---|---|---|---|---|
| **Native Importance** | `models/<model>/feature_importance/native.json` | JSON | `feature_importance_studio` | Feature Studio, Diagnostics |
| **Permutation Importance**| `models/<model>/feature_importance/permutation.json` | JSON | `feature_importance_studio` | Feature Studio, Diagnostics |
| **SHAP Importance** | `models/<model>/feature_importance/shap.json` | JSON | `feature_importance_studio` | Feature Studio, Diagnostics |
| **Importance Comparison** | `models/<model>/feature_importance/comparison.json` | JSON | `feature_importance_studio` | All Studios, Compare |
| **Distribution Stats** | `models/<model>/feature_distribution/holdout.json` | JSON | `feature_distribution_studio`| Distribution Tab, Planner |
| **Drift Ranking** | `models/<model>/feature_drift/ranking.json` | JSON | `feature_drift_studio` | Drift Tab, Diagnostics |
| **Drift Comparison** | `models/<model>/feature_drift/comparison.json` | JSON | `feature_drift_studio` | Diagnostics, Compare |
| **Pair Comparison** | `models/_pairs/<A>__vs__<B>/feature_studio_compare/comparison.json` | JSON | `multi_model_studio` | Studio Compare Tab |
| **Diagnostics Meta** | `models/<model>/diagnostics_studio/meta.json` | JSON | `diagnostics_studio` | Diagnostics Tab |
| **Diagnostics Summary** | `models/<model>/diagnostics_studio/summary.json` | JSON | `diagnostics_studio` | Diagnostics Tab |
| **Unseen Dataset** | `datasets/analysis_datasets/unseen_<slug>_<hash>.parquet` | Parquet | `production_validation` | Validation Engine |
| **Unseen Metadata** | `datasets/analysis_datasets/unseen_<slug>_<hash>.json` | JSON | `production_validation` | Lineage Verifier |
| **Production Metrics** | `models/<model>/production_validation/unseen_metrics.json` | JSON | `production_validation` | Prod Validation Tab |
| **Recommendation History**| `<chart_data_dir>/feature_recommendation_history.json` | JSON | `production_validation` | Feature Project / Model Builder |
| **Feature Projects** | `<chart_data_dir>/feature_registry_projects.json` | JSON | `feature_project_manager`| Master Builder, Model Builder |

---

## 14. Source Code Map

| Component | Source File | Key Class / Functions | Reads | Writes |
|---|---|---|---|---|
| **Studio Shell** | [`apps/master_dataset_tk/feature_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_studio_panel.py) | `FeatureStudioPanel` | Model registry, UI prefs | UI cache |
| **Pipeline Controller** | [`apps/master_dataset_tk/feature_studio_pipeline.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_studio_pipeline.py) | `run_compute_pipeline`, `run_load_pipeline` | Model packages | Studio artifacts |
| **Importance Compute** | [`apps/chain_replay_ml/feature_importance_studio/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/feature_importance_studio/compute.py) | `run_compute`, `_load_holdout_xy` | Model binary, Holdout matrix | `feature_importance/*.json` |
| **Distribution Compute**| [`apps/chain_replay_ml/feature_distribution_studio/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/feature_distribution_studio/compute.py)| `run_compute`, `compute_holdout_stats`| Holdout matrix | `feature_distribution/*.json` |
| **Drift Compute** | [`apps/chain_replay_ml/feature_drift_studio/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/feature_drift_studio/compute.py) | `run_compute`, `load_wf_holdout_xy` | WF & Holdout matrices | `feature_drift/*.json` |
| **Multi-Model Compare** | [`apps/chain_replay_ml/multi_model_studio/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/multi_model_studio/compute.py) | `run_compute`, `build_comparison_rows` | Model A & B artifacts | `models/_pairs/.../comparison.json` |
| **Diagnostics Engine** | [`apps/chain_replay_ml/diagnostics_studio/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/diagnostics_studio/compute.py) | `run_compute`, `summarize_diagnostics` | Importance, Drift, Metrics | `diagnostics_studio/*.json` |
| **Feature Partition** | [`apps/chain_replay_ml/diagnostics_studio/feature_partition.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/diagnostics_studio/feature_partition.py) | `partition_diagnostic_rows` | Dataset metadata, Feature Catalog | Partitioned subsets |
| **Unseen Dataset Builder**| [`apps/chain_replay_ml/production_validation/unseen_dataset.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/unseen_dataset.py) | `resolve_or_create_unseen_dataset` | Master DB, Parent Dataset meta | `unseen_*.parquet` |
| **Validation Compute** | [`apps/chain_replay_ml/production_validation/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/compute.py) | `run_production_validation` | Unseen matrix, Model binary | `production_validation/*.json` |
| **Recommendation Store**| [`apps/chain_replay_ml/production_validation/recommendation_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/recommendation_store.py)| `update_registry_recommendations` | UI recommendation selections | `feature_recommendation_history.json` |

---

## 15. End-to-End Data Flow

```
                      ┌────────────────────────┐
                      │ Feature Project Doc    │
                      │ (e.g. project_id='all')│
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │ Master Dataset Store   │
                      │ (master_nifty_6s.db)   │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │ Analysis Dataset       │
                      │ (Registry + Base + Exp)│
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │ Model Training (XGB)   │
                      │ • Walk-Forward (Train) │
                      │ • Holdout Slice (Eval) │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │ Model Package          │
                      │ models/<model_name>/   │
                      └───────────┬────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│Importance Studio │    │Distribution St.  │    │  Drift Studio    │
│• Native Gain     │    │• Moments (μ, σ)  │    │• Normalized Shift│
│• Permutation     │    │• Percentiles     │    │• KS / Wasserstein│
│• SHAP Holdout    │    │• Null counts     │    │• Composite Risk  │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  Diagnostics Studio    │
                      │  (Holdout vs. WF Split)│
                      │  • Overfitting / Drift │
                      │  • 3-Source Sub-Tabs   │
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │ Production Validation  │
                      │ (Holdout vs. Unseen)   │
                      │ • unseen_* dataset     │
                      │ • 3-Source Radios      │
                      │ • KEEP / WATCH / REMOVE│
                      └───────────┬────────────┘
                                  │
                                  ▼
                      ┌────────────────────────┐
                      │ Recommendation History │
                      │ (JSON Audit Store)     │
                      └────────────────────────┘
```

---

## 16. Architectural Rules

1. **Master Dataset Isolation**: Master Dataset contains only materialized Feature Registry features. Base and Experimental Pipeline features are never materialized into the Master Dataset SQLite store.
2. **Deterministic Classification**: Every model feature must belong to exactly one source category (`Feature Registry`, `Base Pipeline`, or `Selected Experimental`). No feature may appear in multiple tabs or be omitted.
3. **Lineage Preservation in Validation**: `unseen_*` datasets must inherit the exact `feature_project_id`, `pipeline_id`, and `pipeline_snapshot_id` of the parent training dataset.
4. **No Ad-Hoc Recomputation**: Tab switches inside Feature Studio must use cached in-memory structures or reload existing on-disk JSON artifacts. Recomputations occur only via explicit user action ("Compute").
5. **Non-Destructive Recommendations**: Production Validation recommendations (`KEEP`, `WATCH`, `REMOVE`) write to an audit history log and must never mutate or delete canonical Feature Registry definitions or pipeline configurations.
6. **Data Split Boundaries**:
   - Diagnostics Studio evaluates **Walk-Forward vs. Holdout**.
   - Production Validation evaluates **Holdout vs. Unseen Forward Days**.

---

## 17. Known Limitations

### Current Limitations:
- **Studio Compare Source Flattening**: `MultiModelStudioPanel` displays a flat comparison between Model A and Model B without sub-partitioning into the 3 feature sources.
- **Tree SHAP Sample Size**: SHAP importance uses a background sample size of $N=400$ holdout rows to balance latency with fidelity, which provides sample-approximate attribution rather than exhaustive exact Shapley values.

### Intentional Design:
- **Sequential Pipeline Execution**: Features compute sequentially (`Importance` &rarr; `Distribution` &rarr; `Drift` &rarr; `Diagnostics` &rarr; `Planner`) because downstream diagnostics and planner engines require upstream artifact JSON files on disk.
- **Unseen Dataset Caching**: Unseen datasets are indexed by an 8-character identity hash of their data slice and pipeline snapshot, preventing redundant feature generation across validation runs.

---

## 18. Troubleshooting

| Symptom | Probable Cause | Resolution |
|---|---|---|
| `"Select an experimental pipeline when Pipeline Features is enabled"` | Parent dataset metadata lacked `pipeline_id` propagation during unseen dataset resolution. | Resolved in `unseen_dataset.py` by propagating `pipeline_id` and `pipeline_snapshot_id` from `parent_meta` / `train_cfg`. |
| `"Feature 'XYZ' has no primary domain assignment"` | Custom feature name used in project setup is missing from canonical domain mappings in `feature_domains.py`. | Use valid canonical names (e.g., `futures_ltp`, `atm_iv_ce`, `spot_ema20`). |
| `Drift reports 1.0000 (100%) for all features` | Mean shift exceeded $1.5\sigma$ and std shifted by $\ge 50\%$, saturating the normalization clamp. | Inspect data scaling or verify whether market regime (high volatility vs. consolidation) shifted dramatically between WF and Holdout. |
| `No module named 'tkinter'` | PyCharm executed system Python (`C:\python\python.exe`) rather than project virtualenv. | Configure PyCharm SDK to `C:\Users\admin\PycharmProjects\AruMLStudio\.venv\Scripts\python.exe`. |

---

## 19. Glossary

- **Walk-Forward (WF)**: The sequential training and fold-validation slice of the dataset used during model fitting.
- **Holdout**: The out-of-sample time slice set aside during model training to assess validation performance and initial drift.
- **Unseen Days**: Complete trading days present in the Master Dataset that occurred strictly outside the model's training and holdout intervals.
- **`feature_project_id`**: Identifier of the Feature Registry project document organizing canonical registry features.
- **`pipeline_snapshot_id`**: Cryptographic content hash of an experimental pipeline configuration guaranteeing immutable feature generation.
- **`registry_export_features`**: The subset of canonical Feature Registry features selected for dataset export.
- **`base_pipeline_export_features`**: Mathematical transformations (lags, differences, returns, interactions) generated by the base pipeline.
- **Consensus Rank**: Median/average rank of a feature across Native XGBoost, Permutation, and SHAP importance metrics.
