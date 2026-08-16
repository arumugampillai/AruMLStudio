# AruMLStudio Feature Recommendation Lifecycle — Current Implementation

---

## 1. Executive Summary & Purpose

This document provides the authoritative technical reference for the **Feature Recommendation Lifecycle** in **AruMLStudio**.

It details how **Feature Studio** and **Production Validation** evaluate trained models against out-of-sample unseen market data, synthesize feature validation metrics, generate multi-signal evidence recommendations (**KEEP**, **WATCH**, **REMOVE**), and accumulate persistent recommendation history in `feature_recommendation_history.json`.

> [!IMPORTANT]
> **Part A — Current Implementation**: This document strictly describes the **current codebase**. In the current architecture, recommendations are **observational evidence and audit history**. There is **no automatic feature deletion, no automatic retirement, and no automated promotion mechanism**.

---

## 2. Feature Studio Architecture

The **Feature Studio** ([`apps/master_dataset_tk/feature_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_studio_panel.py)) is the central analytics and diagnostic suite for trained machine learning models.

```
                                  FEATURE STUDIO
                             (FeatureStudioPanel)
                                       │
     ┌─────────────┬─────────────┬─────┴───────┬─────────────┬─────────────┬─────────────┐
     ▼             ▼             ▼             ▼             ▼             ▼             ▼
[Importance] [Distribution]   [Drift]   [Studio Compare] [Diagnostics]  [Planner]  [Production]
 (Native/SHAP) (Moments)   (WF vs. HO)  (Model Deltas)  (Root Cause)   (Audit)     [Validation]
```

### 2.1. Feature Studio Responsibilities
1. **Decoupled Artifact Storage**: Feature Studio visualizes on-disk artifacts computed for each model package (`models/<model_name>/`).
2. **Analytical Dimensions**:
   - **Feature Importance**: Native tree split/gain metrics, Holdout Permutation Importance, and Holdout Tree SHAP attributions.
   - **Feature Distribution**: Univariate statistical profiles, missingness, and percentile distributions on Holdout data.
   - **Feature Drift**: Distribution stability benchmarks comparing Walk-Forward (training) data against Holdout data (Normalized Mean Shift, Kolmogorov-Smirnov test, Wasserstein distance).
   - **Studio Compare**: Pairwise delta comparisons between two trained models.
   - **Diagnostics Studio**: Root-cause rule engine diagnosing severe drift, volatility shifts, or over-reliance on unstable features on Holdout vs. Walk-Forward.
   - **Production Validation**: Rigorous forward testing against **true unseen trading days**.

---

## 3. Production Validation & Unseen Data Architecture

**Production Validation** ([`apps/chain_replay_ml/production_validation/`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/)) evaluates how a trained model's selected feature set behaves on forward trading days never encountered during training or holdout testing.

```
                      DATASET REGIMES COMPARISON
                      
  Walk-Forward (WF) Training Days       Holdout Days          True Unseen Days
  ┌─────────────────────────────┐   ┌─────────────────┐   ┌───────────────────────┐
  │      Training & Tuning      │   │ Hyperparameter  │   │ Forward Out-of-Sample │
  │        (In-Sample)          │   │  Benchmarking   │   │  Validation Regime    │
  └─────────────────────────────┘   └─────────────────┘   └───────────────────────┘
  ◄──────────── Diagnostics Studio ──────────►
                                    ◄───────── Production Validation ────────►
```

### 3.1. Holdout Validation vs. True Unseen Validation
- **Diagnostics Studio**: Benchmarks **Walk-Forward Training Days vs. Holdout Days**. Its purpose is to check if the model overfit during development.
- **Production Validation**: Benchmarks **Holdout Days vs. True Unseen Forward Days**. Its purpose is to verify if feature predictive power collapses when deployed to future live market regimes.

### 3.2. Three Feature-Source Populations in Production Validation

There are currently exactly **THREE** feature types:

1. **Feature Registry Features**
   - Canonical features materialized through the Master Dataset.
   - Ownership is controlled by `feature_project_id`.
   - These use the **KEEP / WATCH / REMOVE** recommendation system.
   - They do **NOT** use a Pipeline Feature Score.
   - Registry recommendations are observational evidence for future registry maintenance.
   - There is currently **no automatic retirement or deletion**.

2. **Base Pipeline Features**
   - Pipeline features that are already accepted and belong to the Base Pipeline.
   - They are **NOT** experimental candidates.
   - Base Pipeline features **DO** participate in the Pipeline Feature Score / validation-history concept.
   - Their score is used to monitor their continued stability and performance across unseen validation runs.
   - There is currently **no automatic demotion or deletion**.

3. **Selected Experimental Pipeline Features**
   - Candidate pipeline features generated and selected through the experimental pipeline workflow.
   - They are associated with `pipeline_id` and `pipeline_snapshot_id`.
   - They **DO** participate in the Pipeline Feature Score / validation-history concept.
   - Their unseen validation results provide evidence for whether the feature is strong enough for a future promotion decision.
   - However, there is currently **NO automatic promotion from Experimental &rarr; Base Pipeline**.

---

### 3.3. Important Architectural Distinctions

```
Feature Registry:
    KEEP / WATCH / REMOVE recommendation
    ──► Registry maintenance evidence

Pipeline Features (Base + Experimental):
    Pipeline Feature Score / validation history
    ──► Stability and performance evidence

Experimental Pipeline:
    ──► Future possibility of promotion to Base Pipeline

Base Pipeline:
    ──► Already accepted pipeline features
    ──► Monitored using pipeline validation evidence
```

The current implementation does **NOT** automatically:
- Delete Registry features
- Retire Registry features
- Delete Experimental features
- Remove Base Pipeline features
- Promote Experimental features to Base Pipeline
- Change feature ownership automatically

---

## 4. Unseen Dataset Resolution & Lineage Integrity

The module [`apps/chain_replay_ml/production_validation/unseen_dataset.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/unseen_dataset.py) automatically resolves or builds the required `unseen_*` dataset.

```
                             TRAINED MODEL PACKAGE
                             models/<model_name>/
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     ▼                                 ▼                                 ▼
feature_project_id                pipeline_id                   pipeline_snapshot_id
 (e.g. "chart")                  (e.g. "PL_0005")               ("ca5945f58f8b1a2c")
     │                                 │                                 │
     └─────────────────────────────────┼─────────────────────────────────┘
                                       │
                                       ▼
                       unseen_dataset_identity_hash()
                     (SHA256 content hash of lineage)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
          [Cached Match Found]                   [No Match / Stale]
         Reuse existing parquet:               Generate on-the-fly via
     unseen_<slug>_<hash>.parquet              create_analysis_dataset()
```

### 4.1. Identity Hash Formulation
$$\text{identity\_hash} = \text{SHA256}\Big(\text{master\_db} + \text{unseen\_days} + \text{feature\_project\_id} + \text{pipeline\_id} + \text{pipeline\_snapshot\_id} + \text{flags}\Big)[:8]$$

### 4.2. Zero-Leakage Lineage Guarantees
- **Lineage Reuse**: If an `unseen_*` dataset matching the exact identity hash exists in `datasets/analysis_datasets/`, it is reused immediately.
- **Stale Prevention**: If the model was trained with a different `feature_project_id` or an updated `pipeline_snapshot_id`, the hash changes automatically, triggering a fresh generation. This prevents evaluating models against stale or incompatible feature matrices.

---

## 5. Feature Validation Metrics & Multi-Signal Rules

Located in [`apps/chain_replay_ml/production_validation/rules.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/rules.py):

### 5.1. Comparative Ranking Metrics
For every model feature $f$:
1. **Holdout Rank ($R_{\text{holdout}}$)**: Importance rank on Holdout data ($1 = \text{highest importance}$).
2. **Unseen Rank ($R_{\text{unseen}}$)**: Importance rank on Unseen forward days ($1 = \text{highest importance}$).
3. **Rank Change ($\Delta R$)**:
   $$\Delta R = R_{\text{holdout}} - R_{\text{unseen}}$$
   - $\Delta R < 0 \implies$ Feature fell in importance on unseen data (e.g. Rank 2 &rarr; Rank 10, $\Delta R = -8$).
   - $\Delta R > 0 \implies$ Feature rose in importance on unseen data.
4. **Importance Difference ($\Delta \text{Imp}$)**:
   $$\Delta \text{Imp} = \text{Importance}_{\text{unseen}} - \text{Importance}_{\text{holdout}}$$
5. **Relative Importance Drop ($\text{RelDrop}$)**:
   $$\text{RelDrop} = \frac{\max(0, -\Delta \text{Imp})}{|\text{Importance}_{\text{holdout}}|}$$

---

### 5.2. Multi-Signal Decision Logic (`recommend_feature`)

```
                           Feature Validation Metrics
                                       │
                     ┌─────────────────┼─────────────────┐
                     ▼                 ▼                 ▼
                Rank Drop          Imp Drop        Drift Severity
             (ΔR <= -5: Sev 2) (RelDrop >= 50%: 2) (Drift/KS/W >= Sev 1)
                     │                 │                 │
                     └─────────────────┼─────────────────┘
                                       │
                                       ▼
                   Are ALL critical degradation signals severe?
                                       │
                         YES ──────────┴────────── NO
                          │                         │
                          ▼                         ▼
                       REMOVE               Any signal medium?
                                            (ΔR <= -2 OR RelDrop >= 25%)
                                                    │
                                          YES ──────┴────── NO
                                           │                 │
                                           ▼                 ▼
                                         WATCH              KEEP
```

| Decision | Primary Trigger Criteria | Interpretation |
|---|---|---|
| **`REMOVE`** | Large Rank Drop ($\Delta R \le -5$) **AND** Large Importance Drop ($\text{RelDrop} \ge 50\%$) **AND** High Drift Severity ($\ge 1$). | Feature suffered complete collapse in predictive power and exhibited significant distribution drift. |
| **`WATCH`** | Medium Rank Drop ($\Delta R \le -2$) **OR** Medium Importance Drop ($\text{RelDrop} \ge 25\%$) **OR** Medium Drift. | Feature showed partial degradation or instability between Holdout and Unseen regimes. |
| **`KEEP`** | Stable Rank ($|\Delta R| \le 1$) and stable importance across regimes. | Feature maintained robust predictive utility on true unseen forward trading days. |

---

## 6. Three Feature-Source UI Partitions

In [`apps/master_dataset_tk/production_validation_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/production_validation_panel.py), the comparison table is partitioned using mutually exclusive radio buttons backed by `partition_diagnostic_rows()`:

```
    ┌────────────────────────────────────────────────────────────────────────┐
    │  (o) Selected Experimental (12)    ( ) Base Pipeline (4)    ( ) Feature Registry (28)  │
    └────────────────────────────────────────────────────────────────────────┘
```

1. **Selected Experimental**: Features originating from the model's experimental pipeline (`pipeline_id` + `pipeline_snapshot_id`). Default selection.
2. **Base Pipeline**: Already accepted pipeline features belonging to the Base Pipeline (`base_pipeline_export_features`).
3. **Feature Registry**: Canonical features scoped by `feature_project_id`.

### Enforced Ownership Invariants:
- $\text{Registry Count} + \text{Base Pipeline Count} + \text{Experimental Count} \equiv \text{Total Model Features}$.
- Every feature belongs to exactly one partition; no feature appears in multiple tabs or is omitted.

---

## 7. Recommendation History & Persistence Store

Recommendation history is persisted at `<chart_data_dir>/feature_recommendation_history.json` via [`apps/chain_replay_ml/production_validation/recommendation_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/recommendation_store.py).

### 7.1. Storage Schema
```json
{
  "version": 1,
  "updated_at": "2026-08-16T15:30:00+00:00",
  "entries": [
    {
      "id": 1,
      "feature_id": "FR0012",
      "feature_name": "atm_iv_ce",
      "model_name": "Future_LTP_5m_WF_1168f_XGB_2243_14",
      "recommendation": "REMOVE",
      "generated_date": "2026-08-16T15:30:00+00:00",
      "production_validation_run_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "recommendation_detail": "{'rank_severity': 2, 'imp_severity': 2, 'drift_severity': 1}"
    }
  ],
  "ignored": {},
  "summary": {
    "by_feature": {
      "atm_iv_ce": {
        "feature_id": "FR0012",
        "feature_name": "atm_iv_ce",
        "domain": "implied_volatility",
        "remove_runs": 3,
        "remove_models": 2,
        "watch_runs": 1,
        "watch_models": 1,
        "keep_runs": 0,
        "keep_models": 0,
        "last_recommendation": "REMOVE",
        "last_model": "Future_LTP_5m_WF_1168f_XGB_2243_14",
        "last_date": "2026-08-16T15:30:00+00:00",
        "recommendation_strength": 4,
        "recommendation_strength_stars": "★★★★☆"
      }
    }
  }
}
```

---

## 8. Accumulation of REMOVE Runs

The recommendation store aggregates evidence across multiple validation runs:

### 8.1. De-duplication Rule (`_upsert_entry`)
- Each entry is keyed by `(production_validation_run_id, model_name, feature_name)`.
- Re-running validation for the same run ID updates the existing record rather than double-counting.

### 8.2. Multi-Run Accumulation Example:
```
Validation Run 1 (Model A) ──► Recommendation: REMOVE ──► remove_runs = 1, remove_models = 1
Validation Run 2 (Model B) ──► Recommendation: REMOVE ──► remove_runs = 2, remove_models = 2
Validation Run 3 (Model C) ──► Recommendation: KEEP   ──► remove_runs = 2, keep_runs = 1
Validation Run 4 (Model D) ──► Recommendation: REMOVE ──► remove_runs = 3, remove_models = 3
```

- `remove_runs`: Total count of validation runs that yielded `REMOVE`.
- `remove_models`: Count of **unique models** that recommended `REMOVE`.
- `last_date`: Timestamp of the most recent validation run for that feature.

---

## 9. Recommendation Management UI Actions

### 9.1. "Update Registry Recommendations" Button

**What it does:**
- Persists the current Production Validation results for the selected model into `feature_recommendation_history.json`.
- Stores recommendation evidence for Feature Registry, Base Pipeline, and Selected Experimental Pipeline features according to their respective feature-source classification.
- Rebuilds aggregated feature summaries and validation history.

**What it DOES NOT do:**
- Does **NOT** automatically delete Feature Registry features.
- Does **NOT** automatically remove Base Pipeline features.
- Does **NOT** automatically remove Experimental Pipeline features.
- Does **NOT** automatically promote Experimental features to Base Pipeline.
- Does **NOT** change feature ownership.
- Displays confirmation message: *"No pipeline features were removed. No registry features were retired."*

### 9.2. "Refresh Recommendations"
- **What it does**: Reloads `feature_recommendation_history.json` from disk, rebuilds in-memory summaries, and refreshes recommendation count badges and tree tables in the UI.
- **Computation**: Does not re-run ML models or recompute metrics; reads stored historical entries.

### 9.3. "Ignore Recommendation"
- **What it does**: Writes the feature name to the `ignored` dictionary in `feature_recommendation_history.json`.
- **Effect**: Suppresses the feature from appearing in `recommended_for_removal()` lists in Pipeline/Registry retirement dialogs without deleting historical evidence entries.

### 9.4. "Remove Selected" (in Management Dialogs)
- **What it does**: In the Pipeline Feature Manager or Feature Registry Retirement Dialog, pressing "Remove Selected" invokes permanent exclusion (`retire_pipeline_features` or `retire_registry_features`) to exclude chosen features from future builds.
- **UI Warning**: The UI explicitly clarifies: *"Persists recommendations only — does not remove pipeline features or retire registry features until explicitly confirmed in management dialogs."*

---

## 10. Important Current Architectural Invariants & Limitations

1. **Evidence vs. Lifecycle Mutation**: Recommendations (`KEEP`, `WATCH`, `REMOVE`) and `remove_runs` are **observational audit evidence**. They do **not** trigger automatic deletion or state changes.
2. **No Automatic Registry Deletion**: Canonical features in the Feature Registry are never deleted automatically by validation runs.
3. **No Automatic Pipeline Deletion**: Experimental pipeline candidate features and Base Pipeline features are never removed automatically by validation runs.
4. **No Automated Promotion**: There is currently **no automated workflow to promote Experimental features to Base**.
5. **Deterministic Lineage**: Unseen validation datasets strictly inherit the parent model's `feature_project_id`, `pipeline_id`, and `pipeline_snapshot_id`.

---

## 11. Current End-to-End Recommendation Lifecycle Diagram

```
Feature Registry (Master DB)  ∪  Pipeline Features (Base / Experimental Candidates)
                            │
                            ▼
                Analysis Dataset (.parquet)
                            │
                            ▼
                  Trained Model Package
                 (config.json + lineage)
                            │
                            ▼
                 Production Validation
              (True Unseen Days Resolution)
                            │
                            ▼
                  Feature Validation
              (Holdout vs. Unseen Metrics)
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
           KEEP           WATCH          REMOVE
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
          "Update Registry Recommendations"
                            │
                            ▼
     feature_recommendation_history.json (Audit Store)
       • Increments remove_runs / keep_runs
       • Updates remove_models count
       • Records last_date & run_id
                            │
                            ▼
          Future Validation Runs Accumulate Evidence
```

---

## 12. Part B — Future Lifecycle Design (Placeholder)

> [!NOTE]
> Future lifecycle design is intentionally not defined in this document yet.
> A separate architecture decision is required for evidence scoring,
> feature promotion, Base Pipeline admission, regression/demotion,
> and retirement.
