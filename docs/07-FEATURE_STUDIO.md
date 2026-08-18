# AruMLStudio Feature Studio — Technical & Functional Architecture
## Authoritative Technical Specification (Phase 1 + Phase 2A + Phase 2B Integration)

> **Document Status**: AUTHORITATIVE TECHNICAL ARCHITECTURE  
> **Integrated Subsystems**: Feature Studio Diagnostics Suite, Production Validation Engine, SQLite Evidence Database (Phase 1), Policy Settings & Versioning (Phase 1), Evidence Intelligence (Phase 2A), Stability, Risk Badges & Level-1 Generalization (Phase 2B)  
> **Detailed Lifecycle References**: [`docs/08-FEATURE_RECOMMENDATION_LIFECYCLE.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08-FEATURE_RECOMMENDATION_LIFECYCLE.md), [`docs/08.1-FEATURE_RECOMMENDATION_SCORING_LIFECYCLE_POLICY.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08.1-FEATURE_RECOMMENDATION_SCORING_LIFECYCLE_POLICY.md), [`docs/08.2-FEATURE_RECOMMENDATION_POLICY_SETTINGS.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08.2-FEATURE_RECOMMENDATION_POLICY_SETTINGS.md), [`docs/08.3-FEATURE_RECOMMENDATION_PHASE_2A_EVIDENCE_INTELLIGENCE.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08.3-FEATURE_RECOMMENDATION_PHASE_2A_EVIDENCE_INTELLIGENCE.md), [`docs/08.4-FEATURE_RECOMMENDATION_PHASE_2B_STABILITY_RISK_GENERALIZATION.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08.4-FEATURE_RECOMMENDATION_PHASE_2B_STABILITY_RISK_GENERALIZATION.md)  
> **Future Subsystems**: Phase 3 (Recommendation-to-Training Decision Engine) is documented separately in [`docs/08.5-RECOMMENDATION_TO_TRAINING_DECISION_ENGINE.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08.5-RECOMMENDATION_TO_TRAINING_DECISION_ENGINE.md) as a PROPOSED architectural design.

---

## 1. Executive Summary & Purpose

The **Feature Studio** is the primary analytical and validation suite within **AruMLStudio** for inspecting, profiling, diagnosing, and validating model features across their full operational lifecycle.

Machine learning models for options trading combine features from multiple distinct generative populations (canonical domain features, deterministic mathematical transformations, and experimental statistical pipelines). Feature Studio provides complete end-to-end visibility into:
- **Feature Importance Rankings**: Native tree gain, Permutation degradation, and SHAP attribution.
- **Univariate Distribution Properties**: Data anomalies, moments, null percentages, and percentile profiles.
- **Temporal Feature Drift**: Distribution shifts between training (Walk-Forward) and out-of-sample (Holdout) splits.
- **Multi-Model Delta Comparisons**: Side-by-side artifact joins across paired training architectures.
- **Automated Root-Cause Diagnostics**: Automated detection of overfitting, severe drift, volatility regimes, or distribution shifts.
- **Production Validation Engine**: Validation over true unseen forward trading days to synthesize multi-signal recommendations (**`KEEP`**, **`WATCH`**, **`REMOVE`**).
- **SQLite Evidence DB & Evidence Studio**: Append-only persistence into `feature_recommendation_evidence.db`, dual materialized projections, configurable Policy Settings, and query-time intelligence (Confidence, Consensus, Freshness, Stability, Risk Badges, Level-1 Generalization).

---

## 2. Scope & Subsystems

This document provides technical and functional documentation for:
- **`FeatureStudioPanel`** ([`apps/master_dataset_tk/feature_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_studio_panel.py)) and its diagnostics tabs.
- **Compute and Artifact Pipelines** in `apps/chain_replay_ml/`:
  - Feature Importance (`chain_replay_ml.feature_importance_studio`)
  - Feature Distribution (`chain_replay_ml.feature_distribution_studio`)
  - Feature Drift (`chain_replay_ml.feature_drift_studio`)
  - Studio Compare / Multi-Model (`chain_replay_ml.multi_model_studio`)
  - Diagnostics Studio (`chain_replay_ml.diagnostics_studio`)
  - Production Validation (`chain_replay_ml.production_validation`)
- **Feature Recommendation Evidence Subsystem**:
  - SQLite Evidence Database (`evidence_store.py`)
  - Policy Settings Engine & Versioning (`recommendation_policy.py`)
  - Query-Time Evidence Intelligence (`recommendation_store.py`, `recommendation_policy.py`)
  - Feature Recommendation Evidence Studio GUI (`feature_recommendation_viewer.py`)
- **Three-Population Feature Classification Engine** (`chain_replay_ml.dataset_builder.feature_sources_catalog`).
- **Feature Project Isolation & Lineage** (`chain_replay_ml.dataset_builder.feature_project_organization`, `master_feature_project`).

---

## 3. Architecture Overview

Feature Studio follows a decoupled **Compute &rarr; Persist &rarr; Load &rarr; Populate** execution model. UI panels act as stateless visualizers over on-disk JSON/Parquet artifacts stored directly inside each trained model's package directory (`models/<model_name>/`) and the canonical SQLite Evidence Database (`feature_recommendation_evidence.db`).

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
    (Importance → Dist → Drift → Diag → ProdVal)                     (Importance → Dist → Drift → Diag → ProdVal)
                         │                                                              │
                         ▼                                                              ▼
    ┌───────────────────────────────────────────────┐                  ┌─────────────────────────────────────────┐
    │ Populates UI Tab Payloads                     │                  │ Writes JSON artifacts to model package  │
    │ (importance, distribution, drift, diagnostics)│                  │ models/<model_name>/<studio_dir>/       │
    └───────────────────────────────────────────────┘                  └───────────────────┬─────────────────────┘
                                                                                           │
                                                                                           ▼
                                                                               persist_validation_evidence()
                                                                                           │
                                                                                           ▼
                                                                                feature_recommendation_evidence.db
```

### Core Design Invariants:
1. **Controller-Owned Pipeline**: `FeatureStudioPanel` manages model selection, shared filtering, and background compute threading. Tabs do not trigger ad-hoc recomputations upon tab-switching.
2. **Deterministic Artifact Caching**: Loaded artifacts are cached per model in `_cache: PipelineResult`. Switching tabs reuses cached in-memory structures without disk re-reads.
3. **Strict Separation of Data Regimes**:
   - **Diagnostics Studio** strictly evaluates **Walk-Forward (WF) Training Data vs. Holdout Data**.
   - **Production Validation** strictly evaluates **Holdout Data vs. Unseen Forward Days**.
4. **Authoritative Evidence Immutability**: Historical validation events in SQLite `recommendation_evidence` are append-only facts and are never overwritten.

---

## 4. The Three Feature Populations

The architecture partitions all model features into **three disjoint, exhaustive feature populations**:

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
   • Immune from blocking            • Immune from blocking            • Subject to candidate gate
```

### 4.1. Feature Registry Features
- **Origin**: Canonical features materialized through the Master Dataset catalog (`feature_registry_store` and `feature_domains.py`). Scoped by `feature_project_id`.
- **Governance**: Receives `KEEP` / `WATCH` / `REMOVE` recommendations. REMOVE flags trigger `alert` health states for data curators.
- **Invariants**: Registry features are **never automatically retired, deleted, or blocked** from model training.

### 4.2. Base Pipeline Features
- **Origin**: Approved production transformation generators (`base_pipeline_export_features`).
- **Governance**: Evaluated for degradation. Repeated REMOVEs demote priority rank and trigger health warnings.
- **Invariants**: Base Pipeline features are **never automatically deleted or blocked** from candidate generation.

### 4.3. Selected Experimental Features
- **Origin**: High-velocity exploratory candidate transformations generated within a specific pipeline lineage (`pipeline_id` + `pipeline_snapshot_id`).
- **Governance**:
  - Repeated REMOVEs trigger context-level **candidate blocking** in Auto Candidate Generation.
  - Consistent KEEP performance unlocks **`PROMOTION_CANDIDATE`** status for human architectural review.
- **Invariants**: Promotion is strictly a human governance milestone; there is **no automatic code generation or Base Pipeline mutation**.

---

## 5. Feature Studio Tabs & Diagnostic Suite

### 5.1. Feature Importance Tab
- **Compute Package**: `chain_replay_ml.feature_importance_studio`
- **Metrics**: Native tree gain/split importance, Permutation drop ($\Delta\text{MAE}$ upon shuffling), and SHAP attribution.
- **Outputs**: `models/<model>/feature_importance/native.json`, `permutation.json`, `shap.json`.

### 5.2. Feature Distribution Tab
- **Compute Package**: `chain_replay_ml.feature_distribution_studio`
- **Metrics**: Univariate statistics (mean, std, min, max, skewness, kurtosis, percentiles $p10 \dots p90$, null counts).
- **Outputs**: `models/<model>/feature_distribution/holdout.json`.

### 5.3. Feature Drift Tab
- **Compute Package**: `chain_replay_ml.feature_drift_studio`
- **Metrics**: Distribution divergence between Walk-Forward (WF) and Holdout (HO) splits using Kolmogorov-Smirnov (KS) statistics, Wasserstein distance, and drift severity scoring ($0 \dots 2$).
- **Outputs**: `models/<model>/feature_drift/ranking.json`, `comparison.json`.

### 5.4. Studio Compare Tab
- **Compute Package**: `chain_replay_ml.multi_model_studio`
- **Purpose**: Side-by-side artifact delta joins between Model A and Model B without recomputation.
- **Outputs**: `models/_pairs/<A>__vs__<B>/feature_studio_compare/comparison.json`.

### 5.5. Diagnostics Studio Tab
- **Compute Package**: `chain_replay_ml.diagnostics_studio`
- **Purpose**: Automated root-cause performance diagnosis across Holdout vs. Walk-Forward data splits.
- **Headline Diagnostics**: `Overfitting`, `Severe Drift`, `Feature Shift`, `Target Shift`, `Volatility Shift`, or `Stable/Good`.
- **Three-Source Partitioning**: Partitions features into Registry, Base Pipeline, and Selected Experimental sub-tables.

### 5.6. Production Validation Tab
- **Compute Package**: `chain_replay_ml.production_validation`
- **Purpose**: Forward out-of-sample evaluation against unseen trading days.
- **Outputs**: `unseen_metrics.json`, `feature_comparison.json`, and automatic background persistence into `feature_recommendation_evidence.db`.

---

## 6. Feature Recommendation Evidence Architecture

The authoritative recommendation store is the SQLite database located at `<chart_data_dir>/feature_recommendation_evidence.db`.

```
                         PRODUCTION VALIDATION RUN
                                     │
                                     ▼
                      persist_validation_evidence()
                                     │
                                     ▼
                          recommendation_evidence
                         (IMMUTABLE RAW AUDIT LOG)
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
          feature_context_summary         experimental_lineage_summary
            (Context Projection)               (Lineage Projection)
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     QUERY-TIME INTELLIGENCE ENRICHMENT
                     • Phase 2A: Confidence C, Consensus, Freshness, Advisory Rank
                     • Phase 2B: Volatility σ_S, Range ΔS, Level-1 Gen Index G, Risk Badges
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
       Feature Recommendation Evidence Studio    Auto Candidate Generation
       (5-Tab Operator Workspace)                (Pre-Training Elimination Gate)
```

### 6.1. Raw Evidence vs Dual Projections
1. **`recommendation_evidence`**: Append-only log of every validation event (`evidence_id`, `context_id`, `feature_name`, `feature_source`, `recommendation`, `validation_run_id`, `model_name`, `holdout_rank`, `unseen_rank`, `rank_change`, `relative_imp_drop`, `drift_severity`, `run_timestamp`).
2. **`feature_context_summary`**: Context-level projection aggregating runs, models, streaks, scores, and candidate blocking states per `(context_id, feature_source, feature_name)`.
3. **`experimental_lineage_summary`**: Lineage-specific projection tracking exact algorithmic provenance and `PROMOTION_CANDIDATE` eligibility per `(context_id, pipeline_id, pipeline_snapshot_id, feature_name)`.
4. **`policy_settings_history`**: Audit log recording every policy version, context override, and rollback snapshot.

### 6.2. Cumulative Evidence Scoring Formula (Phase 1)
$$\text{raw\_score} = (w_{\text{keep}} \cdot M_{\text{keep}}) + (w_{\text{remove}} \cdot M_{\text{remove}}) + (w_{\text{watch}} \cdot M_{\text{watch}}) + (B_{\text{keep}} \cdot S_{\text{keep}}) + (P_{\text{remove}} \cdot S_{\text{remove}})$$
$$\text{evidence\_score} = \text{round}\Big(\max(-100.0, \min(+100.0, \text{raw\_score})), 2\Big)$$
- Default weights: $w_{\text{keep}} = +25.0$, $w_{\text{remove}} = -35.0$, $w_{\text{watch}} = -10.0$, $B_{\text{keep}} = +15.0$, $P_{\text{remove}} = -25.0$.

---

## 7. Phase 2A & 2B Query-Time Evidence Intelligence

Phase 2A and Phase 2B add rich statistical intelligence dynamically at query time with zero SQLite schema changes and zero raw evidence mutation:

### 7.1. Phase 2A Intelligence Metrics
1. **Evidence Confidence ($C$)**:
   $$C = \sqrt{C_{\text{runs}} \times C_{\text{models}}} = \sqrt{\left(1 - e^{-N_{\text{runs}}/3.0}\right) \times \left(1 - e^{-M_{\text{unique}}/2.0}\right)}$$
   - $N=1, M=1 \implies 33.4\%$ | $N=2, M=2 \implies 55.5\%$ | $N=3, M=3 \implies 70.1\%$
2. **Model Consensus & Strict Tie Contract**: Reconstructs latest vote per model. 50/50 splits or 3-way deadlocks are classified as **`SPLIT (50%)`**.
3. **Freshness Bands**: $\le 7\text{d} \implies \text{Fresh}$, $8\text{–}30\text{d} \implies \text{Recent}$, $> 30\text{d} \implies \text{Stale}$.
4. **Dual Ranking**:
   - **`priority_rank` (Phase 1 Authoritative)**: Sorted by `evidence_score DESC, keep_runs DESC, feature_name ASC`.
   - **`operational_priority_score`**: $\text{round}(\text{evidence\_score} \times C, 2)$.
   - **`advisory_rank` (Phase 2A Preview)**: Sorted by `operational_priority_score DESC, keep_runs DESC, feature_name ASC`.

### 7.2. Phase 2B Stability, Risk Badges & Generalization
1. **Score Volatility ($\sigma_S$) & Trajectory Spread ($N \ge 3$)**:
   $$\sigma_S = \sqrt{\frac{1}{N - 1} \sum_{t=1}^N (S_t - \bar{S})^2} \quad (\sigma_S < 15.0 \implies \text{Stable},\ 15.0 \le \sigma_S < 35.0 \implies \text{Moderate},\ \sigma_S \ge 35.0 \implies \text{Volatile})$$
   - $N < 3 \implies \text{None}$ (`⚪ N/A (< 3 runs)`).
   - Tracks score range spread $\Delta S = \max(S_t) - \min(S_t)$ and trajectory direction flips $D_{\text{flips}}$.
2. **Level-1 Cross-Context Generalization ($K \ge 2$)**:
   $$G = A_{\text{context}} \times \left(1.0 - \min\left(1.0, \frac{\Delta S_{\text{context}}}{100.0}\right)\right)$$
   - Evaluated across matching market, window, and project dimensions differing only by sampling interval.
   - $G \ge 0.75 \implies \text{Universal}$, $0.50 \le G < 0.75 \implies \text{Scale-Robust}$, $0.25 \le G < 0.50 \implies \text{Scale-Sensitive}$, $G < 0.25 \implies \text{Scale-Specific}$.
3. **Explicit Multi-Dimensional Risk Badges**:
   - `[DEGRADED]` (Score $\le -40.0$), `[SPLIT]` (`is_consensus_tie`), `[STALE]` ($> 30\text{d}$), `[UNSTABLE]` ($\sigma_S \ge 35.0$).

---

## 8. Feature Recommendation Evidence Studio UI (5 Tabs)

Accessible via the **"Evidence DB & Projections"** toolbar button in Feature Studio or Production Validation ([`apps/master_dataset_tk/feature_recommendation_viewer.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_recommendation_viewer.py)):

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Feature Recommendation Evidence Studio                                                            │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Context: [Market: NIFTY ▼] [Interval: 3 ▼] [Window: standard ▼] [Project: all ▼] [x] Legacy  │
│ Context ID: ctx_574ee67348f2                 [Rebuild Projections]  [Refresh Data]                │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ [1. Feature Registry] [2. Base Pipeline] [3. Selected Exp] [4. Raw Log] [5. Policy Settings]      │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Exact Treeview Columns by Tab:
- **Tab 1: Feature Registry**: `Feature Name`, `Runs`, `Models`, `Score`, `Status`, `Confidence`, `Model Consensus`, `Freshness`, `Stability`, `Generalization`, `Badges`, `Last Rec`, `Last Model`.
- **Tab 2: Base Pipeline**: `Priority Rank`, `Feature Name`, `Runs`, `Models`, `Score`, `Confidence`, `Adj Score`, `Advisory Rank`, `Model Consensus`, `Freshness`, `Stability`, `Generalization`, `Badges`, `Status`, `Last Rec`, `Last Model`.
- **Tab 3: Selected Experimental**: `Lineage Status`, `Context Status`, `Feature Name`, `Pipeline ID`, `Snapshot`, `Runs`, `Models`, `Streak`, `Score`, `Confidence`, `Consensus`, `Freshness`, `Stability`, `Generalization`, `Badges`.
- **Tab 4: Raw Evidence Log**: Chronological audit log of all individual validation events.
- **Tab 5: Policy Settings & History**: Interactive editor for scoring weights, experimental blocking/promotion gates, context overrides, version history, preview impact, and rollback.

---

## 9. Policy Settings System Architecture (Tab 5)

Implemented in [`apps/chain_replay_ml/production_validation/recommendation_policy.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/recommendation_policy.py):

- **Tiered Scope Hierarchy**: Global default settings with optional context-specific overrides.
- **Policy Versioning & History**: Every save creates an immutable version snapshot in `policy_settings_history`.
- **No-Op Save Prevention**: Submitting unchanged settings does not create duplicate history rows.
- **Non-Destructive Rollback**: Rolling back to a previous version creates a new active version ($v_{N+1}$) copying the historical state, preserving forward-only audit history.
- **Read-Only Preview Impact**: Evaluates proposed threshold changes in-memory before committing.
- **Projection Rebuild Invariant**: Policy updates automatically trigger `rebuild_all_projections()`, updating projections while preserving raw evidence immutability.

---

## 10. Storage & Artifacts Map

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
| **Production Metrics** | `models/<model>/production_validation/unseen_metrics.json` | JSON | `production_validation` | Prod Validation Tab |
| **SQLite Evidence DB** | `<chart_data_dir>/feature_recommendation_evidence.db` | SQLite | `production_validation` | Evidence Studio, Candidate Gate |
| **Policy Settings DB** | `<chart_data_dir>/feature_recommendation_evidence.db` (table `policy_settings_history`)| SQLite | `recommendation_policy` | Evidence Studio Tab 5 |
| **Feature Projects** | `<chart_data_dir>/feature_registry_projects.json` | JSON | `feature_project_manager`| Master Builder, Model Builder |

---

## 11. Source Code Map

| Component | Source File | Key Class / Functions | Reads | Writes |
|---|---|---|---|---|
| **Studio Shell** | [`apps/master_dataset_tk/feature_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_studio_panel.py) | `FeatureStudioPanel` | Model registry, UI prefs | UI cache, Evidence DB trigger |
| **Pipeline Controller** | [`apps/master_dataset_tk/feature_studio_pipeline.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_studio_pipeline.py) | `run_compute_pipeline`, `run_load_pipeline` | Model packages | Studio artifacts |
| **Diagnostics Engine** | [`apps/chain_replay_ml/diagnostics_studio/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/diagnostics_studio/compute.py) | `run_compute`, `summarize_diagnostics` | Importance, Drift, Metrics | `diagnostics_studio/*.json` |
| **Validation Compute** | [`apps/chain_replay_ml/production_validation/compute.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/compute.py) | `run_production_validation` | Unseen matrix, Model binary | `production_validation/*.json` |
| **Evidence Store** | [`apps/chain_replay_ml/production_validation/evidence_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/evidence_store.py) | `append_validation_evidence`, `rebuild_all_projections`, `query_blocked_candidates` | SQLite schema, Validation rows | `recommendation_evidence`, Projections |
| **Policy Engine** | [`apps/chain_replay_ml/production_validation/recommendation_policy.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/recommendation_policy.py) | `RecommendationPolicy`, `compute_evidence_score`, `compute_score_volatility`, `compute_context_generalization` | Policy config, Evidence history | Projections, Policy History |
| **Recommendation Store**| [`apps/chain_replay_ml/production_validation/recommendation_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/recommendation_store.py)| `get_population_recommendations`, `_enrich_intelligence_metrics` | Projections, Raw Evidence | Query-Time Intelligence Payloads |
| **Dataset Context** | [`apps/chain_replay_ml/production_validation/dataset_context.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/dataset_context.py) | `build_dataset_context`, `resolve_dataset_context_id` | Market, Sampling, Window, Project | Canonical Context ID |
| **Evidence Studio GUI**| [`apps/master_dataset_tk/feature_recommendation_viewer.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_recommendation_viewer.py)| `FeatureRecommendationViewerDialog` | SQLite Evidence DB, Policies | Projections, Policies, UI Views |
| **Candidate Gate** | [`apps/master_dataset_tk/auto_candidate_generation.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/auto_candidate_generation.py) | `select_interaction_parent_features`, `query_blocked_candidates` | `feature_context_summary`, Decision Engine | Filtered Candidate List, Balanced Interaction Sets |
