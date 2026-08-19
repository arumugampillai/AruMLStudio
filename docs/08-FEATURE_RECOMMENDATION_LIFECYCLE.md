# AruMLStudio Feature Recommendation Lifecycle & Evidence Subsystem
## Master Architectural & Technical Reference (Phases 1, 2A, 2B, 3A, 3B, 3C, 3D)

> **Document Status**: AUTHORITATIVE MASTER LIFECYCLE SPECIFICATION  
> **Target Subsystem**: Feature Recommendation Lifecycle, Decision Engine & Governance Subsystem  
> **Subsystems Implemented & Verified (210/210 Tests Passing)**:  
> - **Phase 1**: SQLite Evidence DB, Bounded Evidence Scoring, Dual Projections, Candidate Elimination Gate, Configurable Policy Settings, Versioning & Rollback  
> - **Phase 2A**: Evidence Intelligence (Confidence Saturation, Multi-Model Consensus, Strict SPLIT Tie Handling, Validation Freshness, Operational Priority Score, Advisory Rank)  
> - **Phase 2B**: Behavioral Stability (Score Volatility $\sigma_S$, Score Range Spread $\Delta S$, Direction Flips $D_{\text{flips}}$), Level-1 Cross-Context Generalization Index ($G$), Explicit Risk Badges (`[DEGRADED]`, `[SPLIT]`, `[STALE]`, `[UNSTABLE]`)  
> - **Phase 3A**: Recommendation-to-Training Decision Engine (`training_decision_engine.py`, 4-state qualification `TRAIN_CANDIDATE`, `REVIEW`, `NEW_UNSEEN`, `EXCLUDE`, and `[PROMOTION]` qualification)  
> - **Phase 3B**: Evidence Studio UI Integration (Decision column, Reason Inspector dialog, badge styling)  
> - **Phase 3C**: Model Builder Preset Handoff & Closed-Loop Training Provenance (`model_builder_handoff.py`, `training_provenance_meta.json`)  
> - **Phase 3D**: Feature Promotion, Graduation & Deprecation Governance (Dossier compilation 3D.1, Governance UI 3D.2, Atomic Registry Graduation 3D.3, Base Pipeline Promotion 3D.4A, Feature Deprecation 3D.4B, Multi-Mode Governance UI 3D.4C)  
>
> **Master Documentation Architecture**:
> ```
> Doc 08 — Feature Recommendation Lifecycle MASTER
>   ├── 8.1 — Scoring & Lifecycle Policy (docs/08.1-FEATURE_RECOMMENDATION_SCORING_LIFECYCLE_POLICY.md)
>   ├── 8.2 — Policy Settings & Versioning (docs/08.2-FEATURE_RECOMMENDATION_POLICY_SETTINGS.md)
>   ├── 8.3 — Phase 2A — Evidence Intelligence (docs/08.3-FEATURE_RECOMMENDATION_PHASE_2A_EVIDENCE_INTELLIGENCE.md)
>   ├── 8.4 — Phase 2B — Stability, Risk & Generalization (docs/08.4-FEATURE_RECOMMENDATION_PHASE_2B_STABILITY_RISK_GENERALIZATION.md)
>   ├── 8.5 — Phase 3A/3B/3C — Decision Engine & Training Handoff (docs/08.5-RECOMMENDATION_TO_TRAINING_DECISION_ENGINE.md)
>   └── 8.6 — Phase 3D — Feature Promotion, Graduation & Governance (docs/08.6-FEATURE_PROMOTION_GRADUATION_GOVERNANCE.md)
> ```

---

## 1. Executive Summary & Master Architecture

The **Feature Recommendation Subsystem** in **AruMLStudio** provides end-to-end evidence accumulation, lifecycle governance, multi-model consensus, score stability, and cross-timeframe generalization for all engineered features.

Validation diagnostics benchmark trained machine learning models against out-of-sample unseen market data to synthesize multi-signal recommendations (**`KEEP`**, **`WATCH`**, **`REMOVE`**). These recommendations are automatically persisted into an append-only, immutable SQLite evidence database (`feature_recommendation_evidence.db`), materialized into dual projections, enriched with query-time evidence intelligence, and presented through the 5-tab **Feature Recommendation Evidence Studio**.

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

> [!IMPORTANT]
> **Core Architectural Invariants**:
> 1. `recommendation_evidence` is strictly **append-only and immutable**. Historical events are permanent facts.
> 2. Dual projections (`feature_context_summary`, `experimental_lineage_summary`) are **purely deterministic and rebuildable** from raw evidence at any time.
> 3. Phase 2A and Phase 2B intelligence metrics are derived **dynamically at query time** with zero schema migrations and zero DB mutations.
> 4. **Registry & Base Pipeline Immunity**: Feature Registry and Base Pipeline features are **never automatically deleted or blocked**.
> 5. **Experimental Promotion**: `PROMOTION_CANDIDATE` is an eligibility state for human governance review; it **never automatically mutates the Base Pipeline**.

---

## 2. Production Validation Forward Benchmarking

Diagnostics and forward out-of-sample benchmarking are orchestrated across **Feature Studio** ([`apps/master_dataset_tk/feature_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_studio_panel.py)) and **Production Validation** ([`apps/master_dataset_tk/production_validation_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/production_validation_panel.py)):

```
                                  FEATURE STUDIO
                                (FeatureStudioPanel)
                                         │
       ┌─────────────┬─────────────┬─────┴───────┬─────────────┬─────────────┬─────────────┐
       ▼             ▼             ▼             ▼             ▼             ▼             ▼
  [Importance] [Distribution]   [Drift]   [Studio Compare] [Diagnostics]  [Planner]  [Production]
   (Native/SHAP) (Moments)   (WF vs. HO)  (Model Deltas)  (Root Cause)   (Audit)     [Validation]
```

### 2.1. Holdout vs. True Unseen Forward Benchmarking
- **Diagnostics Studio**: Compares **Walk-Forward Training Days vs. Holdout Days** to identify localized feature drift or overfitting.
- **Production Validation**: Compares **Holdout Days vs. True Unseen Forward Days** never seen during hyperparameter tuning or feature selection. It verifies if predictive utility collapses under live forward market dynamics.

---

## 3. The Three Feature Populations

Every feature entering Production Validation is partitioned into exactly one of three distinct governance populations:

```
                            THREE FEATURE POPULATIONS
                                        │
     ┌──────────────────────────────────┼──────────────────────────────────┐
     ▼                                  ▼                                  ▼
FEATURE REGISTRY                  BASE PIPELINE                  SELECTED EXPERIMENTAL
Canonical Master Dataset         Approved Baseline Features     Experimental Candidate Lineage
(feature_project_id)            (base_pipeline_export_features) (pipeline_id:pipeline_snapshot_id)
     │                                  │                                  │
     ▼                                  ▼                                  ▼
• Scoped by project             • Monitored for degradation     • Tracked by exact lineage
• Observational KEEP/WATCH/REMOVE • Evidence score & rank health • Streak & eligibility tracking
• REMOVE: alert state only      • REMOVE: alert state only      • REMOVE: context candidate gate
• NEVER automatically deleted   • NEVER automatically deleted   • PROMOTION_CANDIDATE eligibility
• NEVER blocked                 • NEVER blocked                 • NEVER automatically promoted
```

1. **Feature Registry Features**: Canonical features defined in the Master Dataset schema. Scoped by `feature_project_id`. Repeated REMOVEs raise observational `[ALERT]` flags for data curators, but registry features are **never blocked or deleted**.
2. **Base Pipeline Features**: Approved baseline pipeline transformations (`base_pipeline_export_features`). Monitored for degradation. Repeated REMOVEs demote rank and trigger health warnings, but base features are **never blocked or deleted**.
3. **Selected Experimental Features**: Candidate transformations generated and evaluated within an explicit pipeline lineage (`pipeline_id` + `pipeline_snapshot_id`). Consecutive REMOVE streaks trigger context-level **candidate blocking** in Auto Candidate Generation. Consistent KEEP performance unlocks **`PROMOTION_CANDIDATE`** status.

---

## 4. Dataset Context & Context Isolation

Feature performance is strictly isolated by dataset regimes to prevent cross-market or cross-timeframe pollution:

```
                                DATASET CONTEXT DIMENSIONS
                                             │
               ┌─────────────────┬───────────┴───────────┬─────────────────┐
               ▼                 ▼                       ▼                 ▼
            Market       Sampling Interval        Sliding Window   Feature Project ID
        (e.g. NIFTY)        (e.g. 3s)           (e.g. standard)       (e.g. all)
```

$$\text{context\_key} = \mathtt{"NIFTY:3:standard:all"}$$
$$\text{context\_id} = \mathtt{"ctx\_"} + \text{SHA256}(\text{context\_key})[:12] \implies \mathtt{"ctx\_574ee67348f2"}$$

- **Cross-Context Isolation**: Evidence from high-frequency $1\text{s}$ datasets never pollutes or blocks features in $3\text{s}$ or $6\text{s}$ datasets. Evidence from `SENSEX` never impacts `NIFTY`.
- **`legacy_unknown` Isolation**: Historical records with unrecoverable dataset metadata are assigned `context_id = "legacy_unknown"`. They are visible for audit in Evidence Studio via the **"Include Legacy Unknown"** checkbox, but are strictly excluded from active candidate blocking.

---

## 5. Recommendation Decision Logic (`recommend_feature`)

Located in [`apps/chain_replay_ml/production_validation/rules.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/rules.py):

| Recommendation | Exact Criteria | Semantic Meaning |
|---|---|---|
| **`REMOVE`** | Severe Rank Drop ($\Delta R \le -5$) **AND** Severe Importance Drop ($\text{RelDrop} \ge 50\%$) **AND** Distribution Drift ($\ge 1$). | Predictive power collapsed on forward data alongside significant distribution drift. |
| **`WATCH`** | Medium Rank Drop ($\Delta R \le -2$) **OR** Medium Importance Drop ($\text{RelDrop} \ge 25\%$) **OR** Moderate Drift. | Partial degradation or distribution instability between Holdout and Unseen regimes. |
| **`KEEP`** | Stable Rank ($|\Delta R| \le 1$) and stable importance across regimes. | Feature maintained robust predictive utility on true unseen forward trading days. |

---

## 6. SQLite Evidence Database Architecture (Phase 1)

The canonical database is located at:
$$\text{Path: }\mathtt{<chart\_data\_dir>/feature\_recommendation\_evidence.db}$$

```
                                SQLite EVIDENCE STORE
                         (feature_recommendation_evidence.db)
                                          │
    ┌───────────────────────────┬─────────┴─────────┬───────────────────────────┬───────────────────────────┐
    ▼                           ▼                   ▼                           ▼                           ▼
dataset_contexts     recommendation_evidence  feature_context_summary  experimental_lineage_summary  policy_settings_history
(Regime Metadata)     (Immutable Raw Log)       (Context Projection)        (Lineage Projection)        (Policy Audit Log)
```

### 6.1. Schema DDL Summary
1. **`dataset_contexts`**: Unique dataset combinations (`context_id`, `market`, `sampling_interval_sec`, `sliding_window`, `feature_project_id`, `context_key`).
2. **`recommendation_evidence`**: Append-only immutable log of every validation event (`evidence_id`, `context_id`, `feature_name`, `feature_source`, `pipeline_id`, `pipeline_snapshot_id`, `recommendation`, `validation_run_id`, `model_name`, `holdout_rank`, `unseen_rank`, `rank_change`, `relative_imp_drop`, `drift_severity`, `run_timestamp`, `created_at`).
3. **`feature_context_summary`**: Context-level projection (`context_id`, `feature_source`, `feature_name`, `total_runs`, `unique_models_count`, `keep_runs`, `watch_runs`, `remove_runs`, `current_streak_type`, `current_streak_count`, `evidence_score`, `lifecycle_status`).
4. **`experimental_lineage_summary`**: Lineage-specific projection (`context_id`, `pipeline_id`, `pipeline_snapshot_id`, `feature_name`, `feature_identity_key`, `total_runs`, `unique_models_count`, `keep_runs`, `watch_runs`, `remove_runs`, `current_streak_type`, `current_streak_count`, `evidence_score`, `lifecycle_status`).
5. **`policy_settings_history`**: Policy version audit log (`policy_id`, `version`, `settings_json`, `created_at`, `created_by`, `change_reason`).

---

## 7. Automatic Validation Evidence Persistence

- **Automatic Flow**: Production Validation automatically executes `persist_validation_evidence(data_dir, model_name)` upon compute completion and cached result loading.
- **Idempotency**: SQLite enforces `ON CONFLICT(evidence_id) DO NOTHING` using deterministic IDs (`ev_{run_id}_{model}_{feature}`).
- **Deterministic Rebuild**: `rebuild_all_projections()` recalculates projections directly from raw evidence without data loss.

---

## 8. Detailed Recommendation Subsystem Architecture

### 8.1. Scoring & Lifecycle Policy (Phase 1)
Implemented in [`apps/chain_replay_ml/production_validation/recommendation_policy.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/recommendation_policy.py) *(detailed in [`docs/08.1`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08.1-FEATURE_RECOMMENDATION_SCORING_LIFECYCLE_POLICY.md))*:

#### Evidence Scoring Formula:
$$\text{raw\_score} = (w_{\text{keep}} \cdot M_{\text{keep}}) + (w_{\text{remove}} \cdot M_{\text{remove}}) + (w_{\text{watch}} \cdot M_{\text{watch}}) + (B_{\text{keep}} \cdot S_{\text{keep}}) + (P_{\text{remove}} \cdot S_{\text{remove}})$$
$$\text{evidence\_score} = \text{round}\Big(\max(S_{\min}, \min(S_{\max}, \text{raw\_score})), 2\Big)$$

- **Default Weights**: $w_{\text{keep}} = +25.0$, $w_{\text{remove}} = -35.0$, $w_{\text{watch}} = -10.0$, $B_{\text{keep}} = +15.0$, $P_{\text{remove}} = -25.0$, $[S_{\min}, S_{\max}] = [-100.0, +100.0]$.
- **Experimental Blocking Thresholds**:
  $$S_{\text{remove}} \ge \mathtt{remove\_block\_consecutive\_threshold}\text{ (2)} \quad\mathbf{OR}\quad \text{remove\_runs} \ge \mathtt{remove\_block\_total\_threshold}\text{ (4)} \implies \mathtt{blocked}$$
- **Experimental Promotion Candidate Rule**:
  $$\text{lineage\_status} = \mathbf{PROMOTION\_CANDIDATE} \iff (S_{\text{keep}} \ge 3) \land (M_{\text{unique}} \ge 2) \land (\text{evidence\_score} \ge 75.0)$$
- **Population Governance Invariant**: Registry and Base Pipeline features are strictly immune from automated candidate blocking and code deletion.

---

### 8.2. Policy Settings & Versioning (Phase 1)
Implemented in `recommendation_policy.py` *(detailed in [`docs/08.2`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08.2-FEATURE_RECOMMENDATION_POLICY_SETTINGS.md))*:

- **Tiered Scope Hierarchy**: Global default settings (`pol_recommended_default`) with optional context-specific overrides (`pol_ctx_<id>_v1`).
- **Policy Versioning & History**: Every save writes an immutable snapshot to `policy_settings_history`.
- **No-Op Save Prevention**: Unchanged submissions do not create duplicate history records.
- **Non-Destructive Rollback**: Rolling back copies the historical state into a new active version ($v_{N+1}$), maintaining forward-only audit integrity.
- **Read-Only Preview Policy Impact**: Simulates proposed threshold changes in-memory before committing.
- **Projection Metadata**: Materialized projections record `projection_policy_id`, `projection_policy_version`, and `projection_rebuilt_at`.
- **Evidence Immutability**: Changing policy rules reinterprets facts and updates projections; it **never modifies historical evidence rows**.

---

### 8.3. Phase 2A — Evidence Intelligence `[IMPLEMENTED]`
Phase 2A adds statistical observation confidence, model consensus, freshness, and dual ranking as query-time intelligence *(detailed in [`docs/08.3`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08.3-FEATURE_RECOMMENDATION_PHASE_2A_EVIDENCE_INTELLIGENCE.md))*:

1. **Evidence Confidence ($C$)**:
   $$C = \sqrt{C_{\text{runs}} \times C_{\text{models}}} = \sqrt{\left(1 - e^{-N_{\text{runs}}/3.0}\right) \times \left(1 - e^{-M_{\text{unique}}/2.0}\right)}$$
   - $N=1, M=1 \implies C \approx 33.4\%$ | $N=2, M=2 \implies C \approx 55.5\%$ | $N=3, M=3 \implies C \approx 70.1\%$
2. **Model Consensus & Strict Tie Contract**: Evaluates the latest vote per unique model package. 50/50 splits or 3-way deadlocks are classified as **`SPLIT (50%)`** with `is_consensus_tie = True`.
3. **Freshness Bands**: $\le 7\text{d} \implies \text{Fresh}$, $8\text{–}30\text{d} \implies \text{Recent}$, $> 30\text{d} \implies \text{Stale}$.
4. **Dual Ranking System**:
   - **`priority_rank` (Phase 1 Authoritative)**: Primary sorting order: `evidence_score DESC, keep_runs DESC, feature_name ASC`.
   - **`operational_priority_score`**: $\text{round}(\text{evidence\_score} \times C, 2)$.
   - **`advisory_rank` (Phase 2A Preview)**: Sorted by `operational_priority_score DESC, keep_runs DESC, feature_name ASC`.

---

### 8.4. Phase 2B — Stability, Risk & Generalization `[IMPLEMENTED]`
Phase 2B implements query-time behavioral stability, score range spread, trajectory reversals, Level-1 cross-context generalization, and explicit multi-dimensional risk badges *(detailed in [`docs/08.4`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08.4-FEATURE_RECOMMENDATION_PHASE_2B_STABILITY_RISK_GENERALIZATION.md))*:

1. **Score Volatility ($\sigma_S$) & Trajectory Spread ($N \ge 3$)**:
   $$\sigma_S = \sqrt{\frac{1}{N - 1} \sum_{t=1}^N (S_t - \bar{S})^2} \quad (\sigma_S < 15.0 \implies \text{Stable},\ 15.0 \le \sigma_S < 35.0 \implies \text{Moderate},\ \sigma_S \ge 35.0 \implies \text{Volatile})$$
   - $N < 3 \implies \text{None}$ (`⚪ N/A (< 3 runs)`).
   - Tracks score range spread $\Delta S = \max(S_t) - \min(S_t)$ and trajectory direction flips $D_{\text{flips}}$.
2. **Level-1 Cross-Context Generalization ($K \ge 2$)**:
   $$G = A_{\text{context}} \times \left(1.0 - \min\left(1.0, \frac{\Delta S_{\text{context}}}{100.0}\right)\right)$$
   - Evaluated across matching market, window, and project dimensions differing only by sampling interval.
   - $G \ge 0.75 \implies \text{Universal}$, $0.50 \le G < 0.75 \implies \text{Scale-Robust}$, $0.25 \le G < 0.50 \implies \text{Scale-Sensitive}$, $G < 0.25 \implies \text{Scale-Specific}$. Single context displays `⚪ Single Context`.
3. **Explicit Multi-Dimensional Risk Badges**:
   - `[DEGRADED]` (Score $\le -40.0$), `[SPLIT]` (`is_consensus_tie`), `[STALE]` ($> 30\text{d}$), `[UNSTABLE]` ($\sigma_S \ge 35.0$).
   - Composite scalar risk score was explicitly rejected to maintain transparency.

---

## 9. Feature Recommendation Evidence Studio UI (5 Tabs)

Located in [`apps/master_dataset_tk/feature_recommendation_viewer.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_recommendation_viewer.py):

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

### Treeview Columns Implemented by Tab:

| Tab | Exact Treeview Columns |
|---|---|
| **Tab 1: Feature Registry** | `Feature Name`, `Runs`, `Models`, `Score`, `Status`, `Confidence`, `Model Consensus`, `Freshness`, `Stability`, `Generalization`, `Badges`, `Last Rec`, `Last Model` |
| **Tab 2: Base Pipeline** | `Priority Rank`, `Feature Name`, `Runs`, `Models`, `Score`, `Confidence`, `Adj Score`, `Advisory Rank`, `Model Consensus`, `Freshness`, `Stability`, `Generalization`, `Badges`, `Status`, `Last Rec`, `Last Model` |
| **Tab 3: Selected Experimental** | `Lineage Status`, `Context Status`, `Feature Name`, `Pipeline ID`, `Snapshot`, `Runs`, `Models`, `Streak`, `Score`, `Confidence`, `Consensus`, `Freshness`, `Stability`, `Generalization`, `Badges` |
| **Tab 4: Raw Evidence Log** | `Evidence ID`, `Context ID`, `Feature`, `Source`, `Pipeline ID`, `Snapshot`, `Run ID`, `Model`, `Rec`, `Holdout Rank`, `Unseen Rank`, `ΔR`, `Imp Drop`, `Drift`, `Timestamp` |
| **Tab 5: Policy Settings** | Interactive config editor for Scoring Weights, Experimental Thresholds, Base Pipeline Gating, Version History, and Rollback controls |

---

## 10. Pre-Training Candidate Elimination Gate & Evidence-Driven Parent Selection

In Auto Candidate Generation ([`apps/master_dataset_tk/auto_candidate_generation.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/auto_candidate_generation.py)):
- **Evidence-Driven Parent Ranking**: Consumes `rank_features_for_candidate_generation()` from the Phase 3A Decision Engine (`training_decision_engine.py`) to rank potential interaction parents by their authoritative standing (`PROMOTION_CANDIDATE_QUALIFIED`, `TRAIN_CANDIDATE`, `NEW_UNSEEN`, `REVIEW`, `EXCLUDE`).
- **Domain-Stratified Quota Allocation**: Allocates up to 36 interaction parents across canonical Feature Registry domains with deterministic round-robin redistribution.
- **Strict Deprecation & Exclusion Gating**: Queries `evaluate_candidate_training_eligibility()` and `query_blocked_candidates(conn, context_id)` prior to feature transformation materialization.
- **Commutative Deduplication**: Prunes duplicate symmetric pairs ($A \le B$) for commutative operators (`multiply`, `add`, `absdiff`, `min`, `max`) while preserving directional pairs for asymmetric operators (`divide`, `subtract`).
- Automatically discards blocked and deprecated candidates, preventing expensive parquet generation and model training on degraded features.

---

## 11. Latest Verified Production Facts & Mathematical Reconciliation

Verified through read-only mathematical consistency audits against the SQLite Evidence DB:

### 11.1. Primary Context (NIFTY 3s standard all)
- **Total Distinct Features**: **`583`** features (110 Registry, 89 Base Pipeline, 384 Experimental)
- **Total Evidence Rows**: **`848`** rows
- **Run Distribution**: $N = 1: \mathbf{373}$ ($C = 33.4\%$) | $N = 2: \mathbf{155}$ ($C = 55.5\%$) | $N = 3: \mathbf{55}$ ($C = 70.1\%$)
- **Model Consensus Distribution**: `KEEP`: 383 | `WATCH`: 75 | `REMOVE`: 38 | `SPLIT/TIE`: 87
- **Phase 2B Stability ($N \ge 3$)**: 45 Moderate ($15 \le \sigma_S < 35$), 10 Volatile ($\sigma_S \ge 35$), 35 range $[20, 50)$, 20 range $[50, 100)$, 20 flip features
- **Phase 2B Generalization ($K \ge 2$)**: 89 multi-context features (49 Universal, 25 Scale-Robust, 10 Scale-Sensitive, 5 Scale-Specific)

### 11.2. Global Multi-Context Verification
- **Total Global Rows**: **`987`** rows
- **DB Checksum Verification**: 100% SHA-256 identical before and after queries (`1f977494f901a915f0e74348585af1ad1d43164baed41aa19cd1fba55227425b`).
- **Schema Alterations**: Zero migrations, zero schema modifications.

---

## 12. Safety & Governance Invariants Summary

1. **Evidence Immutability**: Historical validation events are permanent facts.
2. **Deterministic Projections**: Projections can be rebuilt identically from raw evidence.
3. **Registry Immunity**: Registry features cannot be automatically deleted or blocked.
4. **Base Pipeline Immunity**: Base Pipeline features cannot be automatically deleted or blocked.
5. **Experimental Context Blocking**: Repeated REMOVEs prevent candidate re-generation in Auto Candidate Generation.
6. **Promotion Candidate Governance**: `PROMOTION_CANDIDATE` is an eligibility flag for human review, not automated code insertion.
7. **Query-Time Intelligence**: Phase 2A and Phase 2B metrics compute dynamically without database alterations.
8. **Phase 1 Priority Authority**: Phase 1 `priority_rank` remains authoritative; `advisory_rank` is separate.
9. **Context Isolation**: Cross-market and cross-interval evaluations remain strictly isolated.

---

## 13. Current Implementation Status

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          CURRENT IMPLEMENTATION STATUS                                 │
├────────────────────────────────────────────────────────┬───────────────────────────────┤
│ Phase 1 (Evidence DB, Policy Settings & Gating)        │ ✅ IMPLEMENTED & VERIFIED     │
│ Phase 2A (Evidence Intelligence & Dual Ranking)        │ ✅ IMPLEMENTED & VERIFIED     │
│ Phase 2B (Stability, Risk Badges & Level-1 Gen Index)  │ ✅ IMPLEMENTED & VERIFIED     │
│ Phase 3A (Recommendation-to-Training Decision Engine)   │ ✅ IMPLEMENTED & VERIFIED     │
│ Phase 3B (Evidence Studio UI & Reason Inspector)       │ ✅ IMPLEMENTED & VERIFIED     │
│ Phase 3C (Model Builder Handoff & Provenance Tracking) │ ✅ IMPLEMENTED & VERIFIED     │
│ Phase 3D (Promotion, Graduation & Deprecation Gov)     │ ✅ IMPLEMENTED & VERIFIED     │
└────────────────────────────────────────────────────────┴───────────────────────────────┘
```

---

## 14. Phase 3 & Phase 3D Architecture: Decision Engine & Human Governance `[IMPLEMENTED & VERIFIED]`

> [!NOTE]
> **Phase 3 & Phase 3D Scope**: The Recommendation-to-Training Decision Engine and Promotion/Graduation Governance subsystems are fully implemented and verified across the codebase with **210/210 passing tests**.
> 
> - **Phase 3A/3B/3C**: Fully documented in [`docs/08.5-RECOMMENDATION_TO_TRAINING_DECISION_ENGINE.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08.5-RECOMMENDATION_TO_TRAINING_DECISION_ENGINE.md). Provides context-scoped 4-state qualification (`TRAIN_CANDIDATE`, `REVIEW`, `NEW_UNSEEN`, `EXCLUDE`), Reason Inspector UI, and Model Builder preset export (`save_feature_preset()`).
> - **Phase 3D**: Fully documented in [`docs/08.6-FEATURE_PROMOTION_GRADUATION_GOVERNANCE.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/08.6-FEATURE_PROMOTION_GRADUATION_GOVERNANCE.md). Provides evidence dossier compilation, human governance review dialog, atomic Registry Graduation (`FRxxxx`), Base Pipeline Promotion (`PL_0001`), and feature deprecation governance.
