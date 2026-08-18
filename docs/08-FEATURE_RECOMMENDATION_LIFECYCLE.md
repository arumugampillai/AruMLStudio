# AruMLStudio Feature Recommendation Lifecycle — Current Implementation

---

## 1. Executive Summary & Purpose

This document provides the authoritative technical reference for the **Feature Recommendation Lifecycle** in **AruMLStudio**.

It details how **Production Validation** and **Feature Studio** evaluate trained machine learning models against out-of-sample unseen market data, synthesize multi-signal feature validation metrics (**KEEP**, **WATCH**, **REMOVE**), automatically persist immutable validation evidence into the canonical SQLite database (`feature_recommendation_evidence.db`), and materialize dual derived projections for dataset-context operational state and experimental lineage evaluation.

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
                     ▼                               ▼
         Registry & Base Health /         Experimental Streak / Score /
        Context Pre-Training Gate             Promotion Candidate
                     │                               │
                     └───────────────┬───────────────┘
                                     ▼
                  Feature Recommendation Evidence Studio
```

> [!IMPORTANT]
> **Part A — Current Implementation**: This document strictly describes the **current codebase**.
> - The authoritative persistence store is the SQLite Evidence Database: `feature_recommendation_evidence.db`.
> - The legacy JSON file `feature_recommendation_history.json` is **historical migration source data only** and is **no longer the active store**.
> - Recommendations and scores serve as **observational evidence, pre-training candidate filtering, and human audit history**.
> - In the current architecture, there is **no automatic Feature Registry deletion**, **no automatic Base Pipeline deletion**, and **no automatic Experimental → Base promotion**.

---

## 2. Feature Studio & Production Validation Architecture

The **Feature Studio** ([`apps/master_dataset_tk/feature_studio_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_studio_panel.py)) and **Production Validation Panel** ([`apps/master_dataset_tk/production_validation_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/production_validation_panel.py)) provide comprehensive model diagnostics and out-of-sample forward testing.

```
                                  FEATURE STUDIO
                               (FeatureStudioPanel)
                                        │
      ┌─────────────┬─────────────┬─────┴───────┬─────────────┬─────────────┬─────────────┐
      ▼             ▼             ▼             ▼             ▼             ▼             ▼
 [Importance] [Distribution]   [Drift]   [Studio Compare] [Diagnostics]  [Planner]  [Production]
  (Native/SHAP) (Moments)   (WF vs. HO)  (Model Deltas)  (Root Cause)   (Audit)     [Validation]
```

### 2.1. Holdout Validation vs. True Unseen Validation
- **Diagnostics Studio**: Benchmarks **Walk-Forward Training Days vs. Holdout Days** to identify training overfitting, collinearity, or localized feature drift.
- **Production Validation**: Benchmarks **Holdout Days vs. True Unseen Forward Days** never seen during hyperparameter tuning or training. Its purpose is to verify if predictive utility collapses under live forward market dynamics.

---

## 3. The Three Feature Populations

Every feature entering Production Validation is partitioned into exactly one of three distinct populations with strictly defined lifecycles:

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

### 3.1. Feature Registry Features
- **Origin**: Canonical features materialized through the Master Dataset catalog. Scoped by `feature_project_id`.
- **Validation Role**: Receives `KEEP` / `WATCH` / `REMOVE` recommendations.
- **Invariants**: REMOVE recommendations raise observational alerts for registry curators. Registry features are **never automatically retired, deleted, or blocked** from model training.

### 3.2. Base Pipeline Features
- **Origin**: Already approved and merged transformation pipeline features (`base_pipeline_export_features`).
- **Validation Role**: Participates in evidence accumulation, evidence scoring, and ranking health.
- **Invariants**: Repeated REMOVE recommendations decrease ranking and flag health warnings (e.g. `alert` status when score $\le -40.0$). Base Pipeline features are **never automatically deleted or blocked**.

### 3.3. Selected Experimental Pipeline Features
- **Origin**: Experimental candidate features generated and evaluated within a specific pipeline lineage (`pipeline_id` + `pipeline_snapshot_id`).
- **Validation Role**: Evaluated through exact lineage tracking. Tracks `KEEP` streaks, `REMOVE` streaks, and lineage evidence scores.
- **Invariants**:
  - Consecutive REMOVE recommendations trigger context-level candidate blocking in pre-training candidate generation.
  - Consistent KEEP performance unlocks **`PROMOTION_CANDIDATE`** eligibility for human architectural review.
  - There is **no automatic promotion** from Experimental to Base Pipeline without explicit human review and code integration.

---

## 4. Unseen Dataset Resolution & Deterministic Lineage

The module [`apps/chain_replay_ml/production_validation/unseen_dataset.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/unseen_dataset.py) ensures forward test datasets perfectly reproduce the exact feature transformations of the parent model.

```
                             TRAINED MODEL PACKAGE
                             models/<model_name>/
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     ▼                                 ▼                                 ▼
feature_project_id                pipeline_id                   pipeline_snapshot_id
 (e.g. "all")                    (e.g. "PL_0005")               ("snap_v1")
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

$$\text{identity\_hash} = \text{SHA256}\Big(\text{master\_db} + \text{unseen\_days} + \text{feature\_project\_id} + \text{pipeline\_id} + \text{pipeline\_snapshot\_id} + \text{flags}\Big)[:8]$$

---

## 5. Feature Validation Decision Logic (`recommend_feature`)

Located in [`apps/chain_replay_ml/production_validation/rules.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/rules.py):

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

| Recommendation | Criteria | Interpretation |
|---|---|---|
| **`REMOVE`** | Severe Rank Drop ($\Delta R \le -5$) **AND** Severe Importance Drop ($\text{RelDrop} \ge 50\%$) **AND** Distribution Drift ($\ge 1$). | Predictive power collapsed on forward data alongside significant distribution drift. |
| **`WATCH`** | Medium Rank Drop ($\Delta R \le -2$) **OR** Medium Importance Drop ($\text{RelDrop} \ge 25\%$) **OR** Moderate Drift. | Partial degradation or distribution instability between Holdout and Unseen regimes. |
| **`KEEP`** | Stable Rank ($|\Delta R| \le 1$) and stable importance across regimes. | Feature maintained robust predictive utility on true unseen forward trading days. |

---

## 6. SQLite Evidence Database Architecture

Authoritative persistence is managed by [`apps/chain_replay_ml/production_validation/evidence_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/evidence_store.py) in the SQLite database:
$$\text{Path: }\mathtt{<chart\_data\_dir>/feature\_recommendation\_evidence.db}$$

```
                                SQLite EVIDENCE STORE
                         (feature_recommendation_evidence.db)
                                          │
    ┌───────────────────────────┬─────────┴─────────┬───────────────────────────┐
    ▼                           ▼                   ▼                           ▼
dataset_contexts     recommendation_evidence  feature_context_summary  experimental_lineage_summary
(Regime Metadata)     (Immutable Raw Log)       (Context Projection)        (Lineage Projection)
```

### 6.1. Schema DDL

#### Table 1: `dataset_contexts` (Dataset Regime Identification)
```sql
CREATE TABLE IF NOT EXISTS dataset_contexts (
    context_id TEXT PRIMARY KEY,               -- e.g. 'ctx_574ee67348f2'
    market TEXT NOT NULL,                      -- e.g. 'NIFTY', 'BANKNIFTY', 'SENSEX'
    sampling_interval_sec INTEGER NOT NULL,    -- e.g. 3, 1, 6
    sampling_label TEXT,                       -- e.g. '3s', '6s'
    sliding_window TEXT NOT NULL,              -- e.g. 'standard', 'atm_15'
    feature_project_id TEXT NOT NULL,          -- e.g. 'all', 'chart'
    context_key TEXT NOT NULL UNIQUE,          -- 'NIFTY:3:standard:all'
    created_at TEXT NOT NULL
);
```

#### Table 2: `recommendation_evidence` (Authoritative Immutable Event Log)
```sql
CREATE TABLE IF NOT EXISTS recommendation_evidence (
    evidence_id TEXT PRIMARY KEY,              -- 'ev_{run_id}_{safe_model_name}_{feature_name}'
    context_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_source TEXT NOT NULL,              -- 'registry', 'base_pipeline', 'experimental'
    pipeline_id TEXT,                          -- NULL for registry/base
    pipeline_snapshot_id TEXT,                 -- NULL for registry/base
    recommendation TEXT NOT NULL,              -- 'KEEP', 'WATCH', 'REMOVE'
    validation_run_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    target_column TEXT,
    holdout_rank INTEGER,
    unseen_rank INTEGER,
    rank_change INTEGER,
    relative_imp_drop REAL,
    drift_severity INTEGER,
    evidence_detail_json TEXT,
    run_timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (context_id) REFERENCES dataset_contexts(context_id)
);
```

#### Table 3: `feature_context_summary` (Context-Level Materialized Projection)
```sql
CREATE TABLE IF NOT EXISTS feature_context_summary (
    context_id TEXT NOT NULL,
    feature_source TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    total_runs INTEGER NOT NULL DEFAULT 0,
    unique_models_count INTEGER NOT NULL DEFAULT 0,
    keep_runs INTEGER NOT NULL DEFAULT 0,
    watch_runs INTEGER NOT NULL DEFAULT 0,
    remove_runs INTEGER NOT NULL DEFAULT 0,
    last_recommendation TEXT,
    last_run_id TEXT,
    last_model_name TEXT,
    last_run_timestamp TEXT,
    current_streak_type TEXT,                  -- 'KEEP', 'WATCH', 'REMOVE'
    current_streak_count INTEGER DEFAULT 0,
    evidence_score REAL DEFAULT 0.0,
    lifecycle_status TEXT NOT NULL DEFAULT 'active', -- 'active', 'held', 'blocked', 'alert'
    updated_at TEXT NOT NULL,
    PRIMARY KEY (context_id, feature_source, feature_name),
    FOREIGN KEY (context_id) REFERENCES dataset_contexts(context_id)
);
```

#### Table 4: `experimental_lineage_summary` (Lineage-Specific Promotion Projection)
```sql
CREATE TABLE IF NOT EXISTS experimental_lineage_summary (
    context_id TEXT NOT NULL,
    pipeline_id TEXT NOT NULL,
    pipeline_snapshot_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_identity_key TEXT NOT NULL,        -- 'exp:{feature}:{pipeline_id}:{snapshot_id}'
    total_runs INTEGER NOT NULL DEFAULT 0,
    unique_models_count INTEGER NOT NULL DEFAULT 0,
    keep_runs INTEGER NOT NULL DEFAULT 0,
    watch_runs INTEGER NOT NULL DEFAULT 0,
    remove_runs INTEGER NOT NULL DEFAULT 0,
    last_recommendation TEXT,
    last_run_id TEXT,
    last_model_name TEXT,
    last_run_timestamp TEXT,
    current_streak_type TEXT,
    current_streak_count INTEGER DEFAULT 0,
    evidence_score REAL DEFAULT 0.0,
    lifecycle_status TEXT NOT NULL DEFAULT 'experimental_eval', -- 'experimental_eval', 'promotion_candidate', 'held', 'blocked'
    updated_at TEXT NOT NULL,
    PRIMARY KEY (context_id, pipeline_id, pipeline_snapshot_id, feature_name),
    FOREIGN KEY (context_id) REFERENCES dataset_contexts(context_id)
);
```

#### Table 5: `migration_meta` (Migration Audit Tracking)
```sql
CREATE TABLE IF NOT EXISTS migration_meta (
    meta_key TEXT PRIMARY KEY,                 -- e.g. 'json_migration_completed'
    meta_value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

---

## 7. Dual-Projection Architecture: Why Two Summaries?

A single aggregation table cannot simultaneously serve both context-wide pre-training candidate blocking and lineage-specific experimental promotion.

```
                 recommendation_evidence
                   IMMUTABLE RAW LOG
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
     feature_context_summary   experimental_lineage_summary
       Context Projection        Lineage Projection
              │                       │
              ▼                       ▼
      Registry / Base /        Experimental lifecycle
      context state             and promotion eligibility
```

1. **`feature_context_summary` (Aggregated by `context_id + feature_source + feature_name`)**:
   - Evaluates broad feature performance across models in a dataset regime.
   - Drives candidate blocking for Auto Candidate Generation.
   - Monitors Base Pipeline and Registry health scores.
2. **`experimental_lineage_summary` (Aggregated by `context_id + pipeline_id + pipeline_snapshot_id + feature_name`)**:
   - Preserves exact algorithmic provenance. An experimental transformation that fails in Snapshot A might succeed with altered parameters in Snapshot B.
   - Tracks consecutive KEEP streaks and multi-model consistency to qualify experimental features as **`PROMOTION_CANDIDATE`**.

Both projections are **purely deterministic and rebuildable** from `recommendation_evidence` at any time via `rebuild_all_projections()`.

---

## 8. Raw Evidence Events vs. Projection Summary Rows

During real-world verification of production models, a fundamental architectural distinction exists between **raw validation events** and **distinct feature summaries**:

### Example: Production Model Package `Future_LTP_5m_WF_1168f_XGB_2243_14`
1. **Raw Evidence Log (`recommendation_evidence`)**:
   - Current Model Validation: **583 events** (110 Registry, 89 Base Pipeline, 384 Experimental)
   - Migrated Historical Model (`Future_LTP_5m_WF_232f_XGB_1539_2`): **232 events** (141 Registry, 91 Base Pipeline)
   - **Total Raw Immutable Records**: $583 + 232 = \mathbf{815}$ rows.
2. **Context Materialized Projection (`feature_context_summary`)**:
   - Feature Registry: **183 distinct features**
   - Base Pipeline: **138 distinct features**
   - Selected Experimental: **384 distinct features**
   - **Total Materialized Summary Rows**: $183 + 138 + 384 = \mathbf{705}$ rows.

### Mathematical Proof of Deduplication:
$$\begin{aligned}
\text{Shared Registry Features across both models} &= 68 \\
\text{Shared Base Pipeline Features across both models} &= 42 \\
\text{Total Multi-Model Re-evaluations} &= 68 + 42 = \mathbf{110} \text{ events} \\
\text{Total Raw Events (815)} - \text{Shared Events (110)} &= \mathbf{705} \text{ Distinct Feature Summaries}
\end{aligned}$$

Every one of the 815 raw events is stored and correctly accumulated into the `total_runs`, `unique_models_count`, streaks, and scores of the 705 summary rows.

---

## 9. Automatic Persistence Flow & Idempotency

### 9.1. Persistence Flow
Validation evidence persistence is decoupled from low-level compute algorithms and executed from the orchestration layer upon successful completion:

```
Production Validation Compute (run_production_validation_compute)
    │
    ▼ (Generates local comparison.json, summary.json, run_meta.json)
Production Validation Orchestration Layer (production_validation_panel.py)
    │
    ▼ (Invokes canonical persistence API)
persist_validation_evidence(data_dir, model_name)
    │
    ▼ (Partitions features via partition_diagnostic_rows)
append_validation_evidence(conn, context, evidence_rows, policy)
    │
    ├──▶ Writes to recommendation_evidence (ON CONFLICT DO NOTHING)
    ├──▶ Updates feature_context_summary
    └──▶ Updates experimental_lineage_summary
```

- **Fully Automatic Persistence**: Persistence occurs automatically in the background upon successful Production Validation compute (`_on_compute()`) and when cached validation results are loaded (`_apply_compute_payload()`).
- **User-Initiated Action**: The **"Persist Validation Evidence"** UI button allows users to manually persist/re-persist evidence or view persistence confirmation summaries on demand.

### 9.2. Idempotency Guarantees
- Every evidence record is assigned a deterministic primary key:
  $$\text{evidence\_id} = \text{ev\_}\{\text{run\_id}\}\_\{\text{safe\_model\_name}\}\_\{\text{feature\_name}\}$$
- If a user clicks **"Persist Validation Evidence"** or validation compute runs twice for the same run ID, SQLite enforces `ON CONFLICT(evidence_id) DO NOTHING`.
- Projection updates recalculate scores directly from the deduplicated evidence log, ensuring **zero duplicate records** and 100% stable scores.

---

## 10. Legacy JSON Migration

The legacy JSON store `feature_recommendation_history.json` is automatically and idempotently migrated into the SQLite evidence database by [`apps/chain_replay_ml/production_validation/recommendation_migration.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/recommendation_migration.py).

```
feature_recommendation_history.json (Legacy Historical File)
                            │
                            ▼
           migrate_legacy_recommendation_json(data_dir)
                            │
                            ▼
     • Recovers dataset context and model lineage
     • Classifies features into Registry / Base / Experimental
     • Inserts canonical records into recommendation_evidence
     • Materializes feature_context_summary & experimental_lineage_summary
     • Writes migration_meta['json_migration_completed'] = 'true'
```

- **Idempotency**: Once `json_migration_completed` is set, subsequent checks detect the flag and return immediately without re-migrating or altering data.
- **Audit Preservation**: Historical entries are preserved in full without data loss.

---

## 11. Dataset Context & Isolation

The evidence store isolates feature behavior across different market dynamics through `dataset_contexts`:

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

- Evidence from high-frequency $1\text{s}$ datasets never pollutes or blocks candidate features in swing or standard $3\text{s}$ datasets.
- Evidence from `SENSEX` never impacts `NIFTY` or `BANKNIFTY` models.

### `legacy_unknown` Isolation
Historical legacy records whose dataset metadata cannot be recovered are tagged with `context_id = "legacy_unknown"`. They remain accessible for human historical audit in the Evidence Studio via the **"Include Legacy Unknown"** filter, but are **strictly excluded** from active candidate blocking in Auto Candidate Generation.

---

## 12. Scoring Policy, Streaks & Lifecycle Thresholds

Implemented in [`apps/chain_replay_ml/production_validation/recommendation_policy.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/recommendation_policy.py):

### 12.1. Exact Evidence Scoring Formula (`compute_evidence_score`)
For any given feature across its chronological validation history within a dataset context:

$$\text{raw\_score} = (w_{\text{keep}} \cdot M_{\text{keep}}) + (w_{\text{remove}} \cdot M_{\text{remove}}) + (w_{\text{watch}} \cdot M_{\text{watch}}) + (B_{\text{keep}} \cdot S_{\text{keep}}) + (P_{\text{remove}} \cdot S_{\text{remove}})$$

$$\text{evidence\_score} = \text{round}\Big(\max(S_{\min}, \min(S_{\max}, \text{raw\_score})), 2\Big)$$

Where:
- $M_{\text{keep}}$ = Count of unique models where the feature received `KEEP`
- $M_{\text{remove}}$ = Count of unique models where the feature received `REMOVE`
- $M_{\text{watch}}$ = Count of unique models where the feature received `WATCH`
- $S_{\text{keep}}$ = Current consecutive `KEEP` streak count from the end of the run sequence
- $S_{\text{remove}}$ = Current consecutive `REMOVE` streak count from the end of the run sequence
- Configured Policy Weights:
  - $w_{\text{keep}} = +25.0$
  - $w_{\text{remove}} = -35.0$
  - $w_{\text{watch}} = -10.0$
  - $B_{\text{keep}} = +15.0$ (Streak Bonus per consecutive KEEP)
  - $P_{\text{remove}} = -25.0$ (Streak Penalty per consecutive REMOVE)
  - $[S_{\min}, S_{\max}] = [-100.0, +100.0]$

### 12.2. Experimental Lineage Promotion-Candidate Rule
An experimental feature is classified as **`PROMOTION_CANDIDATE`** in `experimental_lineage_summary` if and only if all three conditions are satisfied:
1. $\text{Current Streak Type} == \text{"KEEP"}$ with $S_{\text{keep}} \ge \mathtt{promotion\_candidate\_consecutive\_keep}\text{ (default: 3)}$.
2. Evaluated across at least $M_{\text{unique}} \ge \mathtt{min\_unique\_models}\text{ (default: 2)}$ unique model packages.
3. $\text{Lineage Evidence Score} \ge \mathtt{promotion\_candidate\_min\_score}\text{ (default: 75.0)}$.

> [!NOTE]
> `PROMOTION_CANDIDATE` is an **eligibility status for human review**. There is **no automated code generation or automatic Base Pipeline insertion**.

### 12.3. Pre-Training Candidate Elimination Gate & Thresholds
In Auto Candidate Generation ([`apps/chain_replay_ml/dataset_builder/auto_candidate_generator.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/auto_candidate_generator.py)):
- Queries `query_blocked_candidates(conn, context_id)` before feature materialization.
- In `feature_context_summary`, an Experimental feature transitions to `lifecycle_status = "blocked"` if:
  $$S_{\text{remove}} \ge \mathtt{remove\_block\_consecutive\_threshold}\text{ (default: 2)}$$
  $$\mathbf{OR}\quad \text{remove\_runs} \ge \mathtt{remove\_block\_total\_threshold}\text{ (default: 4)}$$
- **Immunity Invariant**: Feature Registry and Base Pipeline features are **never blocked** (their status transitions to `alert` or `held`, but never `blocked`).

---

## 13. Feature Recommendation Evidence Studio UI

Accessible via the **"Evidence DB & Projections"** button in Production Validation or Feature Studio ([`apps/master_dataset_tk/feature_recommendation_viewer.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_recommendation_viewer.py)):

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Feature Recommendation Evidence Studio                                                            │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Context: [Market: NIFTY ▼] [Interval: 3 ▼] [Window: standard ▼] [Project: all ▼] [x] Legacy  │
│ Context ID: ctx_574ee67348f2                 [Rebuild Projections]  [Refresh Data]                │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│ [1. Feature Registry (183)] [2. Base Pipeline (138)] [3. Selected Exp (384)] [4. Raw Log (815)]   │
│                                                                                                   │
│ Feature Name    │ Runs │ Models │ Streak  │ Score  │ Status   │ Last Rec │ Last Model             │
│─────────────────┼──────┼────────┼─────────┼────────┼──────────┼──────────┼────────────────────────│
│ atm_pcr_chg_5m  │ 2    │ 2      │ KEEP 2  │ 65.0   │ active   │ KEEP     │ Future_LTP_5m_WF_...   │
│ atm_iv_ce       │ 2    │ 2      │ REMOVE 2│ -60.0  │ alert    │ REMOVE   │ Future_LTP_5m_WF_...   │
│ ...             │      │        │         │        │          │          │                        │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### UI Controls & Tabs:
1. **Context Filters**: Dynamic filtering by `Market`, `Interval`, `Sliding Window`, `Project ID`, and `Include Legacy Unknown`. Automatically initialized to the active model's dataset context.
2. **Rebuild Projections**: Recomputes all summary projections directly from the immutable `recommendation_evidence` log.
3. **Tab 1 — Feature Registry**: Displays registry health, multi-model evaluation counts, accumulated scores, and alert flags.
4. **Tab 2 — Base Pipeline**: Displays Base Pipeline scores, stability rankings, and degradation alerts.
5. **Tab 3 — Selected Experimental**: Displays lineage-specific features, exact `pipeline_id` + `pipeline_snapshot_id`, streak counters, and `PROMOTION_CANDIDATE` badges.
6. **Tab 4 — Raw Evidence Log**: Chronological audit trail of all individual validation events.

---

## 14. Current End-to-End Recommendation Lifecycle Diagram

```
Master Dataset Catalog ∪ Approved Base Pipeline ∪ Experimental Candidate Pipeline
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
                      persist_validation_evidence()
                      ("Persist Validation Evidence")
                                    │
                                    ▼
                         recommendation_evidence
                        (SQLite Immutable Event Log)
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
          feature_context_summary       experimental_lineage_summary
            (Context Projection)             (Lineage Projection)
                     │                             │
                     ▼                             ▼
         Auto Candidate Gate /         Promotion Candidate Review /
         Registry & Base Health           Human Governance Audit
```

---

## 15. Invariants & Current Limitations

1. **No Automatic Registry Deletion**: Feature Registry features are never removed, retired, or blocked by validation runs.
2. **No Automatic Base Pipeline Removal**: Base Pipeline features are never removed or blocked automatically.
3. **No Automatic Experimental Promotion**: `PROMOTION_CANDIDATE` status provides human governance eligibility only. Promotion requires architectural review and manual pipeline promotion.
4. **Idempotent Persistence**: Persisting a validation run multiple times creates zero duplicate rows.
5. **Authoritative Evidence Store**: `feature_recommendation_evidence.db` is the sole authoritative store. `feature_recommendation_history.json` is strictly legacy migration source data.
6. **Context Isolation**: Candidate blocking and evidence scoring are strictly partitioned by `context_id`.

---

## 16. Part B — Future Lifecycle Design (Placeholder)

> [!NOTE]
> Advanced automated lifecycle capabilities (such as automated HCA redundancy pruning, mutual information clustering, autonomous pipeline promotion code synthesis, and multi-regime demotion workflows) are intentionally preserved for future architectural design phases and are not part of the active operational codebase.
