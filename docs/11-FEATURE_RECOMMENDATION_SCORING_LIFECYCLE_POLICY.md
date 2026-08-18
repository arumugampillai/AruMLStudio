# AruMLStudio Feature Recommendation Scoring & Lifecycle Policy

---

## 1. Purpose

This document defines how **AruMLStudio** converts accumulated Production Validation evidence into feature health, scoring, ranking, blocking, and Experimental promotion decisions.

The policy operates on the Recommendation Evidence architecture:

```
    Production Validation
            │
            ▼
    KEEP / WATCH / REMOVE
            │
            ▼
    recommendation_evidence
            │
       ┌────┴────┐
       ▼         ▼
feature_context_summary
       │
       └──────────────► experimental_lineage_summary
                              │
                              ▼
                    Lifecycle Policy
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
          Registry           Base        Experimental
           Health          Priority        Promotion
             │                │                │
             ▼                ▼                ▼
           ALERT           RANKING       PROMOTION_CANDIDATE
                                              │
                                              ▼
                                         Human Review
```

This policy does not replace Feature Analysis Lab.

- **Feature Analysis answers**:
  $$\text{"Which features should be selected for this model?"}$$
- **Production Validation answers**:
  $$\text{"How did those features perform on true unseen data?"}$$
- **This policy answers**:
  $$\text{"What does the accumulated validation evidence mean for the future lifecycle of this feature?"}$$

---

## 2. Architectural Principles

### 2.1. Evidence is Immutable
`recommendation_evidence` is the authoritative historical evidence log. Every Production Validation evaluation represents an independent evidence event.

The policy must never overwrite historical evidence.

```
    Evidence Event
          │
          ▼
    Immutable Evidence Log
          │
          ▼
    Materialized Projections
          │
          ▼
    Current Lifecycle State
```

### 2.2. Dataset Context is the Primary Boundary
All recommendation calculations are scoped to a **Dataset Context**.

The context is determined by:
- `market` (e.g. `NIFTY`, `BANKNIFTY`, `SENSEX`)
- `sampling_interval_sec` (e.g. `3`, `1`, `6`)
- `sliding_window` (e.g. `standard`, `atm_15`)
- `feature_project_id` (e.g. `all`, `chart`)

$$\text{context\_id} = \mathtt{"ctx\_"} + \text{SHA256}(\text{market} + \text{interval} + \text{window} + \text{project})[:12]$$

Example:
- `NIFTY:3:standard:all` ($\mathtt{ctx\_574ee67348f2}$) must not be mixed with `NIFTY:15:atm_15:all` or `SENSEX:3:standard:all`.

A feature's evidence in one context must not automatically alter its lifecycle state in another context.

### 2.3. Feature Population Determines Lifecycle Behavior
A single scoring model must not be used to make identical decisions for all feature populations. The three populations have different responsibilities:
- **Feature Registry**: Canonical feature health
- **Base Pipeline**: Accepted pipeline feature priority and health
- **Experimental Pipeline**: Experimental lineage validation and promotion eligibility

---

## 3. Three Feature Lifecycle Policies

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. FEATURE REGISTRY                                                                         │
│ • Canonical features materialized through Master Dataset                                     │
│ • Foundation inputs: NEVER automatically blocked, retired, or deleted                       │
│ • Recommendation evidence is observational for health monitoring, alerts, and audit history │
│ • Persistent negative validation produces ALERT condition, not retirement                   │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. BASE PIPELINE                                                                            │
│ • Already accepted into the Base Pipeline (base_pipeline_export_features)                   │
│ • Accumulates evidence across runs and models to receive priority/ranking                   │
│ • Can become WATCH / ALERT candidates based on evidence score                               │
│ • NEVER automatically blocked by the candidate gate and NEVER automatically deleted         │
│ • REMOVE result means: "Negative evidence accumulated", NOT "Delete from Base Pipeline"     │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. EXPERIMENTAL PIPELINE                                                                    │
│ • Candidate features generated through experimental pipeline workflows                      │
│ • Objective: Has exact lineage demonstrated enough unseen-data evidence for promotion?      │
│ • Lineage-specific key: context_id + pipeline_id + pipeline_snapshot_id + feature_name      │
│ • Different pipeline snapshots must NOT have evidence merged for promotion decisions        │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Recommendation Evidence Schema

Every validation event produces one recommendation (`KEEP`, `WATCH`, `REMOVE`).

The raw evidence record in `recommendation_evidence` contains:
- `evidence_id`: Deterministic primary key (`ev_{run_id}_{safe_model_name}_{feature_name}`)
- `context_id`: Dataset regime context ID
- `feature_name`: Name of the evaluated feature
- `feature_source`: `'registry'`, `'base_pipeline'`, or `'experimental'`
- `pipeline_id`: Pipeline identifier (for experimental features)
- `pipeline_snapshot_id`: Pipeline snapshot hash (for experimental features)
- `model_name`: Trained model package name
- `validation_run_id`: Unique validation execution run ID
- `recommendation`: `'KEEP'`, `'WATCH'`, or `'REMOVE'`
- `holdout_rank`, `unseen_rank`, `rank_change`: Comparative rank metrics
- `relative_imp_drop`, `drift_severity`: Drift and degradation metrics
- `evidence_detail_json`: Full diagnostic payload
- `run_timestamp`: Execution timestamp

The raw evidence is **immutable** and is **never replaced** by a later summary.

---

## 5. Current Evidence Score

### 5.1. Purpose
The Evidence Score summarizes accumulated validation evidence across historical runs. The score is bounded:
$$-100.0 \le \text{evidence\_score} \le +100.0$$

### 5.2. Exact Implemented Formula (`compute_evidence_score`)
For any given feature across its chronological validation history within a dataset context:

$$\text{raw\_score} = (w_{\text{keep}} \cdot M_{\text{keep}}) + (w_{\text{remove}} \cdot M_{\text{remove}}) + (w_{\text{watch}} \cdot M_{\text{watch}}) + (B_{\text{keep}} \cdot S_{\text{keep}}) + (P_{\text{remove}} \cdot S_{\text{remove}})$$

$$\text{evidence\_score} = \text{round}\Big(\max(-100.0, \min(+100.0, \text{raw\_score})), 2\Big)$$

### 5.3. Configured Policy Weights
Implemented in [`apps/chain_replay_ml/production_validation/recommendation_policy.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/production_validation/recommendation_policy.py):

| Parameter | Symbol | Weight / Value | Description |
|---|---|---|---|
| `weight_keep` | $w_{\text{keep}}$ | **$+25.0$** | Base weight per unique model recommending KEEP |
| `weight_remove` | $w_{\text{remove}}$ | **$-35.0$** | Base weight per unique model recommending REMOVE |
| `weight_watch` | $w_{\text{watch}}$ | **$-10.0$** | Base weight per unique model recommending WATCH |
| `bonus_consecutive_keep` | $B_{\text{keep}}$ | **$+15.0$** | Multiplier per consecutive KEEP streak ($S_{\text{keep}}$) |
| `penalty_consecutive_remove` | $P_{\text{remove}}$ | **$-25.0$** | Multiplier per consecutive REMOVE streak ($S_{\text{remove}}$) |
| Score Bounds | $[S_{\min}, S_{\max}]$ | **$[-100.0, +100.0]$** | Clamped score boundary |

Where:
- $M_{\text{keep}}$ = Count of unique models where the feature received `KEEP`
- $M_{\text{remove}}$ = Count of unique models where the feature received `REMOVE`
- $M_{\text{watch}}$ = Count of unique models where the feature received `WATCH`
- $S_{\text{keep}}$ = Current consecutive `KEEP` streak count from the end of the chronological sequence
- $S_{\text{remove}}$ = Current consecutive `REMOVE` streak count from the end of the chronological sequence

---

## 6. Four Separate Architectural Concepts

To maintain strict conceptual integrity and prevent confusing historical evidence with operational actions, AruMLStudio explicitly defines four separate concepts:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. EVIDENCE SCORE                                                                           │
│ Definition: Historical validation strength calculated directly from the immutable log.      │
│ Current Implementation: Implemented formula bounded in [-100.0, +100.0].                    │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. EVIDENCE CONFIDENCE                                                                      │
│ Definition: Amount and diversity of independent evidence (sample size, unique models).      │
│ Current Implementation: Indicated by unique_models_count and total_runs.                     │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. RISK SCORE                                                                               │
│ Definition: Future operational risk assessment evaluating degradation trends & WATCH depth. │
│ Current Implementation: Conceptual roadmap item (FUTURE ENHANCEMENT).                       │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. LIFECYCLE STATE                                                                          │
│ Definition: Current operational status of a feature within a context.                       │
│ Current Implementation: 'active', 'held', 'blocked', 'promotion_candidate', 'alert'.        │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

The system does not overload the Evidence Score with every operational ranking consideration:
- **Evidence Score** = Historical validation strength
- **Lifecycle Priority** = Operational ranking derived from evidence

This separation allows operational rankings to evolve without modifying historical evidence.

---

## 7. KEEP / WATCH / REMOVE Semantics

```
                     RECOMMENDATION EVIDENCE SEMANTICS
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
       KEEP                        WATCH                       REMOVE
(Positive Evidence)         (Uncertainty/Caution)        (Negative Evidence)
         │                           │                           │
• Increases confidence       • Not equivalent to REMOVE   • Negative score contribution
• Drives Base Pipeline       • Preserves visibility       • Experimental: can trigger
  priority & Experimental    • Does NOT automatically       context-level candidate gate
  promotion eligibility        block candidates           • Registry & Base: health alert
```

### 7.1. KEEP
KEEP is positive evidence. Repeated KEEP results increase confidence that the feature remains useful in the evaluated context. Strong KEEP evidence is essential for:
- Base Pipeline priority ranking
- Experimental promotion candidate eligibility

### 7.2. WATCH
WATCH represents uncertainty or caution. It is **not** equivalent to REMOVE.

$$\text{Example: } \{\text{KEEP}: 5, \text{WATCH}: 4, \text{REMOVE}: 0\} \quad\text{is fundamentally healthier than}\quad \{\text{KEEP}: 1, \text{WATCH}: 0, \text{REMOVE}: 5\}$$

WATCH features remain visible and monitored. A WATCH result does **not** automatically block an Experimental feature.

### 7.3. REMOVE
REMOVE is negative validation evidence and contributes negatively to the Evidence Score.
- **Experimental features**: Repeated REMOVE evidence can activate the historical candidate blocking gate.
- **Base Pipeline & Registry features**: REMOVE creates health/review evidence but does **not** activate the candidate blocking mechanism.

---

## 8. Base Pipeline Scoring & Ranking

### 8.1. Objective
Base Pipeline features are never judged from a single validation run. Evidence accumulates over time:
$$\text{Run 1 (KEEP)} \longrightarrow \text{Run 2 (KEEP)} \longrightarrow \text{Run 3 (WATCH)} \longrightarrow \text{Run 4 (KEEP)} \longrightarrow \text{Run 5 (KEEP)}$$
This feature demonstrates substantially stronger evidence than a feature validated only once.

### 8.2. Current Implementation: Base Pipeline Priority Ranking
In the **current codebase**, Base Pipeline Priority Ranking is directly derived from the implemented **Evidence Score**:
1. Features in the Base Pipeline are sorted by `evidence_score` descending.
2. Ties are resolved using `total_runs`, `keep_streak`, and `unique_models_count`.
3. Higher Evidence Scores place features higher in the review order within Feature Studio.

### 8.3. Future Roadmap: Dedicated Base Pipeline Operational Priority Score
A **Dedicated Base Pipeline Operational Priority Score** is a future enhancement that will provide a multi-factorial composite layer (combining evidence confidence, recency, and operational stability).
> [!NOTE]
> This dedicated operational score is a **future analytical layer** and will **not** replace the underlying immutable Evidence Score.

### 8.4. Independent Model Evidence
Multiple distinct model packages provide stronger evidence than repeatedly evaluating the same model:

$$\text{Feature A: } 10 \text{ runs across } \mathbf{1\text{ model}} \quad<\quad \text{Feature B: } 7 \text{ runs across } \mathbf{4\text{ unique models}}$$

Therefore, `unique_models_count` is a first-class lifecycle signal.

### 8.5. Non-Mutation Invariant
$$\mathbf{Priority\ Rank} \ne \mathbf{Automatic\ Pipeline\ Mutation}$$
The ranking serves informational and review mechanisms. The Base Pipeline remains strictly protected from automatic deletion or blocking.

---

## 9. WATCH Priority & Risk-Oriented Review

WATCH is a monitoring state, not a failure state. WATCH analysis considers:
- WATCH frequency and recency
- Co-occurring REMOVE evidence
- Evidence Score deterioration
- Unique model distribution

```
Feature A: KEEP = 10, WATCH = 8, REMOVE = 0  ──► Low Risk (High positive volume)
Feature B: KEEP = 2,  WATCH = 2, REMOVE = 4  ──► High Risk (Negative dominant)
```

Feature B requires urgent review even though Feature A has more WATCH events. Future automated risk scoring will structure this review; in the current implementation, sorting by Evidence Score naturally penalizes features with co-occurring REMOVEs.

---

## 10. Experimental Lifecycle & Lineage Scoping

Experimental lifecycle is lineage-specific. The identity key is:

$$\text{feature\_identity\_key} = \mathtt{"exp:"} + \text{feature\_name} + \mathtt{":"} + \text{pipeline\_id} + \mathtt{":"} + \text{pipeline\_snapshot\_id}$$

Evidence from different pipeline snapshots is **never merged** when determining promotion eligibility.

---

## 11. Experimental Promotion Candidate Rule

An experimental feature is classified as **`PROMOTION_CANDIDATE`** in `experimental_lineage_summary` if and only if all three conditions are satisfied:

$$\begin{aligned}
1.&\quad S_{\text{keep}} \ge 3 \text{ (Consecutive KEEP streak)} \\
2.&\quad M_{\text{unique}} \ge 2 \text{ (Unique model packages)} \\
3.&\quad \text{evidence\_score} \ge 75.0
\end{aligned}$$

```
                consecutive_keeps >= 3  AND  unique_models >= 2  AND  score >= 75.0
                                              │
                                             YES
                                              │
                                              ▼
                             lifecycle_status = PROMOTION_CANDIDATE
                                              │
                                              ▼
                                    Human Review Queue
```

> [!IMPORTANT]
> **No Automatic Mutation**: `PROMOTION_CANDIDATE` is an eligibility status for human review. It does **not** automatically modify `pipeline_features_config.py` or insert code into the Base Pipeline.

---

## 12. Experimental Lifecycle States

```
    CANDIDATE (Initial Pipeline Proposal)
        │
        ▼
    VALIDATING (Participating in Production Validation)
        │
        ▼
    ACTIVE (Performing within acceptable thresholds)
        │
        ├──────────────► HELD (Under WATCH or ambiguous performance)
        │
        ▼
    PROMOTION_CANDIDATE (Satisfies 3 KEEPs, 2 models, score >= 75.0)
        │
        ▼
    HUMAN REVIEW (Architectural evaluation)
        │
        ├──────────────► REJECTED / HELD
        │
        ▼
    FUTURE BASE PROMOTION (Manual pipeline merge)
```

---

## 13. Experimental Blocking Gate

Experimental candidate blocking is controlled by the **Historical Elimination Gate** (Layer 1):

An Experimental feature becomes context-level **`blocked`** in `feature_context_summary` when:

$$S_{\text{remove}} \ge \mathtt{remove\_block\_consecutive\_threshold}\text{ (default: 2)}$$
$$\mathbf{OR}\quad \text{remove\_runs} \ge \mathtt{remove\_block\_total\_threshold}\text{ (default: 4)}$$

```
    Auto Candidate Generation (auto_candidate_generator.py)
            │
            ▼
    query_blocked_candidates(conn, context_id)
            │
            ▼
    Rejects blocked Experimental candidates
            │
            ▼
    Candidate is NOT materialized in parquet
            │
            ▼
    Expensive downstream training avoided
```

---

## 14. Population-Specific REMOVE Behavior

REMOVE has strictly different consequences depending on feature source:

| Population | REMOVE Meaning | Automatic Block? | Automatic Delete? |
|---|---|:---:|:---:|
| **Feature Registry** | Negative health evidence | **No** (Immunity) | **No** |
| **Base Pipeline** | Negative pipeline evidence | **No** (Immunity) | **No** |
| **Experimental** | Negative candidate evidence | **Yes** (After threshold) | **No** |

```
    REMOVE Recommendation
       │
       ├── Feature Registry   ──► ALERT State (Health Warning)
       │
       ├── Base Pipeline      ──► PRIORITY Impact (Rank Degradation)
       │
       └── Experimental       ──► Historical Candidate BLOCKED Gate
```

---

## 15. Streaks

Streaks represent directional evidence:
- **KEEP Streak ($S_{\text{keep}}$)**: Consecutive `KEEP` results indicate operational stability. Required for Experimental promotion eligibility.
- **REMOVE Streak ($S_{\text{remove}}$)**: Consecutive `REMOVE` results indicate persistent degradation. Drives Experimental candidate blocking.
- **WATCH**: Represents uncertainty. Streak calculation strictly follows the chronological ordering of validation events.

---

## 16. Unique Model Evidence

`unique_models_count` measures how many distinct model packages have validated a feature.

$$\text{Case 1: } 8 \text{ validation runs on } \mathbf{1\text{ model}} \quad\text{vs.}\quad \text{Case 2: } 6 \text{ validation runs across } \mathbf{4\text{ distinct models}}$$

Case 2 represents broader generalizability across hyperparameters and feature compositions.

---

## 17. Recency Weighting (Future Policy Enhancement)

Future lifecycle policies may incorporate time-decay / recency weighting:
$$\text{Recent Evidence} > \text{Stale Evidence}$$
> [!NOTE]
> Recency weighting is a **future policy enhancement** and is **not implemented in the current codebase**. The current Evidence Score treats all valid historical runs equally within a context.

---

## 18. Evidence Confidence vs. Evidence Score

- **Evidence Score**: What does history indicate? (Directional quality: $[-100.0, +100.0]$)
- **Evidence Confidence**: How much independent volume supports it? (Sample size, unique models)

Confidence complements, but does not overwrite, raw validation evidence.

---

## 19. Recommendation Priority Model

The Feature Studio lifecycle conceptual model separates:

```
    Evidence Score ──────► What does historical validation say? ([-100.0, +100.0])
          │
    Evidence Confidence ─► How much independent multi-model evidence exists?
          │
    Lifecycle Priority ──► Current Base ranking derived from Evidence Score
          │
    Lifecycle State ─────► Current operational state (active, held, blocked, alert)
```

---

## 20. Feature Studio UI Presentation

### 20.1. Feature Registry Columns
`Feature Name` · `Health Status` · `Evidence Score` · `KEEP Runs` · `WATCH Runs` · `REMOVE Runs` · `Unique Models` · `Last Validated` · `Alert State`

### 20.2. Base Pipeline Columns
`Priority Rank` · `Feature Name` · `Health Status` · `Evidence Score` · `KEEP Runs` · `WATCH Runs` · `REMOVE Runs` · `KEEP Streak` · `REMOVE Streak` · `Unique Models` · `Total Runs` · `Last Validated`

### 20.3. Selected Experimental Columns
`Pipeline ID` · `Snapshot ID` · `Feature Name` · `Lineage Status` · `Context Gate` · `Evidence Score` · `KEEP Streak` · `REMOVE Streak` · `Unique Models` · `Total Runs` · `Promotion Candidate Badge` · `Last Validated`

---

## 21. Decision Priority Hierarchy

When evaluating multiple lifecycle signals, the system enforces the following priority order:

```
    1. Historical Safety (Prevent blocked candidates from materializing)
           │
           ▼
    2. Repeated Negative Evidence (REMOVE streaks & degradation alerts)
           │
           ▼
    3. Strong Positive Evidence (KEEP streaks & high evidence scores)
           │
           ▼
    4. Evidence Confidence (Multi-model support)
           │
           ▼
    5. WATCH / Uncertainty (Monitoring queue)
           │
           ▼
    6. Human Governance & Review
```

For Experimental candidates:
$$\mathbf{BLOCKED} > \mathbf{PROMOTION\_CANDIDATE} > \mathbf{ACTIVE} > \mathbf{WATCH} > \mathbf{NEW / INSUFFICIENT\ EVIDENCE}$$

---

## 22. No Automatic Base Pipeline Mutation

The Recommendation Lifecycle must **never** automatically modify `pipeline_features_config.py` or generate code adding features to the Base Pipeline.

$$\text{Recommendation} \longrightarrow \text{Promotion Candidate} \longrightarrow \text{Human Review} \longrightarrow \text{Manual Pipeline Promotion}$$

Automatic Experimental &rarr; Base promotion is **not implemented in the codebase**.

---

## 23. No Automatic Feature Deletion

Recommendation processing must **never** automatically delete:
- Feature Registry canonical definitions
- Base Pipeline features
- Experimental feature definitions

The only automated elimination performed is the **Pre-Training Candidate Blocking Gate**, which prevents historically blocked candidates from being regenerated.

---

## 24. Rebuildability Invariant

All lifecycle projections must be mathematically reproducible from `recommendation_evidence`:

```
                    recommendation_evidence (Authoritative Log)
                                       │
                         rebuild_all_projections()
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
          feature_context_summary             experimental_lineage_summary
```

The policy never depends on ephemeral counters that cannot be reconstructed from raw evidence.

---

## 25. Current Implementation vs. Future Policy Enhancements

### 25.1. Current Implementation (Active Codebase)
- [x] **Immutable Evidence Log**: Authoritative event storage in `recommendation_evidence`.
- [x] **Dataset-Context Isolation**: All calculations scoped by `context_id` in `dataset_contexts`.
- [x] **Evidence Score Formula**: Implemented additive formula bounded in $[-100.0, +100.0]$ with weights ($w_{\text{keep}}=+25$, $w_{\text{remove}}=-35$, $w_{\text{watch}}=-10$, $B_{\text{keep}}=+15$, $P_{\text{remove}}=-25$).
- [x] **Base Pipeline Priority Ranking**: Derived directly from the implemented Evidence Score descending.
- [x] **Directional Streaks**: Chronological `KEEP` and `REMOVE` streak tracking.
- [x] **Unique Model Counting**: First-class tracking of distinct model packages (`unique_models_count`).
- [x] **Experimental Candidate Blocking**: Historical Elimination Gate blocking when $S_{\text{remove}} \ge 2$ OR $\text{remove\_runs} \ge 4$.
- [x] **Registry & Base Immunity**: Base Pipeline and Registry features immune from automatic blocking and deletion.
- [x] **Lineage-Specific Tracking**: Tracking scoped by `(context_id, pipeline_id, pipeline_snapshot_id, feature_name)`.
- [x] **Promotion Candidate Gate**: Eligibility when $S_{\text{keep}} \ge 3$, $M_{\text{unique}} \ge 2$, and $\text{score} \ge 75.0$.
- [x] **Human Governance Boundary**: Promotion candidate state surfaces for review without automatic code mutation.
- [x] **Automatic Persistence**: Automatic background SQLite writes upon successful Production Validation compute.
- [x] **Mathematical Rebuildability**: Zero-loss projection reconstruction via `rebuild_all_projections()`.

### 25.2. Future Policy Enhancements (NOT CURRENTLY IMPLEMENTED)
- [ ] **Dedicated Base Pipeline Operational Priority Score**: Multi-factorial operational scoring layer distinct from the raw Evidence Score.
- [ ] **Automated Risk Score**: Composite score evaluating WATCH depth and degradation severity.
- [ ] **Statistical Evidence Confidence Metric**: Normalized confidence index.
- [ ] **Dynamic Recency / Time-Decay Weighting**: Time-attenuated historical weighting.
- [ ] **Multi-Versioned Recommendation Policy Switching**: Pluggable policy versions.
- [ ] **In-App Promotion Review Workflow**: Formal GUI staging queue for candidate promotion.
- [ ] **Automated Code Generation for Base Promotion**: Automatic generation of `pipeline_features_config.py` updates.
- [ ] **Base Pipeline Historical Change Audit Log**: Version control log for pipeline modifications.

---

## 26. Complete System Lifecycle Architecture

```
                               Production Validation
                                         │
                                         ▼
                               KEEP / WATCH / REMOVE
                                         │
                                         ▼
                              recommendation_evidence
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
               feature_context_summary         experimental_lineage_summary
                         │                               │
                ┌────────┼────────┐                      │
                ▼        ▼        ▼                      ▼
             Registry   Base   Context Gate         Promotion Review
              Health   Ranking       │                    │
                │        │           │                    │
                │        │           ▼                    ▼
                │        │       Experimental        PROMOTION_
                │        │         BLOCKED            CANDIDATE
                │        │                                │
                │        │                                ▼
                │        │                           Human Review
                │        │                                │
                │        └────────────────────────────────┤
                │                                         ▼
                │                                Future Base Pipeline
                │
                ▼
              Feature Studio
```

$$\begin{aligned}
\text{Generate} &\longrightarrow \text{Historical Gate} \longrightarrow \text{Materialize} \longrightarrow \text{Feature Selection} \longrightarrow \text{Train} \\
&\longrightarrow \text{Unseen Validation} \longrightarrow \text{Evidence DB} \longrightarrow \text{Scoring} \longrightarrow \text{Lifecycle Decision}
\end{aligned}$$
