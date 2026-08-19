# AruMLStudio Phase 4 Master Roadmap & Step-by-Step Implementation Directive
## Authoritative Engineering Specification: Advanced Quantitative Calculus, Model Taxonomy, Regime Registry, Persistent Benchmarking & Autonomous Research Factory

> **Document Number**: `Doc 11`  
> **Document Type**: AUTHORITATIVE MASTER ROADMAP & STEP-BY-STEP IMPLEMENTATION DIRECTIVE  
> **Operational Baseline**: Phases 1–3D Verified & Operational (**210/210 Tests Passing**), Docs 00–13  
> **Status**: **AUTHORITATIVE ROADMAP SPECIFICATION** (Phases 4A–4H & Phase 5: `PLANNED / NOT IMPLEMENTED`)  
> **Hardware Constraint**: Designed strictly for a **16 GB RAM Local Workstation** (Zero cloud dependencies)

---

## 1. Executive Summary & Strategic Context

The ultimate mission of **AruMLStudio** is to transform from an interactive, human-driven machine learning studio into a **continuous, local, autonomous quantitative research factory**:

$$\boxed{\mathbf{Q}^*: \text{For each market regime } R, \text{ discover, validate, and govern the most robust predictive model or ensemble } \mathcal{M}^*_{R}}$$

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    THE OVERNIGHT ONE-BUTTON EXPERIENCE                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Researcher configures parameters or presses "Start Overnight Campaign" before sleep.                 │
│ 2. AruMLStudio autonomously explores: Features → Pipelines → Datasets → Models → Regimes.               │
│ 3. Prunes non-viable branches via Phase 1–3A Decision Engine gates to prevent combinatorial explosion.  │
│ 4. Benchmarks candidates out-of-sample across market regimes within strict 16 GB RAM memory budgets.    │
│ 5. Researcher wakes up to an Executive Morning Research Report detailing discovered champion models.    │
│ 6. Human governance explicitly reviews and approves production promotions (Phase 3D boundary).          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Formal Lifecycle State Taxonomy

To ensure complete architectural clarity across all documentation and tools, every component and subsystem is assigned exactly one status:

| Status Key | Technical Definition | Active Codebase Meaning |
|---|---|---|
| **`IMPLEMENTED`** | Source code is fully written, integrated, and functioning in the active repository. | Exists in `apps/` or `src/`. |
| **`VERIFIED`** | Code has passed all unit, integration, and regression test suites with 100% assertions met. | 210/210 regression tests passing. |
| **`PLANNED`** | Formally specified in the authoritative roadmap; awaiting explicit implementation turn. | Architectural design finalized. |
| **`PROPOSED`** | Early architectural hypothesis or concept under pre-implementation design review. | Non-binding technical draft. |
| **`DEFERRED`** | Intentionally postponed until dependent prerequisite subsystems are hardened. | Explicitly queued for later phase. |
| **`DEPRECATED`** | Phased out or superseded; maintained strictly for historical replay and audit provenance. | Read-only backward compatibility. |

---

## 3. Comprehensive Master Subsystem Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      ARUMLSTUDIO MASTER ROADMAP MATRIX                                      │
├──────────────┬──────────────────────────────────────────┬─────────────────────────────┬─────────────────────┤
│ Phase        │ Subsystem / Milestone                    │ Category                    │ Verified Status     │
├──────────────┼──────────────────────────────────────────┼─────────────────────────────┼─────────────────────┤
│ Phase 1      │ Evidence DB, Scoring & Elimination Gate  │ Statistical Evidence Engine │ ✅ IMPLEMENTED / VER│
│ Phase 2A     │ Evidence Intelligence (Consensus, Fresh) │ Multi-Model Analytics       │ ✅ IMPLEMENTED / VER│
│ Phase 2B     │ Stability (σ_S, Range) & Level-1 Gen (G) │ Robustness & Risk Badges    │ ✅ IMPLEMENTED / VER│
│ Phase 3A     │ Decision Engine (Candidate Qualification)│ Pre-Training Policy Bridge  │ ✅ IMPLEMENTED / VER│
│ Phase 3B     │ Evidence Studio UI & Reason Inspector    │ Human Interactive UI        │ ✅ IMPLEMENTED / VER│
│ Phase 3C     │ Model Builder Preset Handoff & Provenance│ Closed-Loop Traceability    │ ✅ IMPLEMENTED / VER│
│ Phase 3D     │ Feature Promotion, Graduation & Deprec.  │ Governance Framework        │ ✅ IMPLEMENTED / VER│
│ Auto Cand Up │ Evidence-Driven Parents & Commutative Ded│ Combinatorial Pruning       │ ✅ IMPLEMENTED / VER│
├──────────────┼──────────────────────────────────────────┼─────────────────────────────┼─────────────────────┤
│ Phase 4A     │ Higher-Order Option Surface Engine       │ Advanced Surface Calculus   │ 🔵 PLANNED          │
│ Phase 4B     │ Composite Non-Linear Feature Selection   │ Attributions & Pruning      │ 🔵 PLANNED          │
│ Phase 4C.1   │ Model Taxonomy Foundation                │ 4-Dimensional Meta Schema   │ ✅ IMPLEMENTED / VER│
│ Phase 4C.2   │ Model Registry Extension                 │ SQLite & Metadata Sync      │ 🔵 PLANNED          │
│ Phase 4C.3   │ Regime Registry (`regime_registry_store`)│ Market Regime Catalog       │ 🔵 PLANNED          │
│ Phase 4C.4   │ Model Lab Population Awareness           │ Faceted Research UI         │ 🔵 PLANNED          │
│ Phase 4D     │ Persistent Multi-Model Benchmarking      │ Research Memory (`analysis`)│ 🔵 PLANNED          │
│ Phase 4E     │ Automated Project Recommendations        │ Advisory Grouping Engine    │ 🔵 PLANNED (Advisory│
│ Phase 4F     │ Strategy Evidence Bridge                 │ Isolated Simulation Ledger  │ 🔵 PLANNED          │
│ Phase 4G     │ Lineage & Registry Integrity Auditor     │ Deterministic Read-Only Aud │ 🔵 PLANNED          │
│ Phase 4H     │ Optional Continuous Registry Watcher     │ Passive Real-Time Drift Mon │ 🔵 PLANNED (Passive)│
├──────────────┼──────────────────────────────────────────┼─────────────────────────────┼─────────────────────┤
│ Phase 5      │ Autonomous Quantitative Research Factory │ Overnight Autonomous Engine │ 🌟 STRATEGIC DEST.  │
└──────────────┴──────────────────────────────────────────┴─────────────────────────────┴─────────────────────┘
```

---

## 4. Core Model Architecture: Four Orthogonal Dimensions

Model identity is **multidimensional and cryptographically anchored**. The platform strictly prohibits overloading a single `type` field:

```
                              ┌──────────────────────────────────┐
                              │     FOUR MODEL DIMENSIONS        │
                              └─────────────────┬────────────────┘
                                                │
       ┌────────────────────────┬───────────────┴───────────────┬────────────────────────┐
       ▼                        ▼                               ▼                        ▼
┌──────────────┐        ┌──────────────┐                ┌──────────────┐         ┌──────────────┐
│  DIMENSION 1 │        │  DIMENSION 2 │                │  DIMENSION 3 │         │  DIMENSION 4 │
│  TASK TYPE   │        │MARKET REGIME │                │  POPULATION  │         │LIFECYCLE STAT│
└──────┬───────┘        └──────┬───────┘                └──────┬───────┘         └──────┬───────┘
       │                        │                               │                        │
  Mathematical            Environmental                   Governance &             Operational
  Objective               Condition                       Maturity Tier            Readiness
```

### 4.1. Dimension 1: Task Type Enum (Mathematical Formulation)
- **`DIRECTION_CLASSIFIER`**: Binary/ternary forward sign prediction ($\{-1, 0, +1\}$).
- **`REGIME_CLASSIFIER`**: Multi-class environmental state categorization ($\{R_1, \dots, R_K\}$).
- **`REGRESSION`**: Continuous value estimation (e.g. forward price difference, return in points).
- **`TRIPLE_BARRIER`**: Path-dependent outcome classification ($\{TP, SL, TIME\}$).
- **`CONFIDENCE_CLASSIFIER`**: Conditional win probability / calibration filter ($[0.0, 1.0]$).
- **`VOLATILITY_ESTIMATOR`**: Continuous forward realized variance estimation ($\mathbb{R}^+$).

> [!IMPORTANT]
> **Extensibility Invariant**: "Trend" and "Sideways" are **NEVER** Task Types. Task Type is strictly invariant to market conditions; a `DIRECTION_CLASSIFIER` remains a `DIRECTION_CLASSIFIER` across all market regimes.

### 4.2. Dimension 2: Market Regime Taxonomy
- **Baseline Regimes**: `R000` (`ALL_REGIMES`), `R001` (`TREND`), `R002` (`SIDEWAYS`), `R003` (`HIGH_VOLATILITY`), `R004` (`LOW_VOLATILITY`), `R005` (`BREAKOUT`), `R006` (`REVERSAL`), `R007` (`EXPIRY_PINNING`).
- **Discovered Regimes**: Supports empirical micro-clusters (e.g. `R017`: *"High IV expansion + accelerating gamma + abnormal volume"*) registered dynamically in `regime_registry_store.json`.

### 4.3. Dimension 3: Model Population Tier (Governance Standing)
- **`EXPERIMENTAL`**: Speculative models from parameter sweeps or novel feature sets.
- **`VALIDATED`**: Passed walk-forward out-of-sample holdout validation.
- **`CHALLENGER`**: Validated model actively competing against the incumbent champion.
- **`CHAMPION`**: The single highest-ranked, human-governed production model for that context.

### 4.4. Dimension 4: Lifecycle Status (Operational Readiness)
- **`CANDIDATE`** $\rightarrow$ **`ACTIVE`** $\rightarrow$ **`DEGRADED`** $\rightarrow$ **`DEPRECATED`** $\rightarrow$ **`ARCHIVED`**.

---

## 5. Subsystem Detailed Specifications (Phase 4A through Phase 5)

---

### Phase 4A — Higher-Order Option Surface Engine `[PLANNED]`
- **Purpose**: Calculate continuous volatility surface parameters and higher-order Greeks (Vanna, Volga, SVI Total Implied Variance, SABR strike skew).
- **Architecture**: Isolated in custom additive transformation modules (`surface_svi.py`, `surface_sabr.py`, `higher_greeks.py`).
- **Independence**: Fully decoupled; does not alter existing Phase 1–3D scoring.

---

### Phase 4B — Composite Non-Linear Feature Selection `[PLANNED]`
- **Purpose**: Combine TreeSHAP, Mutual Information entropy, and Permutation Importance into a normalized composite importance score:
  $$\text{CompositeScore}_j = w_{\text{shap}} \cdot \widetilde{\text{SHAP}}_j + w_{\text{mi}} \cdot \widetilde{\text{MI}}_j + w_{\text{perm}} \cdot \widetilde{\text{Perm}}_j$$
- **Safety**: Strictly contained inside walk-forward training folds; row-group subsampled ($N \le 10,000$) for 16 GB RAM safety.

---

### Phase 4C — Model Taxonomy & Regime Architecture `[IMPLEMENTED & VERIFIED]`
- **4C.1: Model Taxonomy Foundation `[IMPLEMENTED & VERIFIED]`**: Enums, dataclasses, and canonical context keys:
  $$\mathbf{K}_{\text{context}} = (\text{Market}, \text{Interval}, \text{Task Type}, \text{Prediction Horizon}, \text{Regime ID})$$
- **4C.2: Model Registry Extension `[IMPLEMENTED & VERIFIED]`**: Additive SQLite schema updates in `.lifecycle_registry.db` (`task_type`, `regime_id`, `context_key`, `champion_model_name`, `challenger_model_name`, `regime_scope`, `metadata_json`), context-scoped champion governance, and package metadata taxonomy stamping.
- **4C.3: Regime Registry (`regime_registry_store.json`) `[IMPLEMENTED & VERIFIED]`**: Authoritative JSON catalog storing declarative regime definitions, parent/child hierarchy, required features, and immutable versioning with canonical `definition_hash`.
- **4C.4: Model Research Lab Population Awareness `[IMPLEMENTED & VERIFIED]`**: 4-dimensional faceted filtering toolbar (Task, Regime, Population, Lifecycle), Treeview taxonomy columns, Population badges (`👑 CHAMPION`, `⚔️ CHALLENGER`, `VALIDATED`, `EXPERIMENTAL`), and context-scoped champion display.

---

### Phase 4D — Persistent Research Memory & Multi-Model Benchmarking `[COMPLETED & VERIFIED]`
- **4D.1: Research Memory Schema & DB Initialization `[IMPLEMENTED & VERIFIED]`**: `<data_dir>/analysis.db` SQLite connection management, WAL mode, foreign keys, and complete 9-table schema (`research_campaigns`, `experiment_signatures`, `campaign_experiments`, `benchmark_runs`, `model_benchmarks`, `benchmark_metrics`, `regime_evaluations`, `feature_set_evaluations`, `champion_history`).
- **4D.2: Experiment Identity & Canonical Deduplication `[IMPLEMENTED & VERIFIED]`**: Pure deterministic canonicalization, 6-decimal float quantization, SHA-256 experiment signature hashing, atomic check-and-register concurrency gate in `analysis.db`.
- **4D.3: Model Benchmark & Metrics Persistence `[IMPLEMENTED & VERIFIED]`**: Persistent benchmark run evaluation events, model benchmark scorecards, and extensible normalized granular metrics (`benchmark_metrics`) in `analysis.db`.
- **4D.4: Regime & Feature Composition Evaluation `[IMPLEMENTED & VERIFIED]`**: Authoritative feature population categorization (Base PL_0001, Canonical Registry, Experimental PL_0002+, Deprecated, Unknown), experimental dependency ratios, cross-regime degradation scoring, and empirical regime-feature affinity summaries in `analysis.db`.
- **4D.5: Robustness Ranking Policy Engine `[IMPLEMENTED & VERIFIED]`**: Multi-factor penalty scoring (`ROB_POLICY_v1.0`), Pareto non-dominated multi-objective frontier calculation, 5-level deterministic tie-breaking, context-scoped candidate ranking, and explainable Ranking Dossiers in `analysis.db`.
- **4D.6: Research Campaign Lifecycle & Champion History `[IMPLEMENTED & VERIFIED]`**: Campaign state machine (`CREATED` -> `RUNNING` -> `PAUSED` -> `COMPLETED`/`FAILED`/`CANCELLED`), atomic single-statement quota allocation (`UPDATE RETURNING`), experiment trial linking, and immutable, context-scoped champion transition audit trail with time-travel query support in `analysis.db`.
- **4D.7: Model Research Lab Leaderboard UI `[IMPLEMENTED & VERIFIED]`**: Context-isolated multi-model research leaderboard, canonical context key resolver, production champion vs. research candidate distinction, multi-tab empirical evidence dossiers (Robustness breakdown, Pareto optimality, Cross-Regime stress matrix, Feature composition governance, Campaign lineage, Champion transition history) in Tkinter desktop UI.
- **4D.8: Full Regression & Lineage Verification `[IMPLEMENTED & VERIFIED]`**: Comprehensive end-to-end lineage integration tests, 4-dimensional orthogonality verification (Task != Regime != Population != Lifecycle), multi-threaded quota concurrency tests, and cryptographic immutability assertions across all production databases.

---

### Phase 4E — Automated Project Recommendations `[PLANNED / ADVISORY ONLY]`
- **Purpose**: Provide data-driven suggestions for feature project groupings based on empirical feature gain.
- **Boundary**: **Strictly Advisory**. Zero automated mutation of `schema_feature_meta.py` or feature projects.

---

### Phase 4F — Strategy Evidence Bridge `[PLANNED]`
- **Purpose**: Record trading simulation performance (PnL, Sharpe, Profit Factor, Drawdown) in `analysis.db`.
- **Isolation**: Strictly separated from statistical feature recommendation scoring.

---

### Phase 4G — Lineage & Registry Integrity Auditor `[PLANNED]`
- **Purpose**: Deterministic, read-only validator verifying cryptographic SHA-256 hashes, snapshot IDs, and lineage graphs from raw market data to deployed models.

---

### Phase 4H — Optional Continuous Registry Watcher `[PLANNED / PASSIVE]`
- **Purpose**: Real-time background file monitor alerting researchers to external file drift or orphaned models.
- **Constraint**: Built strictly **after** Phase 4G is verified; passive alerting only.

---

### Phase 5 — Autonomous Quantitative Research Factory `[STRATEGIC DESTINATION]`
- **Purpose**: End-to-end overnight research engine executing automated feature exploration, pipeline generation (`PL_0002+`), dataset building, training, out-of-sample validation, regime evaluation, and Morning Report synthesis within 16 GB RAM workstation limits.
- **Human Governance Boundary**: Autonomous system discovers and proposes; **Human Governance retains 100% exclusive authority** over production asset promotion (`Phase 3D`).

---

## 6. Local-First 16 GB RAM Workstation Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   16 GB RAM WORKSTATION RESOURCE BUDGET                                 │
├────────────────────────────┬─────────────────────────────┬──────────────────────────────────────────────┤
│ Resource Category          │ Workstation Ceiling         │ Engineering Control Mechanism                │
├────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────────┤
│ Peak System Memory         │ ≤ 12.0 GB (4.0 GB Headroom) │ Row-group streaming, chunked Parquet reads,  │
│                            │                             │ explicit Python GC collection after training.│
├────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────────┤
│ Worker Concurrency         │ Max 4 CPU worker threads    │ `concurrent.futures.ProcessPoolExecutor`     │
│                            │ 1 Dedicated GPU Process     │ with hard process concurrency limits.        │
├────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────────┤
│ Disk Space Management      │ Auto-capped temp files      │ Intermediate analysis parquets auto-pruned;  │
│                            │                             │ model checkpoints & metadata preserved.      │
├────────────────────────────┼─────────────────────────────┼──────────────────────────────────────────────┤
│ Fault Tolerance & Resume   │ Graceful Interruption       │ Checkpointed campaign state in `analysis.db`;│
│                            │                             │ resumable at next session startup.           │
└────────────────────────────┴─────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 7. Mandatory Step-by-Step Implementation Protocol

For every subsequent phase, the following 7-step discipline is strictly enforced:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    7-STEP IMPLEMENTATION PROTOCOL                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Step 1: Pre-Implementation Documentation & Code Inspection (Produce Gap Analysis & Affected Files).     │
│ Step 2: Architecture Checkpoint & Safety Verification (Verify Phase 1–3D, PL_0001, and 16 GB RAM rules).│
│ Step 3: Implement ONLY the Target Sub-Phase (No creeping scope; zero modification of future phases).   │
│ Step 4: Add & Run Comprehensive Tests (Unit, integration, and full 210+ regression suite).              │
│ Step 5: Immediate Documentation Reconciliation (Synchronize all affected markdown files under /docs).   │
│ Step 6: Full Read-Only Audit (Reconcile Code ↔ Docs ↔ DB ↔ Tests ↔ Registry).                          │
│ Step 7: STOP & Report (Provide phase completion report and await explicit user instruction).             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Master Dependency Graph & Execution Sequence

```
                        [ PHASE 4A ]               [ PHASE 4B ]
                   Higher-Order Surfaces       Composite Feature Sel.
                   (Quantitative Engine)       (Importance Pruning)
                            │                           │
                            └─────────────┬─────────────┘
                                          │ (Independent Capabilities)
                                          ▼
                                    [ PHASE 4C.1 ]
                               Model Taxonomy Foundation
                                          │
                                          ▼
                                    [ PHASE 4C.2 ]
                               Model Registry Extension
                                          │
                                          ▼
                                    [ PHASE 4C.3 ]
                                    Regime Registry
                                          │
                                          ▼
                                    [ PHASE 4C.4 ]
                           Model Lab Population Awareness
                                          │
                                          ▼
                                     [ PHASE 4D ]
                         Persistent Multi-Model Benchmarking
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
             [ PHASE 4E ]            [ PHASE 4F ]            [ PHASE 4G ]
          Project Recommend.      Strategy Evidence       Lineage Integrity
             (Advisory)                Bridge                  Auditor
                  │                       │                       │
                  └───────────────────────┼───────────────────────┘
                                          ▼
                                     [ PHASE 4H ]
                             Continuous Registry Watcher
                                          │
                                          ▼
                                     [ PHASE 5 ]
                            AUTONOMOUS RESEARCH FACTORY
```

---

## 9. Protected Architecture & Non-Negotiable Boundaries

1. **`PL_0001` is strictly the Base Pipeline**: Contains only approved, governed baseline transformations and graduated Registry features. Speculative exploration occurs strictly in `PL_0002+`.
2. **Phase 1–3D Decisions are Immutable**: Scoring formulas, intelligence metrics, stability filters, candidate qualification gates, and governance state machines remain authoritative.
3. **Evidence DB Immutability**: Historical rows in `feature_recommendation_evidence.db` are append-only (SHA-256 intact).
4. **Human Governance Boundary**: Machine learning models and autonomous factory engines propose; human researchers govern production assets.
