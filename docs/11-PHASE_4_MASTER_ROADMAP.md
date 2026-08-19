# AruMLStudio Phase 4 Master Roadmap & Step-by-Step Implementation Directive
## Authoritative Engineering Specification: Advanced Quantitative Calculus, Model Taxonomy, Regime Registry, Persistent Benchmarking & Automated Model Discovery

> **Document Number**: `Doc 11`  
> **Document Type**: AUTHORITATIVE MASTER ROADMAP & STEP-BY-STEP IMPLEMENTATION DIRECTIVE  
> **Operational Baseline**: Phases 1–4E Verified & Operational (**472/472 Tests Passing across 33 Test Modules**), Docs 00–15  
> **Status**: **AUTHORITATIVE ROADMAP SPECIFICATION** (Phases 1–4E: `IMPLEMENTED & VERIFIED`; Phase 4F: `PLANNED / IMMEDIATE NEXT`; Phase 4G–4H, Phase 5A–5B: `PLANNED`; Phase 6: `STRATEGIC DESTINATION`)  
> **Hardware Constraint**: Designed strictly for a **16 GB RAM Local Workstation** (Zero cloud dependencies)

---

## 1. Executive Summary & Strategic Context

### Ultimate Practical Purpose
**AruMLStudio** is the **research, machine-learning, validation, evidence, and fine-tuning platform** whose primary purpose is to continuously discover, validate, stress-test, calibrate, and govern the decision models and complete policies used by the **TWO PRODUCTION TRADING ENGINES**:
1. **Strategy Allocation Engine** (Pre-Entry Decision Engine)
2. **Averaging Engine** (Post-Entry Position Management Engine)

AruMLStudio is **NOT** the live trading engine and does **NOT** place broker orders. The two production trading engines are the **products** being continuously improved.

$$\boxed{\mathbf{Q}^*: \text{For each market context and regime } R, \text{ discover, validate, and govern the most robust complete trading policies for Strategy Allocation and Averaging}}$$

```text
                    ARUMLSTUDIO
          RESEARCH + FINE-TUNING PLATFORM
                       │
        ┌──────────────┴──────────────┐
        ↓                             ↓
STRATEGY ALLOCATION              AVERAGING
RESEARCH                          RESEARCH
        │                             │
        ↓                             ↓
Direction/Regime/              Recovery/Target/
Strategy/Capital               Add/Stop classifiers
classifiers                          │
        │                             │
        ↓                             ↓
Meta Confidence                 Meta Confidence
        │                             │
        ↓                             ↓
Complete Allocation             Complete Averaging
Policy Research                 Policy Research
        │                             │
        └──────────────┬──────────────┘
                       ↓
             COMPLETE POLICY SIMULATION
                       ↓
             WALK-FORWARD VALIDATION
                       ↓
              REGIME STRESS TESTING
                       ↓
                 RISK ANALYSIS
                       ↓
               HUMAN GOVERNANCE
                       ↓
             APPROVED POLICY PACKAGE
                       ↓
             PRODUCTION TRADING SYSTEM
                       │
        ┌──────────────┴──────────────┐
        ↓                             ↓
Strategy Allocation              Averaging Engine
     Engine                           │
        │                             │
        └──────────────┬──────────────┘
                       ↓
              LIVE ORDER/EXECUTION
                       ↓
                    TELEMETRY
                       ↓
                  ARUMLSTUDIO
```

---

## 2. Formal Lifecycle State Taxonomy

To ensure complete architectural clarity across all documentation and tools, every component and subsystem is assigned exactly one status:

| Status Key | Technical Definition | Active Codebase Meaning |
|---|---|---|
| **`IMPLEMENTED`** | Source code is fully written, integrated, and functioning in the active repository. | Exists in `apps/` or `src/`. |
| **`VERIFIED`** | Code has passed all unit, integration, and regression test suites with 100% assertions met. | 472/472 regression tests passing. |
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
│ Phase 4C     │ Model Taxonomy & Regime Architecture     │ 4-Dimensional Meta Schema   │ ✅ IMPLEMENTED / VER│
│ Phase 4D     │ Persistent Multi-Model Benchmarking      │ Research Memory (`analysis`)│ ✅ IMPLEMENTED / VER│
│ Phase 4E     │ Automated Project Recommendations        │ Advisory Dossier Engine     │ ✅ IMPLEMENTED / VER│
│ Phase 4F     │ Automated Model Discovery & Fine-Tuning  │ Candidate Discovery & Tuning│ 🔵 PLANNED / NEXT   │
│ Phase 4G     │ Lineage & Registry Integrity Auditor     │ Deterministic Read-Only Aud │ 🔵 PLANNED          │
│ Phase 4H     │ Optional Continuous Registry Watcher     │ Passive Real-Time Drift Mon │ 🔵 PLANNED (Passive)│
├──────────────┼──────────────────────────────────────────┼─────────────────────────────┼─────────────────────┤
│ Phase 5A     │ Production Trading Engine Evidence Bridge│ Model-to-Policy Bridge      │ 🔵 PLANNED (Moved)  │
│ Phase 5B     │ Production Strategy & Averaging Engines  │ Engine Implementation & Live│ 🔵 PLANNED          │
├──────────────┼──────────────────────────────────────────┼─────────────────────────────┼─────────────────────┤
│ Phase 6      │ Autonomous Quantitative Research Factory │ Overnight Autonomous Engine │ 🌟 STRATEGIC DEST.  │
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
- **`STRATEGY_SUITABILITY`**: Discrete option strategy suitability classification.
- **`RECOVERY_CLASSIFIER`**: Discrete position recovery probability classification.
- **`TARGET_CLASS_CLASSIFIER`**: Discrete basket recovery target class selection.
- **`ACTION_CLASSIFIER`**: Discrete position management action recommendation (`HOLD`, `ADD_1_LOT`, `ADD_2_LOTS`, `ADD_4_LOTS`, `STOP`, `EXIT`).
- **`CAPITAL_ALLOCATION_CLASSIFIER`**: Pre-entry initial capital sizing classification (`CONSERVATIVE`, `NORMAL`, `AGGRESSIVE`).
- **`CONFIDENCE_CLASSIFIER`**: Conditional win probability / calibration filter ($[0.0, 1.0]$).
- **`VOLATILITY_ESTIMATOR`**: Discrete volatility state classification.

> [!IMPORTANT]
> **Extensibility Invariant**: "Trend" and "Sideways" are **NEVER** Task Types. Task Type is strictly invariant to market conditions; a `DIRECTION_CLASSIFIER` remains a `DIRECTION_CLASSIFIER` across all market regimes.

### 4.2. Dimension 2: Market Regime Taxonomy
- **Baseline Regimes**: `R000` (`ALL_REGIMES`), `R001` (`TREND`), `R002` (`SIDEWAYS`), `R003` (`HIGH_VOLATILITY`), `R004` (`LOW_VOLATILITY`), `R005` (`BREAKOUT`), `R006` (`REVERSAL`), `R007` (`EXPIRY_PINNING`).
- **Discovered Regimes**: Supports empirical micro-clusters registered dynamically in `regime_registry_store.json`.

### 4.3. Dimension 3: Model Population Tier (Governance Standing)
- **`EXPERIMENTAL`**: Speculative models from parameter sweeps or novel feature sets.
- **`VALIDATED`**: Passed walk-forward out-of-sample holdout validation.
- **`CHALLENGER`**: Validated model actively competing against the incumbent champion.
- **`CHAMPION`**: The single highest-ranked, human-governed production model for that context.

### 4.4. Dimension 4: Lifecycle Status (Operational Readiness)
- **`CANDIDATE`** $\rightarrow$ **`ACTIVE`** $\rightarrow$ **`DEGRADED`** $\rightarrow$ **`DEPRECATED`** $\rightarrow$ **`ARCHIVED`**.

---

## 5. Subsystem Detailed Specifications (Phase 4A through Phase 6)

---

### Phase 4A — Higher-Order Option Surface Engine `[PLANNED]`
- **Purpose**: Calculate continuous volatility surface parameters and higher-order Greeks (Vanna, Volga, SVI Total Implied Variance, SABR strike skew).
- **Architecture**: Isolated in custom additive transformation modules (`surface_svi.py`, `surface_sabr.py`, `higher_greeks.py`).
- **Independence**: Fully decoupled; does not alter existing Phase 1–3D scoring.

---

### Phase 4B — Composite Non-Linear Feature Selection `[PLANNED]`
- **Purpose**: Combine TreeSHAP, Mutual Information entropy, and Permutation Importance into a normalized composite importance score.
- **Safety**: Strictly contained inside walk-forward training folds; row-group subsampled ($N \le 10,000$) for 16 GB RAM safety.

---

### Phase 4C — Model Taxonomy & Regime Architecture `[IMPLEMENTED & VERIFIED]`
- **4C.1: Model Taxonomy Foundation `[IMPLEMENTED & VERIFIED]`**: Enums, dataclasses, and canonical context keys:
  $$\mathbf{K}_{\text{context}} = (\text{Market}, \text{Interval}, \text{Task Type}, \text{Prediction Horizon}, \text{Regime ID})$$
- **4C.2: Model Registry Extension `[IMPLEMENTED & VERIFIED]`**: Additive SQLite schema updates in `analysis.db (champion_history)`, context-scoped champion governance, and package metadata taxonomy stamping.
- **4C.3: Regime Registry (`regime_registry_store.json`) `[IMPLEMENTED & VERIFIED]`**: Authoritative JSON catalog storing declarative regime definitions, parent/child hierarchy, required features, and immutable versioning with canonical `definition_hash`.
- **4C.4: Model Research Lab Population Awareness `[IMPLEMENTED & VERIFIED]`**: 4-dimensional faceted filtering toolbar, Treeview taxonomy columns, Population badges, and context-scoped champion display.

---

### Phase 4D — Persistent Research Memory & Multi-Model Benchmarking `[COMPLETED & VERIFIED]`
- **4D.1: Research Memory Schema & DB Initialization `[IMPLEMENTED & VERIFIED]`**: `<data_dir>/analysis.db` SQLite connection management, WAL mode, foreign keys, and complete 9-table schema.
- **4D.2: Experiment Identity & Canonical Deduplication `[IMPLEMENTED & VERIFIED]`**: Pure deterministic canonicalization, 6-decimal float quantization, SHA-256 experiment signature hashing, check-and-register concurrency gate.
- **4D.3: Model Benchmark & Metrics Persistence `[IMPLEMENTED & VERIFIED]`**: Persistent benchmark run evaluation events, model benchmark scorecards, and extensible normalized granular metrics.
- **4D.4: Regime & Feature Composition Evaluation `[IMPLEMENTED & VERIFIED]`**: Authoritative feature population categorization, experimental dependency ratios, cross-regime degradation scoring, and empirical regime-feature affinity summaries.
- **4D.5: Robustness Ranking Policy Engine `[IMPLEMENTED & VERIFIED]`**: Multi-factor penalty scoring (`ROB_POLICY_v1.0`), Pareto non-dominated multi-objective frontier calculation, 5-level deterministic tie-breaking, context-scoped candidate ranking, and explainable Ranking Dossiers.
- **4D.6: Research Campaign Lifecycle & Champion History `[IMPLEMENTED & VERIFIED]`**: Campaign state machine, atomic single-statement quota allocation (`UPDATE RETURNING`), experiment trial linking, and immutable champion audit trail.
- **4D.7: Model Research Lab Leaderboard UI `[IMPLEMENTED & VERIFIED]`**: Context-isolated multi-model research leaderboard, canonical context key resolver, multi-tab empirical evidence dossiers in Tkinter desktop UI.
- **4D.8: Full Regression & Lineage Verification `[IMPLEMENTED & VERIFIED]`**: Comprehensive end-to-end lineage integration tests, 4-dimensional orthogonality verification, and cryptographic immutability assertions.

---

### Phase 4E — Automated Project Recommendations `[COMPLETE & VERIFIED / ADVISORY ONLY]`
- **Purpose**: Transform accumulated research evidence into intelligent, prioritized recommendations for research exploration while strictly preserving human governance.
- **Boundary**: **Strictly Advisory**. Zero automated mutation of `PL_0001`, `feature_registry_store.json`, `analysis.db (champion_history)`, or live trading configurations.
- **4E.1: Context Coverage & Evidence Density Analyzer `[IMPLEMENTED & VERIFIED]`**: Operational coverage matrix across canonical `ModelContextKey` dimensions, safe bounded evidence density scores ($0.0 \dots 100.0$), and deterministic coverage classification.
- **4E.2: Production Champion Vulnerability & Challenger Gap Auditor `[IMPLEMENTED & VERIFIED]`**: Fragility detection, cross-regime degradation, calibration (ECE) deficits, staleness auditing, and challenger gap analysis with strictly read-only production governance.
- **4E.3: Empirical Feature Affinity & Interaction Recommender `[IMPLEMENTED & VERIFIED]`**: Multi-source empirical feature scoring ($0.0 \dots 100.0$), disaggregated confidence scaling, pairwise interaction synergy modeling, deprecated feature quarantining, and missing champion opportunity detection.
- **4E.4: Negative Evidence Pruning & Search Space Exclusion Engine `[IMPLEMENTED & VERIFIED]`**: Authoritative deduplication suppression, deprecated feature exclusion, chronic low-robustness detection ($\ge 3$ trials $< 40.0$), extreme cross-regime fragility alerts ($> 30\%$), and severe miscalibration warnings ($\text{ECE} \ge 0.10$).
- **4E.5: Multi-Objective Recommendation Priority Scoring Engine `[IMPLEMENTED & VERIFIED]`**: Transparent multi-objective composite priority scoring ($0.0 \dots 100.0$), evidence-confidence categorization, negative-pruning integration, deterministic tie-breaking, and ranked opportunity dossiers.
- **4E.6: Model Research Lab Recommendation Dossier & UI Agenda Integration `[IMPLEMENTED & VERIFIED]`**: Dynamic explainable research recommendation dossiers, component score decomposition, and interactive Research Recommendations tab integration in the Model Research Lab UI.

---

### Phase 4F — Automated Model Discovery, Strategy Validation & Fine-Tuning Loop `[PLANNED / IMMEDIATE NEXT]`

> [!IMPORTANT]
> **Phase 4F Purpose & Strategic Identity**:  
> Phase 4F empowers AruMLStudio to **automatically discover, train, evaluate, strategy-validate, rank, reject, and fine-tune candidate models** in an unattended, autonomous loop.  
> It is **MODEL-FIRST, NOT STRATEGY-FIRST**: The primary research object is the predictive ML model. The trading strategy is a simple, controlled, deterministic evaluation harness used to determine whether the model's predictions have practical trading value under realistic market conditions.

```text
    START CAMPAIGN (Unattended / Overnight Mode)
          ↓
    Generate Candidate (via Phase 4E Priority Dossiers & Lineage)
          ↓
    Train Candidate Model (Algorithms, Features, Hyperparams)
          ↓
    Evaluate Model (Statistical ROC-AUC, LogLoss, Brier, ECE)
          ↓
    Test Model through Simple Deterministic Strategy Evaluation Harness
          ↓
    Walk-Forward Validation (Zero Look-Ahead Leakage)
          ↓
    Regime / Stress Validation (R001–R007)
          ↓
    Robustness Evaluation (ROB_POLICY_v1.0 Multi-Objective Scorecard)
          ↓
    Rank Candidate vs Context Incumbent (Research Champion / Challenger)
          ↓
    Reject Weak Candidate OR Retain Promising Candidate
          ↓
    Fine-Tune / Generate Improved Descendant Candidate
          ↓
    Repeat Cycle Unattended (Until Budget / Resource / Plateau Stop Condition)
          ↓
    Generate Morning Executive Research Report
```

#### 12 Authoritative Architectural Specifications for Phase 4F:

1. **Model-First Architecture (Not Strategy-First)**:
   - The primary research object is the predictive machine learning model.
   - The trading strategy is strictly an **Evaluation Harness / Strategy Replay Proxy** used to determine whether the model's predictions yield genuine trading edge.

2. **Simple & Deterministic Strategy Evaluation Harness (Preventing Overfitting Traps)**:
   - To prevent catastrophic combinatorial search explosion and overfitting, Phase 4F does **NOT** simultaneously optimize models, strategy rules, averaging sequences, target percentages, and lot sizing.
   - The evaluation harness is **fixed, simple, and deterministic**, testing pure signal efficacy under controlled assumptions.

3. **Classification-First Research Candidates**:
   - Evaluates discrete candidate classification models:
     - `Direction Classifier` (Up / Down / Sideways)
     - `Regime Classifier` (`R001`–`R007`)
     - `2% Outcome Classifier` (Research Candidate Hypothesis)
     - `3% Outcome Classifier` (Research Candidate Hypothesis)
     - `4% Outcome Classifier` (Research Candidate Hypothesis)
     - `Meta-Confidence Classifier` (Reliability & Calibration Gate)
   - *Note: 2%, 3%, and 4% are empirical research candidate labels, NOT hardcoded production truth.*

4. **Multi-Faceted Trading-Level Evidence Matrix**:
   - The strategy evaluation harness produces comprehensive empirical trading telemetry:
     - Signal accuracy, precision, recall, and F1 score
     - Confidence calibration (ECE & reliability diagrams)
     - Total signal frequency, profitable signals vs losing signals, and overall win rate
     - Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
     - Maximum drawdown, consecutive losing streaks, and drawdown duration
     - Time-to-target and time-to-failure distributions
     - Temporal stability across trading sessions and cross-regime robustness

5. **Multi-Objective Model Selection (No Single-Metric Traps)**:
   - Candidates are evaluated using the complete Phase 4D/4E evidence infrastructure (`ROB_POLICY_v1.0` penalty scoring, Pareto non-dominated frontiers, cross-regime degradation) **plus** trading-evaluation evidence.
   - A model with slightly lower ROC-AUC will be favored over an overfitted high-AUC model if it produces superior out-of-sample trading stability and drawdown control.

6. **Strict Zero Data Leakage & Out-of-Sample Integrity**:
   - Strategy replay and validation operate strictly on **out-of-fold / out-of-sample predictions**.
   - Training fold data is cryptographically isolated and can never influence evaluation or ranking metrics.

7. **Cryptographic Fine-Tuning Lineage**:
   - Fine-tuning creates an immutable parent-child lineage relationship:
     $$\boxed{\text{Parent Model} \longrightarrow \text{Experiment} \longrightarrow \text{Modified Feature / Hyperparam / Target} \longrightarrow \text{Child Model} \longrightarrow \text{Evaluation} \longrightarrow \text{Comparison with Parent} \longrightarrow \text{Keep / Reject}}$$

8. **Unattended Overnight Research Loop**:
   - Supports autonomous execution: Researcher configures campaign quota/budget $\to$ Engine explores, trains, evaluates, replays, and tunes $\to$ Synthesizes Executive Morning Research Report detailing top candidates, rejected trials, lineage graphs, and recommended next steps.

9. **Deterministic Research Stop Conditions**:
   - The autonomous loop stops cleanly when:
     - Experiment quota or trial budget is exhausted
     - Time limit is reached
     - No promising candidate hypotheses remain in the search queue
     - Repeated descendant generations plateau with zero statistically significant improvement ($< 0.5\%$ gain over 5 generations)
     - Negative evidence pruning excludes remaining branches
     - Workstation memory/resource ceilings are approached ($> 12.0$ GB RAM)

10. **Absolute No Production Promotion (Human Governance Boundary)**:
    - Phase 4F is strictly advisory and research-scoped.
    - May identify and register a **Research Champion** or Challenger in `analysis.db`.
    - **Never** modifies `active_model.json`, never alters production champion state, never modifies `PL_0001`, never sends broker orders, and never activates live trading. Production promotion remains 100% human-governed.

11. **Strict Subsystem Boundary (No Live Engines in 4F)**:
    - Phase 4F does **NOT** build or implement the live production Strategy Allocation Engine or Averaging Engine. Those belong to Phase 5.
    - Phase 4F discovers and validates the **decision models** that will feed those production engines.

12. **Zero Regression Dependency in Trading Evaluation**:
    - The evaluation harness uses discrete classification outputs, threshold triggers, and realized tick/candle market outcomes.
    - Regression prediction of exact future option LTP or rupee P&L is **NOT** required or used.

---

### Phase 4G — Lineage & Registry Integrity Auditor `[PLANNED]`
- **Purpose**: Deterministic, read-only validator verifying cryptographic SHA-256 hashes, snapshot IDs, and lineage graphs from raw market data to deployed models.

---

### Phase 4H — Optional Continuous Registry Watcher `[PLANNED / PASSIVE]`
- **Purpose**: Real-time background file monitor alerting researchers to external file drift or orphaned models.
- **Constraint**: Built strictly **after** Phase 4G is verified; passive alerting only.

---

### Phase 5A — Production Trading Engine Evidence Bridge `[PLANNED]`
- **Purpose**: Formal evidence bridge connecting validated research models to complete production engine policies:
  $$\boxed{\text{Model Evidence} \longrightarrow \text{Complete Engine Policy} \longrightarrow \text{Full Production Simulation} \longrightarrow \text{Walk-Forward Validation} \longrightarrow \text{Regime Stress} \longrightarrow \text{Risk Analysis} \longrightarrow \text{Policy Evidence} \longrightarrow \text{Human Governance}}$$
- **Strategic Placement Rationale**: Moved from previous Phase 4F position. Establishes the formal evidence bridge *after* models are discovered and validated in Phase 4F, immediately preceding live engine implementation in Phase 5B.

---

### Phase 5B — Production Strategy Allocation & Averaging Engine Engineering `[PLANNED]`
- **Purpose**: Implementation of the two live production trading engines and runtime execution components:
  1. **Strategy Allocation Engine**: Pre-entry decision execution, strategy suitability, capital allocation, reserve contracts.
  2. **Averaging Engine**: Post-entry position management, adverse move monitoring, lot progression ladders, basket recovery targeting.
  3. **Runtime Execution Stack**: Position Manager (state tracking), Exit Engine (target/risk execution), and Order Execution Engine (broker routing & telemetry).
- **Human Governance Boundary**: Strategy and averaging policies require explicit human governance approval before live production deployment.

---

### Phase 6 — Autonomous Quantitative Research Factory `[STRATEGIC DESTINATION]`
- **Purpose**: Fully autonomous, continuous quantitative research factory synthesizing features, pipelines (`PL_0002+`), multi-regime ensembles, and morning executive dossiers within 16 GB RAM workstation limits.
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
