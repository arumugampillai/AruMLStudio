# AruMLStudio Project Constitution & High-Level Architecture Context
## Master Technical Context, Operating Constraints & Developer Constitution

> **Document Type**: PROJECT CONSTITUTION & HIGH-LEVEL ARCHITECTURE REFERENCE  
> **Target Audience**: Human Quantitative Researchers, Software Architects, and AI Coding Agents  
> **Authority**: Permanent High-Level Architectural Context  
> **Implementation Status of Foundation**: Phase 1 + Phase 2A + Phase 2B + Phase 3A + Phase 3B + Phase 3C + Phase 3D + Auto Candidate Generation Upgraded (**210/210 Tests Passing**)

---

## 1. Project Identity & Research Mission

### 1.1. Core Identity
- **Project Name**: `AruMLStudio`
- **Platform Type**: Local-first, high-throughput quantitative machine-learning research platform.
- **Primary Domain**: Financial market modeling, derivatives / options analytics, and market microstructure research (Indian Options Markets — NIFTY, BANKNIFTY, SENSEX, FINNIFTY, MIDCPNIFTY).

### 1.2. Primary Research Objectives
The platform is built to solve two core mathematical classification challenges:
1. **Market Regime Prediction**: Identifying underlying market volatility, trend, liquidity, and distribution states using multi-signal classification.
2. **Market Direction Prediction**: Classifying high-probability forward price and premium movement over discrete forward horizons.

### 1.3. Ultimate Value Criterion
The ultimate goal of AruMLStudio is **NOT** to maximize the arbitrary count of engineered features, models, scripts, or GUI panels. 

The ultimate goal is to:
- **Maximize Out-of-Sample Predictive Quality** on true forward unseen market data.
- **Improve Market-Regime Identification** under shifting macroeconomic environments.
- **Improve Market-Direction Classification Accuracy** with robust statistical edge.
- **Maintain Robustness** across diverse market conditions (high volatility, range-bound, structural trend, expiry squeezes).
- **Maintain Temporal Stability** across months and years of market regimes without performance decay.
- **Strictly Eliminate Overfitting and Look-Ahead Leakage** via disciplined out-of-fold validation and timestamp isolation.
- **Continuously Evaluate Advances in Mathematics and Machine Learning** in an isolated, measurable manner.
- **Preserve Reproducibility and Architectural Integrity** across every experiment, dataset snapshot, and model artifact.

> [!IMPORTANT]
> **The Core Value Rule**:  
> *"New technology or mathematics is valuable only when it improves validated predictive capability or research quality."*

---

## 2. Long-Term Research Philosophy & The Iterative Loop

AruMLStudio operates on a continuous, closed-loop quantitative research lifecycle:

```
                      ┌───────────────────────────┐
                      │        MARKET DATA        │
                      │  (Ticks, Options, Greeks) │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    FEATURE ENGINEERING    │
                      │ (Registry, Lags, Transforms)
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │     FEATURE DISCOVERY     │
                      │  (Analysis Lab, Families) │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    FEATURE INTERACTION    │
                      │ (Domain-Stratified Pairs) │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │     FEATURE SELECTION     │
                      │ (Phase 3A Decision Engine)│
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    MODEL ARCHITECTURE     │
                      │  (Gradient Boost, Trees)  │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │  WALK-FORWARD VALIDATION  │
                      │  (OOS Holdout Diagnostics)│
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │ REGIME/TEMPORAL ROBUSTNESS│
                      │   (Unseen Validation Set) │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    STRATEGY RELEVANCE     │
                      │ (Trading Simulation Lab)  │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │  ACCUMULATED EVIDENCE     │
                      │ (recommendation_evidence) │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │   GOVERNED IMPROVEMENT    │
                      │(Phase 3D Human Governance)│
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │       NEW RESEARCH        │
                      │  (Additive Hypotheses)    │
                      └───────────────────────────┘
```

### 2.1. Additive Isolation of New Mathematics
AruMLStudio is designed as a continuously improving research vehicle. However, new mathematical models, alternative loss functions, or experimental features must initially be **strictly additive and isolated**.

- New mathematical features must **never** immediately overwrite or alter production scoring formulas, baseline registries, or decision thresholds.
- **Preferred Lifecycle**:
  $$\text{New Capability} \longrightarrow \text{Independent Implementation} \longrightarrow \text{Validation} \longrightarrow \text{Baseline Comparison} \longrightarrow \text{Accumulated Evidence} \longrightarrow \text{Human Governance Review} \longrightarrow \text{Possible Production Integration}$$

---

## 3. Local-First Computing Policy

AruMLStudio is built under a **strict Local-First architectural policy**.

### 3.1. Architectural Policy Requirements
1. **Zero Cloud Dependency**: Core workflows (data ingestion, feature transformation, model training, diagnostics evaluation, production validation, registry graduation) must execute locally without reliance on external cloud services (AWS, Azure, GCP, Google Colab, remote compute clusters, or cloud-only APIs).
2. **Offline Resilience**: The entire application suite and all research workflows must remain 100% functional without active internet connectivity.
3. **Optional External Cloud Use**: Cloud computing may be utilized externally by researchers for exploratory experiments, but cloud connectivity must **never** become an architectural prerequisite for core studio operations.

### 3.2. Engineering Precedence
Future software architecture and algorithmic implementations must strictly prefer local engineering optimizations before recommending external compute:
$$\text{Partitioning} \longrightarrow \text{Chunking} \longrightarrow \text{Streaming} \longrightarrow \text{Disk-Backed Storage} \longrightarrow \text{Multiprocessing} \longrightarrow \text{Local GPU Acceleration} \longrightarrow \text{Memory-Efficient Data Types}$$

---

## 4. Workstation Hardware Constraints

The primary development and research environment is constrained to a local developer workstation.

### 4.1. Workstation Profile
- **System Memory (RAM)**: **`16 GB`**
- **RAM Speed**: **`2133 MHz`**
- **RAM Configuration**: 2 of 4 memory slots populated (dual-channel).
- **Transient Memory Baseline**: Task Manager captures indicate an active baseline of $\approx 10.6\text{ GB}$ in use and $\approx 5.2\text{ GB}$ available during ordinary OS/IDE multi-tasking. *(Note: This snapshot is a momentary operational observation, not a fixed allocation).*
- **Processor (CPU)**: Multi-core, multi-GHz architecture (clocked at $\approx 4.13\text{ GHz}$ during active load).
- **Graphics Processing Unit (GPU)**: Local NVIDIA GeForce GPU available for compute acceleration.
- **Storage Layer**: High-speed Solid State Drive (SSD) storage.

> [!IMPORTANT]
> **Permanent Resource Invariant**:  
> *"All core AruMLStudio workflows, dataset transformers, model training routines, and diagnostics evaluators must be designed to operate reliably within a **16 GB RAM** workstation without causing Out-Of-Memory (OOM) operating system thrashing."*

---

## 5. Resource-Aware Engineering Policy

Every newly engineered component, transformation generator, and analysis lab module must explicitly budget and control its computational footprint:

### 5.1. Resource Budget Dimensions
Every prospective computationally intensive feature must account for:
- **Expected RAM & Peak RAM**: Maximum memory overhead during vectorized calculations.
- **CPU Time Complexity**: Scaling behavior with respect to rows ($N$) and features ($M$).
- **GPU Acceleration**: Safe fallback to CPU when GPU resources are occupied.
- **Disk I/O & Parquet Compression**: Efficient row-group chunking and columnar reads.
- **Multiprocessing Safety**: Subprocess workers must not replicate massive parent memory frames into worker address spaces.
- **Deterministic Garbage Collection**: Releasing memory frames explicitly after chunk writes.

### 5.2. Anti-Patterns to Avoid
Large datasets **must never** be loaded completely into unchunked memory.
- **Strict Rule**: Avoid full-matrix materialization of entire multi-gigabyte tick tables into unindexed Pandas DataFrames.
- **Required Architecture**:
  $$\text{Partition by Date/Token} \longrightarrow \text{Stream / Load Chunk} \longrightarrow \text{Compute Transform} \longrightarrow \text{Persist to Parquet / DB} \longrightarrow \text{Release Memory} \longrightarrow \text{Iterate}$$

This resource discipline is mandatory for:
- Tree-based and kernel SHAP attribution.
- Permutation importance matrices.
- Mutual information entropy calculations.
- SVI / SABR implied volatility surface calibrations.
- Multi-model cross-validation comparisons.
- Combinatorial candidate feature generation.
- Strategy backtesting simulation loops.

---

## 6. Development Model & Human-Agent Collaboration

AruMLStudio is developed by a collaborative team of quantitative researchers, software engineers, and domain specialists ($\approx 10\text{ contributors}$).

### 6.1. Collaboration Pipeline

```
       Human Research & Domain Team
                    │
                    ▼
      Architecture & Research Alignment
                    │
                    ▼
     ChatGPT / Deep Reasoning Agents
   (Architecture, Math & Plan Reviews)
                    │
                    ▼
           Coding Agents (AGY)
      (Precise Code Implementation)
                    │
                    ▼
     Comprehensive Unit & Regression Tests
                    │
                    ▼
         Read-Only Architectural Audit
                    │
                    ▼
      Synchronized Documentation Updates
```

### 6.2. Role of AI Coding Agents
AI Coding Agents are **implementation assistants**, not autonomous project architects.
- Coding agents **must not** independently redesign the project's foundational architecture, rename core domain terminology, alter mathematical thresholds without authorization, or skip regression suites.
- Major architectural changes require an upfront design proposal, plan review, and explicit human approval before execution.

---

## 7. Permanent Rules for Coding Agents

Every coding agent working on AruMLStudio must strictly adhere to the following rules:

### 7.1. Mandatory Agent Actions
1. **Read Before Writing**: Inspect relevant architecture documentation (`docs/00`–`docs/08.6`) before proposing or modifying code.
2. **Respect Phase Boundaries**: Understand the exact boundary of the active task; do not bleed into future roadmap phases.
3. **Preserve Invariants**: Uphold existing database schemas, immutability guarantees, and mathematical definitions.
4. **Targeted Refactoring Only**: Avoid unrelated cosmetic refactoring of stable working subsystems.
5. **Execute Regression Tests**: Run the targeted test modules and the full regression suite after every code change.
6. **Verify Data Immutability**: Ensure raw historical evidence tables and schemas have zero unexpected mutations.
7. **Transparent Reporting**: Report all modified files, created files, test execution metrics, and error traces.
8. **Keep Documentation Synchronized**: Update documentation when code behavior changes to prevent documentation rot.
9. **Maintain Backward Compatibility**: Ensure legacy model packages, analysis datasets, and feature configs load cleanly.

### 7.2. Prohibited Agent Actions
1. **DO NOT** silently change scoring formulas, weighting constants, or decay factors.
2. **DO NOT** alter Phase 3A decision precedence tiers or 4-state semantic contracts.
3. **DO NOT** mutate, rewrite, or delete historical rows in `recommendation_evidence`.
4. **DO NOT** alter SQLite database schemas without explicit architectural approval.
5. **DO NOT** delete or rename historical feature identities (`FRxxxx`).
6. **DO NOT** introduce external cloud dependencies into core local workflows.
7. **DO NOT** modify production JSON stores or SQLite files during unit test runs (use isolated temporary directories).
8. **DO NOT** implement future roadmap phases without an approved, dedicated implementation plan.
9. **DO NOT** treat proposed architecture documents as if they were already implemented in production code.
10. **DO NOT** modify unrelated subsystems outside the explicit task scope.

---

## 8. Core Architectural Principles

The AruMLStudio codebase is governed by sixteen permanent architectural principles:

1. **Evidence Before Promotion**: A feature earns promotion only through accumulated, multi-model out-of-sample validation evidence.
2. **Human Governance Before Mutation**: Permanent lifecycle transitions (`REGISTRY_GRADUATION`, `BASE_PIPELINE_PROMOTION`, `FEATURE_DEPRECATION`) require explicit Human Governance Review.
3. **Read-Only Analysis Before Mutation**: Thoroughly inspect and trace state using read-only operations before executing state modifications.
4. **Atomic Transactions**: All mutations to JSON stores (`feature_registry_store.json`, `pipeline_registry_store.json`) must use atomic temporary-file replacement.
5. **Historical Evidence Immutability**: Validation evidence records in `recommendation_evidence` are append-only historical facts that are never updated or deleted.
6. **Permanent Feature Identities**: Assigned Feature Registry IDs (`FRxxxx`) are permanent and immutable; retired features remain in history.
7. **Deprecation Without Physical Deletion**: Deprecated features are flagged in metadata to maintain 100% backward compatibility with legacy model packages.
8. **Mandatory Walk-Forward Validation**: Research claims must be backed by time-series walk-forward splits and unseen forward validation days.
9. **Zero Look-Ahead Leakage**: Transformations and feature calculations must never consume forward price action or future label information.
10. **Strict Context Isolation**: Evidence, recommendations, and decisions are isolated by `DatasetContext` (`market`, `sampling_interval_sec`, `sliding_window`, `feature_project_id`).
11. **Scientific Reproducibility**: Given the same dataset snapshot, pipeline configuration, and model seed, calculations must yield identical outputs.
12. **Backward Compatibility**: New software versions must seamlessly load and infer using existing model packages and historical datasets.
13. **Local-First Execution**: The platform operates self-sufficiently on local developer hardware.
14. **Resource-Aware Engineering**: Algorithms must budget memory and compute to fit comfortably within a 16 GB RAM constraint.
15. **Documentation Integrity**: Documentation must accurately reflect actual source code implementation, avoiding speculative or stale claims.
16. **Additive Mathematics**: New mathematical models are initially additive and isolated before influencing production decision logic.

---

## 9. The Feature Lifecycle & Governance Model

Features in AruMLStudio move through a strictly governed lifecycle:

```
                            [ EXPERIMENTAL FEATURE ]
                          (Pipeline candidate generation)
                                       │
                                       ▼
                          [ VALIDATION EVIDENCE ]
                         (Multi-model OOS validation)
                                       │
                                       ▼
                       [ PHASE 3A DECISION ENGINE ]
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
           [ EXCLUDE ]            [ REVIEW ]          [ TRAIN_CANDIDATE ]
        (Blocked from build)   (Needs inspection)    (Approved for training)
                                                              │
                                                              ▼
                                                   [ PROMOTION CANDIDATE ]
                                                  (Consecutive KEEPs streak)
                                                              │
                                                [ HUMAN GOVERNANCE REVIEW 1 ]
                                                              │
                                                              ▼
                                                   [ REGISTRY GRADUATION ]
                                                  (Permanent FRxxxx Assigned)
                                                              │
                                                [ HUMAN GOVERNANCE REVIEW 2 ]
                                                              │
                                                              ▼
                                                  [ BASE PIPELINE PROMOTION ]
                                                  (Added to default PL_0001)
                                                              │
                                                [ HUMAN GOVERNANCE REVIEW 3 ]
                                                              │
                                                              ▼
                                                   [ FEATURE DEPRECATION ]
                                                  (Flagged as deprecated)
```

### 9.1. Taxonomy of Feature Populations
- **Experimental Features**: Newly generated or exploratory features evaluated in custom pipeline snapshots (`PL_0002+`).
- **Feature Registry Features**: Governed production assets cataloged in `feature_registry_store.json` with permanent `FRxxxx` identities.
- **Base Pipeline Features**: Standard default features generated for all production models in `pipeline_registry_store.json` (`PL_0001`).
- **Deprecated Features**: Historical features with persistent performance degradation that are excluded from future training but preserved for backward compatibility.

### 9.2. Runtime Authority
- **Base Pipeline Membership**: The array `pipeline_registry_store.json -> pipelines -> PL_0001 -> registry_feature_ids` is the **sole runtime authority** for Base Pipeline inclusion.
- **Registry Store Synchronization**: The flag `feature_registry_store.json -> feature_identities -> [FRxxxx] -> is_base_pipeline` is a synchronized, denormalized status representation.

---

## 10. Foundational Architecture Overview (Phases 1–3D)

The existing, fully verified foundation includes:

- **Phase 1 — Scoring & Gating**: SQLite Evidence DB, bounded scoring $[-100, +100]$, dual projections (`feature_context_summary`, `experimental_lineage_summary`), policy versioning & rollback.
- **Phase 2A — Evidence Intelligence**: Confidence saturation $C$, multi-model consensus, strict tie/split handling, freshness decay, operational priority score, and advisory rank.
- **Phase 2B — Stability & Generalization**: Score volatility $\sigma_S$, range spread $\Delta S$, direction flips $D_{\text{flips}}$, Level-1 cross-context generalization index $G$, and explicit risk badges (`[DEGRADED]`, `[SPLIT]`, `[STALE]`, `[UNSTABLE]`).
- **Phase 3A — Decision Engine**: Context-scoped, 4-state deterministic qualification (`TRAIN_CANDIDATE`, `REVIEW`, `NEW_UNSEEN`, `EXCLUDE`) with `[PROMOTION]` qualification.
- **Phase 3B — Model Builder Handoff**: Human candidate inspection, candidate handoff dialog, and preset export (`save_feature_preset()`).
- **Phase 3C — Training Provenance**: Closed-loop tracking via `training_provenance_meta.json` linking training inputs to validation outcomes.
- **Phase 3D — Promotion & Graduation Governance**: Complete governance lifecycle including Dossier Compilation (3D.1), Governance Dialog (3D.2), Atomic Registry Graduation (3D.3), Base Pipeline Promotion (3D.4A), Feature Deprecation Governance (3D.4B), and Multi-Mode Governance UI (3D.4C).

---

## 11. Upgraded Auto Candidate Generation

Auto Candidate Generation has been upgraded from an unranked alphabetical slice to a **Context-Aware, Evidence-Driven Domain-Stratified Engine**:

### 11.1. Key Capabilities
- **Evidence-Driven Ranking**: Calls `rank_features_for_candidate_generation()` to sort source features by Phase 1–3A standing (`is_candidate_generation_allowed` > `is_training_candidate` > Priority Tier > `operational_priority_score` > `feature_name`).
- **36-Parent Budget**: Allocates up to 36 parent features across canonical Feature Registry domains (`price_premium`, `spot_futures`, `greeks`, `implied_volatility`, `open_interest`, `volume_liquidity`, `chain_analytics`, `market_structure`, etc.).
- **Deterministic Quota Redistribution**: Domains with fewer features than their base quota surrender unused slots, which are redistributed round-robin in canonical `DOMAIN_ORDER`.
- **Strict Deprecation Gating**: Features marked `deprecated` in the Registry are immediately denied parent eligibility.
- **Commutative Canonical Deduplication ($A \le B$)**:
  - Commutative operators (`multiply`, `add`, `absolute_difference`, `min`, `max`) emit only canonical pairs where $A \le B$ (e.g. `current_iv_x_delta` is emitted; `delta_x_current_iv` is pruned).
  - Asymmetric operators (`divide`, `subtract`) preserve both directional variants ($A / B$ and $B / A$, $A - B$ and $B - A$).
  - Reduces pairwise explosion by $\approx 43.5\%$ with zero loss of expressiveness.

---

## 12. Research Success Criteria

AruMLStudio measures quantitative success through statistical robustness, not code volume:

```
┌────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│               TRUE MEASURES OF SUCCESS                 │                  FALSE METRICS                         │
├────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ • Out-of-sample forward classification accuracy        │ • Arbitrary feature count                              │
│ • Regime identification stability across market shifts │ • Number of complex mathematical transformations      │
│ • Precision / Recall balance on trading signals        │ • Total lines of Python code written                   │
│ • Cross-context temporal generalization                │ • Number of trained models stored on disk              │
│ • Reproducibility of experiment results                │ • Uncontrolled parameter count                         │
│ • Zero look-ahead bias and zero data leakage           │ • Excessive memory consumption                         │
└────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 13. Architectural Change-Control Workflow

Every future non-trivial architectural change must follow a 10-step protocol:

1. **Phase Identification**: Determine which phase the proposal belongs to.
2. **Boundary Definition**: Identify affected components and ensure untouched modules are quarantined.
3. **Invariant Check**: Confirm that all 16 core architectural principles are preserved.
4. **Dependency Audit**: Map data flow and functional dependencies.
5. **Roadmap & Specification Review**: Prepare architectural and mathematical specifications.
6. **Acceptance Criteria**: Establish concrete test cases and resource budgets.
7. **Human Approval**: Obtain explicit review before modifying production code.
8. **Implementation & Unit Testing**: Implement changes with full test coverage.
9. **Regression & Data Audit**: Verify that all 210 existing regression tests pass and database checksums match.
10. **Documentation Synchronization**: Reconcile documentation to reflect the updated implementation.

---

## 14. Phase 4 Entry Gate

To ensure that the platform remains stable, clean, and mathematically rigorous:

> [!CAUTION]
> **Phase 4 Implementation Gate**:  
> *"Phase 4 implementation must NOT begin merely because an interesting new mathematical idea or feature has been identified.*  
> *Every Phase 4 capability must proceed through: Master Roadmap Alignment $\longrightarrow$ Dependency Analysis $\longrightarrow$ Resource & Memory Budgeting $\longrightarrow$ Mathematical Specification $\longrightarrow$ Architectural Review $\longrightarrow$ Verified Implementation."*
