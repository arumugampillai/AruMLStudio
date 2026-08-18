# AruMLStudio Phase 4 Master Roadmap & Architectural Specification
## Next-Generation Quantitative Capabilities & Mathematical Extensions

> **Document Type**: MASTER ARCHITECTURAL ROADMAP & SPECIFICATION  
> **Status**: **PLANNED / NOT IMPLEMENTED** (Strict Design Specification Only)  
> **Subsystem Scope**: Phase 4 Advanced Quantitative Extensions (Phase 4A through Phase 4H)  
> **Prerequisites**: Phase 1, Phase 2A, Phase 2B, Phase 3A, Phase 3B, Phase 3C, Phase 3D (All Verified & Passing 210/210 Tests)  
> **Hardware Constraint**: Designed specifically for a **16 GB RAM Local Workstation** without cloud dependencies.

---

## 1. Executive Summary & Phase 4 Vision

The **Phase 4 Master Roadmap** defines the next generation of mathematical, analytical, and governance extensions for **AruMLStudio**.

Building upon the protected foundation of **Phases 1 through 3D**, Phase 4 introduces higher-order derivatives surface modeling, non-linear feature importance synthesis, population-aware model lab diagnostics, persistent cross-architecture benchmarking, automated project grouping suggestions, strategy-to-governance evidence bridges, and system-wide integrity auditing.

> [!IMPORTANT]
> **Phase 4 Ground Rule**:  
> All Phase 4 capabilities are **currently PLANNED / NOT IMPLEMENTED**.  
> No Phase 4 code shall be written, no database schema altered, and no existing Phase 1–3D scoring logic modified without prior independent architectural review and resource budgeting.

---

## 2. Comprehensive Phase 4 Subsystem Breakdown

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 4 SUBSYSTEM TOPOLOGY                                │
├────────────┬──────────────────────────────────────────┬────────────────────────────────┤
│ Sub-Phase  │ Title                                    │ Category                       │
├────────────┼──────────────────────────────────────────┼────────────────────────────────┤
│ Phase 4A   │ Higher-Order Option Surface Engine       │ Advanced Quantitative Calculus │
│ Phase 4B   │ Composite Non-Linear Feature Selection   │ Explainable AI & Attributions  │
│ Phase 4C   │ Model Lab Population Awareness           │ Diagnostics & Lineage Profiler │
│ Phase 4D   │ Persistent Multi-Model Benchmarking      │ Research Database & Analytics  │
│ Phase 4E   │ Automated Project Recommendations        │ Hypothesis Grouping Assistant  │
│ Phase 4F   │ Strategy Evidence Bridge                 │ PnL & Strategy Alignment       │
│ Phase 4G   │ Lineage & Registry Integrity Auditor     │ Deterministic Audit Validator  │
│ Phase 4H   │ Optional Continuous Registry Watcher     │ Passive Drift Monitor          │
└────────────┴──────────────────────────────────────────┴────────────────────────────────┘
```

---

### 2.1. Phase 4A — Higher-Order Option Surface Engine `[PLANNED]`

#### A. Purpose & Scope
Expands option volatility and derivatives modeling beyond point Black-Scholes Greeks and single ATM implied volatility points into full continuous surface representations.

#### B. Planned Capabilities
- **Higher-Order Analytical Greeks**:
  - $\text{Vanna} = \frac{\partial \Delta}{\partial \sigma} = \frac{\partial \text{Vega}}{\partial S}$ (Spot-volatility sensitivity)
  - $\text{Volga (Vomma)} = \frac{\partial \text{Vega}}{\partial \sigma} = \frac{\partial^2 V}{\partial \sigma^2}$ (Volatility-of-volatility convexity)
- **Parametric Volatility Surface Calibration**:
  - **SVI (Stochastic Volatility Inspired)**: Raw and quasi-explicit formulations parameterizing Total Implied Variance $w(k) = a + b \left( \rho(k - m) + \sqrt{(k - m)^2 + \sigma^2} \right)$.
  - **SABR (Stochastic Alpha Beta Rho)**: Hagan asymptotic expansion for continuous strike skew calibration $\sigma_{\text{SABR}}(K, F)$.
- **Surface Diagnostics**:
  - Strike skew slope $\frac{\partial \sigma}{\partial K}\Big|_{\text{ATM}}$, Smile curvature $\frac{\partial^2 \sigma}{\partial K^2}\Big|_{\text{ATM}}$, and Term Structure gradient $\frac{\partial \sigma}{\partial T}$.
  - Calibration root-mean-square error (RMSE), arbitrage-free condition verifications (calendar spread and butterfly arbitrage elimination).

#### C. Architectural Rules & Invariants
- **Zero Impact on Existing Scoring**: Phase 4A features are initially **strictly additive** and isolated in custom transformation families (`surface_svi.py`, `surface_sabr.py`, `higher_greeks.py`).
- **Phase 1–3D Independence**: Does not modify Phase 1 scoring, Phase 2A confidence, or Phase 3A decision thresholds.

#### D. Core Research Questions
1. Does continuous smile curvature provide superior market regime classification compared to discrete strike spreads?
2. Does Vanna/Volga exposure improve multi-step forward direction classification accuracy?
3. What is the calibration latency overhead per trading day on a 16 GB RAM workstation?

---

### 2.2. Phase 4B — Composite Non-Linear Feature Selection `[PLANNED]`

#### A. Purpose & Scope
Unifies multi-dimensional feature importance metrics (tree gains, permutation drops, SHAP attribution, and mutual information entropy) into a single, standardized composite importance score.

#### B. Planned Capabilities
- **Normalized SHAP ($\\widetilde{\text{SHAP}}$)**: Mean absolute TreeSHAP values normalized to $[0, 1]$.
- **Normalized Mutual Information ($\\widetilde{\text{MI}}$)**: Non-linear binned mutual information $I(X_j; Y)$ normalized to $[0, 1]$.
- **Normalized Permutation Importance ($\\widetilde{\text{Perm}}$)**: Out-of-fold metric degradation normalized to $[0, 1]$.
- **Configurable Composite Score**:
  $$\text{CompositeScore}_j = w_{\text{shap}} \cdot \widetilde{\text{SHAP}}_j + w_{\text{mi}} \cdot \widetilde{\text{MI}}_j + w_{\text{perm}} \cdot \widetilde{\text{Perm}}_j$$
  where $w_{\text{shap}} + w_{\text{mi}} + w_{\text{perm}} = 1.0$.

#### C. Critical Invariants & Safety
- **Zero Look-Ahead Leakage**: All SHAP, MI, and Permutation calculations must be strictly confined within time-series walk-forward training folds.
- **Resource Constraints**: TreeSHAP and Permutation routines must use row-group subsampling ($N \le 10,000$ per evaluation split) to avoid RAM exhaustion.

---

### 2.3. Phase 4C — Model Lab Population Awareness `[PLANNED]`

#### A. Purpose & Scope
Enhances the **Model Research Lab** UI to provide real-time visibility into the exact generative population composition of any trained model package.

#### B. Planned Capabilities
- **Population Classification**:
  - Classifies active features into: `BASE`, `REGISTRY`, `EXPERIMENTAL`, and `DEPRECATED`.
- **Diagnostics Breakdown**:
  - Displays feature counts and percentage proportions for each population.
  - Visualizes model importance concentration across experimental vs. production assets.
  - Highlights whether a trained model relies dangerously on experimental or deprecated features.

---

### 2.4. Phase 4D — Persistent Multi-Model Benchmarking `[PLANNED]`

#### A. Purpose & Scope
Materializes and persists multi-model comparison matrices into a structured research SQLite database (`analysis.db`), enabling longitudinal architecture comparisons over time.

#### B. Planned Capabilities & Metadata
- **Persistent Benchmark Record**:
  - `benchmark_id`, `run_timestamp`, `market`, `sampling_interval_sec`, `sliding_window`.
  - `model_a_identity`, `model_b_identity` (hashes, architectures, hyperparameters).
  - `evaluation_dataset_snapshot_id` (cryptographic parity).
  - Metrics delta: $\Delta\text{LogLoss}$, $\Delta\text{Accuracy}$, $\Delta\text{F1}$, $\Delta\text{ECE}$ (Expected Calibration Error).
  - Volatility regime segmentation (Low, Normal, High Volatility splits).

---

### 2.5. Phase 4E — Automated Project Recommendations `[PLANNED]`

#### A. Purpose & Scope
Acts as an advisory hypothesis-generation assistant that discovers and recommends high-performing feature subsets and logical project groupings based on historical validation evidence.

#### B. Planned Capabilities
- **Advisory Grouping Engine**:
  - Evaluates historical co-occurrence of features in successful `KEEP` models.
  - Recommends coherent feature projects (e.g. `volatility_smile_dynamics`, `order_flow_imbalance`).
- **Human Governance Invariant**:
  - **Advisory Only**: Does **NOT** mutate `feature_projects.json` or project assignments automatically. All project creation or feature assignment remains 100% human-initiated.

---

### 2.6. Phase 4F — Strategy Evidence Bridge `[PLANNED]`

#### A. Purpose & Scope
Establishes a quantitative bridge connecting downstream Strategy Lab simulation results (PnL, Drawdown, Profit Factor) with upstream Feature Governance.

#### B. Planned Capabilities & Safety Boundary
- **Independent Strategy Evidence Layer**:
  - Tracks strategy outcomes per feature: `win_rate`, `profit_factor`, `expectancy_per_trade`, `max_drawdown_contribution`, `trade_count`, and `regime_specific_pnl`.
- **Strict Separation Invariant**:
  - **Zero Direct Pollution**: Strategy simulation results are **NOT** injected directly into the primary Phase 1 `evidence_score` formula.
  - Strategy evidence is stored in an independent ledger (`strategy_evidence.db`) and presented as supplementary advisory context during Phase 3D Human Governance Reviews.

---

### 2.7. Phase 4G — Lineage & Registry Integrity Auditor `[PLANNED]`

#### A. Purpose & Scope
Provides a deterministic, read-only system integrity validator that audits the cross-referential consistency of the entire AruMLStudio metadata and storage ecosystem.

#### B. Planned Audit Boundaries
Executes on-demand or at critical architectural checkpoints:
1. **Pre/Post Training**: Validates dataset snapshot hashes and feature presence.
2. **Pre/Post Governance**: Validates JSON stores, schema contracts, and atomic write locks.
3. **Pre-Prediction**: Validates model artifact integrity and feature column alignments.

#### C. Verification Checklist
- [ ] Orphan Feature Registry IDs in `feature_registry_store.json`.
- [ ] Duplicate IDs or name collisions across pipelines.
- [ ] Base Pipeline membership parity: `is_base_pipeline == (FRxxxx in PL_0001)`.
- [ ] Audit log hash chains and record completeness in `feature_graduation_audit_log.jsonl`.
- [ ] SQLite schema integrity and foreign key consistency.

> [!NOTE]
> **Architecture Precedence**: Phase 4G is a **deterministic, read-only validator tool**. It does not run as a continuous background daemon.

---

### 2.8. Phase 4H — Optional Continuous Registry Watcher `[PLANNED]`

#### A. Purpose & Scope
An optional, passive background monitor that observes registry JSON stores on disk and alerts research users if unexpected manual edits or file corruptions occur.

#### B. Operating Rules
- **Passive & Alert-Only**: The watcher **never** modifies, overwrites, or attempts to "auto-repair" registry stores.
- **Strictly Optional**: Can be enabled or disabled without impacting any core research workflow.
- **Dependency**: Must only be developed after Phase 4G Integrity Auditor is fully implemented, verified, and stabilized.

---

## 3. Recommended Sub-Phase Dependency Order

```
                      ┌───────────────────────────┐
                      │         PHASE 4A          │
                      │ (Higher-Order Surfaces)   │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │         PHASE 4B          │
                      │(Composite Non-Linear FS)  │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │         PHASE 4C          │
                      │ (Model Lab Populations)   │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │         PHASE 4D          │
                      │(Persistent Benchmarking)  │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │         PHASE 4E          │
                      │ (Project Recommendations) │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │         PHASE 4F          │
                      │(Strategy Evidence Bridge) │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │         PHASE 4G          │
                      │(Lineage Integrity Auditor)│
                      └─────────────┬─────────────┘
                                    │
                                    ▼ (Optional)
                      ┌───────────────────────────┐
                      │         PHASE 4H          │
                      │ (Passive Registry Watcher)│
                      └───────────────────────────┘
```

*Note: Individual sub-phases (such as 4C, 4D, and 4G) may proceed concurrently once their respective prerequisites are stabilized.*

---

## 4. Phase 4 Safety Model & Permanent Invariants

Every Phase 4 implementation plan must explicitly guarantee the following invariants:

1. **Phase 1 Scoring Preserved**: Weighted cumulative score $[-100, +100]$ is untouched.
2. **Phase 2A Intelligence Preserved**: Confidence saturation $C$, multi-model consensus, and freshness decay remain authoritative.
3. **Phase 2B Stability & Risk Preserved**: Volatility $\sigma_S$, range spread $\Delta S$, flips $D_{\text{flips}}$, Gen Index $G$, and risk badges remain intact.
4. **Phase 3A Decisions Preserved**: 4-state deterministic qualification tiers remain unchanged.
5. **Phase 3B & 3C Provenance Preserved**: Candidate selection handoffs and training provenance tracking are maintained.
6. **Phase 3D Governance Preserved**: Human governance requirements for Registry Graduation, Base Pipeline Promotion, and Deprecation remain strictly enforced.
7. **Evidence DB Immutability**: Historical rows in `recommendation_evidence` remain 100% append-only and immutable.
8. **No Look-Ahead Leakage**: All calculations obey strict walk-forward temporal isolation.
9. **Backward Compatibility**: All existing trained models and Parquet datasets remain functional.

---

## 5. Workstation Resource Budgeting Policy (16 GB RAM)

Because AruMLStudio runs on a local 16 GB workstation, every Phase 4 implementation plan must provide an explicit **Resource Impact Specification**:

```
┌──────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Resource Dimension       │ Architectural Constraint                                    │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Expected / Peak RAM      │ Must not exceed 4.0 GB active memory per subprocess worker. │
│ CPU Complexity           │ Vectorized NumPy/SciPy operations; avoid Python loops.     │
│ GPU Acceleration         │ CUDA-accelerated where beneficial, with graceful CPU fallback.│
│ Dataset Chunking         │ Stream and process in daily / tokenized partitions.         │
│ Multiprocessing Safety   │ Shared memory or chunked IPC; avoid copying large frames.   │
│ Disk I/O & Storage       │ Snappy-compressed Parquet with dictionary encoding.         │
│ Execution Latency        │ Surface calibration < 2.0s per slice; transforms < 30s/day. │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 6. Phase 4 Implementation Entry Gate

Before any code is authored for a Phase 4 sub-phase, the following 8-point specification must be drafted and approved:

1. **Architectural Specification Document**: Clear module boundaries and interface definitions.
2. **Mathematical Specification Document**: Precise formulas, boundary conditions, and loss functions.
3. **Data-Flow & Storage Specification**: Parquet schemas, table structures, and JSON schemas.
4. **Workstation Resource Budget**: RAM, CPU, GPU, and disk requirements.
5. **Backward Compatibility & Invariant Matrix**: Confirmation of zero regression on Phases 1–3D.
6. **Unit & Integration Test Plan**: Concrete test cases covering nominal, edge, and error scenarios.
7. **Failure Recovery & Rollback Plan**: Handling corrupted inputs and missing data gracefully.
8. **Documentation Plan**: List of documentation files to create or update upon completion.

---

## 7. Definition of Phase Completion Standard

A Phase 4 capability is considered complete **ONLY** when all of the following criteria are verified:

- [ ] Complete, robust implementation meeting all mathematical specifications.
- [ ] 100% test pass rate across dedicated unit tests and subsystem integration tests.
- [ ] Zero regressions across the full 210-test baseline regression suite.
- [ ] Empirical resource measurements confirming RAM usage stays within the 16 GB workstation budget.
- [ ] Verified immutability of historical evidence databases and stores.
- [ ] Read-only architectural audit confirming zero invariant violations.
- [ ] Fully reconciled documentation representing actual code reality.
- [ ] Explicit final status statement issued: `IMPLEMENTED AND VERIFIED`.

---

## 8. Current Implementation Status

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 4 IMPLEMENTATION STATUS                             │
├────────────────────────────────────────────────────────┬───────────────────────────────┤
│ Phase 4A (Higher-Order Option Surface Engine)          │ 🔵 PLANNED / NOT IMPLEMENTED  │
│ Phase 4B (Composite Non-Linear Feature Selection)      │ 🔵 PLANNED / NOT IMPLEMENTED  │
│ Phase 4C (Model Lab Population Awareness)              │ 🔵 PLANNED / NOT IMPLEMENTED  │
│ Phase 4D (Persistent Multi-Model Benchmarking)         │ 🔵 PLANNED / NOT IMPLEMENTED  │
│ Phase 4E (Automated Project Recommendations)           │ 🔵 PLANNED / NOT IMPLEMENTED  │
│ Phase 4F (Strategy Evidence Bridge)                    │ 🔵 PLANNED / NOT IMPLEMENTED  │
│ Phase 4G (Lineage & Registry Integrity Auditor)        │ 🔵 PLANNED / NOT IMPLEMENTED  │
│ Phase 4H (Optional Continuous Registry Watcher)        │ 🔵 PLANNED / NOT IMPLEMENTED  │
└────────────────────────────────────────────────────────┴───────────────────────────────┘
```
