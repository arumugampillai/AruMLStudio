# AruMLStudio Ultimate Vision & Autonomous Research Architecture
## Strategic North-Star Architecture: From Local ML Studio to Autonomous Quantitative Research Factory

> **Document Number**: `Doc 12`  
> **Document Type**: STRATEGIC NORTH-STAR ARCHITECTURAL VISION  
> **Target Audience**: Quantitative Researchers, Platform Architects, Machine Learning Engineers, and Coding Agents  
> **Authority**: Permanent High-Level Strategic Destination Document (Non-Implementation Specification)  
> **Operational Prerequisite Baseline**: Phases 1–3D Verified & Operational (**210/210 Tests Passing**)  
> **Execution Constraint**: Designed for a **16 GB RAM Local Workstation** without external cloud dependencies.

---

## 1. Executive Summary & Strategic Destination

The fundamental mission of **AruMLStudio** is to transform from an interactive, human-driven machine learning studio into a **continuous, local autonomous quantitative research factory**.

The platform does not merely aim to build an arbitrary collection of feature transformations, train isolated gradient boosted trees, or fit a single universal "best model."

The overarching purpose of AruMLStudio is to:
> **"Continuously discover, synthesize, test, validate, and evolve specialized predictive models across distinct market regimes, with the ultimate objective of identifying, maintaining, and governing the most robust model or ensemble of models for each specific market regime."**

```
                                  MARKET STREAM
                                        │
                                        ▼
                                 REGIME DETECTION
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
            TREND REGIME         SIDEWAYS REGIME        HIGH VOL REGIME
                 │                      │                      │
                 ▼                      ▼                      ▼
         Model Search Space     Model Search Space     Model Search Space
                 │                      │                      │
                 ▼                      ▼                      ▼
          Best Trend Model       Best Sideways Model    Best High-Vol Model
                 │                      │                      │
                 └──────────────────────┼──────────────────────┘
                                        ▼
                              REGIME MODEL LIBRARY
                           (Governed & Provenance-Tracked)
```

### The Ultimate Quantitative Research Question:
$$\mathbf{Q}^*: \text{For a particular market regime } R, \text{ what feature subset } \mathcal{F}, \text{ transformation family } \mathcal{T}, \text{ model architecture } \mathcal{M},$$
$$\text{training configuration } \Theta, \text{ and prediction horizon } H \text{ produce the most reliable and robust out-of-sample predictive performance?}$$

---

## 2. Core Research Objectives

AruMLStudio addresses high-frequency derivatives market microstructure (NIFTY, BANKNIFTY, SENSEX, FINNIFTY, MIDCPNIFTY options chains). 

The platform continuously investigates:
1. **Market Regime Prediction**: Multi-signal classification of underlying market volatility, liquidity, trend strength, and distribution dynamics.
2. **Market Direction Prediction**: Probabilistic classification of forward spot, futures, and premium movements over multi-tick horizons.
3. **Regime-Specific Directional Models**: Specialized classifiers optimized exclusively for homogeneous market conditions.
4. **Option-Chain Structural Features**: Real-time open interest (OI) concentrations, strike volume velocity, put-call ratios (PCR), delta skew, and strike-band migrations.
5. **Underlying Cash & Futures Microstructure**: Spot-futures basis, VWAP divergences, order flow imbalances, and tick momentum.
6. **Feature Interactions**: Domain-stratified polynomial, ratio, and non-linear interactions across canonical financial domains.
7. **Volatility & Surface Calculus**: Parametric SVI / SABR calibration, continuous smile curvature, strike skew slopes, term-structure gradients, and higher-order Greeks (Vanna, Volga).
8. **Temporal Dynamics**: Time-decay windows, intraday session transitions, expiry day compression regimes, and rolling moments.
9. **Feature Selection & Pruning**: Multi-dimensional importance scoring (Tree Gain, SHAP attribution, Mutual Information entropy, Permutation loss).
10. **Model Architectures & Hyperparameters**: Gradient boosted decision trees (LightGBM, XGBoost, CatBoost), random forests, neural classifiers, and calibrated ensembles.
11. **Prediction Horizons & Training Windows**: Optimal walk-forward rolling window lengths ($W_{\text{train}}$) and forward labeling horizons ($H_{\text{forward}}$).
12. **Cross-Context & Temporal Robustness**: Generalization across sampling frequencies ($1\text{s}, 3\text{s}, 5\text{s}, 15\text{s}, 60\text{s}$) and temporal regime shifts.

---

## 3. The Ultimate One-Button Workflow (The Overnight Research Factory)

The ultimate user experience of AruMLStudio is zero-friction autonomous exploration:

> [!TIP]
> **The Overnight Autonomous Research Experience**:  
> *The researcher configures parameters or presses a single button before leaving for the night $\longrightarrow$ AruMLStudio initiates an autonomous, resource-budgeted research campaign overnight $\longrightarrow$ The researcher wakes up to a scientifically rigorous, fully reproducible report detailing all discovered features, regime-specific champion models, stability audits, and recommended next hypotheses.*

```
                                  RESEARCHER
                                      │
                                      ▼
                            ┌───────────────────┐
                            │  START OVERNIGHT  │
                            │  RESEARCH CAMPAIGN │
                            └─────────┬─────────┘
                                      │
                                      ▼
                        AUTONOMOUS RESEARCH ORCHESTRATOR
                                      │
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
          Feature Discovery    Dataset Factory     Experiment Planner
          (Auto Candidates)  (Parquet Partitions) (Hypothesis Engine)
                 │                    │                    │
                 └────────────────────┼────────────────────┘
                                      ▼
                                Model Factory
                        (Architecture & Search Space)
                                      │
                                      ▼
                              Walk-Forward Lab
                       (Strict Out-of-Sample Folds)
                                      │
                                      ▼
                            Regime Classification
                                      │
                 ┌────────────────────┼────────────────────┐
                 ▼                    ▼                    ▼
            TREND REGIME       SIDEWAYS REGIME      HIGH VOL REGIME
                 │                    │                    │
                 ▼                    ▼                    ▼
             Model T042           Model S019           Model H088
                 │                    │                    │
                 └────────────────────┼────────────────────┘
                                      ▼
                             Benchmark & Ranking
                       (Multi-Model Cross-Evaluation)
                                      │
                                      ▼
                            Evidence & Provenance
                        (recommendation_evidence.db)
                                      │
                                      ▼
                               Research Memory
                       (Experiment History & Failures)
                                      │
                                      ▼
                            Next Experiment Plan
                                      │
                                      └──────────────► [CONTINUE LOOP]
```

### Campaign Termination Conditions:
The autonomous research loop executes continuously until:
1. The configured computation budget or experiment quota is exhausted.
2. The user-defined research time window (e.g., 8-hour overnight window) elapses.
3. The researcher manually pauses or halts the campaign via the UI.
4. Workstation resource-safety limits (RAM threshold, disk budget, GPU thermal limits) trigger a safe graceful pause.

---

## 4. Regime-Specific Model Discovery

A central thesis of AruMLStudio is that **financial markets do not follow a single, stationary data-generating process**. A single global model trained across all historical data often compromises between conflicting regimes, resulting in mediocre predictive power.

### 4.1. Regime Model Populations
AruMLStudio maintains specialized, segregated model populations:

```
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│      TREND MODELS       │  │     SIDEWAYS MODELS     │  │     HIGH-VOL MODELS     │
├─────────────────────────┤  ├─────────────────────────┤  ├─────────────────────────┤
│ T001: Fast Momentum     │  │ S001: Mean-Reverting    │  │ H001: Vega/Vanna Convex │
│ T002: Breakout Follower │  │ S002: Strike Compression│  │ H002: Skew Spike Squeeze│
│ ...                     │  │ ...                     │  │ ...                     │
│ T200: VWAP Continuation │  │ S200: Gamma Pin Scalper │  │ H200: Extreme Tail Model│
└─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘
```

### 4.2. Discoverable Market Regimes
Potential market regimes investigated by the system include:
- **Trend Regimes**: Strong directional bull/bear runs, persistent delta drift, and order-flow momentum.
- **Sideways Regimes**: Range-bound consolidation, strike pinning, theta decay compression, and mean-reverting basis.
- **High Volatility Regimes**: IV expansion, large tail moves, gamma squeezes, and wide strike dispersion.
- **Low Volatility Regimes**: Compressed volatility smiles, tight bid-ask spreads, and low realized drift.
- **Transition / Breakout Regimes**: Volatility breakout initiation, support/resistance breaches, and volume surges.
- **Reversal Regimes**: Exhaustion volume, extreme RSI/z-score divergences, and strike skew flips.
- **Statistically Discovered Regimes**: Latent Dirichlet Allocation (LDA), Hidden Markov Models (HMM), Gaussian Mixture Models (GMM), or unsupervised clustering over microstructure features.

> [!NOTE]
> **No Hardcoded Regimes Assumption**: AruMLStudio does not assume fixed heuristic regimes. The research factory investigates whether deterministic rules, statistical regime models, clustering, or deep state classifiers discover the most predictive market segmentations.

### 4.3. Regime-Specific Optimization Target
For each discovered regime $R$, the system autonomously discovers:
- The optimal **feature combination** (e.g., trend models favor momentum lags; sideways models favor mean-reverting z-scores).
- The optimal **mathematical transformations** and **domain interactions**.
- The optimal **model architecture** (depth, learning rate, regularization, tree count).
- The optimal **prediction horizon** (e.g., $15\text{s}$ for high vol, $120\text{s}$ for slow trend).
- The optimal **training window** and **ensemble weighting scheme**.

---

## 5. Model Evaluation Philosophy: Robustness Over Peak Score

In quantitative machine learning, a model with the highest single-split validation score is frequently an overfitted artifact of sample noise. AruMLStudio enforces a multi-dimensional evaluation standard:

```
                                  EVALUATION ENGINE
                                          │
        ┌───────────────────┬─────────────┼─────────────┬───────────────────┐
        ▼                   ▼             ▼             ▼                   ▼
   Out-of-Sample       Calibration     Temporal      Regime          Generalization
   Walk-Forward         & LogLoss      Stability    Robustness          & Consensus
   (No leakage)        (ECE Score)    (Low σ_S)     (All States)        (Level-1 G)
```

### Multi-Dimensional Acceptance Criteria:
1. **Walk-Forward Out-of-Sample Performance**: Evaluated across rolling forward folds without overlap.
2. **Probability Calibration (Expected Calibration Error - ECE)**: Predicted probabilities must reflect true empirical win rates.
3. **Temporal Stability ($\\sigma_S, \\Delta S, D_{\\text{flips}}$)**: Consistent performance across weeks and months.
4. **Regime Robustness**: Graceful degradation behavior when out-of-regime conditions occur.
5. **Cross-Context Generalization ($G$)**: Validated performance across multiple sampling intervals ($1\text{s}, 3\text{s}, 15\text{s}$).
6. **Multi-Model Consensus**: Independent validation agreement across LightGBM, XGBoost, and CatBoost architectures.
7. **Computational Efficiency**: Inference latency and memory requirements must fit production execution boundaries.

---

## 6. Local-First Computing Architecture

AruMLStudio operates as an independent, local quantitative research environment.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              LOCAL WORKSTATION ECOSYSTEM                               │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ Local Storage Layer      │ Local Compute Layer      │ Local Governance Layer           │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ • Local Parquet Tables   │ • Multi-Core CPU Threads │ • SQLite Evidence DB             │
│ • PyArrow Chunked IO     │ • NVIDIA GeForce GPU     │ • Atomic JSON Registries         │
│ • Local Model Binaries   │ • 16 GB Workstation RAM  │ • Audit Log JSONL Streams        │
│ • On-Disk Parquet Caches │ • Out-of-Core Processing │ • Provenance Metadata JSON       │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

- **Zero Cloud Prerequisites**: Core workflows do not require AWS, Azure, GCP, Colab, or remote clusters.
- **Privacy & Security**: All proprietary proprietary alphas, features, and model weights remain strictly on the local machine.

---

## 7. Resource-Aware Engineering & 16 GB RAM Invariant

To ensure that autonomous overnight research campaigns do not crash the local developer workstation, the orchestrator enforces strict resource management:

```
┌──────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Hardware Dimension       │ Autonomous Research Policy (16 GB Workstation)              │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ System RAM Budget        │ Active research processes capped at ≤ 6.0 GB total.         │
│ Memory Reclamation       │ Explicit garbage collection & frame release after each fold.│
│ Concurrency Control      │ Process queue limits based on available memory and cores.   │
│ Out-of-Core Execution    │ Polars / PyArrow lazy execution; never load full datasets. │
│ Disk-Backed Spilling     │ Intermediate predictions spilled to disk in chunked Parquet.│
│ GPU Memory Budget        │ Cuda stream management with automatic CPU fallback.         │
│ Workstation Protection   │ Auto-throttling if system memory reaches 85% utilization.   │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 8. Subsystem Factories

The autonomous research architecture is structured into dedicated, decoupled factories:

### 8.1. Feature Discovery Factory
- Generates candidate hypotheses across canonical Registry domains, mathematical transformations (lags, differences, rolling stats, regimes, bucketing), and domain-stratified interaction pairs.
- Maintains cryptographic identity hashes to guarantee **zero redundant re-generation of identical features**.

### 8.2. Dataset Factory
- Compiles immutable, time-partitioned Analysis Datasets with cryptographic snapshot IDs (`pipeline_snapshot_id`).
- Strictly enforces walk-forward boundaries, target lag isolation, and zero forward leakage.

### 8.3. Model Factory
- Instantiates model architectures across gradient boosted trees (LightGBM, XGBoost, CatBoost), linear-regularized models, and neural classifiers.
- Enforces model complexity parsimony: a more complex model is accepted **only** if it demonstrates statistically significant out-of-sample margin over simpler baselines.

---

## 9. The Enormous Search Space & Experiment Pruning

The long-term research search space is mathematically combinatorial:

$$\Omega = \mathcal{F}_{\text{features}} \times \mathcal{S}_{\text{subsets}} \times \mathcal{T}_{\text{transforms}} \times \mathcal{I}_{\text{interactions}} \times \mathcal{U}_{\text{surfaces}} \times \mathcal{C}_{\text{contexts}} \times \mathcal{H}_{\text{horizons}} \times \mathcal{R}_{\text{regimes}} \times \mathcal{M}_{\text{models}} \times \Theta_{\text{params}} \times \mathcal{W}_{\text{windows}}$$

Because exhaustive grid search across $\Omega$ is impossible on any machine, the autonomous engine uses **evidence-driven adaptive pruning**, **Bayesian optimization**, **meta-learning**, and **Research Memory** rather than blind brute force.

---

## 10. Research Memory & Experiment Deduplication

AruMLStudio maintains an append-only, indexed **Research Memory**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                    RESEARCH MEMORY                                     │
├───────────────────────────────────┬────────────────────────────────────────────────────┤
│ Proven Successes                  │ Champion models, high-scoring features, stability. │
│ Documented Failures               │ Degraded transformations, unstable combinations.   │
│ Rejection Logs & Root Causes      │ Features failing validation, high volatility.      │
│ Redundancy Cache                  │ Cryptographic hashes of all tested configurations. │
│ Unexplored Hypothesis Frontiers   │ High-potential feature combinations not yet tried. │
└───────────────────────────────────┴────────────────────────────────────────────────────┘
```

> **Value**: The system never wastes precious compute cycles re-running identical or historically disproven experiments.

---

## 11. Feature Population Awareness & Traceability

Every model package maintains complete transparency regarding the generative origin of its feature inputs:

```
┌────────────────────────────────────────────────────────┐
│               MODEL M1842 POPULATION PROFILE           │
├──────────────────────────┬─────────────────────────────┤
│ BASE Pipeline Features   │ 42 features (54.5%)         │
│ FEATURE REGISTRY Assets  │ 28 features (36.4%)         │
│ EXPERIMENTAL Candidates  │  7 features ( 9.1%)         │
│ DEPRECATED Features      │  0 features ( 0.0%) [STRICT]│
└──────────────────────────┴─────────────────────────────┘
```

- **Safety Invariant**: Models heavily reliant on experimental features are explicitly flagged. Deprecated features are strictly blocked from entering new research pipelines.

---

## 12. Persistent Model Research Library (`analysis.db`)

Every trained model artifact, validation split, and diagnostic metric is persisted into the **Model Research Library**:

| Field | Purpose |
| :--- | :--- |
| `model_id` / `version` | Unique model identity and lineage version. |
| `architecture` | Algorithmic family (`lightgbm`, `xgboost`, `catboost`, `ensemble`). |
| `dataset_snapshot_id` | Exact cryptographic snapshot of training/validation Parquet data. |
| `feature_snapshot_id` | Exact feature set and transformation config. |
| `market` / `context_id` | Market token, sampling frequency, and sliding window. |
| `regime_id` | Target market regime specialization (`TREND`, `SIDEWAYS`, `HIGH_VOL`). |
| `prediction_horizon` | Forward labeling horizon in ticks/seconds. |
| `hyperparameters` | Full serialized model configuration. |
| `metrics_oos` | Out-of-sample LogLoss, Accuracy, Precision, Recall, F1, ECE. |
| `stability_metrics` | Volatility $\sigma_S$, Range $\Delta S$, Flips $D_{\text{flips}}$, Gen Index $G$. |
| `resource_metrics` | Training duration, peak memory footprint, inference latency. |
| `provenance_meta` | Git commit hash, environment signature, random seeds. |

---

## 13. Strict Human Governance Boundary

AruMLStudio enforces a strict separation between **Autonomous Research** and **Production Lifecycle State**:

```
                       AUTONOMOUS RESEARCH FACTORY
                    (Exploration, Training, Validation)
                                     │
                                     ▼
                        COMPILED EVIDENCE DOSSIER
                    (Multi-Model OOS Facts, Stability)
                                     │
                                     ▼
                     =================================
                     HUMAN GOVERNANCE REVIEW BOUNDARY
                     =================================
                                     │
                        [Human Reviewer Approval]
                                     │
                                     ▼
                        PRODUCTION LIFECYCLE STATE
                   (Registry Graduation, Base Pipeline)
```

- **Autonomy Scope**: The autonomous engine may freely generate candidates, train experimental models, evaluate validation sets, log evidence, and propose promotions.
- **Governance Invariant**: The autonomous engine **never** graduates features into `feature_registry_store.json`, merges features into `pipeline_registry_store.json` (`PL_0001`), or alters production trading models without explicit human governance execution.

---

## 14. Phase 4 & Phase 5 Strategic Roadmap

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             STRATEGIC EVOLUTION ROADMAP                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 1–3D: FOUNDATION (COMPLETED & VERIFIED — 210/210 TESTS PASSING)                  │
│ • Evidence DB, Policy Engine, Intelligence, Stability, Decisions, Provenance, Gov UI   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 4: ADVANCED QUANTITATIVE EXTENSIONS (PLANNED)                                    │
│ • Phase 4A: Higher-Order Option Surface Engine (Vanna, Volga, SVI/SABR Surface)        │
│ • Phase 4B: Composite Non-Linear Feature Selection (Normalized SHAP + MI + Perm)      │
│ • Phase 4C: Model Lab Population Awareness (Base / Registry / Experimental Breakdowns) │
│ • Phase 4D: Persistent Multi-Model Benchmarking (analysis.db Cross-Model Ledger)       │
│ • Phase 4E: Automated Project Recommendations (Hypothesis Grouping Assistant)          │
│ • Phase 4F: Strategy Evidence Bridge (Independent PnL / Simulation Evidence Ledger)    │
│ • Phase 4G: Lineage & Registry Integrity Auditor (Deterministic System Validator)     │
│ • Phase 4H: Optional Continuous Registry Watcher (Passive File Drift Monitor)          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 5: AUTONOMOUS RESEARCH ORCHESTRATOR (NORTH-STAR DESTINATION)                     │
│ • 5.1: Autonomous Campaign Manager (Start, Pause, Resume, Schedule Overnights)         │
│ • 5.2: Intelligent Experiment Planner (Adaptive Hypothesis Selection)                  │
│ • 5.3: Research Memory & Experiment Deduplication Engine                               │
│ • 5.4: Automated Model Factory & Regime-Specific Population Manager                    │
│ • 5.5: Resource-Budgeted Concurrency Scheduler (16 GB Workstation Safe)                │
│ • 5.6: Morning Research Briefing Generator (Actionable Discovery Reports)              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 15. The Ultimate Morning Research Report

Upon completing an overnight research campaign, AruMLStudio generates a clean, executive research briefing:

```
========================================================================================
                      ARUMLSTUDIO — OVERNIGHT RESEARCH REPORT
========================================================================================
Campaign Timestamp:       2026-08-19 06:30:00 IST
Active Execution Time:    8h 42m 15s
Target Context:           NIFTY_3s_std_all
Workstation Resource:     Peak RAM 5.1 GB | Peak GPU 3.2 GB | Zero OOM Exceptions

----------------------------------------------------------------------------------------
EXPERIMENT THROUGHPUT & SEARCH SUMMARY
----------------------------------------------------------------------------------------
Total Hypotheses Formulated:     1,420
Experiments Completed:           1,350
Early Pruned / Rejected:            58
Failed / Errored:                   12

New Candidate Features Evaluated:  240
New Domain Interactions Tested:  1,890
New Model Packages Trained:      1,350

----------------------------------------------------------------------------------------
CHAMPION REGIME-SPECIFIC MODELS DISCOVERED
----------------------------------------------------------------------------------------
• BEST OVERALL MODEL:
  - Model ID:       M_20260819_0842 (LightGBM Depth 6)
  - Direction F1:   0.742 | Accuracy: 68.4% | ECE: 0.031
  - Stability σ_S:  8.4 (Highly Stable) | Generalization G: 0.79 (Universal)

• BEST TREND REGIME MODEL:
  - Model ID:       T_20260819_0114 (LightGBM + Basis Momentum)
  - Direction F1:   0.814 | Accuracy: 76.2%
  - Key Drivers:    futures_basis_diff5, spot_ema_120, atm_iv_skew_60s

• BEST SIDEWAYS REGIME MODEL:
  - Model ID:       S_20260819_0402 (CatBoost + Strike Dispersion)
  - Direction F1:   0.698 | Accuracy: 64.1%
  - Key Drivers:    pcr_oi_zscore_300, gamma_pin_spread, volga_atm

• BEST HIGH-VOLATILITY REGIME MODEL:
  - Model ID:       H_20260819_0721 (XGBoost + SVI Surface Curvature)
  - Direction F1:   0.771 | Accuracy: 71.8%
  - Key Drivers:    svi_curvature_atm, vanna_spread_30s, iv_smile_slope

----------------------------------------------------------------------------------------
EVIDENCE & GOVERNANCE HIGHLIGHTS
----------------------------------------------------------------------------------------
• MOST PROMISING NEW FEATURE:
  - Feature:        svi_curvature_atm_x_futures_basis_diff5
  - Evidence Score: +82.5 (Fresh) | Confidence: 78.2% | Consensus: 4 KEEPs (0 REMOVEs)

• NEW PROMOTION CANDIDATE QUALIFIED:
  - Feature:        atm_iv_skew_60s (Lineage EXP_PL0004)
  - Status:         [PROMOTION] Ready for Human Governance Review (Dossier Compiled)

• DISPROVEN HYPOTHESIS / FAILURE ANALYSIS:
  - Feature Family: High-order polynomial lags (lag > 300s) degraded severely during 
                    volatility expansion regimes (Score dropped to -62.0).

----------------------------------------------------------------------------------------
RECOMMENDED NEXT RESEARCH CAMPAIGN
----------------------------------------------------------------------------------------
• Investigate cross-surface SABR Hagan strike skew interactions within high-volatility 
  regimes at 1-second sampling intervals.
========================================================================================
```

---

## 16. Permanent Architectural Principles for Autonomous Research

As autonomy increases, the following 10 principles remain permanently non-negotiable:

1. **Scientific Reproducibility**: Every model and metric must be 100% reconstructible from its cryptographic dataset snapshot, feature config, code commit, and seed.
2. **Zero Look-Ahead Bias**: Forward label information or future ticks must never leak into transformation generation or feature selection.
3. **Walk-Forward Validation**: Research claims must be backed by out-of-sample forward testing.
4. **Strict Context Isolation**: Evidence and models from one context must not contaminate another without governed generalization evaluation.
5. **Full Lineage Provenance**: Every model package must record its exact pipeline, hyperparameters, and feature composition.
6. **Evidence Before Promotion**: Performance must be demonstrated across multi-model validation before any feature or model earns promotion.
7. **Human Governance Boundary**: Autonomous systems explore and recommend; humans govern permanent production assets.
8. **Backward Compatibility**: New research engines must never break inference on existing historical model packages.
9. **Local-First Execution**: The platform must execute self-sufficiently within local workstation hardware limits (16 GB RAM).
10. **Incremental, Controlled Evolution**: New capabilities enter through structured, independently tested phases.

---

## 17. Document Governance

This document is the **Strategic North Star** for AruMLStudio.

- **Role**: Guides long-term architecture, aligns contributors, prevents uncontrolled feature creep, and establishes the criteria for true autonomous research.
- **Authority**: Does **NOT** authorize premature coding of Phase 5. Phase 5 implementation shall begin only after Phase 4 extensions are fully completed, tested, and documented.
