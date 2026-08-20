# AruMLStudio Project Constitution & High-Level Architecture Context
## Master Technical Context, Operating Constraints & Developer Constitution

> **Document Type**: PROJECT CONSTITUTION & HIGH-LEVEL ARCHITECTURE REFERENCE  
> **Target Audience**: Human Quantitative Researchers, Software Architects, and AI Coding Agents  
> **Authority**: Permanent High-Level Architectural Context  
> **Implementation Status of Foundation**: Phase 1 through Phase 4F Complete & Verified (**629/629 Tests Passing across 40 Test Modules**)

---

## 1. Project Identity & Research Mission

### 1.1. Core Identity & Ultimate Practical Purpose
- **Project Name**: `AruMLStudio`
- **Platform Type**: Local-first, high-throughput quantitative machine-learning research, validation, fine-tuning, and governance platform.
- **Primary Domain**: Financial market modeling, derivatives / options analytics, and market microstructure research (Indian Options Markets — NIFTY, BANKNIFTY, SENSEX, FINNIFTY, MIDCPNIFTY).

> [!IMPORTANT]
> **Ultimate Practical Purpose**:  
> **AruMLStudio is the research, validation, fine-tuning, evidence, and governance platform responsible for continuously improving the Strategy Allocation Engine and Averaging Engine used by the production trading system.**

AruMLStudio is **NOT** the live trading engine.  
AruMLStudio does **NOT** directly place broker orders.  
AruMLStudio does **NOT** replace the Strategy Allocation Engine or Averaging Engine.

The final production trading system consists of exactly **TWO Production Decision Engines**:
1. **Strategy Allocation Engine** (Pre-Entry Decision Engine)
2. **Averaging Engine** (Post-Entry Position Management Engine)

### 1.2. Supporting Runtime Execution Components (Not Decision Engines)
The production trading runtime utilizes three deterministic operational components:
- **Position Manager**: Tracks real-time multi-dimensional basket accounting, weighted average entry, unrealized P&L, time in trade, and averaging depth.
- **Exit & Risk Enforcement Engine**: Executes deterministic session cutoffs (15:15 IST), hard dollar stop losses, and target fills.
- **Order Execution Engine**: Handles broker API routing, order slicing, fill confirmations, and slippage telemetry.

These three components are **deterministic runtime execution and bookkeeping modules**, NOT additional ML decision engines.

### 1.3. Primary Research Objectives
The platform is built to solve the multi-dimensional quantitative challenges that power the two decision engines:
1. **Market Regime Prediction**: Identifying underlying market volatility, trend, liquidity, and distribution states (`R001`–`R007`).
2. **Market Direction Prediction**: Classifying high-probability forward price and premium movement over discrete forward horizons.
3. **Strategy Suitability Classification**: Determining the optimal option strategy structure for the active market state.
4. **Averaging Policy & Sequence Discovery**: Empirically discovering safe, robust lot progression ladders, adverse movement triggers, and basket exit targets.
5. **Capital Allocation & Reserve Modeling**: Protecting required averaging reserves and safety buffers.

### 1.4. Classification-First Production Modeling (Zero Regression Dependency)
Production modeling is strictly **classification-first**.

> [!IMPORTANT]
> **Zero Regression Dependency in Production**:  
> Production decision-making must **NOT** depend on regression models predicting exact future option LTP, exact future spot price, exact rupee P&L, or exact target price.  
> The production decision layer utilizes discrete, verifiable classification targets (`Direction`, `Regime`, `Strategy Suitability`, `Recovery Probability`, `Target Class`, `Action Classes [HOLD, ADD_1_LOT, ADD_2_LOTS, ADD_4_LOTS, STOP_AVERAGING, EXIT]`, `Capital Allocation Class`, `Meta-Confidence`).

### 1.5. Deterministic Policy Constraint Layer vs. ML Decision Recommendations
The production architecture strictly separates ML model recommendations from deterministic risk constraints:
- **ML Layer**: Recommends action classes (`ADD_1_LOT`, `HOLD`, `STOP_AVERAGING`), recovery probability classes, and candidate target classes based on empirical statistical patterns.
- **Deterministic Policy Constraint Layer**: Enforces non-negotiable risk ceilings (maximum capital, maximum averaging depth, approved lot progression ladder, safety reserve buffer, daily maximum loss limit, and session expiry cutoffs).

> [!CAUTION]
> **Hard Risk Floor Invariant**:  
> An ML recommendation or high confidence score can **NEVER** bypass, override, or relax a hard risk limit or maximum averaging depth set by the Policy Constraint Layer.

### 1.6. Ultimate Value Criterion: Model Quality vs. Complete Engine Policy Quality
The ultimate goal of AruMLStudio is **NOT** to maximize arbitrary model accuracy or feature counts in isolation.

> [!CAUTION]
> **Most Important Distinction — "Best ML Model" $
eq$ "Best Trading Engine"**:  
> A predictive model is only one component of a production trading system. The complete production chain is:  
> $$oxed{	ext{MODEL} \longrightarrow 	ext{DECISION POLICY} \longrightarrow 	ext{POSITION / CAPITAL POLICY} \longrightarrow 	ext{RISK POLICY} \longrightarrow 	ext{TRADING ENGINE} \longrightarrow 	ext{REAL TRADE OUTCOME}}$$  
> Therefore, AruMLStudio must fine-tune **COMPLETE ENGINE POLICIES**, not merely maximize isolated model metrics. A direction classifier with 80% accuracy is useless if its signals produce poor strategy allocation. Likewise, a recovery classifier with high accuracy is unsafe if the resulting lot sequence creates catastrophic tail risk.

The ultimate goal is to:
- **Maximize Out-of-Sample Predictive Quality** on true forward unseen market data.
- **Improve Complete Trading-Policy Robustness** across market regimes and execution environments.
- **Maintain Temporal Stability** across months and years without performance decay.
- **Strictly Eliminate Overfitting and Look-Ahead Leakage** via disciplined out-of-fold validation and timestamp isolation.
- **Preserve Reproducibility and Architectural Integrity** across every experiment, dataset snapshot, and model artifact.

> [!IMPORTANT]
> **The Core Value Rule**:  
> *"New technology, mathematics, or models are valuable only when they improve validated decision capability or complete trading engine performance."*

---

## 2. The Closed-Loop Research-to-Production Fine-Tuning Loop

AruMLStudio operates on a continuous, closed-loop quantitative research, validation, and fine-tuning lifecycle:

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

### 2.1. The 20 Core Research Responsibilities of AruMLStudio

AruMLStudio is specifically responsible for:

1. **Feature Research**: Engineering and validating domain-stratified microstructure, order-flow, and volatility features.
2. **Classification Model Research**: Training discrete, multi-horizon directional, regime, and strategy classifiers.
3. **Meta-Confidence Research**: Evaluating classification reliability and calibration uncertainty.
4. **Strategy Suitability Research**: Classifying option strategy suitability across shifting market conditions.
5. **Averaging-Policy Research**: Discovering empirical policies for position management under adverse movement.
6. **Capital Allocation Research**: Optimizing initial sizing versus required averaging and safety reserves.
7. **Lot-Sequence Research**: Evaluating linear, stepped, and geometric lot progression ladders.
8. **Adverse-Spacing Research**: Finding optimal fixed and adaptive percentage drop triggers.
9. **Basket-Target Research**: Determining discrete recovery target classes (`TARGET_A`, `TARGET_B`, `TARGET_C`).
10. **Risk-Policy Research**: Establishing hard exposure, daily drawdown, and session cutoff boundaries.
11. **Walk-Forward Validation**: Enforcing strict temporal out-of-fold validation with zero look-ahead leakage.
12. **Regime Stress Testing**: Stress testing models and policies across all 7 baseline market regimes (`R001`–`R007`).
13. **Temporal Robustness**: Auditing cross-month, cross-year performance decay and parameter stability.
14. **Calibration Analysis**: Measuring Expected Calibration Error (ECE) and reliability diagrams.
15. **Negative-Evidence Pruning**: Pruning unproductive or hazardous search branches and deprecated features.
16. **Research Priority Scoring**: Transparent multi-objective ranking of candidate research opportunities.
17. **Recommendation Dossiers**: Generating explainable, human-readable research agendas for quantitative review.
18. **Challenger Evaluation**: Auditing Production Champion vulnerabilities and Challenger performance leads.
19. **Production Candidate Preparation**: Packaging validated artifacts with cryptographic lineage and configuration.
20. **Human-Governed Promotion**: Enforcing strict human approval gates before production activation.

### 2.2. Additive Isolation of New Mathematics & Policies
AruMLStudio is designed as a continuously improving research vehicle. However, new mathematical models, alternative loss functions, or experimental features must initially be **strictly additive and isolated**.

- New mathematical features must **never** immediately overwrite or alter production scoring formulas, baseline registries, or decision thresholds.
- **Preferred Lifecycle**:
  $$	ext{New Capability} \longrightarrow 	ext{Independent Implementation} \longrightarrow 	ext{Validation} \longrightarrow 	ext{Baseline Comparison} \longrightarrow 	ext{Accumulated Evidence} \longrightarrow 	ext{Human Governance Review} \longrightarrow 	ext{Possible Production Integration}$$

---

## 3. Local-First Computing Policy

AruMLStudio is built under a **strict Local-First architectural policy**.

### 3.1. Architectural Policy Requirements
1. **Zero Cloud Dependency**: Core workflows (data ingestion, feature transformation, model training, diagnostics evaluation, production validation, registry graduation) must execute locally without reliance on external cloud services (AWS, Azure, GCP, Google Colab, remote compute clusters, or cloud-only APIs).
2. **Offline Resilience**: The entire application suite and all research workflows must remain 100% functional without active internet connectivity.
3. **Optional External Cloud Use**: Cloud computing may be utilized externally by researchers for exploratory experiments, but cloud connectivity must **never** become an architectural prerequisite for core studio operations.

### 3.2. Engineering Precedence
Future software architecture and algorithmic implementations must strictly prefer local engineering optimizations before recommending external compute:
$$	ext{Partitioning} \longrightarrow 	ext{Chunking} \longrightarrow 	ext{Streaming} \longrightarrow 	ext{Disk-Backed Storage} \longrightarrow 	ext{Multiprocessing} \longrightarrow 	ext{Local GPU Acceleration} \longrightarrow 	ext{Memory-Efficient Data Types}$$

---

## 4. Workstation Hardware Constraints

The primary development and research environment is constrained to a local developer workstation.

### 4.1. Workstation Profile
- **System Memory (RAM)**: **`16 GB`**
- **RAM Speed**: **`2133 MHz`**
- **RAM Configuration**: 2 of 4 memory slots populated (dual-channel).
- **Available Hardware Concurrency**: 4 CPU Cores (Max 4 parallel worker processes).
- **Local GPU**: Dedicated local hardware accelerator available for training acceleration.

### 4.2. Workstation Memory Safety Invariant
All quantitative algorithms, feature matrix pipelines, cross-validation loops, and persistent database queries must operate strictly within a **12 GB peak RAM budget** (reserving 4 GB for OS, background services, and Tkinter UI rendering).
