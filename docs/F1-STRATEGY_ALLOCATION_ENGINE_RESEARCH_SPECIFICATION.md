# Strategy Allocation & Averaging Engines Architecture Specification
## Two Production Trading Engines Powered, Fine-Tuned, and Governed by AruMLStudio

> **Document ID:** `Doc F1` (`F1-STRATEGY_ALLOCATION_ENGINE_RESEARCH_SPECIFICATION.md`)  
> **Document Type:** Authoritative Architecture & Functional Specification  
> **Subsystems:** Strategy Allocation Engine (Pre-Entry), Averaging Engine (Post-Entry), and AruMLStudio Research Platform  
> **Trading Scope:** Intraday NIFTY / SENSEX Options  
> **Status:** `AUTHORITATIVE ARCHITECTURAL SPECIFICATION` — Design Specification Only (No Direct Implementation Authorized)

---

# 1. Executive Overview & Final Project Purpose

**AruMLStudio** is the **research, machine-learning, validation, evidence, and fine-tuning platform**.

It is **NOT** the primary live trading execution engine and does **NOT** place broker orders.

The final production trading system consists of exactly **TWO Production Decision Engines**:

1. **Strategy Allocation Engine** (Pre-Entry Decision Engine)
2. **Averaging Engine** (Post-Entry Position Management Engine)

AruMLStudio researches, trains, validates, stress-tests, calibrates, benchmarks, and continuously fine-tunes the predictive and decision models used by these two production engines.

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

The two production engines must remain structurally separate, independently testable, and fine-tuned on separate objective functions.

---

# 2. Engine 1 — Strategy Allocation Engine (Pre-Entry)

The Strategy Allocation Engine is the **pre-entry decision engine**.

Its core operational questions are:

- *What market condition currently exists?*
- *Which option strategy structure is empirically suitable?*
- *What direction and regime are supported by evidence?*
- *How confident is the system in the strategy selection?*
- *How much capital can safely be allocated to the initial entry?*
- *How many initial lots should be entered?*
- *How much capital must be reserved for the Averaging Engine?*

### What Strategy Allocation Engine Must NOT Do:
- It must **not** manage averaging after entry.
- It must **not** manage an already-open position.
- It must **not** decide subsequent averaging additions.
- It must **not** place broker orders directly.

## 2.1 Strategy Allocation Inputs (Classification-First)

The Strategy Allocation Engine consumes discrete classification outputs:

```text
Direction Classifier:
    UP / DOWN / SIDEWAYS (with calibrated probability vector)

Regime Classifier:
    R001 (Trend) / R002 (Sideways) / R003 (High Vol) / R004 (Low Vol) /
    R005 (Breakout) / R006 (Reversal) / R007 (Expiry Pinning)

Volatility Classifier:
    VERY_LOW / LOW / NORMAL / HIGH / VERY_HIGH

Strategy Suitability Classifier:
    Bull Call Spread / Bear Put Spread / Long Call / Long Put /
    Iron Condor / Long Straddle / Ratio Spreads / Jade Lizard / etc.

Strategy Confidence:
    Meta-confidence / Calibration score [0.0 % .. 100.0 %]

Capital Allocation Classifier:
    CONSERVATIVE / NORMAL / AGGRESSIVE / DEFENSIVE / PROHIBITED

Portfolio Risk State:
    Current exposure, open margin, session drawdown, daily loss limit
```

## 2.2 Strategy Allocation Output Format

The engine produces a structured, governed pre-entry instruction:

```json
{
  "decision_id": "ALLOC_20260819_093015_NIFTY_001",
  "timestamp": "2026-08-19T09:30:15Z",
  "market": "NIFTY",
  "context_key": "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
  "selected_strategy": "BULL_CALL_SPREAD",
  "strategy_confidence": 0.9140,
  "capital_allocation_class": "NORMAL",
  "initial_capital": 45000.0,
  "initial_quantity_lots": 2,
  "required_averaging_reserve": 90000.0,
  "safety_reserve": 15000.0,
  "risk_gate_status": "PASS",
  "final_decision": "APPROVED",
  "governance_rule_checked": "HUMAN_APPROVED_CHAMPION_ONLY"
}
```

This output becomes the initial position instruction for the live trading application.

---

# 3. Engine 2 — Averaging Engine (Post-Entry Position Management)

The Averaging Engine is the **post-entry position-management engine**.

This engine is critical to the final production system because the trading methodology relies on controlled additions when option prices move adversely.

The Averaging Engine determines:

- *Should an additional position be added right now?*
- *At what adverse price movement threshold should an addition occur?*
- *How many lots should be added at this specific level?*
- *What is the next permitted averaging level?*
- *How much capital reserve remains available?*
- *When must averaging stop completely?*
- *What recovery/target class should be targeted for the entire basket?*
- *When should the entire basket position be closed?*

The engine must **NOT** permanently hard-code fixed rules such as *"add 1 lot every 2% drop"*.

Instead, AruMLStudio researches, validates, and fine-tunes whether the averaging sequence should adapt dynamically to:

1. Market regime (`R001`–`R007`)
2. Directional momentum and underlying velocity
3. Volatility state (`IV` / realized volatility)
4. Option premium level and strike moneyness
5. Time to expiry (DTE and minutes to session close)
6. Current basket position size and weighted average entry price
7. Adverse excursion percentage from initial vs average entry
8. Remaining capital in the Averaging Reserve
9. Empirical probability of price recovery
10. Distance to basket profit target

---

# 4. The Averaging Sequence as an Empirical Research Problem

AruMLStudio treats the averaging sequence as a **hypothesis-driven research problem**, not a hard-coded heuristic.

Candidate lot progression ladders (for empirical research evaluation):
- `1 - 1 - 1 - 1 - 1 - 1` (Linear Candidate)
- `1 - 1 - 1 - 1 - 2 - 2` (Stepped Conservative Candidate)
- `1 - 1 - 1 - 1 - 2 - 2 - 4 - 4` (Stepped Progressive Candidate)
- `1 - 1 - 2 - 2 - 4 - 4 - 8 - 8` (Geometric Capped Candidate)

Candidate adverse drop spacings (for empirical research evaluation):
- `0.5%`, `1.0%`, `1.5%`, `2.0%`, `2.5%`, `3.0%`, `4.0%`, `5.0%`
- Adaptive spacing based on option delta, implied volatility, or regime.

### 12 Mandatory Research Questions Evaluated by AruMLStudio:
1. **Safe Maximum Depth**: How many averaging levels are safe before tail risk escalates exponentially?
2. **Spacing Topology**: How far apart should levels be spaced across different volatility regimes?
3. **Trigger Reference**: Should percentage drops be measured from:
   - Initial entry price?
   - Latest entry price?
   - Basket weighted average price?
4. **Adaptive vs Fixed Spacing**: Does regime-dependent spacing improve recovery rates over fixed spacing?
5. **Quantity Progression**: How many lots should be added at each level?
6. **Progression Geometry**: Should lot sizes scale linearly, stepped, or geometrically?
7. **Regime Scaling**: Should lot sizes decrease or halt during adverse regime transitions (e.g. Breakout R005 against position)?
8. **Hard Stop Condition**: Under what exact statistical or drawdown condition must averaging halt completely?
9. **Capital Preservation**: What fraction of total capital must remain unallocated as a safety cushion?
10. **Recovery Gate**: What minimum classification recovery probability is required before authorizing the next addition?
11. **Basket Target Policy**: What discrete target class (`TARGET_A`, `TARGET_B`, `TARGET_C`) should be assigned after each addition?
12. **Timeout and Decay Limits**: At what time-decay threshold must the position be liquidated regardless of P&L?

All 12 questions must be answered empirically using out-of-sample walk-forward validation and regime stress testing in AruMLStudio.

---

# 5. Classification-First Modeling Principle (Zero Regression Dependency)

The production architecture is strictly **classification-first**.

### Core Invariant:
> **AruMLStudio and the Production Engines do NOT require or depend on regression models predicting exact future option LTP, exact future spot, exact rupee P&L, or exact target price.**

Regression models are prone to extreme noise, heavy tails, and overfitting in short-duration option trading.

The production decision layer uses discrete, verifiable classification targets:

### Action Classifications:
- `HOLD`
- `ADD_1_LOT`
- `ADD_2_LOTS`
- `ADD_4_LOTS`
- `STOP_AVERAGING`
- `EXIT_IMMEDIATELY`

### Averaging Level States:
- `NO_ADD`
- `LEVEL_1`
- `LEVEL_2`
- `LEVEL_3`
- `LEVEL_4`
- `LEVEL_5_MAX`

### Recovery Probability Classes:
- `VERY_LOW` ($P < 0.20$)
- `LOW` ($0.20 \le P < 0.40$)
- `MODERATE` ($0.40 \le P < 0.60$)
- `HIGH` ($0.60 \le P < 0.80$)
- `VERY_HIGH` ($P \ge 0.80$)

### Basket Target Classes (Research Candidates):
- `TARGET_A` (Quick recovery candidate: e.g. +1.0% to +1.5% on basket)
- `TARGET_B` (Standard target candidate: e.g. +2.0% to +2.5% on basket)
- `TARGET_C` (Extended target candidate: e.g. +3.5% to +5.0% on basket)
- `EXIT_AT_MARKET` (Risk or timeout breach)

> [!NOTE]
> **Target Class Candidate Distinctions**:  
> The percentage ranges shown above for `TARGET_A`, `TARGET_B`, and `TARGET_C` are **research candidate intervals** for empirical study. AruMLStudio must discover and validate the exact optimal target policy out-of-sample. The system must clearly distinguish between:
> 1. **Prediction Horizon** (e.g. 5m, 15m time window)
> 2. **Recovery Probability Class** (model estimate of recovery likelihood)
> 3. **Basket Target Class** (discrete recovery category)
> 4. **Exact Trading Target** (the resulting rupee / tick exit rule applied to the whole basket).

---

# 6. Deterministic Policy Constraint Layer vs ML Decisions

The architecture strictly separates ML statistical recommendations from deterministic risk constraints:

```text
┌───────────────────────────────────────────────┐
│              ML PREDICTIVE LAYER              │
│  • Recommends Action (ADD / HOLD / STOP / EXIT)│
│  • Estimates Recovery Probability Class       │
│  • Estimates Target Class                     │
│  • Calculates Meta-Confidence Score           │
└───────────────────────┬───────────────────────┘
                        │ Recommendation
                        ▼
┌───────────────────────────────────────────────┐
│     DETERMINISTIC POLICY CONSTRAINT LAYER     │
│  • Enforces Maximum Capital Limit             │
│  • Enforces Hard Maximum Averaging Depth      │
│  • Enforces Approved Lot Ladder Ceilings      │
│  • Protects Safety Reserve Buffer             │
│  • Enforces Daily Loss Limit                  │
│  • Enforces Session Cutoff (15:15 IST)        │
│  • Enforces Minimum Option Delta / Liquidity  │
└───────────────────────┬───────────────────────┘
                        │ Authoritative Command
                        ▼
┌───────────────────────────────────────────────┐
│             EXECUTION & RISK GUARD            │
│         [ ALLOW / REJECT / FORCE EXIT ]       │
└───────────────────────────────────────────────┘
```

> [!CAUTION]
> **Deterministic Priority Rule**:  
> If an ML model recommends `ADD_1_LOT` with 95% confidence, but the position has reached `LEVEL_5_MAX` or the daily drawdown limit is reached, the **Deterministic Policy Constraint Layer immediately vetoes the addition**. ML recommendations can never override hard safety ceilings.

---

# 7. Complete Position State Vector for Averaging

The Averaging Engine evaluates the multi-dimensional position state, never a naive single-variable drop:

```text
AveragingPositionState
{
    // Identification & Chronology
    position_id: "POS_20260819_001"
    entry_timestamp: "2026-08-19T09:30:15Z"
    current_timestamp: "2026-08-19T09:34:20Z"
    time_in_trade_sec: 245
    session_phase: "OPENING_DRIVE"
    time_to_expiry_min: 325

    // Price & Basket State
    initial_entry_price: 120.0
    latest_entry_price: 114.0
    current_option_ltp: 112.5
    weighted_average_entry: 117.0
    current_quantity_lots: 3
    current_averaging_level: 2
    max_permitted_level: 4

    // P&L & Exposure
    unrealized_pnl_rupees: -675.0
    unrealized_pnl_pct: -3.85%
    capital_deployed: 52650.0
    remaining_averaging_reserve: 67350.0

    // Market & Model Signals
    underlying_spot: 24550.0
    underlying_trend: "BULLISH"
    market_regime: "R001"
    regime_confidence: 0.86
    volatility_class: "NORMAL"
    direction_prediction: "UP"
    direction_confidence: 0.78
    option_greeks: { delta: 0.52, gamma: 0.04, theta: -12.5, vega: 8.2 }
    bid_ask_spread_pct: 0.15%

    // Model Recommendations
    recovery_probability_class: "HIGH"
    target_class: "TARGET_B"
    classified_action: "ADD_1_LOT"
}
```

---

# 8. Subsystem Responsibility Matrix & Boundaries

The complete trading topology spans six distinct layers with strict boundaries:

```text
┌───────────────────────────────────────────────────────────────────────┐
│                             ARUMLSTUDIO                               │
│  Research • Training • Validation • Simulation • Evidence • Registry  │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │ Models & Policies
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│                         LIVE TRADING SYSTEM                           │
│                                                                       │
│  1. STRATEGY ALLOCATION ENGINE (Pre-Entry Decision Engine)            │
│     • Strategy Suitability • Initial Allocation • Reserve Policy      │
│                                  │                                    │
│                                  ▼ Initial Order                      │
│  2. POSITION MANAGER (Runtime State Tracker — Supporting Component)   │
│     • Basket Accounting • P&L Tracking • Level Counter                │
│                                  │                                    │
│             ┌────────────────────┴────────────────────┐               │
│             ▼                                         ▼               │
│  3. AVERAGING ENGINE (Post-Entry)        4. EXIT ENGINE (Runtime)     │
│     (Decision Engine)                       • Target Triggers         │
│     • Adverse Move Analysis                 • Stop Loss Triggers      │
│     • Add / Hold / Stop Classification      • Session Cutoffs         │
│     • Ladder Progression                                              │
│             │                                         │               │
│             └────────────────────┬────────────────────┘               │
│                                  ▼ Order Commands                     │
│  5. ORDER EXECUTION ENGINE (Broker Interface — Supporting Component)  │
│     • Order Routing • Slicing • Fills • Slippage Telemetry            │
└──────────────────────────────────┬────────────────────────────────────┘
                                   │ Trade Outcomes & Telemetry
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│                     ARUMLSTUDIO RESEARCH MEMORY                       │
│     Strategy Evidence DB • Averaging Evidence • Feature Intelligence  │
└───────────────────────────────────────────────────────────────────────┘
```

### Responsibility Matrix:

| Subsystem | Subsystem Role | Primary Question | Owns | Does NOT Own |
|---|---|---|---|---|
| **AruMLStudio** | Research Platform | *What decision models & complete policies are statistically robust?* | Feature engineering, model research, walk-forward validation, regime stress, ranking, governance, registries, fine-tuning. | Live order execution, broker connections, real-time tick routing. |
| **Strategy Allocation Engine** | Decision Engine 1 (Pre-Entry) | *What strategy to enter and how much initial capital to deploy?* | Pre-entry strategy suitability, capital allocation class, initial lot sizing, reserve contract enforcement, risk gating. | Averaging decisions, post-entry position management, order placement. |
| **Averaging Engine** | Decision Engine 2 (Post-Entry) | *When and how much to add during adverse movement?* | Adverse drop triggers, lot progression ladders, recovery probability classification, target class assignment, stop-averaging gates. | Pre-entry strategy selection, broker execution, historical model training. |
| **Position Manager** | Supporting Runtime Component | *What is the live status of the active basket?* | Real-time basket accounting, weighted average LTP, margin deployed, time in trade, session clock tracking. | Strategy selection, ML prediction, order routing. |
| **Exit Engine** | Supporting Runtime Component | *When must the position terminate?* | Basket target execution, hard dollar stop loss, session cutoff (e.g. 15:15 IST), expiry liquidation. | Sizing additions, ML training, strategy selection. |
| **Order Execution Engine** | Supporting Runtime Component | *How to route orders to the exchange?* | Broker API communication, multi-leg slicing, limit/market execution, fill confirmations, slippage telemetry. | Trading decisions, ML inference, capital allocation policies. |

---

# 9. Cooperative Shared Capital Reserve Contract

The Strategy Allocation Engine and Averaging Engine share an immutable capital contract.

### Capital Allocation Partition Formula:
$$	ext{Total Available Capital} = 	ext{Initial Allocation} + 	ext{Averaging Reserve} + 	ext{Safety Reserve}$$

### Capital Invariant:
$$	ext{Initial Allocation} + 	ext{Maximum Permitted Averaging Exposure} + 	ext{Safety Reserve} \le 	ext{Total Capital}$$

### Contract Rules:
1. **No Single-Engine Overreach**: The Strategy Allocation Engine must **never** consume capital earmarked for the Averaging Reserve.
2. **Reserve Policy Export**: The Averaging Engine defines the maximum required capital for its active lot progression ladder (e.g., Level 1 to Level 4).
3. **Pre-Entry Verification**: Strategy Allocation verifies that Total Capital $\ge$ Initial Allocation + Required Averaging Reserve + Safety Reserve before approving any trade.
4. **Safety Reserve Immunity**: The Safety Reserve ($\ge 10\% - 20\%$ of account equity) is untouchable by both engines to prevent margin calls and black-swan drawdowns.

---

# 10. Two Distinct Training Objectives & Dataset Structures

AruMLStudio does **not** combine Strategy Allocation and Averaging into a single monolithic model. They have fundamentally different states, decision intervals, and loss functions.

## 10.1 Strategy Allocation Training Dataset

Each observation represents a **Pre-Entry Market State**:

```text
Features:
- Market Context (Spot, Trend, Volatility, Session Phase, DTE)
- Model Predictions (Direction probability vector, Regime ID & confidence)
- Option Chain State (ATM IV, Skew, Put-Call Ratio, Open Interest changes)
- Candidate Strategy Attributes (Max Profit, Max Loss, Margin Required, Greek Profile)
- Portfolio State (Current exposure, Daily P&L, Remaining Capital)

Labels & Target Variables:
- Strategy Suitability Class (VERY_UNFAVORABLE, UNFAVORABLE, NEUTRAL, FAVORABLE, HIGHLY_FAVORABLE)
- Capital Allocation Class (CONSERVATIVE, NORMAL, AGGRESSIVE)

Telemetry Outcomes:
- Outcome Type (TARGET_REACHED, RISK_LIMIT_REACHED, TIMEOUT)
- Trade Performance (Maximum Favorable Excursion [MFE], Maximum Adverse Excursion [MAE], Net P&L, Slippage, Duration)
```

## 10.2 Averaging Training Dataset

Each observation represents a **Live Position State** sampled at each adverse excursion trigger:

```text
Features:
- Position Chronology (Time in trade, Minutes to session close, DTE)
- Price Dynamics (Initial price, Average entry price, Current LTP, Adverse move %)
- Quantity & Depth (Current lots, Level index 1..N, Capital deployed / remaining)
- Real-Time Market State (Underlying spot velocity, Regime, Direction confidence)
- Option State (Live Delta, Gamma, Theta decay rate, Spread %)

Labels & Action Classes:
- Optimal Action Class (ADD_1_LOT, ADD_2_LOTS, ADD_4_LOTS, HOLD, STOP_AVERAGING, EXIT)
- Recovery Probability Class (VERY_LOW, LOW, MODERATE, HIGH, VERY_HIGH)
- Optimal Basket Target Class (TARGET_A, TARGET_B, TARGET_C, EXIT_AT_MARKET)
```

---

# 11. Core Architectural Invariants Preserved

This specification strictly adheres to all established AruMLStudio engineering and mathematical standards:

1. **Evidence Before Promotion**: No model or policy enters production without quantified empirical evidence.
2. **Human Governance Boundary**: All research outputs, dossiers, and recommendations are strictly read-only and advisory.
3. **Context Isolation**: All models and rankings operate strictly within the canonical 5-tuple `ModelContextKey` (`Market + Sampling Interval + Task Type + Prediction Horizon + Regime ID`).
4. **Walk-Forward Validation**: Strict temporal integrity with zero look-ahead leakage.
5. **Hardware Safety**: Designed for a standard **16 GB RAM Local Workstation** without external cloud dependencies.
6. **Production Asset Immutability**: Historical evidence databases (`feature_recommendation_evidence.db`, `analysis.db`, `.active_model.json`, `PL_0001`) remain 100% cryptographically protected.

---

# 12. Implementation Roadmap & Authoritative Phase Sequence

- **Phase 4E**: Research Recommendations `[COMPLETE & VERIFIED]`
- **Phase 4F**: Automated Model Discovery, Strategy Validation & Fine-Tuning Loop `[COMPLETE & VERIFIED — 629/629 Tests Passing across 40 Test Modules]`
- **Phase 4G**: Lineage & Registry Integrity Auditor `[PLANNED]`
- **Phase 4H**: Optional Continuous Registry Watcher `[PLANNED / PASSIVE]`
- **Phase 5A**: Production Trading Engine Evidence Bridge `[PLANNED]` (Moved from old Phase 4F)
- **Phase 5B**: Production Strategy Allocation & Averaging Engines Engineering `[PLANNED]`
- **Phase 6**: Autonomous Quantitative Research Factory `[STRATEGIC DESTINATION]`

> [!IMPORTANT]
> **Strict Governance Boundary**:  
> This document represents Authoritative Architectural & Documentation Reconciliation only. **Zero production code, database entries, or live configurations are modified in this step.**
