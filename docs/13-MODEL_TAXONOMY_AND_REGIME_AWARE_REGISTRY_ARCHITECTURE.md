# AruMLStudio Model Taxonomy & Regime-Aware Model Registry Architecture
## Technical Architectural Specification: Task Taxonomy, Regime Taxonomy, Multi-Dimensional Identity & Research Factory Topology

> **Document Number**: `Doc 13`  
> **Document Status**: `AUTHORITATIVE SPECIFICATION (Phase 4C, Phase 4D, Phase 4E & Phase 4F COMPLETE & VERIFIED — 629/629 Tests Passing; Phase 4G+ PLANNED)`  
> **Operational Baseline**: Phases 1–4F Complete & Verified (**629/629 Tests Passing across 40 Test Modules**), Docs 00–15  
> **Implementing Modules**: `apps/chain_replay_ml/model_taxonomy/`, `apps/chain_replay_ml/research_memory/`, `apps/chain_replay_ml/research_recommendations/` (`coverage.py`, `vulnerability.py`, `feature_affinity.py`, `negative_pruning.py`, `priority_scoring.py`, `dossier.py`), `apps/chain_replay_ml/research_memory/champion_history.py`, `apps/chain_replay_ml/training/registry.py`, `apps/chain_replay_ml/training/artifacts.py`, `apps/master_dataset_tk/model_registry_panel.py`, `apps/master_dataset_tk/model_registry_detail.py`, `apps/master_dataset_tk/model_research_leaderboard_panel.py`  
> **Hardware Constraint**: Designed for a **16 GB RAM Local Workstation** without external cloud dependencies.

---

## 1. Executive Summary & Architectural Motivation

### Ultimate Practical Purpose
**AruMLStudio** is the **research, machine-learning, validation, evidence, and fine-tuning platform** that researches, tests, calibrates, and governs the decision models and policies used by the **TWO PRODUCTION TRADING ENGINES**:
1. **Strategy Allocation Engine** (Pre-Entry Decision Engine)
2. **Averaging Engine** (Post-Entry Position Management Engine)

AruMLStudio is **NOT** the live execution engine. The two production engines are the **products** being continuously improved.

The central research hypothesis is that **no single universal predictive model dominates financial markets across all temporal conditions**. Financial markets undergo structural macro and micro changes (volatility expansion, momentum trending, mean-reverting chop, liquidity vacuums, expiry gamma pinning). Consequently, maximum out-of-sample edge requires **specialized, regime-aware predictive models governed within an explicit regime taxonomy**.

### The Core Architectural Problem in Current Architecture:
In the current implementation:
1. Models are categorized primarily through a hardcoded 2-tab UI dichotomy: `Regression` vs. `Triple Barrier` ([`apps/master_dataset_tk/model_registry_panel.py#L265`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/model_registry_panel.py#L265)).
2. Model identity is coupled to folder/file naming strings (`Future_LTP_5m_WF_1168f_XGB...`).
3. There is no formal metadata separation between:
   - **Task Type** (What is the model predicting mathematically? e.g. continuous return, directional sign, regime classification, barrier hit probability).
   - **Market Regime** (Under what environmental market condition is this model trained or deployed? e.g. Trend, Sideways, High Volatility, Breakout).
   - **Model Population** (Where does this model sit in research/governance? e.g. Experimental, Validated, Production Candidate, Champion, Challenger).
   - **Lifecycle Status** (What is the operational readiness state? e.g. Candidate, Active, Degraded, Deprecated, Retired).

This document establishes the **pre-implementation architectural design** for decoupling these four orthogonal dimensions, defining an extensible regime registry, standardizing permanent model identity, and preparing the foundation for Phase 4 / Phase 5 autonomous multi-regime research.

---

## 2. Current State Architectural Audit (Source-Code Verified)

A rigorous read-only source-code inspection of the active codebase yields the following authoritative inventory:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       CURRENT MODEL SUBSYSTEM AUDIT                                         │
├──────────────────────┬──────────────────────────────────────────────────────────────────┬───────────────────┤
│ Component / Feature  │ Active Codebase Implementation & Location                        │ Verified Status   │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼───────────────────┤
│ Model Creation       │ `chain_replay_ml.training.boost_trainer`, `xgb_trainer.py`,      │ ✅ IMPLEMENTED    │
│                      │ `catboost_trainer.py`, `lgb_trainer.py`                          │                   │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼───────────────────┤
│ Model Package Dir    │ On-disk directory `<data_dir>/models/<safe_model_name>/`         │ ✅ IMPLEMENTED    │
│                      │ (`chain_replay_ml.training.paths.model_package_dir`)             │                   │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼───────────────────┤
│ Package Artifacts    │ `model.json`/`.ubj`, `config.json`, `metrics.json`,              │ ✅ IMPLEMENTED    │
│                      │ `training_metadata.json`, `dataset_build_snapshot.json`          │                   │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼───────────────────┤
│ Local Registry DB    │ SQLite `<data_dir>/analysis.db` (champion_history)                │ ✅ IMPLEMENTED    │
│                      │ (`model_registry`, `model_history` tables)                       │                   │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼───────────────────┤
│ UI Discovery         │ `apps/master_dataset_tk/model_registry_panel.py` scans           │ ✅ IMPLEMENTED    │
│                      │ `<data_dir>/models/` directory for packages                      │                   │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼───────────────────┤
│ Model Research Lab   │ `ModelLabWindow` (`model_lab_<name>_v1.db`), strike dashboard,   │ ✅ IMPLEMENTED    │
│                      │ threshold analysis, strategy simulation                          │                   │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼───────────────────┤
│ Hardcoded Family NB  │ UI tabs hardcoded to `["regression", "triple_barrier"]`          │ ⚠️ PARTIALLY IMPL. │
│                      │ in `model_registry_panel.py#L265`                                │ (Overloaded)      │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼───────────────────┤
│ Task Type Separation │ Inferred ad-hoc from `target` name prefix or `strategy_id`       │ ❌ NOT FOUND      │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼───────────────────┤
│ Market Regime Schema │ No formal `regime_id`, `regime_taxonomy`, or regime registry     │ ❌ NOT FOUND      │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼───────────────────┤
│ Champion/Challenger  │ Context-scoped champion history in `analysis.db` (champion_history); │ ⚠️ PARTIALLY IMPL. │
│ Management           │ no regime-specific champion/challenger tracking                  │                   │
├──────────────────────┼──────────────────────────────────────────────────────────────────┼───────────────────┤
│ Persistent Multi-    │ In-memory comparison in `MultiModelStudioPanel`;                 │ 🔵 PLANNED        │
│ Model Benchmarking   │ persistent schema planned for Phase 4D                           │                   │
└──────────────────────┴──────────────────────────────────────────────────────────────────┴───────────────────┘
```

---

## 3. Four Orthogonal Dimensions of Model Classification

To prevent architectural overloading where a single "type" field attempts to describe target mathematics, market context, governance tier, and execution state simultaneously, the architecture strictly separates **four independent dimensions**:

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
  (e.g. DIRECTION_        (e.g. TREND,                    (e.g. EXPERIMENTAL,      (e.g. CANDIDATE,
   CLASSIFIER)             SIDEWAYS, HIGH_VOL)             CHALLENGER, CHAMPION)    ACTIVE, DEPRECATED)
```

---

## 4. Dimension 1: Task Taxonomy (Mathematical Formulation)

The **Task Type** defines the mathematical formulation of the target variable and the loss function optimized by the model.

### 4.1. Core Canonical Task Types:

| Task Type Enum | Target Mathematical Space | Typical Target Names | Loss / Objective Function | Primary Metrics |
|---|---|---|---|---|
| **`DIRECTION_CLASSIFIER`** | Binary / Ternary discrete direction: $\{-1, 0, +1\}$ or $\{0, 1\}$ | `label_up_5m`, `direction_15m`, `bar_dir_30s` | Binary Cross-Entropy, Multi-Class Log-Loss | ROC-AUC, PR-AUC, Accuracy, Precision, Directional % |
| **`REGIME_CLASSIFIER`** | Categorical regime label: $\{R_1, R_2, \dots, R_K\}$ | `regime_id_target`, `vol_state_label` | Multi-class Categorical Cross-Entropy | Macro F1, Per-Regime Precision/Recall, Log-Loss |
| **`REGRESSION`** | Continuous real values: $\mathbb{R}$ | `future_ltp_5m`, `ormp_return_5m_points`, `iv_diff_60s` | MSE, RMSE, Huber Loss, MAE | RMSE, MAE, MedAE, $R^2$, Directional Accuracy % |
| **`TRIPLE_BARRIER`** | Path-dependent discrete barrier outcome: $\{TP, SL, TIME\}$ | `label_id`, `tb_target` | Multi-Class Log-Loss / Softmax | Precision, Recall, F1, Expected Payoff, Profit Factor |
| **`CONFIDENCE_CLASSIFIER`** | Conditional success probability: $[0.0, 1.0]$ | `target_hit`, `prob_win_filter` | Binary Log-Loss, Focal Loss, Brier Score | ECE (Expected Calibration Error), Brier Score, ROC-AUC |
| **`VOLATILITY_ESTIMATOR`** | Continuous non-negative volatility: $\mathbb{R}^+$ | `future_realized_vol_5m`, `parkinson_vol_15m` | Log-Cosh, MSE on log-variance | QLIKE, RMSE, MAE, Explained Variance |

### 4.2. Task Type Extensibility Invariant:
- "Trend" and "Sideways" are **NEVER** Task Types. They represent market regimes.
- Task Type is strictly invariant to market conditions; a `DIRECTION_CLASSIFIER` remains a `DIRECTION_CLASSIFIER` whether trained on Trending data, Sideways data, or High-Volatility data.

---

## 5. Dimension 2: Market Regime Taxonomy

The **Market Regime** defines the environmental market state under which a model is designed to operate.

### 5.1. Formal Regime Entity Schema

Every market regime in AruMLStudio is uniquely identified and registered in the **Regime Registry**:

```json
{
    "regime_id": "R001",
    "regime_name": "TREND",
    "regime_version": 1,
    "regime_family": "DIRECTIONAL_MOMENTUM",
    "description": "Strong directional price momentum with sustained order flow imbalance and low strike variance.",
    "detection_method": "RULE_BASED",
    "detection_spec": {
        "rule_expression": "abs(ema_slope_60) > 1.5 AND vwap_distance_pct > 0.35 AND adx_14 > 25.0",
        "primary_features": ["spot_ema_60_slope", "vwap_distance_pct", "adx_14", "futures_basis_zscore"]
    },
    "regime_lifecycle": "ACTIVE",
    "created_at": "2026-08-19T00:00:00Z",
    "updated_at": "2026-08-19T00:00:00Z"
}
```

### 5.2. Baseline Regime Catalog:

```
┌───────────┬───────────────────┬───────────────────┬─────────────────────────────────────────────────────────┐
│ Regime ID │ Regime Name       │ Detection Method  │ Primary Characterization / Defining Market Dynamics     │
├───────────┼───────────────────┼───────────────────┼─────────────────────────────────────────────────────────┤
│ `R000`    │ `ALL_REGIMES`     │ DEFAULT_UNIVERSAL │ Universal baseline model evaluated across all samples.  │
│ `R001`    │ `TREND`           │ HYBRID (Rule+ML)  │ Directional continuation, ADX > 25, VWAP divergence.    │
│ `R002`    │ `SIDEWAYS`        │ HYBRID (Rule+ML)  │ Mean-reverting, range-bound oscillator, low ADX < 18.   │
│ `R003`    │ `HIGH_VOLATILITY` │ VOL_SURFACE_CALC  │ Realized IV > 85th percentile, wide bid-ask, IV spike.  │
│ `R004`    │ `LOW_VOLATILITY`  │ VOL_SURFACE_CALC  │ Realized IV < 25th percentile, compressed straddle.     │
│ `R005`    │ `BREAKOUT`        │ LIQUIDITY_MOM     │ Volatility compression breakout with volume explosion.  │
│ `R006`    │ `REVERSAL`        │ MICROSTRUCTURE    │ Exhaustion divergence at major support/resistance.      │
│ `R007`    │ `EXPIRY_PINNING`  │ OPTION_GAMMA      │ Gamma pinning near max-pain strike on expiry afternoon. │
└───────────┴───────────────────┴───────────────────┴─────────────────────────────────────────────────────────┘
```

### 5.3. Discovered Regime Extensibility (The Research Factory Invariant)
The architecture does not limit regimes to predefined static names. The Autonomous Research Factory can discover empirical market micro-clusters:
- Example: **`R017`**: *"High IV expansion + accelerating gamma + abnormal call volume"*.
- It receives an allocated ID (`R017`), a parametric definition, and a classification model (`RC017`), allowing downstream models to associate with `R017` seamlessly without changing the database schema or UI layout.

---

## 6. Dimension 3: Model Population (Governance & Hierarchy Tier)

Model Population tracks the maturity, validation standing, and role of the model package within the platform:

```
                  ┌────────────────────────────────────────┐
                  │           MODEL POPULATIONS            │
                  └───────────────────┬────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌──────────────────┐        ┌──────────────────┐          ┌──────────────────┐
│   EXPERIMENTAL   │        │    VALIDATED     │          │    PRODUCTION    │
│   (Exploratory)  │        │   (Benchmarked)  │          │    (Governed)    │
└───────┬──────────┘        └─────────┬────────┘          └─────────┬────────┘
        │                             │                             │
  • Novel Hypotheses            • Passed Walk-Forward         • Regime Champion
  • Auto-Generated Runs         • Passed Out-of-Sample        • Regime Challenger
  • Training Sandbox            • Cross-Context Validated     • Live Inference
```

1. **`EXPERIMENTAL`**: Speculative models generated during hyperparameter sweeps, feature subset explorations, or novel architectures.
2. **`VALIDATED`**: Models that completed full out-of-sample holdout validation and meet minimal statistical robustness criteria.
3. **`CHALLENGER`**: Validated models competing directly against the current regime champion in backtesting and paper simulation.
4. **`CHAMPION`**: The single highest-ranking, human-governed production model for a specific `(market, sampling_interval, task_type, regime_id)` tuple.

---

## 7. Dimension 4: Lifecycle Status (Operational State)

Lifecycle Status indicates the operational usability of the model package:

1. **`CANDIDATE`**: Training completed; awaiting validation or governance review.
2. **`ACTIVE`**: Fully approved and operational for research, benchmarking, or inference.
3. **`DEGRADED`**: Exhibited performance decay or distribution drift in recent production validation.
4. **`DEPRECATED`**: Formally phased out; preserved for historical replay and provenance audit.
5. **`RETIRED`**: Archived; cannot be selected for live execution or new benchmarks.

---

## 8. Permanent Model Identity & Metadata Specification

Model identity is **cryptographically anchored and metadata-driven**, never derived from transient folder names or arbitrary UI strings.

### 8.1. Canonical Model Package Metadata Structure (`config.json` / `registry.json`):

```json
{
    "model_id": "MD000142",
    "model_name": "DIR_TREND_5M_WF_XGB_1168F",
    "version": 1,
    "model_family_id": "MF_NIFTY_3S_DIR_TREND",
    
    "task": {
        "task_type": "DIRECTION_CLASSIFIER",
        "target": "label_up_5m",
        "target_type": "BINARY_CLASSIFICATION",
        "prediction_horizon": "5m",
        "loss_function": "binary:logistic"
    },
    
    "regime": {
        "regime_id": "R001",
        "regime_name": "TREND",
        "regime_version": 1,
        "regime_scope": "SPECIALIZED"
    },
    
    "market_context": {
        "market": "NIFTY",
        "sampling_interval_sec": 3,
        "underlying_asset": "NIFTY_OPTIONS_CHAIN"
    },
    
    "lineage": {
        "feature_project_id": "all",
        "base_pipeline_id": "PL_0001",
        "experimental_pipeline_id": "PL_0007",
        "pipeline_snapshot_id": "e4a8b29c17df302e",
        "dataset_snapshot_hash": "a1f9c84e62b083dc",
        "feature_count": 1168,
        "feature_selection_preset": "NIFTY_3S_TOP_GAIN_1168"
    },
    
    "training": {
        "algorithm": "xgboost",
        "algorithm_version": "2.0.3",
        "validation_strategy": "walk_forward_cv",
        "train_trading_days": 45,
        "holdout_trading_days": 15,
        "hyperparameters": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8
        }
    },
    
    "governance": {
        "population": "VALIDATED",
        "status": "CHALLENGER",
        "regime_champion_rank": 2,
        "approved_by": "HUMAN_GOVERNANCE",
        "registered_at": "2026-08-19T00:00:00Z"
    }
}
```

---

## 9. Model Naming Convention (Presentation Layer)

While `model_id` (`MD000142`) is the authoritative programmatic key, human researchers benefit from structured, self-describing presentation names:

### 9.1. Recommended Standard Name Formula:
$$\text{NAME} = \langle\text{TASK}\rangle\_\langle\text{REGIME}\rangle\_\langle\text{HORIZON}\rangle\_\langle\text{VALIDATION}\rangle\_\langle\text{ALGORITHM}\rangle\_\langle\text{FEATURES}\rangle$$

### 9.2. Real Examples:
- `DIR_TREND_5M_WF_XGB_1168F` (Direction Classifier, Trend Regime, 5-min Horizon, Walk-Forward, XGBoost, 1168 Features)
- `DIR_SIDEWAYS_5M_WF_CAT_842F` (Direction Classifier, Sideways Regime, 5-min Horizon, Walk-Forward, CatBoost, 842 Features)
- `REGIME_ALL_15M_WF_LGB_156F` (Regime Classifier, All Regimes, 15-min Horizon, Walk-Forward, LightGBM, 156 Features)
- `TB_VOLATILE_1M_WF_XGB_420F` (Triple Barrier Classifier, High Volatility Regime, 1-min Horizon, Walk-Forward, XGBoost, 420 Features)

---

## 10. Regime Model Hierarchy & Inference Topology

The ultimate target runtime architecture combines specialized regime models dynamically:

```
                                  LIVE MARKET TICK STREAM
                                             │
                                             ▼
                             REGIME CLASSIFIER / DETECTOR (RC001)
                             Evaluates: Volatility, ADX, VWAP, Imbalance
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               ▼                             ▼                             ▼
         [TREND REGIME]              [SIDEWAYS REGIME]            [HIGH VOL REGIME]
       Regime State: R001            Regime State: R002           Regime State: R003
               │                             │                             │
               ▼                             ▼                             ▼
       TREND CHAMPION MODEL          SIDEWAYS CHAMPION MODEL       HIGH-VOL CHAMPION MODEL
      (DIR_TREND_5M_WF_XGB)         (DIR_SIDEWAYS_5M_WF_CAT)      (DIR_HIGHVOL_5M_WF_LGB)
               │                             │                             │
               └─────────────────────────────┼─────────────────────────────┘
                                             ▼
                                 CONFIDENCE / SIZING FILTER
                                             │
                                             ▼
                                  FINAL DIRECTIONAL SIGNAL
                                 (Trade Entry / Exit / Size)
```

---

## 11. Model Research Lab UI Architecture (Future Design)

The Model Research Lab UI will evolve from a 2-tab layout (`Regression` vs. `Triple Barrier`) into a **faceted, multi-dimensional exploration studio**:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          MODEL RESEARCH REGISTRY                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Primary Task Filter Tabs:                                                                                   │
│ [ All Models ] [ Direction Classifiers ] [ Regime Classifiers ] [ Regression ] [ Triple Barrier ] [ Conf ] │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Faceted Filter Bar:                                                                                         │
│ Market: [ NIFTY ▼ ]   Interval: [ 3s ▼ ]   Regime: [ All Regimes | Trend | Sideways | High Vol ▼ ]          │
│ Population: [ All | Champion | Challenger | Validated | Experimental ▼ ]   Status: [ Active ▼ ]             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Model Table:                                                                                                │
│ Model ID │ Name                     │ Task       │ Regime   │ Pop.       │ Status  │ Dir %  │ AUC   │ Action│
├──────────┼──────────────────────────┼────────────┼──────────┼────────────┼─────────┼────────┼───────┼───────┤
│ MD000142 │ DIR_TREND_5M_WF_XGB_1168F│ DIRECTION  │ TREND    │ CHAMPION 👑│ ACTIVE  │ 62.4%  │ 0.684 │ [Lab] │
│ MD000143 │ DIR_TREND_5M_WF_CAT_1168F│ DIRECTION  │ TREND    │ CHALLENGER │ ACTIVE  │ 61.8%  │ 0.679 │ [Lab] │
│ MD000144 │ DIR_SIDE_5M_WF_CAT_842F  │ DIRECTION  │ SIDEWAYS │ CHAMPION 👑│ ACTIVE  │ 58.1%  │ 0.631 │ [Lab] │
│ MD000145 │ REGIME_ALL_15M_WF_LGB    │ REGIME_CLS │ ALL      │ CHAMPION 👑│ ACTIVE  │ 74.2%  │ 0.792 │ [Lab] │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Champion / Challenger Governance Protocol

For every unique operational context defined by the key:
$$\mathbf{K}_{\text{regime}} = (\text{Market}, \text{Sampling Interval}, \text{Task Type}, \text{Prediction Horizon}, \text{Regime ID})$$

The platform maintains an explicit competitive ranking:
1. **Regime Champion ($R_1$)**: The single operational model currently holding highest out-of-sample edge.
2. **Regime Challenger ($R_2$)**: The primary alternative model being benchmarked against the Champion.
3. **Backup Candidates ($R_3, R_4$)**: Validated fallback models with distinct feature subsets or algorithms.
4. **Experimental Candidates ($R_5+$)**: Speculative models undergoing automated research training.

### Promotion from Challenger to Champion requires:
- Statistical superiority on true unseen validation days ($p < 0.05$).
- Lower or equivalent Expected Calibration Error ($\text{ECE}$).
- Explicit **Human Governance Review Sign-Off** (strictly maintaining the non-autonomous production boundary).

---

## 13. Research Factory Integration (Document 12 Alignment)

The model taxonomy enables the autonomous research engine to iterate over structured search vectors:

```
                            AUTONOMOUS RESEARCH ENGINE
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
      Sample Subsets            Feature Pipelines          Model Hypotheses
      (Regime-Stratified)       (PL_0001 + PL_0002+)       (XGB, CatBoost, LGB)
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        ▼
                            TRAIN & EVALUATE BATCH
                                        │
                                        ▼
                           TAG: Task Type & Regime ID
                                        │
                                        ▼
                         PERSIST METRICS IN analysis.db
                                        │
                                        ▼
                         RANK: Champion / Challenger Matrix
                                        │
                                        ▼
                       FORMULATE NEXT GENERATION HYPOTHESIS
```

---

## 14. Relationships with Existing AruMLStudio Architecture

The Model Taxonomy cleanly integrates with all existing foundations:

1. **Feature Registry (`FRxxxx`)**: Provides canonical inputs.
2. **Base Pipeline (`PL_0001`)**: Provides the immutable, governed production baseline transforms.
3. **Experimental Pipelines (`PL_0002+`)**: Generates novel features for experimental model variations.
4. **Feature Recommendation & Decision Engine (Phases 1–3C)**: Governs which features enter model candidate sets.
5. **Phase 3D Governance**: Governs feature graduation into Registry and Base Pipeline.
6. **Model Registry & Research Lab**: Hosts the multi-regime champion/challenger population.
7. **Production Validation**: Evaluates models on unseen days, feeding evidence back to both features and models.

---

## 15. Backward Compatibility Strategy

To ensure **zero regression on existing model packages**:

1. **Legacy Model Package Resolution**:
   - If a model package lacks `task` or `regime` metadata in its `config.json`:
     - If `target` starts with `future_ltp_` or `ormp_return_` $\implies$ `task_type = "REGRESSION"`.
     - If `target` starts with `label_up_` or `label_down_` $\implies$ `task_type = "DIRECTION_CLASSIFIER"`.
     - If `strategy_id == "triple_barrier"` or `target == "label_id"` $\implies$ `task_type = "TRIPLE_BARRIER"`.
     - `regime_id` defaults safely to `"R000"` (`ALL_REGIMES` / `UNSPECIFIED`).
     - `population` defaults to `"EXPERIMENTAL"`.
     - `status` defaults to `"ACTIVE"`.
2. **No On-Disk File Rewrites**: Legacy packages continue to function with in-memory metadata adapters without modifying historical JSON files.

---

## 16. Database & Registry Architecture Impact

1. **SQLite `analysis.db` (`champion_history`)**:
   - Additive optional columns: `task_type TEXT`, `regime_id TEXT`, `regime_name TEXT`, `population TEXT`.
   - Existing tables (`model_registry`, `model_history`) remain 100% backward compatible.
2. **Dedicated Regime Registry (`regime_registry_store.json`)**:
   - Stores authoritative regime definitions, parameters, and detection expressions.
3. **Research Memory (`analysis.db` - Phase 4D)**:
   - Stores multi-model, multi-regime cross-validation benchmarks and longitudinal leaderboards.

---

## 17. Comprehensive Lineage Architecture

Every future model package maintains complete end-to-end cryptographic traceability:

$$\text{Model ID} \longrightarrow \text{Task Type} \longrightarrow \text{Regime ID} \longrightarrow \text{Dataset Snapshot} \longrightarrow \text{Pipeline Snapshot (PL\_0001 + PL\_XXXX)} \longrightarrow \text{Validation Hash}$$

```
   MODEL (MD000142)
      ├── Task: DIRECTION_CLASSIFIER
      ├── Target: label_up_5m
      ├── Regime: R001 (TREND, v1)
      ├── Market Context: NIFTY, 3s, 5m Horizon
      ├── Dataset Snapshot Hash: a1f9c84e62b083dc
      ├── Pipeline Snapshot ID: e4a8b29c17df302e (PL_0001 + PL_0007)
      ├── Feature Presets: 1,168 Features
      ├── Training Engine: XGBoost v2.0.3 (Walk-Forward CV)
      ├── Production Validation: Unseen 15-Day Hash Match
      ├── Historical Evidence DB: Recommendation Score +84.2, Confidence 0.88
      └── Governance Tier: VALIDATED / CHALLENGER
```

---

## 18. Roadmap Position & Subsystem Dependencies

In the context of the **Phase 4 Master Roadmap** ([`Doc 11`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/11-PHASE_4_MASTER_ROADMAP.md)):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PHASE 4 ROADMAP TOPOLOGY                                 │
├──────────────┬─────────────────────────────────────────────────────────┬───────────────┤
│ Phase        │ Title                                                   │ Dependency    │
├──────────────┼─────────────────────────────────────────────────────────┼───────────────┤
│ Phase 4A     │ Higher-Order Option Surface Engine                      │ Independent   │
│ Phase 4B     │ Composite Non-Linear Feature Selection                  │ 4A Optional   │
│ Phase 4C.1   │ Model Taxonomy & Regime Registry (THIS SPECIFICATION)   │ Foundation    │
│ Phase 4C.2   │ Model Lab Population Awareness (Registry/Experimental)  │ Requires 4C.1 │
│ Phase 4D     │ Persistent Multi-Model Benchmarking in analysis.db      │ Requires 4C   │
│ Phase 4E     │ Automated Project Recommendations                       │ Requires 4D   │
│ Phase 4F     │ Automated Model Discovery & Fine-Tuning                 │ ✅ VERIFIED   │
│ Phase 4G     │ Lineage & Registry Integrity Auditor                    │ Requires 4C/4D│
│ Phase 4H     │ Continuous Registry Watcher                             │ Requires 4G   │
└──────────────┴─────────────────────────────────────────────────────────┴───────────────┘
```

> **Recommendation**: "Model Taxonomy & Regime-Aware Model Registry" serves as **Phase 4C.1**, providing the essential schema and taxonomy foundation for Phase 4C.2 (Population Awareness) and Phase 4D (Multi-Model Persistent Benchmarking).

---

## 19. Documentation Impact Map

When Phase 4C.1 is eventually implemented, the following documents will be synchronized:

1. **[`docs/00-ABOUT-ARUMLSTUDIO.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/00-ABOUT-ARUMLSTUDIO.md)**: Update Section 10 with the 4-dimensional model taxonomy.
2. **[`docs/01-FINAL_ARCHITECTURE_DIAGRAM.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/01-FINAL_ARCHITECTURE_DIAGRAM.md)**: Add Regime Registry and multi-regime inference flow.
3. **[`docs/03-REGISTRIES_ARCHITECTURE.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/03-REGISTRIES_ARCHITECTURE.md)**: Document `regime_registry_store.json` schema.
4. **[`docs/06-CREATE_MODEL_TRAINING_FEATURE_STUDIO_ACTUAL_FLOW.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/06-CREATE_MODEL_TRAINING_FEATURE_STUDIO_ACTUAL_FLOW.md)**: Document Task Type and Regime selection dropdowns in Create Model UI.
5. **[`docs/09-MODEL_RESEARCH_LAB_ACTUAL_WORKFLOW.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/09-MODEL_RESEARCH_LAB_ACTUAL_WORKFLOW.md)**: Document faceted filtering in Model Registry panel.
6. **[`docs/11-PHASE_4_MASTER_ROADMAP.md`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/docs/11-PHASE_4_MASTER_ROADMAP.md)**: Update Phase 4C specification with Phase 4C.1 deliverables.

---

## 20. Future Test Strategy (Pre-Implementation Specification)

When implementation begins, the following comprehensive test modules will be required:

1. **`test_model_taxonomy_schema.py`**:
   - Validates Task Type enum validation (`DIRECTION_CLASSIFIER`, `REGIME_CLASSIFIER`, etc.).
   - Validates that task types and regimes are strictly separated.
2. **`test_regime_registry.py`**:
   - Validates CRUD operations, JSON serialization, and versioning for `regime_registry_store.json`.
3. **`test_legacy_model_backward_compatibility.py`**:
   - Asserts that legacy models without regime metadata resolve safely to `R000 / UNSPECIFIED` without errors or file modifications.
4. **`test_regime_model_champion_ranking.py`**:
   - Asserts deterministic grouping and ranking of models by `(market, interval, task_type, regime_id)`.
5. **`test_model_lineage_traceability.py`**:
   - Asserts cryptographic lineage linking model ID to dataset hash, pipeline snapshot ID (`PL_0001` + `PL_XXXX`), and training config.

---

## 21. Pre-Implementation Decision Record (Summary)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PRE-IMPLEMENTATION DECISION RECORD                              │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Architectural Dimension  │ Authoritative Decision                                      │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1. Current Architecture  │ Hardcoded 2-tab UI (Regression / Triple Barrier); filenames │
│                          │ act as primary display IDs; no explicit regime taxonomy.    │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2. Proposed Architecture │ 4 Orthogonal Dimensions: Task Type, Market Regime,          │
│                          │ Model Population, and Lifecycle Status.                     │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3. Key Differences       │ Decoupled mathematics from market environment; canonical    │
│                          │ metadata model_id replaces filename source of truth.        │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4. New Entities          │ `RegimeRegistryStore` (`regime_id`, `detection_method`),    │
│                          │ `TaskType` enum, multi-regime Champion/Challenger matrix.   │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 5. Reused Foundations    │ Feature Registry, Base Pipeline PL_0001, Experimental       │
│                          │ Pipelines PL_0002+, Production Validation, Evidence DB.     │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 6. Implementation Guard  │ ⚠️ ZERO CODE TO BE IMPLEMENTED AT THIS STAGE.               │
│                          │ Awaiting explicit Phase 4C implementation directive.       │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```
