# AruMLStudio Strategy Lab — Actual Implementation & Workflow Audit

---

## 1. Strategy Lab Overview & Architecture

The **Strategy Lab** (accessible via the main navigation in [`apps/master_dataset_tk/app.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/app.py)) is the strategy formulation, trade execution simulation, backtesting replay, and automated hypothesis research engine in **AruMLStudio**.

```
                                      STRATEGY LAB NAVIGATION
                                                 │
      ┌────────────────────────┬─────────────────┴───────┬────────────────────────┐
      ▼                        ▼                         ▼                        ▼
[1. Strategies]          [2. Replay]           [3. Research Programs]   [4. Experiment Planner]
 • Strategy Registry      • Fold Replay         • Hypothesis Tracking    • Proposal Generation
 • Version Lifecycle      • Prediction Runs     • Campaign Automation    • Template Freezing
 • Simulation Engine      • Tick-Level Replay   • Parameter Optimization • Multi-Step Pipelines
 • Leaderboard Matrix     • Trade Timeline      • Budget & Stop Criteria • Auto Knowledge Base
```

### 1.1. Core Problem & Lifecycle Role
While **Create Model** trains ML models to forecast directional price movements, **Strategy Lab** solves the subsequent trading problem:
> *"How do we convert raw model price forecasts into risk-managed, positive-expectancy options trading strategies with exact entry filters, exit conditions, stop losses, position sizing, and transaction fee models?"*

### 1.2. High-Level Data Ingestion & Production

```
                                    DATA FLOW INTO STRATEGY LAB
                                    
Model Registry (models/<name>/)  ──► Prediction Runs Store (prediction_runs/registry.db)
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │     STRATEGY LAB      │
                                    │ • Strategies Registry │
                                    │ • Simulation Engine   │
                                    │ • Replay Timeline     │
                                    │ • Campaign Jobs       │
                                    └───────────┬───────────┘
                                                │
                                                ▼
Trade Results & P&L Curves ──► Knowledge Base & Strategy Champions (strategies/registry.db)
```

- **Consumes**: Model prediction outputs (`prediction_runs`), historical tick/candle feeds from Master Dataset, and user-defined strategy parameters.
- **Produces**: Simulated trades, equity curves, risk-adjusted performance metrics, champion strategy configurations, and structured research knowledge.

---

## 2. Strategies Subsystem

The **Strategies** section is implemented in `StrategiesPanel` ([`apps/master_dataset_tk/strategy_lab_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/strategy_lab_panel.py)) and hosts four tabs:
1. **Strategies Tab**: `StrategyRegistryPanel` ([`apps/master_dataset_tk/strategy_registry_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/strategy_registry_panel.py))
2. **Prediction Runs Tab**: `PredictionRunsPanel` ([`apps/master_dataset_tk/prediction_runs_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/prediction_runs_panel.py))
3. **Simulation Tab**: `StrategySimulationPanel` ([`apps/master_dataset_tk/strategy_simulation_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/strategy_simulation_panel.py))
4. **Leaderboard Tab**: `ResearchLabPanel` ([`apps/master_dataset_tk/research_lab_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/research_lab_panel.py))

### 2.1. Strategy Configuration Schema
A Strategy profile is a structured JSON document defining entry, exit, stop, target, and risk rules, defined in [`apps/chain_replay_ml/strategy_registry/schema.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/strategy_registry/schema.py):

```json
{
  "name": "OTM Premium Buyer",
  "description": "Current expiry, OTM ±15, premium ₹15–30, 3s cadence, 30s max hold, 8% target / 5% stop.",
  "entry": {
    "direction": "long",
    "min_confidence": 0.0,
    "premium_min": 15.0,
    "premium_max": 30.0,
    "atm_band": 15,
    "expiry": "current",
    "entry_cadence_sec": 3,
    "minimum_predicted_move_pct": 0.0,
    "use_regression": true,
    "option_types": ["CE", "PE"]
  },
  "exit": {
    "mode": "target_stop_hold"
  },
  "stop": {
    "stop_loss_pct": 5.0
  },
  "target": {
    "target_profit_pct": 8.0,
    "use_predicted_ltp": false
  },
  "hold_time": {
    "max_hold_sec": 30
  },
  "confidence": {
    "min_signal_strength": 0.0,
    "use_model_confidence": false
  },
  "position_size": {
    "lots": 1,
    "qty_per_lot": 65
  },
  "execution": {
    "fees_mode": "rupee_charges",
    "slippage_ticks": 0,
    "allow_averaging": false
  }
}
```

### 2.2. Strategy Storage & Database Schema
Stored in `<chart_data_dir>/strategies/registry.db` via `StrategyRegistryStore` ([`apps/chain_replay_ml/strategy_registry/store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/strategy_registry/store.py)):

| Table Name | Purpose | Important Columns |
|---|---|---|
| `strategy_profiles` | Strategy family metadata and current champion pointer. | `strategy_id` (PK), `display_name`, `slug`, `current_version_id`, `champion_config_hash`, `status`, `updated_on` |
| `strategy_versions` | Immutable version records of strategy configurations. | `version_id` (PK), `strategy_id` (FK), `version_number`, `parent_version_id`, `lifecycle` (`new_strategy`, `clone`, `edit`, `calibration`), `config_hash`, `config_json` |

### 2.3. Strategy Creation & Lifecycle Flow
```
User clicks "New Strategy" / "Clone"
    │
    ▼
StrategyRegistryPanel._create_from_template()
    │
    ▼
StrategyRegistryService.create_strategy_from_template()  [strategy_registry/service.py]
    │
    ├── Computes SHA256 config_hash
    ├── Creates strategy_profiles row
    └── Inserts version 1 into strategy_versions
    │
    ▼
Strategy becomes available in Simulation, Replay, and Campaigns
```

### 2.4. Strategy Simulation & Performance Metrics
Located in [`apps/chain_replay_ml/strategy_simulator/engine.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/strategy_simulator/engine.py) and [`metrics.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/strategy_simulator/metrics.py):
- **Simulation Execution**: Evaluates entry gates on prediction rows; traverses forward ticks to trigger target profit %, stop loss %, or max hold timeout exits.
- **Computed Metrics**:
  - `total_trades`, `winning_trades`, `losing_trades`
  - `win_rate_pct`, `loss_rate_pct`
  - `net_pnl`, `gross_pnl`, `brokerage_charges`
  - `profit_factor` ($\frac{\sum \text{Gains}}{\sum |\text{Losses}|}$)
  - `max_drawdown_rupees`, `max_drawdown_pct`
  - `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`
  - `average_trade_pnl`, `average_win`, `average_loss`, `win_loss_ratio`
  - `max_consecutive_wins`, `max_consecutive_losses`

---

## 3. Replay Subsystem

Implemented in `FoldReplayPanel` ([`apps/master_dataset_tk/fold_replay_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/fold_replay_panel.py)) and backed by [`apps/chain_replay_ml/fold_research/trade_replay.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/fold_research/trade_replay.py).

### 3.1. Replay Purpose & Data Source
Replay provides tick-level visualization of model predictions, trade entries, excursions, and exits across historical walk-forward folds and prediction runs.

- **Primary Source**: `<chart_data_dir>/prediction_runs/registry.db` ([`apps/chain_replay_ml/prediction_runs/store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/prediction_runs/store.py)) containing recorded champion model predictions across trading folds.
- **Tick Resolution**: 3-second or 6-second tick snapshots matching the Master Dataset sampling interval.

### 3.2. Execution Flow

```
Prediction Run Selected (prediction_runs/registry.db)
         │
         ▼
Fold Selected (Fold 1, Fold 2, ... Fold N)
         │
         ▼
Replay Engine matches Strategy Entry Rules against Predictions
         │
         ▼
Generates Timeline Events:
  • prediction (model predicted move)
  • trade_entry (long option purchase)
  • excursion (tick-by-tick unrealized P&L)
  • trade_exit (target profit hit / stop loss / hold expiry)
         │
         ▼
Renders Interactive Canvas:
  • Price Sparklines & Candlesticks
  • Entry/Exit markers
  • P&L Step Chart
  • Confidence Level Bars
```

- **Replay Capabilities**: Replays **model-generated predictions filtered by strategy rules**. (It evaluates how a strategy would have performed on the exact predictions generated by a trained model).

---

## 4. Research Programs Subsystem

Implemented in `ResearchProgramPanel` ([`apps/master_dataset_tk/research_program_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/research_program_panel.py)) and [`apps/chain_replay_ml/fold_research/research_program.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/fold_research/research_program.py).

### 4.1. What is a Research Program?
A **Research Program** is a structured, multi-campaign hypothesis container that coordinates automated optimization campaigns (e.g., `"Stop Optimization"`, `"Target Hurdle Search"`, `"Premium Window Sweep"`).

### 4.2. Campaign Metadata & Schema (`research_program.db`)
Stored in `<chart_data_dir>/research/research_program.db` via `ResearchProgramStore` ([`apps/chain_replay_ml/fold_research/research_program_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/fold_research/research_program_store.py)):

| Field | Meaning | Example |
|---|---|---|
| `campaign_id` | Alphanumeric unique identifier | `"CMP_0007"` |
| `name` | Human-readable campaign label | `"Stop Loss Optimization (3%–8%)"` |
| `status` | Campaign lifecycle state | `"waiting"`, `"running"`, `"completed"`, `"failed"`, `"stopped"` |
| `research_question`| Formal hypothesis tested | `"Does tightening stop loss from 5% to 3.5% reduce max drawdown without degrading win rate?"` |
| `importance` | Priority rating | `"high"`, `"medium"`, `"low"` |
| `objective_json` | Target performance goal | `{"metric": "profit_factor", "target_value": 1.75}` |
| `budget_json` | Resource & time constraints | `{"max_experiments": 25, "max_minutes": 60}` |
| `stopping_json` | Early stopping rule | `{"patience": 5, "min_improvement_pct": 1.0}` |

### 4.3. Campaign Execution Lifecycle
Campaigns are scheduled and run via `ResearchCampaignCoordinator` ([`apps/master_dataset_tk/research_campaign_coordinator.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/research_campaign_coordinator.py)) and `campaign_job_runner.py`:
1. **Waiting**: Campaign enqueued in coordinator queue.
2. **Running**: Spawns worker jobs iterating over strategy parameter grids.
3. **Completed / Stopped**: Halts when target objective is achieved or budget/stopping criteria are met.

---

## 5. Experiment Planner Subsystem

Implemented in `ExperimentPlannerPanel` ([`apps/master_dataset_tk/experiment_planner_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/experiment_planner_panel.py)) and backed by [`apps/chain_replay_ml/fold_research/experiment_pipeline.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/fold_research/experiment_pipeline.py).

### 5.1. The 4-Tier Concept Hierarchy

```
┌────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Concept            │ Definition & Responsibility                                            │
├────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ **Strategy**       │ Concrete set of entry/exit/stop rules applied to prediction feeds.     │
│ **Research Program**│ High-level business goal (e.g. "Maximize OTM Profit Factor").         │
│ **Experiment**     │ Single parameterized trial comparing candidate changes against baseline│
│ **Replay**         │ Visual tick-by-tick inspection tool for a single trade/fold execution. │
└────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

### 5.2. Experiment Pipeline Storage (`experiment_pipeline.db`)
Stored in `<chart_data_dir>/research/experiment_pipeline.db` via `ExperimentPipelineStore` ([`apps/chain_replay_ml/fold_research/experiment_pipeline_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/fold_research/experiment_pipeline_store.py)):

| Table Name | Purpose | Key Columns |
|---|---|---|
| `experiment_proposals` | Generated hypotheses proposing parameter modifications. | `proposal_id`, `goal`, `tags_json`, `available_json`, `selected_json`, `score_json`, `status` |
| `experiment_templates` | Frozen immutable experiment specifications ready for execution. | `template_id`, `proposal_id`, `accepted_changes_json`, `routing_json`, `baseline_json`, `score_json` |
| `experiment_jobs` | Active and historical execution records of multi-step jobs. | `job_id`, `template_id`, `status`, `current_step`, `progress_json`, `results_json`, `comparison_json` |

### 5.3. Multi-Step Execution Pipeline (`JOB_STEPS`)
When an experiment job runs, it advances sequentially through 8 steps:
1. `preparing`: Validates template, baseline runs, and execution phases.
2. `cloning`: Clones strategy version and model config into isolated experiment versions.
3. `training`: (If model changes involved) Retrains model with walk-forward validation.
4. `walk_forward`: Evaluates metrics across folds and feeds prediction run.
5. `simulation`: Replays strategy rules on predictions to generate trade metrics.
6. `research_report`: Generates comparison report against baseline.
7. `knowledge_base`: Auto-closure: computes information gain, root cause, and extracts findings.
8. `complete`: Frozen verdict ready for user review.

---

## 6. Complete End-to-End Flow Across Strategy Lab

```
[1] User creates baseline strategy: "OTM Premium Buyer v1" (Strategies Tab)
                    │
                    ▼
[2] Research Campaign created: "Stop Loss Optimization" (Research Programs Tab)
                    │
                    ▼
[3] Experiment Planner generates proposals & freezes templates (Experiment Planner Tab)
                    │
                    ▼
[4] Multi-step job executes simulation sweeps across prediction runs
                    │
                    ▼
[5] Strategy Simulator evaluates trades & calculates metrics (P&L, Drawdown, Profit Factor)
                    │
                    ▼
[6] User inspects detailed trade entries/excursions in Replay (Replay Tab)
                    │
                    ▼
[7] Winning configuration set as Champion: "OTM Premium Buyer v2 (Champion)"
```

---

## 7. Data Storage & Metadata Inventory

| Component | Storage Type | File / Path | Key Tables / Schemas | Purpose |
|---|---|---|---|---|
| **Strategies** | SQLite | `strategies/registry.db` | `strategy_profiles`, `strategy_versions` | Persistent catalog of strategy configurations & versions. |
| **Prediction Runs** | SQLite | `prediction_runs/registry.db` | `prediction_runs`, `prediction_folds` | Model prediction outputs used as simulation inputs. |
| **Research Programs**| SQLite | `research/research_program.db` | `research_programs`, `research_campaigns`, `campaign_proposals` | Multi-campaign hypothesis tracking & optimization runs. |
| **Experiment Planner**| SQLite | `research/experiment_pipeline.db`| `experiment_proposals`, `experiment_templates`, `experiment_jobs` | Multi-step experiment lifecycle & comparison store. |
| **Knowledge Base** | SQLite | `research/knowledge.db` | `knowledge_entries`, `findings` | Automated research takeaways & rules discovered during experiments. |

---

## 8. Relationship with Model Research Lab & Feature Studio

### 8.1. Relationship with Model Research Lab (Model Lab)
- **Shared Simulation Engine**: Both Strategy Lab Simulation and Model Research Lab use the simulation algorithms in `apps/chain_replay_ml/strategy_simulator/` (`engine.py`, `metrics.py`).
- **Data Source Difference**:
  - Model Research Lab operates on its model-specific `D:\data\model_research\model_lab_<name>_v1.db` (`prediction_dataset`).
  - Strategy Lab Replay operates on `<chart_data_dir>/prediction_runs/registry.db`.

### 8.2. Relationship with Feature Studio
- **NO DIRECT CODE CONNECTION**: Strategy Lab does **not** import, call, or depend on `feature_importance_studio`, `feature_distribution_studio`, `feature_drift_studio`, or `diagnostics_studio`.
- **No Recommendation Feedback**: Strategy Lab results do not modify `feature_recommendation_history.json` or alter `KEEP / WATCH / REMOVE` recommendations.

---

## 9. Current Implementation vs. Not Implemented

### 9.1. CURRENTLY IMPLEMENTED (Verified in Code)
1. **Strategy Registry**: Creation, cloning, versioning, champion assignment, and SQLite persistence in `strategies/registry.db`.
2. **Strategy Simulation**: Full rule evaluation (premium filters, cadence, target %, stop %, max hold, slippage, rupee fees) computing win rates, profit factor, and drawdowns.
3. **Fold Replay UI**: Interactive timeline, candlestick sparklines, P&L step charts, and trade cards.
4. **Research Programs**: Automated multi-campaign coordinator with budget constraints and early stopping.
5. **Experiment Planner**: End-to-end 8-step pipeline (proposals &rarr; templates &rarr; jobs &rarr; comparisons &rarr; knowledge extraction).

### 9.2. NOT IMPLEMENTED / NOT FOUND
1. **Live Broker Order Execution**: Strategy Lab is strictly a research and backtesting simulator; it does **not** send live orders to exchange APIs.
2. **Direct Feature Recommendation Generation**: Strategy backtests do not feed automated feature retirement/promotion recommendations.

---

## 10. Source Code Reference Map

| Subsystem | Primary Source File | Primary Class / Function | Storage Path |
|---|---|---|---|
| **Strategy UI** | [`apps/master_dataset_tk/strategy_registry_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/strategy_registry_panel.py) | `StrategyRegistryPanel` | `strategies/registry.db` |
| **Strategy Store** | [`apps/chain_replay_ml/strategy_registry/store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/strategy_registry/store.py) | `StrategyRegistryStore` | `strategies/registry.db` |
| **Simulation Engine**| [`apps/chain_replay_ml/strategy_simulator/engine.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/strategy_simulator/engine.py) | `run_strategy_simulation` | In-memory / UI results |
| **Replay UI** | [`apps/master_dataset_tk/fold_replay_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/fold_replay_panel.py) | `FoldReplayPanel` | `prediction_runs/registry.db` |
| **Trade Replay** | [`apps/chain_replay_ml/fold_research/trade_replay.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/fold_research/trade_replay.py) | `load_trade_replay_session` | In-memory timeline |
| **Programs UI** | [`apps/master_dataset_tk/research_program_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/research_program_panel.py) | `ResearchProgramPanel` | `research/research_program.db` |
| **Program Store** | [`apps/chain_replay_ml/fold_research/research_program_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/fold_research/research_program_store.py) | `ResearchProgramStore` | `research/research_program.db` |
| **Planner UI** | [`apps/master_dataset_tk/experiment_planner_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/experiment_planner_panel.py) | `ExperimentPlannerPanel` | `research/experiment_pipeline.db` |
| **Pipeline Store** | [`apps/chain_replay_ml/fold_research/experiment_pipeline_store.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/fold_research/experiment_pipeline_store.py) | `ExperimentPipelineStore` | `research/experiment_pipeline.db` |

---

## 11. Final End-to-End Architecture Diagram

```
Master Dataset (Raw Tick Database)
       │
       ▼
Feature Transformation (Transforms & Pipelines)
       │
       ▼
Analysis Dataset (Parquet Feature Matrix)
       │
       ▼
Create Model (Walk-Forward Training & HPO)
       │
       ▼
Model Registry (Trained Model Packages & Prediction Runs)
       │
       ├──────────────────────────────► Feature Studio (Static Diagnostics & Drift)
       │
       ├──────────────────────────────► Model Research Lab (Row-by-Row Tick Excursions)
       │
       └──────────────────────────────► Strategy Lab
                                              │
                                              ├── Strategies (Entry, Exit & Stop Rules)
                                              ├── Replay (Tick-Level Trade Timeline)
                                              ├── Research Programs (Hypothesis Optimization)
                                              └── Experiment Planner (8-Step Automation Pipeline)
```
