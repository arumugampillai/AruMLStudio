# AruMLStudio — Complete Visual System Architecture

---

## 1. Complete System Architecture Overview

```mermaid
flowchart TB
    %% STYLING AND THEME
    classDef dataBox fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef registryBox fill:#0f172a,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef ftBox fill:#1e293b,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef modelBox fill:#1e293b,stroke:#fbbf24,stroke-width:2px,color:#f8fafc;
    classDef studioBox fill:#1e293b,stroke:#f472b6,stroke-width:2px,color:#f8fafc;
    classDef strategyBox fill:#1e293b,stroke:#a78bfa,stroke-width:2px,color:#f8fafc;

    subgraph S_DATA ["1. DATA FOUNDATION"]
        RAW["Raw Tick Data (Options & Spot)"] --> MDB_BUILDER["Master Dataset Builder"]
        MDB_BUILDER --> MASTER_DB[("Master Dataset (SQLite)\n• Feature Registry Features Only\n• feature_project_id")]
    end

    subgraph S_REGISTRIES ["CORE REGISTRIES"]
        REG_FEAT[("Feature Registry\n(Registry Features)")]:::registryBox
        REG_DS[("Dataset Registry\n(Master, Analysis, Unseen)")]:::registryBox
        REG_MODEL[("Model Registry\n(Trained Model Packages)")]:::registryBox
    end

    subgraph S_FT ["2. FEATURE TRANSFORMATION LAB"]
        FT_MANUAL["Manual Pipeline\n(Lags, Diffs, Returns, Technicals)"]:::ftBox
        FT_AUTO["Auto Pipeline\n(Candidate Search & Generations)"]:::ftBox
        FT_ANALYSIS["Analysis Lab\n(Corr, MI, SHAP, HCA Clusters)"]:::ftBox
        
        MASTER_DB --> FT_MANUAL
        MASTER_DB --> FT_AUTO
        FT_MANUAL --> FT_ANALYSIS
        FT_AUTO --> FT_ANALYSIS
        FT_ANALYSIS --> ANALYSIS_DS[("Analysis Dataset (Parquet)\n• Feature Registry\n• Base Pipeline\n• Selected Experimental")]:::dataBox
    end

    subgraph S_TRAIN ["3. MODEL CREATION & TRAINING ENGINE"]
        CREATE_MODEL["Create Model UI\n• Target & Horizon\n• 3 Feature Tabs\n• Walk-Forward Setup\n• Algorithm & HPO"]:::modelBox
        TRAIN_ORCH["Training Orchestrator\n• Out-of-Fold Cross-Validation\n• Baseline vs Tuned Candidates\n• Champion Selection"]:::modelBox
        
        ANALYSIS_DS --> CREATE_MODEL
        CREATE_MODEL --> TRAIN_ORCH
        TRAIN_ORCH --> REG_MODEL
    end

    subgraph S_STUDIO ["4. FEATURE STUDIO & PRODUCTION VALIDATION"]
        direction TB
        FS_POST["Post-Training Automation\n(Imp, Dist, Drift Studios)"]:::studioBox
        FS_LOAD["Feature Studio Viewer\n(Decoupled JSON Viewer)"]:::studioBox
        PV_UNSEEN["Production Validation\n(Unseen Dataset Replay)"]:::studioBox
        PV_RULES["Feature Validation & Health\n(KEEP / WATCH / REMOVE)"]:::studioBox
        REC_HIST[("Recommendation History\n(Persistent Store)")]:::studioBox

        REG_MODEL --> FS_POST
        FS_POST --> FS_LOAD
        REG_MODEL --> PV_UNSEEN
        PV_UNSEEN --> PV_RULES
        PV_RULES --> REC_HIST
    end

    subgraph S_RESEARCH ["5. MODEL RESEARCH LAB"]
        M_LAB["Model Lab Workspace\n(model_lab_<name>_v1.db)"]:::modelBox
        PRED_GEN["Parallel Prediction Builder\n(Tick Predictions & Excursions)"]:::modelBox
        MFE_MAE["Trade Path Analytics\n(MFE, MAE, Time-to-Target)"]:::modelBox
        
        REG_MODEL -->|Open Research| M_LAB
        M_LAB --> PRED_GEN
        PRED_GEN --> MFE_MAE
    end

    subgraph S_STRATEGY ["6. STRATEGY LAB"]
        STRAT_REG["Strategy Registry\n(Profiles, Versions & Champions)"]:::strategyBox
        REPLAY_UI["Fold Replay\n(Tick-Level Trade Timeline)"]:::strategyBox
        PROGRAMS["Research Programs\n(Hypothesis Campaigns & Budgets)"]:::strategyBox
        EXP_PLANNER["Experiment Planner\n(8-Step Automation Pipeline)"]:::strategyBox
        SIM_ENGINE["Strategy Simulator\n(Win Rate, Drawdown, Profit Factor)"]:::strategyBox
        
        REG_MODEL -->|Prediction Feeds| REPLAY_UI
        STRAT_REG --> SIM_ENGINE
        PROGRAMS --> EXP_PLANNER
        EXP_PLANNER --> SIM_ENGINE
    end

    %% Registry Connections
    REG_FEAT -.-> MDB_BUILDER
    REG_DS -.-> MASTER_DB
    REG_DS -.-> ANALYSIS_DS
    REG_DS -.-> PV_UNSEEN
```

---

## 2. Complete End-to-End Data Lifecycle

```mermaid
flowchart LR
    classDef stage fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef artifact fill:#0f172a,stroke:#34d399,stroke-width:2px,color:#f8fafc;

    RAW["Raw Tick Data\n(Quotes & Spreads)"]:::stage
    MASTER["Master Dataset\n(SQLite)\n─────────────\nFeature Registry\nFeatures Only"]:::artifact
    FT["Feature\nTransformation\n(Manual + Auto)"]:::stage
    ANALYSIS["Analysis Dataset\n(Parquet)\n─────────────\n• Registry Features\n• Base Pipeline\n• Selected Experimental"]:::artifact
    CREATE["Create Model\n(Feature Selection\n& WF Split)"]:::stage
    TRAIN["Training &\nWalk-Forward"]:::stage
    MODEL_PKG["Model Package\n(models/<name>/)\n─────────────\n• Weights (.ubj)\n• Lineage (config.json)\n• Diagnostics Artifacts"]:::artifact
    CONSUMERS["Downstream Consuming Studios\n─────────────\n• Feature Studio\n• Production Validation\n• Model Research Lab\n• Strategy Lab"]:::stage

    RAW --> MASTER
    MASTER --> FT
    FT --> ANALYSIS
    ANALYSIS --> CREATE
    CREATE --> TRAIN
    TRAIN --> MODEL_PKG
    MODEL_PKG --> CONSUMERS
```

---

## 3. The Three Distinct Feature Populations

```mermaid
flowchart TB
    classDef regStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef baseStyle fill:#1e293b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef expStyle fill:#1e293b,stroke:#fbbf24,stroke-width:2px,color:#f8fafc;
    classDef unionStyle fill:#0f172a,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef noPromStyle fill:#450a0a,stroke:#f87171,stroke-width:2px,stroke-dasharray: 5 5,color:#fca5a5;

    subgraph POP1 ["1. FEATURE REGISTRY FEATURES"]
        F1["Feature Registry\n(Canonical Feature Definitions)"]:::regStyle
        F1_ID["Governed by feature_project_id"]:::regStyle
        F1_LOC["Materialized in Master Dataset\n(Raw Tick Data Foundation)"]:::regStyle
        F1 --> F1_ID --> F1_LOC
    end

    subgraph POP2 ["2. BASE PIPELINE FEATURES"]
        F2["Base Pipeline\n(Accepted Pipeline Population)"]:::baseStyle
        F2_ID["Active Baseline Pipeline State"]:::baseStyle
        F2_LOC["Non-Experimental Base Population"]:::baseStyle
        F2 --> F2_ID --> F2_LOC
    end

    subgraph POP3 ["3. SELECTED EXPERIMENTAL PIPELINE"]
        F3["Selected Experimental Pipeline\n(Experimental Feature Candidates)"]:::expStyle
        F3_ID["Governed by pipeline_id + pipeline_snapshot_id"]:::expStyle
        F3_LOC["Generated via Manual / Auto Transformation"]:::expStyle
        F3 --> F3_ID --> F3_LOC
    end

    subgraph NO_PROMOTION ["LIFECYCLE BOUNDARY (CURRENT STATE)"]
        NO_PROM["NO Automatic Experimental → Base Promotion\n(Promotion Workflow is NOT Implemented)"]:::noPromStyle
    end

    subgraph CONTAINER ["DATASET CONSUMPTION BOUNDARY"]
        MASTER_DS[("Master Dataset (SQLite)\n• Feature Registry Features Only")]:::regStyle
        ANALYSIS_DS[("Analysis Dataset (Parquet)\n• Feature Registry\n• Base Pipeline\n• Selected Experimental")]:::unionStyle
        MODEL_FEAT["Model Selected Features\n(User-Selected Subset from Analysis Dataset)"]:::unionStyle
    end

    F1_LOC --> MASTER_DS
    MASTER_DS --> ANALYSIS_DS
    F2_LOC --> ANALYSIS_DS
    F3_LOC --> ANALYSIS_DS
    ANALYSIS_DS --> MODEL_FEAT
```

---

## 4. Model Creation &rarr; Training &rarr; Model Registry

```mermaid
flowchart TD
    classDef step fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef artifact fill:#0f172a,stroke:#fbbf24,stroke-width:2px,color:#f8fafc;

    A["Analysis Dataset Selection"]:::step --> B["Target & Horizon Definition\n(e.g. future_ltp_5m)"]:::step
    B --> C["Feature Selection\n(Tab 1: Registry | Tab 2: Base | Tab 3: Experimental)"]:::step
    C --> D["Walk-Forward Split Configuration\n(Expanding / Rolling Folds + Holdout %)"]:::step
    D --> E["Algorithm Selection & Hyperparameter Grid\n(XGBoost / LightGBM / CatBoost + HPO)"]:::step
    E --> F["Train Model Action\n(Background Daemon Thread)"]:::step
    
    subgraph TRAIN_ENGINE ["Training Orchestration Engine"]
        F --> G["Load Parquet (Slices X, y)"]
        G --> H["Sanitize Features & Drop NaNs"]
        H --> I["Walk-Forward Folds Evaluation"]
        I --> J["Optuna HPO Optimization"]
        J --> K["Train Baseline Candidate vs. Tuned Candidate"]
        K --> L["Re-Evaluate on Walk-Forward Folds"]
        L --> M["Crown Champion Model Winner"]
    end

    subgraph ARTIFACTS ["Model Package Serialization (models/<name>/)"]
        M --> N["config.json (Lineage & Snapshots)"]:::artifact
        M --> O["model.ubj (Champion Weights)"]:::artifact
        M --> P["metrics.json (OOF & Holdout Scores)"]:::artifact
        M --> Q["feature_importance.csv (Native Splits/Gains)"]:::artifact
        M --> R["feature_importance_studio/ (Native, Permutation, SHAP)"]:::artifact
        M --> S["feature_distribution_studio/ (Holdout Distributions)"]:::artifact
        M --> T["feature_drift_studio/ (Train vs Holdout Drift)"]:::artifact
        M --> U["diagnostics_studio/ (Summary & Narrative)"]:::artifact
    end

    N --> REG[("Model Registry")]:::artifact
    O --> REG
    P --> REG
```

---

## 5. Model Registry &rarr; Feature Studio &rarr; Unseen Validation

```mermaid
flowchart TD
    classDef modelStyle fill:#1e293b,stroke:#fbbf24,stroke-width:2px,color:#f8fafc;
    classDef studioStyle fill:#1e293b,stroke:#f472b6,stroke-width:2px,color:#f8fafc;
    classDef unseenStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef evalStyle fill:#0f172a,stroke:#34d399,stroke-width:2px,color:#f8fafc;

    MODEL[("Selected Model from Model Registry")]:::modelStyle
    
    subgraph IN_SAMPLE ["In-Sample Post-Training Feature Studio"]
        FS_IMP["Feature Importance Studio\n(Native, Permutation, SHAP)"]:::studioStyle
        FS_DIST["Feature Distribution Studio\n(Holdout Moments & Missingness)"]:::studioStyle
        FS_DRIFT["Feature Drift Studio\n(WF Train vs Holdout KS-Test / Wasserstein)"]:::studioStyle
        FS_DIAG["Diagnostics Studio\n(Automated Root-Cause Diagnostic Rules)"]:::studioStyle
        
        MODEL --> FS_IMP
        MODEL --> FS_DIST
        MODEL --> FS_DRIFT
        FS_IMP --> FS_DIAG
        FS_DIST --> FS_DIAG
        FS_DRIFT --> FS_DIAG
    end

    subgraph OUT_OF_SAMPLE ["Out-of-Sample Production Validation"]
        LINEAGE["Extract Model Lineage\n• feature_project_id\n• pipeline_id & snapshot_id\n• selected features\n• seen training days"]:::unseenStyle
        
        HASH["Compute Identity Hash\nSHA256(Master DB + Unseen Days + Lineage)[:8]"]:::unseenStyle
        
        RESOLVE["Resolve or Generate Unseen Dataset\nunseen_<slug>_<hash>.parquet"]:::unseenStyle
        
        REPLAY["Replay Model Inference on Unseen Days\n• Holdout Rank vs Unseen Rank\n• Relative Importance Drop\n• Unseen Distribution Shift"]:::unseenStyle
        
        MODEL --> LINEAGE
        LINEAGE --> HASH
        HASH --> RESOLVE
        RESOLVE --> REPLAY
    end

    subgraph DECISION ["Tri-Population Feature Health Evaluation"]
        REPLAY --> VAL_REG["Feature Registry Features\n(KEEP / WATCH / REMOVE)"]:::evalStyle
        REPLAY --> VAL_BASE["Base Pipeline Features\n(KEEP / WATCH / REMOVE)"]:::evalStyle
        REPLAY --> VAL_EXP["Selected Experimental Features\n(KEEP / WATCH / REMOVE)"]:::evalStyle
    end
```

---

## 6. Feature Recommendation Lifecycle (Current Implementation)

```mermaid
flowchart TD
    classDef action fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef record fill:#0f172a,stroke:#fbbf24,stroke-width:2px,color:#f8fafc;
    classDef boundary fill:#450a0a,stroke:#f87171,stroke-width:2px,stroke-dasharray: 5 5,color:#fca5a5;

    PV["Production Validation Engine"]:::action
    RULES["Evaluation Rules Matrix\n(Rank Degradation + Importance Drop + Drift)"]:::action
    
    REC["Generate Observational Recommendations\n• KEEP (Stable Forward Generalization)\n• WATCH (Moderate Performance Drop)\n• REMOVE (Severe Drop / Negative Contribution)"]:::record

    PV --> RULES
    RULES --> REC

    subgraph PERSISTENCE ["Recommendation History Persistence"]
        REC --> SAVE["Update Registry Recommendations Button"]:::action
        SAVE --> STORE[("feature_recommendation_history.json\n────────────────────────────────\n• feature_type (registry/base/exp)\n• remove_runs count\n• keep_runs count\n• remove_models list")]:::record
    end

    subgraph USER_ACTIONS ["User Actions (Manual Review)"]
        STORE --> UI_VIEW["Feature Studio / Validation Table"]:::action
        UI_VIEW --> ACT_REMOVE["Remove Selected\n(Manual selection by User)"]:::action
        UI_VIEW --> ACT_IGNORE["Ignore Recommendation\n(Retain feature)"]:::action
    end

    subgraph NOT_IMPLEMENTED ["System Boundaries (NOT Implemented in Current Code)"]
        direction TB
        NO_DEL["NO Automatic Feature Deletion"]:::boundary
        NO_RET["NO Automatic Feature Retirement"]:::boundary
        NO_PROM["NO Automatic Experimental → Base Promotion"]:::boundary
    end
```

---

## 7. Strategy Lab Architecture & Subsystems

```mermaid
flowchart TB
    classDef stratStyle fill:#1e293b,stroke:#a78bfa,stroke-width:2px,color:#f8fafc;
    classDef storeStyle fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef engineStyle fill:#1e293b,stroke:#34d399,stroke-width:2px,color:#f8fafc;

    subgraph INPUTS ["Strategy Lab Inputs"]
        MODEL_REG[("Model Registry\n(models/<name>/)")]:::storeStyle
        PRED_STORE[("Prediction Runs Store\n(prediction_runs/registry.db)")]:::storeStyle
        MODEL_REG --> PRED_STORE
    end

    subgraph SUBSYSTEMS ["Strategy Lab Workbenches"]
        direction TB
        
        subgraph S1 ["1. Strategies Subsystem"]
            STRAT_UI["Strategy Registry Panel"]:::stratStyle
            STRAT_STORE[("strategies/registry.db\n• strategy_profiles\n• strategy_versions")]:::storeStyle
            STRAT_UI <--> STRAT_STORE
        end

        subgraph S2 ["2. Replay Subsystem"]
            REPLAY_PANEL["Fold Replay Panel"]:::stratStyle
            TIMELINE["Tick-Level Trade Timeline\n(Sparklines, P&L Curves, Excursions)"]:::stratStyle
            PRED_STORE --> REPLAY_PANEL
            REPLAY_PANEL --> TIMELINE
        end

        subgraph S3 ["3. Research Programs"]
            PROG_UI["Research Program Panel"]:::stratStyle
            PROG_STORE[("research_program.db\n• research_programs\n• research_campaigns")]:::storeStyle
            PROG_COORD["Campaign Coordinator\n(Automated Parameter Sweeps)"]:::stratStyle
            PROG_UI <--> PROG_STORE
            PROG_STORE --> PROG_COORD
        end

        subgraph S4 ["4. Experiment Planner"]
            PLAN_UI["Experiment Planner Panel"]:::stratStyle
            PLAN_STORE[("experiment_pipeline.db\n• proposals\n• templates\n• jobs")]:::storeStyle
            PLAN_PIPE["8-Step Automation Pipeline\n(Prep → Clone → Train → WF → Sim → Report → KB → Done)"]:::stratStyle
            PLAN_UI <--> PLAN_STORE
            PLAN_STORE --> PLAN_PIPE
        end
    end

    subgraph SIMULATION ["Execution & Analytics Engine"]
        SIM_ENG["Strategy Simulator\n(strategy_simulator/engine.py)"]:::engineStyle
        METRICS["Trade & Portfolio Metrics\n• Win Rate % | Net P&L\n• Profit Factor | Max Drawdown\n• Sharpe / Sortino / Calmar"]:::engineStyle
        
        STRAT_STORE --> SIM_ENG
        PRED_STORE --> SIM_ENG
        PROG_COORD --> SIM_ENG
        PLAN_PIPE --> SIM_ENG
        SIM_ENG --> METRICS
    end
```

---

## 8. Complete AruMLStudio Master System Map

```mermaid
flowchart TD
    %% Big Picture Full System Flow
    classDef raw fill:#1e293b,stroke:#94a3b8,stroke-width:2px,color:#f8fafc;
    classDef data fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef ft fill:#1e293b,stroke:#34d399,stroke-width:2px,color:#f8fafc;
    classDef model fill:#1e293b,stroke:#fbbf24,stroke-width:2px,color:#f8fafc;
    classDef fs fill:#1e293b,stroke:#f472b6,stroke-width:2px,color:#f8fafc;
    classDef res fill:#1e293b,stroke:#fb923c,stroke-width:2px,color:#f8fafc;
    classDef strat fill:#1e293b,stroke:#a78bfa,stroke-width:2px,color:#f8fafc;

    RAW_DATA["RAW TICK DATA\n(Options & Underlying Spot)"]:::raw
    
    MASTER_DS[("MASTER DATASET\n(SQLite Master DB)\n• Feature Registry Features Only\n• feature_project_id")]:::data
    
    FT_LAB["FEATURE TRANSFORMATION\n(Manual + Auto Pipelines + Analysis Lab)"]:::ft
    
    ANALYSIS_DS[("ANALYSIS DATASET\n(Parquet Feature Matrix)\n• Feature Registry\n• Base Pipeline\n• Selected Experimental")]:::data
    
    CREATE_MODEL["CREATE MODEL & TRAINING\n• Target & Horizon Setup\n• 3 Feature Population Tabs\n• Walk-Forward Split & HPO\n• Champion Selection"]:::model
    
    MODEL_REG[("MODEL REGISTRY\n(Serialized Packages in models/<name>/)\n• Model Weights (.ubj)\n• Lineage & Config Snapshots\n• Post-Training Diagnostics")]:::model

    RAW_DATA --> MASTER_DS
    MASTER_DS --> FT_LAB
    FT_LAB --> ANALYSIS_DS
    ANALYSIS_DS --> CREATE_MODEL
    CREATE_MODEL --> MODEL_REG

    %% 3 Downstream Pillars
    MODEL_REG --> FS_BRANCH["FEATURE STUDIO & VALIDATION"]:::fs
    MODEL_REG --> MLAB_BRANCH["MODEL RESEARCH LAB"]:::res
    MODEL_REG --> STRAT_BRANCH["STRATEGY LAB"]:::strat

    %% Feature Studio Details
    subgraph PILLAR_FS ["Feature Studio & Production Validation"]
        FS_BRANCH --> FS_DIAGNOSTICS["In-Sample Diagnostics\n• Importance (Native/Perm/SHAP)\n• Distribution (Moments/Missing)\n• Drift (KS-Test / Wasserstein)"]:::fs
        FS_BRANCH --> FS_UNSEEN["Unseen Dataset Validation\n(unseen_<slug>_<hash>.parquet)"]:::fs
        FS_UNSEEN --> FS_RECOMMEND["Health Recommendations\n(KEEP / WATCH / REMOVE)\n• feature_recommendation_history.json"]:::fs
    end

    %% Model Research Details
    subgraph PILLAR_MLAB ["Model Research Lab (Model Lab)"]
        MLAB_BRANCH --> MLAB_WORKSPACE["Dedicated SQLite Workspace\n(model_lab_<name>_v1.db)"]:::res
        MLAB_WORKSPACE --> MLAB_PRED["Parallel Prediction Builder\n(Tick Predictions & Excursions)"]:::res
        MLAB_PRED --> MLAB_ANALYTICS["Trade Path Analytics\n• MFE / MAE Excursions\n• Time to Target / Strike Slice\n• Meta-Confidence Modeling"]:::res
    end

    %% Strategy Lab Details
    subgraph PILLAR_STRAT ["Strategy Lab Subsystems"]
        STRAT_BRANCH --> STRAT_CATALOG["Strategy Registry\n(strategies/registry.db)"]:::strat
        STRAT_BRANCH --> STRAT_REPLAY["Fold Replay\n(prediction_runs/registry.db)"]:::strat
        STRAT_BRANCH --> STRAT_CAMPAIGNS["Research Programs & Planner\n(research_program.db / experiment_pipeline.db)"]:::strat
        STRAT_CATALOG --> STRAT_SIM["Strategy Simulation Engine\n(P&L, Win Rate %, Drawdown, Profit Factor)"]:::strat
        STRAT_REPLAY --> STRAT_SIM
        STRAT_CAMPAIGNS --> STRAT_SIM
    end
```
