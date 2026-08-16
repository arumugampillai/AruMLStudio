# AruMLStudio Architecture & Dependency Boundaries

## 1. Overview & System Purpose
**AruMLStudio** is an autonomous Machine Learning Research Studio for high-frequency financial time-series engineering, dataset synthesis, and machine learning model exploration.

It is designed as a **completely independent, self-contained desktop and processing framework** with zero runtime code dependencies on legacy trading platforms (such as AruNeo).

---

## 2. Layered Architecture & Dependency Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│               master_dataset_tk (GUI Layer)                 │
│  - MLResearchStudioApp (Main Studio)                        │
│  - CreateDatasetPanel, ResearchLabPanel, ModelExplorer      │
└──────────────────────────────┬──────────────────────────────┘
                               │ (calls into ML/orchestrator)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             chain_replay_ml (Core ML Engine)                │
│  - dataset_builder (Orchestrator, Projects, Transformations)│
│  - model_lab (Model Registry, Prediction Workers)           │
│  - training (XGBoost, LightGBM, CatBoost, Row Prune)        │
│  - post_training & performance (Numba acceleration)         │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
                ▼                             ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│    feature_intelligence      │ │     tick_data_paths        │
│  - Feature metadata catalog  │ │  - Central tick DB locator │
│  - FIC SQLite store          │ │  - Master dataset naming   │
└───────────────┬──────────────┘ └─────────────┬──────────────┘
                │                              │
                └──────────────┬───────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Foundation / Core Anchors                   │
│  - path_config.py (ensure_ml_studio_paths, APPS_DIR)        │
│  - __version__.py (Single Source of Version Truth)          │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Rules:
1. **Unidirectional Flow**:
   - `master_dataset_tk` may import `chain_replay_ml`, `feature_intelligence`, `path_config`, `__version__`.
   - `chain_replay_ml` may import `feature_intelligence`, `path_config`, `__version__`, but **never** imports `master_dataset_tk` UI widgets.
   - `feature_intelligence` is an independent feature catalog and does not import `master_dataset_tk`.
   - Foundation modules (`path_config.py`, `__version__.py`) have **zero** dependencies on upper layers.

---

## 3. Four-Tier Storage Boundary

To ensure complete data integrity, machine portability, and isolation:

| Tier | Path | Policy |
| :--- | :--- | :--- |
| **1. Application Code** | `AruMLStudio/` (`apps/` as import root) | Python source files and launchers. **Never contains user data.** |
| **2. Application State** | `%APPDATA%\AruMLStudio\` | Flat JSON settings, window state, UI prefs, FIC SQLite database. **Never uses legacy AruNeo path.** |
| **3. Market / Tick Data** | `D:\data\ticks\` (or user-configured) | Raw tick SQLite DBs. **Data only — NEVER inserted into `sys.path`.** |
| **4. Master Datasets & Models** | `D:\data\master_dataset\`, `D:\data\model_research\` | Parquet datasets, SQLite job logs, model check-points. **Data only.** |

---

## 4. Environment Variable Precedence

All environment variables follow strict 3-tier precedence:

$$\mathbf{ARUMLSTUDIO\_*} \longrightarrow \mathbf{ARUNEO\_*}\;(\text{legacy fallback}) \longrightarrow \mathbf{Application\;Default}$$

---

## 5. Forbidden Dependency Constraints

1. **No Legacy Application Dependencies**: Production code must **never** import `angelone.*` application code or `AruNeo.*`.
2. **No Developer Machine Paths**: Production code must **never** hardcode `C:\Users\...`, `PycharmProjects`, or developer-specific drive mappings.
3. **No sys.path Hijacking**: External market data and chart directories must **never** be placed at `sys.path[0]` ahead of `apps/`.
