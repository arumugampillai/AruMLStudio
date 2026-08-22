# DATA ROOT AND STORAGE ARCHITECTURE
## Single Authoritative Data Root & Consolidated Storage Specification

```
Document Version: 1.0.0
Author: DeepMind Agentic Pair Programmer / ML Systems Architecture
Status: AUTHORITATIVE ARCHITECTURAL SPECIFICATION & SYSTEM DESIGN
Target Base Path: C:\Users\admin\PycharmProjects\AruMLStudio
Target Document: docs/17-DATA_ROOT_AND_STORAGE_ARCHITECTURE.md
Target Canonical Data Root: D:\data (Configurable via Application Settings)
Related Documents: Docs 02, 03, 04, 05, 08, 12, 13, 14, 15, 16, 17, 18
Databases: analysis.db · feature_recommendation_evidence.db · angel_historic_bars.db · prediction_runs.db · master_dataset_*.db
Workstation Baseline: 16 GB RAM Local Workstation (Zero Cloud Dependencies, Deterministic Offline Execution)
```

---

## 0. Version History & Architectural Milestone

| Version | Date | Author | Status | Key Architectural Milestone |
|---|---|---|---|---|
| **1.0.0** | 2026-08-22 | Agentic ML Team | **Authoritative Design** | Complete Data Root Consolidation Specification. Establishes a single configurable Data Root (`D:\data`), eliminates path fragmentation, prevents duplicate database spawning, and standardizes directory hierarchies for all databases, registries, datasets, models, research campaigns, and prediction runs. |

---

## 1. Executive Purpose

The purpose of this architecture is to transition **ML Research Studio** from a fragmented, multi-path storage model to a **Single Configurable Data Root Architecture**.

Currently, persistent application data is scattered across:
- Source code directories (`apps/data`, `apps/static`)
- Legacy chart directories (`C:\...\AruNeo\angelone\chart\data`)
- Root drive paths (`D:\data\master_dataset`, `D:\data\ticks`)
- Ad-hoc paths constructed via `os.path.dirname(...)` and relative strings

This fragmentation led to severe bugs, including the accidental creation of parallel, empty `analysis.db` files (e.g. `angelone/analysis.db` vs `angelone/chart/data/analysis.db`) and confusing UI configuration with redundant directory pickers.

Under the **Canonical Data Root Architecture**:
1. The user configures exactly **ONE** application setting:
   $$\text{DATA ROOT} = \text{D:\textbackslash data}$$
2. The application derives **all** internal directories, database paths, registries, model stores, and research artifacts deterministically beneath this root.
3. Source code (`C:\Users\admin\PycharmProjects\AruMLStudio`) is strictly isolated from application persistent data (`D:\data`), enabling seamless cross-machine deployment and complete data portability.

---

## 2. Current-State Forensic Findings

A comprehensive read-only audit of the production filesystem and codebase revealed the following distribution of application data:

### 2.1 Storage Distribution by Location

| Location | Role / Content | File Count | Disk Footprint | Primary Assets |
|---|---|---|---|---|
| **`C:\...\angelone\chart\data`** | Active Primary Operational Store | 19 files, 15 subdirs | **19.82 GB** | • `analysis.db` (108.15 MB, 435k correlation rows)<br>• `feature_recommendation_evidence.db` (44.67 MB, 42k evidence rows)<br>• `angel_historic_bars.db` (207.75 MB, 1.66M bars)<br>• `feature_registry_store.json` (180 KB)<br>• `pipeline_registry_store.json` (10 KB)<br>• `datasets/` (11.82 GB Parquets)<br>• `chain_exports/` (5.07 GB JSONs)<br>• `ml_models/` (482.84 MB)<br>• `prediction_runs/registry.db` (653.88 MB)<br>• `strategy_runs/registry.db` (1,019.47 MB) |
| **`D:\data`** | High-Volume Telemetry & Tick Store | 1 file, 6 subdirs | **37.54 GB** | • `ticks/` (32.96 GB, 29 raw tick databases)<br>• `master_dataset/` (2.73 GB, `master_dataset_nifty_*.db`)<br>• `model_research/` (1.85 GB, `model_lab_*.db`)<br>• `nifty/` (5.70 MB, `nifty_spot_bars.db`)<br>• `pipeline_registry_store.json` (10 KB, duplicate) |
| **`C:\...\AruMLStudio\apps\data`** | Development / Bundled Seed Store | 4 items | **0.50 MB** | • `angel_historic_bars.db` (seed empty)<br>• `cache/`<br>• `models/` (sample files) |
| **`C:\...\angelone`** (Accidental) | Bug-Induced Phantom Root | 1 file | **0.20 MB** | • `analysis.db` (Accidentally spawned 204 KB empty database) |

---

## 3. Forensic Inventory of Every Persistent Data Path

| Artifact Name | Owner Module | Current Operational Path | Purpose | Type | Proposed Canonical Data Root Location | Migration Required? |
|---|---|---|---|---|---|---|
| **`analysis.db`** | `chain_replay_ml.research_memory`<br>`dataset_builder` | `angelone/chart/data/analysis.db` | Correlation, SHAP, mutual info, overnight campaigns, research registry, discovery pipelines, candidate rankings. | SQLite (WAL) | `D:\data\databases\analysis.db` | **YES** |
| **`feature_recommendation_evidence.db`** | `chain_replay_ml.production_validation`<br>`discovery_pipeline.bridge` | `angelone/chart/data/feature_recommendation_evidence.db` | Longitudinal empirical evidence, feature context summary, experimental lineage summary. | SQLite (WAL) | `D:\data\databases\feature_recommendation_evidence.db` | **YES** |
| **`angel_historic_bars.db`** | `chain_replay_ml.historic` | `angelone/chart/data/angel_historic_bars.db` | 1-minute historical OHLCV bars cache for underlying indices (1.66M rows). | SQLite (WAL) | `D:\data\databases\angel_historic_bars.db` | **YES** |
| **`prediction_runs/registry.db`** | `chain_replay_ml.prediction_runs` | `angelone/chart/data/prediction_runs/registry.db` | Out-of-sample forward prediction run results, evaluation metrics, and trading signals. | SQLite (WAL) | `D:\data\databases\prediction_runs.db` | **YES** |
| **`strategy_runs/registry.db`** | `chain_replay_ml.strategy_lab` | `angelone/chart/data/strategy_runs/registry.db` | Replay strategy backtest executions, trade logs, and equity curves. | SQLite (WAL) | `D:\data\databases\strategy_runs.db` | **YES** |
| **`master_dataset_nifty_*.db`** | `chain_replay_ml.dataset_builder` | `D:\data\master_dataset\master_dataset_nifty_*.db` | Pre-computed second-by-second master order-flow feature grids (3s, 6s, 9s, 10s). | SQLite (WAL) | `D:\data\datasets\master\master_dataset_nifty_*.db` | **YES** |
| **`angel_market_*.db`** | `tick_data_paths` | `D:\data\ticks\angel_market_*.db` | Raw packet-level tick databases recorded from WebSocket market feed. | SQLite (RO) | `D:\data\ticks\angel_market_*.db` | **NO** (Already in `D:\data\ticks`) |
| **`feature_registry_store.json`** | `chain_replay_ml.dataset_builder.feature_registry_store` | `angelone/chart/data/feature_registry_store.json` | Master Permanent Feature Registry (approved & promoted features). | JSON | `D:\data\registries\feature_registry_store.json` | **YES** |
| **`pipeline_registry_store.json`** | `chain_replay_ml.dataset_builder.pipeline_registry_store` | `angelone/chart/data/pipeline_registry_store.json` | Master Pipeline Registry (Base pipelines PL_0001..PL_0013 and discovery pipelines). | JSON | `D:\data\registries\pipeline_registry_store.json` | **YES** |
| **`.lifecycle_registry.db`** | `chain_replay_ml.model_registry` | `angelone/chart/data/models/.lifecycle_registry.db` | Model Registry metadata, deployment tags, and lifecycle states. | SQLite | `D:\data\registries\model_registry.db` | **YES** |
| **`analysis_*.parquet`** | `chain_replay_ml.dataset_builder` | `angelone/chart/data/datasets/*.parquet` | Generated tabular training and validation datasets. | Parquet | `D:\data\datasets\analysis\*.parquet` | **YES** |
| **`chain_NIFTY_*.json`** | `chain_replay_ml.export` | `angelone/chart/data/chain_exports/*.json` | Exported option chain snapshots. | JSON | `D:\data\datasets\exports\chain_*.json` | **YES** |
| **`ml_models/*.json`** | `chain_replay_ml.model_builder` | `angelone/chart/data/ml_models/*.json` | Training execution reports and hyperparameters. | JSON | `D:\data\models\research\*.json` | **YES** |
| **`model_lab_*.db`** | `chain_replay_ml.prediction_runs` | `D:\data\model_research\model_lab_*.db` | Model evaluation prediction outputs and residual diagnostics. | SQLite | `D:\data\predictions\datasets\model_lab_*.db` | **YES** |
| **`triple_barrier_run_*.parquet`**| `chain_replay_ml.label_runs` | `angelone/chart/data/label_runs/*.parquet` | Triple-barrier labeling outputs and event timestamps. | Parquet | `D:\data\datasets\labels\*.parquet` | **YES** |
| **`OpenAPIScripMaster.json`** | `chain_replay_ml.cache` | `angelone/chart/data/cache/OpenAPIScripMaster.json` | Angel One instrument and strike token master cache. | JSON | `D:\data\cache\OpenAPIScripMaster.json` | **YES** |
| **`ml_research_studio.json`** | `master_dataset_tk.project_config` | `%APPDATA%\AruMLStudio\ml_research_studio.json` | Global application user preferences and Data Root setting. | JSON | `%APPDATA%\AruMLStudio\config.json` | **NO** (User config root) |

---

## 4. Problems with the Current Path Architecture

```
                               CURRENT FRAGMENTED ARCHITECTURE (VULNERABLE)
                               
   ┌───────────────────────────┐         ┌─────────────────────────────┐         ┌───────────────────────────┐
   │  C:\...\AruMLStudio\apps  │         │  C:\...\angelone\chart\data │         │          D:\data          │
   │  (Repo / Source Code)     │         │  (Legacy Operational Store) │         │  (High-Volume Root)       │
   └─────────────┬─────────────┘         └──────────────┬──────────────┘         └─────────────┬─────────────┘
                 │                                      │                                      │
                 ▼                                      ▼                                      ▼
         APPS_DIR / static                       analysis.db                           ticks/
         CHART_DATA_ROOT                         feature_evidence.db                   master_dataset/
         Hardcoded lookups                       feature_registry.json                 model_research/
                 │                               pipeline_registry.json                        │
                 │                                      │                                      │
                 └───────────────────────┬──────────────┴──────────────────────────────────────┘
                                         │
                                         ▼
                 ┌─────────────────────────────────────────────────────────────┐
                 │ PATH BUGS & ACCIDENTAL PHANTOM DATABASES:                   │
                 │ • os.path.dirname(chart_dir) → spawned angelone\analysis.db │
                 │ • fallback chains look in apps\ vs data\ vs cwd             │
                 │ • D:\data contains duplicate pipeline_registry_store.json   │
                 │ • Settings UI requires 4 separate folder pickers            │
                 └─────────────────────────────────────────────────────────────┘
```

1. **Accidental Duplicate Databases**:
   When components used `os.path.dirname(chart_dir)` or relative lookups, a new empty database was silently created (e.g. `C:\...\angelone\analysis.db`).
2. **Ambiguous Ground Truth**:
   `pipeline_registry_store.json` existed in both `angelone/chart/data/` (active) and `D:\data/` (stale duplicate), risking state drift.
3. **Conflation of Code and Data**:
   `apps/path_config.py` defined `CHART_DATA_ROOT = APPS_DIR`, binding persistent runtime data directly into the git repository directory tree.
4. **Scattered Multi-Directory Settings**:
   The Settings UI forced researchers to configure "Project Folder", "Tick Data Folder", "Master Dataset Folder", "Chart Directory", and "Prediction Runs DB" separately, confusing users and introducing misconfigurations.

---

## 5. Canonical Data Root Architecture

Under the new architecture, persistent application data is completely consolidated under **ONE configurable root directory**:

$$\text{DATA ROOT} = \text{D:\textbackslash data}$$

```
D:\data\
├── databases\                    # All authoritative SQLite transactional & analytical stores
│   ├── analysis.db               # Primary analytical memory, correlation, SHAP, research registry, discovery
│   ├── feature_recommendation_evidence.db  # Longitudinal evidence store & graduation summaries
│   ├── angel_historic_bars.db    # Cached 1-minute historical candle bars
│   ├── prediction_runs.db        # Out-of-sample forward prediction runs and performance signals
│   └── strategy_runs.db          # Replay backtest executions and portfolio equity curves
│
├── registries\                   # Master immutable & authoritative definition registries
│   ├── feature_registry_store.json   # Master Permanent Feature Registry (FR_0001..FR_XXXX)
│   ├── pipeline_registry_store.json  # Master Pipeline Registry (PL_0001..PL_XXXX)
│   └── model_registry.db             # Registered candidate & production model metadata
│
├── datasets\                     # Parquet grids, master datasets, and dataset exports
│   ├── master\                   # Pre-computed second-by-second master order flow DBs
│   │   ├── master_dataset_nifty_3s.db
│   │   ├── master_dataset_nifty_6s.db
│   │   └── master_dataset_nifty_10s.db
│   ├── analysis\                 # Tabular ML training/validation Parquet datasets
│   │   ├── analysis_198r_171b_6s_20260820_223630.parquet
│   │   └── analysis_198r_171b_6s_20260820_223630.json
│   ├── labels\                   # Label run Parquets (Triple Barrier, fixed horizon)
│   └── exports\                  # Option chain and tabular CSV/JSON exports
│
├── models\                       # Model artifacts, serialized weights, and training telemetry
│   ├── production\               # Active deployed models (LightGBM, XGBoost, CatBoost, Torch)
│   ├── candidates\               # Candidate models undergoing out-of-sample validation
│   └── research\                 # Training reports, loss curves, and hyperparameter trials
│
├── research\                     # Autonomous research campaigns, snapshots, and dossiers
│   ├── campaigns\                # Overnight campaign state manifests and candidate specs
│   ├── discovery\                # Autonomous Discovery Pipeline evolutionary telemetry
│   ├── snapshots\                # Immutable research and dataset snapshot archives
│   └── dossiers\                 # Generated Markdown & JSON Morning Research Dossiers
│
├── predictions\                  # Prediction evaluation outputs and prediction datasets
│   ├── datasets\                 # Out-of-sample prediction database files (model_lab_*.db)
│   └── artifacts\                # Prediction probability matrices and signal arrays
│
├── ticks\                        # Raw WebSocket tick market feeds (angel_market_YYYY-MM-DD.db)
│
├── logs\                         # Application, worker, and background campaign execution logs
│
└── cache\                        # Temporary token instruments and computation caches
    └── OpenAPIScripMaster.json
```

---

## 6. Directory Specifications & Storage Governance

### 6.1 `databases/`
- **Purpose**: Authoritative transactional and analytical SQLite stores.
- **Allowed Files**: `*.db`, `*.db-wal`, `*.db-shm`.
- **Ownership**:
  - `analysis.db`: Owned by `research_memory`, `discovery_pipeline`, `overnight_campaign`.
  - `feature_recommendation_evidence.db`: Owned by `production_validation`, `discovery_pipeline.bridge`.
  - `prediction_runs.db`: Owned by `prediction_runs`.
  - `strategy_runs.db`: Owned by `strategy_lab`.
- **Governance Invariant**: **Zero duplicate databases.** All components must connect through the centralized connection manager.

### 6.2 `registries/`
- **Purpose**: Ground truth definitions for features, pipelines, and models.
- **Allowed Files**: `feature_registry_store.json`, `pipeline_registry_store.json`, `model_registry.db`.
- **Governance Invariant**:
  - `feature_registry_store.json` contains ONLY promoted, permanent features (`FR_*`).
  - `pipeline_registry_store.json` contains authoritative pipeline definitions (`PL_*`).
  - Discovery pipelines and experimental features ($DF\_*$) are NEVER directly written to `feature_registry_store.json`.

### 6.3 `datasets/`
- **Purpose**: High-throughput columnar datasets and pre-computed master tables.
- **Subdirectories**:
  - `master/`: Contains `master_dataset_*.db` files.
  - `analysis/`: Contains `analysis_*.parquet` and associated `*.json` metadata schemas.
  - `labels/`: Contains `triple_barrier_run_*.parquet`.
  - `exports/`: Contains exported option chain JSON/CSV data.

### 6.4 `models/`
- **Purpose**: Serialized machine learning models and training records.
- **Subdirectories**:
  - `production/`: Stored production weights (`.pkl`, `.onnx`, `.pt`, `.json`).
  - `candidates/`: Validated candidate model checkpoints.
  - `research/`: `training_report_*.json` and hyperparameter search histories.

### 6.5 `research/`
- **Purpose**: Autonomous discovery campaign outputs and research logs.
- **Subdirectories**:
  - `campaigns/`: Campaign configurations and run state.
  - `discovery/`: Generation mutation lineage logs.
  - `snapshots/`: Cryptographic snapshot archives (`DP_SNAP_*`).
  - `dossiers/`: Morning Research Dossier reports (`dossier_*.md`, `dossier_*.json`).

### 6.6 `predictions/`
- **Purpose**: Forward evaluation databases and prediction inference artifacts.
- **Subdirectories**:
  - `datasets/`: Prediction outputs (`model_lab_*.db`).
  - `artifacts/`: Prediction probability matrices (`.npy`, `.parquet`).

### 6.7 `ticks/`
- **Purpose**: Raw historical packet recordings from exchange feeds (`angel_market_YYYY-MM-DD.db`).
- **Access Mode**: Read-only during feature computation; append-only during live ingestion.

### 6.8 `logs/` and `cache/`
- **`logs/`**: Background worker logs, campaign execution traces, and audit logs.
- **`cache/`**: Instrument master files (`OpenAPIScripMaster.json`), ephemeral calculation caches.

---

## 7. Registry Ownership & Boundaries

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 REGISTRY OWNERSHIP BOUNDARIES                                    │
├────────────────────────────────┬────────────────────────────────┬────────────────────────────────┤
│      PERMANENT REGISTRY        │      PIPELINE REGISTRY         │      RESEARCH REGISTRY         │
├────────────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│ Storage:                       │ Storage:                       │ Storage:                       │
│ registries/                    │ registries/                    │ databases/analysis.db          │
│ feature_registry_store.json    │ pipeline_registry_store.json   │ (table: research_registry)     │
│                                │                                │                                │
│ Contents:                      │ Contents:                      │ Contents:                      │
│ • Approved FR_XXXX Features    │ • Base Pipelines (PL_0001..)   │ • Historical Campaign Runs     │
│ • Versioned Math Signatures    │ • Discovery Pipelines          │ • Overnight Benchmark Scores   │
│ • Domain Classifications       │ • Candidate Pipelines          │ • Multi-Generation Evolution   │
│ • Production Graduation Meta   │ • Feature Membership IDs       │ • Research Hypothesis Logs     │
│                                │ • AST Provenance Metadata      │                                │
│ Mutation Rule:                 │ Mutation Rule:                 │ Mutation Rule:                 │
│ STRICT GOVERNANCE PROMOTION    │ PIPELINE BUILDER / SYNTHESIS   │ AUTONOMOUS CAMPAIGN ENGINE     │
│ (Zero automatic DF_* writes)   │ (Authoritative composition)    │ (Continuous longitudinal sync) │
└────────────────────────────────┴────────────────────────────────┴────────────────────────────────┘
```

---

## 8. Path Resolution Architecture & Canonical API

To eliminate all relative pathing, `os.path.dirname()` bugs, and parallel database spawning, a single authoritative path resolver service is defined:

### 8.1 DataRootService API Specification

```python
"""Authoritative Centralized Path Resolver for ML Research Studio."""

from __future__ import annotations
import os
from typing import Literal

class DataRootService:
    """Single Source of Truth for all persistent application storage paths."""
    
    def __init__(self, data_root: str | None = None) -> None:
        self._data_root = os.path.abspath(os.path.normpath(data_root or self._load_default_root()))

    @property
    def data_root(self) -> str:
        return self._data_root

    def get_database_path(self, db_type: Literal["analysis", "feature_evidence", "angel_historic", "predictions", "strategies"]) -> str:
        db_map = {
            "analysis": "analysis.db",
            "feature_evidence": "feature_recommendation_evidence.db",
            "angel_historic": "angel_historic_bars.db",
            "predictions": "prediction_runs.db",
            "strategies": "strategy_runs.db",
        }
        return os.path.join(self._data_root, "databases", db_map[db_type])

    def get_registry_path(self, reg_type: Literal["feature", "pipeline", "model"]) -> str:
        reg_map = {
            "feature": "feature_registry_store.json",
            "pipeline": "pipeline_registry_store.json",
            "model": "model_registry.db",
        }
        return os.path.join(self._data_root, "registries", reg_map[reg_type])

    def get_datasets_dir(self, category: Literal["master", "analysis", "labels", "exports"] = "analysis") -> str:
        return os.path.join(self._data_root, "datasets", category)

    def get_models_dir(self, category: Literal["production", "candidates", "research"] = "research") -> str:
        return os.path.join(self._data_root, "models", category)

    def get_research_dir(self, category: Literal["campaigns", "discovery", "snapshots", "dossiers"] = "discovery") -> str:
        return os.path.join(self._data_root, "research", category)

    def get_predictions_dir(self, category: Literal["datasets", "artifacts"] = "datasets") -> str:
        return os.path.join(self._data_root, "predictions", category)

    def get_ticks_dir(self) -> str:
        return os.path.join(self._data_root, "ticks")

    def get_logs_dir(self) -> str:
        return os.path.join(self._data_root, "logs")

    def get_cache_dir(self) -> str:
        return os.path.join(self._data_root, "cache")

    def ensure_layout(self) -> None:
        """Idempotently create all canonical subdirectories beneath data_root."""
        for path in [
            os.path.join(self._data_root, "databases"),
            os.path.join(self._data_root, "registries"),
            os.path.join(self._data_root, "datasets", "master"),
            os.path.join(self._data_root, "datasets", "analysis"),
            os.path.join(self._data_root, "datasets", "labels"),
            os.path.join(self._data_root, "datasets", "exports"),
            os.path.join(self._data_root, "models", "production"),
            os.path.join(self._data_root, "models", "candidates"),
            os.path.join(self._data_root, "models", "research"),
            os.path.join(self._data_root, "research", "campaigns"),
            os.path.join(self._data_root, "research", "discovery"),
            os.path.join(self._data_root, "research", "snapshots"),
            os.path.join(self._data_root, "research", "dossiers"),
            os.path.join(self._data_root, "predictions", "datasets"),
            os.path.join(self._data_root, "predictions", "artifacts"),
            os.path.join(self._data_root, "ticks"),
            os.path.join(self._data_root, "logs"),
            os.path.join(self._data_root, "cache"),
        ]:
            os.makedirs(path, exist_ok=True)
```

---

## 9. Settings UI Design

The **Settings Panel** is simplified from 4 confusing path pickers down to **1 primary Data Root configuration** with real-time health telemetry:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     ⚙️ APPLICATION SETTINGS                                      │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  📁 APPLICATION DATA ROOT                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────┬───────────────────┐  │
│  │ D:\data                                                                │ [ Browse... ]     │  │
│  └────────────────────────────────────────────────────────────────────────┴───────────────────┘  │
│  All databases, registries, datasets, models, research, and telemetry resolve beneath this root.  │
│                                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 🟢 Storage Status: HEALTHY · Disk Free: 482.4 GB / 931.5 GB · Write Permissions: OK        │  │
│  │                                                                                             │  │
│  │ • Databases:  D:\data\databases\ (analysis.db: 108.1 MB · evidence.db: 44.7 MB)            │  │
│  │ • Registries: D:\data\registries\ (feature_registry: 180 KB · pipeline_registry: 10 KB)   │  │
│  │ • Datasets:   D:\data\datasets\ (145 analysis parquets · 3 master DBs)                      │  │
│  │ • Models:     D:\data\models\ (444 training reports · 393 model checkpoints)                │  │
│  │ • Ticks:      D:\data\ticks\ (29 market day databases · 32.96 GB)                           │  │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                  │
│  ⚙️ ADVANCED / EXTERNAL OVERRIDES (Optional)                                                      │
│  [ ] Override External Tick Data Path                                                            │
│      ┌────────────────────────────────────────────────────────────────────┬───────────────────┐  │
│      │ D:\data\ticks                                                      │ [ Browse... ]     │  │
│      └────────────────────────────────────────────────────────────────────┴───────────────────┘  │
│                                                                                                  │
│  [ 🔄 Verify Storage Integrity ]          [ 📦 Run Data Migration Assistant ]     [ Save Settings ]│
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Audit of Existing Codebase Consumers

| Component / Subsystem | Current Path Resolution Method | Target Canonical Resolution | Risk of Duplicate / Split DB |
|---|---|---|---|
| **ML Research Studio App** (`master_dataset_tk/app.py`) | `chart_dir = resolve_chart_dir()` | `data_root_service = get_data_root_service()` | **High** (Previous anchor for `chart/data`) |
| **Model Registry Panel** (`model_registry_panel.py`) | `chart_data_dir(self.chart_dir)` | `data_root_service.get_database_path("analysis")` | **Critical** (Previously spawned `angelone/analysis.db`) |
| **Discovery Dashboard** (`discovery_dashboard/service.py`) | `data_dir` parameter passed from UI | `data_root_service.data_root` | **Medium** (Relies on caller passing correct directory) |
| **Morning Research Dossier** (`morning_dossier/generator.py`) | `data_dir` parameter | `data_root_service.data_root` | **Low** (Reads `analysis.db` and `evidence.db`) |
| **Feature Recommendation Bridge** (`overnight_campaign/feature_evidence_bridge.py`) | Multi-path fallback chain (`os.path.join("apps", ...)` vs `data_dir`) | `data_root_service.get_database_path("feature_evidence")` | **Critical** (Fallback chain spawned empty evidence DBs) |
| **Research Memory Store** (`research_memory/db.py`) | `os.path.join(data_dir, "analysis.db")` | `data_root_service.get_database_path("analysis")` | **Medium** (Safe if data_dir is canonical) |
| **Evidence Store** (`production_validation/evidence_store.py`) | `os.path.join(data_dir, "feature_recommendation_evidence.db")` | `data_root_service.get_database_path("feature_evidence")` | **Medium** (Safe if data_dir is canonical) |
| **Pipeline Registry Store** (`dataset_builder/pipeline_registry_store.py`) | `os.path.join(data_dir, "pipeline_registry_store.json")` | `data_root_service.get_registry_path("pipeline")` | **Medium** (Stale duplicate exists in `D:\data`) |
| **Feature Registry Store** (`dataset_builder/feature_registry_store.py`) | `os.path.join(data_dir, "feature_registry_store.json")` | `data_root_service.get_registry_path("feature")` | **Medium** (Safe if data_dir is canonical) |
| **Tick Data Path Resolver** (`tick_data_paths.py`) | Fallback list (`D:\data\ticks`, `chart_dir/data`, `old/`) | `data_root_service.get_ticks_dir()` | **Low** (Already defaults to `D:\data\ticks`) |
| **Master Dataset Builder** (`dataset_builder/master_store.py`) | `D:\data\master_dataset` env/prefs lookup | `data_root_service.get_datasets_dir("master")` | **Low** (Already rooted on D:) |

---

## 11. Migration Strategy & Safeguards

```
                               SAFE 5-STAGE MIGRATION WORKFLOW
                               
   ┌───────────────────────┐     1. PRE-FLIGHT VALIDATION
   │  Pre-Flight Checks    │     • Verify D:\ has at least 50 GB free disk space
   │  & Source Discovery   │     • Compute SHA-256 checksums for all source databases
   └───────────┬───────────┘     • Run PRAGMA integrity_check on source SQLite DBs
               │
               ▼
   ┌───────────────────────┐     2. STRUCTURE PROVISIONING
   │  Layout Creation      │     • Provision D:\data\databases, registries, datasets, models, etc.
   │  beneath D:\data\     │     • Set restrictive directory permissions
   └───────────┬───────────┘
               │
               ▼
   ┌───────────────────────┐     3. ATOMIC COPY & CHECKSUM VERIFICATION
   │  Copy & Verify Pass   │     • Copy analysis.db → D:\data\databases\analysis.db
   │  (Zero Overwrite)     │     • Copy evidence.db → D:\data\databases\feature_evidence.db
   └───────────┬───────────┘     • Verify SHA-256(source) == SHA-256(destination)
               │
               ▼
   ┌───────────────────────┐     4. MANIFEST RECORDING & CONFIG SWITCH
   │  Atomic Configuration │     • Write migration_manifest.json with timestamp and hashes
   │  Switchover           │     • Atomically update %APPDATA%\AruMLStudio\ml_research_studio.json
   └───────────┬───────────┘     • Point DATA_ROOT = D:\data
               │
               ▼
   ┌───────────────────────┐     5. ARCHIVE & LOCK OLD LOCATIONS
   │  Post-Migration Lock  │     • Rename source files to *.migrated_backup
   │  & Clean Verification │     • Verify all UI tabs and autonomous engines load from D:\data
   └───────────────────────┘
```

### 11.1 Safeguard Rules
1. **Zero Data Loss**: Source files are NEVER deleted during migration; they are renamed with `.migrated_backup` timestamp suffixes only after full checksum verification.
2. **Checksum Enforcement**: Every SQLite database and JSON store must match source SHA-256 before the configuration pointer is switched.
3. **Database Integrity Audit**: Run `PRAGMA integrity_check;` on the destination database before releasing the connection.
4. **Rollback Strategy**: If verification fails at any stage, the configuration pointer remains on the old location, and the destination files are quarantined in `D:\data\.migration_failed_<timestamp>`.

---

## 12. Architectural Invariants

The consolidated Data Root architecture establishes six inviolable system invariants:

### Invariant 1: Single Data Root Invariant
All application-owned persistent data (databases, registries, datasets, models, research campaigns, predictions, logs, caches) MUST resolve beneath the configured `DATA_ROOT`.

### Invariant 2: Database Singleton Invariant
Each logical database (`analysis.db`, `feature_recommendation_evidence.db`, `prediction_runs.db`, `strategy_runs.db`, `angel_historic_bars.db`) MUST have exactly ONE authoritative path on the filesystem.

### Invariant 3: Zero Code/Data Conflation Invariant
The application source repository (`C:\Users\admin\PycharmProjects\AruMLStudio`) MUST NEVER be used as a storage location for runtime datasets, models, or databases.

### Invariant 4: No Duplicate Database Invariant
No module or test may silently initialize a fallback SQLite database in the current working directory, module folder, or relative parent directory. If a path fails to resolve, the application must raise an explicit configuration error.

### Invariant 5: Universal Path Resolver Invariant
Every UI panel, background research runner, discovery pipeline, model trainer, and test suite MUST resolve storage locations via `DataRootService` instead of ad-hoc path concatenation.

### Invariant 6: Immutability of Master Registries Invariant
`feature_registry_store.json` and `pipeline_registry_store.json` beneath `D:\data\registries\` can ONLY be modified by explicit, authorized promotion and creation routines. Autonomous experimental research NEVER writes unpromoted $DF\_*$ features to `feature_registry_store.json`.

---

## 13. Recommended Implementation Phases

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                RECOMMENDED IMPLEMENTATION ROADMAP                                │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Phase 1: Canonical Path Resolver & Config Schema                                                 │
│ • Implement DataRootService in apps/chain_replay_ml/core/data_root.py.                            │
│ • Update ml_research_studio.json to store single data_root = "D:\\data".                         │
│ • Unit tests for path resolution and directory layout creation.                                  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Phase 2: Database & Registry Path Consolidation                                                  │
│ • Refactor research_memory/db.py, production_validation/evidence_store.py, and                   │
│   dataset_builder/pipeline_registry_store.py to consume DataRootService.                         │
│ • Eliminate all ad-hoc fallback path lists (e.g. os.path.join("apps", ...)).                     │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Phase 3: Datasets, Models, & Prediction Engine Consolidation                                     │
│ • Wire Master Dataset Builder, Parquet generator, and Prediction Lab to DataRootService.         │
│ • Standardize model artifact paths under D:\data\models\.                                        │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Phase 4: Settings UI & Migration Assistant                                                       │
│ • Modernize ML Research Studio Settings tab to display single Data Root picker + health card.    │
│ • Implement automated, safe 5-stage Data Migration Assistant with checksum verification.        │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Phase 5: Deprecation & Final Legacy Path Cleanup                                                 │
│ • Remove dead path resolution helpers (bundled_chart_dir, CHART_DATA_ROOT).                      │
│ • Clean up accidental phantom files (e.g. angelone/analysis.db).                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. Focused Verification Plan

When implementation begins in subsequent feature phases, validation must proceed through focused, non-destructive smoke checks:

1. **Path Resolution Smoke Test**: Verify `DataRootService("D:\\data")` deterministically returns `D:\data\databases\analysis.db` and `D:\data\registries\pipeline_registry_store.json`.
2. **Database Singleton Check**: Verify opening `analysis.db` from any UI panel (Model Registry, Discovery Dashboard, Dossier, Evidence Studio) opens the identical filesystem handle without spawning phantom databases.
3. **Registry Loading Check**: Verify `pipeline_registry_store.json` loads authoritative `PL_0001` (171 base features) directly from `D:\data\registries\`.
4. **Tick DB Access Check**: Verify `angel_market_*.db` files are detected and readable under `D:\data\ticks\`.
5. **Settings UI Check**: Verify changing Data Root in Settings updates the configuration file and reloads all dependent UI panels cleanly.
