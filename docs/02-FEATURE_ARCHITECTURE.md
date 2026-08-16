# AruMLStudio Feature Architecture — Feature Types, Registry, Pipeline Features & Metadata

---

## 1. Executive Summary & Purpose

This document is the authoritative technical reference for the **Feature Architecture** of **AruMLStudio**. 

AruMLStudio operates on an explicit, non-overlapping data and feature lifecycle:

```
                                      ARUMLSTUDIO DATA LIFECYCLE
                                      
  Raw Tick Data ──► Master Dataset ──► Feature Transformation ──► Analysis Dataset ──► Create Model ──► Model Registry ──► Production Validation
   (Market Feeds)    (SQLite .db)      (Pipeline Engine)          (Parquet Matrix)     (3-Source Pick)   (Trained Package)   (Unseen Days Replay)
```

### 1.1. Core Architectural Boundaries
1. **Master Dataset (Foundation)**: Built directly from raw tick data. Contains **Feature Registry features only** (scoped by `feature_project_id`). Does **NOT** contain Base Pipeline or Selected Experimental Pipeline features. Routine feature experimentation **never rebuilds the Master Dataset**.
2. **Feature Transformation (Enrichment)**: Takes the Master Dataset as input. Executes manual/auto pipeline candidate generators. Materializes the **Analysis Dataset** containing Registry, Base Pipeline, and Selected Experimental features.
3. **Create Model (Model Builder)**: Consumes the **Analysis Dataset**, not raw tick data. Reads dataset metadata, displays 3-source trees, and trains on the user-selected feature subset.
4. **Model Registry (Trained Package)**: Stores model weights, training config, and preserves complete lineage (`feature_project_id`, `pipeline_id`, `pipeline_snapshot_id`, `selected_features`).
5. **Production Validation (Verification)**: Uses Model Registry lineage to resolve/materialize the exact **Unseen Dataset** required to forward-test on unseen market regimes.

---

## 2. Current Implementation Status vs. Future Roadmap

To ensure total architectural clarity, the subsystem status is defined as follows:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CURRENT / IMPLEMENTED                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Feature Registry: 206 canonical domain features across 11 financial domains.                  │
│ • Feature Projects (feature_project_id): Project-based scoping and custom UI grouping.         │
│ • Master Dataset Materialization: Materializes Feature Registry features into Master SQLite.    │
│ • Experimental Transformations: Manual & Auto Candidate Generation engine.                      │
│ • Pipeline Registry: Storage of experimental pipelines, candidate lists, and transform configs. │
│ • Analysis Dataset Export: Parquet dataset creation (Registry + Experimental Candidates).       │
│ • Model Builder Feature Selection: Selection of features across Registry & Experimental pools.  │
│ • Cryptographic Provenance: pipeline_snapshot_id for exact replay and unseen validation.        │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FUTURE / NOT YET IMPLEMENTED                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • Promotion Engine: Automated workflow to promote validated experimental features to Base.      │
│ • Base Pipeline Library: Dedicated, stabilized core pipeline feature catalog.                  │
│ • Standalone Base Pipeline Generator: Dedicated engine for generating fixed Base transforms.    │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Feature Populations & Lifecycle Classifications

```
                               ┌───────────────────────────────────────────────┐
                               │           ALL AVAILABLE FEATURES              │
                               │                (N Features)                   │
                               └───────────────────────┬───────────────────────┘
                                                       │
                     ┌─────────────────────────────────┼─────────────────────────────────┐
                     ▼                                 ▼                                 ▼
           ┌───────────────────┐             ┌───────────────────┐             ┌───────────────────┐
           │ Feature Registry  │             │Selected Experim.  │             │   Base Pipeline   │
           │ (Population 1)    │             │   (Population 2)  │             │ (Future State / 3)│
           │   [IMPLEMENTED]   │             │   [IMPLEMENTED]   │             │ [NOT YET IMPL.]   │
           └─────────┬─────────┘             └─────────┬─────────┘             └─────────┬─────────┘
                     │                                 │                                 │
           • 206 Canonical Features          • Speculative Transforms          • Future Promotion Target
           • 11 Financial Domains            • Manual & Auto Candidates        • Accepted Candidates
           • Materialized in Master DB       • Generated in Analysis DB        • Stable Core Library
           • Scoped by feature_project_id    • Scoped by pipeline_id           • (Promotion Engine Pending)
```

---

### 3.1. Population 1: Feature Registry Features (IMPLEMENTED)

| Attribute | Technical Specification |
|---|---|
| **What is it?** | Canonical, domain-specific financial features representing market state, option pricing, Greeks, volatility surfaces, order flow, and microstructure. |
| **Why does it exist?** | Provides a standardized, curated vocabulary of primary financial signals computed directly from normalized market tick feeds. |
| **Problem solved** | Eliminates redundant calculation of core financial variables; standardizes Greek models, implied volatilities, and strike moneyness across models. |
| **Source of Truth** | Canonical Registry (`schema_feature_meta.py`, `feature_domains.py`, `feature_registry_store.py`). |
| **Materialization** | **Materialized in Master Dataset SQLite (`master_dataset_*.db`)**. |
| **Scoping Mechanism** | Scoped by **`feature_project_id`** (e.g. `"all"`, `"chart"`). |
| **Model Selection** | Selected in Create Model **Tab 1: Feature Registry**. |
| **Missing Handling** | Fatal build error if a mandatory canonical extractor fails during Master Dataset materialization. |
| **Real Examples** | `spot`, `futures_ltp`, `atm_iv_ce`, `atm_iv_pe`, `call_delta`, `put_gamma`, `atm_straddle_price`, `pcr_oi`, `futures_basis`, `weighted_spot_ema`. |

---

### 3.2. Population 2: Selected Experimental Pipeline Features (IMPLEMENTED)

| Attribute | Technical Specification |
|---|---|
| **What is it?** | Novel, exploratory transformation features created via interactive Manual configuration or combinatorial Auto Candidate Generation. |
| **Why does it exist?** | Machine learning research requires hypothesis testing across temporal lags, returns, rolling statistics, quantile buckets, and cross-feature interactions. |
| **Problem solved** | Allows safe exploration of hundreds of speculative features without modifying the canonical Feature Registry. |
| **Source of Truth** | Experimental Pipeline records (`PL_0002+`) in `pipeline_registry_store.json`. |
| **Materialization** | **Generated dynamically during Analysis Dataset creation** based on the pipeline's `transformation_config`. |
| **Scoping Mechanism** | Scoped by **`pipeline_id`** (e.g. `PL_0005`), `pipeline_name`, and immutable **`pipeline_snapshot_id`**. |
| **Model Selection** | Selected in Create Model **Tab 3: Selected Experimental Pipeline**. |
| **Missing Handling** | If an experimental pipeline feature cannot be computed or is missing in unseen validation data, Production Validation flags a lineage mismatch. |
| **Real Examples** | `spot_roll_mean_60`, `atm_iv_ce_zscore_120`, `spot_bucket_5`, `delta_spread_abs`, `pcr_oi_ema_300`. |

---

### 3.3. Population 3: Base Pipeline Features (FUTURE / NOT YET IMPLEMENTED)

| Attribute | Conceptual Specification |
|---|---|
| **What is it?** | A planned future catalog of accepted, stabilized transformation features that have passed rigorous validation and are promoted into a permanent core library. |
| **Current Status** | **NOT IMPLEMENTED YET**. Currently exists as an architectural placeholder and UI tab concept. |
| **Intended Role** | When implemented, high-performing experimental features will be promoted out of exploratory pipelines into the Base Pipeline library so that all models can reuse them as standard features. |
| **What is NOT Present Today** | There is currently **no active Base Pipeline Generator**, no automated promotion engine, and no separate Base feature materializer. |

---

## 4. Feature Registry Architecture

### 4.1. Canonical Feature Specification
A feature is **Canonical** if and only if:
1. It is registered in the central schema (`_REGISTRY_FEATURES` in `feature_plugins.py` or `schema_feature_meta.py`).
2. It belongs to exactly one **Primary Domain** in `feature_domains.py`.
3. It has a deterministic calculation extractor operating on raw market data (tick-level spot, option chains, order book).
4. It possesses an immutable canonical identifier (e.g. `FR0001`) and standard snake_case name (e.g. `atm_iv_ce`).

### 4.2. Primary Domain Taxonomy
The business domain taxonomy in [`apps/chain_replay_ml/dataset_builder/feature_domains.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/feature_domains.py) defines 11 mutually exclusive financial domains:

```
                                  FEATURE REGISTRY DOMAINS
                                     (206 Active Features)
                                              │
         ┌───────────────────┬────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                ▼                   ▼                   ▼
  [Price & Premium]   [Spot & Futures]    [Greeks]      [Implied Volatility]   [Open Interest]
   (24 features)       (28 features)    (32 features)      (22 features)        (18 features)
         │                   │                │                   │                   │
         ├───────────────────┼────────────────┴───────────────────┼───────────────────┤
         ▼                   ▼                                    ▼                   ▼
[Volume & Liquidity] [Chain Analytics]                    [Market Structure]   [Time & Session]
   (16 features)       (26 features)                        (14 features)       (12 features)
```

| Domain ID | Display Label | Description & Scope | Representative Features |
|---|---|---|---|
| `price_premium` | Price & Premium | Option LTPs, bid/ask spreads, intrinsic values, time value decay | `ltp`, `bid`, `ask`, `spread`, `strike_moneyness`, `time_value` |
| `spot_futures` | Spot & Futures | Underlying spot index, futures prices, basis, EMAs, VWAP | `spot`, `futures_ltp`, `futures_basis`, `spot_ema9`, `spot_ema20` |
| `greeks` | Greeks | Black-Scholes sensitivities (Delta, Gamma, Theta, Vega, Rho, Vanna, Volga) | `call_delta`, `put_delta`, `call_gamma`, `call_vega`, `call_theta` |
| `implied_volatility` | Implied Volatility | ATM IV, skew, smile curves, rolling IV spreads, term structure | `atm_iv_ce`, `atm_iv_pe`, `current_iv`, `roll_iv`, `iv_skew` |
| `open_interest` | Open Interest | Strike open interest, OI deltas, cumulative build-up, PCR | `oi`, `oi_change`, `pcr_oi`, `oi_buildup_ce`, `oi_buildup_pe` |
| `volume_liquidity` | Volume & Liquidity | Traded volume, volume deltas, trade frequency, turnover | `day_volume`, `volume_delta`, `trade_count`, `turnover_rate` |
| `chain_analytics` | Chain Analytics | Gamma Exposure (GEX), strike pinning, max pain, DGT REIV | `total_gex`, `max_pain`, `dgt_reiv_ce`, `straddle_iv_weighted` |
| `historical_context` | Historical Context | Multi-day reference points, previous day close, session highs/lows | `prev_day_close`, `session_high`, `session_low`, `day_range_pos` |
| `market_structure` | Market Structure | Microstructure order flow, bid/ask depth ratios, tick velocity | `order_flow_imbalance`, `book_pressure`, `spread_cost_bps` |
| `time_session` | Time & Session | Minutes to expiry, session progression, day of week indicators | `time_to_expiry_min`, `session_progress_pct`, `is_expiry_day` |
| `metadata` | Metadata | Partitioning columns and indexing tokens | `timestamp`, `trading_day`, `token`, `strike`, `expiry` |

---

### 4.3. Feature Projects (`feature_project_id`)

A **Feature Project** organizes and scopes Feature Registry features for dataset building and model development.

```
                              FEATURE REGISTRY STORE
                           (feature_registry_store.json)
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     Project: "all" (Reserved)                       Project: "chart" (Custom)
     • All 206 active registry features              • Selected subset (e.g. 45 features)
     • 11 Canonical Domain Groups                    • Custom Project Groups ("Chart Core", "EMA")
```

#### Core Concepts:
- **`feature_project_id`**: A unique string identifier (e.g. `"all"`, `"chart"`, `"intraday_momentum"`).
- **Membership**: A project document contains `feature_names: list[str]`, defining the exact subset of canonical features available in that project.
- **Custom Organization**: Custom projects can define `project_groups: list[dict]` and `feature_group_map: dict[str, str]` to organize features into intuitive sub-groups without altering canonical domain assignments.
- **The `"all"` Project**: A reserved, immutable project containing 100% of all active canonical registry features mapped to their 11 canonical domains.
- **CRITICAL DISTINCTION**:
  - `feature_id` / `name` $\implies$ Identifier of a single feature column (e.g. `FR0012` / `atm_iv_ce`).
  - `feature_project_id` $\implies$ Identifier of the project catalog scoping which features are enabled (e.g. `"chart"`).

---

## 5. Master Dataset & Feature Registry Boundary

The **Master Dataset** is the single source of truth for materialized Feature Registry market data.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               MASTER DATASET BUILDER                                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. User selects Feature Project (e.g. feature_project_id = "chart").                   │
│ 2. Feature Selection Tree displays features:                                           │
│    Active Features = Selected Project Features ∩ Canonical Registry Features           │
│ 3. Master Dataset materializes only these Registry features into SQLite tables.        │
│ 4. Master Dataset metadata records:                                                    │
│    • feature_project_id: "chart"                                                       │
│    • registry_export_features: ["spot", "atm_iv_ce", "futures_ltp", ...]               │
│    • registry_export_count: 45                                                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### What the Master Dataset DOES NOT Contain:
- **NO Pipeline Features**: All transformed features (whether experimental or speculative) are strictly excluded from Master Dataset SQLite tables. Master Dataset owns raw Feature Registry columns only.

---

## 6. Experimental Pipeline Transformations (IMPLEMENTED)

Experimental pipelines allow data scientists to engineer, evaluate, and test speculative features without contaminating the canonical Feature Registry.

```
       ┌────────────────────────────────────────────────────────┐
       │                Transformation Generator                │
       │    (Rolling Z-Score, Quantile Bucket, Math Log)        │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │                   Candidate Feature                    │
       │                 (e.g. spot_roll_mean_60)               │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │              Pipeline Record (PL_0002+)                │
       │          (Stored in pipeline_registry_store)           │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │            Immutable Pipeline Snapshot                 │
       │        (SHA256 Hash = pipeline_snapshot_id)            │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │              Analysis Dataset Export                   │
       │            (Materialized into Parquet)                 │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │         Create Model Selection (Section 5.3)           │
       │             (Selected Model Features)                  │
       └────────────────────────────────────────────────────────┘
```

### Why Experimental Features are Isolated:
1. **Registry Contamination Prevention**: Prevents experimental hypothesis-testing from cluttering the stable financial taxonomy.
2. **Lineage & Replay Fidelity**: A cryptographic `pipeline_snapshot_id` ensures that a model trained on version A of a pipeline can be reproduced exactly, even if version B is later saved under the same pipeline name.
3. **Unseen Validation Safety**: Production Validation knows exactly which pipeline generated each feature for out-of-sample live replay.

---

## 7. Comprehensive Feature Metadata Inventory (Levels 1 to 7)

AruMLStudio tracks metadata across seven hierarchical layers:

### Level 1 — Individual Feature Schema Metadata
Stored in [`apps/chain_replay_ml/dataset_builder/schema_feature_meta.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/schema_feature_meta.py) and `feature_domains.py`:

| Field Name | Type | Description | Example |
|---|---|---|---|
| `name` | `str` | Canonical snake_case feature identifier | `"atm_iv_ce"` |
| `primary_domain` | `str` | Financial domain taxonomy ID | `"implied_volatility"` |
| `display_name` | `str` | Human-readable label for UI | `"ATM Implied Volatility (CE)"` |
| `description` | `str` | Technical and economic explanation | `"Black-Scholes implied volatility for the CE strike closest to spot."` |
| `data_type` | `str` | Value classification | `"Volatility"`, `"Price"`, `"Ratio"`, `"Count"` |
| `ownership` | `str` | Extraction tier | `"base"`, `"computed_base"` |
| `depends_on` | `list[str]` | Input dependencies | `["ltp", "spot", "strike", "time_to_expiry"]` |
| `can_apply_lag` | `bool` | Flag enabling time-lag transforms | `True` |
| `can_apply_return`| `bool` | Flag enabling return transforms | `True` |
| `can_participate_in_interaction` | `bool` | Flag enabling cross-feature math | `True` |
| `status` | `str` | Implementation status | `"production"`, `"experimental"`, `"deprecated"` |

---

### Level 2 — Feature Registry Metadata
Stored in `<chart_data_dir>/feature_registry_store.json`:

| Field Name | Type | Description | Example |
|---|---|---|---|
| `registry_version` | `str` | Store schema version | `"1.0"` |
| `created_on` | `str` | Creation timestamp | `"2026-08-16"` |
| `feature_ids` | `dict[str, str]` | Mapping of canonical ID to name | `{"FR0001": "spot", "FR0012": "atm_iv_ce"}` |
| `disabled_features`| `dict[str, Any]` | Catalog of user-retired features | `{"custom_ratio_1": {"retired_on": "2026-08-10"}}` |
| `next_feature_id_seq` | `int` | Sequential counter for new features | `207` |

---

### Level 3 — Feature Project Metadata
Stored in `<chart_data_dir>/feature_registry_projects.json` (and `feature_registry_store.json`):

| Field Name | Type | Description | Example |
|---|---|---|---|
| `label` | `str` | Project display name | `"Chart Analysis Project"` |
| `description` | `str` | Research purpose | `"Core spot and implied volatility features for chart models."` |
| `feature_names` | `list[str]` | Included canonical registry features | `["spot", "futures_ltp", "atm_iv_ce", "spot_ema20"]` |
| `project_groups` | `list[dict]` | Custom organizational groups | `[{"id": "chart_core", "label": "Chart Core"}]` |
| `feature_group_map`| `dict[str, str]` | Mapping of feature to custom group | `{"spot": "chart_core", "atm_iv_ce": "chart_core"}` |
| `reserved` | `bool` | True for system `"all"` project | `False` |

---

### Level 4 — Pipeline Feature Metadata (Candidate Level)
Stored in transformation definitions:

| Field Name | Type | Description | Example |
|---|---|---|---|
| `feature_name` | `str` | Generated candidate column name | `"spot_roll_mean_60"` |
| `source_feature` | `str` | Input parent feature | `"spot"` |
| `transformation` | `str` | Generator family | `"rolling"` |
| `operation` | `str` | Mathematical operation | `"mean"` |
| `window_seconds` | `int` | Time horizon | `60` |
| `partition_by` | `list[str]` | Grouping keys | `["trading_day", "token"]` |

---

### Level 5 — Pipeline Store Metadata
Stored in `<chart_data_dir>/pipeline_registry_store.json`:

| Field Name | Type | Description | Example |
|---|---|---|---|
| `pipeline_id` | `str` | Unique pipeline ID (`PL_XXXX`) | `"PL_0005"` |
| `name` | `str` | Pipeline display name | `"Pipeline_005 (Auto Momentum)"` |
| `type` | `str` | Pipeline origin type | `"manual"`, `"auto"` |
| `status` | `str` | Lifecycle state | `"draft"`, `"ready"` |
| `candidate_features`| `list[str]` | List of generated feature names | `["spot_roll_mean_60", "atm_iv_ce_zscore_120"]` |
| `transformation_config`| `dict` | Full executable transform graph | `{"transformation_pipeline_version": 1, ...}` |
| `created_at` | `str` | UTC creation timestamp | `"2026-08-16T12:00:00Z"` |

---

### Level 6 — Dataset Feature Metadata
Stored in `datasets/analysis_datasets/analysis_<name>.json`:

| Field Name | Type | Description | Example |
|---|---|---|---|
| `feature_project_id` | `str` | Scoping project ID | `"chart"` |
| `registry_export_features` | `list[str]` | Materialized registry features | `["spot", "atm_iv_ce", "futures_ltp"]` |
| `registry_export_count` | `int` | Count of registry features | `3` |
| `pipeline_id` | `str` | Bound experimental pipeline ID | `"PL_0005"` |
| `pipeline_name` | `str` | Bound pipeline display label | `"Pipeline_005 (Auto Momentum)"` |
| `pipeline_snapshot_id` | `str` | Cryptographic content hash | `"ca5945f58f8b1a2c"` |
| `pipeline_provenance` | `dict` | Snapshot object recording candidates | `{"pipeline_id": "PL_0005", "candidate_features": [...]}` |
| `include_registry` | `bool` | Flag indicating registry inclusion | `True` |
| `include_pipeline` | `bool` | Flag indicating pipeline inclusion | `True` |

---

### Level 7 — Model Feature Metadata
Stored in `models/<model_name>/config.json`:

| Field Name | Type | Description | Example |
|---|---|---|---|
| `features` | `list[str]` | Final selected model features | `["spot", "atm_iv_ce", "spot_roll_mean_60"]` |
| `feature_count` | `int` | Total number of selected features | `3` |
| `feature_project_id` | `str` | Inherited project ID | `"chart"` |
| `pipeline_id` | `str` | Inherited experimental pipeline ID | `"PL_0005"` |
| `pipeline_snapshot_id` | `str` | Inherited snapshot hash | `"ca5945f58f8b1a2c"` |
| `pipeline_provenance` | `dict` | Inherited pipeline provenance | `{"pipeline_id": "PL_0005", "candidate_features": [...]}` |

---

## 8. Feature Source Classification Engine

The classification engine in [`apps/chain_replay_ml/dataset_builder/feature_sources_catalog.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/feature_sources_catalog.py) classifies any feature deterministically using strict precedence rules:

```
                            Feature Name: feat
                                   │
                                   ▼
         Is feat in registry_export_features OR canonical domain?
                                   │
                         YES ──────┴────── NO
                          │                 │
                          ▼                 ▼
          DATASET_SOURCE_FEATURE_REGISTRY   Is feat in experimental candidate_features OR
                                            matches pipeline transform config?
                                                    │
                                          YES ──────┴────── NO
                                           │                 │
                                           ▼                 ▼
                           DATASET_SOURCE_OTHER_PIPELINE    UNCLASSIFIED (Error)
```

---

## 9. Feature Lifecycle

```
Feature Stage        Feature Registry (IMPLEMENTED)     Selected Experimental (IMPLEMENTED)  Base Pipeline (FUTURE)
----------------------------------------------------------------------------------------------------------------------
1. Definition        schema_feature_meta.py             Auto / Manual Transform Builder      Promotion from validated candidate
2. Scoping           feature_project_id                 pipeline_registry_store record       Core library inclusion
3. Materialization   Master Dataset (SQLite)            Analysis Dataset (Parquet)           (Future core materialization)
4. Model Selection   Model Builder Tab 1                Model Builder Tab 3                  (Future Model Builder Tab 2)
5. Model Training    XGBoost Feature Matrix             XGBoost Feature Matrix               XGBoost Feature Matrix
6. Profiling         Feature Studio Diagnostics         Feature Studio Diagnostics           Feature Studio Diagnostics
7. Validation        Unseen Dataset Hash Match          Unseen Dataset Hash Match            Unseen Dataset Hash Match
```

---

## 10. Metadata Propagation & Lineage Replay

```
Feature Registry ──► Feature Project ──► Master Dataset ──► Analysis Dataset ──► Model ──► Validation
──────────────────────────────────────────────────────────────────────────────────────────────────────
Preserved:           Canonical Names     Project ID          Project ID          Project ID   Project ID
                     Domain IDs          Project Groups      Export List         Export List  Export List
                                                             Pipeline ID         Pipeline ID  Pipeline ID
                                                             Snapshot ID         Snapshot ID  Snapshot ID
──────────────────────────────────────────────────────────────────────────────────────────────────────
Added:               feature_project_id  registry_export_*   pipeline_provenance features     unseen_hash
                                         SQLite DB Path      timing metrics      timings      drift_scores
──────────────────────────────────────────────────────────────────────────────────────────────────────
Immutable:           FR IDs & Names      Project Identity    Master Schema       Snapshot ID  Validation Hash
```

### Unseen Lineage Resolution:
- Production Validation computes:
  $$\text{unseen\_dataset\_identity\_hash} = \text{SHA256}\Big(\text{master\_db} + \text{unseen\_days} + \text{feature\_project\_id} + \text{pipeline\_id} + \text{pipeline\_snapshot\_id}\Big)[:8]$$
- Ensures 100% reproducible out-of-sample data generation without manual re-configuration.

---

## 11. Architectural Invariants (Enforced Rules)

1. **Master Dataset Exclusivity**: Master Dataset contains only materialized Feature Registry features.
2. **Analysis Dataset Coexistence**: Analysis Dataset is where Feature Registry and Selected Experimental Pipeline features coexist.
3. **`feature_project_id` Boundary**: `feature_project_id` scopes Feature Registry membership; it is not a feature ID.
4. **Snapshot Immutability**: `pipeline_snapshot_id` represents an immutable cryptographic hash of an experimental pipeline definition.
5. **Deterministic Classification**: Every model feature must belong to exactly one feature-source population.
6. **Non-Destructive Recommendations**: Feature recommendations (`KEEP`, `WATCH`, `REMOVE`) write to an audit trail and never mutate or delete canonical definitions.
7. **Base Pipeline Separation**: Base Pipeline is a future lifecycle promotion state, not an active feature generator today.

---

## 12. Authoritative Comparison Table

| Property | Feature Registry (IMPLEMENTED) | Selected Experimental (IMPLEMENTED) | Base Pipeline (FUTURE / NOT YET IMPL.) |
|---|---|---|---|
| **Purpose** | Core market state & Greek representation | Exploratory hypothesis testing & research | Accepted, stabilized core library |
| **Status** | **CURRENT / IMPLEMENTED** | **CURRENT / IMPLEMENTED** | **FUTURE LIFECYCLE TARGET** |
| **Source of Truth** | Canonical Registry (`schema_feature_meta.py`) | `pipeline_registry_store.json` (`PL_0002+`) | (Future Promoted Catalog) |
| **Created By** | System Engine / Quant Extractor | Manual Builder / Auto Combinatorial Batch | Promotion from validated experimental |
| **Scoping Key** | `feature_project_id` | `pipeline_id` + `pipeline_snapshot_id` | Core platform inclusion |
| **Snapshot Hash** | None (Static Schema) | Cryptographic SHA256 (`pipeline_snapshot_id`) | Immutable Library Version |
| **Stored in Master DB** | **YES (SQLite Table)** | **NO** | **NO** |
| **Generated in Analysis**| Read from Master SQLite | **YES (Generated in Parquet)** | (Future standard inclusion) |
| **Model Selection** | Model Builder Tab 1 | Model Builder Tab 3 | (Future Model Builder Tab 2) |
| **Representative Example**| `atm_iv_ce` | `spot_roll_mean_60` | (Promoted candidate feature) |
