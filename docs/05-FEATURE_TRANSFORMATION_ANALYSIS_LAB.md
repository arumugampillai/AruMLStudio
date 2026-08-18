# AruMLStudio Feature Transformation Analysis Lab — Layer-2 Scientific Feature Selection

---

## 1. Executive Summary & Purpose

The **Feature Analysis Lab** (the "Analysis" tab in Feature Transformations, implemented in [`apps/master_dataset_tk/feature_analysis_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_analysis_panel.py)) is the dedicated pre-training statistical research, collinearity diagnostics, and scientific feature down-selection workbench in **AruMLStudio**.

### 1.1. The Two-Layer Architecture

AruMLStudio enforces a strict architectural decoupling between **Historical Lifecycle Governance** (Layer 1) and **Scientific Model Feature Selection** (Layer 2):

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: HISTORICAL ELIMINATION GATE (Pre-Materialization / Generation Phase)               │
│ • Uses: Production Validation evidence from feature_recommendation_evidence.db              │
│ • Scope: Dataset context-wide (context_id) historical lifecycle state                       │
│ • Function: Prevents historically 'blocked' Experimental features from being regenerated    │
│ • Invariant: Historical lifecycle decision; does NOT evaluate current dataset collinearity  │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: SCIENTIFIC FEATURE SELECTION (Post-Materialization / Analysis Phase)               │
│ • Uses: In-sample statistical metrics over an already materialized Analysis Dataset         │
│ • Scope: Model/dataset-scoped selection decision (.parquet)                                 │
│ • Function: Removes collinear redundancies (HCA/Corr) and selects optimal feature subsets    │
│ • Invariant: Model selection decision; does NOT mutate Evidence DB or emit KEEP/REMOVE      │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Independence Invariant**: Layer 1 and Layer 2 are completely independent and decoupled:
> - Layer 1 decides: *"Is this experimental candidate historically viable in this dataset regime?"*
> - Layer 2 decides: *"Which subset of available candidate features provides optimal, non-redundant signal for this specific model architecture?"*

---

## 2. Distinction: Analysis Exclusion vs. Production Validation REMOVE vs. Historical BLOCKED

The system maintains precise, non-overlapping semantic definitions across the lifecycle:

$$\begin{aligned}
\mathbf{Analysis\ EXCLUDED} &\ne \mathbf{Production\ Validation\ REMOVE} \ne \mathbf{Historical\ BLOCKED}
\end{aligned}$$

| State | Originating Subsystem | Meaning | Persistent Impact |
|---|---|---|---|
| **Analysis EXCLUDED** | Layer 2 — Feature Analysis Lab | Feature was omitted from the final model selection bundle (e.g. dropped due to pairwise correlation $\ge 0.95$ or superseded by a higher-scoring HCA family representative). | **None on Evidence DB**. Feature remains available in the Analysis Dataset and can be included in other models. Does **not** write to SQLite or emit recommendations. |
| **PV REMOVE** | Production Validation | Feature exhibited severe rank collapse ($\Delta R \le -5$), large relative importance drop ($\ge 50\%$), and distribution drift ($\ge 1$) when forward-tested on **true unseen trading days**. | **Persisted to `recommendation_evidence`**. Contributes to accumulated negative evidence scores and streak counts. |
| **Historical BLOCKED** | Layer 1 — Pre-Training Elimination Gate | An Experimental feature accumulated $\ge 2$ consecutive REMOVEs or $\ge 4$ total REMOVEs in `feature_context_summary`. | **Active Gate**. Automatically rejected during Auto Candidate Generation before expensive parquet materialization. |

---

## 3. Complete End-to-End Layer-2 Data Flow

```
                      Materialized Analysis Dataset (.parquet)
                                         │
                                         ▼
                     Multi-Module Statistical Analysis Engine
  ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
  │   Correlation    │       HCA        │   Permutation    │   Mutual Info    │
  │ (Pearson/Spearman│ (Agglomerative   │ (Baseline Tree   │ (Non-Linear k-NN │
  │  Pairwise |r|)   │  Feature Families│  Degradation Δ)  │  Entropy I(X;Y)) │
  └────────┬─────────┴────────┬─────────┴────────┬─────────┴────────┬─────────┘
           │                  │                  │                  │
           └──────────────────┼──────────────────┴──────────────────┘
                              │
                              ▼
                  Feature Selection Strategy
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
 [hca_corr_perm]         [corr_perm]           [perm_only]
  (HCA + Corr + Perm)     (Corr + Perm)         (Permutation)
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                Final Feature Selection Bundle
             (Selected Feature Names + Lineage)
                              │
                              ▼
                     Create Model Builder
                  (Feature Set Preset Saved)
                              │
                              ▼
                        Model Builder
            (Tab 1: Registry · Tab 2: Base · Tab 3: Exp)
                              │
                              ▼
                        Model Training
                    (XGBoost / LightGBM / RF)
                              │
                              ▼
                    Trained Model Package
                    (models/<model_name>/)
                              │
                              ▼
                    Production Validation
                (True Unseen Forward Testing)
                              │
                              ▼
                     KEEP / WATCH / REMOVE
                              │
                              ▼
                 Recommendation Evidence DB
             (feature_recommendation_evidence.db)
```

---

## 4. Module Results vs. Final Selection Participation

The Analysis Lab contains 8 distinct research tabs. It is critical to distinguish between **Diagnostic/Research Modules** and **Modules that actively participate in the Feature Selection Engine**:

```
                       ANALYSIS LAB MODULE TAXONOMY
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
DIAGNOSTIC / RESEARCH MODULES                     SELECTION ENGINE PARTICIPANTS
(Informational & Deep-Dive Analysis)              (Directly Execute Selection Logic)
           │                                                   │
• Mutual Information (analysis_mutual_information.py)  • Correlation Matrix (analysis_correlation.py)
• Fast TreeSHAP (analysis_shap.py)                     • HCA Feature Families (analysis_hca.py)
• Variance Inflation Factors (VIF)                    • Permutation Scoring (analysis_permutation.py)
• Univariate Feature Profiles (analysis_feature_profiles.py) • Discovery Rating / Review (analysis_family_review.py)
```

### 4.1. Selection Engine Participants
1. **Correlation Matrix (`analysis_correlation.py`)**: Computes all pairwise Pearson/Spearman coefficients. Directly executes greedy high-correlation drops.
2. **HCA Clustering (`analysis_hca.py`)**: Partitions correlated features into agglomerative feature families based on distance $d = 1 - |r|$.
3. **Permutation Importance (`analysis_permutation.py`)**: Computes empirical predictive utility via baseline metric degradation on permuted rows.
4. **Family Review & Discovery Rating (`analysis_family_review.py`)**: Ranks family members by discovery score, computes score gaps, and evaluates manual researcher overrides.

### 4.2. Diagnostic / Research Modules
- **Mutual Information (`analysis_mutual_information.py`)**: Estimates non-linear entropy $I(X; Y)$. Used for exploratory ranking of complex interactions (e.g. volatility smiles) that linear correlation misses.
- **TreeSHAP (`analysis_shap.py`)**: Runs TreeSHAP attribution on pre-trained models.
- **Variance Inflation Factor (VIF)**: Computes multicollinearity inflation in linear sub-spaces.
- **Feature Profiles & Roles (`analysis_feature_profiles.py`)**: Univariate statistical summaries (moments, missingness, quantiles).

---

## 5. Selection Strategies: Exact Algorithms & Decision Rules

Implemented in [`apps/chain_replay_ml/dataset_builder/analysis_feature_selection.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_feature_selection.py):

```
                        INPUT: Unified Feature Matrix (N Features)
                                           │
         ┌───────────────────┬─────────────┴───────┬───────────────────┐
         ▼                   ▼                     ▼                   ▼
  [hca_corr_perm]       [corr_perm]           [perm_only]         [corr_only]
  HCA + Corr + Perm   Corr + Perm Only      Permutation Only    Correlation Only
```

### 5.1. Strategy 1: `hca_corr_perm` (HCA + Correlation + Permutation) — Recommended Default
- **Input**: All numerical feature columns in the Analysis Dataset.
- **Modules Required**: `correlation`, `hca`, `permutation`, `feature_profiles` / `family_review`.
- **Configurable Parameters**:
  - `correlation_threshold` (default: $0.95$)
  - `permutation_threshold` (default: $0.001$)
  - `representative_policy` (`top_1`, `top_2`, `top_3`, `top_n`)
  - `min_family_size` (default: $2$)
- **Execution Workflow**:
  1. **Correlation Filtering**: Evaluates all pairwise $|r| \ge \text{correlation\_threshold}$. Drops the lower-scoring feature in each collinear pair.
  2. **HCA Clustering**: Cuts the complete-linkage / average-linkage dendrogram at distance threshold $d = 1 - |r| > 0.15$ to form discrete **Feature Families**.
  3. **Representative Policy Selection (`_select_hca_top_n`)**: Selects Top $N$ representatives per family based on explicit precedence:
     $$\text{Manual Override} \longrightarrow \text{Suggested Discovery Leader} \longrightarrow \text{Cluster Centroid}$$
  4. **Permutation Thresholding**: Retains features meeting or exceeding the permutation importance threshold.
- **Output**: Down-selected feature list, family mapping, and `n_families` count.

### 5.2. Strategy 2: `corr_perm` (Correlation + Permutation Only)
- **Input**: All feature columns.
- **Modules Required**: `correlation`, `permutation`.
- **Configurable Parameters**: `correlation_threshold` (default: $0.95$), `permutation_threshold` (default: $0.001$).
- **Execution Workflow**:
  1. **Greedy Correlation Filter (`correlation_filter`)**: Sorts pairs by $|r|$ descending. For each pair where $|r| \ge \text{correlation\_threshold}$, drops the member with the lower score.
  2. **Permutation Filter (`permutation_filter`)**: From the remaining features, keeps only those where $|\text{importance}| \ge \text{permutation\_threshold}$.
- **Output**: Collinearity-pruned, performance-filtered feature subset.

### 5.3. Strategy 3: `perm_only` (Permutation Importance Only — Research)
- **Input**: All feature columns.
- **Modules Required**: `permutation`.
- **Configurable Parameters**: `permutation_threshold` (default: $0.001$).
- **Execution Workflow**:
  1. Ranks all features by $|\text{importance}|$ descending.
  2. Drops features where $|\text{importance}| < \text{permutation\_threshold}$.
- **Output**: Ranked predictive feature subset without collinearity pruning.

### 5.4. Strategy 4: `corr_only` (Correlation Only — Research)
- **Input**: All feature columns.
- **Modules Required**: `correlation`.
- **Configurable Parameters**: `correlation_threshold` (default: $0.95$).
- **Execution Workflow**:
  1. Runs `correlation_filter()` across all pairs where $|r| \ge \text{correlation\_threshold}$.
  2. Drops collinear duplicates using tie-breaker scoring.
- **Output**: Orthogonal feature subset without performance degradation filtering.

---

## 6. HCA Representative Selection: Precedence & Fallback Logic

Located in [`apps/chain_replay_ml/dataset_builder/analysis_hca.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_hca.py) and [`analysis_family_review.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/chain_replay_ml/dataset_builder/analysis_family_review.py):

For every HCA feature family, representative selection follows an **exact, deterministic 4-stage hierarchy**:

```
                         HCA REPRESENTATIVE RESOLUTION
                                       │
                Does a manual researcher override exist in DB?
                          (family_review table)
                                       │
                        YES ───────────┴─────────── NO
                         │                           │
                         ▼                           ▼
                 Use Selected Rep          Does Discovery Rating exist?
                 (Saved by user)             (feature_profiles table)
                                                     │
                                       YES ──────────┴────────── NO
                                        │                         │
                                        ▼                         ▼
                                Use Suggested Leader       Use Cluster Centroid
                               (Highest feature_score)  (Highest mean |r| to members)
```

1. **Stage 1 — Manual Researcher Override**:
   If the researcher explicitly confirmed or overridden a representative in the Family Review tab (`family_review.selected_representative`), that exact feature is locked as the primary representative.
2. **Stage 2 — Suggested Discovery Score Leader**:
   If no manual decision exists, the algorithm selects the family member with the highest overall Discovery Rating score (`feature_profiles.feature_score` or `rating_score`).
3. **Stage 3 — Cluster Centroid Fallback**:
   If no discovery scores or permutation metrics have been computed, the algorithm selects the feature with the highest average absolute correlation $\overline{|r|}$ to all other cluster members (the mathematical centroid):
   $$\text{Centroid}(f) = \arg\max_{f \in \text{Family}} \left( \frac{1}{|\text{Family}| - 1} \sum_{g \in \text{Family}, g \ne f} |r_{f, g}| \right)$$
4. **Stage 4 — Top N Policy Expansion**:
   If the policy is `top_2`, `top_3`, or `top_n`, the primary representative is placed in Slot 1. Subsequent slots are filled from remaining family members in descending order of discovery score until $N$ representatives are selected.

---

## 7. Feature Source Handling: Unified Pool vs. Downstream Lineage

### 7.1. In-Analysis Unified Representation
The Analysis Lab processes all columns as a **single unified, flat feature matrix**:

$$\mathbf{X}_{\text{Analysis}} = \mathbf{X}_{\text{Registry}} \cup \mathbf{X}_{\text{Base Pipeline}} \cup \mathbf{X}_{\text{Experimental Pipeline}}$$

- It does **not** run separate correlation matrices or distinct HCA algorithms for the three feature populations.
- This ensures that collinearity between a Feature Registry feature (e.g. `atm_iv_ce`) and an Experimental Pipeline feature (e.g. `atm_iv_ce_roll_mean_60`) is detected and resolved scientifically.

### 7.2. Lineage Metadata Preservation
While the statistical computation is unified, full lineage metadata is preserved in companion dataset JSON files and propagated downstream:
- `feature_project_id` (e.g. `"all"`, `"chart"`)
- `pipeline_id` & `pipeline_snapshot_id` (e.g. `"PL_0005"`, `"snap_v1"`)
- `registry_export_features` (canonical registry feature list)
- `base_pipeline_export_features` (approved base pipeline feature list)
- `pipeline_provenance` (candidate snapshot and transformation rules)

---

## 8. Selection Provenance & Bundle Schema

When a selection is finalized via `build_final_feature_dataset()`, a serializable lineage dictionary is constructed by `build_feature_selection_lineage()`:

```json
{
  "source": "analysis",
  "strategy": "hca_corr_perm",
  "strategy_label": "HCA + Correlation + Permutation",
  "strategy_short": "HCA",
  "representative_policy": "top_1",
  "representative_policy_label": "Top 1",
  "top_n": 1,
  "correlation_threshold": 0.95,
  "permutation_threshold": 0.001,
  "min_family_size": 2,
  "n_input_features": 583,
  "n_after_correlation": 420,
  "n_families": 48,
  "n_selected_features": 116,
  "feature_set_hash": "a1b2c3d4e5f67890",
  "analysis_dataset": "analysis_PL0005_198r_447p_6s_20260814_221827",
  "discovery_bundle_id": "disc_20260815_120000",
  "run_id": "run_20260815_120000",
  "resolved_at": "2026-08-16T12:00:00+00:00"
}
```

### Fields Stored in Selection Provenance:
- `source`: Originating subsystem (`"analysis"`).
- `strategy`, `strategy_label`, `strategy_short`: The selected strategy ID and human labels.
- `representative_policy`, `top_n`: HCA policy configuration.
- `correlation_threshold`, `permutation_threshold`: Numerical thresholds applied.
- `n_input_features`, `n_after_correlation`, `n_families`, `n_selected_features`: Pipeline stage counts.
- `feature_set_hash`: SHA256 fingerprint of the sorted selected feature names.
- `analysis_dataset`: Name of the parent dataset.
- `discovery_bundle_id`, `run_id`: Unique identifiers linking to SQLite `analysis.db`.

> [!NOTE]
> Individual per-feature exclusion reasons are displayed dynamically in UI cards during preview, but are not serialized as per-column database rows.

---

## 9. Model Builder Handoff Workflow

When the user clicks **"► Create Model Builder"** in [`apps/master_dataset_tk/feature_analysis_panel.py`](file:///c:/Users/admin/PycharmProjects/AruMLStudio/apps/master_dataset_tk/feature_analysis_panel.py):

```
[1] User clicks "► Create Model Builder"
         │
         ▼
[2] build_final_feature_dataset(...) resolves selected features and lineage metadata
         │
         ▼
[3] save_feature_preset(chart_dir, features=..., dataset=..., analysis_feature_selection=lineage)
    Writes preset to build_config_prefs.json
         │
         ▼
[4] Callback _on_open_model_builder(features, dataset, source_model, lineage) invoked
         │
         ▼
[5] Model Builder opens and loads target dataset
         │
         ▼
[6] apply_feature_preset() intersects preset with dataset catalog
         │
         ▼
[7] Model Builder checks active boxes across its 3 source tabs:
    ├── Tab 1: Feature Registry (Matches against registry_export_features)
    ├── Tab 2: Base Pipeline (Matches against base_pipeline_export_features)
    └── Tab 3: Selected Experimental (Matches against pipeline_provenance candidate features)
```

### Feature Source Resolution in Model Builder:
Model Builder does not require source tags in the feature list. It matches incoming feature names against the dataset's embedded metadata:
- Features in `registry_export_features` $\longrightarrow$ Checked in **Tab 1: Feature Registry**.
- Features in `base_pipeline_export_features` $\longrightarrow$ Checked in **Tab 2: Base Pipeline**.
- Remaining features $\longrightarrow$ Checked in **Tab 3: Selected Experimental**.

---

## 10. Complete Lifecycle Boundary Matrix

To maintain architectural clarity, each subsystem answers a distinct, non-overlapping question:

```
┌──────────────────────────────────────┬─────────────────────────────────────────────────────────────────┐
│ Subsystem / Mechanism                │ Core Architectural Question Answered                            │
├──────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
│ **Layer 2 — Feature Analysis Lab**   │ *"Which features provide optimal, non-redundant signal for      │
│                                      │  this specific model architecture on historical data?"*         │
├──────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
│ **Production Validation**            │ *"How did these trained features perform when forward-tested    │
│                                      │  against true unseen market trading days?"*                     │
├──────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
│ **Recommendation Lifecycle**         │ *"What is the accumulated evidence score and streak health     │
│                                      │  of this feature across multiple production validation runs?"*  │
├──────────────────────────────────────┼─────────────────────────────────────────────────────────────────┤
│ **Layer 1 — Pre-Training Gate**      │ *"Should this historically failed Experimental feature be       │
│                                      │  generated again in this dataset context?"*                     │
└──────────────────────────────────────┴─────────────────────────────────────────────────────────────────┘
```

---

## 11. Complete Closed-Loop System Architecture

```
Feature Registry / Pipeline Candidate Generation
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: HISTORICAL ELIMINATION GATE                                   │
│ • Pre-Materialization / Generation Phase (context_id scoped)           │
│ • Checks feature_context_summary in feature_recommendation_evidence.db │
│ • Filters out historically blocked Experimental features               │
└───────────────────┬────────────────────────────────────────────────────┘
                    │
                    ▼
       Allowed Experimental Candidates
                    │
                    ▼
  Master / Analysis Dataset Materialization
                    │
                    ▼
        Analysis Dataset (.parquet)
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: SCIENTIFIC FEATURE SELECTION (Analysis Lab)                   │
│ • Post-Materialization / In-Sample Statistical Analysis                │
│ • Correlation Matrix, HCA Families, Permutation Scoring                │
│ • Resolves Final Feature Selection Bundle via Strategy (hca_corr_perm) │
└───────────────────┬────────────────────────────────────────────────────┘
                    │
                    ▼
         Feature Selection Bundle
                    │
                    ▼
              Model Builder
(Tab 1: Registry · Tab 2: Base · Tab 3: Exp)
                    │
                    ▼
              Model Training
         (Trained Model Package)
                    │
                    ▼
 Feature Studio / Production Validation
     (True Unseen Forward Testing)
                    │
                    ▼
         KEEP / WATCH / REMOVE
                    │
                    ▼
       Recommendation Evidence DB
(recommendation_evidence + Dual Projections)
                    │
                    ▼
          Historical Evidence
                    │
                    ▼
          [Closes Loop to Layer 1 Gate]
```

---

## 12. Current Invariants & Future Architecture Roadmap

### 12.1. Current Invariants (Implemented Codebase)
1. **Unified Flat Matrix**: Analysis Lab treats all columns as a single pool; feature source partitioning occurs downstream in Model Builder.
2. **Read-Only over Datasets**: Analysis Lab does not alter Parquet feature values.
3. **No Automatic Evidence Mutation**: Analysis Lab selection/exclusion never modifies `feature_recommendation_evidence.db`.
4. **Deterministic Lineage**: Every selection bundle records strategy, thresholds, family counts, and feature set hashes.

### 12.2. Future Architecture Roadmap (NOT CURRENTLY IMPLEMENTED)
> [!NOTE]
> The following capabilities represent potential future enhancements and are **not implemented in the current codebase**:
> - **Automatic Analysis &rarr; Recommendation DB Feedback**: Automated creation of recommendation evidence directly from Analysis Lab scores without running Production Validation.
> - **Autonomous Base Pipeline Promotion**: Automatic code generation or promotion of high-performing Analysis Lab features to Base Pipeline without human review.
> - **Source-Partitioned Analysis Tabs**: Separate visual tabs for Feature Registry, Base Pipeline, and Experimental features inside the Correlation and Permutation modules.
> - **Composite Non-Linear Scoring**: Mathematical combination of SHAP, MI, and Permutation scores into a single weighted selection metric.
