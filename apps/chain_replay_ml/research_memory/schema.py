"""DDL Schema Definitions for Persistent Research Memory (Phase 4D.1).

Defines all tables and indices for <data_dir>/analysis.db:
1. research_campaigns (Campaign lifecycle & resource budgeting)
2. experiment_signatures (Deterministic deduplication registry)
3. campaign_experiments (Campaign execution linker)
4. benchmark_runs (Evaluation event context)
5. model_benchmarks (Aggregated scorecard & robustness summary)
6. benchmark_metrics (Granular normalized metric store)
7. regime_evaluations (Cross-regime stress & degradation store)
8. feature_set_evaluations (Composition & dependency audit)
9. champion_history (Longitudinal promotion log)
"""

from __future__ import annotations

ANALYSIS_DB_TABLES_DDL = """
-- 1. Research Campaigns Table (Session Lifecycle & Resource Budgeting)
CREATE TABLE IF NOT EXISTS research_campaigns (
    campaign_id TEXT PRIMARY KEY,
    campaign_name TEXT NOT NULL,
    context_key TEXT NOT NULL,
    description TEXT,
    ranking_policy_version TEXT NOT NULL DEFAULT 'ROB_POLICY_v1.0',
    status TEXT NOT NULL DEFAULT 'PENDING',
    max_experiments_limit INTEGER NOT NULL DEFAULT 100,
    max_duration_seconds REAL NOT NULL DEFAULT 14400.0,
    memory_limit_mb INTEGER NOT NULL DEFAULT 8192,
    total_planned INTEGER NOT NULL DEFAULT 0,
    completed_count INTEGER NOT NULL DEFAULT 0,
    skipped_duplicate_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    start_time TEXT,
    end_time TEXT,
    elapsed_sec REAL,
    checkpoints_json TEXT,
    termination_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- 2. Experiment Signatures Table (Deterministic Deduplication Registry)
CREATE TABLE IF NOT EXISTS experiment_signatures (
    signature_hash TEXT PRIMARY KEY,
    context_key TEXT NOT NULL,
    market TEXT NOT NULL,
    sampling_interval_sec INTEGER NOT NULL,
    task_type TEXT NOT NULL,
    prediction_horizon TEXT NOT NULL,
    regime_id TEXT NOT NULL,
    regime_definition_hash TEXT NOT NULL,
    dataset_snapshot_hash TEXT NOT NULL,
    feature_set_hash TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    hyperparameters_hash TEXT NOT NULL,
    walk_forward_hash TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    canonical_payload_json TEXT NOT NULL,
    first_executed_at TEXT NOT NULL,
    execution_count INTEGER NOT NULL DEFAULT 1,
    latest_model_name TEXT NOT NULL
);

-- 3. Campaign Experiments Table (Session Execution Linker)
CREATE TABLE IF NOT EXISTS campaign_experiments (
    campaign_exp_id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id TEXT NOT NULL,
    trial_index INTEGER NOT NULL,
    signature_hash TEXT NOT NULL,
    model_name TEXT,
    execution_status TEXT NOT NULL,
    elapsed_sec REAL,
    memory_peak_mb REAL,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES research_campaigns(campaign_id),
    FOREIGN KEY (signature_hash) REFERENCES experiment_signatures(signature_hash)
);

-- 4. Benchmark Runs Table (Evaluation Event Context)
CREATE TABLE IF NOT EXISTS benchmark_runs (
    benchmark_run_id TEXT PRIMARY KEY,
    campaign_id TEXT,
    context_key TEXT NOT NULL,
    benchmark_scope TEXT NOT NULL,
    ranking_policy_version TEXT NOT NULL,
    run_timestamp TEXT NOT NULL,
    models_evaluated_count INTEGER NOT NULL,
    top_model_name TEXT,
    incumbent_champion_name TEXT,
    evaluation_criteria_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES research_campaigns(campaign_id)
);

-- 5. Model Benchmarks Table (Aggregated Scorecard & Robustness Summary)
CREATE TABLE IF NOT EXISTS model_benchmarks (
    benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_run_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    signature_hash TEXT NOT NULL,
    context_key TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    feature_count INTEGER NOT NULL,
    ranking_policy_version TEXT NOT NULL,
    primary_metric_name TEXT NOT NULL,
    primary_metric_value REAL NOT NULL,
    secondary_metric_value REAL,
    wf_folds_count INTEGER NOT NULL,
    fold_metric_mean REAL NOT NULL,
    fold_metric_std REAL NOT NULL,
    fold_metric_min REAL NOT NULL,
    fold_metric_max REAL NOT NULL,
    worst_fold_drawdown REAL,
    temporal_stability_score REAL NOT NULL,
    brier_score REAL,
    log_loss REAL,
    expected_calibration_error REAL,
    training_time_sec REAL,
    inference_latency_us REAL,
    model_size_bytes INTEGER,
    robustness_score REAL NOT NULL,
    rank_in_context INTEGER NOT NULL,
    recommendation_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (benchmark_run_id) REFERENCES benchmark_runs(benchmark_run_id),
    FOREIGN KEY (signature_hash) REFERENCES experiment_signatures(signature_hash)
);

-- 6. Benchmark Metrics Table (Granular Normalized Metric Store)
CREATE TABLE IF NOT EXISTS benchmark_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_id INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_stage TEXT NOT NULL,
    fold_index INTEGER,
    metric_value REAL NOT NULL,
    metric_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (benchmark_id) REFERENCES model_benchmarks(benchmark_id)
);

-- 7. Regime Evaluations Table (Cross-Regime Stress & Degradation Store)
CREATE TABLE IF NOT EXISTS regime_evaluations (
    eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    signature_hash TEXT NOT NULL,
    tested_regime_id TEXT NOT NULL,
    tested_regime_hash TEXT NOT NULL,
    is_native_regime INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    primary_metric REAL NOT NULL,
    regime_degradation_pct REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (signature_hash) REFERENCES experiment_signatures(signature_hash)
);

-- 8. Feature Set Evaluations Table (Composition & Dependency Audit)
CREATE TABLE IF NOT EXISTS feature_set_evaluations (
    feature_eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature_hash TEXT NOT NULL,
    feature_set_hash TEXT NOT NULL,
    total_features INTEGER NOT NULL,
    base_pipeline_count INTEGER NOT NULL,
    registry_feature_count INTEGER NOT NULL,
    experimental_feature_count INTEGER NOT NULL,
    deprecated_feature_count INTEGER NOT NULL,
    experimental_dependency_ratio REAL NOT NULL,
    top_10_features_json TEXT NOT NULL,
    features_list_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (signature_hash) REFERENCES experiment_signatures(signature_hash)
);

-- 9. Champion History Table (Longitudinal Promotion Log)
CREATE TABLE IF NOT EXISTS champion_history (
    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_key TEXT NOT NULL,
    previous_champion_name TEXT,
    new_champion_name TEXT NOT NULL,
    previous_robustness_score REAL,
    new_robustness_score REAL NOT NULL,
    score_delta REAL NOT NULL,
    ranking_policy_version TEXT NOT NULL,
    promoted_by TEXT NOT NULL DEFAULT 'HUMAN_RESEARCHER',
    promotion_reason TEXT NOT NULL,
    transition_timestamp TEXT NOT NULL
);

-- High-Performance Indices
CREATE INDEX IF NOT EXISTS idx_exp_sig_context ON experiment_signatures(context_key);
CREATE INDEX IF NOT EXISTS idx_benchmarks_context_rank ON model_benchmarks(context_key, robustness_score DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_metrics_bm ON benchmark_metrics(benchmark_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_campaign_exp_camp ON campaign_experiments(campaign_id, execution_status);
CREATE INDEX IF NOT EXISTS idx_regime_eval_model ON regime_evaluations(model_name, tested_regime_id);
CREATE INDEX IF NOT EXISTS idx_champ_hist_context ON champion_history(context_key, transition_timestamp DESC);
"""

EXPECTED_TABLES: tuple[str, ...] = (
    "research_campaigns",
    "experiment_signatures",
    "campaign_experiments",
    "benchmark_runs",
    "model_benchmarks",
    "benchmark_metrics",
    "regime_evaluations",
    "feature_set_evaluations",
    "champion_history",
)

EXPECTED_INDICES: tuple[str, ...] = (
    "idx_exp_sig_context",
    "idx_benchmarks_context_rank",
    "idx_benchmark_metrics_bm",
    "idx_campaign_exp_camp",
    "idx_regime_eval_model",
    "idx_champ_hist_context",
)
