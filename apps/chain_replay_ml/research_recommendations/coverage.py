"""Context Coverage & Evidence Density Analyzer (Phase 4E.1).

Constructs the operational coverage matrix across Market, Sampling Interval, Task Type,
Prediction Horizon, and Market Regime using the canonical ModelContextKey.

Calculates evidence density scores bounded in [0.0, 100.0] and categorizes contexts
into deterministic coverage classes (COLD_START, SPARSE, DEVELOPING, MATURE).
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

from chain_replay_ml.model_taxonomy.enums import (
    BASELINE_REGIME_CATALOG,
    DEFAULT_REGIME_ID,
    DEFAULT_REGIME_NAME,
    TaskType,
)
from chain_replay_ml.model_taxonomy.specs import ModelContextKey
from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from chain_replay_ml.research_memory.signature import canonical_context_key


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CoverageClass(str, Enum):
    """Deterministic categorization of research evidence density for a ModelContextKey."""

    COLD_START = "COLD_START"  # 0 experiments / benchmarks
    SPARSE = "SPARSE"          # 1 to 4 experiments / benchmarks
    DEVELOPING = "DEVELOPING"  # 5 to 19 experiments / benchmarks (intermediate transition)
    MATURE = "MATURE"          # 20 or more experiments / benchmarks

    @classmethod
    def from_str(cls, value: str | Any) -> CoverageClass:
        raw = str(value or "").strip().upper()
        try:
            return cls(raw)
        except ValueError:
            pass
        mapping = {
            "COLD": cls.COLD_START,
            "UNEXPLORED": cls.COLD_START,
            "LOW": cls.SPARSE,
            "MODERATE": cls.DEVELOPING,
            "INTERMEDIATE": cls.DEVELOPING,
            "ACTIVE": cls.DEVELOPING,
            "HIGH": cls.MATURE,
            "ESTABLISHED": cls.MATURE,
        }
        if raw in mapping:
            return mapping[raw]
        return cls.COLD_START


def classify_coverage(experiment_or_benchmark_count: int) -> CoverageClass:
    """Deterministically classify coverage based on experiment or benchmark count.
    
    Thresholds:
        - 0: COLD_START
        - 1..4: SPARSE
        - 5..19: DEVELOPING
        - >= 20: MATURE
    """
    count = int(experiment_or_benchmark_count or 0)
    if count <= 0:
        return CoverageClass.COLD_START
    elif count < 5:
        return CoverageClass.SPARSE
    elif count < 20:
        return CoverageClass.DEVELOPING
    else:
        return CoverageClass.MATURE


def compute_evidence_density_score(
    benchmark_count: int,
    unique_features_count: int,
    total_registry_features: int = 50,
) -> float:
    """Calculate the safe bounded Context Evidence Density score in [0.0, 100.0].
    
    Mathematical Formulation:
        D(K) = 100.0 * (1.0 - exp(-N_benchmarks / 10.0)) * (0.5 + 0.5 * (N_features_tested / max(1, N_total_features)))
        
    Properties:
        - Exactly 0.0 when benchmark_count == 0.
        - Strict clamping to [0.0, 100.0].
        - Guaranteed safe against division by zero (handles total_registry_features <= 0).
        - Deterministic rounding to 4 decimal places.
    """
    b_count = max(0, int(benchmark_count or 0))
    if b_count == 0:
        return 0.0

    f_count = max(0, int(unique_features_count or 0))
    tot_feat = max(1, int(total_registry_features or 1))

    # Diminishing marginal returns on benchmark volume (saturates near ~30-50 benchmarks)
    volume_factor = 1.0 - math.exp(-float(b_count) / 10.0)

    # Feature coverage ratio (bounded in [0.0, 1.0])
    feature_ratio = min(1.0, float(f_count) / float(tot_feat))
    diversity_factor = 0.5 + (0.5 * feature_ratio)

    raw_score = 100.0 * volume_factor * diversity_factor
    clamped_score = min(100.0, max(0.0, raw_score))
    return round(clamped_score, 4)


@dataclass(frozen=True)
class ContextCoverage:
    """Coverage and evidence density profile for a single ModelContextKey."""

    context_key: str
    market: str
    sampling_interval_sec: int
    task_type: str
    prediction_horizon: str
    regime_id: str
    regime_name: str
    benchmark_count: int
    completed_campaign_count: int
    failed_campaign_count: int
    unique_experiment_count: int
    unique_feature_set_count: int
    unique_features_count: int
    temporal_span: dict[str, Any]
    evidence_density_score: float
    coverage_class: CoverageClass

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["coverage_class"] = self.coverage_class.value
        return d


@dataclass(frozen=True)
class CoverageMatrix:
    """Complete aggregated matrix across multiple ModelContextKeys."""

    contexts: list[ContextCoverage]
    total_contexts: int
    cold_start_count: int
    sparse_count: int
    developing_count: int
    mature_count: int
    average_density_score: float
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contexts": [c.to_dict() for c in self.contexts],
            "total_contexts": self.total_contexts,
            "cold_start_count": self.cold_start_count,
            "sparse_count": self.sparse_count,
            "developing_count": self.developing_count,
            "mature_count": self.mature_count,
            "average_density_score": self.average_density_score,
            "generated_at": self.generated_at,
        }


def analyze_context_coverage(
    data_dir: str,
    context_key: str,
    *,
    total_registry_features: int | None = None,
) -> ContextCoverage:
    """Analyze empirical evidence density and coverage metrics for a specific ModelContextKey.
    
    Reads exclusively from `analysis.db` using metadata aggregate queries.
    Zero loading of parquet matrices, ticks, or raw datasets.
    """
    init_analysis_db(data_dir)
    c_key_str = str(context_key).strip()

    # Parse and validate components via canonical ModelContextKey
    ctx_obj = ModelContextKey.from_key_str(c_key_str)
    market = ctx_obj.market
    sampling_sec = ctx_obj.sampling_interval_sec
    task_type = ctx_obj.task_type.value if hasattr(ctx_obj.task_type, "value") else str(ctx_obj.task_type)
    horizon = ctx_obj.prediction_horizon
    regime_id = ctx_obj.regime_id

    # Regime name resolution from canonical catalog
    regime_name = BASELINE_REGIME_CATALOG.get(regime_id, {}).get("name", DEFAULT_REGIME_NAME)

    conn = connect_analysis_db(data_dir)
    try:
        # 1. Benchmark aggregates & temporal span
        bm_row = conn.execute(
            """
            SELECT 
                COUNT(*) AS bm_count,
                MIN(created_at) AS min_ts,
                MAX(created_at) AS max_ts
            FROM model_benchmarks
            WHERE context_key = ?;
            """,
            (c_key_str,),
        ).fetchone()

        bm_count = int(bm_row["bm_count"] or 0) if bm_row else 0
        min_ts = bm_row["min_ts"] if bm_row and bm_row["min_ts"] else None
        max_ts = bm_row["max_ts"] if bm_row and bm_row["max_ts"] else None

        span_days = 0.0
        if min_ts and max_ts:
            try:
                t0 = datetime.fromisoformat(min_ts)
                t1 = datetime.fromisoformat(max_ts)
                span_days = round(max(0.0, (t1 - t0).total_seconds() / 86400.0), 2)
            except (ValueError, TypeError):
                span_days = 0.0

        temporal_span = {
            "earliest_benchmark": min_ts,
            "latest_benchmark": max_ts,
            "span_days": span_days,
        }

        # 2. Campaign aggregates
        camp_rows = conn.execute(
            """
            SELECT status, COUNT(*) AS cnt
            FROM research_campaigns
            WHERE context_key = ?
            GROUP BY status;
            """,
            (c_key_str,),
        ).fetchall()

        completed_camps = 0
        failed_camps = 0
        for cr in camp_rows:
            stat = str(cr["status"]).upper()
            cnt = int(cr["cnt"] or 0)
            if stat == "COMPLETED":
                completed_camps = cnt
            elif stat == "FAILED":
                failed_camps = cnt

        # 3. Experiment signatures & feature diversity
        exp_row = conn.execute(
            """
            SELECT 
                COUNT(DISTINCT signature_hash) AS u_exp,
                COUNT(DISTINCT feature_set_hash) AS u_fset
            FROM experiment_signatures
            WHERE context_key = ?;
            """,
            (c_key_str,),
        ).fetchone()

        u_exp_count = int(exp_row["u_exp"] or 0) if exp_row else 0
        u_fset_count = int(exp_row["u_fset"] or 0) if exp_row else 0

        # Extract unique feature names across all signatures in context
        sig_rows = conn.execute(
            "SELECT canonical_payload_json FROM experiment_signatures WHERE context_key = ?;",
            (c_key_str,),
        ).fetchall()

        unique_features: set[str] = set()
        for sr in sig_rows:
            raw_json = sr["canonical_payload_json"]
            if raw_json:
                try:
                    payload = json.loads(raw_json)
                    feats = payload.get("features", [])
                    if isinstance(feats, list):
                        unique_features.update(str(f).strip() for f in feats if f)
                except (json.JSONDecodeError, TypeError):
                    pass

        u_feat_count = len(unique_features)

        # Baseline registry feature count estimation if not explicitly passed
        tot_features = total_registry_features if total_registry_features is not None else max(50, u_feat_count)

        # 4. Density Score & Coverage Classification
        density_score = compute_evidence_density_score(
            benchmark_count=bm_count,
            unique_features_count=u_feat_count,
            total_registry_features=tot_features,
        )

        # Primary classification on total registered experiments (or benchmarks if signatures unavailable)
        exp_metric = max(bm_count, u_exp_count)
        cov_class = classify_coverage(exp_metric)

        return ContextCoverage(
            context_key=c_key_str,
            market=market,
            sampling_interval_sec=sampling_sec,
            task_type=task_type,
            prediction_horizon=horizon,
            regime_id=regime_id,
            regime_name=regime_name,
            benchmark_count=bm_count,
            completed_campaign_count=completed_camps,
            failed_campaign_count=failed_camps,
            unique_experiment_count=u_exp_count,
            unique_feature_set_count=u_fset_count,
            unique_features_count=u_feat_count,
            temporal_span=temporal_span,
            evidence_density_score=density_score,
            coverage_class=cov_class,
        )
    finally:
        conn.close()


def build_coverage_matrix(
    data_dir: str,
    *,
    explicit_context_keys: Sequence[str] | None = None,
    markets: Sequence[str] | None = None,
    sampling_intervals: Sequence[int] | None = None,
    task_types: Sequence[str] | None = None,
    prediction_horizons: Sequence[str] | None = None,
    regimes: Sequence[str] | None = None,
    total_registry_features: int | None = None,
) -> CoverageMatrix:
    """Construct an operational coverage matrix across multiple ModelContextKeys.
    
    If `explicit_context_keys` is provided, analyzes precisely those contexts.
    If filter dimensions (markets, task_types, regimes, etc.) are passed, computes the Cartesian
    grid across them.
    Otherwise, discovers all active context keys present in `analysis.db`.
    """
    init_analysis_db(data_dir)
    target_keys: set[str] = set()

    if explicit_context_keys is not None:
        target_keys.update(str(k).strip() for k in explicit_context_keys if str(k).strip())
    elif (
        markets is not None
        or sampling_intervals is not None
        or task_types is not None
        or prediction_horizons is not None
        or regimes is not None
    ):
        # Build Cartesian product over provided dimensions
        m_list = [str(m).upper().strip() for m in (markets or ["NIFTY"])]
        s_list = [int(s) for s in (sampling_intervals or [3])]
        t_list = [str(t).upper().strip() for t in (task_types or ["DIRECTION_CLASSIFIER"])]
        h_list = [str(h).strip() for h in (prediction_horizons or ["5m"])]
        r_list = [str(r).upper().strip() for r in (regimes or list(BASELINE_REGIME_CATALOG.keys()))]

        for m in m_list:
            for s in s_list:
                for t in t_list:
                    for h in h_list:
                        for r in r_list:
                            ckey = canonical_context_key(
                                market=m,
                                sampling_interval_sec=s,
                                task_type=t,
                                prediction_horizon=h,
                                regime_id=r,
                            )
                            target_keys.add(ckey)
    else:
        # Discover all contexts currently present in analysis.db
        conn = connect_analysis_db(data_dir)
        try:
            rows_bm = conn.execute("SELECT DISTINCT context_key FROM model_benchmarks;").fetchall()
            rows_sig = conn.execute("SELECT DISTINCT context_key FROM experiment_signatures;").fetchall()
            rows_camp = conn.execute("SELECT DISTINCT context_key FROM research_campaigns;").fetchall()

            for r in rows_bm:
                if r["context_key"]:
                    target_keys.add(str(r["context_key"]).strip())
            for r in rows_sig:
                if r["context_key"]:
                    target_keys.add(str(r["context_key"]).strip())
            for r in rows_camp:
                if r["context_key"]:
                    target_keys.add(str(r["context_key"]).strip())
        finally:
            conn.close()

    # Sort context keys deterministically
    sorted_keys = sorted(list(target_keys))

    contexts: list[ContextCoverage] = []
    cold_count = 0
    sparse_count = 0
    dev_count = 0
    mature_count = 0
    total_density = 0.0

    for ckey in sorted_keys:
        cov = analyze_context_coverage(
            data_dir,
            ckey,
            total_registry_features=total_registry_features,
        )
        contexts.append(cov)
        total_density += cov.evidence_density_score

        if cov.coverage_class == CoverageClass.COLD_START:
            cold_count += 1
        elif cov.coverage_class == CoverageClass.SPARSE:
            sparse_count += 1
        elif cov.coverage_class == CoverageClass.DEVELOPING:
            dev_count += 1
        elif cov.coverage_class == CoverageClass.MATURE:
            mature_count += 1

    tot_contexts = len(contexts)
    avg_density = round(total_density / float(tot_contexts), 4) if tot_contexts > 0 else 0.0

    return CoverageMatrix(
        contexts=contexts,
        total_contexts=tot_contexts,
        cold_start_count=cold_count,
        sparse_count=sparse_count,
        developing_count=dev_count,
        mature_count=mature_count,
        average_density_score=avg_density,
        generated_at=_utc_now_iso(),
    )
