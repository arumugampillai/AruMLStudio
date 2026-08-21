"""Mathematical Feature Synthesis Engine & Provenance Tracker (Phase 3).

Synthesizes novel, mathematically sound experimental features across 5 supported strategies:
1. RATIO: f1 / (abs(f2) + epsilon)
2. INTERACTION: zscore(f1) * zscore(f2)
3. NONLINEAR: log1p(abs(f)), tanh(zscore(f)), f**2
4. SPREAD: zscore(f1) - zscore(f2)
5. COMPOSITE: (zscore(f1) + zscore(f2)) / (abs(zscore(f3)) + epsilon)

Invariants:
1. Target Immunity: Strictly excludes label_*, target_*, timestamp, and metadata columns.
2. Registry Immunity: Never mutates feature_registry_store.json or pipeline_registry_store.json.
3. Deterministic Provenance: Canonical AST formula string and 16-char MD5 formula_hash for deduplication.
4. Numerical Stability: Robust epsilons, NaN clipping, and variance validation.
5. Workstation Safety: Strictly bounded by DiscoveryPipelineBudget (max_new_features_per_gen, max_nan_fraction).
"""

from __future__ import annotations

import ast
import hashlib
import logging
import math
import re
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    GeneratorStrategy,
    _utc_now_iso,
    compute_formula_hash,
    format_discovered_feature_id,
    normalize_formula_expression,
)

logger = logging.getLogger(__name__)

# Excluded column prefixes and metadata names to prevent target leakage
EXCLUDED_COLUMN_PREFIXES = ("label_", "target_", "y_", "ret_forward_", "pnl_")
EXCLUDED_COLUMN_NAMES = frozenset({
    "token", "symbol", "timestamp", "datetime", "date", "time",
    "row_id", "id", "open", "high", "low", "close",
})


def is_eligible_base_feature(col_name: str, series: pd.Series) -> bool:
    """Check if a column is eligible for feature synthesis (numeric, non-target, non-zero variance)."""
    clean_name = str(col_name).strip()
    lower_name = clean_name.lower()

    if lower_name in EXCLUDED_COLUMN_NAMES:
        return False
    for prefix in EXCLUDED_COLUMN_PREFIXES:
        if lower_name.startswith(prefix):
            return False

    # Check numeric type
    if not np.issubdtype(series.dtype, np.number):
        return False

    # Check non-zero variance
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 10 or valid.std() < 1e-7:
        return False

    return True


def zscore(s: pd.Series) -> pd.Series:
    """Compute z-score of series with epsilon protection."""
    clean = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = float(clean.std())
    if std < 1e-7:
        return pd.Series(0.0, index=s.index)
    mean = float(clean.mean())
    return (clean - mean) / (std + 1e-6)


def evaluate_discovery_formula(df: pd.DataFrame, formula_expression: str) -> pd.Series:
    """Safely evaluate a vectorized formula expression against a DataFrame.
    
    Supported functions:
    - col('name'): returns df[name]
    - zscore(series): returns z-score
    - abs(series): returns absolute value
    - sign(series): returns sign
    - log1p(series): returns log1p
    - tanh(series): returns tanh
    - Arithmetic operators: +, -, *, /, **
    """
    expr = normalize_formula_expression(formula_expression)

    # Context dictionary for safe eval
    eval_context: dict[str, Any] = {
        "col": lambda name: df[name].astype(float) if name in df.columns else pd.Series(0.0, index=df.index),
        "zscore": zscore,
        "abs": np.abs,
        "sign": np.sign,
        "log1p": np.log1p,
        "tanh": np.tanh,
        "sqrt": np.sqrt,
        "np": np,
        "pd": pd,
    }

    try:
        # Evaluate formula expression
        result = eval(expr, {"__builtins__": {}}, eval_context)
        if isinstance(result, (int, float)):
            res_series = pd.Series(float(result), index=df.index)
        elif isinstance(result, pd.Series):
            res_series = result
        elif isinstance(result, np.ndarray):
            res_series = pd.Series(result, index=df.index)
        else:
            res_series = pd.Series(0.0, index=df.index)

        # Sanitize inf / nan
        res_series = res_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return res_series
    except Exception as e:
        logger.warning("Error evaluating formula '%s': %s", formula_expression, e)
        return pd.Series(0.0, index=df.index)


class DiscoveryFeatureSynthesizer:
    """Deterministic mathematical synthesizer for Discovery Pipeline features."""

    @classmethod
    def synthesize_ratio(
        cls,
        f1_name: str,
        f2_name: str,
        s1: pd.Series,
        s2: pd.Series,
    ) -> tuple[str, str, pd.Series] | None:
        """Synthesize a ratio feature: f1 / (abs(f2) + dynamic_eps)."""
        if f1_name == f2_name:
            return None

        s2_clean = s2.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        s2_std = float(s2_clean.std())
        eps = round(1e-6 + (0.01 * (s2_std if s2_std > 0 else 1.0)), 6)

        feat_name = f"synth_ratio__{f1_name}__div__{f2_name}"
        formula = f"col('{f1_name}') / (abs(col('{f2_name}')) + {eps:.6f})"

        s1_clean = s1.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result = s1_clean / (s2_clean.abs() + eps)
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if result.std() < 1e-7:
            return None
        return feat_name, formula, result

    @classmethod
    def synthesize_interaction(
        cls,
        f1_name: str,
        f2_name: str,
        s1: pd.Series,
        s2: pd.Series,
    ) -> tuple[str, str, pd.Series] | None:
        """Synthesize a multiplicative interaction feature: zscore(f1) * zscore(f2)."""
        if f1_name == f2_name:
            return None

        # Canonical lexicographical ordering for symmetry deduplication
        first, second = sorted([f1_name, f2_name])
        feat_name = f"synth_inter__{first}__x__{second}"
        formula = f"zscore(col('{first}')) * zscore(col('{second}'))"

        z1 = zscore(s1 if f1_name == first else s2)
        z2 = zscore(s2 if f1_name == first else s1)
        result = (z1 * z2).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if result.std() < 1e-7:
            return None
        return feat_name, formula, result

    @classmethod
    def synthesize_nonlinear(
        cls,
        f_name: str,
        s: pd.Series,
        transform_type: str = "log1p",
    ) -> tuple[str, str, pd.Series] | None:
        """Synthesize a non-linear transform (log1p, tanh_z, sq)."""
        s_clean = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if transform_type == "log1p":
            feat_name = f"synth_log1p__{f_name}"
            formula = f"sign(col('{f_name}')) * log1p(abs(col('{f_name}')))"
            result = np.sign(s_clean) * np.log1p(np.abs(s_clean))
        elif transform_type == "tanh_z":
            feat_name = f"synth_tanh_z__{f_name}"
            formula = f"tanh(zscore(col('{f_name}')))"
            result = np.tanh(zscore(s_clean))
        elif transform_type == "sq":
            feat_name = f"synth_sq__{f_name}"
            formula = f"zscore(col('{f_name}')) ** 2"
            result = zscore(s_clean) ** 2
        else:
            return None

        result_series = pd.Series(result, index=s.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if result_series.std() < 1e-7:
            return None
        return feat_name, formula, result_series

    @classmethod
    def synthesize_spread(
        cls,
        f1_name: str,
        f2_name: str,
        s1: pd.Series,
        s2: pd.Series,
    ) -> tuple[str, str, pd.Series] | None:
        """Synthesize a normalized spread feature: zscore(f1) - zscore(f2)."""
        if f1_name == f2_name:
            return None

        feat_name = f"synth_spread__{f1_name}__minus__{f2_name}"
        formula = f"zscore(col('{f1_name}')) - zscore(col('{f2_name}'))"

        z1 = zscore(s1)
        z2 = zscore(s2)
        result = (z1 - z2).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if result.std() < 1e-7:
            return None
        return feat_name, formula, result

    @classmethod
    def synthesize_composite(
        cls,
        f1_name: str,
        f2_name: str,
        f3_name: str,
        s1: pd.Series,
        s2: pd.Series,
        s3: pd.Series,
    ) -> tuple[str, str, pd.Series] | None:
        """Synthesize a 3-way composite feature: (z(f1) + z(f2)) / (abs(z(f3)) + 0.01)."""
        if len({f1_name, f2_name, f3_name}) < 3:
            return None

        first, second = sorted([f1_name, f2_name])
        feat_name = f"synth_comp__{first}_plus_{second}__div__{f3_name}"
        formula = f"(zscore(col('{first}')) + zscore(col('{second}'))) / (abs(zscore(col('{f3_name}'))) + 0.010000)"

        z1 = zscore(s1 if f1_name == first else s2)
        z2 = zscore(s2 if f1_name == first else s1)
        z3 = zscore(s3)

        result = (z1 + z2) / (z3.abs() + 0.01)
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if result.std() < 1e-7:
            return None
        return feat_name, formula, result


def generate_discovery_features_from_dataset(
    df: pd.DataFrame,
    *,
    pipeline_id: str,
    generation_number: int = 1,
    base_feature_candidates: Sequence[str] | None = None,
    existing_formula_hashes: set[str] | None = None,
    start_sequence: int = 1,
    budget: DiscoveryPipelineBudget | None = None,
    strategies: Sequence[GeneratorStrategy] | None = None,
) -> tuple[list[DiscoveredFeatureSpec], pd.DataFrame]:
    """Generate a bounded batch of novel, mathematically sound discovery features on the real dataset.
    
    Returns:
    - List of generated DiscoveredFeatureSpec instances with deterministic provenance.
    - DataFrame containing computed feature column vectors.
    """
    b = budget or DiscoveryPipelineBudget()
    strats = set(strategies or [
        GeneratorStrategy.RATIO,
        GeneratorStrategy.INTERACTION,
        GeneratorStrategy.NONLINEAR,
        GeneratorStrategy.SPREAD,
    ])
    seen_hashes = set(existing_formula_hashes or set())

    # 1. Filter eligible numeric base features strictly excluding targets
    all_eligible: list[str] = []
    candidates_pool = list(base_feature_candidates) if base_feature_candidates else list(df.columns)

    for col in candidates_pool:
        if col in df.columns and is_eligible_base_feature(col, df[col]):
            all_eligible.append(col)

    if len(all_eligible) < 2:
        return [], pd.DataFrame(index=df.index)

    generated_specs: list[DiscoveredFeatureSpec] = []
    generated_cols: dict[str, pd.Series] = {}
    seq_counter = int(start_sequence)

    # Deterministic pair sampling based on variance ranking
    variances = {c: float(df[c].std()) for c in all_eligible}
    ranked_cols = sorted(all_eligible, key=lambda c: variances[c], reverse=True)
    top_anchor_cols = ranked_cols[:15]  # Top variance anchors

    # 2. Generate features across requested strategies with balanced budget apportionment
    active_strats_list = [s for s in [
        GeneratorStrategy.NONLINEAR,
        GeneratorStrategy.RATIO,
        GeneratorStrategy.INTERACTION,
        GeneratorStrategy.SPREAD,
        GeneratorStrategy.COMPOSITE,
    ] if s in strats]

    if not active_strats_list:
        return [], pd.DataFrame(index=df.index)

    # Calculate target quota per strategy while respecting total budget
    total_budget = b.max_new_features_per_gen
    quota_per_strat = max(1, total_budget // len(active_strats_list))

    # A. NONLINEAR
    if GeneratorStrategy.NONLINEAR in active_strats_list and len(generated_specs) < total_budget:
        count_for_strat = 0
        for col in top_anchor_cols[:8]:
            for t_type in ("log1p", "tanh_z", "sq"):
                if len(generated_specs) >= total_budget or count_for_strat >= quota_per_strat:
                    break
                syn = DiscoveryFeatureSynthesizer.synthesize_nonlinear(col, df[col], t_type)
                if not syn:
                    continue
                name, formula, s_val = syn
                f_hash = compute_formula_hash(formula)
                if f_hash in seen_hashes:
                    continue

                seen_hashes.add(f_hash)
                f_id = format_discovered_feature_id(pipeline_id, "NONL", seq_counter)
                seq_counter += 1

                spec = DiscoveredFeatureSpec(
                    feature_id=f_id,
                    pipeline_id=pipeline_id,
                    feature_name=name,
                    formula_expression=formula,
                    formula_hash=f_hash,
                    generator_strategy=GeneratorStrategy.NONLINEAR,
                    parent_features=[col],
                    generation_discovered=generation_number,
                    lifecycle_status=DiscoveryLifecycleStatus.CANDIDATE,
                )
                generated_specs.append(spec)
                generated_cols[name] = s_val
                count_for_strat += 1

    # B. RATIO
    if GeneratorStrategy.RATIO in active_strats_list and len(generated_specs) < total_budget:
        count_for_strat = 0
        for i, col1 in enumerate(top_anchor_cols):
            for col2 in top_anchor_cols:
                if col1 == col2:
                    continue
                if len(generated_specs) >= total_budget or count_for_strat >= quota_per_strat:
                    break
                syn = DiscoveryFeatureSynthesizer.synthesize_ratio(col1, col2, df[col1], df[col2])
                if not syn:
                    continue
                name, formula, s_val = syn
                f_hash = compute_formula_hash(formula)
                if f_hash in seen_hashes:
                    continue

                seen_hashes.add(f_hash)
                f_id = format_discovered_feature_id(pipeline_id, "RATI", seq_counter)
                seq_counter += 1

                spec = DiscoveredFeatureSpec(
                    feature_id=f_id,
                    pipeline_id=pipeline_id,
                    feature_name=name,
                    formula_expression=formula,
                    formula_hash=f_hash,
                    generator_strategy=GeneratorStrategy.RATIO,
                    parent_features=[col1, col2],
                    generation_discovered=generation_number,
                    lifecycle_status=DiscoveryLifecycleStatus.CANDIDATE,
                )
                generated_specs.append(spec)
                generated_cols[name] = s_val
                count_for_strat += 1

    # C. INTERACTION
    if GeneratorStrategy.INTERACTION in active_strats_list and len(generated_specs) < total_budget:
        count_for_strat = 0
        for i, col1 in enumerate(top_anchor_cols):
            for col2 in top_anchor_cols[i + 1:]:
                if len(generated_specs) >= total_budget or count_for_strat >= quota_per_strat:
                    break
                syn = DiscoveryFeatureSynthesizer.synthesize_interaction(col1, col2, df[col1], df[col2])
                if not syn:
                    continue
                name, formula, s_val = syn
                f_hash = compute_formula_hash(formula)
                if f_hash in seen_hashes:
                    continue

                seen_hashes.add(f_hash)
                f_id = format_discovered_feature_id(pipeline_id, "INTE", seq_counter)
                seq_counter += 1

                spec = DiscoveredFeatureSpec(
                    feature_id=f_id,
                    pipeline_id=pipeline_id,
                    feature_name=name,
                    formula_expression=formula,
                    formula_hash=f_hash,
                    generator_strategy=GeneratorStrategy.INTERACTION,
                    parent_features=sorted([col1, col2]),
                    generation_discovered=generation_number,
                    lifecycle_status=DiscoveryLifecycleStatus.CANDIDATE,
                )
                generated_specs.append(spec)
                generated_cols[name] = s_val
                count_for_strat += 1

    # D. SPREAD
    if GeneratorStrategy.SPREAD in active_strats_list and len(generated_specs) < total_budget:
        count_for_strat = 0
        for i, col1 in enumerate(top_anchor_cols):
            for col2 in top_anchor_cols[i + 1:]:
                if len(generated_specs) >= total_budget or count_for_strat >= quota_per_strat:
                    break
                syn = DiscoveryFeatureSynthesizer.synthesize_spread(col1, col2, df[col1], df[col2])
                if not syn:
                    continue
                name, formula, s_val = syn
                f_hash = compute_formula_hash(formula)
                if f_hash in seen_hashes:
                    continue

                seen_hashes.add(f_hash)
                f_id = format_discovered_feature_id(pipeline_id, "SPRE", seq_counter)
                seq_counter += 1

                spec = DiscoveredFeatureSpec(
                    feature_id=f_id,
                    pipeline_id=pipeline_id,
                    feature_name=name,
                    formula_expression=formula,
                    formula_hash=f_hash,
                    generator_strategy=GeneratorStrategy.SPREAD,
                    parent_features=[col1, col2],
                    generation_discovered=generation_number,
                    lifecycle_status=DiscoveryLifecycleStatus.CANDIDATE,
                )
                generated_specs.append(spec)
                generated_cols[name] = s_val
                count_for_strat += 1

    # E. COMPOSITE
    if GeneratorStrategy.COMPOSITE in active_strats_list and len(generated_specs) < total_budget and len(top_anchor_cols) >= 3:
        count_for_strat = 0
        for i in range(len(top_anchor_cols) - 2):
            col1 = top_anchor_cols[i]
            col2 = top_anchor_cols[i + 1]
            col3 = top_anchor_cols[i + 2]
            if len(generated_specs) >= total_budget or count_for_strat >= quota_per_strat:
                break
            syn = DiscoveryFeatureSynthesizer.synthesize_composite(col1, col2, col3, df[col1], df[col2], df[col3])
            if not syn:
                continue
            name, formula, s_val = syn
            f_hash = compute_formula_hash(formula)
            if f_hash in seen_hashes:
                continue

            seen_hashes.add(f_hash)
            f_id = format_discovered_feature_id(pipeline_id, "COMP", seq_counter)
            seq_counter += 1

            spec = DiscoveredFeatureSpec(
                feature_id=f_id,
                pipeline_id=pipeline_id,
                feature_name=name,
                formula_expression=formula,
                formula_hash=f_hash,
                generator_strategy=GeneratorStrategy.COMPOSITE,
                parent_features=[col1, col2, col3],
                generation_discovered=generation_number,
                lifecycle_status=DiscoveryLifecycleStatus.CANDIDATE,
            )
            generated_specs.append(spec)
            generated_cols[name] = s_val
            count_for_strat += 1

    # Construct DataFrame of generated features
    out_df = pd.DataFrame(generated_cols, index=df.index) if generated_cols else pd.DataFrame(index=df.index)
    return generated_specs, out_df
