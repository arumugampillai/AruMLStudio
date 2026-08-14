"""Feature ownership catalog — Base / Computed Base / Pipeline Owned / Retired.

Canonical source for the Feature Ownership architecture
(``docs/master-dataset/FEATURE_OWNERSHIP.md``).

Classification is by computation role, not by name. Historical Master leftovers
have been migrated (Pipeline Owned) or retired; ``HISTORICAL_FEATURES`` is empty.
"""

from __future__ import annotations

from typing import Any, Literal

OwnershipCategory = Literal["base", "computed_base", "historical", "pipeline_owned", "retired"]

OWNERSHIP_BASE = "base"
OWNERSHIP_COMPUTED_BASE = "computed_base"
OWNERSHIP_HISTORICAL = "historical"
OWNERSHIP_PIPELINE_OWNED = "pipeline_owned"
OWNERSHIP_RETIRED = "retired"

# Future generator for Historical / Pipeline-Owned features (pipeline transform family).
FutureGenerator = Literal[
    "lag",
    "difference",
    "return",
    "difference_clip",
    "rolling_zscore",
    "rolling_ohlc",
    "interaction",
    "derived",
    "historical",
]

# ---------------------------------------------------------------------------
# Explicit Historical set — Phase 3 complete: empty (all migrated or retired).
# ---------------------------------------------------------------------------

HISTORICAL_FEATURES: dict[str, FutureGenerator] = {}

# Computed Base — algorithm/streaming PIT (not Historical even if stateful).
_COMPUTED_BASE_MARKERS: tuple[str, ...] = (
    "_ema",
    "ema9_",
    "ema_",
    "std20",
    "_rv_",
    "rv_ratio",
    "spot_vs_ema",
    "time_since_cross",
    "cross_age",
    "price_dist_from_cross",
    "weighted_",
    "channel_width",
    "spot_high_ema",
    "spot_low_ema",
    "spot_up_score_",
    "spot_down_score_",
    "spot_up_sample_count",
    "spot_down_sample_count",
    "oi_abs_delta_",
)

_COMPUTED_BASE_EXACT: frozenset[str] = frozenset({
    "bs_reiv_pred",
    "dgt_reiv_pred",
    "roll_iv",
    "roll_age_min",
    "rows_since_roll",
    "iv_drift_from_roll",
    "iv_rank_session",
    "dgt_prediction_error",
    "ema9_gt_ema20",
    "ema9_slope",
    # Wave A: Market Microstructure Controller levels
    "mid_price",
    "microprice",
    "microprice_bias",
    "book_imbalance_l1",
    "book_imbalance_l1_5",
    "bid_depth_l1_5",
    "ask_depth_l1_5",
    "book_depth_slope_bid",
    "book_depth_slope_ask",
    # Wave B: Chain IV skew
    "iv_skew_atm",
    "iv_call_put_skew",
    "iv_skew_25d",
    "iv_butterfly_25d",
    "iv_rv_spread_5m",
    "iv_rv_spread_10m",
    "delta_w_volume_flow_1m",
    "delta_w_volume_flow_5m",
    "call_gex",
    "put_gex",
    "net_gex",
    "chain_gex",
    "gamma_flip_spot",
    "gamma_flip_distance",
    "synthetic_forward_spot",
    # Freeze Tier 2 / chain totals
    "atm_iv_ce",
    "atm_iv_pe",
    "total_call_oi",
    "total_put_oi",
    "total_ce_volume",
    "total_pe_volume",
    "otm_ce_volume",
    "otm_pe_volume",
    "otm_pcr_volume",
    # Charm / Speed — second-order BS greeks from current state
    "charm",
    "speed",
    # Wave 6: ema_spread_pct / ema_spread_vs_spot_pct → Interaction (pipeline_owned)
})


def _is_computed_base_name(name: str) -> bool:
    if name in _COMPUTED_BASE_EXACT:
        return True
    return any(m in name for m in _COMPUTED_BASE_MARKERS)


def ownership_of(feature: str) -> OwnershipCategory:
    """Return ownership category for a registry feature name."""
    name = str(feature or "").strip()
    if not name:
        return OWNERSHIP_BASE
    if is_interaction_feature(name):
        return OWNERSHIP_PIPELINE_OWNED
    try:
        from .feature_migration import is_pipeline_owned, is_retired

        if is_retired(name):
            return OWNERSHIP_RETIRED
        if is_pipeline_owned(name):
            return OWNERSHIP_PIPELINE_OWNED
    except Exception:
        pass
    if name in HISTORICAL_FEATURES:
        return OWNERSHIP_HISTORICAL
    if _is_computed_base_name(name):
        return OWNERSHIP_COMPUTED_BASE
    return OWNERSHIP_BASE


def future_generator_of(feature: str) -> FutureGenerator | None:
    """Pipeline transform family for Historical / Pipeline-Owned features."""
    name = str(feature or "").strip()
    if name in HISTORICAL_FEATURES:
        return HISTORICAL_FEATURES[name]
    try:
        from .feature_migration import PIPELINE_OWNED_GENERATORS

        gen = PIPELINE_OWNED_GENERATORS.get(name)
        if gen is not None:
            return gen  # type: ignore[return-value]
    except Exception:
        pass
    return None


def is_canonical(feature: str) -> bool:
    """True when the feature belongs in Master / Feature Registry permanently."""
    cat = ownership_of(feature)
    return cat in (OWNERSHIP_BASE, OWNERSHIP_COMPUTED_BASE)


def non_registry_transform_source_names(data_dir: str | None = None) -> frozenset[str]:
    """Names unsuitable as lag/diff/return sources when Master export is the input grid."""
    from .feature_migration import is_pipeline_owned

    names: set[str] = set(_COMPUTED_BASE_EXACT)
    if data_dir:
        try:
            from .feature_sources_catalog import registry_feature_names

            for n in registry_feature_names(data_dir=data_dir):
                if _is_computed_base_name(n) or is_pipeline_owned(n):
                    names.add(n)
        except Exception:
            pass
    return frozenset(names)


def litmus_is_historical(requires_prior_rows: bool) -> bool:
    """Litmus test helper: deleting all previous rows would make it impossible."""
    return bool(requires_prior_rows)


# Name patterns that usually mean history/transform-of-another-feature (not canonical).
_HISTORICAL_NAME_HINTS: tuple[str, ...] = (
    "_lag_",
    "_return_",
    "_change_",
    "_diff_",
    "_zscore_",
    "_prev",
    "_flow_",
    "_slope_",
    "_accel",
    "pct_change_from_",
    "_x_",  # interaction products (e.g. ltp_x_volume_…)
)


class RegistryAdmissionError(ValueError):
    """New Feature Registry entry rejected by the ownership admission gate."""


def looks_historical_by_name(feature: str) -> bool:
    """Heuristic: name suggests a Historical / Derived feature."""
    name = str(feature or "").strip().lower()
    if not name:
        return False
    if name.endswith("_prev1") or name.endswith("_prev2") or name.endswith("_prev3"):
        return True
    return any(h in name for h in _HISTORICAL_NAME_HINTS)


# InteractionTransformation auto-name infixes — *evidence* that a feature was
# produced by the Interaction plugin, not the definition of Registry eligibility.
# Do NOT include ``_minus_`` / ``_min_`` / ``_max_`` — they collide with base names
# (e.g. ``ce_minus_pe_atm6_ltp``, ``distance_to_max_call_oi_strikes``).
_INTERACTION_NAME_INFIXES: tuple[str, ...] = (
    "_x_",
    "_div_",
    "_plus_",
    "_absdiff_",
)

# Semantic admission (see docs/master-dataset/FEATURE_OWNERSHIP.md).
REGISTRY_ADMISSION_RULE = (
    "Decision tree: raw or foundational market observation that every Master "
    "Dataset should expose → Base; else canonical controller/market-model output → "
    "Computed Base; else recreatable from registry features or Dataset Builder "
    "helpers → Transformation Pipeline (never Registry); else review as a new "
    "canonical model before any Registry admission."
)

CONTROLLER_EMISSION_INVARIANT = (
    "A controller should emit the smallest complete set of canonical market-state "
    "values from which all controller-specific derived features can be reconstructed. "
    "Emit levels (e.g. ltp_ema100, iv_ema100, spot_rv_5m, delta) — never presentation "
    "or experiment packaging (÷ltp, ÷spot, ×delta, cross-feature products). Those "
    "belong in the Transformation Pipeline."
)

# Placement leaves from walk_feature_placement_tree().
PLACEMENT_BASE = "base"
PLACEMENT_COMPUTED_BASE = "computed_base"
PLACEMENT_PIPELINE = "transformation_pipeline"
PLACEMENT_REVIEW_NEW_MODEL = "review_new_canonical_model"
PLACEMENT_EDGE = "edge"


# ── EDGE features ──────────────────────────────────────────────────────────
# Features that *could* move to the Pipeline but are blocked by a specific
# missing capability.  Each entry maps ``feature_name`` →
# ``{"reason_code": str, "description": str, "canonical_inputs": list[str]}``.
# When a new Transformation Pipeline capability lands, query this dict by
# ``reason_code`` to find every feature that should be reassessed.

EDGE_REASON_CONDITIONAL_COLUMN_SELECTION = "CONDITIONAL_COLUMN_SELECTION"
EDGE_REASON_WEIGHTED_SCALAR_BLEND = "WEIGHTED_SCALAR_BLEND"

EDGE_FEATURES: dict[str, dict[str, Any]] = {
    "side_to_ltp_ratio": {
        "reason_code": EDGE_REASON_CONDITIONAL_COLUMN_SELECTION,
        "description": (
            "ce_atm6_ltp_sum or pe_atm6_ltp_sum ÷ ltp depending on option_type. "
            "Pipeline cannot express conditional column selection."
        ),
        "canonical_inputs": ["ce_atm6_ltp_sum", "pe_atm6_ltp_sum", "ltp", "is_call"],
    },
}


def edge_features_blocked_by(reason_code: str) -> list[str]:
    """Return feature names whose EDGE status is due to *reason_code*."""
    return sorted(
        name
        for name, meta in EDGE_FEATURES.items()
        if meta.get("reason_code") == reason_code
    )

REGISTRY_ADMISSION_DECISION_TREE = """
New Feature
      │
      ▼
Is it a raw or foundational market observation
that every Master Dataset should expose?
      │
 ┌────┴────┐
 │         │
Yes       No
 │         │
 ▼         ▼
Base   Is it the output of a canonical
       controller / market model?
             │
        ┌────┴────┐
        │         │
       Yes       No
        │         │
        ▼         ▼
Computed Base   Can it be recreated from
                existing registry features
                or Dataset Builder helpers?
                     │
                ┌────┴────┐
                │         │
               Yes       No
                │         │
                ▼         ▼
        Transformation   Review as new
            Pipeline      canonical model
""".strip()


def walk_feature_placement_tree(
    *,
    foundational_market_observation: bool | None = None,
    raw_market_observation: bool | None = None,
    canonical_controller_or_market_model: bool | None = None,
    recreatable_from_registry_or_helpers: bool | None = None,
) -> dict[str, Any]:
    """Walk the approved placement decision tree (semantic — not name-based).

    First branch: is this a **raw or foundational** market observation that every
    Master Dataset should expose? That is why ``ltp_to_spot_ratio`` / ``moneyness``
    stay Base while ordinary compositions like ``spot_rv_ratio`` go to the pipeline.

    ``raw_market_observation`` is accepted as an alias of
    ``foundational_market_observation``.

    Returns ``{"placement": str, "registry_eligible": bool, "reason": str}``.
    """
    if foundational_market_observation is None:
        foundational_market_observation = raw_market_observation

    if foundational_market_observation is True:
        return {
            "placement": PLACEMENT_BASE,
            "registry_eligible": True,
            "reason": (
                "Raw or foundational market observation that every Master Dataset "
                "should expose → Base (Feature Registry / Master)."
            ),
        }
    if foundational_market_observation is False:
        if canonical_controller_or_market_model is True:
            return {
                "placement": PLACEMENT_COMPUTED_BASE,
                "registry_eligible": True,
                "reason": (
                    "Output of a canonical controller / market model → Computed Base "
                    "(Feature Registry / Master)."
                ),
            }
        if canonical_controller_or_market_model is False:
            if recreatable_from_registry_or_helpers is True:
                return {
                    "placement": PLACEMENT_PIPELINE,
                    "registry_eligible": False,
                    "reason": (
                        "Ordinary composition recreatable from existing registry features "
                        "or Dataset Builder helpers → Transformation Pipeline. Never "
                        "considered for Feature Registry."
                    ),
                }
            if recreatable_from_registry_or_helpers is False:
                return {
                    "placement": PLACEMENT_REVIEW_NEW_MODEL,
                    "registry_eligible": False,
                    "reason": (
                        "Not recreatable from registry/helpers and not yet a canonical "
                        "controller output → review as a new canonical market model "
                        "before any Registry admission."
                    ),
                }
            return {
                "placement": None,
                "registry_eligible": False,
                "reason": (
                    "Not foundational / not controller-owned: declare "
                    "recreatable_from_registry_or_helpers=True|False to finish the tree."
                ),
            }
        return {
            "placement": None,
            "registry_eligible": False,
            "reason": (
                "Not a foundational market observation: declare "
                "canonical_controller_or_market_model=True|False to continue the tree."
            ),
        }
    return {
        "placement": None,
        "registry_eligible": False,
        "reason": (
            "Declare foundational_market_observation=True|False to start the "
            "placement tree (alias: raw_market_observation)."
        ),
    }


def is_interaction_feature(feature: str) -> bool:
    """True when evidence says the feature is an InteractionTransformation product.

    This is a **detector**, not the admission rule. Prefer declaring
    ``produced_by=\"interaction\"`` or the placement-tree answers at admission time.
    """
    name = str(feature or "").strip()
    if not name:
        return False
    lower = name.lower()
    if any(infix in lower for infix in _INTERACTION_NAME_INFIXES):
        return True
    try:
        from .feature_migration import PIPELINE_OWNED_GENERATORS

        if PIPELINE_OWNED_GENERATORS.get(name) == "interaction":
            return True
    except Exception:
        pass
    return False


def evaluate_registry_admission(
    feature: str,
    *,
    ownership: str | None = None,
    requires_prior_rows: bool | None = None,
    allow_historical: bool = False,
    historical_exception_reason: str | None = None,
    produced_by: str | None = None,
    dataset_builder_configurable: bool | None = None,
    generic_registry_math: bool | None = None,
    foundational_market_observation: bool | None = None,
    raw_market_observation: bool | None = None,
    canonical_controller_or_market_model: bool | None = None,
    recreatable_from_registry_or_helpers: bool | None = None,
) -> dict[str, Any]:
    """Gate for admitting a **new** name into the Feature Registry.

    Preferred path: answer the placement decision tree
    (``foundational_market_observation`` → ``canonical_controller_or_market_model`` →
    ``recreatable_from_registry_or_helpers``). Pipeline leaves are never Registry-
    eligible. ``review_new_canonical_model`` stays rejected until a controller /
    market model exists.

    ``raw_market_observation`` is an alias of ``foundational_market_observation``.

    Legacy/auxiliary declarations still reject pipeline-bound work:

    - ``produced_by=\"interaction\"``
    - ``dataset_builder_configurable=True``
    - ``generic_registry_math=True``

    Returns ``{"allowed": bool, "reason": str, "category": str | None, "placement": ...}``.
    """
    name = str(feature or "").strip()
    if not name:
        return {
            "allowed": False,
            "reason": "Feature name is empty.",
            "category": None,
            "placement": None,
        }

    if foundational_market_observation is None:
        foundational_market_observation = raw_market_observation

    # --- Decision tree (when any node is answered) ---
    tree_answered = any(
        v is not None
        for v in (
            foundational_market_observation,
            canonical_controller_or_market_model,
            recreatable_from_registry_or_helpers,
        )
    )
    if tree_answered:
        walked = walk_feature_placement_tree(
            foundational_market_observation=foundational_market_observation,
            canonical_controller_or_market_model=canonical_controller_or_market_model,
            recreatable_from_registry_or_helpers=recreatable_from_registry_or_helpers,
        )
        placement = walked.get("placement")
        if placement == PLACEMENT_BASE:
            return {
                "allowed": True,
                "reason": str(walked["reason"]),
                "category": OWNERSHIP_BASE,
                "placement": placement,
            }
        if placement == PLACEMENT_COMPUTED_BASE:
            return {
                "allowed": True,
                "reason": str(walked["reason"]),
                "category": OWNERSHIP_COMPUTED_BASE,
                "placement": placement,
            }
        if placement == PLACEMENT_PIPELINE:
            return {
                "allowed": False,
                "reason": str(walked["reason"]) + f" ({REGISTRY_ADMISSION_RULE})",
                "category": "transformation_pipeline",
                "placement": placement,
            }
        if placement == PLACEMENT_REVIEW_NEW_MODEL:
            return {
                "allowed": False,
                "reason": str(walked["reason"]) + f" ({REGISTRY_ADMISSION_RULE})",
                "category": "review_new_canonical_model",
                "placement": placement,
            }
        # Incomplete tree — fall through to other signals / require completion.
        if (
            foundational_market_observation is False
            and canonical_controller_or_market_model is False
            and recreatable_from_registry_or_helpers is None
        ):
            return {
                "allowed": False,
                "reason": str(walked["reason"]),
                "category": None,
                "placement": None,
            }

    produced = str(produced_by or "").strip().lower() or None
    if produced in ("interaction", "interaction_transformation", "interactiontransformation"):
        return {
            "allowed": False,
            "reason": (
                f"{name!r} is produced by InteractionTransformation. "
                "Transformation Pipeline only — never Feature Registry / Master. "
                f"({REGISTRY_ADMISSION_RULE})"
            ),
            "category": "interaction",
            "placement": PLACEMENT_PIPELINE,
        }

    if dataset_builder_configurable is True:
        return {
            "allowed": False,
            "reason": (
                f"{name!r} is configurable by the Dataset Builder. "
                "Transformation Pipeline only — never Feature Registry. "
                f"({REGISTRY_ADMISSION_RULE})"
            ),
            "category": "dataset_builder_configurable",
            "placement": PLACEMENT_PIPELINE,
        }

    if generic_registry_math is True:
        return {
            "allowed": False,
            "reason": (
                f"{name!r} is recreatable from existing registry columns via generic math. "
                "Transformation Pipeline (Interaction) only — never Feature Registry. "
                f"({REGISTRY_ADMISSION_RULE})"
            ),
            "category": "generic_registry_math",
            "placement": PLACEMENT_PIPELINE,
        }

    # Detector: InteractionTransformation outputs (name / migration ledger).
    if is_interaction_feature(name):
        return {
            "allowed": False,
            "reason": (
                f"{name!r} matches InteractionTransformation evidence "
                "(auto-name / migration generator=interaction). "
                "Transformation Pipeline only — never Feature Registry / Master. "
                f"({REGISTRY_ADMISSION_RULE})"
            ),
            "category": "interaction",
            "placement": PLACEMENT_PIPELINE,
        }

    try:
        from .feature_migration import is_pipeline_owned, is_retired

        if is_retired(name):
            return {
                "allowed": False,
                "reason": (
                    f"{name!r} is retired (incompatible with row-based pipeline semantics). "
                    "Do not re-add to the Feature Registry."
                ),
                "category": "retired",
                "placement": None,
            }
        if is_pipeline_owned(name):
            return {
                "allowed": False,
                "reason": (
                    f"{name!r} is Pipeline Owned. Regenerate via the Transformation Pipeline; "
                    "do not re-add to the Feature Registry."
                ),
                "category": OWNERSHIP_PIPELINE_OWNED,
                "placement": PLACEMENT_PIPELINE,
            }
    except Exception:
        pass

    explicit = str(ownership or "").strip().lower() or None
    if requires_prior_rows is True or explicit in (
        OWNERSHIP_HISTORICAL,
        "derived",
        "historical_derived",
        "pipeline",
        "pipeline_owned",
    ):
        if allow_historical and str(historical_exception_reason or "").strip():
            return {
                "allowed": True,
                "reason": (
                    "Historical exception granted: "
                    + str(historical_exception_reason).strip()
                ),
                "category": OWNERSHIP_HISTORICAL,
                "placement": PLACEMENT_REVIEW_NEW_MODEL,
            }
        return {
            "allowed": False,
            "reason": (
                f"{name!r} is historical/derived. Per the placement tree it belongs in the "
                "Transformation Pipeline (recreatable via Lag / Diff / Return / Rolling / "
                "Interaction / …). Registry only after a new canonical model review."
            ),
            "category": OWNERSHIP_HISTORICAL,
            "placement": PLACEMENT_PIPELINE,
        }

    if looks_historical_by_name(name) and explicit not in (
        OWNERSHIP_BASE,
        OWNERSHIP_COMPUTED_BASE,
        "base",
        "computed_base",
        "computed",
    ):
        if allow_historical and str(historical_exception_reason or "").strip():
            return {
                "allowed": True,
                "reason": (
                    "Historical-looking name allowed via exception: "
                    + str(historical_exception_reason).strip()
                ),
                "category": OWNERSHIP_HISTORICAL,
                "placement": PLACEMENT_REVIEW_NEW_MODEL,
            }
        return {
            "allowed": False,
            "reason": (
                f"{name!r} looks historical/derived. Prefer Transformation Pipeline. "
                "To force Registry admission only after confirming Base / Computed Base "
                "via the placement tree (ownership=base|computed_base)."
            ),
            "category": OWNERSHIP_HISTORICAL,
            "placement": PLACEMENT_PIPELINE,
        }

    category = explicit or ownership_of(name)
    if category in (OWNERSHIP_BASE, OWNERSHIP_COMPUTED_BASE, "base", "computed_base", "computed"):
        is_computed = category in (OWNERSHIP_COMPUTED_BASE, "computed_base", "computed")
        return {
            "allowed": True,
            "reason": (
                "Canonical market-state Base / Computed Base — eligible for Feature Registry "
                "(placement tree: foundational market observation or canonical "
                "controller / market model)."
            ),
            "category": OWNERSHIP_COMPUTED_BASE if is_computed else OWNERSHIP_BASE,
            "placement": PLACEMENT_COMPUTED_BASE if is_computed else PLACEMENT_BASE,
        }

    return {
        "allowed": True,
        "reason": (
            "No pipeline / Interaction / tree rejection signal; treating as canonical "
            "Registry candidate. Prefer answering the placement decision tree explicitly."
        ),
        "category": OWNERSHIP_BASE,
        "placement": PLACEMENT_BASE,
    }


def assert_registry_admission(
    feature: str,
    *,
    ownership: str | None = None,
    requires_prior_rows: bool | None = None,
    allow_historical: bool = False,
    historical_exception_reason: str | None = None,
    produced_by: str | None = None,
    dataset_builder_configurable: bool | None = None,
    generic_registry_math: bool | None = None,
    foundational_market_observation: bool | None = None,
    raw_market_observation: bool | None = None,
    canonical_controller_or_market_model: bool | None = None,
    recreatable_from_registry_or_helpers: bool | None = None,
) -> None:
    """Raise ``RegistryAdmissionError`` when a new Registry name is not allowed."""
    result = evaluate_registry_admission(
        feature,
        ownership=ownership,
        requires_prior_rows=requires_prior_rows,
        allow_historical=allow_historical,
        historical_exception_reason=historical_exception_reason,
        produced_by=produced_by,
        dataset_builder_configurable=dataset_builder_configurable,
        generic_registry_math=generic_registry_math,
        foundational_market_observation=foundational_market_observation,
        raw_market_observation=raw_market_observation,
        canonical_controller_or_market_model=canonical_controller_or_market_model,
        recreatable_from_registry_or_helpers=recreatable_from_registry_or_helpers,
    )
    if not result.get("allowed"):
        raise RegistryAdmissionError(str(result.get("reason") or "Registry admission denied."))


def classify_registry_features(
    features: list[str] | None = None,
) -> dict[OwnershipCategory, list[str]]:
    """Partition features into ownership buckets (sorted)."""
    if features is None:
        from .feature_plugins import _REGISTRY_FEATURES

        features = sorted({f for feats in _REGISTRY_FEATURES.values() for f in feats})
    buckets: dict[OwnershipCategory, list[str]] = {
        OWNERSHIP_BASE: [],
        OWNERSHIP_COMPUTED_BASE: [],
        OWNERSHIP_HISTORICAL: [],
        OWNERSHIP_PIPELINE_OWNED: [],
        OWNERSHIP_RETIRED: [],
    }
    for feat in features:
        buckets[ownership_of(str(feat))].append(str(feat))
    for key in buckets:
        buckets[key] = sorted(buckets[key])
    return buckets


def ownership_summary(features: list[str] | None = None) -> dict[str, int]:
    buckets = classify_registry_features(features)
    return {k: len(v) for k, v in buckets.items()}


def canonical_registry_features(features: list[str] | None = None) -> list[str]:
    """Features that should remain in the Feature Registry (target state)."""
    buckets = classify_registry_features(features)
    return sorted(buckets[OWNERSHIP_BASE] + buckets[OWNERSHIP_COMPUTED_BASE])


def controller_of(feature: str) -> str | None:
    """Return Controller Registry owner of a Computed Base feature (or None).

    Thin Feature Registry integration — does not alter schema or ownership
    classification. Unknown / Base / Pipeline features return None when no
    controller emits them.
    """
    from .controller_registry import controller_owner_of_feature

    return controller_owner_of_feature(feature)


__all__ = [
    "OWNERSHIP_BASE",
    "OWNERSHIP_COMPUTED_BASE",
    "OWNERSHIP_HISTORICAL",
    "OWNERSHIP_PIPELINE_OWNED",
    "OWNERSHIP_RETIRED",
    "HISTORICAL_FEATURES",
    "OwnershipCategory",
    "FutureGenerator",
    "REGISTRY_ADMISSION_RULE",
    "CONTROLLER_EMISSION_INVARIANT",
    "REGISTRY_ADMISSION_DECISION_TREE",
    "PLACEMENT_BASE",
    "PLACEMENT_COMPUTED_BASE",
    "PLACEMENT_PIPELINE",
    "PLACEMENT_REVIEW_NEW_MODEL",
    "PLACEMENT_EDGE",
    "EDGE_FEATURES",
    "EDGE_REASON_CONDITIONAL_COLUMN_SELECTION",
    "EDGE_REASON_WEIGHTED_SCALAR_BLEND",
    "edge_features_blocked_by",
    "ownership_of",
    "future_generator_of",
    "is_canonical",
    "non_registry_transform_source_names",
    "litmus_is_historical",
    "looks_historical_by_name",
    "is_interaction_feature",
    "walk_feature_placement_tree",
    "evaluate_registry_admission",
    "assert_registry_admission",
    "RegistryAdmissionError",
    "classify_registry_features",
    "ownership_summary",
    "canonical_registry_features",
    "controller_of",
]
