"""Canonical enums and constants for Model Taxonomy (Phase 4C.1).

This module defines the four orthogonal dimensions of model classification:
1. Task Type (Mathematical objective formulation)
2. Market Regime (Environmental market context)
3. Model Population Tier (Governance and maturity standing)
4. Model Lifecycle Status (Operational readiness)
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class TaskType(str, Enum):
    """Mathematical formulation of the model's objective function and target space."""

    DIRECTION_CLASSIFIER = "DIRECTION_CLASSIFIER"
    REGIME_CLASSIFIER = "REGIME_CLASSIFIER"
    REGRESSION = "REGRESSION"
    TRIPLE_BARRIER = "TRIPLE_BARRIER"
    CONFIDENCE_CLASSIFIER = "CONFIDENCE_CLASSIFIER"
    VOLATILITY_ESTIMATOR = "VOLATILITY_ESTIMATOR"

    def is_classification(self) -> bool:
        return self in {
            TaskType.DIRECTION_CLASSIFIER,
            TaskType.REGIME_CLASSIFIER,
            TaskType.TRIPLE_BARRIER,
            TaskType.CONFIDENCE_CLASSIFIER,
        }

    def is_regression(self) -> bool:
        return self in {
            TaskType.REGRESSION,
            TaskType.VOLATILITY_ESTIMATOR,
        }

    @classmethod
    def from_str(cls, value: str | Any) -> TaskType:
        raw = str(value or "").strip().upper()
        # Direct match
        try:
            return cls(raw)
        except ValueError:
            pass

        # Normalized mapping for aliases / legacy strings
        mapping = {
            "DIRECTION": cls.DIRECTION_CLASSIFIER,
            "DIRECTION_CLS": cls.DIRECTION_CLASSIFIER,
            "DIRECTIONAL": cls.DIRECTION_CLASSIFIER,
            "BINARY": cls.DIRECTION_CLASSIFIER,
            "CLASSIFICATION": cls.DIRECTION_CLASSIFIER,
            "REGIME": cls.REGIME_CLASSIFIER,
            "REGIME_CLS": cls.REGIME_CLASSIFIER,
            "REG": cls.REGRESSION,
            "REGRESSION": cls.REGRESSION,
            "TB": cls.TRIPLE_BARRIER,
            "TRIPLE_BARRIER": cls.TRIPLE_BARRIER,
            "CONFIDENCE": cls.CONFIDENCE_CLASSIFIER,
            "CONF": cls.CONFIDENCE_CLASSIFIER,
            "CONFIDENCE_CLS": cls.CONFIDENCE_CLASSIFIER,
            "VOLATILITY": cls.VOLATILITY_ESTIMATOR,
            "VOL_ESTIMATOR": cls.VOLATILITY_ESTIMATOR,
        }
        if raw in mapping:
            return mapping[raw]
        raise ValueError(
            f"Invalid TaskType '{value}'. Must be one of {[e.value for e in cls]}."
        )


class ModelPopulationTier(str, Enum):
    """Maturity and governance tier of a model package."""

    EXPERIMENTAL = "EXPERIMENTAL"
    VALIDATED = "VALIDATED"
    CHALLENGER = "CHALLENGER"
    CHAMPION = "CHAMPION"

    @classmethod
    def from_str(cls, value: str | Any) -> ModelPopulationTier:
        raw = str(value or "").strip().upper()
        try:
            return cls(raw)
        except ValueError:
            pass
        mapping = {
            "EXP": cls.EXPERIMENTAL,
            "VAL": cls.VALIDATED,
            "CHALLENGE": cls.CHALLENGER,
            "CHAMP": cls.CHAMPION,
            "PROD": cls.CHAMPION,
            "PRODUCTION": cls.CHAMPION,
        }
        if raw in mapping:
            return mapping[raw]
        return cls.EXPERIMENTAL


class ModelLifecycleStatus(str, Enum):
    """Operational usability and readiness status of a model package."""

    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"

    @classmethod
    def from_str(cls, value: str | Any) -> ModelLifecycleStatus:
        raw = str(value or "").strip().upper()
        try:
            return cls(raw)
        except ValueError:
            pass
        mapping = {
            "READY": cls.ACTIVE,
            "PROD": cls.ACTIVE,
            "LIVE": cls.ACTIVE,
            "ARCHIVED": cls.RETIRED,
            "HISTORICAL": cls.DEPRECATED,
        }
        if raw in mapping:
            return mapping[raw]
        return cls.ACTIVE


class RegimeScope(str, Enum):
    """Scope of market regime applicability."""

    ALL_REGIMES = "ALL_REGIMES"
    SPECIALIZED = "SPECIALIZED"
    DISCOVERED = "DISCOVERED"


DEFAULT_REGIME_ID: str = "R000"
DEFAULT_REGIME_NAME: str = "ALL_REGIMES"

BASELINE_REGIME_CATALOG: dict[str, dict[str, str]] = {
    "R000": {
        "name": "ALL_REGIMES",
        "description": "Universal baseline evaluated across all market conditions.",
        "family": "UNIVERSAL",
    },
    "R001": {
        "name": "TREND",
        "description": "Strong directional momentum with sustained order flow imbalance.",
        "family": "DIRECTIONAL_MOMENTUM",
    },
    "R002": {
        "name": "SIDEWAYS",
        "description": "Range-bound, mean-reverting oscillator with compressed ADX.",
        "family": "MEAN_REVERSION",
    },
    "R003": {
        "name": "HIGH_VOLATILITY",
        "description": "Realized IV > 85th percentile, wide spreads, and elevated strike variance.",
        "family": "VOLATILITY_EXPANSION",
    },
    "R004": {
        "name": "LOW_VOLATILITY",
        "description": "Realized IV < 25th percentile with compressed straddle premiums.",
        "family": "VOLATILITY_COMPRESSION",
    },
    "R005": {
        "name": "BREAKOUT",
        "description": "Volatility compression breakout with volume and order velocity spike.",
        "family": "LIQUIDITY_EXPANSION",
    },
    "R006": {
        "name": "REVERSAL",
        "description": "Exhaustion divergence at key structural liquidity support/resistance.",
        "family": "MICROSTRUCTURE_REVERSAL",
    },
    "R007": {
        "name": "EXPIRY_PINNING",
        "description": "Gamma pinning and decay compression near max-pain strike on expiry.",
        "family": "OPTION_GAMMA",
    },
}
