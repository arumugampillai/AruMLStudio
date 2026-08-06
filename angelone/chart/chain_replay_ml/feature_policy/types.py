"""Feature Policy — core types (no UI dependencies)."""

from __future__ import annotations

from enum import Enum


FEATURE_POLICY_VERSION = "1.0"
DEFAULT_GAP_MAX_SEC = 20.0


class FeatureCategory(str, Enum):
    RAW = "raw"
    ROLLING = "rolling"
    LOOKBACK = "lookback"
    CUMULATIVE = "cumulative"
    DERIVED = "derived"
    TARGET = "target"
    METADATA = "metadata"


class FeatureLifecycle(str, Enum):
    """When feature state resets or how history is scoped."""

    TICK = "tick"
    SESSION = "session"
    SLIDING_WINDOW = "sliding_window"
    DAY = "day"


class RollingType(str, Enum):
    EMA = "ema"
    SMA = "sma"
    STD = "std"
    ATR = "atr"
    RSI = "rsi"
    BOLLINGER = "bollinger"
    RV = "rv"
    ZSCORE = "zscore"
    OTHER = "other"


class WarmupMode(str, Enum):
    SAMPLE_COUNT = "sample_count"
    TIME_SEC = "time_sec"
