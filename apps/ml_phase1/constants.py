"""Phase 1 NIFTY spot 60m classifier — locked constants (spec v1.7)."""

from __future__ import annotations

NIFTY_TOKEN = "99926000"

INTERVAL_1M = 60
INTERVAL_5M = 300
INTERVAL_30M = 1800
INTERVAL_60M = 3600
INTERVAL_1D = 86400

SESSION_OPEN_HOUR = 9
SESSION_OPEN_MINUTE = 15
SESSION_CLOSE_HOUR = 15
SESSION_CLOSE_MINUTE = 30
LAST_TRAINABLE_HOUR = 14
LAST_TRAINABLE_MINUTE = 30

LABEL_HORIZON_MIN = 60
LABEL_MAX_GAP_MIN = 5
PURGE_LABEL_HORIZON_MIN = 60
PURGE_FEATURE_HORIZON_MIN = 120
PURGE_MINUTES = max(PURGE_LABEL_HORIZON_MIN, PURGE_FEATURE_HORIZON_MIN)

LABEL_UP_THRESHOLD = 0.3
LABEL_DOWN_THRESHOLD = -0.3

EMA_WARMUP_BARS = 100
EMA_PERIODS = (9, 20, 50, 100)
ATR_PERIOD = 14

HIGH_CONFIDENCE_THRESHOLD = 0.70

CLASS_DOWN = 0
CLASS_SIDEWAYS = 1
CLASS_UP = 2
CLASS_NAMES = {CLASS_DOWN: "DOWN", CLASS_SIDEWAYS: "SIDEWAYS", CLASS_UP: "UP"}
CLASS_NAME_TO_CODE = {v: k for k, v in CLASS_NAMES.items()}

FEATURE_COLUMNS: list[str] = [
    "MinuteOfDay",
    "DayOfWeek",
    "Return1m",
    "Return5m",
    "Return15m",
    "Return30m",
    "Return60m",
    "Return1d",
    "GapFromPrevClose",
    "DistanceFromPrevClose",
    "1m_EMA9MinusEMA20_pct",
    "1m_EMA20MinusEMA50_pct",
    "1m_EMA50MinusEMA100_pct",
    "5m_EMA9MinusEMA20_pct",
    "5m_EMA20MinusEMA50_pct",
    "5m_EMA50MinusEMA100_pct",
    "30m_EMA9MinusEMA20_pct",
    "30m_EMA20MinusEMA50_pct",
    "30m_EMA50MinusEMA100_pct",
    "60m_EMA9MinusEMA20_pct",
    "60m_EMA20MinusEMA50_pct",
    "60m_EMA50MinusEMA100_pct",
    "1m_EMA20Slope5m_pct",
    "1m_EMA50Slope15m_pct",
    "5m_EMA20Slope15m_pct",
    "5m_EMA50Slope30m_pct",
    "30m_EMA20Slope60m_pct",
    "60m_EMA20Slope120m_pct",
    "ATR14_1m_pct",
    "ATR14_5m_pct",
    "ATR14_30m_pct",
    "StdRet5m",
    "StdRet15m",
    "StdRet60m",
    "DistanceFromPrevDayHigh_pct",
    "DistanceFromPrevDayLow_pct",
    "Daily_EMA20MinusEMA50_pct",
    "DistanceFromDailyEMA20_pct",
]

QA_COLUMNS = [
    "year",
    "month",
    "trading_date",
    "label_offset_min",
]

WALK_FORWARD_FOLDS: list[tuple[tuple[int, ...], int]] = [
    ((2015, 2016, 2017, 2018), 2019),
    ((2015, 2016, 2017, 2018, 2019), 2020),
    ((2015, 2016, 2017, 2018, 2019, 2020), 2021),
    ((2015, 2016, 2017, 2018, 2019, 2020, 2021), 2022),
    ((2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022), 2023),
    ((2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023), 2024),
    ((2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024), 2025),
]

FINAL_TRAIN_YEARS = tuple(range(2015, 2026))
FINAL_TEST_YEAR = 2026

FEATURE_COUNT = len(FEATURE_COLUMNS)
