"""Strategy Simulator — Phase 3 replay on prediction rows."""

from __future__ import annotations

from .probability_filter import (
    PROBABILITY_DISABLED,
    apply_probability_filter,
    option_from_label,
    probability_filter_labels,
    probability_filter_options,
    probability_row_summary,
    resolve_member_threshold_defaults,
)
from .registry import get_strategy_run_detail, get_strategy_run_trades, list_strategy_runs
from .scoring import (
    SCORING_VERSION,
    attach_strategy_score,
    count_active_trading_days,
    evaluate_strategy,
    evaluate_strategy_from_run,
    get_strategy_grade,
    sample_reliability_label,
)
from .service import (
    build_tb_comparison_payload,
    run_strategy_simulation,
    run_strategy_simulation_from_lab,
    run_strategy_simulation_from_lab_with_tb_comparison,
)
from .store import StrategyRunStore
from .triple_barrier_filter import (
    MISSING_TB_REASON,
    TB_CLASS_COLUMN,
    TB_DISABLED,
    TB_PROB_COLUMN,
    apply_tb_filter,
    discover_tb_model_name,
    normalize_tb_threshold,
    resolve_tb_class_options,
    tb_filter_options,
    tb_row_summary,
)

__all__ = [
    "MISSING_TB_REASON",
    "PROBABILITY_DISABLED",
    "SCORING_VERSION",
    "TB_CLASS_COLUMN",
    "TB_DISABLED",
    "TB_PROB_COLUMN",
    "StrategyRunStore",
    "apply_probability_filter",
    "apply_tb_filter",
    "attach_strategy_score",
    "build_tb_comparison_payload",
    "count_active_trading_days",
    "discover_tb_model_name",
    "evaluate_strategy",
    "evaluate_strategy_from_run",
    "get_strategy_grade",
    "get_strategy_run_detail",
    "get_strategy_run_trades",
    "list_strategy_runs",
    "normalize_tb_threshold",
    "option_from_label",
    "probability_filter_labels",
    "probability_filter_options",
    "probability_row_summary",
    "resolve_member_threshold_defaults",
    "resolve_tb_class_options",
    "run_strategy_simulation",
    "run_strategy_simulation_from_lab",
    "run_strategy_simulation_from_lab_with_tb_comparison",
    "sample_reliability_label",
    "tb_filter_options",
    "tb_row_summary",
]
