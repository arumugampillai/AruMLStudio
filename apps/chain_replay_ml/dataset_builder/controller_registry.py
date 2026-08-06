"""Canonical controller registry — single source of controller IDs, warmup, and dependencies.

Must stay aligned with docs/controllers/CONTROLLER_OWNERSHIP.md. The classification generator validates
against this registry (no orphan controllers, no unknown IDs).

Bump FEATURE_REGISTRY_VERSION when CONTROLLER_REGISTRY or CONTROLLER_FEATURES changes.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bump when controller IDs, phases, warmup, or feature mappings change.
FEATURE_REGISTRY_VERSION = 30


@dataclass(frozen=True)
class ControllerSpec:
    controller_id: str
    phase: int
    warmup_type: str  # Sample | Calendar | Session | Immediate
    warmup_value: str  # e.g. "9", "5m", "Session", "0"
    sample_fields: tuple[str, ...] = ()
    source_controllers: tuple[str, ...] = ()


# Features produced per controller (Wave 2/3: canonical levels; packaging → Interaction).
CONTROLLER_FEATURES: dict[str, list[str]] = {
    "token.ltp.ema9": ["ltp_ema9"],
    "token.ltp.ema20": ["ltp_ema20"],
    "token.ltp.ema50": ["ltp_ema50"],
    "token.ltp.ema100": ["ltp_ema100"],
    "token.ltp.ema200": ["ltp_ema200"],
    "token.ltp.ema300": ["ltp_ema300"],
    "composite.weighted_ltp_ema": ["weighted_ltp_ema"],
    "token.ltp.std20": ["ltp_std20"],
    "token.rv.5m": ["opt_rv_5m"],
    "token.rv.10m": ["opt_rv_10m"],
    "token.rv.ratio": [],  # opt_rv_ratio → Interaction
    "token.iv.ema9": ["iv_ema9"],
    "token.iv.ema20": ["iv_ema20"],
    "token.iv.ema50": ["iv_ema50"],
    "token.iv.ema100": ["iv_ema100"],
    "token.iv.ema200": ["iv_ema200"],
    "token.iv.ema300": ["iv_ema300"],
    "token.iv_window.1m": ["iv_zscore_1m"],
    "token.iv_window.5m": ["iv_zscore_5m"],
    "token.iv_window.15m": ["iv_zscore_15m"],
    "token.iv_window.30m": ["iv_zscore_30m"],
    "token.iv_window.session": ["iv_rank_session"],
    # iv_change_* still emitted by Master (registry → Pipeline Owned)
    "token.iv_history.1m": ["iv_change_1m", "iv_pct_change_1m"],
    "token.iv_history.5m": ["iv_change_5m"],
    "token.iv_history.15m": ["iv_change_15m"],
    "token.roll": [
        "roll_iv", "roll_age_min", "rows_since_roll",
        "bs_reiv_pred", "dgt_reiv_pred", "iv_drift_from_roll"],
    "token.dgt": [
        "dgt_prediction_error"],  # dgt_reiv_to_ltp_ratio / dgt_to_spot_ratio → Interaction
    # Wave A: Market Microstructure Controller (Immediate). bid_ask_spread stays RAW Base.
    "token.book": [
        "mid_price",
        "microprice",
        "microprice_bias",
        "book_imbalance_l1",
        "book_imbalance_l1_5",
        "bid_depth_l1_5",
        "ask_depth_l1_5",
        "book_depth_slope_bid",
        "book_depth_slope_ask",
    ],
    # Wave B: Chain Controller — surface / chain-state levels (skew first).
    "token.chain": [
        "iv_skew_atm",
        "iv_call_put_skew",
        "iv_skew_25d",
        "iv_butterfly_25d",
        "atm_iv_ce",
        "atm_iv_pe",
        "total_call_oi",
        "total_put_oi",
        "total_ce_volume",
        "total_pe_volume",
        "otm_ce_volume",
        "otm_pe_volume",
        "otm_pcr_volume",
        "delta_w_volume_flow_1m",
        "delta_w_volume_flow_5m",
        "call_gex",
        "put_gex",
        "net_gex",
        "chain_gex",
        "gamma_flip_spot",
        "gamma_flip_distance",
        "synthetic_forward_spot",
        "oi_abs_delta_0_20_ce",
        "oi_abs_delta_0_20_pe",
        "oi_abs_delta_20_40_ce",
        "oi_abs_delta_20_40_pe",
        "oi_abs_delta_40_60_ce",
        "oi_abs_delta_40_60_pe",
        "oi_abs_delta_60_80_ce",
        "oi_abs_delta_60_80_pe",
        "oi_abs_delta_80_100_ce",
        "oi_abs_delta_80_100_pe",
    ],
    "spot.ema9": ["spot_ema9"],
    "spot.ema20": ["spot_ema20"],
    "spot.ema50": ["spot_ema50"],
    "spot.ema100": ["spot_ema100"],
    "spot.ema200": ["spot_ema200"],
    "spot.ema300": ["spot_ema300"],
    "spot.hl.ema20": ["spot_high_ema20", "spot_low_ema20", "spot_ema20_channel_width"],
    "spot.hl.ema50": ["spot_high_ema50", "spot_low_ema50", "spot_ema50_channel_width"],
    "spot.hl.ema100": ["spot_high_ema100", "spot_low_ema100", "spot_ema100_channel_width"],
    "spot.hl.ema200": ["spot_high_ema200", "spot_low_ema200", "spot_ema200_channel_width"],
    "spot.hl.ema300": ["spot_high_ema300", "spot_low_ema300", "spot_ema300_channel_width"],
    "spot.rv.5m": ["spot_rv_5m"],
    "spot.rv.10m": ["spot_rv_10m"],
    "spot.rv.ratio": [],  # spot_rv_ratio → Interaction
    # Wave B: IV − spot RV levels (packaging of existing IV + RV).
    "composite.iv_rv_spread": ["iv_rv_spread_5m", "iv_rv_spread_10m"],
    "spot.momentum": [
        "ema9_gt_ema20", "ema9_slope", "time_since_cross_min",
        "price_dist_from_cross_pct", "cross_age_decay"],
    # Wave 6: ema_spread_pct / ema_spread_vs_spot_pct / spot_vs_ema20_pct → Interaction
    "composite.weighted_spot_ema": ["weighted_spot_ema"],
    "composite.weighted_spot_hl": [
        "weighted_spot_high_ema", "weighted_spot_low_ema", "weighted_spot_close_ema"],
    # Interaction-only controller — features moved to InteractionTransformation pipeline.
    "composite.iv_x_spot_ema": [],
}

CONTROLLER_REGISTRY: dict[str, ControllerSpec] = {
    cid: ControllerSpec(cid, phase, wtype, wval, fields, deps)
    for cid, phase, wtype, wval, fields, deps in [
        ("token.ltp.ema9", 1, "Sample", "9", ("ltp",), ()),
        ("token.ltp.ema20", 1, "Sample", "20", ("ltp",), ()),
        ("token.ltp.ema50", 1, "Sample", "50", ("ltp",), ()),
        ("token.ltp.ema100", 1, "Sample", "100", ("ltp",), ()),
        ("token.ltp.ema200", 1, "Sample", "200", ("ltp",), ()),
        ("token.ltp.ema300", 1, "Sample", "300", ("ltp",), ()),
        ("composite.weighted_ltp_ema", 1, "Sample", "200", ("ltp",),
         ("token.ltp.ema9", "token.ltp.ema20", "token.ltp.ema50", "token.ltp.ema200")),
        ("token.ltp.std20", 2, "Sample", "20", ("ltp",), ()),
        ("token.rv.5m", 2, "Sample", "30", ("log_return",), ()),
        ("token.rv.10m", 2, "Sample", "60", ("log_return",), ()),
        ("token.rv.ratio", 2, "Immediate", "0", (), ("token.rv.5m", "token.rv.10m")),
        ("token.iv.ema9", 2, "Sample", "9", ("iv",), ()),
        ("token.iv.ema20", 2, "Sample", "20", ("iv",), ()),
        ("token.iv.ema50", 2, "Sample", "50", ("iv",), ()),
        ("token.iv.ema100", 2, "Sample", "100", ("iv",), ()),
        ("token.iv.ema200", 2, "Sample", "200", ("iv",), ()),
        ("token.iv.ema300", 2, "Sample", "300", ("iv",), ()),
        ("token.iv_window.1m", 2, "Calendar", "1m", ("iv",), ()),
        ("token.iv_window.5m", 2, "Calendar", "5m", ("iv",), ()),
        ("token.iv_window.15m", 2, "Calendar", "15m", ("iv",), ()),
        ("token.iv_window.30m", 2, "Calendar", "30m", ("iv",), ()),
        ("token.iv_window.session", 2, "Session", "Session", ("iv",), ()),
        ("token.iv_history.1m", 2, "Calendar", "1m", ("iv",), ()),
        ("token.iv_history.5m", 2, "Calendar", "5m", ("iv",), ()),
        ("token.iv_history.15m", 2, "Calendar", "15m", ("iv",), ()),
        ("token.roll", 2, "Immediate", "0", ("iv",), ()),
        ("token.dgt", 2, "Calendar", "lag horizon", (), ("token.roll",)),
        ("token.book", 1, "Immediate", "0",
         ("bid", "ask", "bid_qty", "ask_qty"), ()),
        ("token.chain", 2, "Immediate", "0",
         ("spot", "ltp", "oi"), ()),
        ("spot.ema9", 3, "Sample", "9", ("spot",), ()),
        ("spot.ema20", 3, "Sample", "20", ("spot",), ()),
        ("spot.ema50", 3, "Sample", "50", ("spot",), ()),
        ("spot.ema100", 3, "Sample", "100", ("spot",), ()),
        ("spot.ema200", 3, "Sample", "200", ("spot",), ()),
        ("spot.ema300", 3, "Sample", "300", ("spot",), ()),
        ("spot.hl.ema20", 3, "Sample", "20", ("spot_high", "spot_low"), ()),
        ("spot.hl.ema50", 3, "Sample", "50", ("spot_high", "spot_low"), ()),
        ("spot.hl.ema100", 3, "Sample", "100", ("spot_high", "spot_low"), ()),
        ("spot.hl.ema200", 3, "Sample", "200", ("spot_high", "spot_low"), ()),
        ("spot.hl.ema300", 3, "Sample", "300", ("spot_high", "spot_low"), ()),
        ("spot.rv.5m", 3, "Sample", "30", ("spot_log_return",), ()),
        ("spot.rv.10m", 3, "Sample", "60", ("spot_log_return",), ()),
        ("spot.rv.ratio", 3, "Immediate", "0", (), ("spot.rv.5m", "spot.rv.10m")),
        ("composite.iv_rv_spread", 3, "Immediate", "0", ("iv",),
         ("spot.rv.5m", "spot.rv.10m")),
        ("spot.momentum", 3, "Sample", "20", ("spot",), ("spot.ema9", "spot.ema20")),
        ("composite.weighted_spot_ema", 3, "Sample", "200", ("ltp",),
         ("spot.ema9", "spot.ema20", "spot.ema50", "spot.ema100", "spot.ema200")),
        ("composite.weighted_spot_hl", 3, "Sample", "300", ("ltp",),
         ("spot.hl.ema20", "spot.hl.ema50", "spot.hl.ema100", "spot.hl.ema200", "spot.hl.ema300")),
        ("composite.iv_x_spot_ema", 3, "Sample", "200", ("ltp", "iv",),
         ("token.iv_window.1m", "token.iv_window.5m", "token.iv_window.15m",
          "spot.ema9", "spot.ema20", "spot.ema50", "spot.ema100", "spot.ema200")),
    ]
}

# Docs / classification may still say "ChainController"; live ID is token.chain.
# Keep the PascalCase name as a future-alias until the generator is updated.
FUTURE_CONTROLLER_IDS = frozenset({"ChainController"})


def all_registry_controller_ids() -> frozenset[str]:
    return frozenset(CONTROLLER_REGISTRY.keys())


def detect_dependency_cycles() -> list[list[str]]:
    """Return cycles in source_controllers graph (empty if acyclic)."""
    graph: dict[str, list[str]] = {
        cid: list(spec.source_controllers) for cid, spec in CONTROLLER_REGISTRY.items()
    }
    cycles: list[list[str]] = []
    visited: set[str] = set()
    stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> None:
        if node in stack:
            if node in path:
                i = path.index(node)
                cycles.append(path[i:] + [node])
            return
        if node in visited:
            return
        visited.add(node)
        stack.add(node)
        path.append(node)
        for dep in graph.get(node, []):
            if dep in graph:
                dfs(dep)
        path.pop()
        stack.remove(node)

    for cid in graph:
        dfs(cid)
    return cycles


def ensure_architecture_registry() -> None:
    """Bootstrap the Controller Registry architecture layer (idempotent).

    Does not alter CONTROLLER_REGISTRY / CONTROLLER_FEATURES or any emission.
    """
    from .controller_bootstrap import ensure_controller_registry

    ensure_controller_registry()


def controller_owner_of_feature(feature: str) -> str | None:
    """Return controller_id that emits ``feature`` via the Controller Registry."""
    from .controller_bootstrap import owner_of_feature

    return owner_of_feature(feature)
