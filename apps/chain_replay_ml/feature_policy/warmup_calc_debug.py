"""Warm-up simulator — feature calculation debugger (formula + operand values)."""

from __future__ import annotations

import math
import re
from typing import Any

from chain_replay_ml.dataset_builder.schema_column_docs import RICH_COLUMN_DOCS

CHANNEL_EPS = 1e-6


def _fmt_num(val: Any, *, digits: int = 5) -> str | None:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if math.isnan(f) or math.isinf(f):
        return None
    if abs(f) >= 1000:
        return f"{f:.2f}"
    if abs(f) >= 1:
        return f"{f:.4f}"
    return f"{f:.{digits}f}"


def _null_display(val: Any) -> str:
    if val is None:
        return "NULL"
    return str(val)


def formula_doc_for(feature_name: str) -> str:
    doc = RICH_COLUMN_DOCS.get(feature_name) or {}
    return str(doc.get("formula_doc") or doc.get("formula_ref") or feature_name)


def normalize_epoch_ts(raw: Any) -> int | None:
    """Convert replay / trace timestamps to epoch-second lookup keys."""
    if raw is None:
        return None
    if hasattr(raw, "timestamp"):
        try:
            return int(round(float(raw.timestamp())))
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    if abs(val) >= 1e15:
        val /= 1e9
    elif abs(val) >= 1e12:
        val /= 1e3
    return int(round(val))


def replay_columns_for(feature_name: str) -> list[str]:
    """Replay frame columns needed to debug *feature_name*."""
    cols = ["ltp", "spot", feature_name]
    if feature_name == "weighted_ltp_ema_to_ltp_ratio":
        return [
            "ltp",
            "spot",
            feature_name,
            "ltp_ema9",
            "ltp_ema20",
            "ltp_ema50",
            "ltp_ema200",
        ]
    if feature_name == "weighted_spot_ema_to_ltp_ratio":
        return [
            "ltp",
            "spot",
            feature_name,
            "spot_ema9",
            "spot_ema20",
            "spot_ema50",
            "spot_ema200",
        ]
    m = re.match(r"ltp_ema(\d+)$", feature_name)
    if m:
        return ["ltp", feature_name]
    m = re.match(r"spot_ema(\d+)$", feature_name)
    if m:
        return ["ltp", "spot", feature_name]
    m = re.match(r"ltp_ema(\d+)_to_spot_ratio$", feature_name)
    if m:
        return ["ltp", "spot", feature_name]
    m = re.match(r"ltp_ema(\d+)_to_ltp_ratio$", feature_name)
    if m:
        return ["ltp", feature_name]
    m = re.match(r"spot_ema(\d+)_to_ltp_ratio$", feature_name)
    if m:
        return ["ltp", "spot", feature_name]
    m = re.match(r"spot_high_ema(\d+)$", feature_name)
    if m:
        return ["ltp", "spot", feature_name]
    m = re.match(r"spot_low_ema(\d+)$", feature_name)
    if m:
        return ["ltp", "spot", feature_name]
    m = re.match(r"spot_high_ema(\d+)_to_ltp_ratio$", feature_name)
    if m:
        return ["ltp", feature_name]
    m = re.match(r"spot_low_ema(\d+)_to_ltp_ratio$", feature_name)
    if m:
        return ["ltp", feature_name]
    m = re.match(r"ltp_to_spot_ema(\d+)_channel_width_ratio$", feature_name)
    if m:
        p = m.group(1)
        return [
            "ltp",
            feature_name,
            f"spot_high_ema{p}",
            f"spot_low_ema{p}",
        ]
    return list(dict.fromkeys(cols))


def _ema_period_from_name(feature_name: str) -> int | None:
    m = re.search(r"ema(\d+)", feature_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _controller_label(period: int | None) -> str | None:
    return f"EMA{period}" if period else None


def build_formula_spec(feature_name: str) -> dict[str, Any]:
    period = _ema_period_from_name(feature_name)
    ctrl = _controller_label(period)
    formula_doc = formula_doc_for(feature_name)
    kind = "generic"
    tree: list[dict[str, str]] = [{"label": feature_name, "role": "feature"}]

    if re.match(r"ltp_ema\d+$", feature_name):
        kind = "ltp_ema_level"
        p = period or "?"
        tree = [
            {"label": feature_name, "role": "result"},
            {"label": f"EMA{p}(LTP)", "role": "value"},
        ]
    elif re.match(r"spot_ema\d+$", feature_name):
        kind = "spot_ema_level"
        p = period or "?"
        tree = [
            {"label": feature_name, "role": "result"},
            {"label": f"EMA{p}(SPOT)", "role": "value"},
        ]
    elif re.match(r"ltp_ema\d+_to_spot_ratio$", feature_name):
        kind = "ltp_ema_to_spot"
        p = period or "?"
        tree = [
            {"label": feature_name, "role": "result"},
            {"label": "÷", "role": "op"},
            {"label": f"EMA{p}(LTP)", "role": "numerator"},
            {"label": "SPOT", "role": "denominator"},
        ]
    elif re.match(r"ltp_ema\d+_to_ltp_ratio$", feature_name):
        kind = "ltp_ema_to_ltp"
        p = period or "?"
        tree = [
            {"label": feature_name, "role": "result"},
            {"label": "÷", "role": "op"},
            {"label": f"EMA{p}(LTP)", "role": "numerator"},
            {"label": "LTP", "role": "denominator"},
        ]
    elif re.match(r"spot_ema\d+_to_ltp_ratio$", feature_name):
        kind = "spot_ema_to_ltp"
        p = period or "?"
        tree = [
            {"label": feature_name, "role": "result"},
            {"label": "÷", "role": "op"},
            {"label": f"EMA{p}(SPOT)", "role": "numerator"},
            {"label": "LTP", "role": "denominator"},
        ]
    elif re.match(r"ltp_to_spot_ema\d+_channel_width_ratio$", feature_name):
        kind = "channel_width"
        p = period or "?"
        tree = [
            {"label": feature_name, "role": "result"},
            {"label": "÷", "role": "op"},
            {"label": "LTP", "role": "numerator"},
            {"label": "Channel Width", "role": "denominator"},
            {"label": "−", "role": "op"},
            {"label": f"High EMA{p}(SPOT)", "role": "operand"},
            {"label": f"Low EMA{p}(SPOT)", "role": "operand"},
        ]
    elif feature_name == "weighted_ltp_ema_to_ltp_ratio":
        kind = "weighted_ltp_ema"
        tree = [
            {"label": feature_name, "role": "result"},
            {"label": "÷", "role": "op"},
            {"label": "Weighted EMA blend", "role": "numerator"},
            {"label": "LTP", "role": "denominator"},
        ]
    elif feature_name == "weighted_spot_ema_to_ltp_ratio":
        kind = "weighted_spot_ema"
        tree = [
            {"label": feature_name, "role": "result"},
            {"label": "÷", "role": "op"},
            {"label": "Weighted Spot EMA blend", "role": "numerator"},
            {"label": "LTP", "role": "denominator"},
        ]

    return {
        "feature_name": feature_name,
        "formula_doc": formula_doc,
        "kind": kind,
        "controller_label": ctrl,
        "tree": tree,
        "table_columns": _table_columns_for_kind(kind, period),
    }


def _table_columns_for_kind(kind: str, period: int | None) -> list[str]:
    p = period or 0
    if kind in ("ltp_ema_level", "ltp_ema_to_spot", "ltp_ema_to_ltp"):
        if kind == "ltp_ema_level":
            return ["time", "sample", "ltp", f"ema{p}", "feature_value"]
        if kind == "ltp_ema_to_spot":
            return ["time", "sample", "ltp", "spot", f"ema{p}", "feature_value"]
        return ["time", "sample", "ltp", f"ema{p}", "feature_value"]
    if kind in ("spot_ema_level", "spot_ema_to_ltp"):
        return ["time", "sample", "ltp", f"spot_ema{p}", "feature_value"]
    if kind == "channel_width":
        return [
            "time", "sample", "ltp",
            f"high_ema{p}", f"low_ema{p}", "channel_width", "feature_value",
        ]
    if kind == "weighted_ltp_ema":
        return [
            "time", "sample", "ltp", "spot",
            "ema9_r", "ema20_r", "ema50_r", "ema200_r", "feature_value",
        ]
    if kind == "weighted_spot_ema":
        return [
            "time", "sample", "ltp", "spot",
            "spot_ema9_r", "spot_ema20_r", "spot_ema50_r", "spot_ema200_r", "feature_value",
        ]
    return ["time", "sample", "ltp", "spot", "feature_value"]


def _display_headers(columns: list[str], period: int | None) -> list[str]:
    p = period or 0
    labels = {
        "time": "Time",
        "sample": "Sample",
        "ltp": "LTP",
        "spot": "SPOT",
        "feature_value": "Feature Value",
        f"ema{p}": f"EMA{p}",
        f"spot_ema{p}": f"Spot EMA{p}",
        f"high_ema{p}": f"High EMA{p}",
        f"low_ema{p}": f"Low EMA{p}",
        "channel_width": "Channel Width",
        "ema9_r": "EMA9/LTP",
        "ema20_r": "EMA20/LTP",
        "ema50_r": "EMA50/LTP",
        "ema200_r": "EMA200/LTP",
        "spot_ema9_r": "Spot EMA9/LTP",
        "spot_ema20_r": "Spot EMA20/LTP",
        "spot_ema50_r": "Spot EMA50/LTP",
        "spot_ema200_r": "Spot EMA200/LTP",
    }
    return [labels.get(c, c.replace("_", " ").title()) for c in columns]


def _operand_values(
    feature_name: str,
    *,
    kind: str,
    period: int | None,
    replay_vals: dict[str, Any],
    feature_ready: bool,
) -> dict[str, Any]:
    ltp = replay_vals.get("ltp")
    spot = replay_vals.get("spot")
    feat = replay_vals.get(feature_name)
    out: dict[str, Any] = {
        "ltp": ltp,
        "spot": spot,
        "feature_value": feat if feature_ready else None,
    }
    p = period or 0

    if not feature_ready:
        if kind in ("ltp_ema_to_spot", "ltp_ema_to_ltp", "ltp_ema_level"):
            out[f"ema{p}"] = None
        if kind in ("spot_ema_to_ltp", "spot_ema_level"):
            out[f"spot_ema{p}"] = None
        if kind == "channel_width":
            out[f"high_ema{p}"] = None
            out[f"low_ema{p}"] = None
            out["channel_width"] = None
        if kind == "weighted_ltp_ema":
            out.update({
                "ema9_r": None, "ema20_r": None, "ema50_r": None, "ema200_r": None,
            })
        if kind == "weighted_spot_ema":
            out.update({
                "spot_ema9_r": None, "spot_ema20_r": None,
                "spot_ema50_r": None, "spot_ema200_r": None,
            })
        return out

    try:
        ltp_f = float(ltp) if ltp is not None else None
        spot_f = float(spot) if spot is not None else None
        feat_f = float(feat) if feat is not None else None
    except (TypeError, ValueError):
        return out

    if kind == "ltp_ema_level" and feat_f is not None:
        out[f"ema{p}"] = feat_f
    elif kind == "spot_ema_level" and feat_f is not None:
        out[f"spot_ema{p}"] = feat_f
    elif kind == "ltp_ema_to_spot" and feat_f is not None and spot_f:
        out[f"ema{p}"] = feat_f * spot_f
    elif kind == "ltp_ema_to_ltp" and feat_f is not None and ltp_f:
        out[f"ema{p}"] = feat_f * ltp_f
    elif kind == "spot_ema_to_ltp" and feat_f is not None and ltp_f:
        out[f"spot_ema{p}"] = feat_f * ltp_f
    elif kind == "channel_width" and ltp_f:
        hi = replay_vals.get(f"spot_high_ema{p}")
        lo = replay_vals.get(f"spot_low_ema{p}")
        if hi is None:
            hi_r = replay_vals.get(f"spot_high_ema{p}_to_ltp_ratio")
            try:
                hi = float(hi_r) * ltp_f if hi_r is not None else None
            except (TypeError, ValueError):
                hi = None
        if lo is None:
            lo_r = replay_vals.get(f"spot_low_ema{p}_to_ltp_ratio")
            try:
                lo = float(lo_r) * ltp_f if lo_r is not None else None
            except (TypeError, ValueError):
                lo = None
        try:
            hi_f = float(hi) if hi is not None else None
            lo_f = float(lo) if lo is not None else None
        except (TypeError, ValueError):
            hi_f = lo_f = None
        if hi_f is not None and lo_f is not None:
            out[f"high_ema{p}"] = hi_f
            out[f"low_ema{p}"] = lo_f
            out["channel_width"] = hi_f - lo_f
        else:
            out[f"high_ema{p}"] = None
            out[f"low_ema{p}"] = None
            out["channel_width"] = None
    elif kind == "weighted_ltp_ema":
        for key, src_level, src_ratio in (
            ("ema9_r", "ltp_ema9", "ltp_ema9_to_ltp_ratio"),
            ("ema20_r", "ltp_ema20", "ltp_ema20_to_ltp_ratio"),
            ("ema50_r", "ltp_ema50", "ltp_ema50_to_ltp_ratio"),
            ("ema200_r", "ltp_ema200", "ltp_ema200_to_ltp_ratio"),
        ):
            raw = replay_vals.get(src_ratio)
            if raw is None and ltp_f:
                level = replay_vals.get(src_level)
                try:
                    raw = float(level) / ltp_f if level is not None else None
                except (TypeError, ValueError):
                    raw = None
            try:
                out[key] = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                out[key] = None
    elif kind == "weighted_spot_ema":
        for key, src_level, src_ratio in (
            ("spot_ema9_r", "spot_ema9", "spot_ema9_to_ltp_ratio"),
            ("spot_ema20_r", "spot_ema20", "spot_ema20_to_ltp_ratio"),
            ("spot_ema50_r", "spot_ema50", "spot_ema50_to_ltp_ratio"),
            ("spot_ema200_r", "spot_ema200", "spot_ema200_to_ltp_ratio"),
        ):
            raw = replay_vals.get(src_ratio)
            if raw is None and ltp_f:
                level = replay_vals.get(src_level)
                try:
                    raw = float(level) / ltp_f if level is not None else None
                except (TypeError, ValueError):
                    raw = None
            try:
                out[key] = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                out[key] = None

    return out

def build_row_breakdown(
    feature_name: str,
    *,
    sample: int,
    time: str,
    operands: dict[str, Any],
    formula_spec: dict[str, Any],
) -> dict[str, Any]:
    kind = formula_spec.get("kind") or "generic"
    period = _ema_period_from_name(feature_name)
    p = period or 0
    formula_doc = formula_spec.get("formula_doc") or feature_name
    feat = operands.get("feature_value")
    lines: list[str] = [f"Sample {sample}", f"Time {time}", ""]
    deps_lines: list[str] = []

    ltp = operands.get("ltp")
    spot = operands.get("spot")
    if ltp is not None:
        deps_lines.append(f"LTP = {_fmt_num(ltp)}")
    if spot is not None:
        deps_lines.append(f"SPOT = {_fmt_num(spot)}")

    steps: list[dict[str, str]] = []
    tree_lines: list[str] = [feature_name, "↓"]

    if kind == "ltp_ema_level":
        ema = operands.get(f"ema{p}")
        if ema is not None:
            deps_lines.append(f"EMA{p} = {_fmt_num(ema)}")
        tree_lines.extend([f"EMA{p}(LTP)", "↓", formula_doc])
        if ema is not None:
            steps = [
                {"label": "Formula", "text": f"EMA{p}(LTP)"},
                {"label": "=", "text": _fmt_num(ema) or "—"},
            ]
    elif kind == "spot_ema_level":
        ema = operands.get(f"spot_ema{p}")
        if ema is not None:
            deps_lines.append(f"Spot EMA{p} = {_fmt_num(ema)}")
        tree_lines.extend([f"EMA{p}(SPOT)", "↓", formula_doc])
        if ema is not None:
            steps = [
                {"label": "Formula", "text": f"EMA{p}(SPOT)"},
                {"label": "=", "text": _fmt_num(ema) or "—"},
            ]
    elif kind == "ltp_ema_to_spot":
        ema = operands.get(f"ema{p}")
        if ema is not None:
            deps_lines.append(f"EMA{p} = {_fmt_num(ema)}")
        tree_lines.extend([f"EMA{p}(LTP) / SPOT", "↓", formula_doc])
        if ema is not None and spot is not None and float(spot) != 0:
            calc = float(ema) / float(spot)
            steps = [
                {"label": "Formula", "text": f"EMA{p}(LTP) / SPOT"},
                {"label": "Substitute", "text": f"{_fmt_num(ema)} / {_fmt_num(spot)}"},
                {"label": "=", "text": _fmt_num(calc) or "—"},
            ]
    elif kind == "ltp_ema_to_ltp":
        ema = operands.get(f"ema{p}")
        if ema is not None:
            deps_lines.append(f"EMA{p} = {_fmt_num(ema)}")
        tree_lines.extend([f"EMA{p}(LTP) / LTP", "↓", formula_doc])
        if ema is not None and ltp is not None and float(ltp) != 0:
            calc = float(ema) / float(ltp)
            steps = [
                {"label": "Formula", "text": f"EMA{p}(LTP) / LTP"},
                {"label": "Substitute", "text": f"{_fmt_num(ema)} / {_fmt_num(ltp)}"},
                {"label": "=", "text": _fmt_num(calc) or "—"},
            ]
    elif kind == "spot_ema_to_ltp":
        ema = operands.get(f"spot_ema{p}")
        if ema is not None:
            deps_lines.append(f"Spot EMA{p} = {_fmt_num(ema)}")
        tree_lines.extend([f"EMA{p}(SPOT) / LTP", "↓", formula_doc])
        if ema is not None and ltp is not None and float(ltp) != 0:
            calc = float(ema) / float(ltp)
            steps = [
                {"label": "Formula", "text": f"EMA{p}(SPOT) / LTP"},
                {"label": "Substitute", "text": f"{_fmt_num(ema)} / {_fmt_num(ltp)}"},
                {"label": "=", "text": _fmt_num(calc) or "—"},
            ]
    elif kind == "channel_width":
        high = operands.get(f"high_ema{p}")
        low = operands.get(f"low_ema{p}")
        width = operands.get("channel_width")
        if high is not None:
            deps_lines.append(f"High EMA{p} = {_fmt_num(high)}")
        if low is not None:
            deps_lines.append(f"Low EMA{p} = {_fmt_num(low)}")
        if width is not None:
            deps_lines.append(f"Channel Width = {_fmt_num(width)}")
        tree_lines.extend([
            "Channel Width",
            "↓",
            f"High EMA{p} − Low EMA{p}",
            "↓",
            f"Width = {_fmt_num(width) or 'NULL'}",
            "↓",
            "LTP / Width",
            "↓",
            _fmt_num(feat) or "NULL",
        ])
        if width is not None and ltp is not None:
            den = abs(float(width)) + CHANNEL_EPS
            calc = float(ltp) / den
            steps = [
                {"label": "Channel Width", "text": f"{_fmt_num(high)} − {_fmt_num(low)} = {_fmt_num(width)}"},
                {"label": "Formula", "text": "LTP / (channel_width + ε)"},
                {"label": "Substitute", "text": f"{_fmt_num(ltp)} / {_fmt_num(den)}"},
                {"label": "=", "text": _fmt_num(calc) or "—"},
            ]
    elif kind == "weighted_ltp_ema":
        for label, key in (
            ("EMA9/LTP", "ema9_r"), ("EMA20/LTP", "ema20_r"),
            ("EMA50/LTP", "ema50_r"), ("EMA200/LTP", "ema200_r"),
        ):
            val = operands.get(key)
            if val is not None:
                deps_lines.append(f"{label} = {_fmt_num(val)}")
        tree_lines.extend(["Weighted EMA blend / LTP", "↓", formula_doc])
    elif kind == "weighted_spot_ema":
        for label, key in (
            ("Spot EMA9/LTP", "spot_ema9_r"), ("Spot EMA20/LTP", "spot_ema20_r"),
            ("Spot EMA50/LTP", "spot_ema50_r"), ("Spot EMA200/LTP", "spot_ema200_r"),
        ):
            val = operands.get(key)
            if val is not None:
                deps_lines.append(f"{label} = {_fmt_num(val)}")
        tree_lines.extend(["Weighted Spot EMA blend / LTP", "↓", formula_doc])
    else:
        tree_lines.append(formula_doc)
        if feat is not None:
            steps = [{"label": "Feature Value", "text": _fmt_num(feat) or "—"}]

    if feat is not None and steps:
        steps.append({"label": "Dataset", "text": _fmt_num(feat) or "—"})

    return {
        "sample": sample,
        "time": time,
        "dependencies": deps_lines,
        "formula_doc": formula_doc,
        "steps": steps,
        "tree_lines": tree_lines,
        "feature_value": feat,
    }


def build_calculation_rows(
    trace: list[dict[str, Any]],
    *,
    feature_name: str,
    replay_lookup: dict[int, dict[str, Any]],
    step_sec: int,
    sample_indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build calculation table rows for *sample_indices* (default: full trace)."""
    spec = build_formula_spec(feature_name)
    kind = spec["kind"]
    period = _ema_period_from_name(feature_name)
    columns = spec["table_columns"]
    headers = _display_headers(columns, period)

    if sample_indices is None:
        indices = list(range(len(trace)))
    else:
        indices = list(sample_indices)

    rows_out: list[dict[str, Any]] = []
    for idx in indices:
        if idx < 0 or idx >= len(trace):
            continue
        tr = trace[idx]
        replay_vals = lookup_replay_values(replay_lookup, tr["ts"], step_sec=step_sec)

        operands = _operand_values(
            feature_name,
            kind=kind,
            period=period,
            replay_vals=replay_vals,
            feature_ready=bool(tr.get("feature_ready")),
        )
        display: dict[str, str] = {
            "time": str(tr.get("time", "—")),
            "sample": str(tr.get("samples", "")),
        }
        for col in columns:
            if col in ("time", "sample"):
                continue
            val = operands.get(col)
            if col == "feature_value":
                val = operands.get("feature_value")
            display[col] = _null_display(_fmt_num(val) if val is not None else None)

        breakdown = build_row_breakdown(
            feature_name,
            sample=int(tr.get("samples") or 0),
            time=str(tr.get("time") or ""),
            operands=operands,
            formula_spec=spec,
        )
        rows_out.append({
            "index": idx,
            "display": display,
            "operands": operands,
            "breakdown": breakdown,
            "feature_ready": bool(tr.get("feature_ready")),
        })

    spec["table_headers"] = headers
    return rows_out, spec


def _row_score(row: dict[str, Any]) -> tuple[int, float]:
    """Prefer ATM-ish rows with core fields populated."""
    score = 0
    if row.get("ltp") is not None:
        score += 4
    if row.get("spot") is not None:
        score += 4
    delta = row.get("delta")
    if delta is not None:
        try:
            score += max(0, 3 - int(abs(float(delta)) * 10))
        except (TypeError, ValueError):
            pass
    return score, -float(row.get("ltp") or 0)


def resolve_primary_replay_token(
    rows: list[dict[str, Any]],
    *,
    anchor_ts: float | None = None,
    step_sec: int = 3,
) -> str | None:
    """Pick one option token for stable replay lookup (avoids ATM row switching)."""
    if not rows:
        return None
    counts: dict[str, int] = {}
    for raw in rows:
        tok = str(raw.get("token") or "").strip()
        if tok:
            counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        return None

    if anchor_ts is not None:
        ts_key = normalize_epoch_ts(anchor_ts)
        if ts_key is not None:
            step_i = max(int(step_sec), 1)
            at_anchor: set[str] = set()
            for raw in rows:
                row_ts = normalize_epoch_ts(raw.get("timestamp"))
                if row_ts is None:
                    continue
                if abs(row_ts - ts_key) <= max(step_i * 3, 6):
                    tok = str(raw.get("token") or "").strip()
                    if tok:
                        at_anchor.add(tok)
            if at_anchor:
                return sorted(at_anchor, key=lambda tok: (-counts.get(tok, 0), tok))[0]

    return sorted(counts, key=lambda tok: (-counts[tok], tok))[0]


def _normalize_replay_cell(val: Any) -> Any | None:
    """Map replay cell to a value or explicit NULL (None)."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s or s.upper() == "NULL" or s.lower() == "nan":
            return None
        try:
            f = float(s)
            return None if math.isnan(f) else f
        except ValueError:
            return val
    if isinstance(val, (int, float)):
        return float(val)
    return val


def _ingest_lookup_row(
    lookup: dict[int, dict[str, Any]],
    ts_key: int,
    row: dict[str, Any],
) -> None:
    if not row:
        return
    prev = lookup.get(ts_key)
    if prev is None or _row_score(row) > _row_score(prev):
        lookup[ts_key] = row


def build_replay_lookup_from_rows(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    token: str | None = None,
    anchor_ts: float | None = None,
    step_sec: int = 3,
) -> dict[int, dict[str, Any]]:
    """Build timestamp lookup from serialized chain replay rows."""
    token_key = str(token).strip() if token else None
    if token_key is None:
        token_key = resolve_primary_replay_token(
            rows,
            anchor_ts=anchor_ts,
            step_sec=step_sec,
        )
    filtered = rows
    if token_key:
        filtered = [r for r in rows if str(r.get("token") or "").strip() == token_key]
    lookup: dict[int, dict[str, Any]] = {}
    use_cols = list(dict.fromkeys(columns + ["ltp", "spot", "delta", "token"]))
    for raw in filtered:
        ts_key = normalize_epoch_ts(raw.get("timestamp"))
        if ts_key is None:
            continue
        bucket: dict[str, Any] = {}
        for col in use_cols:
            if col not in raw:
                continue
            bucket[col] = _normalize_replay_cell(raw[col])
        _ingest_lookup_row(lookup, ts_key, bucket)
    if not lookup and token_key and rows:
        return build_replay_lookup_from_rows(
            rows,
            columns,
            token=None,
            anchor_ts=anchor_ts,
            step_sec=step_sec,
        )
    return lookup


def align_replay_lookup_to_trace(
    lookup: dict[int, dict[str, Any]],
    *,
    chain_rows: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    columns: list[str],
    step_sec: int,
    token: str | None = None,
    anchor_ts: float | None = None,
) -> dict[int, dict[str, Any]]:
    """Prefer chain rows that overlap the policy trace when lookup hit rate is low."""
    if not trace or not chain_rows:
        return lookup
    hit_rate = lookup_replay_hit_rate(lookup, trace, step_sec=step_sec)
    if hit_rate >= 0.95:
        return lookup
    trace_keys: set[int] = set()
    for tr in trace:
        key = normalize_epoch_ts(tr.get("ts"))
        if key is not None:
            trace_keys.add(key)
    if not trace_keys:
        return lookup
    step_i = max(int(step_sec), 1)
    tol = max(step_i * 3, 6)
    filtered = [
        raw for raw in chain_rows
        if (ts_key := normalize_epoch_ts(raw.get("timestamp"))) is not None
        and any(abs(ts_key - tk) <= tol for tk in trace_keys)
    ]
    if not filtered:
        return lookup
    rebuilt = build_replay_lookup_from_rows(
        filtered,
        columns,
        token=token,
        anchor_ts=anchor_ts,
        step_sec=step_sec,
    )
    if not rebuilt:
        rebuilt = build_replay_lookup_from_rows(
            filtered,
            columns,
            token=None,
            anchor_ts=anchor_ts,
            step_sec=step_sec,
        )
    if lookup_replay_hit_rate(rebuilt, trace, step_sec=step_sec) > hit_rate:
        return rebuilt
    return lookup


def resolve_replay_lookup_from_result(
    result: Any,
    *,
    columns: list[str] | None = None,
) -> dict[int, dict[str, Any]]:
    """Best replay lookup: stored dict, else rebuild from chain rows on the result."""
    step_sec = int(max(getattr(result, "sampling_interval_sec", 3) or 3, 1))
    trace = list(getattr(result, "full_trace", None) or [])
    lookup = (
        getattr(result, "all_features_lookup", None)
        or getattr(result, "maturity_replay_lookup", None)
        or {}
    )
    if lookup:
        chain_rows = list(getattr(result, "all_features_rows", None) or [])
        if chain_rows and trace:
            cols = list(columns or getattr(result, "maturity_feature_names", None) or [])
            if cols:
                lookup = align_replay_lookup_to_trace(
                    lookup,
                    chain_rows=chain_rows,
                    trace=trace,
                    columns=cols,
                    step_sec=step_sec,
                    token=getattr(result, "replay_token", None),
                    anchor_ts=float(trace[0]["ts"]) if trace else None,
                )
        return lookup
    chain_rows = list(getattr(result, "all_features_rows", None) or [])
    if not chain_rows:
        return {}
    cols = list(columns or getattr(result, "maturity_feature_names", None) or [])
    if not cols:
        seen: set[str] = set()
        for raw in chain_rows[:500]:
            if isinstance(raw, dict):
                seen.update(str(k) for k in raw)
        cols = sorted(seen)
    anchor_ts = float(trace[0]["ts"]) if trace else None
    token = getattr(result, "replay_token", None)
    token = token or resolve_primary_replay_token(
        chain_rows,
        anchor_ts=anchor_ts,
        step_sec=step_sec,
    )
    rebuilt = build_replay_lookup_from_rows(
        chain_rows,
        cols,
        token=token,
        anchor_ts=anchor_ts,
        step_sec=step_sec,
    )
    if rebuilt:
        return align_replay_lookup_to_trace(
            rebuilt,
            chain_rows=chain_rows,
            trace=trace,
            columns=cols,
            step_sec=step_sec,
            token=token,
            anchor_ts=anchor_ts,
        )
    fallback = build_replay_lookup_from_rows(
        chain_rows,
        cols,
        token=None,
        anchor_ts=anchor_ts,
        step_sec=step_sec,
    )
    return align_replay_lookup_to_trace(
        fallback,
        chain_rows=chain_rows,
        trace=trace,
        columns=cols,
        step_sec=step_sec,
        token=token,
        anchor_ts=anchor_ts,
    )


def build_replay_lookup(
    df: Any,
    columns: list[str],
    *,
    step_sec: int,
    token: str | None = None,
    anchor_ts: float | None = None,
) -> dict[int, dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return {}
    if "timestamp" not in df.columns:
        return {}
    records = df.to_dict(orient="records")
    return build_replay_lookup_from_rows(
        records,
        columns,
        token=token,
        anchor_ts=anchor_ts,
        step_sec=step_sec,
    )


def lookup_replay_hit_rate(
    lookup: dict[int, dict[str, Any]],
    trace: list[dict[str, Any]],
    *,
    step_sec: int,
) -> float:
    if not lookup or not trace:
        return 0.0
    hits = sum(
        1 for tr in trace
        if lookup_replay_values(lookup, float(tr["ts"]), step_sec=step_sec)
    )
    return hits / len(trace)


def lookup_replay_values(
    lookup: dict[int, dict[str, Any]],
    ts: float,
    *,
    step_sec: int,
) -> dict[str, Any]:
    """Resolve replay row for a trace timestamp with step tolerance."""
    if not lookup:
        return {}
    ts_key = normalize_epoch_ts(ts)
    if ts_key is None:
        return {}
    step_i = max(int(step_sec), 1)
    for delta in (0, -step_i, step_i, -2 * step_i, 2 * step_i, -3 * step_i, 3 * step_i):
        key = ts_key + delta
        if key in lookup:
            return lookup[key]
    nearest_key = min(lookup.keys(), key=lambda k: abs(k - ts_key))
    if abs(nearest_key - ts_key) <= max(step_i * 3, 6):
        return lookup.get(nearest_key) or {}
    return {}
