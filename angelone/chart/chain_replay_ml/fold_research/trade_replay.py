"""Second-by-second trade replay timeline and entry decision explanation."""

from __future__ import annotations

import math
from typing import Any

from chain_replay_ml.strategy_registry.schema import normalize_strategy_config


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _fmt_time(ts: Any) -> str:
    try:
        s = int(float(ts))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"
    except (TypeError, ValueError):
        return "—"


def _rel_sec(ts: Any, base: float) -> str:
    try:
        d = float(ts) - base
        if abs(d) < 0.05:
            return "Entry"
        return f"+{int(round(d))}s"
    except (TypeError, ValueError):
        return "—"


def _find_entry_row(trade: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ep_id = trade.get("entry_prediction_id")
    if ep_id:
        for row in rows:
            if row.get("prediction_id") == ep_id:
                return row
    token = str(trade.get("token") or "")
    entry_ts = _num(trade.get("entry_ts"))
    if entry_ts is None:
        return None
    for row in rows:
        if str(row.get("token") or "") != token:
            continue
        ts = _num(row.get("timestamp"))
        if ts is not None and abs(ts - entry_ts) < 0.01:
            return row
    return None


def _prediction_context(row: dict[str, Any] | None, *, direction: str = "long") -> dict[str, Any]:
    if not row:
        return {}
    ltp = _num(row.get("ltp"))
    pred = _num(row.get("predicted_ltp"))
    actual = _num(row.get("actual_ltp"))
    pred_pct = None
    if ltp and pred and ltp > 0:
        pred_pct = round((pred - ltp) / ltp * 100.0, 2)

    expected_profit = None
    actual_profit = None
    if ltp is not None and pred is not None:
        expected_profit = round(pred - ltp, 4) if direction == "long" else round(ltp - pred, 4)
    if ltp is not None and actual is not None:
        actual_profit = round(actual - ltp, 4) if direction == "long" else round(ltp - actual, 4)

    model_error = None
    if pred is not None and actual is not None:
        model_error = round(actual - pred, 4)
    pred_error_pct = None
    if pred is not None and actual is not None and actual != 0:
        pred_error_pct = round((pred - actual) / abs(actual) * 100.0, 2)
    elif row.get("prediction_error") is not None and ltp and ltp > 0:
        pred_error_pct = round(float(row["prediction_error"]) / ltp * 100.0, 2)

    conf = _num(row.get("confidence"))
    prob = _estimate_success_probability(pred_pct, conf, direction_correct=row.get("direction_correct"))
    return {
        "prediction_pct": pred_pct,
        "current_ltp": ltp,
        "predicted_ltp": pred,
        "actual_ltp": actual,
        "expected_profit": expected_profit,
        "actual_profit": actual_profit,
        "model_error": model_error,
        "prediction_error": row.get("prediction_error"),
        "prediction_error_pct": pred_error_pct,
        "direction_correct": row.get("direction_correct"),
        "confidence": conf,
        "confidence_pct": round(conf, 1) if conf is not None else None,
        "confidence_note": "Model confidence not persisted on prediction rows yet." if conf is None else None,
        "probability_success_pct": prob.get("success_pct"),
        "probability_failure_pct": prob.get("failure_pct"),
        "probability_source": prob.get("source"),
        "probability_note": prob.get("note"),
    }


def _estimate_success_probability(
    pred_pct: float | None,
    confidence_pct: float | None,
    *,
    direction_correct: Any = None,
    audit_pass_ratio: float | None = None,
) -> dict[str, Any]:
    """Heuristic success probability until ensemble/calibration models are wired."""
    magnitude = abs(pred_pct or 0.0)
    base = 52.0 + min(magnitude, 18.0) * 1.35
    if audit_pass_ratio is not None:
        base += audit_pass_ratio * 12.0
    if confidence_pct is not None:
        success = 0.65 * confidence_pct + 0.35 * base
        source = "confidence_blend"
        note = None
    else:
        success = base
        source = "heuristic"
        note = "Estimated from prediction strength and entry checks until calibration is persisted."
    if direction_correct in (1, True, "1"):
        success += 4.0
    elif direction_correct in (0, False, "0"):
        success -= 8.0
    success = max(5.0, min(95.0, round(success, 1)))
    return {
        "success_pct": success,
        "failure_pct": round(100.0 - success, 1),
        "source": source,
        "note": note,
    }


def explain_entry_decision(row: dict[str, Any], cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Explain why strategy entered with structured audit checks."""
    cfg = normalize_strategy_config(cfg or {})
    entry = cfg["entry"]
    ltp = _num(row.get("ltp"))
    pred = _num(row.get("predicted_ltp"))
    spot = _num(row.get("spot"))
    direction = str(entry.get("direction") or "long").lower()
    decision = "BUY" if direction == "long" else "SELL"

    pred_pct = None
    if ltp and pred and ltp > 0:
        pred_pct = round((pred - ltp) / ltp * 100.0, 2)

    audit_checks: list[dict[str, Any]] = []
    blocked: list[str] = []

    premium_ok = ltp is not None and entry["premium_min"] <= ltp <= entry["premium_max"]
    audit_checks.append({
        "label": "Premium inside band",
        "passed": premium_ok,
        "detail": f"₹{ltp:.2f} in ₹{entry['premium_min']}–₹{entry['premium_max']}" if ltp is not None else "—",
    })
    if not premium_ok:
        blocked.append("Premium outside band")

    pred_positive = pred is not None and ltp is not None and ((pred > ltp) if direction == "long" else (pred < ltp))
    audit_checks.append({
        "label": "Prediction positive" if direction == "long" else "Prediction negative (short)",
        "passed": pred_positive,
        "detail": f"{pred_pct:+.2f}%" if pred_pct is not None else "—",
    })
    if not pred_positive:
        blocked.append("Prediction direction mismatch")

    min_move = float(entry.get("minimum_predicted_move_pct") or 0.0)
    signed_move = None
    if pred_pct is not None:
        signed_move = pred_pct if direction != "short" else -pred_pct
    if min_move > 0:
        move_ok = signed_move is not None and signed_move >= min_move
        audit_checks.append({
            "label": f"Predicted move ≥ {min_move:g}%",
            "passed": move_ok,
            "detail": f"{signed_move:+.2f}%" if signed_move is not None else "—",
        })
        if not move_ok:
            blocked.append("Predicted move below minimum")

    conf_cfg = cfg["confidence"]
    row_conf = _num(row.get("confidence"))
    min_sig = float(conf_cfg.get("min_signal_strength") or 0)
    if conf_cfg.get("use_model_confidence"):
        conf_ok = row_conf is not None and row_conf >= min_sig
        audit_checks.append({
            "label": f"Confidence ≥ {min_sig:.0f}%",
            "passed": conf_ok,
            "detail": f"{row_conf:.0f}%" if row_conf is not None else "—",
        })
        if not conf_ok:
            blocked.append("Confidence too low")
    else:
        audit_checks.append({
            "label": "Confidence gate",
            "passed": True,
            "detail": f"{row_conf:.0f}%" if row_conf is not None else "Not required",
        })

    risk = "Low"
    if pred_pct is not None and abs(pred_pct) > 12:
        risk = "High"
    elif pred_pct is not None and abs(pred_pct) > 6:
        risk = "Medium"
    risk_ok = risk != "High"
    audit_checks.append({
        "label": "Risk acceptable",
        "passed": risk_ok,
        "detail": risk,
    })

    audit_checks.append({"label": "Spread OK", "passed": True, "detail": "Not tracked in replay"})
    audit_checks.append({"label": "Volume OK", "passed": True, "detail": "Not tracked in replay"})

    decision_quality = _build_decision_quality(
        premium_ok=premium_ok,
        pred_positive=pred_positive,
        pred_pct=pred_pct,
        row_conf=row_conf,
        conf_required=bool(conf_cfg.get("use_model_confidence")),
        min_conf=min_sig,
    )

    prediction_ctx = _prediction_context(row, direction=direction)
    passed = sum(1 for c in audit_checks if c.get("passed"))
    total = len(audit_checks) or 1
    prob = _estimate_success_probability(
        pred_pct,
        prediction_ctx.get("confidence_pct"),
        direction_correct=row.get("direction_correct"),
        audit_pass_ratio=passed / total,
    )
    prediction_ctx["probability_success_pct"] = prob["success_pct"]
    prediction_ctx["probability_failure_pct"] = prob["failure_pct"]
    prediction_ctx["probability_source"] = prob["source"]
    prediction_ctx["probability_note"] = prob["note"]

    return {
        "decision": decision if not blocked else "BLOCKED",
        "prediction_pct": pred_pct,
        "confidence": row_conf,
        "premium": ltp,
        "spot": spot,
        "predicted_ltp": pred,
        "risk_score": risk,
        "audit_checks": audit_checks,
        "decision_quality": decision_quality,
        "blocked": blocked,
        "summary": "; ".join(c["label"] for c in audit_checks if c["passed"]) if not blocked else "; ".join(blocked),
        "prediction": prediction_ctx,
    }


def _build_exit_analysis(
    trade: dict[str, Any],
    entry_row: dict[str, Any] | None,
    *,
    target_price: float | None,
    stop_price: float | None,
    stop_hit: bool,
    target_hit: bool,
    premium_path: list[dict[str, Any]],
    direction: str,
) -> dict[str, Any]:
    entry_price = _num(trade.get("entry_price"))
    exit_price = _num(trade.get("exit_price"))
    exit_reason = str(trade.get("exit_reason") or "exit")
    net_pnl = _num(trade.get("net_pnl"))

    premiums = [_num(p.get("value")) for p in premium_path if _num(p.get("value")) is not None]
    max_premium = max(premiums) if premiums else exit_price
    min_premium = min(premiums) if premiums else exit_price

    target_missed_by = None
    if target_price is not None and entry_price and not target_hit:
        if direction == "long" and max_premium is not None:
            target_missed_by = round(target_price - max_premium, 4)
        elif direction == "short" and min_premium is not None:
            target_missed_by = round(min_premium - target_price, 4)

    max_fav_pct = _num(trade.get("max_favorable_pct"))
    max_adv_pct = _num(trade.get("max_adverse_pct"))
    max_profit = None
    max_dd = None
    if entry_price and entry_price > 0:
        if max_fav_pct is not None:
            max_profit = round(entry_price * max_fav_pct / 100.0, 4)
        if max_adv_pct is not None:
            max_dd = round(abs(entry_price * max_adv_pct / 100.0), 4)

    pred_ctx = _prediction_context(entry_row, direction=direction) if entry_row else {}
    pred_correct = pred_ctx.get("direction_correct")
    if pred_correct is None and entry_row:
        pred_correct = None
    strategy_correct = net_pnl is not None and net_pnl > 0

    return {
        "exit_reason": exit_reason,
        "exit_reason_label": {
            "target": "Target Hit",
            "stop": "Stop Hit",
            "max_hold": "Max Hold",
            "end_of_path": "End of Path",
        }.get(exit_reason, exit_reason.replace("_", " ").title()),
        "prediction_correct": pred_correct,
        "strategy_correct": strategy_correct,
        "target_missed_by": target_missed_by,
        "stop_hit": stop_hit,
        "stop_never_hit": not stop_hit,
        "target_hit": target_hit,
        "maximum_profit": max_profit,
        "maximum_drawdown": max_dd,
        "held_seconds": trade.get("holding_seconds"),
        "net_pnl": net_pnl,
        "model_error": pred_ctx.get("model_error"),
        "prediction_error_pct": pred_ctx.get("prediction_error_pct"),
    }


def _find_exit_row(trade: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    token = str(trade.get("token") or "")
    exit_ts = _num(trade.get("exit_ts"))
    if exit_ts is None:
        return None
    best: dict[str, Any] | None = None
    best_dt = 999.0
    for row in rows:
        if str(row.get("token") or "") != token:
            continue
        ts = _num(row.get("timestamp"))
        if ts is None:
            continue
        dt = abs(ts - exit_ts)
        if dt < best_dt:
            best_dt = dt
            best = row
    return best if best_dt < 2.0 else None


def _build_maximum_opportunity(
    *,
    entry_price: float | None,
    exit_price: float | None,
    premium_path: list[dict[str, Any]],
    net_pnl: float | None,
    direction: str,
) -> dict[str, Any]:
    premiums = [_num(p.get("value")) for p in premium_path if _num(p.get("value")) is not None]
    if entry_price is None:
        return {}
    highest = max(premiums) if premiums else entry_price
    lowest = min(premiums) if premiums else entry_price
    if direction == "long":
        max_possible = round(highest - entry_price, 4)
        captured = round((exit_price or entry_price) - entry_price, 4) if exit_price is not None else None
    else:
        max_possible = round(entry_price - lowest, 4)
        captured = round(entry_price - (exit_price or entry_price), 4) if exit_price is not None else None
    efficiency = None
    if max_possible and max_possible > 0 and captured is not None:
        efficiency = round(captured / max_possible * 100.0, 1)
    elif max_possible and max_possible < 0 and captured is not None:
        efficiency = 0.0
    return {
        "entry_premium": entry_price,
        "highest_premium": highest,
        "lowest_premium": lowest,
        "exit_premium": exit_price,
        "maximum_profit_possible": max_possible,
        "maximum_possible": max_possible,
        "captured_profit": captured,
        "capture_efficiency_pct": efficiency,
        "efficiency_pct": efficiency,
        "net_pnl": net_pnl,
    }


def _build_pnl_path(
    *,
    entry_price: float | None,
    premium_path: list[dict[str, Any]],
    direction: str,
) -> list[dict[str, Any]]:
    if entry_price is None:
        return []
    out: list[dict[str, Any]] = []
    for pt in premium_path:
        prem = _num(pt.get("value"))
        if prem is None:
            continue
        pnl = round(prem - entry_price, 4) if direction == "long" else round(entry_price - prem, 4)
        out.append({
            "timestamp": pt.get("timestamp"),
            "time_label": pt.get("time_label"),
            "rel_label": pt.get("rel_label"),
            "premium": prem,
            "pnl": pnl,
        })
    return out


def _build_rule_timeline(events: list[dict[str, Any]], decision: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for ev in events:
        et = str(ev.get("event_type") or "")
        if et == "prediction":
            steps.append({"label": "Prediction", "detail": ev.get("label"), "kind": "prediction"})
        elif et == "strategy_entry":
            steps.append({"label": "Signal Generated", "detail": "Entry signal from strategy rules", "kind": "signal"})
            steps.append({"label": decision.get("decision") or "BUY", "detail": ev.get("label"), "kind": "entry"})
        elif et == "premium_drop":
            steps.append({"label": "Premium dropped", "detail": ev.get("label"), "kind": "premium_drop"})
        elif et == "premium_tick":
            if steps and steps[-1].get("kind") == "premium_tick":
                steps[-1]["detail"] = ev.get("label")
            else:
                steps.append({"label": "Premium moved", "detail": ev.get("label"), "kind": "premium_tick"})
        elif et == "stop_hit":
            steps.append({"label": "Stop triggered", "detail": ev.get("label"), "kind": "stop"})
        elif et == "target_hit":
            steps.append({"label": "Target reached", "detail": ev.get("label"), "kind": "target"})
        elif et == "trade_exit":
            steps.append({"label": "Exit", "detail": ev.get("label"), "kind": "exit"})
    conf = decision.get("confidence")
    if conf is not None and steps:
        insert_at = 3 if len(steps) > 3 else len(steps)
        steps.insert(insert_at, {
            "label": "Confidence gate",
            "detail": f"Confidence {conf:.0f}%",
            "kind": "confidence",
        })
    return steps


def _pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return round((new - old) / abs(old) * 100.0, 2)


def _build_since_entry(
    entry_row: dict[str, Any] | None,
    exit_row: dict[str, Any] | None,
    *,
    feature_rows: dict[int, dict[str, Any]] | None,
    direction: str,
) -> dict[str, Any]:
    if not entry_row:
        return {}
    entry_feats: dict[str, Any] = {}
    exit_feats: dict[str, Any] = {}
    if feature_rows:
        eri = entry_row.get("row_index")
        xri = (exit_row or {}).get("row_index")
        if eri is not None and int(eri) in feature_rows:
            entry_feats = feature_rows[int(eri)]
        if xri is not None and int(xri) in feature_rows:
            exit_feats = feature_rows[int(xri)]

    entry_spot = _num(entry_row.get("spot"))
    exit_spot = _num((exit_row or {}).get("spot"))
    entry_ltp = _num(entry_row.get("ltp"))
    exit_ltp = _num((exit_row or {}).get("actual_ltp")) or _num((exit_row or {}).get("ltp"))

    metrics = {
        "spot": _pct_change(exit_spot, entry_spot),
        "premium": _pct_change(exit_ltp, entry_ltp),
        "iv": _pct_change(_num(exit_feats.get("current_iv")), _num(entry_feats.get("current_iv"))),
        "pcr": _pct_change(_num(exit_feats.get("chain_pcr")), _num(entry_feats.get("chain_pcr"))),
        "delta": _delta_abs(_num(exit_feats.get("delta")), _num(entry_feats.get("delta"))),
        "theta": _delta_abs(_num(exit_feats.get("theta")), _num(entry_feats.get("theta"))),
        "gamma": _delta_abs(_num(exit_feats.get("gamma")), _num(entry_feats.get("gamma"))),
    }
    return {
        "direction": direction,
        "metrics": metrics,
        "entry": {"spot": entry_spot, "premium": entry_ltp, **entry_feats},
        "exit": {"spot": exit_spot, "premium": exit_ltp, **exit_feats},
    }


def _delta_abs(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return round(new - old, 4)


def _severity_for_delta(delta: float | None, *, pct: bool = False) -> str:
    if delta is None:
        return "LOW"
    mag = abs(delta)
    if pct:
        if mag >= 4.0:
            return "HIGH"
        if mag >= 1.5:
            return "MEDIUM"
        return "LOW"
    if mag >= 3.0:
        return "HIGH"
    if mag >= 1.0:
        return "MEDIUM"
    return "LOW"


def _build_feature_alerts(feature_series: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    for fname, series in feature_series.items():
        if len(series) < 2:
            continue
        entry_val = _num(series[0].get("value"))
        worst_delta = 0.0
        worst_label = ""
        for pt in series[1:]:
            d = _num(pt.get("delta"))
            if d is None:
                continue
            if abs(d) > abs(worst_delta):
                worst_delta = d
                worst_label = str(pt.get("rel_label") or pt.get("time_label") or "")
        total_delta = None
        if entry_val is not None and series[-1].get("value") is not None:
            last = _num(series[-1].get("value"))
            if last is not None:
                total_delta = round(last - entry_val, 4)
        use_delta = total_delta if total_delta is not None else worst_delta
        if use_delta is None or abs(use_delta) < 0.01:
            continue
        direction = "↑" if use_delta > 0 else "↓"
        label = fname.replace("_", " ").title()
        if "negative" in fname or use_delta < 0:
            drift = f"Large {'negative' if use_delta < 0 else 'positive'} drift"
        else:
            drift = f"Large {'positive' if use_delta > 0 else 'negative'} drift"
        alerts.append({
            "feature": fname,
            "label": label,
            "value": series[-1].get("value"),
            "delta": use_delta,
            "delta_label": f"({use_delta:+.2f})",
            "direction": direction,
            "drift_text": drift,
            "when": worst_label,
            "severity": _severity_for_delta(use_delta),
        })
    alerts.sort(key=lambda a: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}[a["severity"]], -abs(a.get("delta") or 0)))
    return alerts


def _build_decision_quality(
    *,
    premium_ok: bool,
    pred_positive: bool,
    pred_pct: float | None,
    row_conf: float | None,
    conf_required: bool,
    min_conf: float,
) -> dict[str, Any]:
    premium_score = 20 if premium_ok else 6
    if pred_positive:
        pred_score = min(20, int(14 + min(abs(pred_pct or 0.0), 6.0)))
    else:
        pred_score = 4
    spread_score = 20
    if conf_required and row_conf is not None:
        conf_score = min(20, int(row_conf * 20.0 / max(min_conf, 1.0) * 0.85))
    elif row_conf is not None:
        conf_score = min(20, int(row_conf / 5.0))
    else:
        conf_score = 14
    liquidity_score = 19
    dimensions = [
        {"label": "Premium", "score": premium_score, "max": 20},
        {"label": "Prediction", "score": pred_score, "max": 20},
        {"label": "Spread", "score": spread_score, "max": 20},
        {"label": "Confidence", "score": conf_score, "max": 20},
        {"label": "Liquidity", "score": liquidity_score, "max": 20},
    ]
    total = sum(d["score"] for d in dimensions)
    return {"total": total, "max": 100, "dimensions": dimensions}


def _build_trade_verdict(
    trade: dict[str, Any],
    exit_analysis: dict[str, Any],
    since_entry: dict[str, Any],
    decision: dict[str, Any],
    *,
    stop_hit: bool,
    target_hit: bool,
    max_opportunity: dict[str, Any],
) -> dict[str, Any]:
    net_pnl = _num(trade.get("net_pnl"))
    if net_pnl is not None and net_pnl > 0:
        outcome = "TRADE SUCCEEDED"
    elif net_pnl is not None and net_pnl < 0:
        outcome = "TRADE FAILED"
    else:
        outcome = "TRADE BREAKEVEN"

    model_reasons: list[dict[str, Any]] = []
    strategy_reasons: list[dict[str, Any]] = []
    pred_ok = exit_analysis.get("prediction_correct")
    pred_pct = (decision.get("prediction") or {}).get("prediction_pct")

    if pred_ok in (1, True, "1"):
        model_verdict = "Correct"
        model_reasons.append({"passed": True, "label": "Direction correct", "detail": ""})
    elif pred_ok in (0, False, "0"):
        model_verdict = "Wrong"
        model_reasons.append({"passed": False, "label": "Wrong", "detail": "Direction did not match outcome"})
        if pred_pct is not None and abs(pred_pct) > 8:
            model_reasons.append({
                "passed": False,
                "label": "Prediction overshot",
                "detail": f"{pred_pct:+.1f}%",
            })
    else:
        model_err = exit_analysis.get("prediction_error_pct")
        if model_err is not None and abs(model_err) > 5:
            model_verdict = "Wrong"
            model_reasons.append({
                "passed": False,
                "label": "Prediction overshot",
                "detail": f"{model_err:+.1f}%",
            })
        else:
            model_verdict = "Unclear"
            model_reasons.append({"passed": True, "label": "Insufficient label data", "detail": ""})

    metrics = (since_entry or {}).get("metrics") or {}
    spot_chg = metrics.get("spot")
    prem_chg = metrics.get("premium")
    iv_chg = metrics.get("iv")

    entry_checks = decision.get("audit_checks") or []
    premium_ok = any(c.get("label", "").startswith("Premium") and c.get("passed") for c in entry_checks)
    pred_gate_ok = any("Prediction" in str(c.get("label", "")) and c.get("passed") for c in entry_checks)
    if premium_ok and pred_gate_ok:
        strategy_reasons.append({"passed": True, "label": "Entry correct", "detail": ""})

    if stop_hit:
        max_fav = exit_analysis.get("maximum_profit")
        if max_fav is not None and max_fav > 0:
            strategy_reasons.append({
                "passed": False,
                "label": "Stop too tight",
                "detail": f"Reached +₹{max_fav:.2f} before stop",
            })
            strategy_reasons.append({
                "passed": False,
                "label": "Could have survived",
                "detail": "Favorable move before stop",
            })
        else:
            strategy_reasons.append({
                "passed": False,
                "label": "Stop hit",
                "detail": exit_analysis.get("exit_reason_label") or "",
            })

    if prem_chg is not None and prem_chg < -2.0:
        strategy_reasons.append({"passed": False, "label": "Premium decay", "detail": f"{prem_chg:.2f}%"})

    if iv_chg is not None and iv_chg < -1.0:
        strategy_reasons.append({"passed": False, "label": "IV contraction", "detail": f"{iv_chg:.2f}%"})

    if spot_chg is not None and prem_chg is not None:
        if direction := since_entry.get("direction"):
            if direction == "long" and spot_chg > 0.1 and prem_chg < -1.0:
                strategy_reasons.append({
                    "passed": False,
                    "label": "Spot rose, premium fell",
                    "detail": "Premium did not follow spot",
                })
            if spot_chg < -0.15:
                model_reasons.append({
                    "passed": False,
                    "label": "Spot reversed",
                    "detail": f"{spot_chg:.2f}% unexpectedly",
                })

    eff = max_opportunity.get("efficiency_pct")
    max_possible = max_opportunity.get("maximum_profit_possible")
    if target_hit:
        strategy_reasons.append({"passed": True, "label": "Target captured", "detail": ""})
    elif eff is not None and eff < 30 and net_pnl is not None and net_pnl < 0:
        strategy_reasons.append({
            "passed": False,
            "label": "Poor capture efficiency",
            "detail": f"{eff:.0f}% of max opportunity",
        })
    if max_possible is not None and max_possible <= 0 and net_pnl is not None and net_pnl < 0:
        strategy_reasons.append({
            "passed": False,
            "label": "No opportunity given",
            "detail": "Market never offered favorable move",
        })

    strategy_failed = any(not r.get("passed") for r in strategy_reasons)
    if net_pnl is not None and net_pnl > 0:
        strategy_verdict = "Succeeded"
        if not strategy_reasons:
            strategy_reasons.append({"passed": True, "label": "Execution worked", "detail": ""})
    elif strategy_failed:
        strategy_verdict = "Failed"
    else:
        strategy_verdict = "Neutral"

    reasons = model_reasons + strategy_reasons

    return {
        "outcome": outcome,
        "model": {"verdict": model_verdict, "reasons": model_reasons},
        "strategy": {"verdict": strategy_verdict, "reasons": strategy_reasons},
        "reasons": reasons,
    }


def _trade_signature(
    trade: dict[str, Any],
    entry_row: dict[str, Any] | None,
    *,
    pred_pct: float | None,
    feature_rows: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    feats: dict[str, Any] = {}
    if entry_row and feature_rows:
        ri = entry_row.get("row_index")
        if ri is not None:
            feats = feature_rows.get(int(ri)) or {}
    return {
        "premium": _num(trade.get("entry_price")),
        "delta": _num(feats.get("delta")),
        "hold_seconds": float(trade.get("holding_seconds") or 0),
        "prediction_pct": pred_pct,
        "iv": _num(feats.get("current_iv")),
    }


def _dimension_match(label: str, a: float | None, b: float | None, *, rel_tol: float, abs_tol: float | None = None) -> bool:
    if a is None or b is None:
        return False
    if abs_tol is not None and abs(a - b) <= abs_tol:
        return True
    base = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / base <= rel_tol


def _matched_on_dimensions(current: dict[str, Any], peer: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        ("Premium", "premium", {"rel_tol": 0.12}),
        ("Delta", "delta", {"rel_tol": 0.15, "abs_tol": 0.05}),
        ("Hold Time", "hold_seconds", {"rel_tol": 0.25, "abs_tol": 5.0}),
        ("Prediction %", "prediction_pct", {"rel_tol": 0.20, "abs_tol": 1.5}),
        ("IV", "iv", {"rel_tol": 0.10, "abs_tol": 0.5}),
    ]
    out: list[dict[str, Any]] = []
    for label, key, tol in checks:
        matched = _dimension_match(
            label,
            _num(current.get(key)),
            _num(peer.get(key)),
            rel_tol=float(tol["rel_tol"]),
            abs_tol=_num(tol.get("abs_tol")),
        )
        out.append({"label": label, "matched": matched})
    return out


def _similarity_from_matched(matched_on: list[dict[str, Any]]) -> float:
    if not matched_on:
        return 0.0
    return sum(1 for m in matched_on if m.get("matched")) / len(matched_on)


def _trade_similarity_vector(
    trade: dict[str, Any],
    entry_row: dict[str, Any] | None,
    *,
    pred_pct: float | None,
) -> list[float]:
    entry_price = _num(trade.get("entry_price")) or 0.0
    hold = float(trade.get("holding_seconds") or 0)
    reason_map = {"target": 1.0, "stop": 2.0, "max_hold": 3.0}
    reason_code = reason_map.get(str(trade.get("exit_reason") or ""), 0.0)
    return [
        entry_price / 50.0,
        abs(pred_pct or 0.0) / 20.0,
        hold / 120.0,
        reason_code / 3.0,
        1.0 if (_num(trade.get("net_pnl")) or 0) > 0 else 0.0,
    ]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _find_similar_trades(
    trade: dict[str, Any],
    entry_row: dict[str, Any] | None,
    peer_trades: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    pred_pct: float | None,
    feature_rows: dict[int, dict[str, Any]] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    current_id = str(trade.get("trade_id") or "")
    current_sig = _trade_signature(trade, entry_row, pred_pct=pred_pct, feature_rows=feature_rows)
    vec = _trade_similarity_vector(trade, entry_row, pred_pct=pred_pct)
    scored: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []
    for peer in peer_trades:
        pid = str(peer.get("trade_id") or "")
        if pid == current_id:
            continue
        peer_entry = _find_entry_row(peer, rows)
        peer_pred = None
        if peer_entry:
            ltp = _num(peer_entry.get("ltp"))
            pred = _num(peer_entry.get("predicted_ltp"))
            if ltp and pred and ltp > 0:
                peer_pred = round((pred - ltp) / ltp * 100.0, 2)
        peer_sig = _trade_signature(peer, peer_entry, pred_pct=peer_pred, feature_rows=feature_rows)
        matched_on = _matched_on_dimensions(current_sig, peer_sig)
        blend = 0.55 * _similarity_from_matched(matched_on) + 0.45 * _cosine_similarity(
            vec, _trade_similarity_vector(peer, peer_entry, pred_pct=peer_pred),
        )
        scored.append((blend, peer, matched_on))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for sim, peer, matched_on in scored[:limit]:
        out.append({
            "trade_id": peer.get("trade_id"),
            "token": peer.get("token"),
            "net_pnl": peer.get("net_pnl"),
            "exit_reason": peer.get("exit_reason"),
            "similarity_pct": round(sim * 100.0, 1),
            "matched_on": matched_on,
        })
    return out


def _feature_direction_label(fname: str, pct_chg: float, *, trade_direction: str) -> str:
    bullish_up = {"spot_ema20_to_ltp_ratio", "delta", "chain_pcr", "current_iv"}
    if fname in bullish_up:
        aligned = pct_chg > 0 if trade_direction == "long" else pct_chg < 0
    elif fname == "theta":
        aligned = pct_chg < 0 if trade_direction == "long" else pct_chg > 0
    else:
        aligned = pct_chg > 0
    if abs(pct_chg) < 0.5:
        return "Neutral feature"
    return "Most bullish feature" if aligned else "Most bearish feature"


def _build_mini_shap(
    feature_series: dict[str, list[dict[str, Any]]],
    *,
    direction: str = "long",
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for fname, series in feature_series.items():
        if not series:
            continue
        entry = _num(series[0].get("value"))
        last = _num(series[-1].get("value"))
        if entry is None or last is None:
            continue
        pct_chg = round((last - entry) / abs(entry) * 100.0, 2) if entry != 0 else 0.0
        arrow = "▲" if pct_chg > 0 else ("▼" if pct_chg < 0 else "—")
        items.append({
            "feature": fname,
            "label": fname.replace("_", " "),
            "value": last,
            "pct_change": pct_chg,
            "arrow": arrow,
            "direction_label": _feature_direction_label(fname, pct_chg, trade_direction=direction),
        })
    items.sort(key=lambda x: abs(x.get("pct_change") or 0), reverse=True)
    return items


def _build_regime_badges(
    entry_row: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    feature_rows: dict[int, dict[str, Any]] | None,
    *,
    direction: str,
) -> list[str]:
    if not entry_row:
        return []
    from .regime_iv import _classify_iv_row

    feats: dict[str, Any] = {}
    if feature_rows:
        ri = entry_row.get("row_index")
        if ri is not None:
            feats = feature_rows.get(int(ri)) or {}

    entry_ts = _num(entry_row.get("timestamp"))
    prev_spot = None
    for row in sorted(rows, key=lambda r: _num(r.get("timestamp")) or 0):
        ts = _num(row.get("timestamp"))
        if ts is None or entry_ts is None or ts >= entry_ts:
            break
        prev_spot = _num(row.get("spot"))

    tags = _classify_iv_row(entry_row, feats, prev_spot=prev_spot, prev_iv=None, prev_pcr=None)
    badges = list(tags[:5])
    pred = _num(entry_row.get("predicted_ltp"))
    ltp = _num(entry_row.get("ltp"))
    if ltp and pred:
        if (pred > ltp and direction == "long") or (pred < ltp and direction == "short"):
            badges.append("Bullish" if direction == "long" else "Bearish")
    iv = _num(feats.get("current_iv"))
    if iv is not None and iv < 12:
        badges.append("Low IV")
    elif iv is not None and iv > 20:
        badges.append("High IV")
    return list(dict.fromkeys(badges))[:8]


def _append_feature_point(
    feature_series: dict[str, list[dict[str, Any]]],
    fname: str,
    *,
    timestamp: float,
    time_label: str,
    rel_label: str,
    value: Any,
) -> None:
    series = feature_series.setdefault(fname, [])
    prev = _num(series[-1]["value"]) if series else None
    cur = _num(value)
    delta = round(cur - prev, 4) if cur is not None and prev is not None else None
    series.append({
        "timestamp": timestamp,
        "time_label": time_label,
        "rel_label": rel_label,
        "value": cur,
        "delta": delta,
    })


def build_trade_replay(
    trade: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    cfg: dict[str, Any] | None = None,
    feature_rows: dict[int, dict[str, Any]] | None = None,
    peer_trades: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a chronological trade replay from entry through exit."""
    cfg_norm = normalize_strategy_config(cfg or {}) if cfg else None
    entry_row = _find_entry_row(trade, rows)
    token = str(trade.get("token") or "")
    entry_ts = _num(trade.get("entry_ts"))
    exit_ts = _num(trade.get("exit_ts"))
    entry_price = _num(trade.get("entry_price"))
    if entry_ts is None or exit_ts is None:
        return {"ok": False, "error": "trade missing entry/exit timestamps"}

    direction = str((cfg_norm or {}).get("entry", {}).get("direction") or "long").lower()
    decision = explain_entry_decision(entry_row or {}, cfg_norm or {}) if entry_row else {}

    events: list[dict[str, Any]] = []
    spot_path: list[dict[str, Any]] = []
    premium_path: list[dict[str, Any]] = []

    if entry_row:
        ets = float(entry_row.get("timestamp") or entry_ts)
        events.append({
            "event_type": "prediction",
            "display_type": "Prediction",
            "timestamp": ets,
            "time_label": _fmt_time(ets),
            "spot": entry_row.get("spot"),
            "ltp": entry_row.get("ltp"),
            "predicted_ltp": entry_row.get("predicted_ltp"),
            "actual_ltp": entry_row.get("actual_ltp"),
            "label": f"Spot {entry_row.get('spot')} · LTP {entry_row.get('ltp')}",
            "prediction_id": entry_row.get("prediction_id"),
            "row_index": entry_row.get("row_index"),
        })
        sp = _num(entry_row.get("spot"))
        if sp is not None:
            spot_path.append({"timestamp": ets, "time_label": _fmt_time(ets), "rel_label": "Entry", "value": sp})
        lp = _num(entry_row.get("ltp"))
        if lp is not None:
            premium_path.append({"timestamp": ets, "time_label": _fmt_time(ets), "rel_label": "Entry", "value": lp})

    events.append({
        "event_type": "strategy_entry",
        "display_type": "Strategy Entry",
        "timestamp": entry_ts,
        "time_label": _fmt_time(entry_ts),
        "price": entry_price,
        "token": token,
        "label": f"{decision.get('decision', 'ENTRY')} @ ₹{entry_price}",
    })

    target_pct = float((cfg_norm or {}).get("target", {}).get("target_profit_pct") or 8.0)
    stop_pct = float((cfg_norm or {}).get("stop", {}).get("stop_loss_pct") or 5.0)
    target_price = stop_price = None
    if entry_price and entry_price > 0:
        if direction == "long":
            target_price = entry_price * (1.0 + target_pct / 100.0)
            stop_price = entry_price * (1.0 - stop_pct / 100.0)
        else:
            target_price = entry_price * (1.0 - target_pct / 100.0)
            stop_price = entry_price * (1.0 + stop_pct / 100.0)

    hold_rows = [
        r for r in rows
        if str(r.get("token") or "") == token
        and (_num(r.get("timestamp")) or 0) >= entry_ts
        and (_num(r.get("timestamp")) or 0) <= exit_ts
    ]
    hold_rows.sort(key=lambda r: (_num(r.get("timestamp")) or 0, int(r.get("row_index") or 0)))

    target_hit = False
    stop_hit = False
    feature_series: dict[str, list[dict[str, Any]]] = {}
    track_features = (
        "spot_ema20_to_ltp_ratio", "delta", "gamma", "current_iv", "chain_pcr", "theta",
    )
    prev_premium = entry_price

    for row in hold_rows:
        ts = _num(row.get("timestamp"))
        if ts is None or ts <= entry_ts:
            continue
        premium = _num(row.get("actual_ltp")) or _num(row.get("ltp"))
        if premium is None:
            continue
        tl = _fmt_time(ts)
        rl = _rel_sec(ts, entry_ts)

        events.append({
            "event_type": "premium_tick",
            "display_type": "Premium",
            "timestamp": ts,
            "time_label": tl,
            "spot": row.get("spot"),
            "ltp": premium,
            "predicted_ltp": row.get("predicted_ltp"),
            "actual_ltp": row.get("actual_ltp"),
            "label": f"Premium ₹{premium:.2f}",
            "prediction_id": row.get("prediction_id"),
            "row_index": row.get("row_index"),
        })
        premium_path.append({"timestamp": ts, "time_label": tl, "rel_label": rl, "value": premium})
        if prev_premium is not None and premium < prev_premium * 0.995 and direction == "long":
            events.append({
                "event_type": "premium_drop",
                "display_type": "Premium Drop",
                "timestamp": ts,
                "time_label": tl,
                "label": f"Premium dropped to ₹{premium:.2f}",
            })
        prev_premium = premium
        sp = _num(row.get("spot"))
        if sp is not None:
            spot_path.append({"timestamp": ts, "time_label": tl, "rel_label": rl, "value": sp})

        ri = row.get("row_index")
        if feature_rows and ri is not None and int(ri) in feature_rows:
            feats = feature_rows[int(ri)]
            for fname in track_features:
                if fname in feats and feats[fname] is not None:
                    _append_feature_point(
                        feature_series, fname,
                        timestamp=ts, time_label=tl, rel_label=rl, value=feats[fname],
                    )

        if target_price is not None and not target_hit:
            hit = premium >= target_price if direction == "long" else premium <= target_price
            if hit:
                target_hit = True
                events.append({
                    "event_type": "target_hit",
                    "display_type": "Target Hit",
                    "timestamp": ts,
                    "time_label": tl,
                    "price": premium,
                    "label": f"Target hit @ ₹{premium:.2f} (target ₹{target_price:.2f})",
                })
        if stop_price is not None and not stop_hit:
            hit = premium <= stop_price if direction == "long" else premium >= stop_price
            if hit:
                stop_hit = True
                events.append({
                    "event_type": "stop_hit",
                    "display_type": "Stop Hit",
                    "timestamp": ts,
                    "time_label": tl,
                    "price": premium,
                    "label": f"Stop hit @ ₹{premium:.2f} (stop ₹{stop_price:.2f})",
                })

    exit_reason = str(trade.get("exit_reason") or "exit")
    reason_label = {
        "target": "Target Hit",
        "stop": "Stop Hit",
        "max_hold": "Max Hold",
        "end_of_path": "End of Path",
    }.get(exit_reason, exit_reason.replace("_", " ").title())

    events.append({
        "event_type": "trade_exit",
        "display_type": "Exit",
        "timestamp": exit_ts,
        "time_label": _fmt_time(exit_ts),
        "price": trade.get("exit_price"),
        "net_pnl": trade.get("net_pnl"),
        "return_pct": trade.get("return_pct"),
        "exit_reason": exit_reason,
        "label": f"Exit — {reason_label} · PnL ₹{trade.get('net_pnl')}",
    })

    events.sort(key=lambda e: (e.get("timestamp") or 0, _event_order(e.get("event_type"))))
    for i, ev in enumerate(events):
        ev["sequence"] = i + 1

    exit_analysis = _build_exit_analysis(
        trade,
        entry_row,
        target_price=target_price,
        stop_price=stop_price,
        stop_hit=stop_hit,
        target_hit=target_hit,
        premium_path=premium_path,
        direction=direction,
    )
    exit_row = _find_exit_row(trade, rows)
    exit_price = _num(trade.get("exit_price"))
    max_opportunity = _build_maximum_opportunity(
        entry_price=entry_price,
        exit_price=exit_price,
        premium_path=premium_path,
        net_pnl=_num(trade.get("net_pnl")),
        direction=direction,
    )
    pnl_path = _build_pnl_path(entry_price=entry_price, premium_path=premium_path, direction=direction)
    since_entry = _build_since_entry(entry_row, exit_row, feature_rows=feature_rows, direction=direction)
    feature_alerts = _build_feature_alerts(feature_series)
    rule_timeline = _build_rule_timeline(events, decision)
    trade_verdict = _build_trade_verdict(
        trade,
        exit_analysis,
        since_entry,
        decision,
        stop_hit=stop_hit,
        target_hit=target_hit,
        max_opportunity=max_opportunity,
    )
    pred_pct = (decision.get("prediction") or {}).get("prediction_pct")
    similar_trades = _find_similar_trades(
        trade,
        entry_row,
        peer_trades or [],
        rows,
        pred_pct=pred_pct,
        feature_rows=feature_rows,
    )
    mini_shap = _build_mini_shap(feature_series, direction=direction)
    regime_badges = _build_regime_badges(entry_row, rows, feature_rows, direction=direction)

    from .counterfactual import build_counterfactuals
    from .feature_time_machine import build_feature_time_machine
    from .trade_narrative import explain_trade_narrative

    counterfactuals = build_counterfactuals(
        trade,
        premium_path,
        cfg=cfg_norm or {},
        direction=direction,
    )
    feature_time_machine = build_feature_time_machine(feature_series, direction=direction)

    from .trade_replay_insights import (
        build_prediction_failure_analysis,
        build_research_conclusion,
        classify_trade,
        confidence_tier,
    )

    prediction_failure = build_prediction_failure_analysis(
        decision,
        since_entry,
        feature_alerts,
        exit_analysis=exit_analysis,
    )
    trade_classification = classify_trade(
        trade,
        trade_verdict,
        exit_analysis,
        max_opportunity,
        since_entry,
        regime_badges,
    )
    research_conclusion = build_research_conclusion(
        trade,
        decision,
        trade_classification,
        prediction_failure,
        since_entry,
        max_opportunity,
        counterfactuals,
        regime_badges,
    )
    if decision.get("prediction"):
        prob = decision["prediction"].get("probability_success_pct")
        decision["prediction"]["confidence_tier"] = confidence_tier(
            decision["prediction"].get("confidence_pct") or prob,
        )

    replay_doc_partial = {
        "trade": trade,
        "since_entry": since_entry,
        "trade_verdict": trade_verdict,
        "regime_badges": regime_badges,
        "similar_trades": similar_trades,
        "counterfactuals": counterfactuals,
    }
    trade_explanation = explain_trade_narrative(replay_doc_partial)

    return {
        "ok": True,
        "trade": trade,
        "decision": decision,
        "events": events,
        "feature_series": feature_series,
        "feature_alerts": feature_alerts,
        "mini_shap": mini_shap,
        "regime_badges": regime_badges,
        "price_paths": {"spot": spot_path, "premium": premium_path},
        "pnl_path": pnl_path,
        "since_entry": since_entry,
        "maximum_opportunity": max_opportunity,
        "rule_timeline": rule_timeline,
        "trade_verdict": trade_verdict,
        "similar_trades": similar_trades,
        "counterfactuals": counterfactuals,
        "feature_time_machine": feature_time_machine,
        "trade_explanation": trade_explanation,
        "prediction_failure": prediction_failure,
        "trade_classification": trade_classification,
        "research_conclusion": research_conclusion,
        "exit_analysis": exit_analysis,
        "target_price": round(target_price, 4) if target_price else None,
        "stop_price": round(stop_price, 4) if stop_price else None,
    }


def _event_order(event_type: str | None) -> int:
    order = {
        "prediction": 0,
        "strategy_entry": 1,
        "premium_tick": 2,
        "premium_drop": 2,
        "stop_hit": 3,
        "target_hit": 4,
        "trade_exit": 5,
    }
    return order.get(str(event_type or ""), 9)


def _observation_bullet(label: str, detail: str = "") -> str:
    text = str(label or "").strip()
    if detail:
        text = f"{text} ({detail})" if "(" not in text else text
    return text


def _bullet_to_tag(text: str) -> str:
    return text.split("(")[0].strip().lower()[:48]


def explain_trade_narrative(replay_doc: dict[str, Any]) -> dict[str, Any]:
    """Produce a paragraph explanation for Explain This Trade."""
    from .trade_narrative import explain_trade_narrative as _explain

    return _explain(replay_doc)


def generate_trade_observation(replay_doc: dict[str, Any]) -> dict[str, Any]:
    """Build a research notebook observation from trade replay intelligence."""
    trade = replay_doc.get("trade") or {}
    verdict = replay_doc.get("trade_verdict") or {}
    since = replay_doc.get("since_entry") or {}
    decision = replay_doc.get("decision") or {}
    pred = decision.get("prediction") or {}
    feature_alerts = replay_doc.get("feature_alerts") or []
    regime_badges = replay_doc.get("regime_badges") or []
    max_opp = replay_doc.get("maximum_opportunity") or {}

    outcome = str(verdict.get("outcome") or "")
    net_pnl = _num(trade.get("net_pnl"))
    if "FAILED" in outcome or (net_pnl is not None and net_pnl < 0):
        opener = "Trade lost because"
    elif "SUCCEEDED" in outcome or (net_pnl is not None and net_pnl > 0):
        opener = "Trade won because"
    else:
        opener = "Trade observation"

    bullets: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        key = text.lower()
        if key not in seen:
            seen.add(key)
            bullets.append(text)

    for section in (verdict.get("model") or {}, verdict.get("strategy") or {}):
        for reason in section.get("reasons") or []:
            if reason.get("passed"):
                continue
            _add(_observation_bullet(str(reason.get("label") or ""), str(reason.get("detail") or "")))

    metrics = since.get("metrics") or {}
    entry = since.get("entry") or {}
    prem_chg = metrics.get("premium")
    if prem_chg is not None and prem_chg < -1.5:
        _add(f"Premium decay ({prem_chg:+.2f}%)")
    iv_chg = metrics.get("iv")
    if iv_chg is not None and iv_chg < -1.0:
        _add(f"IV contraction ({iv_chg:+.2f}%)")
    theta = _num(entry.get("theta"))
    if theta is not None and abs(theta) > 0.45:
        _add(f"High theta ({theta:+.3f})")
    conf = pred.get("confidence_pct")
    if conf is not None and conf < 70:
        _add(f"Low confidence ({conf:.0f}%)")
    elif conf is None and pred.get("probability_success_pct") is not None:
        prob = float(pred["probability_success_pct"])
        if prob < 60:
            _add(f"Low estimated success ({prob:.0f}%)")

    pred_pct = pred.get("prediction_pct")
    if pred_pct is not None and abs(pred_pct) > 12:
        _add(f"Large prediction ({pred_pct:+.1f}%)")

    eff = max_opp.get("capture_efficiency_pct")
    max_possible = max_opp.get("maximum_possible")
    if max_possible is not None and max_possible <= 0:
        _add("No favorable opportunity during hold")
    elif eff is not None and eff < 25:
        _add(f"Poor capture efficiency ({eff:.0f}%)")

    for alert in feature_alerts:
        if alert.get("severity") not in ("HIGH", "MEDIUM"):
            continue
        _add(f"{alert.get('label')} {alert.get('drift_text', 'drift')} {alert.get('delta_label', '')}".strip())

    for badge in regime_badges[:4]:
        tag = str(badge)
        if tag.lower() not in seen:
            _add(tag)

    if not bullets:
        if "SUCCEEDED" in outcome:
            _add("Strategy execution captured the move")
        elif "FAILED" in outcome:
            _add("Exit without meaningful favorable excursion")
        else:
            _add("No dominant factor identified")

    body_lines = [opener + ":"] + [f"• {b}" for b in bullets]
    token = trade.get("token") or "trade"
    trade_id = str(trade.get("trade_id") or "")[:8]
    title = f"{token} — {opener.replace(' because', '')}"
    reason = opener.replace(" because", "")
    if bullets:
        reason = f"{reason} — {bullets[0].split('(')[0].strip()}"

    tags = [_bullet_to_tag(b) for b in bullets[:8]]
    for badge in regime_badges[:3]:
        t = str(badge).lower()
        if t not in tags:
            tags.append(t)

    return {
        "title": title,
        "body": "\n".join(body_lines),
        "reason": reason,
        "tags": tags,
        "bullets": bullets,
        "opener": opener,
    }
