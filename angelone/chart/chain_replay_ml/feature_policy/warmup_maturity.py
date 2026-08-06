"""Dataset maturity timeline — ready feature counts across all policy features."""

from __future__ import annotations

import re
from typing import Any

from .types import FeatureCategory


def _anchor_label(anchor: str | None) -> str | None:
    if not anchor:
        return None
    parts = str(anchor).split(".")
    if len(parts) >= 3 and parts[2].startswith("ema"):
        series = parts[1].upper() if len(parts) > 1 else ""
        ema = f"EMA{parts[2].replace('ema', '')}"
        return f"{series} {ema}" if series else ema
    return anchor.replace("__roll.", "").replace(".", " ").upper()


_SKIP_ROW_READY_PCT = 95.0
_CATEGORY_KEYS = ("raw", "rolling", "lookback", "derived", "cumulative", "target", "metadata")
_MILESTONE_DETAIL_SAMPLES = frozenset({1, 9, 20, 50, 100, 200, 300})


def _category_id(meta: Any) -> str:
    cat = getattr(meta, "feature_category", None)
    if cat is None:
        return "raw"
    val = cat.value if hasattr(cat, "value") else str(cat)
    return str(val).lower()


def _controller_rows(eng: Any) -> list[dict[str, Any]]:
    snap = eng.readiness_snapshot()
    rows: list[dict[str, Any]] = []
    for key in sorted(snap.keys()):
        if not str(key).startswith("__roll."):
            continue
        entry = snap.get(key) or {}
        label = _anchor_label(key) or key
        rows.append({
            "key": key,
            "label": label,
            "ready": bool(entry.get("ready")),
            "samples_seen": int(entry.get("samples_seen") or 0),
            "period": _ema_period_from_label(label),
        })
    rows.sort(key=lambda r: (r.get("period") or 0, r.get("label") or ""))
    return rows


def _ema_period_from_label(label: str) -> int:
    m = re.search(r"EMA(\d+)", str(label or ""), re.I)
    return int(m.group(1)) if m else 0


def _blocking_controller_labels(
    eng: Any,
    reg: Any,
    not_ready_names: list[str],
) -> list[str]:
    """Rolling controllers that block at least one not-ready feature."""
    blocking_keys: set[str] = set()
    for name in not_ready_names:
        meta = reg.get(name)
        if not meta:
            continue
        anchor = getattr(meta, "policy_anchor", None)
        if anchor and not eng.is_ready(anchor):
            blocking_keys.add(str(anchor))
        for dep in getattr(meta, "dependencies", ()) or ():
            dep_id = str(dep)
            if dep_id.startswith("__roll.") and not eng.is_ready(dep_id):
                blocking_keys.add(dep_id)
    labels: list[str] = []
    for key in sorted(blocking_keys):
        label = _anchor_label(key)
        if label:
            labels.append(label)
    return labels


def explain_feature_readiness(
    feature_name: str,
    *,
    eng: Any,
    reg: Any,
    sample: int,
    sampling_interval_sec: float,
) -> dict[str, Any]:
    """Why *feature_name* is ready / not ready at current engine state."""
    meta = reg.get(feature_name)
    if not meta:
        return {"ok": False, "feature": feature_name}
    ready = eng.is_ready(feature_name)
    cat = _category_id(meta)
    needed = int(getattr(meta, "effective_warmup_samples", 0) or getattr(meta, "intrinsic_warmup_samples", 0) or 0)
    sec = int(getattr(meta, "intrinsic_warmup_sec", 0) or 0)
    if sec > 0 and sampling_interval_sec > 0:
        needed = max(needed, int(sec / sampling_interval_sec))
    snap = eng.readiness_snapshot()
    st = snap.get(feature_name) or {}
    seen = int(st.get("samples_seen") or 0)
    blocking: list[str] = []
    anchor = getattr(meta, "policy_anchor", None)
    if anchor and not eng.is_ready(anchor):
        blocking.append(_anchor_label(anchor) or anchor)
    for dep in getattr(meta, "dependencies", ()) or ():
        dep_id = str(dep)
        if dep_id.startswith("__roll.") and not eng.is_ready(dep_id):
            blocking.append(_anchor_label(dep_id) or dep_id)
    reason = ""
    if ready:
        reason = "Ready — policy warm-up complete."
    elif cat == "lookback" and needed > 0:
        reason = (
            f"Lookback warm-up: needs {needed} samples "
            f"({sec or int(needed * sampling_interval_sec)}s @ {sampling_interval_sec:g}s), "
            f"at sample {sample} only {seen}."
        )
    elif blocking:
        reason = f"Blocked by: {', '.join(blocking)}"
    elif needed > 0:
        reason = f"Needs {needed} samples; at sample {sample} only {seen}."
    else:
        reason = "Not ready — dependencies or inputs incomplete."
    return {
        "ok": True,
        "feature": feature_name,
        "ready": ready,
        "category": cat,
        "needed_samples": needed,
        "samples_seen": seen,
        "ready_at_sample": needed if needed > 0 else 1,
        "blocking_controllers": blocking,
        "reason": reason,
    }


def snapshot_maturity(
    eng: Any,
    reg: Any,
    feature_names: list[str],
    *,
    sample: int,
    time: str,
    include_detail: bool = False,
) -> dict[str, Any]:
    """Count ready features by category at current engine state."""
    names = [n for n in feature_names if n and not str(n).startswith("__roll.")]
    cats: dict[str, int] = {k: 0 for k in _CATEGORY_KEYS}
    ready_names: list[str] = []
    not_ready_names: list[str] = []
    for name in names:
        meta = reg.get(name)
        if not meta:
            continue
        cat = _category_id(meta)
        if eng.is_ready(name):
            ready_names.append(name)
            if cat in cats:
                cats[cat] += 1
        else:
            not_ready_names.append(name)

    total = len(names)
    ready_n = len(ready_names)
    not_ready_n = max(total - ready_n, 0)
    ready_pct = round(ready_n / max(total, 1) * 100.0, 1)
    controllers = _controller_rows(eng)
    not_ready_ctrl = [c["label"] for c in controllers if not c.get("ready")]
    blocking_ctrl = _blocking_controller_labels(eng, reg, not_ready_names)
    not_ready_lookback = [
        n for n in not_ready_names
        if reg.get(n) and _category_id(reg.get(n)) == FeatureCategory.LOOKBACK.value
    ]
    derived_blocked: list[str] = []
    for name in not_ready_names:
        meta = reg.get(name)
        if not meta or _category_id(meta) != FeatureCategory.DERIVED.value:
            continue
        anchor = getattr(meta, "policy_anchor", None)
        if anchor and not eng.is_ready(anchor):
            derived_blocked.append(name)

    row: dict[str, Any] = {
        "sample": sample,
        "time": time,
        "ready": ready_n,
        "not_ready": not_ready_n,
        "ready_pct": ready_pct,
        "raw": cats["raw"],
        "rolling": cats["rolling"],
        "lookback": cats["lookback"],
        "derived": cats["derived"],
        "cumulative": cats["cumulative"],
        "total": total,
        "skip_row": ready_pct < _SKIP_ROW_READY_PCT,
        "derived_blocked_count": len(derived_blocked),
    }

    detail: dict[str, Any] = {
        "controllers": controllers,
        "not_ready_controllers": not_ready_ctrl,
        "blocking_controllers": blocking_ctrl,
        "ready_controllers": [c["label"] for c in controllers if c.get("ready")],
        "derived_blocked_count": len(derived_blocked),
        "not_ready_lookback_count": len(not_ready_lookback),
    }
    if include_detail or sample in _MILESTONE_DETAIL_SAMPLES or sample <= 1:
        detail["not_ready_derived"] = derived_blocked[:80]
        detail["not_ready_features"] = not_ready_names[:80]
        detail["not_ready_lookback"] = not_ready_lookback[:40]
        detail["ready_features"] = ready_names[:40]
    row["detail"] = detail
    return row


def milestone_table_rows(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick milestone samples for compact table display."""
    if not timeline:
        return []
    n = len(timeline)
    picks = {1, 9, 20, 50, 100, 200, 300, n}
    step = max(1, n // 12)
    for i in range(step, n, step):
        picks.add(i)
    picks.add(n)
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in timeline:
        s = int(row.get("sample") or 0)
        if s in picks and s not in seen:
            seen.add(s)
            out.append(row)
    out.sort(key=lambda r: int(r.get("sample") or 0))
    return out


def maturity_chart_points(timeline: list[dict[str, Any]], *, buckets: int = 12) -> list[tuple[int, int]]:
    if not timeline:
        return []
    n = len(timeline)
    if n <= buckets:
        return [(int(r.get("sample", 0)), int(r.get("ready") or 0)) for r in timeline]
    out: list[tuple[int, int]] = []
    for b in range(buckets):
        start = int(b * n / buckets)
        end = max(start + 1, int((b + 1) * n / buckets))
        chunk = timeline[start:end]
        out.append((int(chunk[-1].get("sample", 0)), int(chunk[-1].get("ready") or 0)))
    return out


def maturity_buckets(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average ready % over sample ranges."""
    if not timeline:
        return []
    ranges = [
        (1, 50, "0–50"),
        (51, 100, "50–100"),
        (101, 200, "100–200"),
        (201, 10_000_000, "200+"),
    ]
    out: list[dict[str, Any]] = []
    for lo, hi, label in ranges:
        chunk = [
            r for r in timeline
            if lo <= int(r.get("sample") or 0) <= hi
        ]
        if not chunk:
            continue
        avg = round(sum(float(r.get("ready_pct") or 0) for r in chunk) / len(chunk), 1)
        out.append({"label": label, "samples": len(chunk), "avg_ready_pct": avg})
    return out


def build_maturity_summary(
    timeline: list[dict[str, Any]],
    *,
    feature_total: int,
) -> dict[str, Any]:
    if not timeline:
        return {}
    last = timeline[-1]
    peak = max(timeline, key=lambda r: float(r.get("ready_pct") or 0))
    return {
        "feature_total": feature_total,
        "last_sample": int(last.get("sample") or 0),
        "last_ready": int(last.get("ready") or 0),
        "last_ready_pct": float(last.get("ready_pct") or 0),
        "peak_ready_pct": float(peak.get("ready_pct") or 0),
        "peak_sample": int(peak.get("sample") or 0),
        "buckets": maturity_buckets(timeline),
        "chart_points": maturity_chart_points(timeline),
        "milestone_rows": milestone_table_rows(timeline),
    }
