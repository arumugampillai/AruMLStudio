"""Plain-text formatters for Feature Policy views (Tk)."""

from __future__ import annotations

import csv
import io
import re
from typing import Any

CATEGORY_LABELS = {
    "raw": "Raw",
    "rolling": "Rolling",
    "lookback": "Lookback",
    "cumulative": "Cumulative",
    "derived": "Derived",
    "target": "Target",
    "metadata": "Metadata",
}

LIFECYCLE_LABELS = {
    "tick": "Tick",
    "session": "Session",
    "sliding_window": "Sliding Window",
    "day": "Day",
}

CATEGORY_FILTER_OPTIONS: list[tuple[str, str]] = [("all", "All")] + list(CATEGORY_LABELS.items())

_EMA_IN_NAME = re.compile(r"(?:ltp_ema|spot_ema|ema)(\d+)", re.I)
_VIRTUAL_DEP = frozenset({"timestamp", "token", "symbol", "feature_grid"})


def category_label(category: str | None) -> str:
    cat = str(category or "").strip().lower()
    if not cat or cat == "—":
        return "—"
    return CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())


def category_filter_key(combo_val: str) -> str:
    return str(combo_val or "all|All").split("|", 1)[0].strip().lower()


def normalize_policy(feature: dict[str, Any]) -> dict[str, Any]:
    """Merge catalog row + nested policy block into one policy dict."""
    if not feature:
        return {}
    pol = feature.get("policy")
    if isinstance(pol, dict) and pol:
        base = dict(pol)
    else:
        base = {}
    for key in (
        "feature_category", "lifecycle", "dependencies", "intrinsic_warmup_samples",
        "effective_warmup_samples", "effective_warmup_inherited", "policy_anchor",
        "reset_on_gap", "gap_sensitive", "rolling_type", "warmup_mode",
        "intrinsic_warmup_sec", "formula_version", "policy_version",
    ):
        if key not in base and feature.get(key) is not None:
            base[key] = feature.get(key)
    if "name" not in base:
        base["name"] = feature.get("name")
    return base


def anchor_display_name(anchor: str | None) -> str | None:
    if not anchor:
        return None
    parts = str(anchor).split(".")
    if len(parts) >= 3 and parts[2].startswith("ema"):
        try:
            period = parts[2].replace("ema", "")
            return f"EMA{period}"
        except ValueError:
            pass
    if len(parts) >= 3 and parts[2].startswith("std"):
        return f"STD{parts[2].replace('std', '')}"
    return anchor.replace("__roll.", "").replace(".", " ").upper()


def dep_display_name(dep: str) -> str:
    if dep.startswith("__roll."):
        return anchor_display_name(dep) or dep
    if dep in ("ltp", "spot"):
        return dep.upper()
    return dep.replace("_", " ").title()


def warmup_samples(pol: dict[str, Any], *, sampling_interval_sec: float = 0.0) -> int:
    samples = int(pol.get("effective_warmup_samples") or pol.get("intrinsic_warmup_samples") or 0)
    sec = int(pol.get("intrinsic_warmup_sec") or 0)
    if sec > 0 and sampling_interval_sec > 0:
        samples = max(samples, int(sec / sampling_interval_sec))
    return samples


def is_inherited_warmup(pol: dict[str, Any]) -> bool:
    if pol.get("effective_warmup_inherited"):
        return True
    if pol.get("policy_anchor"):
        return True
    cat = str(pol.get("feature_category") or "").lower()
    return cat == "derived" and warmup_samples(pol) > 0


def inherited_warmup_source(
    pol: dict[str, Any],
    *,
    feature_name: str = "",
    features_by_name: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    ctrl = resolve_warmup_controller(
        pol,
        feature_name=feature_name or pol.get("name") or "",
        features_by_name=features_by_name or {},
    )
    if ctrl["controller_label"] and ctrl["controller_samples"] > 0:
        if ctrl["is_self"]:
            return f"{ctrl['controller_label']} ({ctrl['controller_samples']})"
        return f"{ctrl['controller_label']} ({ctrl['controller_samples']})"
    anchor = pol.get("policy_anchor")
    if anchor:
        label = anchor_display_name(anchor)
        samples = warmup_samples(pol)
        if label and samples:
            return f"{label} ({samples})"
        if label:
            return label
    m = _EMA_IN_NAME.search(feature_name or pol.get("name") or "")
    if m and is_inherited_warmup(pol):
        period = m.group(1)
        return f"EMA{period} ({period})"
    return None


def warmup_tier(samples: int) -> str:
    if samples <= 0:
        return "none"
    if samples <= 20:
        return "low"
    if samples <= 100:
        return "mid"
    return "high"


def warmup_tier_icon(samples: int) -> str:
    tier = warmup_tier(samples)
    if tier == "low":
        return "🟢"
    if tier == "mid":
        return "🟡"
    if tier == "high":
        return "🔴"
    return "·"


def format_warmup_duration(samples: int, *, sampling_interval_sec: float = 10.0) -> str:
    if samples <= 0:
        return "0 sec"
    sec = samples * max(sampling_interval_sec, 0.001)
    if sec < 60:
        return f"≈{sec:.0f} sec"
    if sec < 3600:
        mins = sec / 60.0
        return f"≈{mins:.1f} min"
    return f"≈{sec / 3600.0:.1f} h"


def dep_warmup_samples(dep_id: str, features_by_name: dict[str, dict[str, Any]]) -> int:
    """Warm-up sample count for a single dependency id."""
    if dep_id in _VIRTUAL_DEP or dep_id.startswith("feature_grid"):
        return 0
    feat = features_by_name.get(dep_id)
    if feat:
        return warmup_samples(normalize_policy(feat))
    if dep_id.startswith("__roll."):
        parts = dep_id.split(".")
        if len(parts) >= 3:
            tail = parts[2]
            if tail.startswith("ema"):
                try:
                    return int(tail.replace("ema", ""))
                except ValueError:
                    pass
            if tail.startswith("std"):
                try:
                    return int(tail.replace("std", ""))
                except ValueError:
                    pass
    m = _EMA_IN_NAME.search(dep_id)
    if m:
        return int(m.group(1))
    return 0


def rolling_feature_label(feature_name: str, pol: dict[str, Any] | None = None) -> str:
    pol = pol or {}
    anchor = pol.get("policy_anchor")
    if anchor:
        label = anchor_display_name(anchor)
        if label:
            return label
    m = _EMA_IN_NAME.search(feature_name)
    if m:
        return f"EMA{m.group(1)}"
    return dep_display_name(feature_name)


def collect_dependency_warmups(
    pol: dict[str, Any],
    *,
    feature_name: str = "",
    features_by_name: dict[str, dict[str, Any]],
) -> list[tuple[str, str, int]]:
    """(display_label, dep_id, samples) for each warm-up-relevant dependency."""
    entries: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for dep_id in raw_dependency_ids(pol):
        label = dep_display_name(dep_id)
        if label in seen:
            continue
        seen.add(label)
        entries.append((label, dep_id, dep_warmup_samples(dep_id, features_by_name)))
    intrinsic = int(pol.get("intrinsic_warmup_samples") or 0)
    cat = str(pol.get("feature_category") or "").lower()
    if intrinsic > 0 and cat == "rolling":
        label = rolling_feature_label(feature_name or pol.get("name") or "", pol)
        fname = feature_name or pol.get("name") or ""
        entries = [e for e in entries if e[1] != fname]
        if label not in {e[0] for e in entries}:
            entries.append((label, fname or label, intrinsic))
        else:
            entries = [
                (label, dep_id, max(samples, intrinsic) if dep_id == feature_name else samples)
                for label, dep_id, samples in entries
            ]
    entries.sort(key=lambda row: (-row[2], row[0]))
    return entries


def resolve_warmup_controller(
    pol: dict[str, Any],
    *,
    feature_name: str = "",
    features_by_name: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Identify the governing warm-up dependency (bottleneck) for a feature."""
    features_by_name = features_by_name or {}
    name = feature_name or pol.get("name") or ""
    effective = warmup_samples(pol)
    intrinsic = int(pol.get("intrinsic_warmup_samples") or 0)
    entries = collect_dependency_warmups(
        pol, feature_name=name, features_by_name=features_by_name,
    )

    if not entries and effective <= 0:
        return {
            "controller_label": None,
            "controller_id": None,
            "controller_samples": 0,
            "effective_samples": 0,
            "entries": [],
            "reason": "No warm-up required",
            "is_self": False,
        }

    max_samples = max((e[2] for e in entries), default=0)
    max_samples = max(max_samples, effective, intrinsic)

    if max_samples <= 0:
        return {
            "controller_label": None,
            "controller_id": None,
            "controller_samples": 0,
            "effective_samples": effective,
            "entries": entries,
            "reason": "No warm-up required",
            "is_self": False,
        }

    governors = [e for e in entries if e[2] == max_samples]
    if not governors and max_samples > 0:
        label = rolling_feature_label(name, pol)
        governors = [(label, name, max_samples)]
        entries.append((label, name, max_samples))

    anchor = pol.get("policy_anchor")
    controller = governors[0]
    if anchor:
        anchor_label = dep_display_name(anchor)
        for candidate in governors:
            if candidate[1] == anchor or candidate[0] == anchor_label:
                controller = candidate
                break

    ctrl_label, ctrl_id, ctrl_samples = controller
    is_self = ctrl_id == name or (
        intrinsic > 0 and str(pol.get("feature_category") or "").lower() == "rolling"
        and ctrl_samples == intrinsic
    )

    if is_self:
        reason = f"Intrinsic rolling warm-up ({ctrl_samples} samples)"
    elif len(governors) > 1:
        reason = f"Tied longest warm-up ({ctrl_samples} samples)"
    else:
        reason = f"Longest dependency warm-up ({ctrl_samples} samples)"

    return {
        "controller_label": ctrl_label,
        "controller_id": ctrl_id,
        "controller_samples": ctrl_samples,
        "effective_samples": effective,
        "entries": entries,
        "reason": reason,
        "is_self": is_self,
    }


def format_warmup_controller_section(
    pol: dict[str, Any],
    *,
    feature_name: str = "",
    features_by_name: dict[str, dict[str, Any]] | None = None,
    sampling_interval_sec: float = 10.0,
) -> str:
    features_by_name = features_by_name or {}
    name = feature_name or pol.get("name") or "—"
    ctrl = resolve_warmup_controller(
        pol, feature_name=name, features_by_name=features_by_name,
    )
    effective = int(ctrl["effective_samples"])
    lines = ["Warm-up Controller", "-" * 40]

    if ctrl["controller_label"]:
        icon = warmup_tier_icon(int(ctrl["controller_samples"]))
        lines.append(f"  {icon} {ctrl['controller_label']}")
        lines.append("")
        lines.append("Reason:")
        lines.append(f"  {ctrl['reason']}")
        if not ctrl["is_self"] and int(ctrl["controller_samples"]) > 0 and (
            str(pol.get("feature_category") or "").lower() == "derived"
        ):
            lines.extend([
                "",
                "Controlled by:",
                f"  {icon} {ctrl['controller_label']}",
                "       ↓",
                f"  {name}",
            ])
    else:
        lines.append("  — (none)")
        lines.append("")
        lines.append("Reason:")
        lines.append("  No rolling dependencies")

    entries = ctrl["entries"]
    show_table = any(e[2] > 0 for e in entries)
    if show_table and entries:
        lines.extend(["", "Dependency Warm-ups", "-" * 40])
        label_w = max(len(e[0]) for e in entries)
        sorted_entries = sorted(entries, key=lambda row: (-row[2], row[0]))
        for label, _dep_id, samples in sorted_entries:
            governing = (
                label == ctrl["controller_label"]
                and samples == int(ctrl["controller_samples"])
            )
            mark = "  ← Governing" if governing else ""
            lines.append(f"  {label:<{label_w}}  {samples:>4}{mark}")
        if len(entries) > 1:
            sample_vals = ", ".join(str(e[2]) for e in sorted_entries)
            lines.append("")
            lines.append(
                f"  Effective Warm-up = MAX({sample_vals}) = {effective}",
            )

    dur = format_warmup_duration(effective, sampling_interval_sec=sampling_interval_sec)
    lines.extend(["", "Effective Warm-up:"])
    if effective > 0:
        lines.append(f"  {effective} samples ({dur} @ {sampling_interval_sec:g}s)")
    else:
        lines.append("  0 — ready immediately once raw inputs exist")
    return "\n".join(lines)


def format_warmup_cell(
    pol: dict[str, Any],
    *,
    feature_name: str = "",
    sampling_interval_sec: float = 10.0,
    features_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    samples = warmup_samples(pol)
    if samples <= 0:
        return "—"
    icon = warmup_tier_icon(samples)
    ctrl = resolve_warmup_controller(
        pol,
        feature_name=feature_name,
        features_by_name=features_by_name or {},
    )
    dur = format_warmup_duration(samples, sampling_interval_sec=sampling_interval_sec)
    if ctrl["controller_label"] and not ctrl["is_self"] and is_inherited_warmup(pol):
        return f"{icon} {samples} ← {ctrl['controller_label']} ({dur})"
    return f"{icon} {samples} ({dur})"


def format_warmup_detail(
    pol: dict[str, Any],
    *,
    feature_name: str = "",
    sampling_interval_sec: float = 10.0,
    features_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    return format_warmup_controller_section(
        pol,
        feature_name=feature_name,
        features_by_name=features_by_name,
        sampling_interval_sec=sampling_interval_sec,
    )


def display_dependencies(pol: dict[str, Any]) -> list[str]:
    deps = list(pol.get("dependencies") or [])
    out: list[str] = []
    seen: set[str] = set()
    anchor = pol.get("policy_anchor")
    if anchor:
        label = dep_display_name(anchor)
        if label not in seen:
            out.append(label)
            seen.add(label)
    for dep in deps:
        if dep in _VIRTUAL_DEP or dep.startswith("feature_grid"):
            continue
        if dep.startswith("__roll."):
            label = dep_display_name(dep)
        else:
            label = dep_display_name(dep)
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def raw_dependency_ids(pol: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for dep in pol.get("dependencies") or []:
        if dep in _VIRTUAL_DEP or dep.startswith("feature_grid"):
            continue
        if dep not in seen:
            seen.add(dep)
            out.append(dep)
    anchor = pol.get("policy_anchor")
    if anchor and anchor not in seen:
        out.append(anchor)
    return out


def lookup_keys_for_feature(name: str, pol: dict[str, Any]) -> set[str]:
    """Keys for reverse dependency lookup (name, anchor, EMA period aliases)."""
    keys = {name}
    anchor = pol.get("policy_anchor")
    if anchor:
        keys.add(anchor)
    m = _EMA_IN_NAME.search(name)
    if m:
        period = m.group(1)
        keys.add(f"EMA{period}")
        keys.add(f"__roll.ltp.ema{period}")
        keys.add(f"__roll.spot.ema{period}")
    label = anchor_display_name(anchor) if anchor else None
    if label:
        keys.add(label)
    return keys


def build_used_by_index(
    feature_names: list[str],
    features_by_name: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for feat_name in feature_names:
        if feat_name.startswith("__roll."):
            continue
        pol = normalize_policy(features_by_name.get(feat_name) or {"name": feat_name})
        for dep in raw_dependency_ids(pol):
            index.setdefault(dep, []).append(feat_name)
    for users in index.values():
        users.sort()
    return index


def resolve_used_by(
    name: str,
    pol: dict[str, Any],
    used_by_index: dict[str, list[str]],
) -> list[str]:
    users: list[str] = []
    seen: set[str] = set()
    for key in lookup_keys_for_feature(name, pol):
        for u in used_by_index.get(key, []):
            if u != name and u not in seen:
                seen.add(u)
                users.append(u)
    return sorted(users)


def _rolling_parent_ref(
    pol: dict[str, Any],
    features_by_name: dict[str, dict[str, Any]],
) -> tuple[str, str] | None:
    anchor = pol.get("policy_anchor")
    if anchor:
        return ("anchor", anchor)
    rolling_deps: list[str] = []
    for dep in pol.get("dependencies") or []:
        if dep in _VIRTUAL_DEP or dep.startswith("__roll."):
            continue
        dep_pol = normalize_policy(features_by_name.get(dep) or {"name": dep})
        cat = str(dep_pol.get("feature_category") or "").lower()
        if cat == "rolling" or warmup_samples(dep_pol) > 0 or dep_pol.get("policy_anchor"):
            rolling_deps.append(dep)
    if len(rolling_deps) == 1:
        return ("dep", rolling_deps[0])
    return None


def _is_hierarchical_dependencies(
    pol: dict[str, Any],
    features_by_name: dict[str, dict[str, Any]],
) -> bool:
    return _rolling_parent_ref(pol, features_by_name) is not None


def format_dependency_display(
    feature_name: str,
    pol: dict[str, Any],
    features_by_name: dict[str, dict[str, Any]],
) -> str:
    name = feature_name or pol.get("name") or "—"
    deps = display_dependencies(pol)
    if not deps:
        return "  (none)"

    if not _is_hierarchical_dependencies(pol, features_by_name):
        lines = [f"  {name}"]
        for i, d in enumerate(deps):
            branch = "└── " if i == len(deps) - 1 else "├── "
            lines.append(f"  {branch}{d}")
        return "\n".join(lines)

    nodes: list[str] = [name]
    visited: set[str] = {name}
    cur_pol = pol
    while True:
        parent = _rolling_parent_ref(cur_pol, features_by_name)
        if not parent:
            break
        kind, ref = parent
        if kind == "anchor":
            label = dep_display_name(ref)
            if label not in nodes:
                nodes.append(label)
            for dep in cur_pol.get("dependencies") or []:
                if dep in _VIRTUAL_DEP or dep.startswith("__roll."):
                    continue
                dep_pol = normalize_policy(features_by_name.get(dep) or {"name": dep})
                dcat = str(dep_pol.get("feature_category") or "").lower()
                if dcat in ("rolling", "derived"):
                    continue
                label = dep_display_name(dep)
                if label not in nodes:
                    nodes.append(label)
            break
        dep_name = ref
        if dep_name in visited:
            break
        visited.add(dep_name)
        nodes.append(dep_display_name(dep_name))
        cur_pol = normalize_policy(features_by_name.get(dep_name) or {"name": dep_name})
        if _rolling_parent_ref(cur_pol, features_by_name) is None:
            for dep in cur_pol.get("dependencies") or []:
                if dep in _VIRTUAL_DEP or dep.startswith("__roll."):
                    continue
                dep_pol = normalize_policy(features_by_name.get(dep) or {"name": dep})
                dcat = str(dep_pol.get("feature_category") or "").lower()
                if dcat not in ("rolling", "derived"):
                    label = dep_display_name(dep)
                    if label not in nodes:
                        nodes.append(label)
            break

    lines = [f"  {nodes[0]}"]
    for node in nodes[1:]:
        lines.append("  ↓")
        lines.append(f"  {node}")
    return "\n".join(lines)


def rolling_parent_label(
    pol: dict[str, Any],
    *,
    feature_name: str = "",
    features_by_name: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    ctrl = resolve_warmup_controller(
        pol,
        feature_name=feature_name,
        features_by_name=features_by_name or {},
    )
    if ctrl["controller_label"] and not ctrl["is_self"]:
        return ctrl["controller_label"]
    features_by_name = features_by_name or {}
    parent = _rolling_parent_ref(pol, features_by_name)
    if not parent:
        return None
    kind, ref = parent
    if kind == "anchor":
        return anchor_display_name(ref) or inherited_warmup_source(
            pol, feature_name=feature_name, features_by_name=features_by_name,
        )
    dep_pol = normalize_policy(features_by_name.get(ref) or {"name": ref})
    return dep_display_name(ref) or inherited_warmup_source(
        dep_pol, feature_name=ref, features_by_name=features_by_name,
    )


def format_current_policy(
    pol: dict[str, Any],
    *,
    feature_name: str = "",
    features_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    cat = str(pol.get("feature_category") or "").lower()
    features_by_name = features_by_name or {}
    ctrl = resolve_warmup_controller(
        pol, feature_name=feature_name, features_by_name=features_by_name,
    )
    if cat == "rolling":
        return (
            "  If warm-up incomplete or after gap reset\n"
            "  → Output NULL"
        )
    if cat == "derived" and ctrl["controller_label"] and not ctrl["is_self"]:
        return (
            f"  If {ctrl['controller_label']} not ready\n"
            "  → Output NULL"
        )
    return (
        "  No rolling dependencies.\n"
        "  Always available once raw inputs exist."
    )


def format_readiness_graph(
    pol: dict[str, Any],
    *,
    sampling_interval_sec: float = 10.0,
) -> str:
    samples = warmup_samples(pol)
    if samples <= 0:
        return ""
    cat = str(pol.get("feature_category") or "").lower()
    if cat not in ("rolling", "derived", "lookback") and not pol.get("policy_anchor"):
        return ""

    lines = ["Readiness Graph", "-" * 40]
    milestones = [0.0, 0.25, 0.5, 0.75, 1.0]
    seen_cp: set[int] = set()
    for frac in milestones:
        cp = int(round(samples * frac))
        if cp in seen_cp:
            continue
        seen_cp.add(cp)
        t_min = (cp * sampling_interval_sec) / 60.0
        ready = cp >= samples
        mark = "✅" if ready else "❌"
        if t_min < 1 and cp > 0:
            lines.append(f"  {t_min * 60:.0f} sec   {mark}")
        elif cp == 0:
            lines.append(f"  0 min   {mark}")
        else:
            lines.append(f"  {t_min:.0f} min   {mark}")

    bar = "■" * 10
    lines.extend([
        "",
        "Warm-up",
        f"  [{bar}]",
        f"  Ready after {samples} / {samples} samples",
        f"  ({format_warmup_duration(samples, sampling_interval_sec=sampling_interval_sec)})",
    ])
    return "\n".join(lines)


def format_used_by_section(
    name: str,
    pol: dict[str, Any],
    used_by_index: dict[str, list[str]],
    *,
    limit: int = 12,
) -> str:
    users = resolve_used_by(name, pol, used_by_index)
    if not users:
        return ""
    lines = ["Used By", "-" * 40]
    for u in users[:limit]:
        lines.append(f"  · {u}")
    if len(users) > limit:
        lines.append(f"  … +{len(users) - limit} more ({len(users)} derived features)")
    elif len(users) > 1:
        lines.append(f"  ({len(users)} downstream features)")
    return "\n".join(lines)


def format_dependency_chain(
    feature_name: str,
    pol: dict[str, Any],
    features_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    return format_dependency_display(
        feature_name, pol, features_by_name or {},
    )


_EMA_LABEL = re.compile(r"^EMA(\d+)$", re.I)


def normalize_ema_label(label: str | None) -> str | None:
    if not label:
        return None
    m = _EMA_LABEL.match(str(label).strip())
    if m:
        return f"EMA{m.group(1)}"
    m = _EMA_IN_NAME.search(str(label))
    if m:
        return f"EMA{m.group(1)}"
    return None


def _ema_period_sort_key(label: str) -> int:
    m = _EMA_LABEL.match(label)
    return int(m.group(1)) if m else 9999


def compute_ema_controller_breakdown(
    feature_names: list[str],
    features_by_name: dict[str, dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Per EMA controller: rolling feature count vs derived dependents."""
    breakdown: dict[str, dict[str, int]] = {}
    for name in feature_names:
        feat = features_by_name.get(name) or {"name": name}
        pol = normalize_policy(feat)
        cat = str(pol.get("feature_category") or "raw").lower()
        ctrl = resolve_warmup_controller(
            pol, feature_name=name, features_by_name=features_by_name,
        )
        ema_key = normalize_ema_label(ctrl.get("controller_label"))
        if not ema_key and cat == "rolling":
            ema_key = normalize_ema_label(rolling_feature_label(name, pol))
        if not ema_key:
            continue
        bucket = breakdown.setdefault(ema_key, {"rolling": 0, "derived": 0})
        if ctrl.get("is_self") or cat == "rolling":
            bucket["rolling"] += 1
        else:
            bucket["derived"] += 1
    return breakdown


def format_rolling_controllers_section(
    ema_controllers: dict[str, dict[str, int]],
) -> list[str]:
    if not ema_controllers:
        return []
    lines = ["", "Rolling Controllers", "-" * 40, ""]
    for ema_key in sorted(ema_controllers.keys(), key=_ema_period_sort_key):
        counts = ema_controllers[ema_key]
        rolling_n = int(counts.get("rolling") or 0)
        derived_n = int(counts.get("derived") or 0)
        if rolling_n <= 0 and derived_n <= 0:
            continue
        lines.append(f"  {ema_key}")
        lines.append(f"    Rolling Features : {rolling_n}")
        lines.append(f"    Derived Features : {derived_n}")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def format_ready_time_compact(samples: int, *, sampling_interval_sec: float = 10.0) -> str:
    if samples <= 0:
        return "—"
    sec = samples * max(sampling_interval_sec, 0.001)
    if sec < 60:
        return f"{sec:.0f} sec"
    if sec < 3600:
        mins = sec / 60.0
        return f"{mins:.0f} min" if abs(mins - round(mins)) < 0.05 else f"{mins:.1f} min"
    return f"{sec / 3600.0:.1f} h"


def format_warmup_time_short(samples: int, *, sampling_interval_sec: float = 10.0) -> str:
    """Compact duration for inline UI labels, e.g. ``10m`` or ``45s``."""
    if samples <= 0:
        return "—"
    sec = samples * max(sampling_interval_sec, 0.001)
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        mins = sec / 60.0
        if abs(mins - round(mins)) < 0.05:
            return f"{int(round(mins))}m"
        return f"{mins:.1f}m"
    return f"{sec / 3600.0:.1f}h"


def format_required_warmup_label(
    feature_names: list[str],
    *,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = 20.0,
) -> str:
    """Single-line warm-up summary for build feature selection UI."""
    names = [
        n for n in dict.fromkeys(feature_names)
        if n and not str(n).startswith("__roll.")
    ]
    if not names:
        return "Required Warm-up: none"
    from chain_replay_ml.feature_policy import build_validation_preview

    preview = build_validation_preview(
        names,
        sampling_interval_sec=float(sampling_interval_sec),
        gap_max_sec=float(gap_max_sec),
    )
    samples = int(preview.get("max_warmup_samples") or 0)
    if samples <= 0:
        return "Required Warm-up: none"
    dur = format_warmup_time_short(samples, sampling_interval_sec=sampling_interval_sec)
    return f"Required Warm-up: {samples} samples ({dur})"


def compute_selection_stats(
    feature_names: list[str],
    features_by_name: dict[str, dict[str, Any]],
    *,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = 20.0,
) -> dict[str, Any]:
    names = list(dict.fromkeys(feature_names))
    counts: dict[str, int] = {k: 0 for k in CATEGORY_LABELS}
    max_warmup = 0
    max_warmup_name = ""
    max_warmup_label = ""
    gap_reset_n = 0
    inherited_n = 0
    inherited_examples: list[str] = []
    ema_controllers: dict[str, dict[str, int]] = {}

    for name in names:
        feat = features_by_name.get(name) or {"name": name}
        pol = normalize_policy(feat)
        cat = str(pol.get("feature_category") or "raw").lower()
        if cat in counts:
            counts[cat] += 1
        samples = warmup_samples(pol)
        if samples > max_warmup:
            max_warmup = samples
            max_warmup_name = name
            ctrl = resolve_warmup_controller(
                pol, feature_name=name, features_by_name=features_by_name,
            )
            if ctrl["controller_label"]:
                max_warmup_label = ctrl["controller_label"]
            else:
                max_warmup_label = dep_display_name(name)
        if pol.get("reset_on_gap") or cat == "rolling":
            gap_reset_n += 1
        if is_inherited_warmup(pol):
            inherited_n += 1
            if len(inherited_examples) < 5:
                src = inherited_warmup_source(
                    pol, feature_name=name, features_by_name=features_by_name,
                )
                inherited_examples.append(f"{name} → {src or 'inherited'}")

    ema_controllers = compute_ema_controller_breakdown(names, features_by_name)
    ema_depend_counts = {
        k: v["rolling"] + v["derived"] for k, v in ema_controllers.items()
    }
    ema_sorted = sorted(ema_depend_counts.items(), key=lambda x: (_ema_period_sort_key(x[0]), -x[1]))
    longest_sec = max_warmup * max(sampling_interval_sec, 0.001)
    return {
        "total": len(names),
        "counts": counts,
        "max_warmup": max_warmup,
        "max_warmup_name": max_warmup_name,
        "max_warmup_label": max_warmup_label or max_warmup_name,
        "longest_ready_time": format_warmup_duration(max_warmup, sampling_interval_sec=sampling_interval_sec),
        "longest_ready_time_compact": format_ready_time_compact(
            max_warmup, sampling_interval_sec=sampling_interval_sec,
        ),
        "longest_ready_sec": longest_sec,
        "gap_reset_n": gap_reset_n,
        "inherited_n": inherited_n,
        "inherited_examples": inherited_examples,
        "ema_controllers": ema_controllers,
        "ema_depend_counts": ema_sorted,
        "gap_max_sec": gap_max_sec,
        "sampling_interval_sec": sampling_interval_sec,
    }


def format_policy_summary_tab(
    feature_names: list[str],
    features_by_name: dict[str, dict[str, Any]],
    *,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = 20.0,
) -> str:
    names = list(dict.fromkeys(feature_names))
    if not names:
        return "No features selected."

    st = compute_selection_stats(
        names, features_by_name,
        sampling_interval_sec=sampling_interval_sec,
        gap_max_sec=gap_max_sec,
    )
    c = st["counts"]
    lines = [
        "Summary",
        "=" * 40,
        "",
        f"  Total Features           {st['total']}",
        f"  Rolling                  {c['rolling']}",
        f"  Derived                  {c['derived']}",
    ]
    if st["max_warmup"]:
        icon = warmup_tier_icon(st["max_warmup"])
        lines.append(
            f"  Longest Warm-up          {icon} {st['max_warmup_label']} "
            f"({st['max_warmup']} samples)",
        )
        lines.append(
            f"  Longest Ready Time       {st['longest_ready_time_compact']}",
        )
    else:
        lines.append("  Longest Warm-up          —")
        lines.append("  Longest Ready Time       —")
    lines.extend(format_rolling_controllers_section(st.get("ema_controllers") or {}))
    return "\n".join(lines)


def format_simulator_policy_sidebar(
    feature_name: str | None,
    feature_names: list[str],
    features_by_name: dict[str, dict[str, Any]],
    *,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = 20.0,
) -> str:
    """Compact policy summary for Warm-up Simulator left panel."""
    names = list(dict.fromkeys(feature_names))
    if not names:
        return "No features loaded."

    st = compute_selection_stats(
        names, features_by_name,
        sampling_interval_sec=sampling_interval_sec,
        gap_max_sec=gap_max_sec,
    )
    c = st["counts"]
    lines = [
        "Policy Summary",
        "-" * 28,
        f"Total Features    {st['total']}",
        f"Rolling           {c['rolling']}",
        f"Derived           {c['derived']}",
    ]
    if st["max_warmup"]:
        icon = warmup_tier_icon(st["max_warmup"])
        lines.append(
            f"Global Longest   {icon} {st['max_warmup_label']} ({st['max_warmup']})",
        )
        lines.append(f"Global Ready     {st['longest_ready_time_compact']}")

    ema_controllers = st.get("ema_controllers") or {}
    if ema_controllers:
        lines.extend(["", "Rolling Controllers", "-" * 28])
        for ema_key in sorted(ema_controllers.keys(), key=_ema_period_sort_key):
            counts = ema_controllers[ema_key]
            rolling_n = int(counts.get("rolling") or 0)
            derived_n = int(counts.get("derived") or 0)
            if rolling_n <= 0 and derived_n <= 0:
                continue
            lines.append(f"{ema_key}")
            lines.append(f"  Rolling : {rolling_n}  Derived : {derived_n}")

    fname = (feature_name or "").strip()
    if fname:
        feat = features_by_name.get(fname) or {"name": fname}
        pol = normalize_policy({**feat, "name": fname})
        cat = category_label(str(pol.get("feature_category") or ""))
        ctrl = resolve_warmup_controller(
            pol, feature_name=fname, features_by_name=features_by_name,
        )
        samples = warmup_samples(pol, sampling_interval_sec=sampling_interval_sec)
        dur = format_warmup_duration(samples, sampling_interval_sec=sampling_interval_sec)
        deps = display_dependencies(pol)
        lines.extend(["", "Selected Feature", "-" * 28, fname, f"Category  {cat}"])
        if ctrl.get("controller_label"):
            icon = warmup_tier_icon(int(ctrl.get("controller_samples") or 0))
            lines.append(f"Controller {icon} {ctrl['controller_label']}")
        lines.append(f"Warm-up   {samples} samples ({dur})")
        lines.append(f"Ready Time {dur}")
        if deps:
            lines.append(f"Depends   {', '.join(deps[:5])}")
            if len(deps) > 5:
                lines.append(f"          +{len(deps) - 5} more")
        cat_key = str(pol.get("feature_category") or "").lower()
        if cat_key == "lookback" and int(pol.get("intrinsic_warmup_sec") or 0) > 0:
            lines.append(
                f"Policy    NULL until {int(pol.get('intrinsic_warmup_sec') or 0)}s "
                f"lookback ({samples} samples @ {sampling_interval_sec:g}s)",
            )
        elif ctrl.get("controller_label") and not ctrl.get("is_self"):
            lines.append(f"Policy    NULL until {ctrl['controller_label']} ready")
        elif samples <= 0:
            lines.append("Policy    Ready when inputs exist")
        else:
            lines.append(f"Policy    NULL for first {samples} samples")

    return "\n".join(lines)


def format_selection_policy_summary(
    feature_names: list[str],
    features_by_name: dict[str, dict[str, Any]],
    *,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = 20.0,
    used_by_index: dict[str, list[str]] | None = None,
) -> str:
    _ = used_by_index
    return format_policy_summary_tab(
        feature_names,
        features_by_name,
        sampling_interval_sec=sampling_interval_sec,
        gap_max_sec=gap_max_sec,
    )


def format_feature_policy_detail(
    feature: dict[str, Any] | None,
    *,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = 20.0,
    features_by_name: dict[str, dict[str, Any]] | None = None,
    used_by_index: dict[str, list[str]] | None = None,
) -> str:
    if not feature:
        return "Select a feature to inspect category, lifecycle, warm-up, and dependencies."
    features_by_name = features_by_name or {}
    used_by_index = used_by_index or {}
    pol = normalize_policy(feature)
    name = feature.get("name") or feature.get("display_name") or pol.get("name") or "—"
    cat = str(pol.get("feature_category") or "—")
    lifecycle = str(pol.get("lifecycle") or "—")
    deps = display_dependencies(pol)
    samples = warmup_samples(pol)
    gap_sensitive = pol.get("gap_sensitive")
    if gap_sensitive is None:
        gap_sensitive = str(cat).lower() == "rolling"
    reset_on_gap = pol.get("reset_on_gap")
    if reset_on_gap is None:
        reset_on_gap = str(cat).lower() == "rolling"

    lines = [
        "Feature",
        "-" * 40,
        f"  Name: {name}",
        f"  Category: {category_label(cat)}",
        f"  Lifecycle: {LIFECYCLE_LABELS.get(lifecycle, lifecycle.replace('_', ' ').title())}",
        "",
        "Depends On",
        "-" * 40,
    ]
    if deps:
        for d in deps:
            lines.append(f"  {d}")
    else:
        lines.append("  —")
    used_by = format_used_by_section(str(name), pol, used_by_index)
    if used_by:
        lines.extend(["", used_by])
    warmup_section = format_warmup_controller_section(
        pol,
        feature_name=str(name),
        features_by_name=features_by_name,
        sampling_interval_sec=sampling_interval_sec,
    )
    lines.extend(["", warmup_section, "", "Ready Rule", "-" * 40])
    ctrl = resolve_warmup_controller(
        pol, feature_name=str(name), features_by_name=features_by_name,
    )
    if str(cat).lower() == "derived":
        if ctrl["controller_label"] and not ctrl["is_self"]:
            lines.append(
                f"  Governing dependency ready ({ctrl['controller_label']})",
            )
        elif ctrl["controller_label"]:
            lines.append(
                f"  After {ctrl['controller_label']} warm-up ({samples} samples)",
            )
        else:
            lines.append("  All raw inputs present")
    elif str(cat).lower() == "rolling":
        ctrl_label = ctrl["controller_label"] or rolling_feature_label(str(name), pol)
        lines.append(
            f"  After {ctrl_label} warm-up ({samples} samples) without gap reset",
        )
    elif samples > 0:
        lines.append(f"  After {samples} samples")
    else:
        lines.append("  Immediately ready")
    lines.extend([
        "",
        "Gap Sensitive",
        "-" * 40,
        f"  {'YES' if gap_sensitive else 'NO'}",
        "",
        "Reset On Gap",
        "-" * 40,
        f"  {'YES (>' + str(gap_max_sec) + ' sec)' if reset_on_gap else 'NO'}",
        "",
        "Current Policy",
        "-" * 40,
        format_current_policy(
            pol, feature_name=str(name), features_by_name=features_by_name,
        ),
    ])
    readiness = format_readiness_graph(pol, sampling_interval_sec=sampling_interval_sec)
    if readiness:
        lines.extend(["", readiness])
    lines.extend([
        "",
        "Dependency Structure",
        "-" * 40,
        format_dependency_display(str(name), pol, features_by_name),
    ])
    desc = feature.get("description")
    if desc:
        lines.extend(["", "Description", "-" * 40, f"  {desc}"])
    return "\n".join(lines).rstrip()


def format_feature_policy_section(
    feature: dict[str, Any],
    *,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = 20.0,
    features_by_name: dict[str, dict[str, Any]] | None = None,
    used_by_index: dict[str, list[str]] | None = None,
) -> str:
    return format_feature_policy_detail(
        feature,
        sampling_interval_sec=sampling_interval_sec,
        gap_max_sec=gap_max_sec,
        features_by_name=features_by_name,
        used_by_index=used_by_index,
    )


def format_build_summary_preview(preview: dict[str, Any]) -> str:
    if not preview:
        return "No build summary available."
    lines = [
        "Build Summary",
        "=" * 40,
    ]
    cfg = preview.get("build_config") or {}
    if cfg:
        lines.extend([
            "",
            "Build configuration",
            "-" * 40,
            f"  Sampling interval    {cfg.get('sampling_label') or cfg.get('sampling_interval_sec', '—')}",
            f"  Sliding stride       {cfg.get('sliding_stride_label') or cfg.get('sliding_stride_sec', '—')}",
            f"  Feature window       {cfg.get('feature_window_sec', cfg.get('sampling_interval_sec', '—'))} sec",
            f"  Strike selection     {cfg.get('strike_label') or '—'}",
            f"  Target labels        {cfg.get('target_labels_text') or ', '.join(cfg.get('target_labels') or []) or '—'}",
        ])
    lines.extend([
        "",
        "Features",
        "-" * 40,
        f"  Selected: {preview.get('feature_count', 0)}",
    ])
    cls = preview.get("classification") or {}
    lines.append(
        f"  Raw {cls.get('raw', 0)} · Rolling {cls.get('rolling', 0)} · "
        f"Lookback {cls.get('lookback', 0)} · Derived {cls.get('derived', 0)}",
    )
    inh = preview.get("inherited_features")
    if inh is not None:
        lines.append(f"  Inherited Features: {inh}")
    lines.append("")
    lines.append("Warm-up budget")
    lines.append("-" * 40)
    max_feat = preview.get("max_warmup_feature") or "—"
    max_samples = preview.get("max_warmup_samples") or 0
    icon = warmup_tier_icon(int(max_samples))
    lines.append(f"  Longest: {icon} {max_feat} ({max_samples} samples)")
    if preview.get("longest_ready_time"):
        lines.append(f"  Longest Ready Time: {preview['longest_ready_time']}")
    lines.append(f"  Per session: ~{preview.get('warmup_rows_per_session', 0)} rows")
    est = preview.get("estimated_warmup_rows")
    if est is not None:
        lines.append(f"  Estimated total warm-up rows: ~{int(est):,}")
    lines.append("")
    lines.append("Checks")
    lines.append("-" * 40)
    for chk in preview.get("checks") or []:
        icon_c = "✓" if chk.get("status") == "pass" else "✗"
        lines.append(f"  {icon_c} {chk.get('label') or chk.get('id')}")
    lines.append("")
    warmup = preview.get("warmup_preview") or []
    if warmup:
        lines.append("Top warm-up features")
        lines.append("-" * 40)
        for row in warmup[:10]:
            samples = int(row.get("samples") or 0)
            icon = warmup_tier_icon(samples)
            inh_lbl = row.get("inherited_from") or ""
            suffix = f" ← {inh_lbl}" if inh_lbl else ""
            lines.append(
                f"  {icon} {row.get('name')}: {samples} samples{suffix}",
            )
    return "\n".join(lines)


def format_build_validation_preview(preview: dict[str, Any]) -> str:
    """Backward-compatible alias for build summary text."""
    return format_build_summary_preview(preview)


def format_feature_health_report(report: dict[str, Any]) -> str:
    if not report:
        return "No feature health report available."
    lines = [
        "Feature Health",
        "=" * 40,
    ]
    hs = report.get("health_summary") or {}
    if hs:
        lines.append(f"  Features tracked: {hs.get('features_tracked', '—')}")
        lines.append(f"  Avg ready: {hs.get('avg_ready_pct', '—')}%")
        if hs.get("worst_missing_feature"):
            lines.append(
                f"  Worst missing: {hs['worst_missing_feature']} "
                f"({hs.get('worst_missing_pct', '—')}%)",
            )
    pol = report.get("policy") or {}
    if pol:
        lines.append(
            f"  Gap policy: {pol.get('gap_max_sec', '—')}s · "
            f"grid {pol.get('sampling_interval_sec', '—')}s",
        )
    lines.append("")
    health = report.get("feature_health") or []
    if not health:
        lines.append("  No per-feature health sampled.")
        return "\n".join(lines)

    for row in health[:30]:
        name = row.get("name") or "—"
        ready = row.get("ready_pct", "—")
        missing = row.get("missing_pct", "—")
        warmup_pct = float(row.get("warmup_pct") or 0)
        total_rows = int(row.get("rows") or 0)
        warmup_rows = int(row.get("warmup_rows") or round(total_rows * warmup_pct / 100.0))
        gap_n = int(row.get("gap_reset_count") or 0)
        lines.extend([
            name,
            "-" * min(40, len(str(name))),
            f"  Ready: {ready}%",
            f"  Gap Reset: {gap_n}",
            f"  Warm-up skipped: {warmup_rows} rows",
            f"  Missing: {missing}%",
            "",
        ])
    if len(health) > 30:
        lines.append(f"… +{len(health) - 30} more features")
    return "\n".join(lines).rstrip()

FEATURE_POLICY_CSV_COLUMNS = (
    "feature",
    "category",
    "lifecycle",
    "warmup",
    "warmup_samples",
    "dependencies",
    "used_by",
    "gap_sensitive",
    "reset_on_gap",
    "rolling_type",
    "warmup_mode",
    "intrinsic_warmup_sec",
    "formula_version",
    "policy_version",
)


def feature_policy_csv_rows(
    feature_names: list[str],
    features_by_name: dict[str, dict[str, Any]],
    *,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """One CSV row per feature (same fields as the Feature Policy list)."""
    _ = gap_max_sec  # reserved for parity with formatters
    used_by_index = build_used_by_index(feature_names, features_by_name)
    rows: list[dict[str, Any]] = []
    for name in feature_names:
        feat = features_by_name.get(name) or {"name": name}
        pol = normalize_policy({**feat, "name": name})
        deps = display_dependencies(pol)
        users = resolve_used_by(name, pol, used_by_index)
        samples = warmup_samples(pol, sampling_interval_sec=sampling_interval_sec)
        rows.append(
            {
                "feature": name,
                "category": category_label(pol.get("feature_category")),
                "lifecycle": LIFECYCLE_LABELS.get(
                    str(pol.get("lifecycle") or "").lower(),
                    str(pol.get("lifecycle") or "—"),
                ),
                "warmup": format_warmup_cell(
                    pol,
                    feature_name=name,
                    sampling_interval_sec=sampling_interval_sec,
                    features_by_name=features_by_name,
                ),
                "warmup_samples": samples,
                "dependencies": ", ".join(deps) if deps else "",
                "used_by": ", ".join(users) if users else "",
                "gap_sensitive": pol.get("gap_sensitive"),
                "reset_on_gap": pol.get("reset_on_gap"),
                "rolling_type": pol.get("rolling_type") or "",
                "warmup_mode": pol.get("warmup_mode") or "",
                "intrinsic_warmup_sec": pol.get("intrinsic_warmup_sec") or "",
                "formula_version": pol.get("formula_version") or "",
                "policy_version": pol.get("policy_version") or "",
            }
        )
    rows.sort(key=lambda r: (-int(r.get("warmup_samples") or 0), str(r.get("feature") or "")))
    return rows


def feature_policy_csv_text(
    feature_names: list[str],
    features_by_name: dict[str, dict[str, Any]],
    *,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = 0.0,
) -> str:
    """CSV string for Feature Policy download."""
    rows = feature_policy_csv_rows(
        feature_names,
        features_by_name,
        sampling_interval_sec=sampling_interval_sec,
        gap_max_sec=gap_max_sec,
    )
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(FEATURE_POLICY_CSV_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in FEATURE_POLICY_CSV_COLUMNS})
    return buf.getvalue()

