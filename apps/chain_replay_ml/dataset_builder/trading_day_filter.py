"""Trading-day filters for master → registry export (extensible day tags)."""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Mapping, Sequence

# Export modes (registry: trading_day_filter.mode)
MODE_ALL = "all"
MODE_EXCLUDE_EXPIRY = "exclude_expiry"
MODE_EXPIRY_ONLY = "expiry_only"

KNOWN_MODES = frozenset({MODE_ALL, MODE_EXCLUDE_EXPIRY, MODE_EXPIRY_ONLY})

# Day tags — add holiday / budget / rbi / fno_expiry_week here later
TAG_EXPIRY = "expiry"

_DAY_TAG_PREDICATES: dict[str, Callable[[Mapping[str, Any]], bool]] = {}


def register_day_tag(tag: str, predicate: Callable[[Mapping[str, Any]], bool]) -> None:
    """Register or replace a day-tag predicate (future filters plug in here)."""
    key = str(tag or "").strip().lower()
    if not key:
        raise ValueError("day tag name is required")
    _DAY_TAG_PREDICATES[key] = predicate


def _default_is_expiry_day(day: Mapping[str, Any]) -> bool:
    if "is_expiry_day" in day and day.get("is_expiry_day") is not None:
        try:
            return int(day.get("is_expiry_day") or 0) != 0
        except (TypeError, ValueError):
            return bool(day.get("is_expiry_day"))
    td = str(day.get("trading_day") or "").strip()
    dom = str(day.get("dominant_expiry") or "").strip()
    if td and dom and td == dom:
        return True
    return False


register_day_tag(TAG_EXPIRY, _default_is_expiry_day)


def normalize_mode(mode: str | None) -> str:
    raw = str(mode or MODE_ALL).strip().lower() or MODE_ALL
    if raw in ("all_selected", "all_days", ""):
        return MODE_ALL
    if raw not in KNOWN_MODES:
        raise ValueError(f"Unknown trading day filter mode: {mode!r}")
    return raw


def day_has_tag(day: Mapping[str, Any] | None, tag: str) -> bool:
    """True when *day* metadata carries the named tag."""
    key = str(tag or "").strip().lower()
    pred = _DAY_TAG_PREDICATES.get(key)
    if pred is None:
        raise ValueError(f"Unknown trading day tag: {tag!r}")
    return bool(pred(day or {}))


def apply_trading_day_filter(
    selected_days: Sequence[str],
    day_by_key: Mapping[str, Mapping[str, Any]],
    mode: str | None,
) -> tuple[list[str], dict[str, Any]]:
    """
    Filter checkbox-selected trading days by mode.

    Returns (exported_days_sorted, trading_day_filter metadata dict).
    """
    selected = sorted({str(d).strip() for d in selected_days if str(d).strip()})
    selected_n = len(selected)
    norm = normalize_mode(mode)
    expiry_dates = [
        d for d in selected
        if day_has_tag(day_by_key.get(d), TAG_EXPIRY)
    ]

    if norm == MODE_ALL:
        exported = list(selected)
    elif norm == MODE_EXCLUDE_EXPIRY:
        exported = [d for d in selected if d not in set(expiry_dates)]
    elif norm == MODE_EXPIRY_ONLY:
        exported = list(expiry_dates)
    else:
        raise ValueError(f"Unknown trading day filter mode: {mode!r}")

    exported_set = set(exported)
    excluded = [d for d in selected if d not in exported_set]
    meta: dict[str, Any] = {
        "mode": norm,
        "selected_days": selected_n,
        "exported_days": len(exported),
        "selected_dates": selected,
        "exported_dates": exported,
        "excluded_dates": excluded,
        "expiry_dates": expiry_dates,
    }
    return exported, meta


def master_trading_days(master_db_path: str | None) -> list[str]:
    """Sorted trading-day keys available in a master DB.

    Used to resolve "All days" scope to a concrete date list — same source
    (``MasterStore.read_master_days``) the Master Dataset panel uses to
    populate its trading-day checkbox list.
    """
    path = str(master_db_path or "").strip()
    if not path or not os.path.isfile(path):
        return []
    from .master_store import MasterStore

    store = MasterStore(path)
    store.open()
    try:
        rows = store.read_master_days()
    finally:
        store.close()
    return sorted({
        str(r.get("trading_day") or "").strip()
        for r in rows
        if str(r.get("trading_day") or "").strip()
    })


def resolve_day_scope_filter(
    *,
    scope: str,
    selected_days: Iterable[str] | None,
    master_days: Sequence[str],
) -> tuple[bool, list[str], dict[str, Any]]:
    """Map an explicit All-days/Selected-days UI scope to export kwargs + meta.

    Mirrors the Master Dataset panel pattern: regardless of scope, the
    concrete resolved date list is always recorded in the returned
    ``trading_day_filter`` metadata (``selected_dates`` / ``exported_dates``),
    so downstream metadata never has to fall back on a vague "all days" label.

    Returns ``(all_days, explicit_days, trading_day_filter_meta)`` where
    ``explicit_days`` is the list to pass as ``selected_days`` to the export
    builder (empty when ``all_days`` is True — the builder scans everything).
    """
    norm_scope = str(scope or "all").strip().lower()
    master = sorted({str(d).strip() for d in master_days if str(d).strip()})
    if norm_scope == "selected":
        chosen = sorted({str(d).strip() for d in (selected_days or []) if str(d).strip()})
        if master:
            chosen = [d for d in chosen if d in set(master)]
        exported, meta = apply_trading_day_filter(chosen, {}, MODE_ALL)
        return False, exported, meta
    exported, meta = apply_trading_day_filter(master, {}, MODE_ALL)
    return True, [], meta


def trading_day_filter_label(mode: str | None) -> str:
    norm = normalize_mode(mode)
    return {
        MODE_ALL: "All selected days",
        MODE_EXCLUDE_EXPIRY: "Exclude expiry days",
        MODE_EXPIRY_ONLY: "Expiry days only",
    }.get(norm, norm)


def format_date_list(dates: Sequence[str] | None) -> str:
    cleaned = sorted({str(d).strip() for d in (dates or []) if str(d).strip()})
    return ", ".join(cleaned) if cleaned else "—"


def _clean_dates(dates: Sequence[str] | None) -> list[str]:
    return sorted({str(d).strip() for d in (dates or []) if str(d).strip()})


def enrich_trading_day_filter_dates(
    tdf: Mapping[str, Any] | None,
    *,
    exported_dates: Sequence[str] | None = None,
    master_day_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Ensure trading_day_filter carries concrete date lists.

    Older exports stored only selected/exported counts; this backfills
    excluded/expiry/exported dates when possible.
    """
    if not isinstance(tdf, Mapping) or not tdf:
        return {}
    out: dict[str, Any] = dict(tdf)
    try:
        mode = normalize_mode(str(out.get("mode") or MODE_ALL))
    except ValueError:
        mode = MODE_ALL
    out["mode"] = mode

    selected = _clean_dates(out.get("selected_dates"))
    exported = _clean_dates(out.get("exported_dates") or exported_dates)
    excluded = _clean_dates(out.get("excluded_dates"))
    expiry = _clean_dates(out.get("expiry_dates"))

    master_by_day = {
        str(d.get("trading_day") or "").strip(): d
        for d in (master_day_rows or [])
        if str(d.get("trading_day") or "").strip()
    }
    if master_by_day and not selected:
        selected_n = int(out.get("selected_days") or 0)
        if selected_n and selected_n == len(master_by_day):
            selected = sorted(master_by_day)
        elif exported and selected_n > len(exported):
            # Prefer: exported ∪ master expiry days (covers exclude_expiry).
            expiry_in_master = [
                td for td, row in master_by_day.items()
                if day_has_tag(row, TAG_EXPIRY)
            ]
            candidate = sorted(set(exported) | set(expiry_in_master))
            if len(candidate) == selected_n:
                selected = candidate

    if selected and not expiry and master_by_day:
        expiry = [d for d in selected if day_has_tag(master_by_day.get(d), TAG_EXPIRY)]
    elif selected and not expiry:
        # Keep empty; caller may only have selected/exported lists.
        pass

    if selected and not exported and mode == MODE_ALL:
        exported = list(selected)
    elif selected and not exported and mode == MODE_EXCLUDE_EXPIRY:
        exported = [d for d in selected if d not in set(expiry)]
    elif selected and not exported and mode == MODE_EXPIRY_ONLY:
        exported = list(expiry)

    if selected and not excluded:
        excluded = [d for d in selected if d not in set(exported)]

    if mode == MODE_EXCLUDE_EXPIRY and not excluded and expiry:
        excluded = [d for d in expiry if d not in set(exported)]

    if mode == MODE_EXCLUDE_EXPIRY and not excluded and master_by_day and exported:
        excluded = [
            td for td, row in master_by_day.items()
            if day_has_tag(row, TAG_EXPIRY) and td not in set(exported)
        ]
        selected_n = int(out.get("selected_days") or 0)
        if selected_n and len(exported) + len(excluded) != selected_n:
            # Keep if math matches; otherwise still show likely expiry exclusions.
            pass

    out["selected_dates"] = selected
    out["exported_dates"] = exported
    out["excluded_dates"] = excluded
    out["expiry_dates"] = expiry or [
        d for d in excluded if mode == MODE_EXCLUDE_EXPIRY
    ] or expiry
    if not out.get("selected_days"):
        out["selected_days"] = len(selected) if selected else int(out.get("selected_days") or 0)
    if not out.get("exported_days"):
        out["exported_days"] = len(exported) if exported else int(out.get("exported_days") or 0)
    out["selected_days"] = int(out.get("selected_days") or len(selected) or 0)
    out["exported_days"] = int(out.get("exported_days") or len(exported) or 0)
    return out


def trading_day_filter_summary_rows(
    tdf: Mapping[str, Any] | None,
    *,
    exported_dates: Sequence[str] | None = None,
    master_day_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Registry Filters section rows from trading_day_filter metadata."""
    if not isinstance(tdf, Mapping) or not tdf:
        return []
    enriched = enrich_trading_day_filter_dates(
        tdf,
        exported_dates=exported_dates,
        master_day_rows=master_day_rows,
    )
    try:
        mode = normalize_mode(str(enriched.get("mode") or MODE_ALL))
    except ValueError:
        mode = MODE_ALL
    label = trading_day_filter_label(mode)
    selected_n = int(enriched.get("selected_days") or 0)
    exported_n = int(enriched.get("exported_days") or len(enriched.get("exported_dates") or []) or 0)
    rows: list[dict[str, str]] = [
        {
            "label": "Trading day filter",
            "value": f"{label} ({exported_n}/{selected_n} days)" if selected_n else label,
        },
    ]
    excluded = _clean_dates(enriched.get("excluded_dates"))
    expiry = _clean_dates(enriched.get("expiry_dates"))
    exported = _clean_dates(enriched.get("exported_dates"))
    if mode == MODE_EXCLUDE_EXPIRY:
        show = excluded or expiry
        rows.append({"label": "Excluded expiry dates", "value": format_date_list(show)})
    elif mode == MODE_EXPIRY_ONLY:
        rows.append({"label": "Expiry dates included", "value": format_date_list(exported or expiry)})
        if excluded:
            rows.append({"label": "Excluded dates", "value": format_date_list(excluded)})
    elif excluded:
        rows.append({"label": "Excluded dates", "value": format_date_list(excluded)})
    elif expiry:
        rows.append({"label": "Expiry dates (in selection)", "value": format_date_list(expiry)})
    return rows
