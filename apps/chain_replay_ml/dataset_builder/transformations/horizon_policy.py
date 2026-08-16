"""Shared interval-specific time-horizon policy.

Used by Lag, Difference, Return, Difference Clip (and warm-up UI warnings).
Edit ``horizon_policy.json`` (or set ``ARUNEO_HORIZON_POLICY``) to add sampling
intervals without code changes.

Horizons are generated dynamically::

    min, min+step, min+2*step, …, max
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .time_shift import LagConfigError

_DEFAULT_POLICY_PATH = Path(__file__).with_name("horizon_policy.json")
_PRIMARY_ENV_POLICY_PATH = "ARUMLSTUDIO_HORIZON_POLICY"
_FALLBACK_ENV_POLICY_PATH = "ARUNEO_HORIZON_POLICY"


@dataclass(frozen=True)
class HorizonIntervalPolicy:
    """Policy for one sample interval."""

    interval_sec: int
    min_horizon_sec: int
    step_sec: int
    max_horizon_sec: int
    warmup_sec: int

    def horizons(self) -> tuple[int, ...]:
        """Inclusive arithmetic sequence from min to max by step."""
        if self.step_sec <= 0:
            raise LagConfigError(
                "Horizon Policy\n"
                f"step_sec must be > 0 for {self.interval_sec}s (got {self.step_sec})."
            )
        out: list[int] = []
        value = int(self.min_horizon_sec)
        while value <= int(self.max_horizon_sec):
            out.append(value)
            value += int(self.step_sec)
        return tuple(out)


def _policy_path() -> Path:
    override = (
        os.environ.get(_PRIMARY_ENV_POLICY_PATH, "").strip()
        or os.environ.get(_FALLBACK_ENV_POLICY_PATH, "").strip()
    )
    if override:
        return Path(override)
    return _DEFAULT_POLICY_PATH


def _parse_interval_entry(interval_key: Any, raw: Any) -> HorizonIntervalPolicy:
    try:
        interval = int(float(interval_key))
    except (TypeError, ValueError) as exc:
        raise LagConfigError(
            "Horizon Policy\n"
            f"Invalid interval key={interval_key!r}."
        ) from exc
    if interval <= 0:
        raise LagConfigError(
            "Horizon Policy\n"
            f"Interval must be > 0 (got {interval})."
        )
    if not isinstance(raw, dict):
        raise LagConfigError(
            "Horizon Policy\n"
            f"Interval {interval}s entry must be an object (got {raw!r})."
        )
    try:
        min_h = int(raw.get("min_horizon_sec"))
        max_h = int(raw.get("max_horizon_sec"))
        warm = int(raw.get("warmup_sec", max_h))
    except (TypeError, ValueError) as exc:
        raise LagConfigError(
            "Horizon Policy\n"
            f"Invalid min/max/warmup for {interval}s: {raw!r}"
        ) from exc
    step_raw = raw.get("step_sec", None)
    if step_raw is None:
        step = min_h
    else:
        try:
            step = int(step_raw)
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "Horizon Policy\n"
                f"Invalid step_sec for {interval}s: {step_raw!r}"
            ) from exc
    if min_h <= 0 or max_h <= 0 or step <= 0 or warm <= 0:
        raise LagConfigError(
            "Horizon Policy\n"
            f"Interval {interval}s requires positive min/step/max/warmup."
        )
    if max_h < min_h:
        raise LagConfigError(
            "Horizon Policy\n"
            f"Interval {interval}s: max_horizon_sec ({max_h}) < min ({min_h})."
        )
    if min_h % interval != 0 or step % interval != 0 or max_h % interval != 0:
        raise LagConfigError(
            "Horizon Policy\n"
            f"Interval {interval}s: min/step/max must be exact multiples of "
            f"the sample interval (min={min_h}, step={step}, max={max_h})."
        )
    return HorizonIntervalPolicy(
        interval_sec=interval,
        min_horizon_sec=min_h,
        step_sec=step,
        max_horizon_sec=max_h,
        warmup_sec=warm,
    )


def _parse_policy_document(doc: dict[str, Any]) -> dict[int, HorizonIntervalPolicy]:
    raw = doc.get("intervals")
    if not isinstance(raw, dict) or not raw:
        raise LagConfigError(
            "Horizon Policy\n"
            "Document missing non-empty 'intervals' map."
        )
    out: dict[int, HorizonIntervalPolicy] = {}
    for key, entry in raw.items():
        policy = _parse_interval_entry(key, entry)
        out[policy.interval_sec] = policy
    return out


def load_horizon_policy(
    path: str | Path | None = None,
    *,
    document: dict[str, Any] | None = None,
) -> dict[int, HorizonIntervalPolicy]:
    if document is not None:
        if not isinstance(document, dict):
            raise LagConfigError("Horizon Policy\nDocument must be a dict.")
        return _parse_policy_document(document)
    policy_path = Path(path) if path is not None else _policy_path()
    try:
        text = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LagConfigError(
            "Horizon Policy\n"
            f"Cannot read policy file: {policy_path}\n{exc}"
        ) from exc
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LagConfigError(
            "Horizon Policy\n"
            f"Invalid JSON in {policy_path}\n{exc}"
        ) from exc
    if not isinstance(doc, dict):
        raise LagConfigError(
            "Horizon Policy\n"
            f"Root must be an object: {policy_path}"
        )
    return _parse_policy_document(doc)


@lru_cache(maxsize=4)
def _cached_default_policy(path_str: str) -> dict[int, HorizonIntervalPolicy]:
    return load_horizon_policy(path_str)


def default_horizon_policy() -> dict[int, HorizonIntervalPolicy]:
    return dict(_cached_default_policy(str(_policy_path().resolve())))


def clear_horizon_policy_cache() -> None:
    _cached_default_policy.cache_clear()


def normalize_horizon_interval(sample_interval_sec: float | int) -> int:
    try:
        interval = float(sample_interval_sec)
    except (TypeError, ValueError) as exc:
        raise LagConfigError(
            "Horizon Policy\n"
            f"Invalid sample_interval_sec={sample_interval_sec!r}."
        ) from exc
    if interval <= 0:
        raise LagConfigError(
            "Horizon Policy\n"
            f"sample_interval_sec must be > 0 (got {interval})."
        )
    as_int = int(round(interval))
    if abs(interval - as_int) > 1e-9:
        raise LagConfigError(
            "Horizon Policy\n"
            f"sample_interval_sec={interval} must be an integer second grid."
        )
    return as_int


def get_horizon_policy(
    sample_interval_sec: float | int,
    *,
    policies: dict[int, HorizonIntervalPolicy] | None = None,
) -> HorizonIntervalPolicy:
    interval = normalize_horizon_interval(sample_interval_sec)
    catalog = policies if policies is not None else default_horizon_policy()
    if interval not in catalog:
        raise LagConfigError(
            "Horizon Policy\n"
            f"No horizon policy for sample_interval_sec={interval}. "
            f"Configured: {sorted(catalog)}. "
            f"Add an entry to horizon_policy.json (or set {_ENV_POLICY_PATH})."
        )
    return catalog[interval]


def default_horizons_for_interval(
    sample_interval_sec: float | int,
    *,
    policies: dict[int, HorizonIntervalPolicy] | None = None,
) -> tuple[int, ...]:
    """Dynamic horizon list for Lag / Difference / Return / Difference Clip."""
    return get_horizon_policy(sample_interval_sec, policies=policies).horizons()


def warmup_seconds_for_interval(
    sample_interval_sec: float | int,
    *,
    policies: dict[int, HorizonIntervalPolicy] | None = None,
) -> float:
    return float(get_horizon_policy(sample_interval_sec, policies=policies).warmup_sec)


def format_horizon_policy_summary(sample_interval_sec: float | int) -> str:
    p = get_horizon_policy(sample_interval_sec)
    n = len(p.horizons())
    return (
        f"{p.interval_sec}s → min {p.min_horizon_sec}s, step {p.step_sec}s, "
        f"max {p.max_horizon_sec}s ({n} horizons), warm-up {p.warmup_sec}s"
    )
