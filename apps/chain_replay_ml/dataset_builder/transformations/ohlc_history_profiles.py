"""Load configurable OHLC Aggregation history profiles.

History retention is product configuration keyed by sample interval — not derived
from warm-up. Edit ``ohlc_history_profiles.json`` (or set
``ARUNEO_OHLC_HISTORY_PROFILES`` / pass an in-memory document) to add intervals or
timeframe/history combinations without code changes.

When a labeled period is not divisible by the sample interval, keep the label and
set ``seconds`` to the actual grid duration plus optional ``nominal_seconds`` so
UI/metadata stay transparent (e.g. 5m @ 9s → 297s / 33 samples, nominal 300).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .time_shift import LagConfigError

_DEFAULT_PROFILE_PATH = Path(__file__).with_name("ohlc_history_profiles.json")
_ENV_PROFILE_PATH = "ARUNEO_OHLC_HISTORY_PROFILES"


@dataclass(frozen=True)
class OhlcTimeframeSpec:
    """One timeframe inside an interval profile.

    ``key`` / ``timeframe_label`` is the product label shown in the UI (e.g. ``5m``).
    ``seconds`` / ``actual_duration_sec`` is the candle duration actually used.
    When ``nominal_seconds`` differs from ``seconds``, the candle is an explicit
    sample-grid approximation of the labeled period (documented in metadata).
    """

    key: str
    seconds: int
    history: int  # completed candles retained (index 1 = newest)
    nominal_seconds: int | None = None

    @property
    def timeframe_label(self) -> str:
        return self.key

    @property
    def actual_duration_sec(self) -> int:
        return self.seconds

    @property
    def nominal_duration_sec(self) -> int:
        return int(self.nominal_seconds) if self.nominal_seconds is not None else self.seconds

    @property
    def is_approximate(self) -> bool:
        return self.nominal_duration_sec != self.actual_duration_sec

    def sample_count(self, sample_interval_sec: float | int) -> int:
        interval = int(round(float(sample_interval_sec)))
        if interval <= 0:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"sample_interval_sec must be > 0 (got {sample_interval_sec!r})."
            )
        return self.seconds // interval

    def to_metadata(self, sample_interval_sec: float | int) -> dict[str, Any]:
        """Transparent timeframe record for dataset / experiment metadata."""
        interval = int(round(float(sample_interval_sec)))
        meta: dict[str, Any] = {
            "timeframe_label": self.timeframe_label,
            "actual_duration_sec": self.actual_duration_sec,
            "sample_count": self.sample_count(interval),
            "history": self.history,
            "nominal_duration_sec": self.nominal_duration_sec,
            "is_approximate": self.is_approximate,
        }
        return meta


@dataclass(frozen=True)
class OhlcIntervalProfile:
    """History profile for one sample interval."""

    interval_sec: int
    timeframes: dict[str, OhlcTimeframeSpec]

    def timeframe_keys(self) -> tuple[str, ...]:
        return tuple(self.timeframes.keys())

    def get(self, timeframe: str) -> OhlcTimeframeSpec:
        key = str(timeframe or "").strip().lower()
        if key not in self.timeframes:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"Timeframe {timeframe!r} is not in the history profile for "
                f"{self.interval_sec}s sampling. "
                f"Available: {list(self.timeframes)}."
            )
        return self.timeframes[key]


def _profile_path() -> Path:
    override = os.environ.get(_ENV_PROFILE_PATH, "").strip()
    if override:
        return Path(override)
    return _DEFAULT_PROFILE_PATH


def _parse_entry(tf_key: str, raw: Any) -> OhlcTimeframeSpec:
    key = str(tf_key or "").strip().lower()
    if not key:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            "History profile entry missing timeframe key."
        )
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Profile entry for {key!r} must be an object with "
            f"'seconds' and 'history' (got {raw!r})."
        )
    if not isinstance(raw, dict):
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Invalid profile entry for {key!r}: {raw!r}"
        )
    try:
        seconds = int(raw.get("seconds"))
        history = int(raw.get("history"))
    except (TypeError, ValueError) as exc:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Invalid seconds/history for timeframe {key!r}: {raw!r}"
        ) from exc
    if seconds <= 0 or history <= 0:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Timeframe {key!r} requires seconds>0 and history>0 "
            f"(got seconds={seconds}, history={history})."
        )
    nominal_seconds: int | None = None
    if "nominal_seconds" in raw and raw.get("nominal_seconds") is not None:
        try:
            nominal_seconds = int(raw.get("nominal_seconds"))
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"Invalid nominal_seconds for timeframe {key!r}: {raw!r}"
            ) from exc
        if nominal_seconds <= 0:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"Timeframe {key!r} nominal_seconds must be > 0 "
                f"(got {nominal_seconds})."
            )
    return OhlcTimeframeSpec(
        key=key,
        seconds=seconds,
        history=history,
        nominal_seconds=nominal_seconds,
    )


def _parse_profiles_document(doc: dict[str, Any]) -> dict[int, OhlcIntervalProfile]:
    raw_profiles = doc.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            "History profiles document missing non-empty 'profiles' map."
        )
    out: dict[int, OhlcIntervalProfile] = {}
    for interval_key, entries in raw_profiles.items():
        try:
            interval = int(float(interval_key))
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"Invalid profile interval key={interval_key!r}."
            ) from exc
        if interval <= 0:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"Profile interval must be > 0 (got {interval})."
            )
        if not isinstance(entries, dict) or not entries:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"Profile for {interval}s has no timeframes."
            )
        tfs: dict[str, OhlcTimeframeSpec] = {}
        for tf_key, raw in entries.items():
            spec = _parse_entry(tf_key, raw)
            if spec.seconds % interval != 0:
                raise LagConfigError(
                    "OHLC Aggregation Transformation\n"
                    f"Profile {interval}s / {spec.key}: seconds={spec.seconds} "
                    f"is not divisible by sample interval {interval}."
                )
            tfs[spec.key] = spec
        out[interval] = OhlcIntervalProfile(interval_sec=interval, timeframes=tfs)
    return out


def _parse_unavailable(doc: dict[str, Any]) -> dict[int, dict[str, str]]:
    """Optional map: interval → {timeframe → reason string} for UI messaging."""
    raw = doc.get("unavailable")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            "'unavailable' must be an object keyed by sample interval."
        )
    out: dict[int, dict[str, str]] = {}
    for interval_key, entries in raw.items():
        try:
            interval = int(float(interval_key))
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"Invalid unavailable interval key={interval_key!r}."
            ) from exc
        if not isinstance(entries, dict):
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"unavailable[{interval}] must be an object of timeframe→reason."
            )
        messages: dict[str, str] = {}
        for tf_key, reason in entries.items():
            key = str(tf_key or "").strip().lower()
            text = str(reason or "").strip()
            if key and text:
                messages[key] = text
        if messages:
            out[interval] = messages
    return out


def _load_document(path: str | Path | None = None) -> dict[str, Any]:
    profile_path = Path(path) if path is not None else _profile_path()
    try:
        text = profile_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Cannot read history profiles file: {profile_path}\n{exc}"
        ) from exc
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Invalid JSON in history profiles file: {profile_path}\n{exc}"
        ) from exc
    if not isinstance(doc, dict):
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"History profiles root must be an object: {profile_path}"
        )
    return doc


def load_ohlc_history_profiles(
    path: str | Path | None = None,
    *,
    document: dict[str, Any] | None = None,
) -> dict[int, OhlcIntervalProfile]:
    """Load profiles from JSON path or an in-memory document."""
    if document is not None:
        if not isinstance(document, dict):
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                "History profile document must be a dict."
            )
        return _parse_profiles_document(document)
    return _parse_profiles_document(_load_document(path))


@lru_cache(maxsize=4)
def _cached_default_catalog(
    path_str: str,
) -> tuple[dict[int, OhlcIntervalProfile], dict[int, dict[str, str]]]:
    doc = _load_document(path_str)
    return _parse_profiles_document(doc), _parse_unavailable(doc)


def default_ohlc_history_profiles() -> dict[int, OhlcIntervalProfile]:
    """Cached default profiles from package JSON (or env override path)."""
    profiles, _unavailable = _cached_default_catalog(str(_profile_path().resolve()))
    return dict(profiles)


def clear_ohlc_history_profile_cache() -> None:
    _cached_default_catalog.cache_clear()


def unavailable_ohlc_timeframe_messages(
    sample_interval_sec: float | int,
) -> dict[str, str]:
    """Configured reasons for timeframes intentionally omitted at this interval."""
    interval = normalize_profile_interval(sample_interval_sec)
    _profiles, unavailable = _cached_default_catalog(str(_profile_path().resolve()))
    return dict(unavailable.get(interval) or {})


def format_ohlc_unavailable_hint(sample_interval_sec: float | int) -> str:
    messages = unavailable_ohlc_timeframe_messages(sample_interval_sec)
    if not messages:
        return ""
    return " ".join(messages.values())


def format_ohlc_approximation_hint(sample_interval_sec: float | int) -> str:
    """UI note listing approximate timeframe labels for this sample interval."""
    try:
        profile = get_ohlc_interval_profile(sample_interval_sec)
    except LagConfigError:
        return ""
    parts: list[str] = []
    for key in profile.timeframe_keys():
        spec = profile.get(key)
        if not spec.is_approximate:
            continue
        samples = spec.sample_count(profile.interval_sec)
        parts.append(
            f"{spec.timeframe_label} uses {spec.actual_duration_sec}s "
            f"({samples} samples; nominal {spec.nominal_duration_sec}s)"
        )
    if not parts:
        return ""
    return "Approximate candles: " + "; ".join(parts) + "."


def timeframe_specs_metadata(
    sample_interval_sec: float | int,
    timeframes: list[str] | tuple[str, ...],
    *,
    history_overrides: dict[str, Any] | None = None,
    profiles: dict[int, OhlcIntervalProfile] | None = None,
) -> list[dict[str, Any]]:
    """Build transparent timeframe metadata for selected labels."""
    out: list[dict[str, Any]] = []
    for tf in timeframes:
        spec = resolve_timeframe_spec(
            sample_interval_sec,
            tf,
            profiles=profiles,
            history_overrides=history_overrides,
        )
        out.append(spec.to_metadata(sample_interval_sec))
    return out


def normalize_profile_interval(sample_interval_sec: float | int) -> int:
    try:
        interval = float(sample_interval_sec)
    except (TypeError, ValueError) as exc:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"Invalid sample_interval_sec={sample_interval_sec!r}."
        ) from exc
    if interval <= 0:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"sample_interval_sec must be > 0 (got {interval})."
        )
    as_int = int(round(interval))
    if abs(interval - as_int) > 1e-9:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"sample_interval_sec={interval} must be an integer second grid "
            "to match a history profile key."
        )
    return as_int


def get_ohlc_interval_profile(
    sample_interval_sec: float | int,
    *,
    profiles: dict[int, OhlcIntervalProfile] | None = None,
) -> OhlcIntervalProfile:
    """Resolve the history profile for a dataset sampling interval."""
    interval = normalize_profile_interval(sample_interval_sec)
    catalog = profiles if profiles is not None else default_ohlc_history_profiles()
    if interval not in catalog:
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"No OHLC history profile for sample_interval_sec={interval}. "
            f"Configured profiles: {sorted(catalog)}. "
            "Add an entry to ohlc_history_profiles.json (or override via "
            f"{_ENV_PROFILE_PATH})."
        )
    return catalog[interval]


def resolve_timeframe_spec(
    sample_interval_sec: float | int,
    timeframe: str,
    *,
    profiles: dict[int, OhlcIntervalProfile] | None = None,
    history_overrides: dict[str, Any] | None = None,
) -> OhlcTimeframeSpec:
    """Resolve timeframe seconds/history for an interval (optional per-run overrides)."""
    profile = get_ohlc_interval_profile(sample_interval_sec, profiles=profiles)
    spec = profile.get(timeframe)
    if not history_overrides:
        return spec
    raw = history_overrides.get(spec.key)
    if raw is None:
        return spec
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        hist = int(raw)
        if hist <= 0:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"history override for {spec.key} must be > 0 (got {hist})."
            )
        return OhlcTimeframeSpec(
            key=spec.key,
            seconds=spec.seconds,
            history=hist,
            nominal_seconds=spec.nominal_seconds,
        )
    if isinstance(raw, dict):
        seconds = int(raw.get("seconds", spec.seconds))
        history = int(raw.get("history", spec.history))
        if seconds <= 0 or history <= 0:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"Invalid history override for {spec.key}: {raw!r}"
            )
        interval = normalize_profile_interval(sample_interval_sec)
        if seconds % interval != 0:
            raise LagConfigError(
                "OHLC Aggregation Transformation\n"
                f"Override seconds={seconds} for {spec.key} is not divisible "
                f"by sample interval {interval}."
            )
        nominal_raw = raw.get("nominal_seconds", spec.nominal_seconds)
        nominal_seconds = int(nominal_raw) if nominal_raw is not None else None
        return OhlcTimeframeSpec(
            key=spec.key,
            seconds=seconds,
            history=history,
            nominal_seconds=nominal_seconds,
        )
    raise LagConfigError(
        "OHLC Aggregation Transformation\n"
        f"Invalid history override for {spec.key}: {raw!r}"
    )


def available_ohlc_timeframes(
    sample_interval_sec: float | int,
    *,
    profiles: dict[int, OhlcIntervalProfile] | None = None,
) -> tuple[str, ...]:
    return get_ohlc_interval_profile(
        sample_interval_sec, profiles=profiles
    ).timeframe_keys()


def parse_history_overrides(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Optional transform params: ``history_by_timeframe`` or ``history_overrides``."""
    if not params:
        return None
    raw = params.get("history_by_timeframe")
    if raw is None:
        raw = params.get("history_overrides")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise LagConfigError(
            "OHLC Aggregation Transformation\n"
            f"history_by_timeframe must be an object (got {raw!r})."
        )
    return {
        str(k).strip().lower(): v
        for k, v in raw.items()
        if str(k).strip()
    }
