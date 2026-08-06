"""Unified dataset selection — single source of truth for all filtering."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from .master_distribution import DIST_ATM, DIST_DELTA, DIST_PREMIUM
from .master_fingerprint import METADATA_STATUS_VALID, normalize_metadata_status
from .master_naming import path_relative_to_data_dir, resolve_master_db_path

SelectionAccuracy = Literal["exact", "estimated", "partial", "unavailable"]
TableProfile = Literal["master_samples", "prediction_meta"]


def format_day_scope_label(
    *,
    all_days: bool,
    selected_days: Sequence[str] | None = None,
    trading_day: str | None = None,
) -> str:
    """Human-readable day scope for UI and registry summaries."""
    if all_days:
        return "All days"
    days = sorted({str(d).strip() for d in (selected_days or []) if str(d).strip()})
    if len(days) == 1:
        return days[0]
    if len(days) > 1:
        return ", ".join(days)
    td = str(trading_day or "").strip()
    return td or "—"


@dataclass
class DatasetSelectionSpec:
    """Canonical filter specification shared by preview, build, and export."""

    market: str = "NIFTY"
    interval_sec: int | None = None
    master_db_path: str | None = None

    selected_days: list[str] = field(default_factory=list)
    single_day: str | None = None
    all_days: bool = False
    token: str | None = None

    mode: str = "post_filter"
    atm_band: int | None = None
    premium_min: float | None = None
    premium_max: float | None = None
    delta_min: float | None = None
    delta_max: float | None = None
    delta_type: str = "absolute"
    custom_offsets: list[int] | None = None

    premium_enabled: bool = False
    delta_enabled: bool = False

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def has_distribution_filters(self) -> bool:
        if self.atm_band is not None and int(self.atm_band) >= 0:
            return True
        if self.premium_min is not None and self.premium_max is not None:
            return True
        if self.delta_enabled and self.delta_min is not None and self.delta_max is not None:
            return True
        if self.delta_min is not None and self.delta_max is not None and self.mode == "delta_range":
            return True
        return False

    def effective_trading_days(self, available_days: list[str]) -> list[str]:
        if self.all_days:
            return list(available_days)
        if self.single_day:
            td = str(self.single_day).strip()
            return [td] if td in available_days else []
        selected = [str(d).strip() for d in self.selected_days if str(d).strip()]
        if selected:
            avail = set(available_days)
            return [d for d in selected if d in avail]
        return list(available_days)

    def to_strike_selection_dict(self) -> dict[str, Any]:
        mode = str(self.mode or "atm_band").lower()
        if mode == "post_filter":
            mode = "atm_band"
        out: dict[str, Any] = {"mode": mode}
        if mode == "atm_band":
            out["atmBand"] = self.atm_band if self.atm_band is not None else 10
        elif mode == "premium_band":
            out["premiumMin"] = self.premium_min
            out["premiumMax"] = self.premium_max
        elif mode == "delta_range":
            out["deltaType"] = self.delta_type
            out["deltaMin"] = self.delta_min
            out["deltaMax"] = self.delta_max
        elif mode == "custom":
            out["customOffsets"] = list(self.custom_offsets or [])
        return out

    def to_filter_summary_dict(self) -> dict[str, Any]:
        return {
            "all_days": bool(self.all_days),
            "trading_day": self.single_day,
            "token": self.token,
            "atm_band_filter": self.atm_band,
            "premium_enabled": bool(
                self.premium_enabled
                or (self.premium_min is not None and self.premium_max is not None)
            ),
            "premium_min": self.premium_min,
            "premium_max": self.premium_max,
            "delta_enabled": bool(
                self.delta_enabled
                or (self.delta_min is not None and self.delta_max is not None)
            ),
            "delta_min": self.delta_min,
            "delta_max": self.delta_max,
            "selected_days": list(self.selected_days),
        }

    @classmethod
    def from_strike_selection(
        cls,
        strike_selection: dict[str, Any],
        *,
        selected_days: list[str] | None = None,
        market: str = "NIFTY",
        interval_sec: int | None = None,
    ) -> DatasetSelectionSpec:
        ss = dict(strike_selection or {})
        mode = str(ss.get("mode") or "atm_band").lower()
        spec = cls(
            market=str(market or "NIFTY").upper(),
            interval_sec=interval_sec,
            selected_days=list(selected_days or []),
            mode=mode,
            delta_type=str(ss.get("deltaType") or "absolute").lower(),
        )
        if mode == "atm_band":
            band = ss.get("atmBand", 10)
            if str(band).lower() == "all":
                spec.atm_band = 50
            else:
                spec.atm_band = int(band or 10)
        elif mode == "premium_band":
            spec.premium_min = float(ss.get("premiumMin") or 15)
            spec.premium_max = float(ss.get("premiumMax") or 30)
        elif mode == "delta_range":
            spec.delta_min = float(ss.get("deltaMin") or 0.15)
            spec.delta_max = float(ss.get("deltaMax") or 0.50)
            spec.delta_enabled = True
        elif mode == "custom":
            spec.custom_offsets = [int(x) for x in (ss.get("customOffsets") or [])]
        return spec

    @classmethod
    def from_registry_criteria(cls, criteria: dict[str, Any] | None) -> DatasetSelectionSpec:
        crit = dict(criteria or {})
        spec = cls(mode="post_filter")
        spec.all_days = bool(crit.get("all_days"))
        spec.single_day = str(crit.get("trading_day") or "").strip() or None
        spec.token = str(crit.get("token") or "").strip() or None
        if crit.get("atm_band_filter") is not None:
            spec.atm_band = int(crit["atm_band_filter"])
        spec.premium_enabled = bool(crit.get("premium_enabled"))
        spec.premium_min = crit.get("premium_min")
        spec.premium_max = crit.get("premium_max")
        if spec.premium_min is not None:
            spec.premium_min = float(spec.premium_min)
        if spec.premium_max is not None:
            spec.premium_max = float(spec.premium_max)
        spec.delta_enabled = bool(crit.get("delta_enabled"))
        spec.delta_min = crit.get("delta_min")
        spec.delta_max = crit.get("delta_max")
        if spec.delta_min is not None:
            spec.delta_min = float(spec.delta_min)
        if spec.delta_max is not None:
            spec.delta_max = float(spec.delta_max)
        days = crit.get("selected_days")
        if isinstance(days, list):
            spec.selected_days = [str(d) for d in days]
        return spec

    @classmethod
    def from_api_body(cls, body: dict[str, Any]) -> DatasetSelectionSpec:
        market = str(body.get("market") or "NIFTY").upper()
        interval_sec = body.get("interval_sec")
        if interval_sec is not None and interval_sec != "":
            interval_sec = int(interval_sec)
        else:
            interval_sec = None
            master_dataset = body.get("master_dataset")
            if master_dataset:
                m = re.search(r"_(\d+)s\.db$", str(master_dataset), re.I)
                if m:
                    interval_sec = int(m.group(1))

        selected_days = body.get("selected_days") or []
        if not isinstance(selected_days, list):
            selected_days = []

        premium_range = body.get("premium_range")
        delta_range = body.get("delta_range")
        atm_band = body.get("atm_band")
        atm_val = int(atm_band) if atm_band is not None and atm_band != "" else None

        spec = cls(
            market=market,
            interval_sec=interval_sec,
            master_db_path=str(body.get("master_dataset") or "") or None,
            selected_days=[str(d) for d in selected_days],
            atm_band=atm_val,
        )
        if isinstance(premium_range, dict):
            pmin = premium_range.get("min")
            pmax = premium_range.get("max")
            if pmin is not None and pmax is not None:
                spec.premium_min = float(pmin)
                spec.premium_max = float(pmax)
                spec.premium_enabled = True
        if isinstance(delta_range, dict):
            dmin = delta_range.get("min")
            dmax = delta_range.get("max")
            if dmin is not None and dmax is not None:
                spec.delta_min = float(dmin)
                spec.delta_max = float(dmax)
                spec.delta_enabled = True
        strike = body.get("strike_selection")
        if isinstance(strike, dict):
            built = cls.from_strike_selection(
                strike,
                selected_days=spec.selected_days,
                market=market,
                interval_sec=interval_sec,
            )
            if built.mode != "post_filter":
                spec.mode = built.mode
                spec.atm_band = built.atm_band
                spec.premium_min = built.premium_min
                spec.premium_max = built.premium_max
                spec.delta_min = built.delta_min
                spec.delta_max = built.delta_max
                spec.delta_type = built.delta_type
                spec.custom_offsets = built.custom_offsets
                spec.delta_enabled = built.delta_enabled
        return spec


@dataclass
class SelectionPreviewResult:
    estimated_rows: int
    estimated_tokens: int
    estimated_days: int
    estimated_size_mb: float
    estimated_build_time_sec: float
    metadata_version: int
    accuracy: SelectionAccuracy
    warnings: list[str] = field(default_factory=list)
    base_rows: int = 0
    filter_factor: float = 1.0

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "estimated_rows": self.estimated_rows,
            "estimated_tokens": self.estimated_tokens,
            "estimated_days": self.estimated_days,
            "estimated_size_mb": self.estimated_size_mb,
            "estimated_build_time_sec": self.estimated_build_time_sec,
            "metadata_version": self.metadata_version,
            "accuracy": self.accuracy,
            "warnings": self.warnings,
            "base_rows": self.base_rows,
            "filter_factor": round(self.filter_factor, 6),
        }


def _parse_premium_bucket(bucket: str) -> tuple[float, float | None]:
    label = str(bucket).strip()
    if label.endswith("+"):
        return float(label[:-1]), None
    if "-" in label:
        parts = label.split("-", 1)
        return float(parts[0]), float(parts[1])
    return float(label), float(label)


def _parse_delta_bucket(bucket: str) -> tuple[float, float | None]:
    label = str(bucket).strip()
    if label.endswith("+"):
        return float(label[:-1]), None
    if "-" in label:
        parts = label.split("-", 1)
        return float(parts[0]), float(parts[1])
    return float(label), float(label)


def _ranges_overlap(
    a_lo: float,
    a_hi: float | None,
    b_lo: float,
    b_hi: float | None,
) -> bool:
    a_hi_v = a_hi if a_hi is not None else math.inf
    b_hi_v = b_hi if b_hi is not None else math.inf
    return a_lo < b_hi_v and b_lo < a_hi_v


def _premium_factor(
    distributions: list[dict[str, Any]],
    prem_min: float,
    prem_max: float,
) -> tuple[float, bool]:
    rows = [d for d in distributions if d.get("distribution_type") == DIST_PREMIUM]
    total = sum(int(d.get("rows") or 0) for d in rows)
    if total <= 0:
        return 1.0, False
    lo = min(float(prem_min), float(prem_max))
    hi = max(float(prem_min), float(prem_max))
    matched = sum(
        int(item.get("rows") or 0)
        for item in rows
        if _ranges_overlap(*_parse_premium_bucket(str(item.get("bucket"))), lo, hi)
    )
    return matched / total, True


def _atm_factor(distributions: list[dict[str, Any]], atm_band: int) -> tuple[float, bool]:
    rows = [d for d in distributions if d.get("distribution_type") == DIST_ATM]
    total = sum(int(d.get("rows") or 0) for d in rows)
    if total <= 0:
        return 1.0, False
    band = max(0, int(atm_band))
    matched = sum(
        int(d.get("rows") or 0)
        for d in rows
        if str(d.get("bucket", "")).isdigit() and int(d["bucket"]) <= band
    )
    return matched / total, True


def _delta_factor(
    distributions: list[dict[str, Any]],
    delta_min: float,
    delta_max: float,
) -> tuple[float, bool]:
    rows = [d for d in distributions if d.get("distribution_type") == DIST_DELTA]
    total = sum(int(d.get("rows") or 0) for d in rows)
    if total <= 0:
        return 1.0, False
    lo = min(float(delta_min), float(delta_max))
    hi = max(float(delta_min), float(delta_max))
    matched = sum(
        int(item.get("rows") or 0)
        for item in rows
        if _ranges_overlap(*_parse_delta_bucket(str(item.get("bucket"))), lo, hi)
    )
    return matched / total, True


def _token_factor(distributions: list[dict[str, Any]], dist_type: str) -> float | None:
    rows = [d for d in distributions if d.get("distribution_type") == dist_type]
    total_rows = sum(int(d.get("rows") or 0) for d in rows)
    total_tokens = sum(int(d.get("tokens") or 0) for d in rows if d.get("tokens") is not None)
    if total_rows <= 0 or total_tokens <= 0:
        return None
    return total_tokens / total_rows


def _estimate_build_time_sec(rows: int, interval_sec: int | None) -> float:
    if rows <= 0:
        return 0.0
    base = rows / 12000.0
    if interval_sec is not None and int(interval_sec) <= 3:
        base *= 1.35
    return round(max(5.0, base), 1)


def _profile_columns(profile: TableProfile) -> dict[str, Any]:
    if profile == "prediction_meta":
        return {
            "ltp": "current_ltp",
            "abs_delta": "abs_delta",
            "delta": "delta",
            "strike_distance": None,
            "strike": "strike",
            "current_spot": "current_spot",
            "strike_step": 50,
        }
    return {
        "ltp": "ltp",
        "abs_delta": "abs_delta",
        "delta": "delta",
        "strike_distance": "strike_distance_from_atm",
        "strike": None,
        "current_spot": None,
        "strike_step": None,
    }


def build_selection_sql_where(
    spec: DatasetSelectionSpec,
    *,
    profile: TableProfile = "master_samples",
    column_names: set[str] | None = None,
    param_style: Literal["qmark", "inline"] = "qmark",
) -> tuple[str, list[Any]]:
    """Build SQL WHERE fragment (no leading WHERE) for the given table profile."""
    cols = column_names or set()
    pc = _profile_columns(profile)
    where_parts: list[str] = []
    params: list[Any] = []

    def _lit(val: Any) -> str:
        if isinstance(val, float):
            return repr(float(val))
        return repr(val)

    def _add(part_q: str, part_inline: str, *values: Any) -> None:
        if param_style == "inline":
            where_parts.append(part_inline)
        else:
            where_parts.append(part_q)
            params.extend(values)

    td = str(spec.single_day or "").strip()
    if td and not spec.all_days:
        _add("trading_day = ?", f"trading_day = {_lit(td)}", td)
    elif spec.selected_days and not spec.all_days:
        days = [str(d).strip() for d in spec.selected_days if str(d).strip()]
        if days:
            placeholders = ", ".join("?" for _ in days)
            inline_list = ", ".join(_lit(d) for d in days)
            _add(
                f"trading_day IN ({placeholders})",
                f"trading_day IN ({inline_list})",
                *days,
            )

    if spec.token:
        _add("token = ?", f"token = {_lit(spec.token)}", spec.token)

    ltp_col = pc["ltp"]
    prem_active = spec.premium_enabled or (
        spec.premium_min is not None and spec.premium_max is not None
    )
    if prem_active and spec.premium_min is not None and spec.premium_max is not None:
        if not cols or ltp_col in cols:
            lo = float(spec.premium_min)
            hi = float(spec.premium_max)
            if lo > hi:
                lo, hi = hi, lo
            _add(
                f'"{ltp_col}" >= ? AND "{ltp_col}" <= ?',
                f"{ltp_col} >= {_lit(lo)} AND {ltp_col} <= {_lit(hi)}",
                lo,
                hi,
            )

    if spec.atm_band is not None and int(spec.atm_band) >= 0:
        band = int(spec.atm_band)
        dist_col = pc["strike_distance"]
        if dist_col and (not cols or dist_col in cols):
            _add(
                f'ABS("{dist_col}") <= ?',
                f'ABS("{dist_col}") <= {_lit(band)}',
                band,
            )
        elif pc["strike"] and pc["current_spot"] and {"strike", pc["current_spot"]}.issubset(cols or {"strike", pc["current_spot"]}):
            step = int(pc["strike_step"] or 50)
            atm_expr = f"(CAST(ROUND({pc['current_spot']} / {step}.0) AS INTEGER) * {step})"
            _add(
                f"ABS({pc['strike']} - {atm_expr}) <= ?",
                f"ABS({pc['strike']} - {atm_expr}) <= {_lit(band * step)}",
                band * step,
            )

    delta_active = spec.delta_enabled or (
        spec.delta_min is not None and spec.delta_max is not None
    )
    if delta_active and spec.delta_min is not None and spec.delta_max is not None:
        lo = float(spec.delta_min)
        hi = float(spec.delta_max)
        if lo > hi:
            lo, hi = hi, lo
        abs_col = pc["abs_delta"]
        delta_col = pc["delta"]
        if abs_col and (not cols or abs_col in cols):
            _add(
                f'"{abs_col}" >= ? AND "{abs_col}" <= ?',
                f"{abs_col} >= {_lit(lo)} AND {abs_col} <= {_lit(hi)}",
                lo,
                hi,
            )
        elif delta_col and (not cols or delta_col in cols):
            _add(
                f'ABS("{delta_col}") >= ? AND ABS("{delta_col}") <= ?',
                f"ABS({delta_col}) >= {_lit(lo)} AND ABS({delta_col}) <= {_lit(hi)}",
                lo,
                hi,
            )

    if not where_parts:
        return "1=1", params
    return " AND ".join(where_parts), params


class DatasetSelectionEngine:
    """Single entry point for preview estimation, SQL filters, and calibration."""

    def __init__(self, spec: DatasetSelectionSpec, db_path: str | None = None) -> None:
        self.spec = spec
        self.db_path = os.path.abspath(db_path) if db_path else None

    @classmethod
    def resolve_master_db_path(
        cls,
        data_dir: str,
        spec: DatasetSelectionSpec,
    ) -> str:
        if spec.master_db_path:
            name = os.path.basename(str(spec.master_db_path))
            return os.path.join(data_dir, "datasets", name)
        if spec.interval_sec is None:
            raise ValueError("interval_sec is required when master_dataset is not provided")
        return resolve_master_db_path(
            data_dir,
            market=spec.market,
            sampling_interval_sec=int(spec.interval_sec),
        )

    def build_sql_where(
        self,
        *,
        profile: TableProfile = "master_samples",
        column_names: set[str] | None = None,
        param_style: Literal["qmark", "inline"] = "qmark",
    ) -> tuple[str, list[Any]]:
        return build_selection_sql_where(
            self.spec,
            profile=profile,
            column_names=column_names,
            param_style=param_style,
        )

    def count_rows_sql(
        self,
        conn: sqlite3.Connection,
        *,
        table: str = "samples",
        profile: TableProfile = "master_samples",
    ) -> int:
        col_rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        col_names = {str(r[1]) for r in col_rows}
        where_sql, params = self.build_sql_where(
            profile=profile,
            column_names=col_names,
        )
        row = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE {where_sql}',
            params,
        ).fetchone()
        return int(row[0]) if row else 0

    def estimate_from_metadata(
        self,
        meta: dict[str, Any],
        days: list[dict[str, Any]],
        distributions: list[dict[str, Any]],
    ) -> SelectionPreviewResult:
        warnings: list[str] = []
        metadata_version = int(meta.get("metadata_version") or 0)
        status = normalize_metadata_status(meta.get("metadata_status"))
        if status != METADATA_STATUS_VALID:
            warnings.append(f"Metadata status is {status}")

        available = [str(d.get("trading_day") or "") for d in days if d.get("trading_day")]
        effective = self.spec.effective_trading_days(available)
        day_map = {d["trading_day"]: d for d in days}
        day_rows = [day_map[d] for d in effective if d in day_map]

        if self.spec.selected_days and not self.spec.all_days:
            missing = [d for d in self.spec.selected_days if d not in day_map]
            if missing:
                warnings.append(f"{len(missing)} selected day(s) not found in master metadata")
        if not day_rows:
            warnings.append("No trading days in master metadata")

        base_rows = sum(int(d.get("row_count") or 0) for d in day_rows)
        base_tokens = sum(
            int(d.get("token_count") or 0) for d in day_rows if d.get("token_count") is not None
        )

        factors: list[float] = []
        filter_count = 0

        prem_active = self.spec.premium_enabled or (
            self.spec.premium_min is not None and self.spec.premium_max is not None
        )
        if prem_active and self.spec.premium_min is not None and self.spec.premium_max is not None:
            factor, ok = _premium_factor(
                distributions, float(self.spec.premium_min), float(self.spec.premium_max),
            )
            factors.append(factor)
            filter_count += 1
            if not ok:
                warnings.append("Premium distribution unavailable — premium filter ignored")

        if self.spec.atm_band is not None:
            factor, ok = _atm_factor(distributions, int(self.spec.atm_band))
            factors.append(factor)
            filter_count += 1
            if not ok:
                warnings.append("ATM distribution unavailable — ATM band filter ignored")

        delta_active = self.spec.delta_enabled or (
            self.spec.delta_min is not None and self.spec.delta_max is not None
        )
        if delta_active and self.spec.delta_min is not None and self.spec.delta_max is not None:
            factor, ok = _delta_factor(
                distributions, float(self.spec.delta_min), float(self.spec.delta_max),
            )
            factors.append(factor)
            filter_count += 1
            if not ok:
                warnings.append("Delta distribution unavailable — delta filter ignored")

        row_factor = 1.0
        for f in factors:
            row_factor *= f

        estimated_rows = int(round(base_rows * row_factor))
        if filter_count > 1:
            warnings.append("Combined filters assume bucket independence (estimated, not exact)")
        if not distributions and filter_count > 0:
            warnings.append("Distribution metadata empty — row estimate may be inaccurate")

        token_ratio = _token_factor(distributions, DIST_ATM)
        if token_ratio is None and base_rows > 0 and base_tokens > 0:
            token_ratio = base_tokens / base_rows
        estimated_tokens = (
            int(round(estimated_rows * token_ratio))
            if token_ratio is not None and estimated_rows > 0
            else 0
        )

        total_rows_meta = int(meta.get("total_rows") or 0)
        db_bytes = int(meta.get("database_size") or 0)
        if estimated_rows > 0 and total_rows_meta > 0 and db_bytes > 0:
            estimated_size_mb = round(
                (db_bytes / total_rows_meta) * estimated_rows / (1024 * 1024), 1,
            )
        else:
            col_count = int(meta.get("feature_count") or 0) + int(meta.get("target_count") or 0) + 12
            raw_bytes = estimated_rows * (col_count * 8 + 48)
            estimated_size_mb = round(raw_bytes / (1024 * 1024), 1) if raw_bytes > 0 else 0.0
            if estimated_rows > 0:
                warnings.append("Database size estimated from column count (no master DB size metadata)")

        interval = self.spec.interval_sec or meta.get("sampling_interval_sec")
        build_time = _estimate_build_time_sec(
            estimated_rows,
            int(interval) if interval else None,
        )

        accuracy = self._resolve_preview_accuracy(
            day_rows=day_rows,
            filter_count=filter_count,
            warnings=warnings,
        )

        return SelectionPreviewResult(
            estimated_rows=estimated_rows,
            estimated_tokens=estimated_tokens,
            estimated_days=len(day_rows),
            estimated_size_mb=estimated_size_mb,
            estimated_build_time_sec=build_time,
            metadata_version=metadata_version,
            accuracy=accuracy,
            warnings=warnings,
            base_rows=base_rows,
            filter_factor=row_factor,
        )

    @staticmethod
    def _resolve_preview_accuracy(
        *,
        day_rows: list[dict[str, Any]],
        filter_count: int,
        warnings: list[str],
    ) -> SelectionAccuracy:
        if not day_rows:
            return "unavailable"
        if warnings and any(
            "unavailable" in w.lower() or "not found" in w.lower() for w in warnings
        ):
            return "partial"
        if filter_count == 0:
            return "exact"
        if filter_count == 1 and not any("independence" in w for w in warnings):
            return "estimated"
        return "estimated"

    def preview(self) -> SelectionPreviewResult:
        if not self.db_path or not os.path.isfile(self.db_path):
            return SelectionPreviewResult(
                estimated_rows=0,
                estimated_tokens=0,
                estimated_days=0,
                estimated_size_mb=0.0,
                estimated_build_time_sec=0.0,
                metadata_version=0,
                accuracy="unavailable",
                warnings=["Master database file does not exist"],
            )

        from .master_store import MasterStore

        store = MasterStore(self.db_path)
        store.open()
        try:
            meta = store.read_master_meta_dict()
            days = store.read_master_days()
            distributions = store.read_master_distributions()
        finally:
            store.close()
        return self.estimate_from_metadata(meta, days, distributions)

    @classmethod
    def preview_from_body(cls, data_dir: str, body: dict[str, Any]) -> dict[str, Any]:
        spec = DatasetSelectionSpec.from_api_body(body)
        db_path = cls.resolve_master_db_path(data_dir, spec)
        engine = cls(spec, db_path)
        result = engine.preview()
        out = result.to_api_dict()
        out["master_db_path"] = path_relative_to_data_dir(db_path, data_dir)
        return out

    @classmethod
    def from_build_config(
        cls,
        *,
        strike_selection: dict[str, Any],
        sources: list[dict[str, Any]],
        market: str = "NIFTY",
        interval_sec: int | None = None,
        master_db_path: str | None = None,
    ) -> DatasetSelectionEngine:
        days = [str(s.get("trading_day") or "") for s in sources if s.get("trading_day")]
        spec = DatasetSelectionSpec.from_strike_selection(
            strike_selection,
            selected_days=days,
            market=market,
            interval_sec=interval_sec,
        )
        spec.master_db_path = master_db_path
        return cls(spec)
