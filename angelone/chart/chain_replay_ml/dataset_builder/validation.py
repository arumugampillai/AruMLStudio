"""Pre-write dataset validation with live per-check progress."""

from __future__ import annotations

import math
from typing import Any, Callable

CHUNK_SIZE = 25_000
_PANDAS_DUP_THRESHOLD = 50_000
_DUP_KEY_COLS = ("trading_day", "market", "expiry", "timestamp", "strike", "option_type")

CheckCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]

_REQUIRED_META = ["trading_day", "market", "expiry", "timestamp", "strike", "option_type"]


def _is_bad(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


def _count_duplicate_rows(
    rows: list[dict[str, Any]],
    *,
    cancel_check: CancelCheck | None = None,
    on_chunk: Callable[[int, int], None] | None = None,
) -> int:
    n_rows = len(rows)
    if n_rows < _PANDAS_DUP_THRESHOLD:
        seen: set[tuple[Any, ...]] = set()
        dup_count = 0
        for i in range(0, n_rows, CHUNK_SIZE):
            if cancel_check and cancel_check():
                raise _ValidationCancelled()
            end = min(i + CHUNK_SIZE, n_rows)
            for row in rows[i:end]:
                key = tuple(row.get(col) for col in _DUP_KEY_COLS)
                if key in seen:
                    dup_count += 1
                seen.add(key)
            if on_chunk:
                on_chunk(end, n_rows)
        return dup_count

    import pandas as pd

    parts: list[pd.DataFrame] = []
    for i in range(0, n_rows, CHUNK_SIZE):
        if cancel_check and cancel_check():
            raise _ValidationCancelled()
        end = min(i + CHUNK_SIZE, n_rows)
        chunk = rows[i:end]
        parts.append(pd.DataFrame.from_records(
            [{col: row.get(col) for col in _DUP_KEY_COLS} for row in chunk]
        ))
        if on_chunk:
            on_chunk(end, n_rows)
    keys_df = pd.concat(parts, ignore_index=True)
    return int(keys_df.duplicated(keep="first").sum())


def _count_missing_target_rows(
    rows: list[dict[str, Any]],
    target_columns: list[str],
    *,
    cancel_check: CancelCheck | None = None,
    on_chunk: Callable[[int, int], None] | None = None,
) -> int:
    n_rows = len(rows)
    bad_rows = 0
    for i in range(0, n_rows, CHUNK_SIZE):
        if cancel_check and cancel_check():
            raise _ValidationCancelled()
        end = min(i + CHUNK_SIZE, n_rows)
        for row in rows[i:end]:
            if any(_is_bad(row.get(col)) for col in target_columns):
                bad_rows += 1
        if on_chunk:
            on_chunk(end, n_rows)
    return bad_rows


class _ValidationCancelled(Exception):
    pass


def _build_check_list(target_columns: list[str]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = [
        {"id": "row_count", "label": "Row count", "status": "pending"},
        {"id": "required_columns", "label": "Required columns", "status": "pending"},
        {"id": "target_columns", "label": "Target columns", "status": "pending"},
        {"id": "feature_columns", "label": "Feature columns", "status": "pending"},
        {"id": "feature_count", "label": "Metadata", "status": "pending"},
        {"id": "duplicates", "label": "Duplicate keys", "status": "pending"},
    ]
    if target_columns:
        checks.append({
            "id": "missing_values",
            "label": "Missing values",
            "status": "pending",
        })
    return checks


def _emit(
    checks: list[dict[str, Any]],
    *,
    completed: int,
    on_check: CheckCallback | None,
    message: str,
    sub_current: int | None = None,
    sub_total: int | None = None,
) -> None:
    if not on_check:
        return
    running = next((c for c in checks if c["status"] == "running"), None)
    on_check({
        "checks": [dict(c) for c in checks],
        "current": completed,
        "total": len(checks),
        "message": message,
        "current_check": running["label"] if running else message,
        "sub_current": sub_current,
        "sub_total": sub_total,
    })


def _set_status(checks: list[dict[str, Any]], check_id: str, status: str, detail: str | None = None) -> None:
    for c in checks:
        if c["id"] == check_id:
            c["status"] = status
            if detail:
                c["detail"] = detail
            return


def validate_dataset(
    rows: list[dict[str, Any]],
    *,
    target_columns: list[str],
    feature_columns: list[str],
    expected_feature_count: int | None = None,
    required_meta: list[str] | None = None,
    on_check: CheckCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    required_meta = required_meta or _REQUIRED_META
    checks = _build_check_list(target_columns)
    completed = 0
    total = len(checks)

    def start(check_id: str, message: str) -> bool:
        if cancel_check and cancel_check():
            return False
        _set_status(checks, check_id, "running")
        _emit(checks, completed=completed, on_check=on_check, message=message)
        return True

    def finish(check_id: str, passed: bool, detail: str | None = None) -> bool:
        nonlocal completed
        if not passed:
            _set_status(checks, check_id, "fail", detail)
            if detail:
                issues.append(detail)
        else:
            _set_status(checks, check_id, "pass", detail)
        completed += 1
        _emit(checks, completed=completed, on_check=on_check, message=checks[-1]["label"] if completed < total else "Validation complete")
        return passed

    # 1. Row count
    if not start("row_count", "Checking row count…"):
        return False, ["Cancelled"]
    if not rows:
        finish("row_count", False, "Row count is 0")
        return False, issues
    finish("row_count", True, f"{len(rows):,} rows")

    sample = rows[0]
    n_rows = len(rows)

    # 2. Required columns
    if not start("required_columns", "Checking required columns…"):
        return False, ["Cancelled"]
    missing_meta = [c for c in required_meta if c not in sample]
    if missing_meta:
        finish("required_columns", False, f"Missing metadata: {', '.join(missing_meta)}")
    else:
        finish("required_columns", True)

    # 3. Target columns
    if not start("target_columns", "Checking prediction target columns…"):
        return False, ["Cancelled"]
    missing_targets = [c for c in target_columns if c not in sample]
    if missing_targets:
        finish("target_columns", False, f"Missing targets: {', '.join(missing_targets)}")
    else:
        finish("target_columns", True, f"{len(target_columns)} target columns")

    # 4. Feature columns
    if not start("feature_columns", "Checking feature columns…"):
        return False, ["Cancelled"]
    missing_feats = [c for c in feature_columns if c not in sample]
    if missing_feats:
        finish("feature_columns", False, f"Missing features: {', '.join(missing_feats[:5])}{'…' if len(missing_feats) > 5 else ''}")
    else:
        finish("feature_columns", True, f"{len(feature_columns)} feature columns")

    # 5. Feature count
    if not start("feature_count", "Verifying feature count…"):
        return False, ["Cancelled"]
    present_feats = [c for c in feature_columns if c in sample]
    expected = expected_feature_count if expected_feature_count is not None else len(feature_columns)
    if len(present_feats) != expected:
        finish("feature_count", False, f"Expected {expected} features, found {len(present_feats)}")
    else:
        finish("feature_count", True, f"{len(present_feats)} features")

    # 6. Duplicates (chunked scan)
    if not start("duplicates", f"Scanning {n_rows:,} rows for duplicate keys…"):
        return False, ["Cancelled"]
    try:
        def _on_dup_chunk(end: int, total: int) -> None:
            _emit(
                checks,
                completed=completed,
                on_check=on_check,
                message=f"Scanning duplicate keys… {end:,}/{total:,}",
                sub_current=end,
                sub_total=total,
            )

        dup_count = _count_duplicate_rows(
            rows,
            cancel_check=cancel_check,
            on_chunk=_on_dup_chunk,
        )
    except _ValidationCancelled:
        return False, ["Cancelled"]
    if dup_count:
        finish("duplicates", False, f"Duplicate rows: {dup_count:,}")
    else:
        finish("duplicates", True)

    # 7. Missing values in target columns (chunked)
    if target_columns:
        if not start("missing_values", f"Checking missing targets in {n_rows:,} rows…"):
            return False, ["Cancelled"]
        try:
            def _on_missing_chunk(end: int, total: int) -> None:
                _emit(
                    checks,
                    completed=completed,
                    on_check=on_check,
                    message=f"Checking missing values… {end:,}/{total:,}",
                    sub_current=end,
                    sub_total=total,
                )

            bad_rows = _count_missing_target_rows(
                rows,
                target_columns,
                cancel_check=cancel_check,
                on_chunk=_on_missing_chunk,
            )
        except _ValidationCancelled:
            return False, ["Cancelled"]
        if bad_rows:
            finish("missing_values", False, f"{bad_rows:,} rows with missing target values")
        else:
            finish("missing_values", True)

    return len(issues) == 0, issues
