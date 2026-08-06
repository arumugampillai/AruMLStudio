"""Shared feature access for Research Lab prediction datasets.

Master Dataset is the sole feature store. New prediction builds store outcomes +
``master_row_id`` only (``feature_storage_mode=referenced``). Legacy labs keep
embedded ``sf_*`` columns (``embedded``). All consumers should go through this
module — do not JOIN master manually elsewhere.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Any, Mapping

from .prediction_schema import (
    FEATURE_STORAGE_EMBEDDED,
    FEATURE_STORAGE_REFERENCED,
    sanitize_feature_column,
)
from .store import ModelLabStore

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MASTER_ATTACH = "master_feat"


def _safe_ident(name: str) -> bool:
    return bool(name) and bool(_SAFE_IDENT.match(name))


def master_dataset_id_from_path(path: str | None) -> str | None:
    if not path:
        return None
    base = os.path.basename(str(path).replace("\\", "/"))
    if base.lower().endswith(".db"):
        base = base[:-3]
    return base or None


def resolve_master_db_path(
    *,
    data_dir: str | None,
    master_db_path: str | None,
    lab_db_path: str | None = None,
) -> str | None:
    """Resolve relative or absolute master DB path; return existing file path."""
    raw = str(master_db_path or "").strip()
    if not raw:
        return None

    def _ok(path: str) -> str | None:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
        return None

    def _score(path: str) -> int:
        """Prefer real masters (has samples + master_row_id) over empty stubs."""
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error:
            return -1
        try:
            cols = {
                str(r[1]) for r in conn.execute("PRAGMA table_info(samples)").fetchall()
            }
            if not cols:
                return -1
            score = 1
            if "master_row_id" in cols:
                score += 100
            try:
                n = int(conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] or 0)
            except sqlite3.Error:
                n = 0
            if n > 0:
                score += 50
            if n > 1000:
                score += 10
            return score
        except sqlite3.Error:
            return -1
        finally:
            conn.close()

    candidates: list[str] = []
    if os.path.isabs(raw):
        found = _ok(raw)
        if found:
            candidates.append(found)

    basename = os.path.basename(raw.replace("\\", "/"))
    if not basename.lower().endswith(".db"):
        basename = f"{basename}.db"

    roots: list[str] = []
    if data_dir:
        roots.append(str(data_dir))
    if lab_db_path:
        lab_dir = os.path.dirname(os.path.abspath(lab_db_path))
        roots.append(lab_dir)
        parent = lab_dir
        for _ in range(5):
            parent = os.path.dirname(parent)
            if not parent or parent in roots:
                break
            roots.append(parent)

    seen: set[str] = set()
    for root in roots:
        for rel in (
            raw.replace("/", os.sep),
            os.path.join("datasets", basename),
            basename,
        ):
            cand = rel if os.path.isabs(rel) else os.path.join(root, rel)
            key = os.path.normcase(os.path.abspath(cand))
            if key in seen:
                continue
            seen.add(key)
            found = _ok(cand)
            if found:
                candidates.append(found)
        # Sibling layouts: <parent>/<child>/datasets/<basename>
        try:
            for child in os.listdir(root):
                child_path = os.path.join(root, child)
                if not os.path.isdir(child_path):
                    continue
                cand = os.path.join(child_path, "datasets", basename)
                key = os.path.normcase(os.path.abspath(cand))
                if key in seen:
                    continue
                seen.add(key)
                found = _ok(cand)
                if found:
                    candidates.append(found)
        except OSError:
            pass

    found = _ok(raw)
    if found:
        candidates.append(found)

    if not candidates:
        return None

    # Prefer usable masters (rows and/or master_row_id). Skip empty stubs.
    best_path: str | None = None
    best_score = -1
    for cand in candidates:
        sc = _score(cand)
        if sc > best_score:
            best_score = sc
            best_path = cand
            # Good enough: has IDs + rows
            if sc >= 150:
                break
    # Require at least some samples (score>=51) or master_row_id (score>=101)
    if best_score < 51:
        return None
    return best_path


def load_day_master_row_id_map(
    master_db_path: str,
    trading_day: str,
    *,
    timeout_sec: float = 30.0,
) -> dict[tuple[float, str], int]:
    """Stamp FK during prediction build only — not for ongoing feature joins."""
    out: dict[tuple[float, str], int] = {}
    try:
        conn = sqlite3.connect(master_db_path, timeout=float(timeout_sec))
        conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)}")
    except sqlite3.Error:
        return out
    try:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(samples)").fetchall()}
        if "master_row_id" not in cols:
            return out
        rows = conn.execute(
            """
            SELECT timestamp, token, master_row_id
            FROM samples
            WHERE trading_day = ? AND master_row_id IS NOT NULL
            """,
            (str(trading_day),),
        ).fetchall()
        for ts, token, rid in rows:
            try:
                key = (float(ts), str(token or ""))
                out[key] = int(rid)
            except (TypeError, ValueError):
                continue
    except sqlite3.Error:
        return out
    finally:
        conn.close()
    return out


def master_sample_columns(
    master_db_path: str,
    *,
    timeout_sec: float = 10.0,
) -> set[str]:
    """Column names of Master's ``samples`` table (empty set if unavailable).

    Master stores raw/near-term derived columns only (e.g. ``future_ltp_10s``,
    ``future_ltp_1m``) — longer-horizon labels such as ``future_ltp_5m`` are
    computed at analysis-dataset build time and are **not** backfilled into
    Master. Callers that need a specific target column present must check
    this before relying on the Master fallback for a given day.
    """
    path = str(master_db_path or "").strip()
    if not path or not os.path.isfile(path):
        return set()
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=float(timeout_sec))
        conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)}")
        return {str(r[1]) for r in conn.execute("PRAGMA table_info(samples)").fetchall()}
    except sqlite3.Error:
        return set()
    finally:
        if conn is not None:
            conn.close()


def count_trading_day_rows_in_master(
    master_db_path: str,
    trading_day: str,
    *,
    master_filter: dict[str, Any] | Mapping[str, Any] | None = None,
    timeout_sec: float = 60.0,
) -> int:
    path = str(master_db_path or "").strip()
    day = str(trading_day or "").strip()
    if not path or not day or not os.path.isfile(path):
        return 0
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=float(timeout_sec))
        conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)}")
        where_sql, params = _master_day_filter_where(conn, day, master_filter)
        row = conn.execute(
            f"SELECT COUNT(*) FROM samples WHERE {where_sql}",
            params,
        ).fetchone()
        return int(row[0] or 0) if row else 0
    except sqlite3.Error:
        return 0
    finally:
        if conn is not None:
            conn.close()


def load_trading_day_frame_from_master(
    master_db_path: str,
    trading_day: str,
    columns: list[str],
    *,
    master_filter: dict[str, Any] | Mapping[str, Any] | None = None,
    timeout_sec: float = 120.0,
) -> "Any":
    """Load one trading day from Master ``samples`` for Unseen / Master-only builds.

    When ``master_filter`` is provided (from this model's training export metadata),
    applies the same ATM / premium / delta / token / no-null rules used to create
    the training dataset — not the full unfiltered Master day.
    """
    import pandas as pd

    path = str(master_db_path or "").strip()
    day = str(trading_day or "").strip()
    if not path or not day or not os.path.isfile(path):
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=float(timeout_sec))
        conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)}")
    except sqlite3.Error:
        return pd.DataFrame()

    try:
        available = {
            str(r[1]) for r in conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        if not available:
            return pd.DataFrame()
        wanted: list[str] = []
        for col in columns:
            name = str(col or "").strip()
            if name and name in available and name not in wanted:
                wanted.append(name)
        if "master_row_id" in available and "master_row_id" not in wanted:
            wanted.append("master_row_id")
        if "trading_day" in available and "trading_day" not in wanted:
            wanted.insert(0, "trading_day")
        if not wanted:
            return pd.DataFrame()
        where_sql, params = _master_day_filter_where(conn, day, master_filter)
        select_sql = ", ".join(f'"{c}"' for c in wanted)
        sql = (
            f'SELECT {select_sql} FROM samples '
            f'WHERE {where_sql} ORDER BY timestamp, token'
        )
        df = pd.read_sql_query(sql, conn, params=params)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except (sqlite3.Error, ValueError, TypeError, OSError):
        return pd.DataFrame()
    finally:
        conn.close()


def _master_day_filter_where(
    conn: sqlite3.Connection,
    trading_day: str,
    master_filter: Mapping[str, Any] | None,
) -> tuple[str, list[Any]]:
    """WHERE clause for one day using this model's training Master filters."""
    from chain_replay_ml.dataset_builder.master_status import (
        _no_null_complete_where,
        _sample_filter_where,
    )

    day = str(trading_day or "").strip()
    col_names = [str(r[1]) for r in conn.execute("PRAGMA table_info(samples)").fetchall()]
    mf = dict(master_filter or {})

    prem_on = bool(mf.get("premium_enabled")) or (
        mf.get("premium_min") is not None and mf.get("premium_max") is not None
    )
    delta_on = bool(mf.get("delta_enabled")) or (
        mf.get("delta_min") is not None and mf.get("delta_max") is not None
    )
    atm = mf.get("atm_band_filter")
    try:
        atm_i = int(atm) if atm is not None and str(atm).lower() != "all" else None
    except (TypeError, ValueError):
        atm_i = None

    where_sql, params = _sample_filter_where(
        trading_day=day,
        selected_days=None,
        all_days=False,
        token=str(mf.get("token") or "").strip() or None,
        atm_band_filter=atm_i,
        premium_min=float(mf["premium_min"]) if prem_on and mf.get("premium_min") is not None else None,
        premium_max=float(mf["premium_max"]) if prem_on and mf.get("premium_max") is not None else None,
        delta_min=float(mf["delta_min"]) if delta_on and mf.get("delta_min") is not None else None,
        delta_max=float(mf["delta_max"]) if delta_on and mf.get("delta_max") is not None else None,
        column_names=col_names,
    )
    if mf.get("no_null_data"):
        where_sql, _active, _dropped = _no_null_complete_where(
            conn, col_names, where_sql, params
        )
    return where_sql, params


def ensure_master_row_id_light(
    master_db_path: str,
    *,
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """
    Ensure ``master_row_id`` on an existing master DB without opening MasterStore.

    Avoids MasterStore's full CREATE TABLE bootstrap (which needs a write lock
    and fails when the Master Dataset Manager already has the file open).

    Uses ``EXISTS … LIMIT 1`` instead of ``COUNT(*)`` so Build+Compute prepare
    does not stall for minutes on large master DBs when IDs are already filled.

    Returns ``{ok, has_column, wrote, error}``.
    """
    path = str(master_db_path or "").strip()
    if not path or not os.path.isfile(path):
        return {
            "ok": False,
            "has_column": False,
            "wrote": False,
            "error": f"Master DB not found: {path}",
        }
    try:
        conn = sqlite3.connect(path, timeout=float(timeout_sec))
        conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)}")
    except sqlite3.Error as exc:
        return {
            "ok": False,
            "has_column": False,
            "wrote": False,
            "error": str(exc),
        }
    wrote = False
    try:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(samples)").fetchall()}
        if "samples" not in {
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }:
            return {
                "ok": False,
                "has_column": False,
                "wrote": False,
                "error": "Master DB has no samples table",
            }
        if "master_row_id" not in cols:
            conn.execute("ALTER TABLE samples ADD COLUMN master_row_id INTEGER")
            wrote = True
        # Cheap probe — never full-table COUNT(*) on multi-GB masters.
        has_null = conn.execute(
            "SELECT 1 FROM samples WHERE master_row_id IS NULL LIMIT 1"
        ).fetchone()
        if has_null:
            conn.execute(
                """
                UPDATE samples
                SET master_row_id = rowid
                WHERE master_row_id IS NULL
                """
            )
            wrote = True
        idx_exists = conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'index' AND name = 'idx_samples_master_row_id'
            LIMIT 1
            """
        ).fetchone()
        if not idx_exists:
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_samples_master_row_id
                ON samples(master_row_id)
                WHERE master_row_id IS NOT NULL
                """
            )
            wrote = True
        conn.commit()
        return {"ok": True, "has_column": True, "wrote": wrote, "error": None}
    except sqlite3.OperationalError as exc:
        # Likely locked by another writer — check if column already usable via retry read.
        msg = str(exc)
        try:
            cols = {
                str(r[1]) for r in conn.execute("PRAGMA table_info(samples)").fetchall()
            }
            if "master_row_id" in cols:
                return {
                    "ok": True,
                    "has_column": True,
                    "wrote": False,
                    "error": None,
                }
        except sqlite3.Error:
            pass
        return {
            "ok": False,
            "has_column": False,
            "wrote": False,
            "error": msg,
        }
    except sqlite3.Error as exc:
        return {
            "ok": False,
            "has_column": False,
            "wrote": False,
            "error": str(exc),
        }
    finally:
        conn.close()


def master_has_row_id_column(
    master_db_path: str,
    *,
    timeout_sec: float = 10.0,
) -> bool:
    """Read-only check whether samples.master_row_id exists."""
    try:
        conn = sqlite3.connect(
            f"file:{master_db_path}?mode=ro",
            uri=True,
            timeout=float(timeout_sec),
        )
        conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)}")
    except sqlite3.Error:
        try:
            conn = sqlite3.connect(master_db_path, timeout=float(timeout_sec))
            conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)}")
        except sqlite3.Error:
            return False
    try:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(samples)").fetchall()}
        return "master_row_id" in cols
    except sqlite3.Error:
        return False
    finally:
        conn.close()


class PredictionFeatureStore:
    """Transparent embedded / referenced feature access for one lab DB."""

    def __init__(
        self,
        lab_db_path: str,
        *,
        data_dir: str | None = None,
        store: ModelLabStore | None = None,
    ) -> None:
        self.lab_db_path = lab_db_path
        self.data_dir = data_dir
        self._owns_store = store is None
        self._store = store
        self._attached_path: str | None = None

    def __enter__(self) -> PredictionFeatureStore:
        if self._store is None:
            self._store = ModelLabStore(self.lab_db_path)
            self._store.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.detach()
        if self._owns_store and self._store is not None:
            self._store.close()
            self._store = None

    @property
    def store(self) -> ModelLabStore:
        if self._store is None:
            raise RuntimeError("PredictionFeatureStore not open")
        return self._store

    @classmethod
    def from_store(
        cls,
        store: ModelLabStore,
        *,
        data_dir: str | None = None,
    ) -> PredictionFeatureStore:
        return cls(store.db_path, data_dir=data_dir, store=store)

    def summary(self) -> dict[str, Any]:
        return self.store.read_prediction_summary() or {}

    def lab_info(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        info = self.store.read_info()
        if info is not None and hasattr(info, "to_overview_dict"):
            out = dict(info.to_overview_dict())
        try:
            row = self.store.conn.execute(
                "SELECT master_dataset_id, master_db_path FROM model_lab_info WHERE id = 1"
            ).fetchone()
            if row:
                if row[0]:
                    out["master_dataset_id"] = row[0]
                if row[1]:
                    out["master_db_path"] = row[1]
        except sqlite3.Error:
            pass
        return out

    def storage_mode(self) -> str:
        summary = self.summary()
        mode = str(summary.get("feature_storage_mode") or "").strip().lower()
        if mode in (FEATURE_STORAGE_EMBEDDED, FEATURE_STORAGE_REFERENCED):
            return mode
        cols = self.store._prediction_table_columns()
        has_sf = any(str(c).startswith("sf_") for c in cols)
        has_fk = "master_row_id" in cols and self.resolved_master_path() is not None
        if has_fk and not has_sf:
            return FEATURE_STORAGE_REFERENCED
        return FEATURE_STORAGE_EMBEDDED

    def is_referenced(self) -> bool:
        return self.storage_mode() == FEATURE_STORAGE_REFERENCED

    def master_pointer(self) -> dict[str, str | None]:
        summary = self.summary()
        info = self.lab_info()
        path = (
            summary.get("master_db_path")
            or info.get("master_db_path")
            or (info.get("artifact_pointers") or {}).get("master_db_path")
        )
        mid = summary.get("master_dataset_id") or info.get("master_dataset_id")
        return {
            "master_db_path": str(path).strip() if path else None,
            "master_dataset_id": str(mid).strip() if mid else master_dataset_id_from_path(
                str(path) if path else None
            ),
        }

    def resolved_master_path(self) -> str | None:
        ptr = self.master_pointer()
        return resolve_master_db_path(
            data_dir=self.data_dir,
            master_db_path=ptr.get("master_db_path"),
            lab_db_path=self.lab_db_path,
        )

    def feature_map(self) -> list[tuple[str, str]]:
        """Return [(feature_name, physical_column), ...] for the active mode."""
        cols = self.store._prediction_table_columns()
        summary = self.summary()
        mapping: dict[str, str] = {}
        raw = summary.get("feature_columns_json")
        if raw:
            try:
                loaded = json.loads(str(raw))
                if isinstance(loaded, dict):
                    mapping = {str(k): str(v) for k, v in loaded.items()}
            except (TypeError, json.JSONDecodeError):
                mapping = {}

        if self.is_referenced():
            if mapping:
                return sorted(mapping.items(), key=lambda kv: kv[0].lower())
            # Identity map from selected features when JSON missing
            info = self.lab_info()
            feats = info.get("selected_features") or []
            if isinstance(feats, list) and feats:
                return [(str(f), str(f)) for f in feats if str(f).strip()]
            return []

        out: list[tuple[str, str]] = []
        if mapping:
            for name, col in sorted(mapping.items(), key=lambda kv: kv[0].lower()):
                if col in cols:
                    out.append((name, col))
            return out
        for col in sorted(c for c in cols if str(c).startswith("sf_")):
            out.append((col.removeprefix("sf_"), col))
        return out

    def feature_sql(
        self,
        feature_name: str,
        *,
        pred_alias: str = "p",
        master_alias: str = "m",
    ) -> str:
        """Qualified SQL expression for one feature (mode-aware)."""
        name = str(feature_name or "").strip()
        pairs = dict(self.feature_map())
        physical = pairs.get(name, name)
        if not _safe_ident(physical):
            raise ValueError(f"Unsafe feature column: {physical}")
        if self.is_referenced():
            return f'{master_alias}."{physical}"'
        # Embedded: physical is sf_* on prediction table
        if pred_alias:
            return f'{pred_alias}."{physical}"'
        return f'"{physical}"'

    def from_clause(self, *, pred_alias: str = "p") -> str:
        if self.is_referenced():
            return (
                f'prediction_dataset {pred_alias} '
                f'LEFT JOIN {_MASTER_ATTACH}.samples m '
                f'ON {pred_alias}.master_row_id = m.master_row_id'
            )
        if pred_alias:
            return f"prediction_dataset {pred_alias}"
        return "prediction_dataset"

    def attach(self) -> bool:
        """ATTACH master DB when in referenced mode. No-op for embedded."""
        if not self.is_referenced():
            return False
        path = self.resolved_master_path()
        if not path:
            raise FileNotFoundError(
                "Referenced prediction dataset is missing master_db_path "
                "(or the master file was moved)."
            )
        if self._attached_path == path:
            return True
        if self._attached_path:
            self.detach()
        self.store.conn.execute(f'ATTACH DATABASE ? AS "{_MASTER_ATTACH}"', (path,))
        self._attached_path = path
        return True

    def detach(self) -> None:
        if self._attached_path is None or self._store is None:
            return
        try:
            self._store.conn.execute(f'DETACH DATABASE "{_MASTER_ATTACH}"')
        except sqlite3.Error:
            pass
        self._attached_path = None

    def fetch_rows(
        self,
        *,
        outcome_cols: list[str] | None = None,
        feature_names: list[str] | None = None,
        where_sql: str = "",
        where_args: list[Any] | None = None,
        limit: int | None = None,
        require_feature_nonnull: str | None = None,
        order_by_sql: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch prediction outcome columns + selected features as dict rows.

        Feature keys use registry names (not ``sf_*``). Joins master when needed.
        """
        self.store.ensure_prediction_schema()
        cols = self.store._prediction_table_columns()
        pairs = self.feature_map()
        if feature_names is None:
            feat_pairs = pairs
        else:
            wanted = set(feature_names)
            feat_pairs = [(n, c) for n, c in pairs if n in wanted]

        pred_cols = [c for c in (outcome_cols or []) if c in cols]
        select_parts: list[str] = [f'p."{c}" AS "{c}"' for c in pred_cols]
        aliases: list[str] = list(pred_cols)

        if self.is_referenced():
            self.attach()
            for name, physical in feat_pairs:
                if not _safe_ident(physical):
                    continue
                select_parts.append(f'm."{physical}" AS "{name}"')
                aliases.append(name)
        else:
            for name, physical in feat_pairs:
                if physical not in cols or not _safe_ident(physical):
                    continue
                select_parts.append(f'p."{physical}" AS "{name}"')
                aliases.append(name)

        if not select_parts:
            return []

        where = ""
        args = list(where_args or [])
        clauses: list[str] = []
        if where_sql.strip():
            clauses.append(f"({where_sql})")
        if require_feature_nonnull:
            expr = self.feature_sql(require_feature_nonnull)
            clauses.append(f"{expr} IS NOT NULL")
        if clauses:
            where = " WHERE " + " AND ".join(clauses)

        order = ""
        if order_by_sql and order_by_sql.strip():
            # Allow only simple ORDER BY expressions from trusted callers
            order = f" ORDER BY {order_by_sql.strip()}"
        lim = f" LIMIT {int(limit)}" if limit is not None else ""
        sql = (
            f"SELECT {', '.join(select_parts)} "
            f"FROM {self.from_clause(pred_alias='p')}{where}{order}{lim}"
        )
        raw = self.store.conn.execute(sql, args).fetchall()
        return [{aliases[i]: row[i] for i in range(len(aliases))} for row in raw]

    def sql_feature_avg(
        self,
        feature_pairs: list[tuple[str, str]],
        *,
        where_sql: str = "",
        where_args: list[Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Per-feature AVG + count for a cohort (JOIN-aware)."""
        if self.is_referenced():
            self.attach()
        where = f" WHERE ({where_sql})" if where_sql.strip() else ""
        args = list(where_args or [])
        out: dict[str, dict[str, Any]] = {}
        for name, physical in feature_pairs:
            if not _safe_ident(physical):
                out[name] = {"mean": None, "n": 0}
                continue
            if self.is_referenced():
                expr = f'm."{physical}"'
            else:
                cols = self.store._prediction_table_columns()
                if physical not in cols:
                    out[name] = {"mean": None, "n": 0}
                    continue
                expr = f'p."{physical}"'
            and_nn = f" AND {expr} IS NOT NULL" if where else f" WHERE {expr} IS NOT NULL"
            sql = (
                f"SELECT AVG({expr}), COUNT({expr}) "
                f"FROM {self.from_clause(pred_alias='p')}{where}{and_nn}"
            )
            row = self.store.conn.execute(sql, args).fetchone()
            if not row or row[1] is None or int(row[1] or 0) == 0:
                out[name] = {"mean": None, "n": 0}
            else:
                out[name] = {
                    "mean": float(row[0]) if row[0] is not None else None,
                    "n": int(row[1] or 0),
                }
        return out


def referenced_feature_column_map(features: list[str]) -> dict[str, str]:
    """feature_name -> master column name (identity; no sf_ prefix)."""
    out: dict[str, str] = {}
    for feat in features:
        name = str(feat).strip()
        if not name:
            continue
        out[name] = name
    return out


def detect_feature_storage_mode(
    *,
    parquet_columns: set[str] | list[str],
    master_db_path: str | None,
    data_dir: str | None = None,
) -> tuple[str, str | None]:
    """
    Decide embedded vs referenced for a new prediction build.

    Referenced when a resolvable master DB exists (master_row_id can come from
    parquet or be stamped from master during the build).
    """
    resolved = resolve_master_db_path(data_dir=data_dir, master_db_path=master_db_path)
    if not resolved:
        return FEATURE_STORAGE_EMBEDDED, None
    # Referenced whenever master pointer resolves; builder stamps IDs if parquet
    # lacks master_row_id (lookup by trading_day+timestamp+token once).
    _ = parquet_columns  # reserved for future gating
    return FEATURE_STORAGE_REFERENCED, resolved


# Re-export for callers that still build embedded maps
def legacy_sanitize_feature_column(name: str) -> str:
    return sanitize_feature_column(name)
