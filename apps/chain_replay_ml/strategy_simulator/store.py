"""SQLite store for strategy runs and trades."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from .paths import strategy_runs_db_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrategyRunStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = strategy_runs_db_path(data_dir)
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        import os

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> StrategyRunStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("StrategyRunStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS strategy_runs (
                strategy_run_id TEXT PRIMARY KEY,
                prediction_run_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                strategy_version_id TEXT NOT NULL,
                strategy_config_hash TEXT,
                prediction_run_hash TEXT,
                model_id TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                scope TEXT NOT NULL DEFAULT 'all_folds',
                fold_id TEXT,
                fold_number INTEGER,
                created_on TEXT NOT NULL,
                finished_on TEXT,
                trade_count INTEGER DEFAULT 0,
                signal_count INTEGER DEFAULT 0,
                skipped_signals INTEGER DEFAULT 0,
                metrics_json TEXT,
                meta_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_strategy_runs_prediction
                ON strategy_runs(prediction_run_id, created_on DESC);
            CREATE INDEX IF NOT EXISTS idx_strategy_runs_strategy
                ON strategy_runs(strategy_id, created_on DESC);

            CREATE TABLE IF NOT EXISTS strategy_trades (
                trade_id TEXT PRIMARY KEY,
                strategy_run_id TEXT NOT NULL,
                prediction_run_id TEXT NOT NULL,
                fold_id TEXT,
                fold_number INTEGER,
                entry_prediction_id TEXT,
                exit_prediction_id TEXT,
                strategy_version_id TEXT NOT NULL,
                trading_day TEXT,
                token TEXT,
                strike REAL,
                option_type TEXT,
                entry_ts REAL,
                exit_ts REAL,
                entry_price REAL,
                exit_price REAL,
                qty INTEGER,
                gross_pnl REAL,
                fees REAL,
                net_pnl REAL,
                return_pct REAL,
                holding_seconds REAL,
                exit_reason TEXT,
                max_favorable_pct REAL,
                max_adverse_pct REAL,
                meta_json TEXT,
                FOREIGN KEY (strategy_run_id) REFERENCES strategy_runs(strategy_run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_strategy_trades_run
                ON strategy_trades(strategy_run_id, entry_ts);
            CREATE INDEX IF NOT EXISTS idx_strategy_trades_prediction
                ON strategy_trades(prediction_run_id, fold_id);
            """
        )

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        if d.get("metrics_json"):
            try:
                d["metrics"] = json.loads(d["metrics_json"])
            except json.JSONDecodeError:
                d["metrics"] = {}
            del d["metrics_json"]
        elif "metrics_json" in d:
            d["metrics"] = {}
            del d["metrics_json"]
        if d.get("meta_json"):
            try:
                d["meta"] = json.loads(d["meta_json"])
            except json.JSONDecodeError:
                d["meta"] = {}
            del d["meta_json"]
        elif "meta_json" in d:
            d["meta"] = {}
            del d["meta_json"]
        return d

    def create_run(self, doc: dict[str, Any]) -> dict[str, Any]:
        run_id = str(doc.get("strategy_run_id") or uuid.uuid4().hex)
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO strategy_runs (
                strategy_run_id, prediction_run_id, strategy_id, strategy_version_id,
                strategy_config_hash, prediction_run_hash, model_id, status, scope,
                fold_id, fold_number, created_on, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                doc["prediction_run_id"],
                doc["strategy_id"],
                doc["strategy_version_id"],
                doc.get("strategy_config_hash"),
                doc.get("prediction_run_hash"),
                doc.get("model_id"),
                doc.get("status") or "running",
                doc.get("scope") or "all_folds",
                doc.get("fold_id"),
                doc.get("fold_number"),
                now,
                json.dumps(doc.get("meta") or {}, default=str),
            ),
        )
        self.conn.commit()
        return self.get_run(run_id) or {"strategy_run_id": run_id}

    def finalize_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        trade_count: int,
        signal_count: int,
        skipped_signals: int,
        metrics: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """
            UPDATE strategy_runs
            SET status = ?, finished_on = ?, trade_count = ?,
                signal_count = ?, skipped_signals = ?, metrics_json = ?
            WHERE strategy_run_id = ?
            """,
            (
                status,
                _utc_now(),
                trade_count,
                signal_count,
                skipped_signals,
                json.dumps(metrics, default=str),
                run_id,
            ),
        )
        self.conn.commit()

    def insert_trades_batch(self, trades: Sequence[dict[str, Any]]) -> int:
        if not trades:
            return 0
        self.conn.executemany(
            """
            INSERT INTO strategy_trades (
                trade_id, strategy_run_id, prediction_run_id, fold_id, fold_number,
                entry_prediction_id, exit_prediction_id, strategy_version_id,
                trading_day, token, strike, option_type,
                entry_ts, exit_ts, entry_price, exit_price, qty,
                gross_pnl, fees, net_pnl, return_pct, holding_seconds,
                exit_reason, max_favorable_pct, max_adverse_pct, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    t["trade_id"],
                    t["strategy_run_id"],
                    t["prediction_run_id"],
                    t.get("fold_id"),
                    t.get("fold_number"),
                    t.get("entry_prediction_id"),
                    t.get("exit_prediction_id"),
                    t["strategy_version_id"],
                    t.get("trading_day"),
                    t.get("token"),
                    t.get("strike"),
                    t.get("option_type"),
                    t.get("entry_ts"),
                    t.get("exit_ts"),
                    t.get("entry_price"),
                    t.get("exit_price"),
                    t.get("qty"),
                    t.get("gross_pnl"),
                    t.get("fees"),
                    t.get("net_pnl"),
                    t.get("return_pct"),
                    t.get("holding_seconds"),
                    t.get("exit_reason"),
                    t.get("max_favorable_pct"),
                    t.get("max_adverse_pct"),
                    json.dumps(
                        {
                            **(t.get("meta") if isinstance(t.get("meta"), dict) else {}),
                            "direction": t.get("direction"),
                            "stop_price": t.get("stop_price"),
                            "target_price": t.get("target_price"),
                            "use_predicted_ltp": t.get("use_predicted_ltp"),
                            "stop_loss_pct": t.get("stop_loss_pct"),
                            "stop_risk_rupees": t.get("stop_risk_rupees"),
                            "expected_stop_loss_rupees": t.get("expected_stop_loss_rupees"),
                            "lowest_mark_price": t.get("lowest_mark_price"),
                            "highest_mark_price": t.get("highest_mark_price"),
                            "lowest_unrealized_pnl": t.get("lowest_unrealized_pnl"),
                            "stop_trigger_ltp": t.get("stop_trigger_ltp"),
                            "target_trigger_ltp": t.get("target_trigger_ltp"),
                            "sample_exit_ltp": t.get("sample_exit_ltp"),
                            "fill_at_sample_ltp": t.get("fill_at_sample_ltp"),
                            "exit_sample_index": t.get("exit_sample_index"),
                            "exit_row_index": t.get("exit_row_index"),
                            "gap_beyond_stop": t.get("gap_beyond_stop"),
                        },
                        default=str,
                    ),
                )
                for t in trades
            ],
        )
        self.conn.commit()
        return len(trades)

    def get_run(self, strategy_run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM strategy_runs WHERE strategy_run_id = ?", (strategy_run_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def list_runs(
        self,
        *,
        prediction_run_id: str | None = None,
        strategy_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if prediction_run_id:
            clauses.append("prediction_run_id = ?")
            params.append(prediction_run_id)
        if strategy_id:
            clauses.append("strategy_id = ?")
            params.append(strategy_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT * FROM strategy_runs
            {where}
            ORDER BY created_on DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def list_trades(
        self,
        strategy_run_id: str,
        *,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM strategy_trades
            WHERE strategy_run_id = ?
            ORDER BY entry_ts, trade_id
            LIMIT ? OFFSET ?
            """,
            (strategy_run_id, limit, offset),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            if d.get("meta_json"):
                try:
                    d["meta"] = json.loads(d["meta_json"])
                except json.JSONDecodeError:
                    d["meta"] = {}
                del d["meta_json"]
            meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
            for k, v in meta.items():
                if d.get(k) is None:
                    d[k] = v
            out.append(d)
        return out

    def count_trades(self, strategy_run_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM strategy_trades WHERE strategy_run_id = ?",
            (strategy_run_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def count_trades_for_fold(self, strategy_run_id: str, fold_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM strategy_trades WHERE strategy_run_id = ? AND fold_id = ?",
            (strategy_run_id, fold_id),
        ).fetchone()
        return int(row[0]) if row else 0
