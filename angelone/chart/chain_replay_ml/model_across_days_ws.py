"""WebSocket transport for model across-days comparison — /ws/model-across-days."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect

from chain_replay_ml.model_day_comparison import compare_model_across_trading_days

_lock = threading.RLock()
_ws: ModelAcrossDaysWsManager | None = None
_log = logging.getLogger(__name__)


class ModelAcrossDaysWsManager:
    def __init__(self, *, data_dir_resolver: Callable[[], str]) -> None:
        self._data_dir_resolver = data_dir_resolver
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[WebSocket] = set()
        self._active_thread: threading.Thread | None = None
        self._cancel = threading.Event()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        try:
            await websocket.send_json({"type": "MODEL_ACROSS_DAYS", "event": "HELLO_ACK"})
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._handle_message(websocket, msg)
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.discard(websocket)

    async def _handle_message(self, websocket: WebSocket, msg: dict[str, Any]) -> None:
        mtype = str(msg.get("type") or "").lower()
        if mtype == "model_across_days":
            await self._start_job(websocket, msg)
        elif mtype == "cancel":
            self._cancel.set()

    async def _start_job(self, websocket: WebSocket, msg: dict[str, Any]) -> None:
        with _lock:
            if self._active_thread and self._active_thread.is_alive():
                await websocket.send_json({
                    "type": "MODEL_ACROSS_DAYS",
                    "event": "ERROR",
                    "detail": "A comparison job is already running",
                })
                return

        model_name = str(msg.get("model_name") or "").strip()
        if not model_name:
            await websocket.send_json({
                "type": "MODEL_ACROSS_DAYS",
                "event": "ERROR",
                "detail": "model_name is required",
            })
            return

        underlying = str(msg.get("underlying") or "NIFTY").strip()
        position_limit = int(msg.get("position_limit") or 1)
        fill_missing = bool(msg.get("fill_missing"))
        max_workers = max(1, min(int(msg.get("max_workers") or 1), 4))
        max_compute_days = max(1, min(int(msg.get("max_compute_days") or 30), 60))
        date_from = str(msg.get("date_from") or "").strip() or None
        date_to = str(msg.get("date_to") or "").strip() or None

        self._cancel.clear()
        data_dir = self._data_dir_resolver()

        def on_progress(payload: dict[str, Any]) -> None:
            if self._cancel.is_set():
                return
            self._broadcast_json({
                "type": "MODEL_ACROSS_DAYS",
                "event": str(payload.get("event") or "PROGRESS"),
                **{k: v for k, v in payload.items() if k != "event"},
            })

        def run_job() -> None:
            try:
                result = compare_model_across_trading_days(
                    data_dir,
                    model_name=model_name,
                    underlying=underlying,
                    position_limit=position_limit,
                    fill_missing=fill_missing,
                    max_workers=max_workers,
                    max_compute_days=max_compute_days,
                    date_from=date_from,
                    date_to=date_to,
                    on_progress=on_progress,
                    cancel_event=self._cancel,
                )
                if self._cancel.is_set():
                    self._broadcast_json({
                        "type": "MODEL_ACROSS_DAYS",
                        "event": "CANCELLED",
                        **result,
                    })
                else:
                    self._broadcast_json({
                        "type": "MODEL_ACROSS_DAYS",
                        "event": "DONE",
                        **result,
                    })
            except Exception as exc:
                _log.exception("Model across-days job failed")
                self._broadcast_json({
                    "type": "MODEL_ACROSS_DAYS",
                    "event": "ERROR",
                    "detail": str(exc),
                })
            finally:
                with _lock:
                    self._active_thread = None

        thread = threading.Thread(target=run_job, daemon=True, name="model-across-days")
        with _lock:
            self._active_thread = thread
        thread.start()
        await websocket.send_json({
            "type": "MODEL_ACROSS_DAYS",
            "event": "STARTED",
            "model_name": model_name,
            "underlying": underlying,
            "fill_missing": fill_missing,
        })

    def _broadcast_json(self, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            _log.warning("Model across-days progress dropped — WebSocket event loop not running")
            return
        asyncio.run_coroutine_threadsafe(self._async_broadcast(payload), loop)

    async def _async_broadcast(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


def get_model_across_days_ws(*, data_dir_resolver: Callable[[], str]) -> ModelAcrossDaysWsManager:
    global _ws
    with _lock:
        if _ws is None:
            _ws = ModelAcrossDaysWsManager(data_dir_resolver=data_dir_resolver)
        return _ws


def register_model_across_days_ws_routes(app, *, data_dir_resolver: Callable[[], str]) -> None:
    ws_mgr = get_model_across_days_ws(data_dir_resolver=data_dir_resolver)

    @app.on_event("startup")
    async def _model_across_days_ws_startup() -> None:
        loop = asyncio.get_running_loop()
        ws_mgr.set_event_loop(loop)

    @app.websocket("/ws/model-across-days")
    async def ws_model_across_days(websocket: WebSocket) -> None:
        await ws_mgr.connect(websocket)
