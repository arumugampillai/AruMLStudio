"""Wall-clock profiler for the full inference pipeline — accounts for every millisecond."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

_SLOW_MS = 50.0


@dataclass
class _SpanRecord:
    span_id: str
    name: str
    parent_id: str | None
    start_ms: float
    duration_ms: float
    meta: dict[str, Any] = field(default_factory=dict)


class InferenceTimingProfiler:
    """Hierarchical span profiler rooted at inference origin."""

    def __init__(self, *, label: str = "inference_total") -> None:
        self._origin = time.perf_counter()
        self._root_id = uuid.uuid4().hex[:12]
        self._stack: list[tuple[str, str, float]] = []  # span_id, name, t0
        self._spans: list[_SpanRecord] = []
        self._marks: list[dict[str, Any]] = []
        self._counters: dict[str, float] = {}
        self._root_open = True
        self._stack.append((self._root_id, label, self._origin))

    def mark(self, name: str, **meta: Any) -> None:
        now = time.perf_counter()
        abs_ms = round((now - self._origin) * 1000.0, 3)
        prev_ms = self._marks[-1]["abs_ms"] if self._marks else 0.0
        self._marks.append({
            "name": name,
            "abs_ms": abs_ms,
            "since_prev_ms": round(abs_ms - prev_ms, 3),
            **meta,
        })

    def begin(self, name: str, **meta: Any) -> str:
        span_id = uuid.uuid4().hex[:12]
        parent_id = self._stack[-1][0] if self._stack else None
        t0 = time.perf_counter()
        self._stack.append((span_id, name, t0))
        if meta:
            self._marks.append({
                "name": f"{name}:begin",
                "abs_ms": round((t0 - self._origin) * 1000.0, 3),
                "since_prev_ms": 0.0,
                **meta,
            })
        return span_id

    def end(self, name: str | None = None, **meta: Any) -> float:
        if not self._stack:
            return 0.0
        span_id, span_name, t0 = self._stack.pop()
        if name and name != span_name:
            _log.warning("timing profiler end mismatch: expected %s got %s", span_name, name)
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        parent_id = self._stack[-1][0] if self._stack else None
        self._spans.append(_SpanRecord(
            span_id=span_id,
            name=span_name,
            parent_id=parent_id,
            start_ms=round((t0 - self._origin) * 1000.0, 3),
            duration_ms=duration_ms,
            meta=dict(meta),
        ))
        return duration_ms

    def add_ms(self, bucket: str, ms: float, **meta: Any) -> None:
        self._counters[bucket] = round(float(self._counters.get(bucket, 0.0)) + float(ms), 3)
        if meta:
            key = f"{bucket}_meta"
            existing = self._counters.get(key)
            if not isinstance(existing, list):
                existing = []
            existing.append(meta)
            self._counters[key] = existing

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._origin) * 1000.0, 3)

    def finalize(self) -> dict[str, Any]:
        while len(self._stack) > 1:
            self.end()
        if self._root_open and self._stack:
            self._root_open = False
            self.end()

        wall_ms = round((time.perf_counter() - self._origin) * 1000.0, 3)
        span_sum_ms = round(sum(s.duration_ms for s in self._spans if s.parent_id == self._root_id), 3)
        # Top-level children of root; root span duration should ~= wall_ms
        root_span = next((s for s in self._spans if s.span_id == self._root_id), None)
        root_ms = root_span.duration_ms if root_span else wall_ms
        accounted_ms = round(sum(s.duration_ms for s in self._spans if s.parent_id != self._root_id or s.span_id != self._root_id), 3)

        slow = [
            {"name": s.name, "ms": s.duration_ms, "meta": s.meta}
            for s in sorted(self._spans, key=lambda x: x.duration_ms, reverse=True)
            if s.duration_ms >= _SLOW_MS and s.span_id != self._root_id
        ]
        for name, ms in self._counters.items():
            if name.endswith("_meta"):
                continue
            if float(ms) >= _SLOW_MS:
                slow.append({"name": name, "ms": float(ms), "meta": {}})

        for item in slow:
            _log.info(
                "inference timing >%.0fms: %s = %.1fms %s",
                _SLOW_MS,
                item["name"],
                item["ms"],
                item.get("meta") or "",
            )

        tree = self._build_tree(self._root_id)
        flat = [
            {
                "id": s.span_id,
                "name": s.name,
                "parent_id": s.parent_id,
                "start_ms": s.start_ms,
                "ms": s.duration_ms,
                "meta": s.meta,
            }
            for s in self._spans
        ]

        return {
            "wall_total_ms": wall_ms,
            "root_ms": root_ms,
            "span_children_sum_ms": span_sum_ms,
            "unaccounted_ms": round(max(0.0, wall_ms - root_ms), 3),
            "slow_stages": slow,
            "marks": self._marks,
            "counters": {k: v for k, v in self._counters.items() if not k.endswith("_meta")},
            "spans": flat,
            "tree": tree,
        }

    def _build_tree(self, node_id: str) -> list[dict[str, Any]]:
        children = [s for s in self._spans if s.parent_id == node_id]
        children.sort(key=lambda s: s.start_ms)
        out: list[dict[str, Any]] = []
        for s in children:
            node: dict[str, Any] = {
                "name": s.name,
                "ms": s.duration_ms,
                "start_ms": s.start_ms,
                "meta": s.meta,
            }
            sub = self._build_tree(s.span_id)
            if sub:
                node["children"] = sub
            out.append(node)
        return out
