"""Strategy protocol, day-chunk runner, immutable artifact writer (Phase 5 hardened)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, runtime_checkable

from .contracts import assert_run_meta_complete, sanitize_label_row
from .types import (
    ENGINE_VERSION,
    LabelBatchResult,
    LabelRunMeta,
    LabelSourceContext,
    LabelStrategyConfig,
    StrategyCapabilities,
    StrategyMetadata,
    TargetDefinitions,
    validate_config_against_schema,
)


@runtime_checkable
class OutcomeLabelStrategy(Protocol):
    """Common contract for all labeling strategies."""

    @property
    def metadata(self) -> StrategyMetadata: ...

    @property
    def capabilities(self) -> StrategyCapabilities: ...

    def get_config_schema(self) -> dict[str, Any]:
        """UI/param schema: {field: {type, default, ...}}. Strategy owns config."""
        ...

    def get_target_definitions(self) -> TargetDefinitions:
        """primary_target / display_target / label_encoding — training never guesses."""
        ...

    def build_labels(
        self,
        source: LabelSourceContext,
        samples: Any,
        config: LabelStrategyConfig,
    ) -> LabelBatchResult:
        """Compute labels for one chunk (typically one trading day)."""
        ...


@runtime_checkable
class DaySampleSource(Protocol):
    """Read-only day-chunked sample provider (Master / Prediction adapters)."""

    source_kind: str

    def iter_days(self) -> Iterable[str]: ...

    def load_day(self, day: str) -> Any: ...


@dataclass
class ArtifactPaths:
    """Paths for one immutable Training Dataset labeling run."""

    root: Path
    artifact_id: str
    artifact_dir: Path
    rows_path: Path
    run_meta_path: Path


class ImmutableArtifactWriter:
    """Append-only writer: mint a new artifact dir; never overwrite an existing one.

    Stores day batches as JSONL + atomic ``run_meta.json``.
    """

    def __init__(self, root: str | Path, artifact_id: str) -> None:
        self._root = Path(root)
        self._artifact_id = str(artifact_id)
        self._artifact_dir = self._root / self._artifact_id
        self._rows_path = self._artifact_dir / "rows.jsonl"
        self._run_meta_path = self._artifact_dir / "run_meta.json"
        self._days_written: list[str] = []
        self._row_count = 0
        self._valid_rows = 0
        self._invalid_rows = 0
        self._finalized = False
        self._opened = False

    @property
    def paths(self) -> ArtifactPaths:
        return ArtifactPaths(
            root=self._root,
            artifact_id=self._artifact_id,
            artifact_dir=self._artifact_dir,
            rows_path=self._rows_path,
            run_meta_path=self._run_meta_path,
        )

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def days_written(self) -> list[str]:
        return list(self._days_written)

    @property
    def valid_rows(self) -> int:
        return self._valid_rows

    @property
    def invalid_rows(self) -> int:
        return self._invalid_rows

    def open(self) -> None:
        if self._artifact_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing Training Dataset artifact: "
                f"{self._artifact_dir}"
            )
        self._artifact_dir.mkdir(parents=True, exist_ok=False)
        self._rows_path.touch(exist_ok=False)
        self._opened = True

    def append_day(self, day: str, batch: LabelBatchResult) -> None:
        if not self._opened:
            raise RuntimeError("call open() before append_day()")
        if self._finalized:
            raise RuntimeError("artifact already finalized")
        with self._rows_path.open("a", encoding="utf-8") as fh:
            for row in batch.rows:
                payload = sanitize_label_row(dict(row))
                payload.setdefault("_label_day", day)
                fh.write(json.dumps(payload, default=str) + "\n")
                self._row_count += 1
                if payload.get("is_valid") is False:
                    self._invalid_rows += 1
                else:
                    # Treat missing is_valid as valid (FH-style continuous targets).
                    self._valid_rows += 1
        self._days_written.append(day)

    def finalize(self, run_meta: LabelRunMeta) -> Path:
        if not self._opened:
            raise RuntimeError("call open() before finalize()")
        if self._finalized:
            raise RuntimeError("artifact already finalized")
        if self._run_meta_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing run_meta.json: {self._run_meta_path}"
            )
        created = run_meta.created_at_utc or datetime.now(timezone.utc).isoformat()
        meta = LabelRunMeta(
            strategy=run_meta.strategy,
            version=run_meta.version,
            engine_version=run_meta.engine_version,
            source=run_meta.source,
            params=dict(run_meta.params),
            rows=run_meta.rows if run_meta.rows else self._row_count,
            compute_time_sec=run_meta.compute_time_sec,
            supported_problem_types=list(run_meta.supported_problem_types),
            target_columns=list(run_meta.target_columns),
            target_definitions=dict(run_meta.target_definitions),
            days_processed=list(run_meta.days_processed or self._days_written),
            valid_rows=(
                run_meta.valid_rows
                if run_meta.valid_rows is not None
                else self._valid_rows
            ),
            invalid_rows=(
                run_meta.invalid_rows
                if run_meta.invalid_rows is not None
                else self._invalid_rows
            ),
            created_at_utc=created,
        )
        assert_run_meta_complete(meta)
        tmp = self._run_meta_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(meta.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._run_meta_path)
        self._finalized = True
        return self._run_meta_path


@dataclass
class DayChunkRunResult:
    artifact_id: str
    artifact_dir: Path
    run_meta: LabelRunMeta
    days_processed: list[str] = field(default_factory=list)


class DayChunkRunner:
    """Engine orchestration: day → label → append. Strategies stay chunk-agnostic."""

    def run(
        self,
        strategy: OutcomeLabelStrategy,
        source: DaySampleSource,
        config: LabelStrategyConfig,
        writer: ImmutableArtifactWriter,
        *,
        days: Iterable[str] | None = None,
        engine_version: str = ENGINE_VERSION,
    ) -> DayChunkRunResult:
        schema = strategy.get_config_schema()
        raw_params = dict(config.params)
        normalize = getattr(strategy, "normalize_config_params", None)
        if callable(normalize):
            raw_params = dict(normalize(raw_params))
        params = validate_config_against_schema(raw_params, schema)
        effective = LabelStrategyConfig(
            strategy_id=config.strategy_id or strategy.metadata.strategy_id,
            version=config.version or strategy.metadata.version,
            params=params,
        )
        if effective.strategy_id != strategy.metadata.strategy_id:
            raise ValueError(
                f"config.strategy_id={effective.strategy_id!r} does not match "
                f"strategy {strategy.metadata.strategy_id!r}"
            )

        day_list = list(days) if days is not None else list(source.iter_days())
        writer.open()
        t0 = time.perf_counter()
        last_batch: LabelBatchResult | None = None
        for day in day_list:
            samples = source.load_day(day)
            handles: dict[str, Any] = {}
            # Optional session close from day-source adapters.
            close_attr = getattr(source, "session_close_by_day", None)
            if isinstance(close_attr, dict) and day in close_attr:
                handles["session_close_ts"] = close_attr[day]
            ctx = LabelSourceContext(
                source_kind=source.source_kind,
                day=day,
                handles=handles,
            )
            batch = strategy.build_labels(ctx, samples, effective)
            writer.append_day(day, batch)
            last_batch = batch
            # Drop day reference promptly for streaming memory behavior.
            del samples
        elapsed = time.perf_counter() - t0

        caps = strategy.capabilities
        defs = strategy.get_target_definitions()
        if last_batch is not None:
            target_columns = list(last_batch.target_columns)
            target_definitions = last_batch.target_definitions.to_dict()
        else:
            target_columns = [defs.primary_target]
            if defs.display_target:
                target_columns.append(defs.display_target)
            target_definitions = defs.to_dict()

        run_meta = LabelRunMeta(
            strategy=strategy.metadata.strategy_id,
            version=strategy.metadata.version,
            engine_version=engine_version,
            source=source.source_kind,
            params=params,
            rows=writer.row_count,
            compute_time_sec=elapsed,
            supported_problem_types=sorted(caps.supported_problem_types),
            target_columns=target_columns,
            target_definitions=target_definitions,
            days_processed=list(day_list),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        writer.finalize(run_meta)
        finalized = LabelRunMeta.from_dict(
            json.loads(writer.paths.run_meta_path.read_text(encoding="utf-8"))
        )
        return DayChunkRunResult(
            artifact_id=writer.paths.artifact_id,
            artifact_dir=writer.paths.artifact_dir,
            run_meta=finalized,
            days_processed=list(day_list),
        )


def mint_artifact_id(strategy_id: str, *, suffix: str | None = None) -> str:
    """Create a unique Training Dataset artifact id (never reused for overwrite)."""
    import uuid

    token = suffix or uuid.uuid4().hex[:12]
    return f"training_dataset_{strategy_id}_{token}"


def create_immutable_writer(
    root: str | Path,
    strategy_id: str,
    *,
    suffix: str | None = None,
) -> ImmutableArtifactWriter:
    """Always mint a new artifact id — never reopen/overwrite an existing run."""
    return ImmutableArtifactWriter(root, mint_artifact_id(strategy_id, suffix=suffix))


def run_labeling(
    *,
    strategy: OutcomeLabelStrategy,
    source: DaySampleSource,
    root: str | Path,
    params: dict[str, Any] | None = None,
    days: Iterable[str] | None = None,
    engine_version: str = ENGINE_VERSION,
) -> DayChunkRunResult:
    """Production entry: new immutable artifact + day-chunked labeling run."""
    writer = create_immutable_writer(root, strategy.metadata.strategy_id)
    config = LabelStrategyConfig(
        strategy_id=strategy.metadata.strategy_id,
        version=strategy.metadata.version,
        params=dict(params or {}),
    )
    return DayChunkRunner().run(
        strategy,
        source,
        config,
        writer,
        days=days,
        engine_version=engine_version,
    )
