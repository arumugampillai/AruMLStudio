"""Optional CSV export artifact for Dataset Registry entries (Parquet remains canonical)."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any

from .writer import _safe_filename, datasets_dir, read_dataset_parquet


class CsvExportAlreadyExistsError(Exception):
    """Raised when a CSV export already exists and ``replace`` is false."""

    def __init__(self, dataset_name: str, csv_path: str) -> None:
        self.dataset_name = dataset_name
        self.csv_path = csv_path
        super().__init__(f"CSV export already exists for {dataset_name}: {csv_path}")


class CsvExportError(Exception):
    """CSV export failed; source Parquet was not modified."""


def csv_export_path(data_dir: str, safe_name: str) -> str:
    return os.path.join(datasets_dir(data_dir), f"{safe_name}.csv")


def csv_export_sidecar_path(data_dir: str, safe_name: str) -> str:
    return os.path.join(datasets_dir(data_dir), f"{safe_name}.csv-export.json")


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc if isinstance(doc, dict) else {}


def _load_source_metadata(data_dir: str, safe_name: str) -> dict[str, Any]:
    meta_path = os.path.join(datasets_dir(data_dir), f"{safe_name}.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        return _load_json(meta_path)
    except (OSError, json.JSONDecodeError):
        return {}


def _parquet_path(data_dir: str, safe_name: str) -> str:
    return os.path.join(datasets_dir(data_dir), f"{safe_name}.parquet")


def _source_identity(meta: dict[str, Any], safe_name: str) -> dict[str, Any]:
    return {
        "kind": "parquet",
        "dataset_name": safe_name,
        "dataset_id": str(meta.get("dataset_id") or safe_name),
        "dataset_version": str(
            meta.get("dataset_version") or meta.get("builder_version") or ""
        ),
    }


def _is_valid_export(csv_path: str, sidecar_path: str) -> bool:
    return os.path.isfile(csv_path) and os.path.isfile(sidecar_path)


def build_csv_export_metadata(data_dir: str, dataset_name: str) -> dict[str, Any]:
    """Registry metadata view: CSV export status and artifact details."""
    safe_name = _safe_filename(dataset_name)
    parquet_path = _parquet_path(data_dir, safe_name)
    meta = _load_source_metadata(data_dir, safe_name)
    source = _source_identity(meta, safe_name)
    source["path"] = parquet_path

    csv_path = csv_export_path(data_dir, safe_name)
    sidecar_path = csv_export_sidecar_path(data_dir, safe_name)
    filename = os.path.basename(csv_path)

    if not _is_valid_export(csv_path, sidecar_path):
        return {
            "status": "Not Generated",
            "csv_filename": filename,
            "csv_path": csv_path,
            "row_count": None,
            "column_count": None,
            "file_size_bytes": None,
            "generated_at": None,
            "source_dataset": source,
            "export_artifact": {
                "kind": "csv",
                "status": "Not Generated",
            },
        }

    sidecar = _load_json(sidecar_path)
    file_size = os.path.getsize(csv_path)
    return {
        "status": "Generated",
        "csv_filename": sidecar.get("csv_filename") or filename,
        "csv_path": sidecar.get("csv_path") or csv_path,
        "row_count": sidecar.get("row_count"),
        "column_count": sidecar.get("column_count"),
        "file_size_bytes": sidecar.get("file_size_bytes", file_size),
        "generated_at": sidecar.get("generated_at"),
        "source_dataset": sidecar.get("source_dataset") or source,
        "export_artifact": {
            "kind": "csv",
            "status": "Generated",
            "path": sidecar.get("csv_path") or csv_path,
            "filename": sidecar.get("csv_filename") or filename,
        },
    }


def generate_dataset_csv_export(
    data_dir: str,
    dataset_name: str,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    """Read Parquet, write a separate CSV + sidecar metadata. Parquet/json metadata unchanged."""
    safe_name = _safe_filename(dataset_name)
    parquet_path = _parquet_path(data_dir, safe_name)
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet not found for dataset {safe_name}")

    csv_path = csv_export_path(data_dir, safe_name)
    sidecar_path = csv_export_sidecar_path(data_dir, safe_name)
    out_dir = datasets_dir(data_dir)

    if _is_valid_export(csv_path, sidecar_path) and not replace:
        raise CsvExportAlreadyExistsError(safe_name, csv_path)

    parquet_stat_before = os.stat(parquet_path)
    meta = _load_source_metadata(data_dir, safe_name)
    source = _source_identity(meta, safe_name)
    source["path"] = parquet_path

    csv_part = os.path.join(out_dir, f"{safe_name}.csv.part")
    sidecar_part = os.path.join(out_dir, f"{safe_name}.csv-export.json.part")
    csv_bak = csv_path + ".bak"
    sidecar_bak = sidecar_path + ".bak"

    for part in (csv_part, sidecar_part, csv_bak, sidecar_bak):
        if os.path.isfile(part):
            try:
                os.remove(part)
            except OSError:
                pass

    had_existing = _is_valid_export(csv_path, sidecar_path)
    if replace and had_existing:
        shutil.copy2(csv_path, csv_bak)
        shutil.copy2(sidecar_path, sidecar_bak)

    try:
        df = read_dataset_parquet(parquet_path)
        row_count = int(len(df))
        column_count = int(len(df.columns))
        df.to_csv(csv_part, index=False)
        generated_at = datetime.now(timezone.utc).isoformat()
        sidecar_doc: dict[str, Any] = {
            "artifact_type": "csv_export",
            "export_artifact": {"kind": "csv", "status": "Generated"},
            "source_dataset": source,
            "dataset_name": safe_name,
            "csv_filename": os.path.basename(csv_path),
            "csv_path": csv_path,
            "row_count": row_count,
            "column_count": column_count,
            "file_size_bytes": os.path.getsize(csv_part),
            "generated_at": generated_at,
        }
        with open(sidecar_part, "w", encoding="utf-8") as fh:
            json.dump(sidecar_doc, fh, indent=2)
            fh.write("\n")

        os.replace(csv_part, csv_path)
        try:
            os.replace(sidecar_part, sidecar_path)
        except OSError:
            if not had_existing and os.path.isfile(csv_path):
                try:
                    os.remove(csv_path)
                except OSError:
                    pass
            raise

        for bak in (csv_bak, sidecar_bak):
            if os.path.isfile(bak):
                try:
                    os.remove(bak)
                except OSError:
                    pass
    except CsvExportAlreadyExistsError:
        raise
    except Exception as exc:
        for part in (csv_part, sidecar_part):
            if os.path.isfile(part):
                try:
                    os.remove(part)
                except OSError:
                    pass
        if os.path.isfile(csv_bak):
            try:
                os.replace(csv_bak, csv_path)
            except OSError:
                pass
        if os.path.isfile(sidecar_bak):
            try:
                os.replace(sidecar_bak, sidecar_path)
            except OSError:
                pass
        for bak in (csv_bak, sidecar_bak):
            if os.path.isfile(bak):
                try:
                    os.remove(bak)
                except OSError:
                    pass
        raise CsvExportError(str(exc)) from exc

    parquet_stat_after = os.stat(parquet_path)
    if (
        parquet_stat_before.st_size != parquet_stat_after.st_size
        or parquet_stat_before.st_mtime_ns != parquet_stat_after.st_mtime_ns
    ):
        raise CsvExportError("Parquet file changed during CSV export (unexpected)")

    return build_csv_export_metadata(data_dir, safe_name)


def delete_dataset_csv_export(data_dir: str, dataset_name: str) -> dict[str, Any]:
    """Remove CSV export artifact only (Parquet and registry metadata JSON stay)."""
    safe_name = _safe_filename(dataset_name)
    csv_path = csv_export_path(data_dir, safe_name)
    sidecar_path = csv_export_sidecar_path(data_dir, safe_name)
    deleted: list[str] = []
    for path in (csv_path, sidecar_path):
        if os.path.isfile(path):
            os.remove(path)
            deleted.append(os.path.basename(path))
    for suffix in (".csv.part", ".csv-export.json.part", ".csv.bak", ".csv-export.json.bak"):
        stray = os.path.join(datasets_dir(data_dir), f"{safe_name}{suffix}")
        if os.path.isfile(stray):
            try:
                os.remove(stray)
            except OSError:
                pass
    return {
        "dataset_name": safe_name,
        "deleted": deleted,
        "csv_export": build_csv_export_metadata(data_dir, safe_name),
    }
