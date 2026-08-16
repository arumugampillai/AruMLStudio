"""Background resource monitoring for model training jobs."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class TrainingMonitor:
    """Collect CPU/RAM/GPU samples every second in a daemon thread."""

    CSV_COLUMNS = [
        "timestamp",
        "phase",
        "trial",
        "cpu_percent",
        "ram_used_gb",
        "ram_percent",
        "gpu_percent",
        "gpu_memory_used_gb",
        "gpu_memory_total_gb",
        "gpu_temperature",
        "gpu_power",
        "elapsed_seconds",
    ]

    def __init__(
        self,
        *,
        csv_path: str,
        interval_sec: float = 1.0,
        on_sample: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.csv_path = str(csv_path)
        self.interval_sec = max(1.0, float(interval_sec))
        self.on_sample = on_sample

        self._samples: list[dict[str, Any]] = []
        self._latest: dict[str, Any] = {}
        self._phase: str = "Initializing"
        self._trial: int | None = None
        self._total_trials: int | None = None
        self._best_score: float | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._running = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._errors: list[str] = []
        self._psutil_sample_warned = False
        self._psutil_hw_warned = False

        self._nvml = None
        self._nvml_handle = None
        self._gpu_static = self._detect_gpu_static()
        self._hardware = self._collect_hardware()
        self._software = self._collect_software()

    def start(self) -> None:
        if self._running.is_set():
            return
        self._started_at = time.monotonic()
        self._stop.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True, name="training-monitor")
        self._thread.start()

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._stopped_at = time.monotonic()
        self._running.clear()
        self._shutdown_nvml()
        try:
            self.save_csv()
        except Exception as exc:  # pragma: no cover - best effort
            self._errors.append(f"CSV write failed: {exc}")

    def set_phase(
        self,
        phase: str,
        *,
        trial: int | None = None,
        total_trials: int | None = None,
        best_score: float | None = None,
    ) -> None:
        with self._lock:
            self._phase = str(phase or "Unknown")
            if trial is not None:
                self._trial = int(trial)
            if total_trials is not None:
                self._total_trials = int(total_trials)
            if best_score is not None:
                self._best_score = float(best_score)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def save_csv(self) -> str:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            for row in self._samples:
                writer.writerow({key: row.get(key) for key in self.CSV_COLUMNS})
        return self.csv_path

    def build_summary(
        self,
        *,
        dataset: str,
        feature_count: int,
        target: str,
        device: str,
        tree_method: str,
        n_trials: int | None,
        best_trial: int | None,
        training_duration_sec: float,
        hpo_duration_sec: float = 0.0,
        git_commit: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        samples = list(self._samples)
        cpu_vals = [v for v in (_safe_float(s.get("cpu_percent")) for s in samples) if v is not None]
        ram_vals = [v for v in (_safe_float(s.get("ram_used_gb")) for s in samples) if v is not None]
        gpu_vals = [v for v in (_safe_float(s.get("gpu_percent")) for s in samples) if v is not None]
        gpu_mem_vals = [v for v in (_safe_float(s.get("gpu_memory_used_gb")) for s in samples) if v is not None]
        gpu_temp_vals = [v for v in (_safe_float(s.get("gpu_temperature")) for s in samples) if v is not None]
        gpu_power_vals = [v for v in (_safe_float(s.get("gpu_power")) for s in samples) if v is not None]

        total_elapsed = 0.0
        if self._started_at is not None:
            end = self._stopped_at or time.monotonic()
            total_elapsed = max(0.0, end - self._started_at)

        phases = [str(s.get("phase") or "").lower() for s in samples]
        hpo_secs = sum(1.0 for p in phases if "hyperparameter" in p)
        if hpo_duration_sec > 0:
            hpo_secs = float(hpo_duration_sec)
        final_secs = max(0.0, float(training_duration_sec))

        return {
            "model_name": model_name,
            "hardware": self._hardware,
            "software": {**self._software, "git_commit": git_commit},
            "training": {
                "dataset": dataset,
                "feature_count": int(feature_count),
                "target": target,
                "device": device,
                "tree_method": tree_method,
                "number_of_trials": int(n_trials or 0),
                "best_trial": best_trial,
                "training_duration_sec": round(total_elapsed, 2),
                "hyperparameter_search_duration_sec": round(hpo_secs, 2),
                "final_training_duration_sec": round(final_secs, 2),
                "total_training_duration_sec": round(total_elapsed, 2),
            },
            "resource_summary": {
                "average_cpu_percent": round(sum(cpu_vals) / len(cpu_vals), 2) if cpu_vals else None,
                "peak_cpu_percent": round(max(cpu_vals), 2) if cpu_vals else None,
                "average_ram_used_gb": round(sum(ram_vals) / len(ram_vals), 3) if ram_vals else None,
                "peak_ram_used_gb": round(max(ram_vals), 3) if ram_vals else None,
                "average_gpu_percent": round(sum(gpu_vals) / len(gpu_vals), 2) if gpu_vals else None,
                "peak_gpu_percent": round(max(gpu_vals), 2) if gpu_vals else None,
                "average_gpu_memory_used_gb": round(sum(gpu_mem_vals) / len(gpu_mem_vals), 3) if gpu_mem_vals else None,
                "peak_gpu_memory_used_gb": round(max(gpu_mem_vals), 3) if gpu_mem_vals else None,
                "peak_gpu_temperature_c": round(max(gpu_temp_vals), 1) if gpu_temp_vals else None,
                "peak_gpu_power_w": round(max(gpu_power_vals), 1) if gpu_power_vals else None,
            },
            "monitoring": {
                "csv_file": os.path.basename(self.csv_path),
                "samples": len(samples),
                "gpu_monitoring_available": self._gpu_static.get("available", False),
                "errors": list(self._errors),
            },
        }

    def save_summary(self, path: str, summary: dict[str, Any]) -> None:
        import json

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self._collect_sample()
                with self._lock:
                    self._samples.append(sample)
                    self._latest = sample
                if self.on_sample:
                    try:
                        self.on_sample(dict(sample))
                    except Exception:
                        pass
            except Exception as exc:  # pragma: no cover - monitor must not break training
                self._errors.append(f"sample failed: {exc}")
            self._stop.wait(self.interval_sec)

    def _collect_sample(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        cpu_percent = None
        vm_used_gb = None
        vm_percent = None
        vm_total_gb = None
        vm_available_gb = None
        try:
            import psutil

            vm = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)
            vm_used_gb = round(float(vm.used) / (1024 ** 3), 3)
            vm_percent = round(float(vm.percent), 2)
            vm_total_gb = round(float(vm.total) / (1024 ** 3), 3)
            vm_available_gb = round(float(vm.available) / (1024 ** 3), 3)
        except Exception:
            if not self._psutil_sample_warned:
                self._errors.append("psutil unavailable for sample")
                self._psutil_sample_warned = True
        elapsed = 0.0
        if self._started_at is not None:
            elapsed = max(0.0, time.monotonic() - self._started_at)
        with self._lock:
            phase = self._phase
            trial = self._trial
            total_trials = self._total_trials
            best_score = self._best_score
        gpu = self._collect_gpu_live()
        row = {
            "timestamp": now,
            "phase": phase,
            "trial": trial if trial is not None else "",
            "cpu_percent": round(float(cpu_percent), 2) if cpu_percent is not None else None,
            "ram_used_gb": vm_used_gb,
            "ram_percent": vm_percent,
            "gpu_percent": gpu.get("gpu_percent"),
            "gpu_memory_used_gb": gpu.get("gpu_memory_used_gb"),
            "gpu_memory_total_gb": gpu.get("gpu_memory_total_gb"),
            "gpu_temperature": gpu.get("gpu_temperature"),
            "gpu_power": gpu.get("gpu_power"),
            "elapsed_seconds": round(elapsed, 2),
            # Live-only extras for progress UI/dashboard.
            "ram_total_gb": vm_total_gb,
            "ram_available_gb": vm_available_gb,
            "trial_total": total_trials,
            "best_score": best_score,
            "gpu_model": self._gpu_static.get("gpu_model"),
        }
        return row

    def _collect_hardware(self) -> dict[str, Any]:
        total_ram_gb = None
        cores_phys = None
        cores_logical = None
        try:
            import psutil

            vm = psutil.virtual_memory()
            total_ram_gb = round(float(vm.total) / (1024 ** 3), 2)
            cores_phys = psutil.cpu_count(logical=False)
            cores_logical = psutil.cpu_count(logical=True)
        except Exception:
            if not self._psutil_hw_warned:
                self._errors.append("psutil unavailable for hardware profile")
                self._psutil_hw_warned = True
        gpu_static = self._gpu_static
        return {
            "cpu_model": platform.processor() or platform.machine(),
            "core_count_physical": cores_phys,
            "core_count_logical": cores_logical,
            "total_ram_gb": total_ram_gb,
            "gpu_model": gpu_static.get("gpu_model"),
            "gpu_vram_gb": gpu_static.get("gpu_vram_gb"),
            "cuda_version": gpu_static.get("cuda_version"),
            "driver_version": gpu_static.get("driver_version"),
        }

    def _collect_software(self) -> dict[str, Any]:
        import sys
        import xgboost

        app_version = "1.0.0"
        try:
            import master_dataset_tk

            app_version = getattr(master_dataset_tk, "__version__", "1.0.0")
        except Exception:
            app_version = "1.0.0"

        return {
            "python_version": sys.version.split()[0],
            "xgboost_version": getattr(xgboost, "__version__", None),
            "application_version": app_version,
            "platform": platform.platform(),
        }

    def _detect_gpu_static(self) -> dict[str, Any]:
        info = {
            "available": False,
            "gpu_model": None,
            "gpu_vram_gb": None,
            "driver_version": None,
            "cuda_version": None,
        }
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                return info
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_handle = handle
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8", errors="ignore")
            drv = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(drv, bytes):
                drv = drv.decode("utf-8", errors="ignore")
            info.update({
                "available": True,
                "gpu_model": gpu_name,
                "gpu_vram_gb": round(float(mem.total) / (1024 ** 3), 3),
                "driver_version": drv,
                "cuda_version": self._decode_cuda_version(pynvml.nvmlSystemGetCudaDriverVersion()),
            })
            return info
        except Exception:
            self._nvml = None
            self._nvml_handle = None

        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
            if out.returncode != 0 or not out.stdout.strip():
                return info
            first = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            mem_total_gb = round(float(parts[1]) / 1024.0, 3) if len(parts) > 1 else None
            info.update({
                "available": True,
                "gpu_model": parts[0] if parts else None,
                "gpu_vram_gb": mem_total_gb,
                "driver_version": parts[2] if len(parts) > 2 else None,
            })
            return info
        except Exception:
            return info

    def _collect_gpu_live(self) -> dict[str, Any]:
        none_gpu = {
            "gpu_percent": None,
            "gpu_memory_used_gb": None,
            "gpu_memory_total_gb": self._gpu_static.get("gpu_vram_gb"),
            "gpu_temperature": None,
            "gpu_power": None,
        }
        if self._nvml is not None and self._nvml_handle is not None:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                temp = self._nvml.nvmlDeviceGetTemperature(self._nvml_handle, self._nvml.NVML_TEMPERATURE_GPU)
                power_mw = self._nvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                return {
                    "gpu_percent": round(float(util.gpu), 2),
                    "gpu_memory_used_gb": round(float(mem.used) / (1024 ** 3), 3),
                    "gpu_memory_total_gb": round(float(mem.total) / (1024 ** 3), 3),
                    "gpu_temperature": round(float(temp), 1),
                    "gpu_power": round(float(power_mw) / 1000.0, 2),
                }
            except Exception:
                return none_gpu

        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
            if out.returncode != 0 or not out.stdout.strip():
                return none_gpu
            first = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            return {
                "gpu_percent": _safe_float(parts[0] if len(parts) > 0 else None),
                "gpu_memory_used_gb": round((_safe_float(parts[1]) or 0.0) / 1024.0, 3) if len(parts) > 1 else None,
                "gpu_memory_total_gb": round((_safe_float(parts[2]) or 0.0) / 1024.0, 3) if len(parts) > 2 else none_gpu["gpu_memory_total_gb"],
                "gpu_temperature": _safe_float(parts[3] if len(parts) > 3 else None),
                "gpu_power": _safe_float(parts[4] if len(parts) > 4 else None),
            }
        except Exception:
            return none_gpu

    def _shutdown_nvml(self) -> None:
        if self._nvml is None:
            return
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass
        finally:
            self._nvml = None
            self._nvml_handle = None

    @staticmethod
    def _decode_cuda_version(raw: int | bytes | str | None) -> str | None:
        if raw is None:
            return None
        if isinstance(raw, bytes):
            try:
                return raw.decode("utf-8", errors="ignore")
            except Exception:
                return None
        if isinstance(raw, str):
            return raw
        try:
            val = int(raw)
            major = val // 1000
            minor = (val % 1000) // 10
            return f"{major}.{minor}"
        except Exception:
            return None
