"""Phase 2 baseline regression report — frozen gate before RollController (Commit 5).

Run:
    cd angelone/chart
    python -m chain_replay_ml.dataset_builder.phase2_baseline_report
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import unittest
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence

PHASE2_BASELINE_TAG = "Phase 2 baseline"

# Spot RV methods inside RvControllerTests (EffectiveSessionStartTests are separate).
_SPOT_RV_METHODS = frozenset({
    "test_spot_rv_warmup_periods_from_registry",
    "test_spot_rv_dedupes_one_sample_per_timestamp",
    "test_rows_1_to_30_null_for_spot_rv_5m",
    "test_row_31_first_valid_spot_rv_5m",
    "test_spot_rv_interleaved_tokens_do_not_reset",
    "test_token_gap_does_not_reset_spot_rv",
})

_NO_HIDDEN_HISTORY_METHODS = frozenset({
    "test_permanent_regression_no_hidden_history",
    "test_permanent_regression_no_hidden_iv_history",
})

_GAP_RESET_METHODS = frozenset({
    "test_gap_resets_warmup",
    "test_token_gap_resets_rv_warmup",
    "test_gap_reset_clears_iv_controllers",
    "test_gap_resets_iv_history",
    "test_gap_reset_clears_roll_state",
    "test_interleaved_tokens_no_false_gap_reset",
    "test_row_gap_exceeds_3s_under_10s_limit",
})

_MONOTONIC_TIMESTAMP_METHODS = frozenset({
    "test_permanent_monotonic_timestamp_rolling_controller",
    "test_permanent_monotonic_timestamp_iv_history",
    "test_permanent_non_monotonic_gap_replay_fails",
})

_RESET_COMPLETENESS_METHODS = frozenset({
    "test_permanent_reset_complete_all_rolling_controller_types",
    "test_permanent_reset_complete_rv_first_return_on_second_update",
    "test_permanent_reset_complete_iv_history",
    "test_permanent_reset_complete_token_controllers_reset_all",
    "test_permanent_reset_complete_spot_controllers",
})


@dataclass
class CaseResult:
    test_id: str
    status: str
    detail: str = ""


@dataclass
class BaselineRun:
    results: list[CaseResult] = field(default_factory=list)
    stdout: str = ""

    def by_id(self) -> dict[str, CaseResult]:
        return {r.test_id: r for r in self.results}

    def count_status(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)


class _CollectingResult(unittest.TestResult):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.case_results: list[CaseResult] = []

    def _record(self, test: unittest.TestCase, status: str, detail: str = "") -> None:
        self.case_results.append(CaseResult(test.id(), status, detail))

    def addSuccess(self, test: unittest.TestCase) -> None:
        super().addSuccess(test)
        self._record(test, "PASS")

    def addFailure(self, test: unittest.TestCase, err: Any) -> None:
        super().addFailure(test, err)
        self._record(test, "FAIL", self._exc_info(err))

    def addError(self, test: unittest.TestCase, err: Any) -> None:
        super().addError(test, err)
        self._record(test, "ERROR", self._exc_info(err))

    def addSkip(self, test: unittest.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self._record(test, "SKIP", reason or "")

    @staticmethod
    def _exc_info(err: Any) -> str:
        if not err:
            return ""
        exc_type, exc_val, _ = err
        return f"{getattr(exc_type, '__name__', exc_type)}: {exc_val}"


def _test_method_name(test_id: str) -> str:
    return test_id.rsplit(".", 1)[-1]


def _test_class_name(test_id: str) -> str:
    parts = test_id.split(".")
    return parts[-2] if len(parts) >= 2 else ""


def build_phase2_baseline_suite() -> unittest.TestSuite:
    """Unittest suite frozen for Phase 2 baseline (pre RollController)."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    from chain_replay_ml.tests import test_controller_warmup_regression as warmup_mod
    from chain_replay_ml.tests import test_ema_reset_gap as ema_gap_mod
    from chain_replay_ml.tests import test_rolling_controllers as rolling_mod
    from chain_replay_ml.tests.test_replay_parallel_features import TestReplayParallelFeatures

    suite.addTests(loader.loadTestsFromModule(rolling_mod))
    suite.addTests(loader.loadTestsFromModule(warmup_mod))
    suite.addTests(loader.loadTestsFromModule(ema_gap_mod))
    suite.addTests(loader.loadTestsFromTestCase(TestReplayParallelFeatures))
    return suite


def run_phase2_baseline(*, verbosity: int = 0) -> BaselineRun:
    suite = build_phase2_baseline_suite()
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity, resultclass=_CollectingResult)
    result = runner.run(suite)
    assert isinstance(result, _CollectingResult)
    return BaselineRun(results=result.case_results, stdout=stream.getvalue())


def _category_status(
    test_ids: Sequence[str],
    results: Mapping[str, CaseResult],
    *,
    allow_skip: bool = False,
    skip_reason: str = "",
) -> tuple[str, str]:
    """Return (PASS|FAIL|SKIP, note)."""
    if not test_ids:
        return "SKIP", skip_reason or "No mapped tests"

    statuses = [results[tid].status for tid in test_ids if tid in results]
    missing = [tid for tid in test_ids if tid not in results]
    if missing:
        return "FAIL", f"missing results for {len(missing)} test(s)"

    if all(s == "SKIP" for s in statuses):
        return "SKIP", skip_reason or (results[test_ids[0]].detail if test_ids else "")

    if any(s in ("FAIL", "ERROR") for s in statuses):
        failed = [tid for tid in test_ids if results[tid].status in ("FAIL", "ERROR")]
        return "FAIL", failed[0].rsplit(".", 1)[-1]

    if any(s == "SKIP" for s in statuses):
        if allow_skip:
            return "PASS", ""
        return "SKIP", skip_reason or "unexpected SKIP"

    return "PASS", ""


def _tests_for_class(module: str, class_name: str, results: Mapping[str, CaseResult]) -> list[str]:
    prefix = f"{module}.{class_name}."
    return sorted(tid for tid in results if tid.startswith(prefix))


def _tests_for_methods(
    module: str,
    class_name: str,
    methods: frozenset[str],
    results: Mapping[str, CaseResult],
) -> list[str]:
    prefix = f"{module}.{class_name}."
    return sorted(tid for tid in results if tid.startswith(prefix) and _test_method_name(tid) in methods)


def _token_rv_tests(results: Mapping[str, CaseResult]) -> list[str]:
    module = "chain_replay_ml.tests.test_rolling_controllers"
    prefix = f"{module}.RvControllerTests."
    return sorted(
        tid for tid in results
        if tid.startswith(prefix) and _test_method_name(tid) not in _SPOT_RV_METHODS
    )


def _spot_rv_tests(results: Mapping[str, CaseResult]) -> list[str]:
    module = "chain_replay_ml.tests.test_rolling_controllers"
    spot = _tests_for_methods(module, "RvControllerTests", _SPOT_RV_METHODS, results)
    spot += _tests_for_class(module, "EffectiveSessionStartTests", results)
    return sorted(set(spot))


def _warmup_tests(results: Mapping[str, CaseResult]) -> list[str]:
    return _tests_for_class(
        "chain_replay_ml.tests.test_controller_warmup_regression",
        "ControllerWarmupRegressionTests",
        results,
    )


def _replay_parity_tests(results: Mapping[str, CaseResult]) -> list[str]:
    module = "chain_replay_ml.tests.test_ema_reset_gap"
    out: list[str] = []
    out += _tests_for_class(module, "SerialInterleavedGapTests", results)
    out += _tests_for_class(module, "ReplayLookupTokenTests", results)
    out += _tests_for_class(module, "TokenKeyNormalizationTests", results)
    return sorted(out)


def _parallel_unit_tests(results: Mapping[str, CaseResult]) -> list[str]:
    module = "chain_replay_ml.tests.test_replay_parallel_features"
    return _tests_for_class(module, "TestReplayParallelFeatures", results)


def _no_hidden_history_tests(results: Mapping[str, CaseResult]) -> list[str]:
    module = "chain_replay_ml.tests.test_rolling_controllers"
    prefix = f"{module}."
    return sorted(
        tid for tid in results
        if tid.startswith(prefix) and _test_method_name(tid) in _NO_HIDDEN_HISTORY_METHODS
    )


def _gap_reset_tests(results: Mapping[str, CaseResult]) -> list[str]:
    out: list[str] = []
    for tid in results:
        if _test_method_name(tid) in _GAP_RESET_METHODS:
            out.append(tid)
    return sorted(out)


def _monotonic_timestamp_tests(results: Mapping[str, CaseResult]) -> list[str]:
    module = "chain_replay_ml.tests.test_rolling_controllers"
    prefix = f"{module}.MonotonicTimestampInvariantTests."
    return sorted(
        tid for tid in results
        if tid.startswith(prefix) and _test_method_name(tid) in _MONOTONIC_TIMESTAMP_METHODS
    )


def _reset_completeness_tests(results: Mapping[str, CaseResult]) -> list[str]:
    module = "chain_replay_ml.tests.test_rolling_controllers"
    prefix = f"{module}.ControllerResetCompletenessTests."
    return sorted(
        tid for tid in results
        if tid.startswith(prefix) and _test_method_name(tid) in _RESET_COMPLETENESS_METHODS
    )


def _git_revision(chart_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(chart_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _version_info() -> dict[str, str]:
    chart_root = Path(__file__).resolve().parents[2]
    info = {
        "tag": PHASE2_BASELINE_TAG,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "git_revision": _git_revision(chart_root),
    }
    try:
        from chain_replay_ml.dataset_builder.classification_validate import GENERATOR_VERSION
        from chain_replay_ml.dataset_builder.controller_registry import FEATURE_REGISTRY_VERSION

        info["feature_registry_version"] = str(FEATURE_REGISTRY_VERSION)
        info["generator_version"] = str(GENERATOR_VERSION)
    except Exception:
        pass
    return info


def format_phase2_baseline_report(run: BaselineRun) -> str:
    results = run.by_id()
    module = "chain_replay_ml.tests.test_rolling_controllers"

    migration_rows = [
        ("EMA", _tests_for_class(module, "EmaControllerTests", results)),
        ("STD20", _tests_for_class(module, "StdControllerTests", results)),
        ("Token RV", _token_rv_tests(results)),
        ("Spot RV", _spot_rv_tests(results)),
        ("IV Window", _tests_for_class(module, "IvControllerTests", results)),
        ("IV History", _tests_for_class(module, "IvHistoryControllerTests", results)),
        ("Roll", _tests_for_class(module, "RollControllerTests", results)),
    ]

    regression_rows = [
        ("No hidden history", _no_hidden_history_tests(results), False, ""),
        ("Gap reset", _gap_reset_tests(results), False, ""),
        ("Monotonic timestamps", _monotonic_timestamp_tests(results), False, ""),
        ("Reset completeness", _reset_completeness_tests(results), False, ""),
        ("Warmup", _warmup_tests(results), False, ""),
        ("Replay parity", _replay_parity_tests(results), False, ""),
        (
            "Parallel parity",
            _parallel_unit_tests(results),
            False,
            "",
        ),
    ]

    total = len(run.results)
    passed = run.count_status("PASS")
    failed = run.count_status("FAIL") + run.count_status("ERROR")
    skipped = run.count_status("SKIP")

    lines: list[str] = []
    lines.append(PHASE2_BASELINE_TAG)
    lines.append("")
    lines.append("Controller Migration Status")
    for label, test_ids in migration_rows:
        status, note = _category_status(test_ids, results)
        suffix = f"  ({note})" if note and status != "PASS" else ""
        lines.append(f"{label:<12} {status}{suffix}")

    lines.append("")
    if failed == 0 and skipped == 0:
        lines.append(f"Total tests: {passed} / {total} PASS")
    elif failed == 0:
        lines.append(f"Total tests: {passed} / {total} PASS ({skipped} SKIP)")
    else:
        lines.append(f"Total tests: {passed} / {total} PASS, {failed} FAIL, {skipped} SKIP")

    lines.append("")
    lines.append("Regression:")
    for label, test_ids, allow_skip, skip_reason in regression_rows:
        status, note = _category_status(test_ids, results, allow_skip=allow_skip, skip_reason=skip_reason)
        suffix = f"  ({note})" if note and status not in ("PASS",) else ""
        lines.append(f"{label:<22} {status}{suffix}")

    return "\n".join(lines)


def write_phase2_baseline_markdown(report_text: str, path: Path | None = None) -> Path:
    if path is None:
        repo_root = Path(__file__).resolve().parents[4]
        path = repo_root / "docs" / "controllers" / "PHASE2_BASELINE.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    info = _version_info()
    header = [
        f"# {PHASE2_BASELINE_TAG}",
        "",
        f"- Timestamp: {info['timestamp_utc']}",
        f"- Git revision: `{info['git_revision']}`",
    ]
    if "feature_registry_version" in info:
        header.append(f"- FEATURE_REGISTRY_VERSION: {info['feature_registry_version']}")
    if "generator_version" in info:
        header.append(f"- GENERATOR_VERSION: {info['generator_version']}")
    header.append("")
    header.append("```")
    header.append(report_text)
    header.append("```")
    header.append("")
    path.write_text("\n".join(header), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 baseline regression report.")
    parser.add_argument(
        "--no-write-md",
        action="store_true",
        help="Do not write docs/controllers/PHASE2_BASELINE.md",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose unittest output")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run = run_phase2_baseline(verbosity=2 if args.verbose else 0)
    report = format_phase2_baseline_report(run)
    print(report)

    if not args.no_write_md:
        md_path = write_phase2_baseline_markdown(report)
        print(f"\nWrote {md_path}")

    failed = run.count_status("FAIL") + run.count_status("ERROR")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
