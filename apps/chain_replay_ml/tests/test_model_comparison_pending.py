"""Tests for pending model comparison resolution."""

from __future__ import annotations

import unittest


def resolve_pending_comparison(
    names: list[str],
    pending: tuple[str, str] | None,
) -> tuple[tuple[str, str] | None, str | None]:
    if not pending:
        return None, None
    if not names:
        return None, None
    model_a, model_b = pending
    if model_a not in names or model_b not in names:
        return None, "One or both selected models are no longer available."
    return (model_a, model_b), None


class PendingComparisonTests(unittest.TestCase):
    def test_waits_for_names(self) -> None:
        self.assertEqual(resolve_pending_comparison([], ("A", "B")), (None, None))

    def test_resolves_valid_pair(self) -> None:
        pair, err = resolve_pending_comparison(["A", "B", "C"], ("A", "C"))
        self.assertIsNone(err)
        self.assertEqual(pair, ("A", "C"))

    def test_missing_model(self) -> None:
        pair, err = resolve_pending_comparison(["A"], ("A", "B"))
        self.assertIsNone(pair)
        self.assertIn("no longer available", err or "")


if __name__ == "__main__":
    unittest.main()
