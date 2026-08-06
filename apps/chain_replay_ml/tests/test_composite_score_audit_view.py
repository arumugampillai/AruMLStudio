import unittest

from master_dataset_tk.model_registry_detail import composite_score_audit_view


class CompositeScoreAuditViewTests(unittest.TestCase):
    def test_builds_table_from_structured_blocks(self) -> None:
        view = composite_score_audit_view(
            {
                "best_validation_composite": {
                    "score": 0.44,
                    "source": "Optuna validation during HPO",
                    "source_file": "walk_forward/best_parameters.json",
                    "source_path": "$.best_evaluation.composite_score",
                    "purpose": "Model selection",
                },
                "production_composite": {
                    "score": 0.452371,
                    "source": "Retrained production model",
                    "source_file": "metrics.json",
                    "source_path": "$.production_walk_forward",
                    "purpose": "Deployed model performance",
                    "champion": "baseline",
                },
                "difference_abs": 0.012371,
                "difference_pct": 2.81,
                "values_differ": True,
            }
        )
        self.assertIsNotNone(view)
        assert view is not None
        self.assertEqual(len(view["table_rows"]), 2)
        self.assertEqual(view["table_rows"][0][0], "Best Validation Composite")
        self.assertEqual(view["table_rows"][0][1], "0.4400")
        self.assertIn("walk_forward/best_parameters.json", view["table_rows"][0][3])
        self.assertIn("champion: baseline", view["table_rows"][1][2])
        self.assertTrue(view["footnote_warn"])
        self.assertIn("Difference:", view["footnote"] or "")

    def test_returns_none_when_no_scores(self) -> None:
        self.assertIsNone(composite_score_audit_view({}))
        self.assertIsNone(
            composite_score_audit_view(
                {
                    "best_validation_composite": {"score": None},
                    "production_composite": {"score": None},
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
