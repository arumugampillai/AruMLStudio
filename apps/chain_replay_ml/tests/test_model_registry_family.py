"""Model Registry family tabs — regression vs triple_barrier."""

from __future__ import annotations

import unittest

from chain_replay_ml.training.config import normalize_training_config
from chain_replay_ml.training.registry import resolve_model_registry_family


class ModelRegistryFamilyTests(unittest.TestCase):
    def test_resolve_by_label_strategy(self) -> None:
        self.assertEqual(
            resolve_model_registry_family({"label_strategy": "triple_barrier"}),
            "triple_barrier",
        )
        self.assertEqual(
            resolve_model_registry_family({"label_strategy": "fixed_horizon"}),
            "regression",
        )

    def test_resolve_by_label_id_target(self) -> None:
        self.assertEqual(
            resolve_model_registry_family({"target": "label_id"}),
            "triple_barrier",
        )
        self.assertEqual(
            resolve_model_registry_family({"target": "future_ltp_5m"}),
            "regression",
        )

    def test_training_config_persists_label_strategy(self) -> None:
        cfg = normalize_training_config(
            {
                "dataset": "ds",
                "target": "label_id",
                "prediction_type": "classification",
                "label_strategy": "triple_barrier",
                "label_strategy_params": {
                    "barrier_type": "percentage",
                    "tp_value": 20,
                    "sl_value": 10,
                },
                "features": ["f1"],
            }
        )
        self.assertEqual(cfg.label_strategy, "triple_barrier")
        self.assertEqual(cfg.label_strategy_params["barrier_type"], "percentage")
        doc = cfg.to_dict()
        self.assertEqual(doc["label_strategy"], "triple_barrier")
        self.assertEqual(
            resolve_model_registry_family(doc),
            "triple_barrier",
        )


if __name__ == "__main__":
    unittest.main()
