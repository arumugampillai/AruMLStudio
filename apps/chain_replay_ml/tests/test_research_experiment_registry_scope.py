"""Production Model Registry excludes Analysis Lab experiment packages."""

from __future__ import annotations

import json
import os

from chain_replay_ml.training.registry import (
    is_research_experiment_model,
    list_trained_models,
)


def test_is_research_experiment_model_by_name_and_scope() -> None:
    assert is_research_experiment_model("Exp_Exp_001")
    assert is_research_experiment_model("Exp_019")
    assert is_research_experiment_model(
        "Custom",
        {"registry_scope": "experiment", "experiment_id": "Exp_001"},
    )
    assert is_research_experiment_model(
        "Custom",
        {"origin": "research_experiment"},
    )
    assert not is_research_experiment_model("Future_LTP_5m_WF_151f_XGB_2308_29")
    assert not is_research_experiment_model(
        "Prod_Model",
        {"status": "trained", "algorithm": "xgboost"},
    )


def test_list_trained_models_excludes_experiments_by_default(tmp_path) -> None:
    models = tmp_path / "models"
    prod = models / "Prod_WF_Model"
    exp = models / "Exp_Exp_042"
    prod.mkdir(parents=True)
    exp.mkdir(parents=True)
    (prod / "registry.json").write_text(
        json.dumps(
            {
                "model_name": "Prod_WF_Model",
                "status": "trained",
                "algorithm": "xgboost",
                "target": "future_ltp_5m",
            }
        ),
        encoding="utf-8",
    )
    (exp / "registry.json").write_text(
        json.dumps(
            {
                "model_name": "Exp_Exp_042",
                "status": "trained",
                "experiment_id": "Exp_042",
                "registry_scope": "experiment",
                "origin": "research_experiment",
            }
        ),
        encoding="utf-8",
    )

    data_dir = str(tmp_path)
    prod_only = list_trained_models(data_dir, lightweight=True)
    names = [r["model_name"] for r in prod_only]
    assert names == ["Prod_WF_Model"]
    assert all(not r.get("is_research_experiment") for r in prod_only)

    both = list_trained_models(
        data_dir, lightweight=True, include_experiments=True
    )
    both_names = {r["model_name"] for r in both}
    assert both_names == {"Prod_WF_Model", "Exp_Exp_042"}
    exp_row = next(r for r in both if r["model_name"] == "Exp_Exp_042")
    assert exp_row["is_research_experiment"] is True
    assert exp_row["registry_scope"] == "experiment"
