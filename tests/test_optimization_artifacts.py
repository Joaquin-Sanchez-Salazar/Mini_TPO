import json
from pathlib import Path

import pandas as pd

from mini_tpo.model_registry import load_model


def test_optimization_outputs_and_three_unique_skus():
    required = [
        "data/processed/optimization_scenarios.parquet",
        "reports/tables/selected_optimization_cases.csv",
        "reports/tables/optimization_contexts.csv",
        "reports/tables/optimal_recommendations.csv",
        "reports/tables/mathematical_roi_optima.csv",
        "reports/tables/profitable_growth_alternatives.csv",
        "reports/tables/optimization_pareto_frontier.csv",
        "reports/tables/optimization_sensitivity.csv",
        "reports/tables/optimization_grid_summary.csv",
        "models/uplift_production.joblib", "models/roi_production.joblib",
    ]
    assert all(Path(path).exists() for path in required)
    cases = pd.read_csv(required[1])
    contexts = pd.read_csv(required[2])
    recommendations = pd.read_csv(required[3])
    assert cases["id_material"].nunique() == len(cases) == 3
    assert len(recommendations.merge(contexts, on="case_id", validate="one_to_one")) == 3
    assert all(Path("reports/figures/optimization").glob("*.png"))


def test_production_models_are_registered_and_reloadable():
    registry = json.loads(Path("models/model_registry.json").read_text(encoding="utf-8"))
    production = [m for m in registry["models"] if m["role"] == "production_refit"]
    scenarios = pd.read_parquet("data/processed/optimization_scenarios.parquet")
    assert len(production) == 2
    for entry in production:
        model = load_model(Path(entry["artifact"]))
        assert len(model.predict(scenarios.iloc[:2][entry["features"]])) == 2
        assert entry["selection_changed"] is False
