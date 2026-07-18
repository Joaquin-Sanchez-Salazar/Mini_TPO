import json
from pathlib import Path

import numpy as np
import pandas as pd

from mini_tpo.data_loading import load_config
from mini_tpo.model_registry import load_model
from mini_tpo.modeling import load_modeling_frame
from mini_tpo.paths import PROJECT_ROOT
from mini_tpo.temporal_validation import expanding_window_splits


def test_registry_models_reload_and_predict_with_declared_features():
    registry_path = Path("models/model_registry.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    frame = load_modeling_frame(PROJECT_ROOT, load_config())
    evaluation = [item for item in registry["models"] if item["role"] != "production_refit"]
    assert len(evaluation) == 4
    assert {(item["target"], item["role"]) for item in evaluation} == {
        ("uplift_real", "champion"), ("uplift_real", "challenger"),
        ("roi", "champion"), ("roi", "challenger"),
    }
    for item in registry["models"]:
        assert item["target"] not in item["features"]
    production = [item for item in registry["models"] if item["role"] == "production_refit"]
    assert len(production) == 2
    for item in production:
        model = load_model(Path(item["artifact"]))
        prediction = model.predict(frame.iloc[:3][item["features"]])
        assert prediction.shape == (3,)
        assert np.isfinite(prediction).all()


def test_oof_predictions_cover_every_temporal_validation_row_for_both_roles():
    cfg = load_config()
    frame = load_modeling_frame(PROJECT_ROOT, cfg)
    test_start = pd.Timestamp("2025-10-01")
    development = frame[frame["fecha_inicio_tanda"].lt(test_start)].reset_index(drop=True)
    folds = expanding_window_splits(development, n_splits=cfg["modeling"]["n_temporal_splits"])
    expected = set().union(*(set(development.iloc[fold.validation_indices]["row_id"]) for fold in folds))
    for target in ["uplift", "roi"]:
        oof = pd.read_parquet(f"data/processed/oof_predictions_{target}.parquet")
        assert set(oof["rol"]) == {"champion", "challenger"}
        assert oof.groupby("rol")["row_id"].apply(set).eq(expected).all()
        assert not oof.duplicated(["rol", "row_id"]).any()
        assert oof["fecha"].equals(oof["fecha_inicio_tanda"])


def test_final_test_artifacts_do_not_overlap_oof_dates():
    for target in ["uplift", "roi"]:
        oof = pd.read_parquet(f"data/processed/oof_predictions_{target}.parquet")
        final = pd.read_parquet(f"data/processed/final_test_predictions_{target}.parquet")
        assert pd.to_datetime(oof["fecha"]).max() < pd.to_datetime(final["fecha"]).min()
        assert final["decision_fijada_antes_test"].all() if "decision_fijada_antes_test" in final else True
