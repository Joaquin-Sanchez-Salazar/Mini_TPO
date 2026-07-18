import json

import pandas as pd

from mini_tpo.constants import FEATURES_MODEL_BASE
from mini_tpo.data_cleaning import clean_data
from mini_tpo.data_loading import load_config, read_raw_data
from mini_tpo.feature_manifest import create_safe_feature_dataset


def test_row_id_unique_stable_and_present_in_artifacts():
    cfg = load_config()
    raw = read_raw_data(cfg)
    clean_a, modeling_a = clean_data(raw, cfg)
    clean_b, _ = clean_data(raw, cfg)
    assert clean_a["row_id"].is_unique
    assert clean_a["row_id"].tolist() == clean_b["row_id"].tolist()
    features, targets, index = create_safe_feature_dataset(modeling_a)
    for frame in [features, targets, index, modeling_a]:
        assert "row_id" in frame.columns
        assert frame["row_id"].is_unique


def test_safe_artifacts_can_join_one_to_one():
    features = pd.read_parquet("data/processed/model_features_safe.parquet")
    targets = pd.read_parquet("data/processed/model_targets.parquet")
    index = pd.read_parquet("data/processed/model_index.parquet")
    assert len(features) == len(targets) == len(index)
    merged = features.merge(targets, on="row_id", validate="one_to_one").merge(index, on="row_id", validate="one_to_one")
    assert len(merged) == len(features)
    assert not features.columns.duplicated().any()
    assert not targets.columns.duplicated().any()
    assert not index.columns.duplicated().any()


def test_feature_manifest_is_valid_json():
    with open("data/processed/feature_manifest.json", encoding="utf-8") as file:
        manifest = json.load(file)
    assert "variables" in manifest
    assert manifest["features_model_base"]
    variables = {item["name"]: item for item in manifest["variables"]}
    assert set(FEATURES_MODEL_BASE).issubset(variables)
    assert variables["volumen_base_sem"]["baseline_included"] is True
    assert variables["volumen_base_sem"]["availability"] == "confirmed_assumption"
    assert variables["elasticidad_estimada"]["baseline_included"] is True
    assert variables["elasticidad_estimada"]["availability"] == "confirmed_assumption"
    assert variables["roi"]["role"] == "target"
    assert variables["roi"]["baseline_included"] is False
    assert variables["roi"]["leakage_risk"] == "target"


def test_preparation_artifacts_remain_complete_and_readable():
    features = pd.read_parquet("data/processed/model_features_safe.parquet")
    targets = pd.read_parquet("data/processed/model_targets.parquet")
    index = pd.read_parquet("data/processed/model_index.parquet")
    assert features.shape == (2048, 8)
    assert len(features) == len(targets) == len(index)
