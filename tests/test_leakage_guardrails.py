import pandas as pd

from mini_tpo.constants import AUDIT_COLUMNS, FEATURES_MODEL_BASE, LEAKAGE_COLUMNS, POST_PROMO_COLUMNS, TARGET_ROI, TARGET_UPLIFT


EXPECTED_SAFE_COLUMNS = {
    "row_id",
    "id_material",
    "subcadena",
    "factor_descuento",
    "duracion_dias",
    "volumen_base_sem",
    "elasticidad_estimada",
    "flag_secundario",
}


def test_no_leakage_columns_in_feature_lists():
    forbidden = set(LEAKAGE_COLUMNS) | set(POST_PROMO_COLUMNS) | {TARGET_UPLIFT, TARGET_ROI}
    assert not forbidden.intersection(FEATURES_MODEL_BASE)
    assert "volumen_base_sem" in FEATURES_MODEL_BASE
    assert "elasticidad_estimada" in FEATURES_MODEL_BASE


def test_safe_features_exclude_postpromo_audit_targets_and_redundant_description():
    features = pd.read_parquet("data/processed/model_features_safe.parquet")
    forbidden = set(LEAKAGE_COLUMNS) | set(POST_PROMO_COLUMNS) | set(AUDIT_COLUMNS) | {TARGET_UPLIFT, TARGET_ROI}
    assert not forbidden.intersection(features.columns)
    assert "des_material" not in features.columns
    assert "des_marca" not in features.columns
    assert "precio_base" not in features.columns
    assert set(features.columns) == EXPECTED_SAFE_COLUMNS
    assert features.shape == (2048, 8)


def test_targets_are_separate_from_features():
    features = pd.read_parquet("data/processed/model_features_safe.parquet")
    targets = pd.read_parquet("data/processed/model_targets.parquet")
    assert {TARGET_UPLIFT, TARGET_ROI}.issubset(targets.columns)
    assert not {TARGET_UPLIFT, TARGET_ROI}.intersection(features.columns)
