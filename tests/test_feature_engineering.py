import json
from pathlib import Path

import numpy as np
import pandas as pd

from mini_tpo.constants import AUDIT_COLUMNS, POST_PROMO_COLUMNS, TARGET_ROI, TARGET_UPLIFT
from mini_tpo.data_loading import load_config
from mini_tpo.feature_engineering import (
    build_engineered_core,
    build_feature_catalog,
    build_preprocessor,
    load_feature_engineering_inputs,
    validate_input_alignment,
)
from mini_tpo.feature_sets import (
    FEATURES_CHANGE_WITH_DISCOUNT,
    FEATURES_CHANGE_WITH_DURATION,
    FEATURES_ENGINEERED_CORE,
    FEATURES_ENGINEERED_OPTIONAL,
)


CORE_PATH = Path("data/processed/model_features_engineered_core.parquet")
OPTIONAL_PATH = Path("data/processed/model_features_engineered_optional.parquet")


def test_engineered_formulas_are_correct():
    core = pd.read_parquet(CORE_PATH)
    assert np.allclose(core["duracion_semanas"], core["duracion_dias"] / 7)
    assert np.allclose(
        core["volumen_base_tanda"],
        core["volumen_base_sem"] * core["duracion_dias"] / 7,
    )
    assert np.allclose(core["elasticidad_abs"], core["elasticidad_estimada"].abs())
    assert np.allclose(
        core["descuento_x_elasticidad"],
        core["factor_descuento"] * core["elasticidad_abs"],
    )
    assert np.allclose(
        core["descuento_x_duracion"],
        core["factor_descuento"] * core["duracion_dias"],
    )
    assert np.allclose(core["factor_descuento_sq"], core["factor_descuento"].pow(2))
    assert np.allclose(core["duracion_dias_sq"], core["duracion_dias"].pow(2))
    assert np.allclose(core["log1p_volumen_base_sem"], np.log1p(core["volumen_base_sem"]))
    assert np.allclose(core["log1p_volumen_base_tanda"], np.log1p(core["volumen_base_tanda"]))


def test_engineered_alignment_order_and_leakage_guardrails():
    cfg = load_config()
    safe, targets, index = load_feature_engineering_inputs(cfg)
    validate_input_alignment(safe, targets, index)
    core = pd.read_parquet(CORE_PATH)
    optional = pd.read_parquet(OPTIONAL_PATH)
    assert len(core) == len(optional) == len(safe) == 2048
    assert core["row_id"].is_unique and optional["row_id"].is_unique
    assert core["row_id"].tolist() == safe["row_id"].tolist()
    assert optional["row_id"].tolist() == safe["row_id"].tolist()
    assert len(core.merge(targets, on="row_id", validate="one_to_one")) == len(core)
    assert len(core.merge(index, on="row_id", validate="one_to_one")) == len(core)
    forbidden = set(POST_PROMO_COLUMNS) | set(AUDIT_COLUMNS) | {TARGET_UPLIFT, TARGET_ROI}
    for frame in [core, optional]:
        assert not forbidden.intersection(frame.columns)
        assert not any(col.startswith("audit_") for col in frame.columns)
        assert not any("mean_uplift" in col or "mean_roi" in col for col in frame.columns)
        assert not frame.columns.duplicated().any()
        assert not np.isinf(frame.select_dtypes(include=[np.number]).to_numpy()).any()


def test_temporal_features_are_valid_and_consistent_for_same_date():
    core = pd.read_parquet(CORE_PATH)
    index = pd.read_parquet("data/processed/model_index.parquet")
    temporal = core.merge(index[["row_id", "fecha_inicio_tanda"]], on="row_id", validate="one_to_one")
    assert temporal["mes"].between(1, 12).all()
    assert temporal["trimestre"].between(1, 4).all()
    assert temporal["semana_anio"].between(1, 53).all()
    cyclical = ["mes_sin", "mes_cos", "semana_anio_sin", "semana_anio_cos"]
    assert temporal[cyclical].apply(lambda s: s.between(-1, 1).all()).all()
    assert (
        temporal.groupby("fecha_inicio_tanda")[
            ["mes", "trimestre", "semana_anio", *cyclical]
        ].nunique().max().max()
        == 1
    )


def _changed_columns(base: pd.DataFrame, scenario: pd.DataFrame) -> set[str]:
    changed = set()
    for column in FEATURES_ENGINEERED_CORE:
        left = base[column].iloc[0]
        right = scenario[column].iloc[0]
        if isinstance(left, (float, np.floating)):
            if not np.isclose(left, right):
                changed.add(column)
        elif left != right:
            changed.add(column)
    return changed


def test_optimizer_candidate_recalculation_changes_only_dependencies():
    cfg = load_config()
    safe, _, index = load_feature_engineering_inputs(cfg)
    base_input = safe.iloc[[0]].copy()
    base_index = index.loc[index["row_id"].eq(base_input["row_id"].iloc[0])].copy()
    base = build_engineered_core(base_input, base_index, cfg)

    discount_input = base_input.copy()
    discount_input["factor_descuento"] = min(0.40, float(base_input["factor_descuento"].iloc[0]) + 0.05)
    if np.isclose(discount_input["factor_descuento"].iloc[0], base_input["factor_descuento"].iloc[0]):
        discount_input["factor_descuento"] = 0.05
    discount_scenario = build_engineered_core(discount_input, base_index, cfg)
    assert _changed_columns(base, discount_scenario) == set(FEATURES_CHANGE_WITH_DISCOUNT)

    duration_input = base_input.copy()
    duration_input["duracion_dias"] = 21 if int(base_input["duracion_dias"].iloc[0]) != 21 else 14
    duration_scenario = build_engineered_core(duration_input, base_index, cfg)
    assert _changed_columns(base, duration_scenario) == set(FEATURES_CHANGE_WITH_DURATION)


def test_feature_engineering_artifacts_catalog_and_manifest():
    required = [
        CORE_PATH,
        OPTIONAL_PATH,
        Path("data/processed/feature_engineering_manifest.json"),
        Path("reports/tables/feature_engineering_catalog.csv"),
        Path("reports/tables/feature_optimizer_compatibility.csv"),
        Path("reports/tables/engineered_feature_correlation.csv"),
        Path("reports/tables/engineered_feature_redundancy.csv"),
        Path("reports/tables/engineered_feature_target_association.csv"),
        Path("reports/feature_engineering_report.md"),
    ]
    assert all(path.exists() for path in required)
    core = pd.read_parquet(CORE_PATH)
    optional = pd.read_parquet(OPTIONAL_PATH)
    assert core.columns.tolist() == ["row_id", *FEATURES_ENGINEERED_CORE]
    assert optional.columns.tolist() == [
        "row_id", *FEATURES_ENGINEERED_CORE, *FEATURES_ENGINEERED_OPTIONAL
    ]
    catalog = pd.read_csv("reports/tables/feature_engineering_catalog.csv")
    assert set(core.columns).union(optional.columns).issubset(set(catalog["nombre"]))
    assert catalog["interpretacion_tecnica"].str.strip().ne("").all()
    assert catalog["interpretacion_negocio"].str.strip().ne("").all()
    manifest = json.loads(
        Path("data/processed/feature_engineering_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["row_count"] == 2048
    assert manifest["feature_sets"]["FEATURES_ENGINEERED_CORE"] == FEATURES_ENGINEERED_CORE


def test_preprocessors_are_templates_not_fitted_globally():
    for family in ["linear", "tree"]:
        preprocessor = build_preprocessor(family)
        assert not hasattr(preprocessor, "transformers_")
    catalog = build_feature_catalog()
    future = catalog[catalog["estado"].eq("future fold-aware")]
    assert set(future["nombre"]) == {
        "mean_uplift_by_sku_shifted",
        "mean_roi_by_sku_chain_shifted",
    }
