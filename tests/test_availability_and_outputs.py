from pathlib import Path

import pandas as pd


EXPECTED_TABLES = [
    "support_sku_chain.csv",
    "support_discount_by_sku_chain.csv",
    "support_duration_by_sku_chain.csv",
    "feature_availability_audit.csv",
    "uplift_floor_by_sku.csv",
    "uplift_floor_by_chain.csv",
    "uplift_floor_by_discount_band.csv",
    "uplift_floor_by_time.csv",
    "temporal_profile_monthly.csv",
    "temporal_profile_quarterly.csv",
    "sku_history_profile.csv",
    "cardinality_and_variance_profile.csv",
    "top_10_roi.csv",
    "bottom_10_roi.csv",
    "discount_uplift_summary.csv",
    "discount_roi_summary.csv",
    "duration_uplift_summary.csv",
    "duration_roi_summary.csv",
    "elasticity_discount_uplift_matrix.csv",
    "elasticity_discount_count_matrix.csv",
    "secondary_promo_summary.csv",
    "baseline_impact_summary.csv",
]


def test_assumed_prepromo_variables_are_confirmed_and_postpromo_excluded():
    audit = pd.read_csv("reports/tables/feature_availability_audit.csv")
    confirmed = audit[audit["variable"].isin(["volumen_base_sem", "elasticidad_estimada"])]
    assert set(confirmed["estado"]) == {"confirmada por supuesto de la prueba"}
    assert set(confirmed["decision_modelado"]) == {"baseline"}
    post = audit[audit["variable"].isin(["volumen_promo", "venta_promo", "inversion_promo", "roi"])]
    assert set(post.loc[post["variable"].ne("roi"), "estado"]) == {"excluida"}
    assert set(post.loc[post["variable"].eq("roi"), "estado"]) == {"target"}


def test_expected_outputs_exist_and_load():
    for name in EXPECTED_TABLES:
        path = Path("reports/tables") / name
        assert path.exists(), name
        pd.read_csv(path)
    for path in [
        Path("data/processed/model_features_safe.parquet"),
        Path("data/processed/model_targets.parquet"),
        Path("data/processed/model_index.parquet"),
    ]:
        assert path.exists()
        assert len(pd.read_parquet(path)) > 0
    assert Path("reports/eda_data_quality_report.md").exists()
    assert Path("reports/data_preparation_report.md").exists()
