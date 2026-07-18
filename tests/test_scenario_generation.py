import numpy as np
import pandas as pd

from mini_tpo.data_loading import load_config
from mini_tpo.scenario_generation import build_scenario_grid, discount_grid


def _contexts():
    return pd.DataFrame([{
        "case_id": "A", "id_material": "PD001", "subcadena": "Cadena01",
        "volumen_base_sem": 1000.0, "elasticidad_estimada": -1.5,
        "flag_secundario": "no", "fecha_referencia": pd.Timestamp("2025-12-29"),
    }])


def test_grid_is_deterministic_unique_and_inside_domain():
    cfg = load_config()
    discounts = discount_grid(.05, .40, .01)
    first = build_scenario_grid(_contexts(), discounts, [5, 7, 10, 14, 21], cfg)
    second = build_scenario_grid(_contexts(), discounts, [5, 7, 10, 14, 21], cfg)
    assert len(first) == 36 * 5
    assert not first.duplicated(["case_id", "factor_descuento", "duracion_dias"]).any()
    assert first["factor_descuento"].between(.05, .40).all()
    assert first["duracion_dias"].between(5, 21).all()
    pd.testing.assert_frame_equal(first, second)


def test_engineered_decision_features_and_fixed_context():
    cfg = load_config()
    grid = build_scenario_grid(_contexts(), [.10, .20], [7, 14], cfg)
    assert np.allclose(grid["volumen_base_tanda"], grid["volumen_base_sem"] * grid["duracion_dias"] / 7)
    assert np.allclose(grid["descuento_x_elasticidad"], grid["factor_descuento"] * grid["elasticidad_estimada"].abs())
    assert np.allclose(grid["descuento_x_duracion"], grid["factor_descuento"] * grid["duracion_dias"])
    assert grid[["id_material", "subcadena", "volumen_base_sem", "elasticidad_estimada", "flag_secundario"]].nunique().eq(1).all()
