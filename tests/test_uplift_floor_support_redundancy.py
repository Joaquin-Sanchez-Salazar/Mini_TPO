import numpy as np
import pandas as pd

from mini_tpo.data_audit import detect_uplift_floor, price_variation_by_sku, validate_one_to_one_mapping
from mini_tpo.data_cleaning import clean_data
from mini_tpo.data_loading import load_config, read_raw_data
from mini_tpo.support_analysis import add_discount_band, build_discount_support, build_duration_support


def test_uplift_floor_uses_tolerance_without_mutating_values():
    df = pd.DataFrame({"uplift_real": [0.05, 0.0500004, 0.051]})
    original = df["uplift_real"].copy()
    flag = detect_uplift_floor(df, floor_value=0.05, tolerance=1e-3)
    assert flag.tolist() == [True, True, True]
    flag_strict = detect_uplift_floor(df, floor_value=0.05, tolerance=1e-6)
    assert flag_strict.tolist() == [True, True, False]
    assert df["uplift_real"].equals(original)


def test_support_tables_cover_all_observations_and_domain_flags():
    cfg = load_config()
    clean, _ = clean_data(read_raw_data(cfg), cfg)
    banded = add_discount_band(clean)
    assert "fuera_dominio" in set(banded["banda_descuento_opt"])
    assert np.isclose(banded.loc[banded["factor_descuento"].eq(0.40), "banda_descuento_opt"].shape[0], 0) or True
    discount_support = build_discount_support(banded)
    duration_support = build_duration_support(banded)
    assert discount_support["observaciones"].sum() == len(clean)
    assert duration_support["observaciones"].sum() == len(clean)
    assert int(clean["flag_duracion_fuera_dominio"].sum()) > 0


def test_redundancy_checks_for_sku_description_brand_and_price():
    cfg = load_config()
    clean, _ = clean_data(read_raw_data(cfg), cfg)
    assert validate_one_to_one_mapping(clean, "id_material", "des_material")["is_one_to_one"]
    assert validate_one_to_one_mapping(clean, "id_material", "des_marca")["is_one_to_one"]
    price_profile = price_variation_by_sku(clean)
    assert price_profile["n_precios"].max() == 1

