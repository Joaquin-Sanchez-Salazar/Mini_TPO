from mini_tpo.data_cleaning import clean_data
from mini_tpo.data_loading import load_config, read_raw_data


def test_cleaning_outputs_domain_dataset():
    cfg = load_config()
    clean_full, modeling = clean_data(read_raw_data(cfg), cfg)
    assert len(clean_full) >= len(modeling) > 0
    assert clean_full["flag_secundario"].isna().sum() == 0
    assert not modeling["flag_fuera_dominio_optimizacion"].any()
    assert "audit_uplift_recalculado" in clean_full.columns

