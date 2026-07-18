import numpy as np
import pandas as pd

from mini_tpo.optimization_evaluation import (
    apply_guardrails,
    deterministic_best,
    select_case_solutions,
)


def _candidates():
    return pd.DataFrame({
        "factor_descuento": [.05, .10, .15], "duracion_dias": [5, 7, 14],
        "roi_esperado": [3.0, 2.8, 2.0], "roi_lower_90": [2.9, 2.7, 1.9],
        "roi_robusto": [2.9, 2.7, 1.9], "uplift_esperado": [.1, .2, .4],
        "volumen_incremental_esperado": [100., 250., 500.], "volumen_promo_esperado": [1100., 1250., 1500.],
        "local_support_count": [2, 3, 4], "nivel_soporte": ["soporte_medio"] * 3,
        "flag_duracion_observada": [True] * 3, "flag_descuento_en_rango_local": [True] * 3,
        "flag_extrapolacion": [False] * 3, "flag_uplift_invalido": [False] * 3,
        "flag_demanda_invalida": [False] * 3, "flag_model_disagreement": [False] * 3,
    })


def test_mathematical_robust_and_growth_objectives():
    frame = apply_guardrails(_candidates(), 1)
    solutions = select_case_solutions(frame, epsilon_roi=.01)
    assert solutions["mathematical"]["factor_descuento"] == .05
    assert solutions["robust"]["factor_descuento"] == .05
    assert solutions["growth"]["factor_descuento"] == .15


def test_tie_break_and_no_recommendation_are_deterministic():
    frame = _candidates()
    frame.loc[:1, "roi_robusto"] = 2.0
    best = deterministic_best(frame.iloc[:2], "roi_robusto", epsilon=.01)
    assert best["factor_descuento"] == .10  # more incremental volume
    blocked = frame.assign(nivel_soporte="soporte_insuficiente")
    blocked = apply_guardrails(blocked, 1)
    assert select_case_solutions(blocked, .01)["robust"] is None


def test_prediction_identities_have_no_future_targets():
    scenarios = pd.read_parquet("data/processed/optimization_scenarios.parquet")
    assert not {"uplift_real", "roi"}.intersection(scenarios.columns)
    assert np.allclose(scenarios["volumen_incremental_esperado"], scenarios["volumen_base_tanda"] * scenarios["uplift_esperado"])
    assert np.allclose(scenarios["volumen_promo_esperado"], scenarios["volumen_base_tanda"] * (1 + scenarios["uplift_esperado"]))
    assert (scenarios["uplift_lower_90"] <= scenarios["uplift_esperado"]).all()
    assert (scenarios["uplift_esperado"] <= scenarios["uplift_upper_90"]).all()
    assert (scenarios["roi_lower_90"] <= scenarios["roi_esperado"]).all()
    assert (scenarios["roi_esperado"] <= scenarios["roi_upper_90"]).all()
