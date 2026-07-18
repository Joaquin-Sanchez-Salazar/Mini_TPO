from __future__ import annotations

import numpy as np
import pandas as pd


def oof_error_quantile(path, role: str = "champion", quantile: float = 0.90) -> float:
    predictions = pd.read_parquet(path)
    selected = predictions[predictions["rol"].eq(role)]
    if selected.empty:
        raise ValueError(f"No OOF predictions for role={role}")
    return float((selected["target_real"] - selected["prediccion"]).abs().quantile(quantile))


def add_prediction_outputs(
    scenarios: pd.DataFrame,
    uplift_model,
    roi_model,
    uplift_features: list[str],
    roi_features: list[str],
    q_uplift: float,
    q_roi: float,
    support_penalty_uplift: dict,
    support_penalty_roi: dict,
) -> pd.DataFrame:
    result = scenarios.copy()
    result["uplift_esperado"] = uplift_model.predict(result[uplift_features])
    result["roi_esperado"] = roi_model.predict(result[roi_features])
    result["volumen_incremental_esperado"] = (
        result["volumen_base_tanda"] * result["uplift_esperado"]
    )
    result["volumen_promo_esperado"] = result["volumen_base_tanda"] * (
        1 + result["uplift_esperado"]
    )
    result["uplift_lower_90"] = result["uplift_esperado"] - q_uplift
    result["uplift_upper_90"] = result["uplift_esperado"] + q_uplift
    result["roi_lower_90"] = result["roi_esperado"] - q_roi
    result["roi_upper_90"] = result["roi_esperado"] + q_roi
    result["volumen_incremental_lower_90"] = result["volumen_base_tanda"] * result["uplift_lower_90"]
    result["volumen_incremental_upper_90"] = result["volumen_base_tanda"] * result["uplift_upper_90"]
    result["volumen_promo_lower_90"] = result["volumen_base_tanda"] * (1 + result["uplift_lower_90"])
    result["volumen_promo_upper_90"] = result["volumen_base_tanda"] * (1 + result["uplift_upper_90"])
    result["penalizacion_soporte_uplift"] = result["nivel_soporte"].map(
        support_penalty_uplift
    ).fillna(max(support_penalty_uplift.values()))
    result["penalizacion_soporte_roi"] = result["nivel_soporte"].map(
        support_penalty_roi
    ).fillna(max(support_penalty_roi.values()))
    result["uplift_robusto"] = result["uplift_lower_90"] - result["penalizacion_soporte_uplift"]
    result["roi_robusto"] = result["roi_lower_90"] - result["penalizacion_soporte_roi"]
    result["flag_uplift_invalido"] = result["uplift_esperado"].lt(-1)
    result["flag_demanda_invalida"] = result["volumen_promo_esperado"].lt(0)
    result["flag_optimo_en_limite_descuento"] = result["factor_descuento"].isin([0.05, 0.40])
    result["flag_optimo_en_limite_duracion"] = result["duracion_dias"].isin([5, 21])
    numeric = result.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Optimization predictions contain NaN or infinite values")
    return result


def apply_guardrails(frame: pd.DataFrame, min_local_support: int) -> pd.DataFrame:
    result = frame.copy()
    result["flag_guardrails_robustos"] = (
        ~result["nivel_soporte"].eq("soporte_insuficiente")
        & result["flag_duracion_observada"]
        & result["flag_descuento_en_rango_local"]
        & ~result["flag_extrapolacion"]
        & result["local_support_count"].ge(min_local_support)
        & result["roi_lower_90"].gt(0)
        & result["uplift_esperado"].ge(0)
        & ~result["flag_uplift_invalido"]
        & ~result["flag_demanda_invalida"]
    )
    if "flag_model_disagreement" in result:
        result["flag_guardrails_robustos"] &= ~result["flag_model_disagreement"]
    return result


def deterministic_best(
    frame: pd.DataFrame,
    objective: str,
    epsilon: float = 0.0,
) -> pd.Series | None:
    if frame.empty:
        return None
    best = float(frame[objective].max())
    tied = frame[frame[objective].ge(best - epsilon)].sort_values(
        ["volumen_incremental_esperado", "local_support_count", "factor_descuento", "duracion_dias"],
        ascending=[False, False, True, True],
    )
    return tied.iloc[0]


def select_case_solutions(
    case_frame: pd.DataFrame,
    epsilon_roi: float,
) -> dict[str, pd.Series | None]:
    mathematical = deterministic_best(case_frame, "roi_esperado")
    valid = case_frame[case_frame["flag_guardrails_robustos"]]
    robust = deterministic_best(valid, "roi_robusto", epsilon=epsilon_roi)
    growth = deterministic_best(valid, "volumen_incremental_esperado")
    return {"mathematical": mathematical, "robust": robust, "growth": growth}


def recommendation_record(
    context: pd.Series,
    solution: pd.Series | None,
    mathematical: pd.Series,
) -> dict:
    selected = solution if solution is not None else mathematical
    automatic = solution is not None
    manual = context["nivel_soporte"] in {"soporte_bajo", "soporte_insuficiente"}
    recommendation_type = (
        "NO_RECOMMENDATION"
        if not automatic
        else "REQUIERE_REVISION_HUMANA"
        if manual
        else "RECOMENDACION_ROBUSTA"
    )
    warning = (
        "Sin escenario que cumpla guardrails; usar solo como exploracion y validar con experimento."
        if not automatic
        else "SKU reciente o soporte bajo: validar con Trade Marketing antes de aprobar."
        if manual
        else "Prediccion sujeta a error OOF; confirmar stock, presupuesto y ejecucion fuera del modelo."
    )
    return {
        "case_id": context["case_id"],
        "id_material": context["id_material"],
        "des_material": context["des_material"],
        "des_marca": context["des_marca"],
        "subcadena": context["subcadena"],
        "fecha_referencia": context["fecha_referencia"],
        "nivel_soporte": context["nivel_soporte"],
        "observaciones_historicas": context["observaciones_historicas"],
        "volumen_base_sem": context["volumen_base_sem"],
        "elasticidad_estimada": context["elasticidad_estimada"],
        "factor_descuento_recomendado": selected["factor_descuento"],
        "duracion_dias_recomendada": int(selected["duracion_dias"]),
        "uplift_esperado": selected["uplift_esperado"],
        "uplift_lower_90": selected["uplift_lower_90"],
        "uplift_upper_90": selected["uplift_upper_90"],
        "volumen_base_tanda": selected["volumen_base_tanda"],
        "volumen_promo_esperado": selected["volumen_promo_esperado"],
        "volumen_incremental_esperado": selected["volumen_incremental_esperado"],
        "roi_esperado": selected["roi_esperado"],
        "roi_lower_90": selected["roi_lower_90"],
        "roi_upper_90": selected["roi_upper_90"],
        "roi_robusto": selected["roi_robusto"],
        "soporte_local": int(selected["local_support_count"]),
        "flag_extrapolacion": bool(selected["flag_extrapolacion"]),
        "flag_optimo_en_limite_descuento": bool(selected["factor_descuento"] in {0.05, 0.40}),
        "flag_optimo_en_limite_duracion": bool(selected["duracion_dias"] in {5, 21}),
        "tipo_recomendacion": recommendation_type,
        "razon_seleccion": "maximo ROI robusto con tolerancia y desempate por volumen/soporte/costo operativo" if automatic else "mejor ROI esperado exploratorio",
        "advertencia_negocio": warning,
    }
