from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import (
    f1_score,
    precision_score,
    r2_score,
    recall_score,
)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def wape(y_true, y_pred) -> float:
    denominator = np.abs(np.asarray(y_true)).sum()
    return float(np.abs(np.asarray(y_pred) - np.asarray(y_true)).sum() / denominator) if denominator else np.nan


def smape(y_true, y_pred) -> float:
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    denominator = np.abs(actual) + np.abs(predicted)
    terms = np.divide(
        2 * np.abs(predicted - actual),
        denominator,
        out=np.zeros_like(actual, dtype=float),
        where=denominator != 0,
    )
    return float(np.mean(terms))


def bias(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def uplift_metrics(y_true, y_pred, volumen_base_tanda) -> dict:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    base = np.asarray(volumen_base_tanda, dtype=float)
    residual = predicted - actual
    unit_error = residual * base
    actual_incremental = actual * base
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "wape": wape(actual, predicted),
        "smape": smape(actual, predicted),
        "bias": bias(actual, predicted),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        "mae_unidades": float(np.mean(np.abs(unit_error))),
        "wape_unidades": float(np.abs(unit_error).sum() / np.abs(actual_incremental).sum()) if np.abs(actual_incremental).sum() else np.nan,
        "error_abs_total_unidades": float(np.abs(unit_error).sum()),
        "bias_unidades": float(np.mean(unit_error)),
        "n": len(actual),
    }


def roi_metrics(y_true, y_pred) -> dict:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    actual_positive = actual > 0
    predicted_positive = predicted > 0
    actual_negative = actual < 0
    predicted_negative = predicted < 0
    rho = (
        spearmanr(actual, predicted, nan_policy="omit").statistic
        if len(actual) > 1 and np.ptp(actual) > 0 and np.ptp(predicted) > 0
        else np.nan
    )
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "median_ae": float(np.median(np.abs(predicted - actual))),
        "bias": bias(actual, predicted),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        "spearman": float(rho),
        "sign_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
        "precision_roi_positivo": float(precision_score(actual_positive, predicted_positive, zero_division=0)),
        "recall_roi_positivo": float(recall_score(actual_positive, predicted_positive, zero_division=0)),
        "f1_roi_positivo": float(f1_score(actual_positive, predicted_positive, zero_division=0)),
        "precision_roi_negativo": float(precision_score(actual_negative, predicted_negative, zero_division=0)),
        "recall_roi_negativo": float(recall_score(actual_negative, predicted_negative, zero_division=0)),
        "f1_roi_negativo": float(f1_score(actual_negative, predicted_negative, zero_division=0)),
        "falsos_positivos": int(np.sum(predicted_positive & actual_negative)),
        "falsos_negativos": int(np.sum(predicted_negative & actual_positive)),
        "false_positive_rate": float(np.mean(predicted_positive & actual_negative)),
        "n": len(actual),
    }


def evaluate_prediction_frame(predictions: pd.DataFrame, target: str) -> dict:
    if target == "uplift_real":
        return uplift_metrics(
            predictions["target_real"],
            predictions["prediccion"],
            predictions["volumen_base_tanda"],
        )
    if target == "roi":
        return roi_metrics(predictions["target_real"], predictions["prediccion"])
    raise ValueError(f"Unsupported target: {target}")


def metrics_by_group(
    predictions: pd.DataFrame,
    target: str,
    group_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(group_columns, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(evaluate_prediction_frame(group, target))
        rows.append(row)
    return pd.DataFrame(rows)


def uncertainty_summary(
    predictions: pd.DataFrame,
    target: str,
    alpha: float = 0.10,
    group_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, float]:
    residual_abs = np.abs(predictions["target_real"] - predictions["prediccion"])
    radius = float(residual_abs.quantile(1 - alpha))
    work = predictions.copy()
    work["interval_lower"] = work["prediccion"] - radius
    work["interval_upper"] = work["prediccion"] + radius
    work["covered"] = work["target_real"].between(
        work["interval_lower"], work["interval_upper"], inclusive="both"
    )
    groups = group_columns or []
    rows = [
        {
            "target": target,
            "segmento": "global",
            "valor_segmento": "global",
            "cobertura": work["covered"].mean(),
            "ancho_promedio": 2 * radius,
            "radio": radius,
            "n": len(work),
            "metodo": "cuantil de residuos OOF absolutos",
        }
    ]
    for column in groups:
        for value, group in work.groupby(column, observed=True, dropna=False):
            rows.append(
                {
                    "target": target,
                    "segmento": column,
                    "valor_segmento": value,
                    "cobertura": group["covered"].mean(),
                    "ancho_promedio": 2 * radius,
                    "radio": radius,
                    "n": len(group),
                    "metodo": "cuantil de residuos OOF absolutos",
                }
            )
    return pd.DataFrame(rows), radius


def compare_models_by_date(
    champion: pd.DataFrame, challenger: pd.DataFrame, target: str
) -> pd.DataFrame:
    left = champion.groupby("fecha_inicio_tanda").apply(
        lambda g: mae(g["target_real"], g["prediccion"]), include_groups=False
    )
    right = challenger.groupby("fecha_inicio_tanda").apply(
        lambda g: mae(g["target_real"], g["prediccion"]), include_groups=False
    )
    paired = pd.concat([left.rename("champion"), right.rename("challenger")], axis=1).dropna()
    difference = paired["challenger"] - paired["champion"]
    if len(paired) >= 5 and not np.allclose(difference, 0):
        statistic, p_value = wilcoxon(difference)
    else:
        statistic, p_value = np.nan, np.nan
    mean_difference = float(difference.mean()) if len(difference) else np.nan
    if abs(mean_difference) < 0.01 * max(float(paired["champion"].mean()), 1e-12):
        conclusion = "empate tecnico"
    elif p_value < 0.05 and mean_difference > 0:
        conclusion = "diferencia clara"
    else:
        conclusion = "diferencia pequena"
    return pd.DataFrame(
        [
            {
                "target": target,
                "fechas_comparadas": len(paired),
                "mae_champion": paired["champion"].mean(),
                "mae_challenger": paired["challenger"].mean(),
                "diferencia_challenger_menos_champion": mean_difference,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "conclusion": conclusion,
                "alcance": "comparacion exploratoria por fecha; no prueba causal",
            }
        ]
    )
