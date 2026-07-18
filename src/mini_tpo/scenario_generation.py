from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from mini_tpo.feature_engineering import build_engineered_core


def next_weekly_date(dates: pd.Series) -> pd.Timestamp:
    latest = pd.to_datetime(dates, errors="raise").max()
    days_ahead = (7 - latest.weekday()) % 7
    days_ahead = 7 if days_ahead == 0 else days_ahead
    return (latest + pd.Timedelta(days=days_ahead)).normalize()


def discount_grid(minimum: float, maximum: float, step: float) -> list[float]:
    count = int(round((maximum - minimum) / step))
    return [round(minimum + i * step, 3) for i in range(count + 1)]


def build_scenario_grid(
    contexts: pd.DataFrame,
    discounts: list[float],
    durations: list[int],
    config: dict,
    grid_type: str = "principal_soportada",
) -> pd.DataFrame:
    rows = []
    scenario_id = 1
    for context in contexts.itertuples(index=False):
        for discount, duration in product(discounts, durations):
            rows.append(
                {
                    "row_id": scenario_id,
                    "case_id": context.case_id,
                    "id_material": context.id_material,
                    "subcadena": context.subcadena,
                    "factor_descuento": discount,
                    "duracion_dias": int(duration),
                    "volumen_base_sem": context.volumen_base_sem,
                    "elasticidad_estimada": context.elasticidad_estimada,
                    "flag_secundario": context.flag_secundario,
                    "fecha_referencia": context.fecha_referencia,
                    "grid_type": grid_type,
                }
            )
            scenario_id += 1
    base = pd.DataFrame(rows)
    safe = base[
        [
            "row_id", "id_material", "subcadena", "factor_descuento",
            "duracion_dias", "volumen_base_sem", "elasticidad_estimada",
            "flag_secundario",
        ]
    ]
    index = base[["row_id"]].assign(fecha_inicio_tanda=base["fecha_referencia"])
    engineered = build_engineered_core(safe, index, config)
    return base.merge(engineered, on=[
        "row_id", "id_material", "subcadena", "factor_descuento",
        "duracion_dias", "volumen_base_sem", "elasticidad_estimada",
        "flag_secundario",
    ], validate="one_to_one")


def add_local_support(scenarios: pd.DataFrame, history: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows = []
    for scenario in scenarios.itertuples(index=False):
        local = history[
            history["id_material"].eq(scenario.id_material)
            & history["subcadena"].eq(scenario.subcadena)
        ]
        same_duration = local["duracion_dias"].eq(scenario.duracion_dias)
        distance = (local["factor_descuento"] - scenario.factor_descuento).abs()
        duration_observed = bool(same_duration.any())
        nearest = float(distance.min()) if len(local) else np.inf
        inside_range = bool(
            len(local)
            and local["factor_descuento"].min() <= scenario.factor_descuento
            <= local["factor_descuento"].max()
        )
        count = int((same_duration & distance.le(tolerance)).sum())
        rows.append(
            {
                "row_id": scenario.row_id,
                "local_support_count": count,
                "distancia_descuento_mas_cercano": nearest,
                "flag_duracion_observada": duration_observed,
                "flag_descuento_en_rango_local": inside_range,
                "flag_extrapolacion": not (duration_observed and inside_range and nearest <= tolerance),
            }
        )
    return scenarios.merge(pd.DataFrame(rows), on="row_id", validate="one_to_one")
