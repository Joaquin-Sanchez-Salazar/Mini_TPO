from __future__ import annotations

import numpy as np
import pandas as pd

from mini_tpo.constants import (
    FEATURES_EXCLUDED_BASE,
    FEATURES_MODEL_BASE,
    FEATURES_MODEL_OPTIONAL,
    LEAKAGE_COLUMNS,
    POST_PROMO_COLUMNS,
    PRE_PROMO_CONFIRMED,
    REPORTING_ONLY_COLUMNS,
)


def build_row_id(df: pd.DataFrame) -> pd.Series:
    return pd.Series([f"row_{i + 1:06d}" for i in range(len(df))], index=df.index, dtype="string")


def detect_uplift_floor(df: pd.DataFrame, floor_value: float = 0.05, tolerance: float = 1e-6) -> pd.Series:
    return pd.Series(np.isclose(df["uplift_real"], floor_value, atol=tolerance, rtol=0), index=df.index)


def validate_one_to_one_mapping(df: pd.DataFrame, left: str, right: str) -> dict:
    counts = df.groupby(left, observed=True)[right].nunique(dropna=False)
    return {
        "left": left,
        "right": right,
        "is_one_to_one": bool(counts.max() == 1),
        "max_values_per_left": int(counts.max()),
        "violating_left_values": int((counts > 1).sum()),
    }


def price_variation_by_sku(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("id_material", observed=True)["precio_base"]
        .agg(n_precios="nunique", precio_min="min", precio_max="max")
        .reset_index()
        .sort_values(["n_precios", "id_material"], ascending=[False, True])
    )


def feature_availability_audit() -> pd.DataFrame:
    rows = [
        ("fecha_inicio_tanda", "fecha/split", "si", "si", "no", "bajo", "conservar en indice, no feature numerica directa", "campo de inicio", "confirmada"),
        ("id_material", "id/predictor", "si", "si", "no", "bajo", "baseline", "SKU conocido al decidir", "confirmada"),
        ("des_material", "reporting", "si", "si", "no", "bajo", "excluir baseline por redundancia con SKU", "1:1 con id_material", "confirmada"),
        ("des_marca", "categorica/reporting", "si", "si", "no", "medio", "reporting/cold start, no aporta adicional si SKU esta incluido", "deterministica por SKU", "confirmada"),
        ("subcadena", "predictor", "si", "si", "no", "bajo", "baseline", "cadena conocida al disenar promocion", "confirmada"),
        ("precio_base", "numerica/reporting", "si", "si", "no", "medio", "excluir baseline si es constante por SKU; opcional si no se usa SKU", "constante por SKU en esta base", "confirmada"),
        ("factor_descuento", "palanca", "si", "si", "no", "bajo", "baseline", "decision de promocion", "confirmada"),
        ("duracion_dias", "palanca", "si", "si", "no", "bajo", "baseline", "decision de promocion", "confirmada"),
        ("volumen_base_sem", "predictor prepromocion", "si", "asumida", "no", "bajo", "baseline", "estimacion previa que escala el impacto absoluto", "confirmada por supuesto de la prueba"),
        ("elasticidad_estimada", "predictor prepromocion", "si", "asumida", "no", "bajo", "baseline", "estimacion interna previa de sensibilidad al precio", "confirmada por supuesto de la prueba"),
        ("flag_secundario", "mecanica", "si", "si", "no", "bajo", "baseline si se conoce al recomendar", "mecanica promocional", "confirmada"),
        ("flag_secundario_missing", "calidad", "condicionado", "si", "posible", "medio", "opcional sensibilidad, no baseline", "estado missing podria no existir en decision futura", "condicionada"),
        ("volumen_promo", "resultado", "no", "si", "si", "alto", "excluida", "observado post-promocion", "excluida"),
        ("venta_promo", "resultado derivado", "no", "si", "si", "alto", "excluida", "deriva de volumen observado", "excluida"),
        ("inversion_promo", "resultado derivado", "no", "si", "si", "alto", "excluida", "deriva de volumen observado", "excluida"),
        ("uplift_real", "target", "no", "si", "si", "alto", "target", "resultado observado", "excluida"),
        ("roi", "target/KPI oficial", "no", "no disponible", "si", "target", "target valido, nunca predictor", "KPI oficial entregado por Alicorp", "target"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "variable",
            "rol",
            "disponible_antes_tanda",
            "metodologia_conocida",
            "puede_usar_info_futura",
            "riesgo_leakage",
            "decision_modelado",
            "evidencia",
            "estado",
        ],
    )


def build_cardinality_profile(df: pd.DataFrame, near_constant_threshold: float = 0.95) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        vc = df[col].value_counts(dropna=False)
        dominant = int(vc.iloc[0]) if len(vc) else 0
        prop = dominant / len(df) if len(df) else 0
        nunique = int(df[col].nunique(dropna=False))
        if nunique == 1:
            klass = "constante"
        elif prop >= near_constant_threshold:
            klass = "casi_constante"
        elif nunique <= 20:
            klass = "baja_cardinalidad"
        else:
            klass = "alta_cardinalidad"
        rows.append(
            {
                "variable": col,
                "tipo": str(df[col].dtype),
                "n_valores_unicos": nunique,
                "frecuencia_categoria_dominante": dominant,
                "proporcion_dominante": prop,
                "clasificacion": klass,
            }
        )
    return pd.DataFrame(rows)


def redundancy_audit(df: pd.DataFrame) -> pd.DataFrame:
    desc = validate_one_to_one_mapping(df, "id_material", "des_material")
    brand = validate_one_to_one_mapping(df, "id_material", "des_marca")
    price = price_variation_by_sku(df)
    return pd.DataFrame(
        [
            {
                "chequeo": "id_material -> des_material",
                "resultado": desc["is_one_to_one"],
                "evidencia": f"max descripciones por SKU={desc['max_values_per_left']}",
                "decision": "des_material queda para reporting; excluida del baseline.",
            },
            {
                "chequeo": "id_material -> des_marca",
                "resultado": brand["is_one_to_one"],
                "evidencia": f"max marcas por SKU={brand['max_values_per_left']}",
                "decision": "des_marca queda para reporting/cold start; redundante con SKU en baseline.",
            },
            {
                "chequeo": "precio_base constante por SKU",
                "resultado": bool(price["n_precios"].max() == 1),
                "evidencia": f"max precios por SKU={int(price['n_precios'].max())}",
                "decision": "precio_base queda fuera del baseline con SKU; opcion si se modela sin SKU.",
            },
        ]
    )


def model_role_summary() -> pd.DataFrame:
    rows = []
    for col in PRE_PROMO_CONFIRMED:
        rows.append((col, "prepromo_confirmada", col in FEATURES_MODEL_BASE, col in FEATURES_MODEL_OPTIONAL, ""))
    for col in POST_PROMO_COLUMNS:
        rows.append((col, "postpromo/leakage", False, False, "excluida"))
    for col in FEATURES_MODEL_OPTIONAL:
        if col not in [r[0] for r in rows]:
            rows.append((col, "opcional/sensibilidad", False, True, "no baseline"))
    for col in REPORTING_ONLY_COLUMNS:
        if col not in [r[0] for r in rows]:
            rows.append((col, "reporting", False, False, "no baseline"))
    for col in FEATURES_EXCLUDED_BASE:
        if col not in [r[0] for r in rows]:
            rows.append((col, "excluida", False, False, "guardrail"))
    return pd.DataFrame(rows, columns=["variable", "rol", "baseline", "opcional", "motivo"])
