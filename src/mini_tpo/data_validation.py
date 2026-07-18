from __future__ import annotations

import numpy as np
import pandas as pd


BUSINESS_KEY = ["fecha_inicio_tanda", "id_material", "subcadena", "factor_descuento", "duracion_dias"]
NON_NEGATIVE_COLUMNS = [
    "precio_base",
    "factor_descuento",
    "duracion_dias",
    "volumen_base_sem",
    "volumen_promo",
    "venta_promo",
    "inversion_promo",
]


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    rows = []
    present = set(df.columns)
    for column in required_columns:
        rows.append(
            {
                "check": "required_column",
                "column": column,
                "status": "ok" if column in present else "fail",
                "affected_rows": 0 if column in present else len(df),
                "detail": "presente" if column in present else "ausente",
            }
        )
    return pd.DataFrame(rows)


def validation_summary(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    rules = config["business_rules"]
    rows = []
    required = config["columns"]["required_columns"]
    rows.extend(validate_required_columns(df, required).to_dict("records"))
    rows.append({"check": "exact_duplicates", "column": "*", "status": "ok", "affected_rows": int(df.duplicated().sum()), "detail": "duplicados exactos"})
    available_key = [c for c in BUSINESS_KEY if c in df.columns]
    rows.append(
        {
            "check": "business_key_duplicates",
            "column": ",".join(available_key),
            "status": "ok",
            "affected_rows": int(df.duplicated(available_key).sum()) if available_key else len(df),
            "detail": "duplicados por clave de negocio candidata",
        }
    )
    numeric_cols = [c for c in config["columns"]["numeric_columns"] if c in df.columns]
    for column in numeric_cols:
        values = pd.to_numeric(df[column], errors="coerce")
        rows.append({"check": "missing", "column": column, "status": "ok", "affected_rows": int(values.isna().sum()), "detail": "valores faltantes"})
        rows.append({"check": "infinite", "column": column, "status": "ok", "affected_rows": int(np.isinf(values.dropna()).sum()), "detail": "valores infinitos"})
    for column in [c for c in NON_NEGATIVE_COLUMNS if c in df.columns]:
        values = pd.to_numeric(df[column], errors="coerce")
        rows.append({"check": "negative_not_allowed", "column": column, "status": "ok", "affected_rows": int((values < 0).sum()), "detail": "negativos no esperados"})
    if "factor_descuento" in df.columns:
        disc = pd.to_numeric(df["factor_descuento"], errors="coerce")
        rows.append({"check": "discount_outside_optimizer_domain", "column": "factor_descuento", "status": "warn", "affected_rows": int(((disc < rules["discount_min"]) | (disc > rules["discount_max"])).sum()), "detail": "fuera de [5%, 40%]"})
    if "duracion_dias" in df.columns:
        dur = pd.to_numeric(df["duracion_dias"], errors="coerce")
        rows.append({"check": "duration_outside_optimizer_domain", "column": "duracion_dias", "status": "warn", "affected_rows": int(((dur < rules["duration_min"]) | (dur > rules["duration_max"])).sum()), "detail": "fuera de [5, 21] dias"})
    return pd.DataFrame(rows)


def add_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["audit_volumen_base_tanda"] = out["volumen_base_sem"] * out["duracion_dias"] / 7
    out["audit_volumen_incremental_observado"] = out["volumen_promo"] - out["audit_volumen_base_tanda"]
    out["audit_uplift_recalculado"] = out["volumen_promo"] / out["audit_volumen_base_tanda"] - 1
    out["audit_venta_promo_recalculada"] = out["volumen_promo"] * out["precio_base"] * (1 - out["factor_descuento"])
    out["audit_inversion_promo_recalculada"] = out["volumen_promo"] * out["precio_base"] * out["factor_descuento"]
    for original, audit in [
        ("uplift_real", "audit_uplift_recalculado"),
        ("venta_promo", "audit_venta_promo_recalculada"),
        ("inversion_promo", "audit_inversion_promo_recalculada"),
    ]:
        out[f"audit_diff_abs_{original}"] = (out[audit] - out[original]).abs()
        out[f"audit_diff_rel_{original}"] = out[f"audit_diff_abs_{original}"] / out[original].abs().replace(0, np.nan)
    out["flag_inconsistencia_uplift"] = out["audit_diff_abs_uplift_real"] > 0.01
    out["flag_inconsistencia_venta"] = out["audit_diff_rel_venta_promo"] > 0.001
    out["flag_inconsistencia_inversion"] = out["audit_diff_rel_inversion_promo"] > 0.001
    return out


def leakage_classification() -> pd.DataFrame:
    rows = [
        ("fecha_inicio_tanda", "Fecha", "Disponible pre-promocion", "Base para split temporal."),
        ("id_material", "ID", "Disponible pre-promocion", "Identificador de SKU."),
        ("des_material", "ID", "Disponible pre-promocion", "Descripcion/codigo comercial de SKU."),
        ("des_marca", "Categorica", "Disponible pre-promocion", "Marca o familia comercial."),
        ("subcadena", "Categorica", "Disponible pre-promocion", "Cadena donde se ejecuta la promocion."),
        ("precio_base", "Numerica", "Disponible pre-promocion", "Precio de referencia previo al descuento."),
        ("factor_descuento", "Numerica", "Conocida al disenar promocion", "Palanca de decision futura."),
        ("duracion_dias", "Numerica", "Conocida al disenar promocion", "Palanca de decision futura."),
        ("volumen_base_sem", "Numerica", "Disponible pre-promocion", "Feature valida por supuesto de la prueba; estimada sin informacion posterior."),
        ("elasticidad_estimada", "Numerica", "Disponible pre-promocion", "Feature valida por supuesto de la prueba; estimada sin informacion posterior."),
        ("flag_secundario", "Categorica", "Conocida al disenar promocion", "Mecanica/visibilidad secundaria; missing es informacion."),
        ("volumen_promo", "Resultado", "Post-promocion", "No apta como feature para predecir uplift."),
        ("venta_promo", "Variable derivada", "Leakage", "Deriva de volumen observado, precio y descuento."),
        ("inversion_promo", "Variable derivada", "Leakage", "Deriva de volumen observado, precio y descuento."),
        ("uplift_real", "Target de uplift", "Target", "Resultado observado a predecir."),
        ("roi", "Target ROI", "Target", "KPI oficial de Alicorp; valido como target y prohibido como predictor."),
    ]
    return pd.DataFrame(rows, columns=["variable", "tipo", "clasificacion_modelado", "comentario"])
