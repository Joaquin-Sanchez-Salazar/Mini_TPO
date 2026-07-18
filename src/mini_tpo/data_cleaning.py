from __future__ import annotations

import pandas as pd

from mini_tpo.constants import (
    FEATURES_MODEL_BASE,
    FEATURES_MODEL_OPTIONAL,
    LEAKAGE_COLUMNS,
    POST_PROMO_COLUMNS,
    TARGET_ROI,
    TARGET_UPLIFT,
)
from mini_tpo.data_audit import build_row_id, detect_uplift_floor
from mini_tpo.data_validation import add_audit_columns

FEATURES_PRE_PROMO = [*FEATURES_MODEL_BASE, *FEATURES_MODEL_OPTIONAL]
AUDIT_COLUMNS_PREFIX = "audit_"


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fecha_inicio_tanda"] = pd.to_datetime(out["fecha_inicio_tanda"], dayfirst=True, errors="raise")
    string_cols = ["id_material", "des_material", "des_marca", "subcadena"]
    for column in string_cols:
        out[column] = out[column].astype("string")
    numeric_cols = [
        "precio_base",
        "factor_descuento",
        "duracion_dias",
        "volumen_base_sem",
        "volumen_promo",
        "venta_promo",
        "inversion_promo",
        "uplift_real",
        "elasticidad_estimada",
        "roi",
    ]
    for column in numeric_cols:
        out[column] = pd.to_numeric(out[column], errors="raise")
    out["duracion_dias"] = out["duracion_dias"].astype("int64")
    return out


def clean_data(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = config["business_rules"]
    out = convert_types(df)
    out.insert(0, "row_id", build_row_id(out))
    out["flag_secundario_missing"] = out["flag_secundario"].isna()
    out["flag_secundario"] = out["flag_secundario"].map({0.0: "no", 1.0: "si"}).fillna("desconocido").astype("category")
    out["des_marca"] = out["des_marca"].astype("category")
    out["subcadena"] = out["subcadena"].astype("category")
    out["flag_descuento_fuera_dominio"] = ~out["factor_descuento"].between(rules["discount_min"], rules["discount_max"], inclusive="both")
    out["flag_duracion_fuera_dominio"] = ~out["duracion_dias"].between(rules["duration_min"], rules["duration_max"], inclusive="both")
    out["flag_fuera_dominio_optimizacion"] = out["flag_descuento_fuera_dominio"] | out["flag_duracion_fuera_dominio"]
    out["flag_uplift_en_piso"] = detect_uplift_floor(
        out,
        floor_value=rules.get("uplift_floor_value", 0.05),
        tolerance=rules.get("uplift_floor_tolerance", 1e-6),
    )
    out["flag_roi_extremo_descriptivo"] = out["roi"].abs() > out["roi"].abs().quantile(0.99)
    out["flag_uplift_extremo_descriptivo"] = out["uplift_real"] > out["uplift_real"].quantile(0.99)
    out = add_audit_columns(out)
    sort_cols = ["fecha_inicio_tanda", "id_material", "subcadena", "factor_descuento", "duracion_dias"]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    modeling = out.loc[~out["flag_fuera_dominio_optimizacion"]].copy().reset_index(drop=True)
    return out, modeling


def build_cleaning_log(raw: pd.DataFrame, clean_full: pd.DataFrame, modeling: pd.DataFrame) -> pd.DataFrame:
    process_version = "0.3"
    return pd.DataFrame(
        [
            {
                "paso": "Conversion explicita de tipos",
                "funcion_responsable": "convert_types",
                "razon_tecnica": "Evitar inferencias silenciosas y habilitar validaciones reproducibles.",
                "razon_negocio": "Asegurar que descuentos, duraciones y resultados sean comparables.",
                "filas_antes": len(raw),
                "filas_despues": len(clean_full),
                "filas_afectadas": len(clean_full),
                "columnas_afectadas": "fecha, categorias, numericas",
                "reversible": "si",
                "output_generado": "data/interim/base_mini_tpo_clean_full.parquet",
                "version_proceso": process_version,
            },
            {
                "paso": "Creacion de row_id",
                "funcion_responsable": "build_row_id",
                "razon_tecnica": "Identificador secuencial reproducible para el archivo raw actual.",
                "razon_negocio": "Mantener trazabilidad de cada promocion; su estabilidad depende de conservar el orden del raw.",
                "filas_antes": len(raw),
                "filas_despues": len(clean_full),
                "filas_afectadas": len(clean_full),
                "columnas_afectadas": "row_id",
                "reversible": "si",
                "output_generado": "todos los datasets derivados",
                "version_proceso": process_version,
            },
            {
                "paso": "Categoria explicita para flag_secundario faltante",
                "funcion_responsable": "clean_data",
                "razon_tecnica": "Null no necesariamente equivale a cero; puede capturar ausencia de informacion.",
                "razon_negocio": "Evitar confundir desconocido con ausencia real de exhibicion secundaria.",
                "filas_antes": len(raw),
                "filas_despues": len(clean_full),
                "filas_afectadas": int(raw["flag_secundario"].isna().sum()),
                "columnas_afectadas": "flag_secundario, flag_secundario_missing",
                "reversible": "si",
                "output_generado": "clean_full, modeling, targets",
                "version_proceso": process_version,
            },
            {
                "paso": "Creacion de flag_uplift_en_piso",
                "funcion_responsable": "detect_uplift_floor",
                "razon_tecnica": "Detectar concentracion en posible piso con tolerancia numerica.",
                "razon_negocio": "Controlar sesgo en calibracion y recomendaciones de bajo uplift.",
                "filas_antes": len(raw),
                "filas_despues": len(clean_full),
                "filas_afectadas": int(clean_full["flag_uplift_en_piso"].sum()),
                "columnas_afectadas": "flag_uplift_en_piso",
                "reversible": "si",
                "output_generado": "clean_full, targets, tablas uplift floor",
                "version_proceso": process_version,
            },
            {
                "paso": "Flags de dominio operativo",
                "funcion_responsable": "clean_data",
                "razon_tecnica": "El optimizador futuro operara entre 5%-40% y 5-21 dias.",
                "razon_negocio": "Evitar extrapolacion comercial no soportada.",
                "filas_antes": len(raw),
                "filas_despues": len(clean_full),
                "filas_afectadas": int(clean_full["flag_fuera_dominio_optimizacion"].sum()),
                "columnas_afectadas": "factor_descuento, duracion_dias",
                "reversible": "si",
                "output_generado": "base_mini_tpo_modeling.parquet",
                "version_proceso": process_version,
            },
            {
                "paso": "Dataset de modelado dentro del dominio",
                "funcion_responsable": "clean_data",
                "razon_tecnica": "Entrenar inicialmente donde existe soporte operativo definido.",
                "razon_negocio": "Reducir riesgo de recomendaciones fuera de politica promocional.",
                "filas_antes": len(clean_full),
                "filas_despues": len(modeling),
                "filas_afectadas": len(clean_full) - len(modeling),
                "columnas_afectadas": "todas",
                "reversible": "si",
                "output_generado": "data/processed/base_mini_tpo_modeling.parquet",
                "version_proceso": process_version,
            },
            {
                "paso": "Separacion features-target-index y datasets seguros",
                "funcion_responsable": "export_model_artifacts",
                "razon_tecnica": "Excluir targets, postpromocion, auditoria y columnas redundantes del baseline.",
                "razon_negocio": "Preparar modelado futuro sin leakage.",
                "filas_antes": len(modeling),
                "filas_despues": len(modeling),
                "filas_afectadas": len(modeling),
                "columnas_afectadas": "FEATURES_MODEL_BASE, targets, indice",
                "reversible": "si",
                "output_generado": "model_features_safe.parquet; model_targets.parquet; model_index.parquet; feature_manifest.json",
                "version_proceso": process_version,
            },
        ]
    )
