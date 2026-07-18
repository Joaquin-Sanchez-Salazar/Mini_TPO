from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mini_tpo.constants import (
    AUDIT_COLUMNS,
    FEATURES_EXCLUDED_BASE,
    FEATURES_MODEL_BASE,
    FEATURES_MODEL_MINIMAL,
    FEATURES_MODEL_OPTIONAL,
    LEAKAGE_COLUMNS,
    POST_PROMO_COLUMNS,
    TARGET_ROI,
    TARGET_UPLIFT,
)


def create_safe_feature_dataset(modeling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = ["row_id", *FEATURES_MODEL_BASE]
    features = modeling[feature_cols].copy()
    targets = modeling[
        [
            "row_id",
            TARGET_UPLIFT,
            TARGET_ROI,
            "flag_descuento_fuera_dominio",
            "flag_duracion_fuera_dominio",
            "flag_fuera_dominio_optimizacion",
            "flag_uplift_en_piso",
            "flag_roi_extremo_descriptivo",
            "flag_uplift_extremo_descriptivo",
        ]
    ].copy()
    index = modeling[["row_id", "fecha_inicio_tanda", "id_material", "des_material", "des_marca", "subcadena"]].copy()
    return features, targets, index


def create_feature_manifest(df: pd.DataFrame) -> dict:
    overrides = {
        "volumen_base_sem": {
            "availability": "confirmed_assumption",
            "leakage_risk": "low",
            "reason": "Escala de demanda estimada antes de la promocion.",
            "future_transformation": "log1p opcional e interaccion con duracion.",
            "business_interpretation": "Permite convertir uplift porcentual en unidades incrementales.",
        },
        "elasticidad_estimada": {
            "availability": "confirmed_assumption",
            "leakage_risk": "low",
            "reason": "Sensibilidad historica del SKU al precio, disponible antes de la tanda.",
            "future_transformation": "Valor absoluto e interaccion con factor_descuento.",
            "business_interpretation": "Diferencia la respuesta esperada ante una misma profundidad promocional.",
        },
        "roi": {
            "availability": "post_promotion",
            "leakage_risk": "target",
            "reason": "KPI oficial de negocio entregado por Alicorp.",
            "future_transformation": "Evaluar transformaciones robustas sin alterar el target original.",
            "business_interpretation": "Retorno observado de la tanda; valido como target, nunca predictor.",
        },
    }
    entries = []
    for col in df.columns:
        if col in FEATURES_MODEL_BASE:
            role = "feature_baseline"
            include_base = True
            include_optional = False
            reason = "Predictor disponible antes de la promocion o conocido al disenar la tanda."
            availability = "confirmed"
            leakage = "low"
        elif col in FEATURES_MODEL_OPTIONAL:
            role = "feature_opcional"
            include_base = False
            include_optional = True
            reason = "Variable reservada para analisis de sensibilidad, no para el baseline principal."
            availability = "confirmed_or_quality"
            leakage = "low"
        elif col in [TARGET_UPLIFT, TARGET_ROI]:
            role = "target"
            include_base = False
            include_optional = False
            reason = "Resultado observado que se desea predecir."
            availability = "post_promotion"
            leakage = "target"
        elif col in POST_PROMO_COLUMNS or col in LEAKAGE_COLUMNS:
            role = "postpromocion"
            include_base = False
            include_optional = False
            reason = "Resultado conocido despues de ejecutar la promocion."
            availability = "post_promotion"
            leakage = "high"
        elif col in AUDIT_COLUMNS or col.startswith("audit_") or col.startswith("flag_inconsistencia"):
            role = "auditoria"
            include_base = False
            include_optional = False
            reason = "Variable derivada para control de consistencia, no predictor."
            availability = "derived_post_promotion"
            leakage = "high"
        elif col in FEATURES_EXCLUDED_BASE:
            role = "excluida_baseline"
            include_base = False
            include_optional = False
            reason = "Indice, variable redundante o control fuera del baseline."
            availability = "support_or_reporting"
            leakage = "medium"
        else:
            role = "control_o_reporting"
            include_base = False
            include_optional = False
            reason = "Variable conservada para control, reporting o trazabilidad."
            availability = "depends_on_source"
            leakage = "medium"
        override = overrides.get(col, {})
        future = (
            "Encoding o transformacion definida en feature engineering."
            if include_base or include_optional
            else ""
        )
        entries.append(
            {
                "name": col,
                "role": role,
                "dtype": str(df[col].dtype),
                "availability": override.get("availability", availability),
                "leakage_risk": override.get("leakage_risk", leakage),
                "baseline_included": include_base,
                "optional_included": include_optional,
                "reason": override.get("reason", reason),
                "future_transformation": override.get("future_transformation", future),
                "source": "dataset proporcionado por Alicorp",
                "business_interpretation": override.get(
                    "business_interpretation",
                    "Variable documentada para gobierno y uso consistente en fases posteriores.",
                ),
            }
        )
    return {
        "version": "0.3",
        "assumptions": [
            "volumen_base_sem y elasticidad_estimada estan disponibles antes de la tanda por supuesto de la prueba.",
            "roi es el KPI oficial de Alicorp y un target valido, nunca un predictor.",
        ],
        "features_model_base": FEATURES_MODEL_BASE,
        "features_model_minimal": FEATURES_MODEL_MINIMAL,
        "features_model_optional": FEATURES_MODEL_OPTIONAL,
        "variables": entries,
    }


def validate_no_leakage(features: pd.DataFrame) -> None:
    forbidden = set(POST_PROMO_COLUMNS) | set(LEAKAGE_COLUMNS) | {TARGET_UPLIFT, TARGET_ROI}
    audit_cols = {c for c in features.columns if c.startswith("audit_") or c.startswith("flag_inconsistencia")}
    overlap = forbidden.intersection(features.columns) | audit_cols
    if overlap:
        raise ValueError(f"Features seguras contienen columnas no permitidas: {sorted(overlap)}")


def export_model_artifacts(modeling: pd.DataFrame, processed_dir: Path) -> dict[str, Path]:
    features, targets, index = create_safe_feature_dataset(modeling)
    validate_no_leakage(features.drop(columns=["row_id"], errors="ignore"))
    paths = {
        "safe_features": processed_dir / "model_features_safe.parquet",
        "safe_targets": processed_dir / "model_targets.parquet",
        "safe_index": processed_dir / "model_index.parquet",
        "feature_manifest": processed_dir / "feature_manifest.json",
    }
    features.to_parquet(paths["safe_features"], index=False)
    targets.to_parquet(paths["safe_targets"], index=False)
    index.to_parquet(paths["safe_index"], index=False)
    features.to_csv(paths["safe_features"].with_suffix(".csv"), index=False)
    targets.to_csv(paths["safe_targets"].with_suffix(".csv"), index=False)
    index.to_csv(paths["safe_index"].with_suffix(".csv"), index=False)
    manifest = create_feature_manifest(modeling)
    paths["feature_manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
