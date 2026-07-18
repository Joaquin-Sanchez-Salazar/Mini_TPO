from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mini_tpo.constants import (
    AUDIT_COLUMNS,
    FORBIDDEN_ENGINEERED_COLUMNS,
    POST_PROMO_COLUMNS,
    TARGET_ROI,
    TARGET_UPLIFT,
)
from mini_tpo.feature_sets import (
    FEATURES_CATEGORICAL,
    FEATURES_CHANGE_WITH_DISCOUNT,
    FEATURES_CHANGE_WITH_DURATION,
    FEATURES_ENGINEERED_CORE,
    FEATURES_ENGINEERED_OPTIONAL,
    FEATURES_ENGINEERED_TEMPORAL,
    FEATURES_FOR_ROI,
    FEATURES_FOR_UPLIFT,
    FEATURES_NUMERICAL,
)
from mini_tpo.paths import PROJECT_ROOT


INPUT_SAFE_COLUMNS = [
    "row_id",
    "id_material",
    "subcadena",
    "factor_descuento",
    "duracion_dias",
    "volumen_base_sem",
    "elasticidad_estimada",
    "flag_secundario",
]


def load_feature_engineering_inputs(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project = config["project"]
    features = pd.read_parquet(PROJECT_ROOT / project["safe_features"])
    targets = pd.read_parquet(PROJECT_ROOT / project["safe_targets"])
    index = pd.read_parquet(PROJECT_ROOT / project["safe_index"])
    return features, targets, index


def validate_input_alignment(
    features: pd.DataFrame, targets: pd.DataFrame, index: pd.DataFrame
) -> pd.DataFrame:
    checks = [
        ("same_row_count", len(features) == len(targets) == len(index), f"{len(features)}, {len(targets)}, {len(index)}"),
        ("features_row_id_unique", features["row_id"].is_unique, str(features["row_id"].nunique())),
        ("targets_row_id_unique", targets["row_id"].is_unique, str(targets["row_id"].nunique())),
        ("index_row_id_unique", index["row_id"].is_unique, str(index["row_id"].nunique())),
        ("safe_schema_exact", features.columns.tolist() == INPUT_SAFE_COLUMNS, str(features.columns.tolist())),
        ("targets_absent_from_features", not {TARGET_UPLIFT, TARGET_ROI}.intersection(features.columns), "targets excluded"),
        ("postpromo_absent_from_features", not set(POST_PROMO_COLUMNS).intersection(features.columns), "postpromo excluded"),
        ("audit_absent_from_features", not set(AUDIT_COLUMNS).intersection(features.columns), "audit excluded"),
    ]
    try:
        joined = features[["row_id"]].merge(
            targets[["row_id"]], on="row_id", validate="one_to_one"
        ).merge(index[["row_id"]], on="row_id", validate="one_to_one")
        join_ok = len(joined) == len(features)
        join_detail = str(len(joined))
    except Exception as exc:  # pragma: no cover - detail is tested through failure status
        join_ok = False
        join_detail = str(exc)
    checks.append(("one_to_one_join", join_ok, join_detail))
    result = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise ValueError(f"Feature engineering inputs failed validation: {failed}")
    return result


def _add_engineered_columns(frame: pd.DataFrame, dates: pd.Series, config: dict) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(dates, errors="raise")
    month_period = config["feature_engineering"].get("month_period", 12)
    week_period = config["feature_engineering"].get("week_period", 52)

    out["duracion_semanas"] = out["duracion_dias"] / 7.0
    out["volumen_base_tanda"] = out["volumen_base_sem"] * out["duracion_dias"] / 7.0
    out["elasticidad_abs"] = out["elasticidad_estimada"].abs()
    out["descuento_x_elasticidad"] = out["factor_descuento"] * out["elasticidad_abs"]
    out["descuento_x_duracion"] = out["factor_descuento"] * out["duracion_dias"]
    out["factor_descuento_sq"] = out["factor_descuento"].pow(2)
    out["duracion_dias_sq"] = out["duracion_dias"].pow(2)
    out["log1p_volumen_base_sem"] = np.log1p(out["volumen_base_sem"])
    out["log1p_volumen_base_tanda"] = np.log1p(out["volumen_base_tanda"])

    out["mes"] = dates.dt.month.astype("int64").to_numpy()
    out["trimestre"] = dates.dt.quarter.astype("int64").to_numpy()
    out["semana_anio"] = dates.dt.isocalendar().week.astype("int64").to_numpy()
    out["mes_sin"] = np.sin(2 * np.pi * out["mes"] / month_period)
    out["mes_cos"] = np.cos(2 * np.pi * out["mes"] / month_period)
    out["semana_anio_sin"] = np.sin(2 * np.pi * out["semana_anio"] / week_period)
    out["semana_anio_cos"] = np.cos(2 * np.pi * out["semana_anio"] / week_period)
    return out


def build_engineered_core(
    features: pd.DataFrame, index: pd.DataFrame, config: dict
) -> pd.DataFrame:
    date_index = index[["row_id", "fecha_inicio_tanda"]].copy()
    aligned = features.merge(date_index, on="row_id", how="left", validate="one_to_one", sort=False)
    if aligned["fecha_inicio_tanda"].isna().any():
        raise ValueError("Missing fecha_inicio_tanda after one-to-one index join")
    engineered = _add_engineered_columns(
        aligned.drop(columns="fecha_inicio_tanda"), aligned["fecha_inicio_tanda"], config
    )
    result = engineered[["row_id", *FEATURES_ENGINEERED_CORE]].copy()
    result["id_material"] = result["id_material"].astype("string")
    result["subcadena"] = result["subcadena"].astype("category")
    result["flag_secundario"] = result["flag_secundario"].astype("category")
    return result


def load_optional_safe_context(config: dict, index: pd.DataFrame) -> pd.DataFrame:
    source_path = PROJECT_ROOT / config["project"]["processed_modeling"]
    source = pd.read_parquet(source_path, columns=["row_id", "precio_base", "flag_secundario_missing"])
    brand = index[["row_id", "des_marca"]].copy()
    context = source.merge(brand, on="row_id", validate="one_to_one", sort=False)
    if set(context.columns) != {"row_id", "precio_base", "flag_secundario_missing", "des_marca"}:
        raise ValueError("Optional context contains unexpected columns")
    return context


def build_engineered_optional(
    core: pd.DataFrame,
    features: pd.DataFrame,
    index: pd.DataFrame,
    optional_context: pd.DataFrame,
) -> pd.DataFrame:
    dates = index[["row_id", "fecha_inicio_tanda"]].copy()
    context = optional_context.merge(dates, on="row_id", validate="one_to_one", sort=False)
    start_date = pd.to_datetime(context["fecha_inicio_tanda"], errors="raise").min()
    context["dias_desde_inicio_dataset"] = (
        pd.to_datetime(context["fecha_inicio_tanda"], errors="raise") - start_date
    ).dt.days.astype("int64")
    context = context.drop(columns="fecha_inicio_tanda")
    sku_chain = features[["row_id", "id_material", "subcadena"]].copy()
    sku_chain["sku_cadena"] = (
        sku_chain["id_material"].astype("string")
        + "__"
        + sku_chain["subcadena"].astype("string")
    )
    context = context.merge(
        sku_chain[["row_id", "sku_cadena"]], on="row_id", validate="one_to_one", sort=False
    )
    result = core.merge(context, on="row_id", validate="one_to_one", sort=False)
    result["des_marca"] = result["des_marca"].astype("category")
    result["sku_cadena"] = result["sku_cadena"].astype("category")
    return result[["row_id", *FEATURES_ENGINEERED_CORE, *FEATURES_ENGINEERED_OPTIONAL]]


def validate_engineered_dataset(
    frame: pd.DataFrame, expected_row_ids: pd.Series, expected_columns: list[str]
) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number])
    forbidden = set(FORBIDDEN_ENGINEERED_COLUMNS).intersection(frame.columns)
    audit_like = {
        col for col in frame.columns if col.startswith("audit_") or col.startswith("flag_inconsistencia")
    }
    checks = [
        ("row_count", len(frame) == len(expected_row_ids), f"{len(frame)}"),
        ("row_id_unique", frame["row_id"].is_unique, str(frame["row_id"].nunique())),
        ("deterministic_order", frame["row_id"].tolist() == expected_row_ids.tolist(), "matches safe input"),
        ("columns_exact", frame.columns.tolist() == ["row_id", *expected_columns], str(frame.columns.tolist())),
        ("no_duplicate_columns", not frame.columns.duplicated().any(), "unique column names"),
        ("no_targets_or_postpromo", not forbidden, str(sorted(forbidden))),
        ("no_audit_columns", not audit_like, str(sorted(audit_like))),
        ("no_infinite_numeric", not np.isinf(numeric.to_numpy()).any(), "finite numeric values"),
        ("no_unexpected_missing", not frame.isna().any().any(), str(frame.isna().sum().sum())),
        ("month_range", frame["mes"].between(1, 12).all(), "1..12"),
        ("quarter_range", frame["trimestre"].between(1, 4).all(), "1..4"),
        ("week_range", frame["semana_anio"].between(1, 53).all(), "1..53"),
    ]
    for column in ["mes_sin", "mes_cos", "semana_anio_sin", "semana_anio_cos"]:
        checks.append((f"{column}_range", frame[column].between(-1, 1).all(), "-1..1"))
    result = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise ValueError(f"Engineered dataset failed validation: {failed}")
    return result


def build_feature_catalog() -> pd.DataFrame:
    rows: list[dict] = []

    def add(
        name: str,
        feature_type: str,
        feature_set: str,
        formula: str,
        sources: str,
        technical: str,
        business: str,
        changes_discount: bool = False,
        changes_duration: bool = False,
        uplift_use: str = "si",
        roi_use: str = "si",
        optimizer_use: str = "si",
        status: str = "core",
        availability: str = "prepromocion",
        leakage: str = "bajo",
    ) -> None:
        rows.append(
            {
                "nombre": name,
                "tipo": feature_type,
                "feature_set": feature_set,
                "formula": formula,
                "variables_fuente": sources,
                "disponibilidad_temporal": availability,
                "riesgo_leakage": leakage,
                "interpretacion_tecnica": technical,
                "interpretacion_negocio": business,
                "cambia_con_descuento": changes_discount,
                "cambia_con_duracion": changes_duration,
                "uso_uplift": uplift_use,
                "uso_roi": roi_use,
                "uso_optimizacion": optimizer_use,
                "estado": status,
            }
        )

    add("row_id", "id", "traceability", "identificador secuencial", "raw order", "Une artifacts 1:1; no es predictor.", "Rastrea cada tanda.", uplift_use="no", roi_use="no", optimizer_use="no", status="excluded")
    add("id_material", "original_categorica", "core", "original", "id_material", "Identifica heterogeneidad por SKU.", "Distingue respuesta por producto.")
    add("subcadena", "original_categorica", "core", "original", "subcadena", "Contexto de cadena.", "Captura ejecucion y shopper distintos.")
    add("factor_descuento", "decision", "core", "original", "factor_descuento", "Palanca continua del escenario.", "Profundidad promocional.", changes_discount=True)
    add("duracion_dias", "decision", "core", "original", "duracion_dias", "Palanca discreta del escenario.", "Dias de exposicion promocional.", changes_duration=True)
    add("volumen_base_sem", "original_numerica", "core", "original", "volumen_base_sem", "Baseline previo semanal.", "Escala el impacto absoluto del uplift.")
    add("elasticidad_estimada", "original_numerica", "core", "original", "elasticidad_estimada", "Sensibilidad con signo.", "Respuesta historica estimada al precio.")
    add("flag_secundario", "original_categorica", "core", "original", "flag_secundario", "Control de otra mecanica conocida.", "Separa secundaria, no secundaria y desconocido.")
    add("duracion_semanas", "derivada", "core", "duracion_dias / 7", "duracion_dias", "Alinea duracion con baseline semanal.", "Expresa la tanda en semanas.", changes_duration=True)
    add("volumen_base_tanda", "derivada", "core", "volumen_base_sem * duracion_dias / 7", "volumen_base_sem; duracion_dias", "Unidades esperadas sin promocion durante la tanda.", "Convierte uplift esperado en unidades incrementales.", changes_duration=True)
    add("elasticidad_abs", "derivada", "core", "abs(elasticidad_estimada)", "elasticidad_estimada", "Magnitud de sensibilidad.", "Facilita interpretar sensibilidad al descuento.")
    add("descuento_x_elasticidad", "interaccion_comercial", "core", "factor_descuento * elasticidad_abs", "factor_descuento; elasticidad_abs", "Presion promocional ajustada por sensibilidad.", "Mismo descuento puede responder distinto por SKU.", changes_discount=True)
    add("descuento_x_duracion", "interaccion_comercial", "core", "factor_descuento * duracion_dias", "factor_descuento; duracion_dias", "Intensidad por exposicion; no es inversion.", "Distingue tandas de igual descuento y distinta duracion.", changes_discount=True, changes_duration=True)
    add("factor_descuento_sq", "no_lineal", "core", "factor_descuento ** 2", "factor_descuento", "Permite curvatura sin imponer causalidad.", "Representa saturacion potencial.", changes_discount=True)
    add("duracion_dias_sq", "no_lineal", "core", "duracion_dias ** 2", "duracion_dias", "Permite curvatura sin imponer forma verdadera.", "Representa rendimientos decrecientes potenciales.", changes_duration=True)
    add("log1p_volumen_base_sem", "transformacion_log", "core", "log1p(volumen_base_sem)", "volumen_base_sem", "Reduce asimetria conservando original.", "Evita que SKUs grandes dominen la escala.")
    add("log1p_volumen_base_tanda", "transformacion_log", "core", "log1p(volumen_base_tanda)", "volumen_base_tanda", "Representa escala relativa de tanda.", "Compara impactos entre tamanos de negocio.", changes_duration=True)
    for name, formula, business in [
        ("mes", "month(fecha_inicio_tanda)", "Mes conocido de ejecucion."),
        ("trimestre", "quarter(fecha_inicio_tanda)", "Trimestre conocido de ejecucion."),
        ("semana_anio", "ISO week(fecha_inicio_tanda)", "Semana conocida de ejecucion."),
        ("mes_sin", "sin(2*pi*mes/12)", "Continuidad estacional entre diciembre y enero."),
        ("mes_cos", "cos(2*pi*mes/12)", "Continuidad estacional entre diciembre y enero."),
        ("semana_anio_sin", "sin(2*pi*semana_anio/52)", "Continuidad estacional semanal."),
        ("semana_anio_cos", "cos(2*pi*semana_anio/52)", "Continuidad estacional semanal."),
    ]:
        add(name, "temporal", "core", formula, "fecha_inicio_tanda", "Componente calendario prepromocion.", business)

    add("precio_base", "opcional_numerica", "optional", "original via row_id", "processed_modeling.precio_base", "Constante por SKU; potencialmente redundante.", "Puede aportar escala monetaria para ROI.", uplift_use="sensibilidad", roi_use="sensibilidad", optimizer_use="contexto", status="optional")
    add("des_marca", "opcional_categorica", "optional", "original via row_id", "model_index.des_marca", "Deterministica respecto a SKU en este historico.", "Puede ayudar en cold start.", uplift_use="sensibilidad", roi_use="sensibilidad", optimizer_use="contexto", status="optional")
    add("flag_secundario_missing", "opcional_calidad", "optional", "original via row_id", "processed_modeling.flag_secundario_missing", "Distingue ausencia de registro.", "Evita confundir desconocido con no secundaria.", uplift_use="sensibilidad", roi_use="sensibilidad", optimizer_use="contexto", status="optional")
    add("sku_cadena", "opcional_interaccion_categorica", "optional", "id_material + '__' + subcadena", "id_material; subcadena", "Interaccion local de 45 combinaciones con riesgo de sobreajuste.", "Captura ejecucion especifica producto-cadena.", uplift_use="sensibilidad", roi_use="sensibilidad", optimizer_use="contexto", status="optional")
    add("dias_desde_inicio_dataset", "opcional_tendencia", "optional", "fecha - min(fecha)", "fecha_inicio_tanda", "Tendencia temporal que puede extrapolar mal.", "Proxy de cambios estructurales, no causal.", uplift_use="sensibilidad", roi_use="sensibilidad", optimizer_use="contexto", status="optional")

    for name, target in [("mean_uplift_by_sku_shifted", "uplift_real"), ("mean_roi_by_sku_chain_shifted", "roi")]:
        add(name, "historica_target", "future_fold_aware", "expanding mean con shift y corte temporal", target, "Solo observaciones estrictamente anteriores dentro del fold.", "Resume historia previa sin mirar el futuro.", uplift_use="futuro", roi_use="futuro", optimizer_use="contexto", status="future fold-aware", availability="historica previa", leakage="alto si no es fold-aware")
    for name in [TARGET_UPLIFT, TARGET_ROI, "flag_uplift_en_piso", "fecha_inicio_tanda"]:
        add(name, "target_o_guardrail", "excluded", "original", name, "Reservada para target, evaluacion o indice.", "No disponible como predictor directo.", uplift_use="no", roi_use="no", optimizer_use="no", status="excluded", availability="postpromocion o indice", leakage="alto/target")
    return pd.DataFrame(rows)


def build_optimizer_compatibility(catalog: pd.DataFrame) -> pd.DataFrame:
    active = catalog[catalog["estado"].isin(["core", "optional"])].copy()
    return active[
        [
            "nombre",
            "tipo",
            "cambia_con_descuento",
            "cambia_con_duracion",
            "disponibilidad_temporal",
            "uso_optimizacion",
        ]
    ].rename(
        columns={
            "nombre": "feature",
            "disponibilidad_temporal": "disponible_antes_tanda",
        }
    )


def engineered_feature_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        col for col in frame.select_dtypes(include=[np.number]).columns if col != "row_id"
    ]
    return frame[numeric_columns].corr(method="spearman")


def engineered_target_association(
    core: pd.DataFrame, targets: pd.DataFrame
) -> pd.DataFrame:
    """Descriptive target associations for reporting, never feature selection."""
    analysis = core.merge(
        targets[["row_id", TARGET_UPLIFT, TARGET_ROI]],
        on="row_id",
        validate="one_to_one",
        sort=False,
    )
    numeric_columns = [
        col for col in core.select_dtypes(include=[np.number]).columns if col != "row_id"
    ]
    rows = []
    for column in numeric_columns:
        rows.append(
            {
                "feature": column,
                "spearman_uplift": analysis[column].corr(analysis[TARGET_UPLIFT], method="spearman"),
                "spearman_roi": analysis[column].corr(analysis[TARGET_ROI], method="spearman"),
                "observaciones": len(analysis),
                "uso": "diagnostico descriptivo; no seleccion automatica",
            }
        )
    return pd.DataFrame(rows)


def engineered_feature_redundancy(
    correlation: pd.DataFrame, threshold: float = 0.90
) -> pd.DataFrame:
    direct = {
        frozenset(["duracion_dias", "duracion_semanas"]): "transformacion lineal exacta",
        frozenset(["volumen_base_sem", "log1p_volumen_base_sem"]): "transformacion monotona",
        frozenset(["volumen_base_tanda", "log1p_volumen_base_tanda"]): "transformacion monotona",
        frozenset(["elasticidad_estimada", "elasticidad_abs"]): "magnitud derivada",
        frozenset(["factor_descuento", "factor_descuento_sq"]): "termino cuadratico",
        frozenset(["duracion_dias", "duracion_dias_sq"]): "termino cuadratico",
    }
    rows = []
    columns = list(correlation.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            corr = float(correlation.loc[left, right])
            relation = direct.get(frozenset([left, right]), "correlacion empirica alta")
            if abs(corr) >= threshold or frozenset([left, right]) in direct:
                rows.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "spearman": corr,
                        "relacion": relation,
                        "impacto_modelo": "posible multicolinealidad en modelos lineales",
                        "decision": "conservar; evaluar regularizacion y ablation temporal",
                        "arboles": "potencialmente redundante, no necesariamente perjudicial",
                    }
                )
    return pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def validate_catalog_coverage(
    core: pd.DataFrame, optional: pd.DataFrame, catalog: pd.DataFrame
) -> None:
    artifact_features = set(core.columns) | set(optional.columns)
    missing = artifact_features - set(catalog["nombre"])
    empty_justification = catalog[
        catalog["interpretacion_tecnica"].str.strip().eq("")
        | catalog["interpretacion_negocio"].str.strip().eq("")
    ]
    if missing or not empty_justification.empty:
        raise ValueError(
            f"Feature catalog invalid. Missing={sorted(missing)}, empty_justification={empty_justification['nombre'].tolist()}"
        )


def create_feature_engineering_manifest(
    core: pd.DataFrame,
    optional: pd.DataFrame,
    catalog: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict:
    return {
        "version": "1.0",
        "phase": "feature_engineering",
        "row_count": len(core),
        "artifacts": {
            "core": {"columns": core.columns.tolist(), "shape": list(core.shape)},
            "optional_extended": {"columns": optional.columns.tolist(), "shape": list(optional.shape)},
        },
        "feature_sets": {
            "FEATURES_ENGINEERED_CORE": FEATURES_ENGINEERED_CORE,
            "FEATURES_ENGINEERED_OPTIONAL": FEATURES_ENGINEERED_OPTIONAL,
            "FEATURES_ENGINEERED_TEMPORAL": FEATURES_ENGINEERED_TEMPORAL,
            "FEATURES_CATEGORICAL": FEATURES_CATEGORICAL,
            "FEATURES_NUMERICAL": FEATURES_NUMERICAL,
            "FEATURES_FOR_UPLIFT": FEATURES_FOR_UPLIFT,
            "FEATURES_FOR_ROI": FEATURES_FOR_ROI,
        },
        "guardrails": [
            "No targets, postpromotion outcomes or audit columns in feature artifacts.",
            "No global fitting of encoders, scalers, imputers or supervised selectors.",
            "Historical target features remain future fold-aware and are not materialized.",
            "A future uplift prediction for ROI must be generated out-of-fold during training.",
        ],
        "validation": validation.to_dict("records"),
        "catalog": catalog.to_dict("records"),
    }


def build_preprocessor(model_family: str) -> ColumnTransformer:
    """Return an unfitted preprocessing template for use inside temporal folds."""
    if model_family not in {"linear", "tree"}:
        raise ValueError("model_family must be 'linear' or 'tree'")
    numeric_transformer = StandardScaler() if model_family == "linear" else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAL),
            ("numeric", numeric_transformer, FEATURES_NUMERICAL),
        ],
        remainder="drop",
    )


def write_feature_engineering_manifest(manifest: dict, path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
