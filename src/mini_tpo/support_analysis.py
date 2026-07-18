from __future__ import annotations

import numpy as np
import pandas as pd


DISCOUNT_LABELS = ["[5%,10%)", "[10%,15%)", "[15%,20%)", "[20%,25%)", "[25%,30%)", "[30%,35%)", "[35%,40%]"]

DEFAULT_SUPPORT_THRESHOLDS = {
    "alto": {"min_observations": 60, "min_discount_bands": 4, "min_durations": 4},
    "medio": {"min_observations": 35, "min_discount_bands": 3, "min_durations": 3},
    "bajo": {"min_observations": 15, "min_discount_bands": 2, "min_durations": 2},
}


def add_discount_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4000000001]
    out["banda_descuento_opt"] = pd.cut(
        out["factor_descuento"],
        bins=bins,
        labels=DISCOUNT_LABELS,
        include_lowest=True,
        right=False,
    ).astype("object")
    out["banda_descuento_opt"] = out["banda_descuento_opt"].fillna("fuera_dominio")
    return out


def _joined_unique(values: pd.Series) -> str:
    return ",".join(str(v) for v in sorted(values.dropna().unique()))


def build_sku_chain_support(
    df: pd.DataFrame,
    floor_value: float = 0.05,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    base = add_discount_band(df)
    grouped = base.groupby(["id_material", "subcadena"], observed=True)
    support = grouped.agg(
        promociones=("row_id", "size"),
        fecha_min=("fecha_inicio_tanda", "min"),
        fecha_max=("fecha_inicio_tanda", "max"),
        n_fechas=("fecha_inicio_tanda", "nunique"),
        descuento_min=("factor_descuento", "min"),
        descuento_max=("factor_descuento", "max"),
        descuento_mediano=("factor_descuento", "median"),
        duraciones_observadas=("duracion_dias", _joined_unique),
        n_duraciones=("duracion_dias", "nunique"),
        volumen_base_mediano=("volumen_base_sem", "median"),
        uplift_medio=("uplift_real", "mean"),
        uplift_mediano=("uplift_real", "median"),
        roi_medio=("roi", "mean"),
        roi_mediano=("roi", "median"),
        pct_roi_negativo=("roi", lambda s: float((s < 0).mean())),
        pct_uplift_en_piso=("flag_uplift_en_piso", "mean"),
        observaciones_dentro_dominio=("flag_fuera_dominio_optimizacion", lambda s: int((~s).sum())),
        observaciones_fuera_dominio=("flag_fuera_dominio_optimizacion", "sum"),
        n_bandas_descuento=("banda_descuento_opt", lambda s: int(s[s != "fuera_dominio"].nunique())),
    ).reset_index()
    support["meses_historia"] = (
        (support["fecha_max"].dt.year - support["fecha_min"].dt.year) * 12
        + (support["fecha_max"].dt.month - support["fecha_min"].dt.month)
        + 1
    )
    support["clasificacion_soporte"] = support.apply(
        classify_support_row, axis=1, thresholds=thresholds
    )
    return support.sort_values(["clasificacion_soporte", "promociones", "id_material", "subcadena"]).reset_index(drop=True)


def classify_support_row(row: pd.Series, thresholds: dict | None = None) -> str:
    rules = thresholds or DEFAULT_SUPPORT_THRESHOLDS
    n = row["promociones"]
    durations = row["n_duraciones"]
    bands = row["n_bandas_descuento"]
    if n >= rules["alto"]["min_observations"] and durations >= rules["alto"]["min_durations"] and bands >= rules["alto"]["min_discount_bands"]:
        return "soporte_alto"
    if n >= rules["medio"]["min_observations"] and durations >= rules["medio"]["min_durations"] and bands >= rules["medio"]["min_discount_bands"]:
        return "soporte_medio"
    if n >= rules["bajo"]["min_observations"] and durations >= rules["bajo"]["min_durations"] and bands >= rules["bajo"]["min_discount_bands"]:
        return "soporte_bajo"
    return "soporte_insuficiente"


def support_rules_table(thresholds: dict | None = None) -> pd.DataFrame:
    rules = thresholds or DEFAULT_SUPPORT_THRESHOLDS
    rows = []
    for level in ["alto", "medio", "bajo"]:
        rule = rules[level]
        rows.append({"nivel": level, **rule})
    rows.append(
        {
            "nivel": "insuficiente",
            "min_observations": 0,
            "min_discount_bands": 0,
            "min_durations": 0,
        }
    )
    return pd.DataFrame(rows)


def build_discount_support(df: pd.DataFrame) -> pd.DataFrame:
    base = add_discount_band(df)
    table = (
        base.groupby(["id_material", "subcadena", "banda_descuento_opt"], observed=True)
        .size()
        .reset_index(name="observaciones")
        .sort_values(["id_material", "subcadena", "banda_descuento_opt"])
    )
    return table


def build_duration_support(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["id_material", "subcadena", "duracion_dias"], observed=True)
        .size()
        .reset_index(name="observaciones")
        .sort_values(["id_material", "subcadena", "duracion_dias"])
    )


def discount_uplift_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = add_discount_band(df)
    return (
        base.groupby(["banda_descuento_opt", "subcadena"], observed=True)
        .agg(
            observaciones=("uplift_real", "size"),
            uplift_medio=("uplift_real", "mean"),
            uplift_mediano=("uplift_real", "median"),
            uplift_std=("uplift_real", "std"),
            uplift_q25=("uplift_real", lambda s: s.quantile(0.25)),
            uplift_q75=("uplift_real", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )


def discount_roi_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = add_discount_band(df)
    return (
        base.groupby(["banda_descuento_opt", "subcadena"], observed=True)
        .agg(
            observaciones=("roi", "size"),
            roi_medio=("roi", "mean"),
            roi_mediano=("roi", "median"),
            roi_q25=("roi", lambda s: s.quantile(0.25)),
            roi_q75=("roi", lambda s: s.quantile(0.75)),
            pct_roi_negativo=("roi", lambda s: float((s < 0).mean())),
        )
        .reset_index()
    )


def duration_uplift_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["duracion_dias", "subcadena"], observed=True)
        .agg(
            observaciones=("uplift_real", "size"),
            uplift_medio=("uplift_real", "mean"),
            uplift_mediano=("uplift_real", "median"),
            uplift_std=("uplift_real", "std"),
        )
        .reset_index()
    )


def duration_roi_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["duracion_dias", "subcadena"], observed=True)
        .agg(
            observaciones=("roi", "size"),
            roi_medio=("roi", "mean"),
            roi_mediano=("roi", "median"),
            pct_roi_negativo=("roi", lambda s: float((s < 0).mean())),
        )
        .reset_index()
    )


def secondary_promo_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("flag_secundario", observed=True)
        .agg(
            observaciones=("row_id", "size"),
            uplift_medio=("uplift_real", "mean"),
            uplift_mediano=("uplift_real", "median"),
            roi_medio=("roi", "mean"),
            roi_mediano=("roi", "median"),
            pct_roi_negativo=("roi", lambda s: float((s < 0).mean())),
            descuento_medio=("factor_descuento", "mean"),
            duracion_media=("duracion_dias", "mean"),
        )
        .reset_index()
    )


def baseline_impact_summary(df: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    work = df.copy()
    work["banda_volumen_base"] = pd.qcut(
        work["volumen_base_sem"], q=quantiles, duplicates="drop"
    ).astype("string")
    return (
        work.groupby("banda_volumen_base", observed=True)
        .agg(
            observaciones=("row_id", "size"),
            volumen_base_mediano=("volumen_base_sem", "median"),
            uplift_mediano=("uplift_real", "median"),
            roi_mediano=("roi", "median"),
            volumen_incremental_mediano=("audit_volumen_incremental_observado", "median"),
        )
        .reset_index()
    )


def elasticity_discount_uplift_tables(
    df: pd.DataFrame, elasticity_quantiles: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = add_discount_band(df)
    work["banda_elasticidad"] = pd.qcut(
        work["elasticidad_estimada"], q=elasticity_quantiles, duplicates="drop"
    ).astype("string")
    grouped = (
        work.groupby(["banda_elasticidad", "banda_descuento_opt"], observed=True)
        .agg(uplift_mediano=("uplift_real", "median"), observaciones=("row_id", "size"))
        .reset_index()
    )
    uplift = grouped.pivot(
        index="banda_elasticidad", columns="banda_descuento_opt", values="uplift_mediano"
    ).reset_index()
    counts = grouped.pivot(
        index="banda_elasticidad", columns="banda_descuento_opt", values="observaciones"
    ).reset_index()
    return uplift, counts


def roi_tail_tables(df: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "fecha_inicio_tanda",
        "id_material",
        "des_material",
        "des_marca",
        "subcadena",
        "precio_base",
        "factor_descuento",
        "duracion_dias",
        "volumen_base_sem",
        "volumen_promo",
        "uplift_real",
        "venta_promo",
        "inversion_promo",
        "roi",
        "flag_secundario",
    ]
    return df.nlargest(n, "roi")[cols].reset_index(drop=True), df.nsmallest(n, "roi")[cols].reset_index(drop=True)


def roi_distribution_audit(df: pd.DataFrame) -> pd.DataFrame:
    quantiles = df["roi"].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    rows = {
        "media": df["roi"].mean(),
        "mediana": df["roi"].median(),
        "desv_std": df["roi"].std(),
        "min": df["roi"].min(),
        "max": df["roi"].max(),
        "pct_negativo": (df["roi"] < 0).mean(),
        "pct_cero": (df["roi"] == 0).mean(),
        "pct_positivo": (df["roi"] > 0).mean(),
    }
    for q, value in quantiles.items():
        rows[f"q{int(q * 100):02d}"] = value
    return pd.DataFrame([rows])


def floor_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, observed=True)["flag_uplift_en_piso"]
        .agg(observaciones="size", registros_en_piso="sum", pct_en_piso="mean")
        .reset_index()
        .sort_values(["pct_en_piso", "observaciones"], ascending=[False, False])
    )


def build_temporal_profile(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    work = df.copy()
    period_col = "periodo"
    work[period_col] = work["fecha_inicio_tanda"].dt.to_period(freq).astype(str)
    return (
        work.groupby(period_col, observed=True)
        .agg(
            promociones=("row_id", "size"),
            descuento_medio=("factor_descuento", "mean"),
            descuento_mediano=("factor_descuento", "median"),
            duracion_media=("duracion_dias", "mean"),
            duracion_mediana=("duracion_dias", "median"),
            uplift_medio=("uplift_real", "mean"),
            uplift_mediano=("uplift_real", "median"),
            roi_medio=("roi", "mean"),
            roi_mediano=("roi", "median"),
            volumen_base_medio=("volumen_base_sem", "mean"),
            volumen_base_mediano=("volumen_base_sem", "median"),
            elasticidad_media=("elasticidad_estimada", "mean"),
            elasticidad_mediana=("elasticidad_estimada", "median"),
            pct_roi_negativo=("roi", lambda s: float((s < 0).mean())),
            pct_promociones_secundarias=("flag_secundario", lambda s: float((s == "si").mean())),
            pct_flag_secundario_missing=("flag_secundario_missing", "mean"),
            pct_uplift_en_piso=("flag_uplift_en_piso", "mean"),
            n_skus=("id_material", "nunique"),
            n_cadenas=("subcadena", "nunique"),
        )
        .reset_index()
    )


def sku_history_profile(df: pd.DataFrame) -> pd.DataFrame:
    promo_chain = (
        df.groupby(["id_material", "subcadena"], observed=True)
        .size()
        .reset_index(name="n")
        .groupby("id_material", observed=True)
        .apply(lambda x: "; ".join(f"{r.subcadena}:{r.n}" for r in x.itertuples()), include_groups=False)
        .reset_index(name="promociones_por_cadena")
    )
    profile = (
        df.groupby("id_material", observed=True)
        .agg(
            primera_fecha=("fecha_inicio_tanda", "min"),
            ultima_fecha=("fecha_inicio_tanda", "max"),
            observaciones=("row_id", "size"),
            n_cadenas=("subcadena", "nunique"),
            descuento_min=("factor_descuento", "min"),
            descuento_max=("factor_descuento", "max"),
            duraciones_observadas=("duracion_dias", _joined_unique),
        )
        .reset_index()
    )
    profile["meses_historia"] = (
        (profile["ultima_fecha"].dt.year - profile["primera_fecha"].dt.year) * 12
        + (profile["ultima_fecha"].dt.month - profile["primera_fecha"].dt.month)
        + 1
    )
    profile = profile.merge(promo_chain, on="id_material", how="left")
    profile["clasificacion_sku"] = np.select(
        [
            (profile["observaciones"] >= 120) & (profile["meses_historia"] >= 12) & (profile["n_cadenas"] >= 3),
            (profile["observaciones"] >= 60) & (profile["meses_historia"] >= 8),
            (profile["observaciones"] >= 25) & (profile["meses_historia"] >= 4),
        ],
        ["sku_maduro", "sku_historia_media", "sku_reciente"],
        default="sku_soporte_insuficiente",
    )
    return profile.sort_values(["clasificacion_sku", "observaciones", "id_material"]).reset_index(drop=True)
