"""Central feature sets for feature engineering and future modeling."""

FEATURES_ENGINEERED_INTERACTIONS = [
    "descuento_x_elasticidad",
    "descuento_x_duracion",
]

FEATURES_ENGINEERED_TEMPORAL = [
    "mes",
    "trimestre",
    "semana_anio",
    "mes_sin",
    "mes_cos",
    "semana_anio_sin",
    "semana_anio_cos",
]

FEATURES_ENGINEERED_CORE = [
    "id_material",
    "subcadena",
    "factor_descuento",
    "duracion_dias",
    "volumen_base_sem",
    "elasticidad_estimada",
    "flag_secundario",
    "duracion_semanas",
    "volumen_base_tanda",
    "elasticidad_abs",
    "descuento_x_elasticidad",
    "descuento_x_duracion",
    "factor_descuento_sq",
    "duracion_dias_sq",
    "log1p_volumen_base_sem",
    "log1p_volumen_base_tanda",
    *FEATURES_ENGINEERED_TEMPORAL,
]

FEATURES_ENGINEERED_OPTIONAL = [
    "precio_base",
    "des_marca",
    "flag_secundario_missing",
    "sku_cadena",
    "dias_desde_inicio_dataset",
]

FEATURES_CATEGORICAL = [
    "id_material",
    "subcadena",
    "flag_secundario",
    "mes",
    "trimestre",
]

FEATURES_NUMERICAL = [
    column for column in FEATURES_ENGINEERED_CORE if column not in FEATURES_CATEGORICAL
]

FEATURES_FOR_UPLIFT = FEATURES_ENGINEERED_CORE.copy()
FEATURES_FOR_ROI = FEATURES_ENGINEERED_CORE.copy()

FEATURES_CONTEXT_FIXED = [
    "id_material",
    "subcadena",
    "volumen_base_sem",
    "elasticidad_estimada",
    "flag_secundario",
    "elasticidad_abs",
    "log1p_volumen_base_sem",
    *FEATURES_ENGINEERED_TEMPORAL,
]

FEATURES_CHANGE_WITH_DISCOUNT = [
    "factor_descuento",
    "descuento_x_elasticidad",
    "descuento_x_duracion",
    "factor_descuento_sq",
]

FEATURES_CHANGE_WITH_DURATION = [
    "duracion_dias",
    "duracion_semanas",
    "volumen_base_tanda",
    "descuento_x_duracion",
    "duracion_dias_sq",
    "log1p_volumen_base_tanda",
]
