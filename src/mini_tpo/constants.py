"""Centralized column roles and modeling guardrails."""

ID_COLUMNS = ["row_id", "id_material"]
DATE_COLUMNS = ["fecha_inicio_tanda"]
TARGET_UPLIFT = "uplift_real"
TARGET_ROI = "roi"
RANDOM_SEED = 42
MODEL_FAMILIES = ["ridge", "hist_gradient_boosting", "extra_trees"]
OPTIMIZATION_CASES = [
    ("PD008", "Cadena01", "caso_a_soporte_alto"),
    ("PD013", "Cadena02", "caso_b_alto_volumen"),
    ("PD015", "Cadena03", "caso_c_sku_reciente"),
]

PRE_PROMO_CONFIRMED = [
    "id_material",
    "subcadena",
    "factor_descuento",
    "duracion_dias",
    "volumen_base_sem",
    "elasticidad_estimada",
    "flag_secundario",
]

PRE_PROMO_CONDITIONAL = []

POST_PROMO_COLUMNS = [
    "volumen_promo",
    "venta_promo",
    "inversion_promo",
    "uplift_real",
    "roi",
]

LEAKAGE_COLUMNS = [
    "volumen_promo",
    "venta_promo",
    "inversion_promo",
    "uplift_real",
    "roi",
]

AUDIT_COLUMNS = [
    "audit_volumen_base_tanda",
    "audit_volumen_incremental_observado",
    "audit_uplift_recalculado",
    "audit_venta_promo_recalculada",
    "audit_inversion_promo_recalculada",
    "audit_diff_abs_uplift_real",
    "audit_diff_rel_uplift_real",
    "audit_diff_abs_venta_promo",
    "audit_diff_rel_venta_promo",
    "audit_diff_abs_inversion_promo",
    "audit_diff_rel_inversion_promo",
    "flag_inconsistencia_uplift",
    "flag_inconsistencia_venta",
    "flag_inconsistencia_inversion",
]

REPORTING_ONLY_COLUMNS = [
    "des_material",
    "fecha_inicio_tanda",
]

FEATURES_MODEL_BASE = [
    "id_material",
    "subcadena",
    "factor_descuento",
    "duracion_dias",
    "volumen_base_sem",
    "elasticidad_estimada",
    "flag_secundario",
]

FEATURES_MODEL_MINIMAL = [
    "id_material",
    "subcadena",
    "factor_descuento",
    "duracion_dias",
    "flag_secundario",
]

FEATURES_MODEL_OPTIONAL = [
    "precio_base",
    "des_marca",
    "flag_secundario_missing",
]

FEATURES_EXCLUDED_BASE = [
    "row_id",
    "fecha_inicio_tanda",
    "des_material",
    "des_marca",
    "precio_base",
    "volumen_promo",
    "venta_promo",
    "inversion_promo",
    "uplift_real",
    "roi",
    "flag_secundario_missing",
]

FORBIDDEN_ENGINEERED_COLUMNS = sorted(
    set(POST_PROMO_COLUMNS + LEAKAGE_COLUMNS + AUDIT_COLUMNS)
)
