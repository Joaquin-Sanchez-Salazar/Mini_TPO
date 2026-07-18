from __future__ import annotations

import pandas as pd


def executive_findings_table(df: pd.DataFrame) -> pd.DataFrame:
    outside = int(df["flag_fuera_dominio_optimizacion"].sum()) if "flag_fuera_dominio_optimizacion" in df else 0
    missing_flag = int(df["flag_secundario_missing"].sum()) if "flag_secundario_missing" in df else int(df["flag_secundario"].isna().sum())
    positive_roi = (df["roi"] > 0).mean()
    floor = df["flag_uplift_en_piso"].mean() if "flag_uplift_en_piso" in df else 0
    support_low = ""
    if {"id_material", "subcadena"}.issubset(df.columns):
        counts = df.groupby(["id_material", "subcadena"], observed=True).size()
        support_low = f"{int((counts < 35).sum())} combinaciones SKU x cadena tienen menos de 35 observaciones."
    return pd.DataFrame(
        [
            {
                "hallazgo": "Existen promociones fuera del dominio operativo futuro.",
                "evidencia": f"{outside} registros fuera de descuento 5%-40% o duracion 5-21 dias.",
                "impacto_modelo": "Riesgo de extrapolacion si se entrenan o recomiendan zonas con poco soporte.",
                "impacto_negocio": "Puede sugerir descuentos o duraciones con ROI incierto.",
                "decision_propuesta": "Mantener en auditoria y excluir del dataset inicial de modelado.",
                "riesgo_prioridad": "Alta",
            },
            {
                "hallazgo": "flag_secundario tiene valores faltantes informativos.",
                "evidencia": f"{missing_flag} registros sin valor observado.",
                "impacto_modelo": "Imputar cero podria confundir desconocido con ausencia real.",
                "impacto_negocio": "Puede sesgar la lectura de mecanicas de exhibicion.",
                "decision_propuesta": "Crear categoria desconocido e indicador missing.",
                "riesgo_prioridad": "Media",
            },
            {
                "hallazgo": "El soporte historico no es homogeneo por SKU x cadena.",
                "evidencia": support_low,
                "impacto_modelo": "El error esperado puede variar localmente aunque el volumen global sea suficiente.",
                "impacto_negocio": "Una recomendacion dentro del rango global puede carecer de soporte para una combinacion especifica.",
                "decision_propuesta": "Usar clasificacion de soporte local como guardrail del optimizador.",
                "riesgo_prioridad": "Alta",
            },
            {
                "hallazgo": "Existe concentracion en el posible piso de uplift.",
                "evidencia": f"{floor:.1%} de registros con uplift cercano a 0.05.",
                "impacto_modelo": "Puede afectar calibracion, residuos y perdida en promociones de bajo uplift.",
                "impacto_negocio": "El optimizador podria subestimar o sobrerrecomendar zonas de bajo descuento si no se controla.",
                "decision_propuesta": "Mantener valores originales y modelar/diagnosticar el flag en fases posteriores.",
                "riesgo_prioridad": "Media",
            },
            {
                "hallazgo": "Uplift y ROI capturan objetivos distintos.",
                "evidencia": f"{positive_roi:.1%} de promociones con ROI positivo; correlacion uplift-ROI debe interpretarse descriptivamente.",
                "impacto_modelo": "Optimizar solo uplift no maximiza necesariamente rentabilidad.",
                "impacto_negocio": "Mayor volumen puede requerir inversion que destruya valor.",
                "decision_propuesta": "Modelar uplift y ROI como targets complementarios y optimizar con restricciones de soporte.",
                "riesgo_prioridad": "Alta",
            },
        ]
    )
