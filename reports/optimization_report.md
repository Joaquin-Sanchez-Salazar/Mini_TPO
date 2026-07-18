# Optimization Report

La seleccion de modelos se mantuvo congelada. Se crearon refits de produccion con las 2,048 observaciones despues de completar la evaluacion; no hubo tuning adicional.

- Fecha de referencia: 2025-12-29.
- Error absoluto OOF q90 uplift: 0.2955.
- Error absoluto OOF q90 ROI: 0.0324.
- Tolerancia de empate ROI: 0.0100.
- Grilla principal: descuento 5%-40% cada 1 pp y duraciones 5, 7, 10, 14 y 21 dias.

## Recomendaciones

| case_id | id_material | des_material | des_marca | subcadena | fecha_referencia | nivel_soporte | observaciones_historicas | volumen_base_sem | elasticidad_estimada | factor_descuento_recomendado | duracion_dias_recomendada | uplift_esperado | uplift_lower_90 | uplift_upper_90 | volumen_base_tanda | volumen_promo_esperado | volumen_incremental_esperado | roi_esperado | roi_lower_90 | roi_upper_90 | roi_robusto | soporte_local | flag_extrapolacion | flag_optimo_en_limite_descuento | flag_optimo_en_limite_duracion | tipo_recomendacion | razon_seleccion | advertencia_negocio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| caso_a_soporte_alto | PD008 | SKU06 | CAT02 | Cadena01 | 2025-12-29 00:00:00 | soporte_alto | 62 | 2849.45 | -2.0084999999999997 | 0.09 | 14 | 0.25029283462793017 | -0.0452062677898834 | 0.5457919370457438 | 5698.9 | 7125.293835261112 | 1426.3938352611112 | 2.9783107366119714 | 2.94595386449127 | 3.010667608732673 | 2.94595386449127 | 1 | False | False | False | RECOMENDACION_ROBUSTA | maximo ROI robusto con tolerancia y desempate por volumen/soporte/costo operativo | Prediccion sujeta a error OOF; confirmar stock, presupuesto y ejecucion fuera del modelo. |
| caso_b_alto_volumen | PD013 | SKU11 | CAT04 | Cadena02 | 2025-12-29 00:00:00 | soporte_medio | 52 | 3828.55 | -1.0735000000000001 | 0.1 | 14 | 0.17741043716331262 | -0.11808866525450096 | 0.4729095395811262 | 7657.1 | 9015.549458403202 | 1358.449458403201 | 2.5120118036791683 | 2.479654931558467 | 2.54436867579987 | 2.469654931558467 | 1 | False | False | False | RECOMENDACION_ROBUSTA | maximo ROI robusto con tolerancia y desempate por volumen/soporte/costo operativo | Prediccion sujeta a error OOF; confirmar stock, presupuesto y ejecucion fuera del modelo. |
| caso_c_sku_reciente | PD015 | SKU15 | CAT05 | Cadena03 | 2025-12-29 00:00:00 | soporte_bajo | 15 | 4367.549999999999 | -0.718 | 0.12 | 21 | 0.11978205590646085 | -0.17571704651135273 | 0.4152811583242744 | 13102.649999999998 | 14672.112354822786 | 1569.462354822789 | 1.765906073134446 | 1.7335492010137448 | 1.7982629452551473 | 1.7035492010137447 | 1 | False | False | True | REQUIERE_REVISION_HUMANA | maximo ROI robusto con tolerancia y desempate por volumen/soporte/costo operativo | SKU reciente o soporte bajo: validar con Trade Marketing antes de aprobar. |

## Lectura de negocio

El optimo matematico maximiza ROI puntual. La recomendacion robusta exige intervalo inferior positivo, soporte local y ausencia de extrapolacion critica. La alternativa de crecimiento maximiza unidades incrementales dentro de los mismos guardrails. Ninguna prediccion es una garantia causal y las restricciones de stock, presupuesto y margen no estan disponibles.
