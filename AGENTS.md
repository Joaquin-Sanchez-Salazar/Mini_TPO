# AGENTS

Este repositorio corresponde a una prueba tecnica de analitica promocional para Alicorp.

## Estado

- EDA, limpieza, preparacion, feature engineering y comparacion de modelos completos.
- Artifacts de features, targets e indice separados y trazables por `row_id`.
- La fase de optimizacion esta completa; no representa un sistema productivo de aprobacion automatica.

## Optimizacion

- Champions heredados: Ridge para uplift y HistGradientBoosting directo para ROI.
- Refits production: mismos features e hiperparametros, ajustados sobre 2,048 filas sin retuning.
- Casos: `PD008 x Cadena01`, `PD013 x Cadena02` y `PD015 x Cadena03`.
- Fecha de referencia: 2025-12-29; contexto principal: mediana de ultimas cuatro observaciones y promocion no secundaria.
- Grilla: descuento 5%-40% cada 1 pp; duraciones principales 5, 7, 10, 14 y 21 dias.
- Objetivos: ROI puntual, ROI robusto con intervalo/penalizacion de soporte y crecimiento rentable.
- Incertidumbre: cuantiles de errores absolutos OOF de champions; intervalos empiricos, no garantias.
- Soporte local: misma duracion y descuento dentro de +/-2.5 pp.
- `PD015` requiere revision humana por ser reciente y de soporte bajo.
- Recomendaciones actuales: `PD008 x Cadena01` 9%/14 dias; `PD013 x Cadena02` 10%/14 dias; `PD015 x Cadena03` 12%/21 dias con revision humana.

## Diseno temporal y seleccion

- Test final aislado: octubre a diciembre de 2025, 357 filas, 15 SKUs y 3 cadenas.
- Desarrollo: julio de 2024 a septiembre de 2025, con cuatro ventanas expansivas por fecha completa.
- Baselines: mediana global, jerarquia SKU x cadena con fallback SKU, cadena y global; para ROI, mediana por banda de descuento.
- Familias: Ridge, HistGradientBoosting y ExtraTrees. Los grids son pequenos y se ajustan solo en desarrollo.
- Metricas uplift: MAE, RMSE, WAPE, sMAPE, bias, R2 y error en unidades.
- Metricas ROI: MAE, RMSE, mediana AE, bias, R2, Spearman, signo y falsos positivos/negativos.
- Champion uplift: Ridge semanal, escala original, `alpha=100`; challenger: Ridge semanal con `log1p`, `alpha=10`.
- Champion ROI: HistGradientBoosting robusto; challenger: HistGradientBoosting mas regularizado.
- El ROI de dos etapas fue peor en todos los folds comparables y no se usa.
- La incertidumbre es un intervalo aproximado por cuantil de residuos OOF, no una garantia causal ni conformal estricta.

## Supuestos

- `volumen_base_sem` y `elasticidad_estimada` estan disponibles antes de la tanda.
- `roi` es el KPI oficial de Alicorp y un target valido; nunca es predictor.
- La fecha se conoce al disenar la promocion, pero solo sus derivados de calendario entran como features.
- `row_id` es trazabilidad, no predictor; es estable mientras se preserve el orden del raw.

## Features core

Originales: `id_material`, `subcadena`, `factor_descuento`, `duracion_dias`, `volumen_base_sem`, `elasticidad_estimada` y `flag_secundario`.

Derivadas principales:

- `duracion_semanas = duracion_dias / 7`.
- `volumen_base_tanda = volumen_base_sem * duracion_dias / 7`.
- `elasticidad_abs = abs(elasticidad_estimada)`.
- `descuento_x_elasticidad = factor_descuento * elasticidad_abs`.
- `descuento_x_duracion = factor_descuento * duracion_dias`.
- Cuadrados de descuento y duracion.
- `log1p` de volumen base semanal y de tanda.
- Mes, trimestre, semana y codificacion ciclica mensual/semanal.

## Features opcionales

- `precio_base`: constante por SKU; sensibilidad para ROI o modelos no lineales.
- `des_marca`: redundante con SKU; sensibilidad para cold start.
- `flag_secundario_missing`: indicador de calidad.
- `sku_cadena`: heterogeneidad local con riesgo de sobreajuste.
- `dias_desde_inicio_dataset`: tendencia temporal con riesgo de extrapolacion.

## Excluidas o futuras

- Targets, resultados postpromocion, auditoria y fecha original estan excluidos.
- El piso de uplift es evaluacion/guardrail, no feature.
- Promedios historicos de uplift o ROI no estan materializados; solo podran evaluarse con `shift`, fechas estrictamente anteriores y tratamiento fold-aware.
- `uplift_real` nunca puede predecir ROI.
- Un futuro `uplift_predicho` para ROI debe generarse out-of-fold durante entrenamiento.

## Compatibilidad con el optimizador

- Cambian con descuento: descuento, interaccion descuento-elasticidad, interaccion descuento-duracion y descuento cuadratico.
- Cambian con duracion: duracion, semanas, volumen base de tanda, interaccion descuento-duracion, duracion cuadratica y log del volumen base de tanda.
- El optimizador futuro debe recalcular todas las dependencias para cada candidato.
- `descuento_x_duracion` no representa inversion real.

## Guardrails

- No modificar `data/raw/` ni sobrescribir artifacts seguros de fase 02.
- Usar rutas relativas y centralizar listas en `feature_sets.py`, `constants.py` o configuracion.
- No usar targets, postpromocion ni auditoria como features.
- No ajustar encoders, escaladores, imputadores o selectores sobre todo el dataset.
- Encoders y escaladores deben ajustarse dentro de cada fold temporal.
- No usar target encoding global.
- No seleccionar features solo por correlacion.
- No interpretar elasticidad, interacciones o relaciones historicas como causalidad.
- Evaluar error porcentual y error en unidades por SKU y cadena.
- Restringir futuras recomendaciones al soporte historico.
- No usar el test final para tuning ni cambiar modelos despues de observarlo.
- No usar `uplift_real` como predictor de ROI; cualquier encadenamiento debe usar uplift OOF.
- Recalcular features dependientes de descuento y duracion para cada candidato del optimizador.
- Considerar incertidumbre y evitar explotar picos artificiales de modelos de arboles.
- No optimizar fuera de 5%-40% o 5-21 dias.
- No usar una prediccion puntual sin incertidumbre.
- No recomendar soporte insuficiente automaticamente.
- No ocultar optimos en limites ni forzar un optimo interior.
- No usar `uplift_real` o `roi` en escenarios futuros.
- No inventar costos, margenes, stock o presupuesto ausentes.
- Recalcular todas las features dependientes de descuento y duracion.

## Artifacts de fase 03

- `data/processed/model_features_engineered_core.parquet`.
- `data/processed/model_features_engineered_optional.parquet`.
- `data/processed/feature_engineering_manifest.json`.
- `reports/tables/feature_engineering_catalog.csv`.
- `reports/tables/feature_optimizer_compatibility.csv`.
- `reports/tables/engineered_feature_correlation.csv`.
- `reports/tables/engineered_feature_redundancy.csv`.
- `reports/tables/engineered_feature_target_association.csv`.
- `reports/feature_engineering_report.md`.

## Artifacts de fase 04

- `data/processed/oof_predictions_uplift.parquet` y `oof_predictions_roi.parquet`.
- `data/processed/final_test_predictions_uplift.parquet` y `final_test_predictions_roi.parquet`.
- `models/uplift_champion.joblib`, `uplift_challenger.joblib`, `roi_champion.joblib` y `roi_challenger.joblib`.
- `models/model_registry.json`.
- Tablas CV, segmentos, incertidumbre, importancia, scorecard y test en `reports/tables/`.
- `reports/modeling_report.md` y `reports/04_modeling_comparison.html`.

## Artifacts de fase 05

- `data/processed/optimization_scenarios.parquet`.
- Casos, contextos, recomendaciones, optimos, crecimiento, Pareto y sensibilidad en `reports/tables/`.
- Quince figuras en `reports/figures/optimization/`.
- `models/uplift_production.joblib` y `models/roi_production.joblib`.
- `reports/optimization_report.md` y `reports/05_optimization.html`.

## Proxima fase

Antes de uso real: incorporar margen, inversion, presupuesto, stock y reglas comerciales; validar en experimentos controlados y monitorear drift, cobertura y error por SKU/cadena.
