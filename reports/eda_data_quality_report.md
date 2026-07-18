# EDA Data Quality Report

Este reporte resume calidad, soporte y relaciones comerciales descriptivas. No establece causalidad.

- Registros raw: 2062
- Registros clean full: 2062
- Registros modeling dentro del dominio: 2048
- Registros fuera del dominio operativo: 14
- Faltantes en flag_secundario: 60

## Supuesto sobre ROI

Para esta prueba se asume que `roi` es el KPI oficial calculado por Alicorp y se acepta como target valido. Como su formula interna no esta disponible, no se realiza una descomposicion contable ni se atribuyen efectos causales a sus componentes.

## Outputs comerciales

Se generaron resumenes de descuento, duracion, elasticidad, volumen base y promocion secundaria en `reports/tables/`. Los extremos se conservan en las tablas.

## Hallazgos ejecutivos

| hallazgo | evidencia | impacto_modelo | impacto_negocio | decision_propuesta | riesgo_prioridad |
| --- | --- | --- | --- | --- | --- |
| Existen promociones fuera del dominio operativo futuro. | 14 registros fuera de descuento 5%-40% o duracion 5-21 dias. | Riesgo de extrapolacion si se entrenan o recomiendan zonas con poco soporte. | Puede sugerir descuentos o duraciones con ROI incierto. | Mantener en auditoria y excluir del dataset inicial de modelado. | Alta |
| flag_secundario tiene valores faltantes informativos. | 60 registros sin valor observado. | Imputar cero podria confundir desconocido con ausencia real. | Puede sesgar la lectura de mecanicas de exhibicion. | Crear categoria desconocido e indicador missing. | Media |
| El soporte historico no es homogeneo por SKU x cadena. | 6 combinaciones SKU x cadena tienen menos de 35 observaciones. | El error esperado puede variar localmente aunque el volumen global sea suficiente. | Una recomendacion dentro del rango global puede carecer de soporte para una combinacion especifica. | Usar clasificacion de soporte local como guardrail del optimizador. | Alta |
| Existe concentracion en el posible piso de uplift. | 6.2% de registros con uplift cercano a 0.05. | Puede afectar calibracion, residuos y perdida en promociones de bajo uplift. | El optimizador podria subestimar o sobrerrecomendar zonas de bajo descuento si no se controla. | Mantener valores originales y modelar/diagnosticar el flag en fases posteriores. | Media |
| Uplift y ROI capturan objetivos distintos. | 56.9% de promociones con ROI positivo; correlacion uplift-ROI debe interpretarse descriptivamente. | Optimizar solo uplift no maximiza necesariamente rentabilidad. | Mayor volumen puede requerir inversion que destruya valor. | Modelar uplift y ROI como targets complementarios y optimizar con restricciones de soporte. | Alta |
