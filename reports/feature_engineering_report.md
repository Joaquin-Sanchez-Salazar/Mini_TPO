# Feature Engineering Report

Fase de construccion de predictores prepromocion. No se entrenaron modelos, encoders ni escaladores, y no se construyo el optimizador.

- Filas de entrada: 2048
- Artifact core: 2048 filas y 24 columnas, incluyendo `row_id`.
- Artifact opcional extendido: 2048 filas y 29 columnas.
- Features core: 23
- Features opcionales: 5
- Controles ejecutados: 41; fallidos: 0.

## Logica de negocio

`volumen_base_tanda` alinea el baseline semanal con la duracion. Las interacciones de descuento con elasticidad y duracion representan presion promocional, pero no inversion real ni causalidad. Las transformaciones logaritmicas reducen asimetria sin eliminar los valores originales.

## Guardrails

- Targets, resultados postpromocion y variables de auditoria estan excluidos.
- La fecha original permanece en el indice; solo se exportan derivados de calendario.
- `sku_cadena` y tendencia temporal quedan en sensibilidad, no en core.
- Promedios historicos de uplift y ROI no se materializan: requeriran shift y tratamiento fold-aware.
- Encoders y escaladores deberan ajustarse dentro de cada fold temporal.
- Un futuro uplift predicho para ROI debera generarse out-of-fold.

## Compatibilidad con optimizacion

Para cada candidato deben recalcularse las features marcadas como dependientes del descuento o de la duracion en `feature_optimizer_compatibility.csv`.
