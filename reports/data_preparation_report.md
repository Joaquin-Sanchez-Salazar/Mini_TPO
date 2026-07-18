# Data Preparation Report

- Raw: 2062 filas, 16 columnas.
- Clean full: 2062 filas, 38 columnas.
- Modeling domain: 2048 filas, 38 columnas.
- Features seguras: 2048 filas y 8 columnas incluyendo `row_id`.
- Targets: `uplift_real` y `roi`.

## Supuestos de disponibilidad

`volumen_base_sem` y `elasticidad_estimada` se aceptan como estimaciones disponibles antes de la tanda. `roi` se acepta como KPI oficial y target postpromocion; nunca se usa como predictor.

## Artifacts

Se exportaron features, targets e indice separados, junto con un feature manifest y el log de transformaciones. Las variables de auditoria permanecen fuera de las features candidatas.
