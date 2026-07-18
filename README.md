# Mini Trade Promo Optimizer - Alicorp

Mini-TPO reproducible para analizar promociones temporales de pasta dental en tres cadenas del canal moderno. El proyecto estima **uplift** y **ROI**, compara modelos con validación temporal y simula combinaciones de:

- descuento entre **5% y 40%**;
- duración entre **5 y 21 días**.

El resultado final incluye curvas de demanda, incertidumbre, soporte histórico y recomendaciones de descuento/duración para tres SKU. Los resultados son predictivos y descriptivos: no demuestran causalidad ni garantizan el retorno de una promoción futura.

## Objetivo

Apoyar decisiones de Trade Marketing mediante:

- diagnóstico de calidad, cobertura y leakage;
- preparación de datos y features prepromoción;
- estimación de uplift y ROI;
- comparación de modelos con ventanas temporales expansivas;
- simulación de demanda, volumen incremental y ROI;
- optimización con incertidumbre, soporte local y guardrails.

## Estructura del repositorio

```text
.
├── configs/                 # Rutas, reglas de negocio y parámetros
├── data/
│   ├── raw/                 # Entradas inmutables
│   ├── interim/             # Dataset limpio completo
│   └── processed/           # Features, targets, predicciones y escenarios
├── models/                  # Champion, challenger, production y registry
├── notebooks/               # Fases 01 a 05
├── reports/
│   ├── figures/             # EDA, modelado y optimización
│   └── tables/              # Calidad, métricas, soporte y recomendaciones
├── src/mini_tpo/            # Lógica reutilizable del pipeline
├── tests/                   # Pruebas de datos, modelado y optimización
├── AGENTS.md                # Decisiones y guardrails técnicos
├── NOTICE.md                # Estado de uso y publicación
├── README.md
├── pyproject.toml
└── uv.lock
```

## Requisitos

- Python **3.11 o superior**, según [`pyproject.toml`](pyproject.toml).
- Entorno recomendado: [`uv`](https://docs.astral.sh/uv/).
- Dependencias principales: pandas, NumPy, matplotlib, scikit-learn, SciPy, PyYAML y PyArrow.
- Jupyter y nbconvert para ejecutar y exportar notebooks.
- pytest para validación automatizada.

El pipeline utiliza CPU y trabaja con 2,048 observaciones; no requiere GPU. `uv.lock` fija el entorno reproducible y `pyproject.toml` alinea scikit-learn 1.9.0 con los artifacts registrados.

## Instalación

Desde un clon limpio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

### Opción recomendada: uv

```bash
uv sync
```

Ejecute luego los comandos con `uv run`, por ejemplo `uv run pytest -q` o `uv run python -m jupyter ...`.

### Alternativa: venv y pip

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Verifique el entorno antes de ejecutar modelos serializados:

```bash
python --version
python -c "import sklearn; print(sklearn.__version__)"
```

## Datos de entrada

El pipeline usa estas entradas locales:

| Archivo | Ubicación | Uso |
|---|---|---|
| Histórico promocional CSV | `data/raw/base_mini_tpo.csv` | Entrada obligatoria del pipeline analítico |
| Prueba técnica PDF | `data/raw/prueba_tecnica_ds.pdf` | Contexto del caso; no es leído por el pipeline |

El CSV contiene fecha, SKU, marca, cadena, precio, descuento, duración, volumen base/promocional, ventas, inversión, uplift, elasticidad, flag secundario y ROI. El esquema exacto y las rutas están en [`configs/project_config.yaml`](configs/project_config.yaml).

El proyecto no descarga datos. `data/raw/` está excluido por `.gitignore` porque no existe autorización explícita para publicar datos empresariales ni el PDF de la prueba. En un clon limpio, coloque el CSV con el nombre exacto indicado; consulte [`data/raw/README.md`](data/raw/README.md). Los archivos raw nunca se sobrescriben.

# Reproducir el proyecto completo desde cero

No existe una función única que ejecute las cinco fases y genere también sus HTML. Los notebooks deben ejecutarse **en orden**, dentro del mismo entorno, porque cada fase consume artifacts de la anterior.

## Resumen de fases

| Fase | Notebook | Propósito | Outputs principales |
|---|---|---|---|
| EDA | `01_eda.ipynb` | Calidad, cobertura, uplift, ROI y soporte | Tablas, figuras y reporte EDA |
| Preparación | `02_data_cleaning_preparation.ipynb` | Limpieza, auditoría y contrato seguro | Parquet, manifests y log de limpieza |
| Features | `03_feature_engineering.ipynb` | Features prepromoción y compatibilidad | Core, extended y catálogo |
| Modelado | `04_modeling_comparison.ipynb` | CV temporal, baselines y selección | OOF, test, modelos y métricas |
| Optimización | `05_optimization.ipynb` | Curvas, guardrails y recomendaciones | Escenarios, óptimos, Pareto y production |

## 1. EDA

Analiza calidad, cobertura temporal, distribuciones de uplift/ROI, fuga de información y soporte por SKU × cadena. También ejecuta `run_preparation()` para disponer de los artifacts base.

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.executed.ipynb --ExecutePreprocessor.timeout=900
```

Inputs principales:

- `data/raw/base_mini_tpo.csv`;
- `configs/project_config.yaml`.

Outputs principales:

- `reports/eda_data_quality_report.md`;
- `reports/tables/eda_validation_summary.csv`;
- `reports/tables/eda_business_driven_summary.csv`;
- `reports/tables/support_sku_chain.csv`;
- figuras generales en `reports/figures/`.

## 2. Limpieza y preparación

Convierte tipos, audita faltantes y consistencia, crea flags de dominio/outliers y separa features, targets e índice. El notebook vuelve a ejecutar `run_preparation()` de forma determinista y agrega la comparación de datasets.

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/02_data_cleaning_preparation.ipynb --output 02_data_cleaning_preparation.executed.ipynb --ExecutePreprocessor.timeout=900
```

Outputs principales:

- `data/interim/base_mini_tpo_clean_full.parquet`;
- `data/processed/base_mini_tpo_modeling.parquet`;
- `data/processed/model_features_safe.parquet`;
- `data/processed/model_targets.parquet`;
- `data/processed/model_index.parquet`;
- `data/processed/feature_manifest.json`;
- `reports/tables/data_cleaning_log.csv`;
- `reports/data_preparation_report.md`.

## 3. Feature engineering

Construye features prepromoción de calendario, volumen de tanda, elasticidad, interacciones y términos no lineales. No ajusta modelos ni transformadores globales.

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/03_feature_engineering.ipynb --output 03_feature_engineering.executed.ipynb --ExecutePreprocessor.timeout=900
```

Inputs principales:

- `model_features_safe.parquet`;
- `model_targets.parquet`;
- `model_index.parquet`.

Outputs principales:

- `data/processed/model_features_engineered_core.parquet`;
- `data/processed/model_features_engineered_optional.parquet`;
- `data/processed/feature_engineering_manifest.json`;
- `reports/tables/feature_engineering_catalog.csv`;
- `reports/tables/feature_optimizer_compatibility.csv`;
- `reports/feature_engineering_report.md`;
- figuras en `reports/figures/feature_engineering/`.

## 4. Modelado y comparación

Reserva el trimestre final como test aislado, crea cuatro ventanas expansivas, evalúa baselines y compara Ridge, HistGradientBoosting y ExtraTrees. El preprocesamiento se ajusta dentro de cada fold.

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/04_modeling_comparison.ipynb --output 04_modeling_comparison.executed.ipynb --ExecutePreprocessor.timeout=1800
```

Outputs principales:

- `data/processed/oof_predictions_uplift.parquet` y `oof_predictions_roi.parquet`;
- `data/processed/final_test_predictions_uplift.parquet` y `final_test_predictions_roi.parquet`;
- cuatro modelos evaluation en `models/`;
- `models/model_registry.json`;
- `reports/tables/model_selection_scorecard.csv`;
- `reports/tables/final_test_metrics_uplift.csv` y `final_test_metrics_roi.csv`;
- `reports/modeling_report.md`;
- curvas en `reports/figures/modeling/response_curves/`.

## 5. Optimización

Reajusta versiones production de los champions sobre las 2,048 filas, selecciona tres casos y evalúa 540 escenarios de descuento × duración. Calcula demanda, volumen incremental, ROI, incertidumbre OOF, soporte local, óptimos y Pareto.

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/05_optimization.ipynb --output 05_optimization.executed.ipynb --ExecutePreprocessor.timeout=1800
```

Outputs principales:

- `data/processed/optimization_scenarios.parquet`;
- `models/uplift_production.joblib` y `models/roi_production.joblib`;
- `reports/tables/optimal_recommendations.csv`;
- `reports/tables/mathematical_roi_optima.csv`;
- `reports/tables/profitable_growth_alternatives.csv`;
- `reports/tables/optimization_pareto_frontier.csv`;
- `reports/optimization_report.md`;
- 15 figuras en `reports/figures/optimization/`.

## Ejecución secuencial

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.executed.ipynb --ExecutePreprocessor.timeout=900
python -m jupyter nbconvert --to notebook --execute notebooks/02_data_cleaning_preparation.ipynb --output 02_data_cleaning_preparation.executed.ipynb --ExecutePreprocessor.timeout=900
python -m jupyter nbconvert --to notebook --execute notebooks/03_feature_engineering.ipynb --output 03_feature_engineering.executed.ipynb --ExecutePreprocessor.timeout=900
python -m jupyter nbconvert --to notebook --execute notebooks/04_modeling_comparison.ipynb --output 04_modeling_comparison.executed.ipynb --ExecutePreprocessor.timeout=1800
python -m jupyter nbconvert --to notebook --execute notebooks/05_optimization.ipynb --output 05_optimization.executed.ipynb --ExecutePreprocessor.timeout=1800
```

Los notebooks ejecutados se escriben dentro de `notebooks/`. Para generar los HTML:

```bash
python -m jupyter nbconvert --to html notebooks/01_eda.executed.ipynb --output-dir reports --output 01_eda.html
python -m jupyter nbconvert --to html notebooks/02_data_cleaning_preparation.executed.ipynb --output-dir reports --output 02_data_cleaning_preparation.html
python -m jupyter nbconvert --to html notebooks/03_feature_engineering.executed.ipynb --output-dir reports --output 03_feature_engineering.html
python -m jupyter nbconvert --to html notebooks/04_modeling_comparison.executed.ipynb --output-dir reports --output 04_modeling_comparison.html
python -m jupyter nbconvert --to html notebooks/05_optimization.executed.ipynb --output-dir reports --output 05_optimization.html
```

Las funciones públicas `run_preparation()`, `run_feature_engineering()`, `run_modeling_comparison()` y `run_optimization()` están en `mini_tpo.pipeline`, pero no reemplazan las visualizaciones y HTML generados por los notebooks.

## Modelado y validación temporal

El split no es aleatorio porque el objetivo es predecir promociones futuras. El desarrollo cubre julio de 2024 a septiembre de 2025; octubre-diciembre de 2025 queda aislado como test final. Las cuatro ventanas expansivas entrenan siempre con fechas anteriores a la validación y mantienen todas las promociones de una fecha en el mismo fold.

Según [`models/model_registry.json`](models/model_registry.json):

| Target / rol | Familia | Feature set | Parámetros principales |
|---|---|---|---|
| Uplift champion | Ridge | `temporal_weekly` | `alpha=100` |
| Uplift challenger | Ridge + `log1p` | `temporal_weekly` | `alpha=10` |
| ROI champion | HistGradientBoosting | `nonlinear` | LR 0.05, 220 iter., 15 hojas, L2 1 |
| ROI challenger | HistGradientBoosting | `nonlinear` | LR 0.04, 300 iter., 9 hojas, L2 3 |

- **Champion:** modelo seleccionado con CV y scorecard antes de abrir el test.
- **Challenger:** alternativa conservada para comparación y sensibilidad.
- **Production refit:** misma selección e hiperparámetros, reajustada sobre las 2,048 filas después de cerrar la evaluación.

Métricas detalladas: [`reports/modeling_report.md`](reports/modeling_report.md), [`model_selection_scorecard.csv`](reports/tables/model_selection_scorecard.csv) y tablas de test final.

## Optimización

Casos seleccionados:

- `PD008 × Cadena01`: soporte alto;
- `PD013 × Cadena02`: alto volumen base;
- `PD015 × Cadena03`: SKU reciente y soporte bajo.

La grilla principal usa descuentos de 5%-40% cada 1 pp y duraciones `5, 7, 10, 14, 21`. Cada escenario calcula uplift, demanda promocional, volumen incremental y ROI. Los intervalos usan residuos OOF; el soporte local exige misma duración y descuento histórico dentro de ±2.5 pp.

- **Óptimo matemático:** maximiza ROI puntual, aunque puede extrapolar.
- **Recomendación robusta:** exige `ROI lower 90 > 0`, soporte y guardrails.
- **Crecimiento rentable:** maximiza unidades incrementales entre escenarios robustos.
- **Pareto:** muestra escenarios no dominados en ROI y volumen incremental.

Recomendaciones verificadas en [`optimal_recommendations.csv`](reports/tables/optimal_recommendations.csv):

| SKU | Cadena | Descuento | Duración | Uplift esperado | Volumen incremental | ROI esperado | ROI lower 90 | Soporte | Decisión |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| PD008 | Cadena01 | 9% | 14 días | 0.250 | 1,426 | 2.987 | 2.961 | Alto | Robusta |
| PD013 | Cadena02 | 10% | 14 días | 0.177 | 1,358 | 2.526 | 2.500 | Medio | Robusta |
| PD015 | Cadena03 | 12% | 21 días | 0.120 | 1,569 | 1.749 | 1.723 | Bajo | Revisión humana |

Estas recomendaciones no incorporan stock, presupuesto, margen o canibalización y deben reevaluarse cuando cambien baseline, elasticidad o soporte.

## Artifacts principales

Los Parquet de `data/interim/` y `data/processed/`, los modelos binarios, los notebooks ejecutados y los HTML son reproducibles y se mantienen fuera de Git. Sus carpetas conservan instrucciones; las tablas, figuras y reportes Markdown resumidos permanecen disponibles para revisión. Antes de una publicación abierta, confirme que esos resultados derivados también estén autorizados.

### `data/processed/`

- datasets de modelado y contratos separados por `row_id`;
- features core/extended;
- predicciones OOF y test final;
- `optimization_scenarios.parquet`.

### `models/`

- `uplift_champion.joblib` y `uplift_challenger.joblib`;
- `roi_champion.joblib` y `roi_challenger.joblib`;
- `uplift_production.joblib` y `roi_production.joblib`;
- `model_registry.json` con features, parámetros, periodos, métricas y versión.

Los archivos `*.joblib` se regeneran con los notebooks 04 y 05; vea [`models/README.md`](models/README.md). `model_registry.json` sí se conserva como contrato liviano.

### `reports/`

- `tables/`: calidad, soporte, métricas, selección y recomendaciones;
- `figures/`: EDA, features, curvas de respuesta y optimización;
- `01_eda.html` a `05_optimization.html`: reportes ejecutivos navegables;
- reportes Markdown por fase.

Los HTML se regeneran con los comandos de exportación anteriores y no se versionan.

## Tests y validación

Con el entorno activado:

```bash
python -m pytest -q
```

Con uv:

```bash
uv run pytest -q
```

Los tests cubren fórmulas, alineamiento por `row_id`, leakage, splits temporales, métricas, fallbacks, artifacts de modelos, grillas, incertidumbre, guardrails, `NO_RECOMMENDATION` y frontera de Pareto.

Lista de comprobación final:

- [ ] Los cinco notebooks ejecutaron en orden y sin celdas de error.
- [ ] Existen los Parquet de preparación, features, OOF, test y optimización.
- [ ] Existen cuatro modelos evaluation y dos production.
- [ ] Existen `reports/01_eda.html` a `reports/05_optimization.html`.
- [ ] Existen las tablas de recomendaciones, óptimos, crecimiento y Pareto.
- [ ] `python -m pytest -q` termina sin fallos en el mismo entorno usado para entrenar.

## Reproducibilidad

- Semilla global: `42`.
- Configuración central: [`configs/project_config.yaml`](configs/project_config.yaml).
- Todas las rutas de código son relativas a la raíz.
- Los notebooks delegan lógica en `src/mini_tpo/`.
- Encoders y escaladores se ajustan dentro de cada fold.
- El test final no participa en tuning o selección.
- `data/raw/` no se modifica.
- No edite manualmente manifests, Parquet, predicciones o registry; regenérelos ejecutando las fases en orden.
- Los outputs pesados o dependientes del entorno se excluyen mediante `.gitignore`; sus README locales documentan cómo regenerarlos.

## Resultados principales

- En OOF, Ridge champion de uplift obtuvo MAE `0.140` frente a `0.191` del baseline jerárquico; el challenger tuvo menor MAE, pero el champion fue elegido por scorecard de estabilidad, suavidad e interpretabilidad.
- El champion ROI obtuvo MAE OOF `0.115` frente a `0.266` del baseline por banda de descuento.
- En test final, uplift champion obtuvo MAE `0.148`; ROI champion obtuvo MAE `0.077`, accuracy de signo `99.44%` y cero falsos positivos.
- Se optimizaron PD008, PD013 y PD015. El descuento domina el ROI y los óptimos puntuales de bajo descuento suelen carecer de soporte local; por eso la recomendación robusta difiere del máximo matemático.

No deben mezclarse métricas OOF, test final y predicciones production: cumplen funciones diferentes.

## Limitaciones

- Modelos predictivos sobre datos observacionales; no estiman causalidad.
- Histórico pequeño y soporte desigual por SKU × cadena.
- Intervalos amplios de uplift; los tres intervalos inferiores de las recomendaciones cruzan cero.
- PD015 es reciente y requiere revisión humana.
- El ROI está fuertemente condicionado por descuento.
- No existen restricciones reales de stock, presupuesto, margen o inversión futura.
- Los modelos requieren monitoreo de drift, cobertura y error tras despliegue.

## Solución de problemas

| Problema | Acción |
|---|---|
| Falta el raw | Verifique `data/raw/base_mini_tpo.csv` y `data/raw/prueba_tecnica_ds.pdf`. |
| `ModuleNotFoundError: mini_tpo` | Active el entorno y ejecute `python -m pip install -e .`. |
| Jupyter/nbconvert no existe | Ejecute `python -m pip install -e .` dentro del entorno correcto. |
| Falta un artifact intermedio | Ejecute nuevamente los notebooks desde la primera fase faltante. |
| Error al cargar `.joblib` | Use la versión registrada de scikit-learn o regenere 04 y 05 en el mismo entorno. |
| El notebook supera el timeout | Aumente `--ExecutePreprocessor.timeout`; no interrumpa 04 o 05 mientras entrenan/escriben modelos. |
| Tests fallan por versiones | Confirme Python ≥3.11 y ejecute 04, 05 y pytest en el mismo entorno. |

## Estado de uso

Este repositorio corresponde a una prueba técnica y no declara una licencia de código abierto. Consulte [`NOTICE.md`](NOTICE.md) y confirme los permisos sobre código, datos, documentos y resultados antes de publicarlo o reutilizarlo.
