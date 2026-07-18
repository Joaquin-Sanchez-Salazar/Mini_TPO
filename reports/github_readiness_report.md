# GitHub readiness report

Fecha de revision: 2026-07-18.

## Alcance

Se preparo el proyecto para una futura publicacion sin inicializar Git, crear commits, configurar remotos ni transferir archivos. La revision fue local y no envio contenido a servicios externos.

## Cambios realizados

Modificados:

- `.gitignore`: cobertura de Python, entornos, Jupyter, IDE, secretos, temporales, datos privados, modelos binarios y exports regenerables.
- `README.md`: entorno bloqueado con `uv.lock`, politica de datos, regeneracion de artifacts y estado de uso.
- `pyproject.toml`: configuracion reproducible del paquete `src/mini_tpo` y scikit-learn 1.9.0, alineado con el registry.
- `uv.lock`: lock reproducible del entorno.
- `notebooks/01_eda.ipynb` a `notebooks/05_optimization.ipynb`: outputs y contadores de ejecucion eliminados; codigo y narrativa conservados.

Creados:

- `NOTICE.md`.
- `data/raw/README.md`.
- `data/interim/README.md`.
- `data/processed/README.md`.
- `models/README.md`.
- `reports/github_readiness_report.md`.

Eliminados por ser copias regenerables o duplicados exactos:

- cinco notebooks `*.executed.ipynb`;
- `base (3).csv`, duplicado exacto del CSV normalizado de `data/raw/`.

El entorno rechazo la eliminacion del PDF binario duplicado y de caches mediante shell; permanecen cubiertos por `.gitignore`.

## Politica de publicacion

Excluidos por `.gitignore`:

- `data/raw/`: CSV empresarial y PDF de la prueba.
- `data/interim/` y `data/processed/`: datos derivados a nivel fila y artifacts regenerables.
- `models/*.joblib` y `models/*.pkl`: binarios dependientes del entorno.
- `notebooks/*.executed.ipynb` y checkpoints.
- `reports/*.html`: exports regenerables.
- `reports/tables/top_10_roi.csv` y `bottom_10_roi.csv`: extractos a nivel fila.
- `.venv/`, caches, logs, temporales, archivos de IDE y formatos habituales de credenciales.
- PDF y duplicados originales ubicados en la raiz.

Recomendados para conservar:

- codigo de `src/mini_tpo/`, notebooks fuente limpios, tests y configuracion;
- `pyproject.toml`, `uv.lock`, README, AGENTS y NOTICE;
- `models/model_registry.json` como contrato liviano;
- reportes Markdown, tablas agregadas y figuras necesarias para sustentar resultados, sujetos a autorizacion de publicacion.

Los README de `data/` y `models/` indican entradas, dependencias y comandos de regeneracion. Los HTML se reconstruyen desde los notebooks ejecutados con los comandos documentados en el README principal.

## Seguridad y privacidad

La busqueda en archivos candidatos a publicacion no encontro:

- rutas personales de Windows, macOS o Linux;
- asignaciones evidentes de API keys, tokens, passwords o client secrets;
- bloques de claves privadas;
- direcciones de correo.

No se encontro un secreto que requiera rotacion. Aun asi, los datos raw, el PDF y los resultados derivados requieren confirmacion explicita del titular antes de una publicacion abierta. La ausencia de un secreto detectable no equivale a autorizacion para compartir informacion empresarial.

Se confirmaron por SHA-256 dos duplicados exactos en la raiz. Se elimino el CSV duplicado; `Prueba Tecnica_DS (2).pdf` permanece local e ignorado. Las copias normalizadas se conservan en `data/raw/`.

## Archivos grandes

No hay archivos mayores de 10 MB fuera de `.venv/`, ni archivos mayores de 100 MB en el proyecto inspeccionado.

Los unicos archivos superiores a 10 MB son dependencias binarias dentro del entorno ignorado:

| Ruta | Tamano aprox. | Tipo | Decision |
|---|---:|---|---|
| `.venv/Lib/site-packages/pyarrow/arrow.dll` | 20.97 MB | DLL | Ignorar; se reinstala con `uv sync` |
| `.venv/Lib/site-packages/numpy.libs/libscipy_openblas*.dll` | 19.46 MB | DLL | Ignorar; se reinstala con `uv sync` |
| `.venv/Lib/site-packages/scipy.libs/libscipy_openblas*.dll` | 19.32 MB | DLL | Ignorar; se reinstala con `uv sync` |
| `.venv/Lib/site-packages/pyarrow/arrow_flight.dll` | 13.98 MB | DLL | Ignorar; se reinstala con `uv sync` |

No se recomienda Git LFS para el inventario publicable actual.

## Validaciones ejecutadas

Comandos ejecutados desde la raiz:

```text
uv lock
uv sync
uv run python -c "import mini_tpo, sklearn"
uv run pytest -q
uv run jupyter nbconvert --clear-output --inplace <notebook>
```

Resultados:

- importacion de `mini_tpo`: correcta;
- scikit-learn: 1.9.0, consistente con `models/model_registry.json`;
- tests: 40 aprobados, 0 fallos;
- warnings: 1,846 advertencias deprecadas de Joblib/NumPy al recargar arrays serializados;
- notebooks fuente: 0 outputs y 0 contadores de ejecucion en los cinco archivos;
- enlaces relativos documentados y archivos principales del README: presentes;
- directorio `.git`: ausente al finalizar la revision.

No se ejecuto nuevamente el pipeline completo porque no era necesario para validar packaging, importacion y tests. Los resultados analiticos no fueron recalculados ni modificados.

## Pendientes manuales

1. Confirmar autorizacion para publicar tablas, figuras, reportes Markdown y `model_registry.json` derivados de datos empresariales.
2. Decidir una licencia solo con autorizacion del propietario; `NOTICE.md` no concede reutilizacion abierta.
3. Eliminar, si se desea, el PDF duplicado raiz y los caches locales ya ignorados.
4. Sustituir los placeholders de clon del README cuando exista una URL real.
5. Revisar el inventario publicable antes de inicializar y publicar el repositorio manualmente.

No se inicializo Git, no se creo ningun commit, no se configuro ningun remoto y no se subio contenido.
