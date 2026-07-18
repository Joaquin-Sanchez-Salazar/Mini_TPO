# Modelos

Los binarios `*.joblib` y `*.pkl` no se versionan: dependen de las versiones de Python, NumPy y scikit-learn y pueden regenerarse.

Artifacts:

- Notebook 04 genera champion y challenger para uplift y ROI.
- Notebook 05 genera los refits production.
- `model_registry.json` sí se conserva como contrato de features, hiperparámetros, métricas y trazabilidad.

Regeneración desde la raíz, después de ejecutar las fases 01 a 03:

```powershell
uv run python -m jupyter nbconvert --to notebook --execute notebooks/04_modeling_comparison.ipynb --output 04_modeling_comparison.executed.ipynb --ExecutePreprocessor.timeout=1800
uv run python -m jupyter nbconvert --to notebook --execute notebooks/05_optimization.ipynb --output 05_optimization.executed.ipynb --ExecutePreprocessor.timeout=1800
```

No cargue modelos serializados con una versión distinta de scikit-learn a la declarada en `pyproject.toml` y `uv.lock`.
