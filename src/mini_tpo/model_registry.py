from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def save_model(model, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path):
    return joblib.load(path)


def build_registry_entry(
    *,
    target: str,
    role: str,
    model_name: str,
    family: str,
    feature_set: str,
    features: list[str],
    hyperparameters: dict,
    train_period: tuple[str, str],
    test_period: tuple[str, str],
    cv_metrics: dict,
    final_test_metrics: dict,
    artifact_path: Path,
    data_version: str,
    random_seed: int,
) -> dict:
    return {
        "target": target,
        "role": role,
        "model": model_name,
        "family": family,
        "feature_set": feature_set,
        "features": features,
        "hyperparameters": hyperparameters,
        "train_period": {"start": train_period[0], "end": train_period[1]},
        "test_period": {"start": test_period[0], "end": test_period[1]},
        "cv_metrics": cv_metrics,
        "final_test_metrics": final_test_metrics,
        "artifact": str(artifact_path.as_posix()),
        "data_version": data_version,
        "random_seed": random_seed,
        "dependencies": {"scikit_learn": sklearn.__version__},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_registry(entries: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1.0",
        "models": entries,
        "guardrails": [
            "Final test was isolated from tuning and model selection.",
            "Targets and postpromotion variables are absent from features.",
            "A future optimizer must recalculate discount and duration dependencies.",
        ],
    }
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path
