from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from mini_tpo.constants import RANDOM_SEED


FEATURE_SET_BASE = [
    "id_material",
    "subcadena",
    "factor_descuento",
    "duracion_dias",
    "volumen_base_sem",
    "elasticidad_estimada",
    "flag_secundario",
]

FEATURE_SET_INTERACTIONS = [
    *FEATURE_SET_BASE,
    "volumen_base_tanda",
    "elasticidad_abs",
    "descuento_x_elasticidad",
    "descuento_x_duracion",
]

FEATURE_SET_NONLINEAR = [
    *FEATURE_SET_INTERACTIONS,
    "factor_descuento_sq",
    "duracion_dias_sq",
]

FEATURE_SET_TEMPORAL_MONTHLY = [
    *FEATURE_SET_NONLINEAR,
    "mes_sin",
    "mes_cos",
]

FEATURE_SET_TEMPORAL_WEEKLY = [
    *FEATURE_SET_NONLINEAR,
    "semana_anio_sin",
    "semana_anio_cos",
]

FEATURE_SET_EXTENDED = [
    *FEATURE_SET_TEMPORAL_MONTHLY,
    "precio_base",
    "des_marca",
    "flag_secundario_missing",
    "sku_cadena",
    "dias_desde_inicio_dataset",
]

MODELING_FEATURE_SETS = {
    "base": FEATURE_SET_BASE,
    "interactions": FEATURE_SET_INTERACTIONS,
    "nonlinear": FEATURE_SET_NONLINEAR,
    "temporal_monthly": FEATURE_SET_TEMPORAL_MONTHLY,
    "temporal_weekly": FEATURE_SET_TEMPORAL_WEEKLY,
    "extended": FEATURE_SET_EXTENDED,
}

CATEGORICAL_CANDIDATES = {
    "id_material",
    "subcadena",
    "flag_secundario",
    "des_marca",
    "sku_cadena",
}


def get_feature_set(name: str) -> list[str]:
    if name not in MODELING_FEATURE_SETS:
        raise KeyError(f"Unknown modeling feature set: {name}")
    return MODELING_FEATURE_SETS[name].copy()


def split_feature_types(features: list[str]) -> tuple[list[str], list[str]]:
    categorical = [column for column in features if column in CATEGORICAL_CANDIDATES]
    numerical = [column for column in features if column not in categorical]
    return categorical, numerical


def _dense_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(family: str, features: list[str]) -> ColumnTransformer:
    categorical, numerical = split_feature_types(features)
    if family == "ridge":
        categorical_transformer = _dense_one_hot_encoder()
        numerical_transformer = StandardScaler()
    elif family in {"hist_gradient_boosting", "extra_trees"}:
        categorical_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        numerical_transformer = "passthrough"
    else:
        raise ValueError(f"Unsupported model family: {family}")
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical),
            ("numerical", numerical_transformer, numerical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_estimator(
    family: str,
    target: str,
    params: dict,
    random_seed: int = RANDOM_SEED,
):
    if family == "ridge":
        return Ridge(alpha=float(params.get("alpha", 10.0)))
    if family == "hist_gradient_boosting":
        loss = params.get(
            "loss", "absolute_error" if target == "roi" else "squared_error"
        )
        return HistGradientBoostingRegressor(
            loss=loss,
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_iter=int(params.get("max_iter", 250)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            l2_regularization=float(params.get("l2_regularization", 1.0)),
            early_stopping=False,
            random_state=random_seed,
        )
    if family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 180)),
            max_depth=params.get("max_depth", 10),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            max_features=params.get("max_features", 0.8),
            n_jobs=1,
            random_state=random_seed,
        )
    raise ValueError(f"Unsupported model family: {family}")


def build_model_pipeline(
    family: str,
    target: str,
    features: list[str],
    params: dict,
    target_transform: str = "original",
    random_seed: int = RANDOM_SEED,
):
    preprocessor = build_preprocessor(family, features)
    estimator = build_estimator(family, target, params, random_seed)
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", estimator)]
    )
    if target_transform == "log1p":
        if target != "uplift_real":
            raise ValueError("log1p target transform is only allowed for uplift_real")
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    if target_transform != "original":
        raise ValueError(f"Unsupported target transform: {target_transform}")
    return pipeline


def unwrap_fitted_pipeline(model):
    if isinstance(model, TransformedTargetRegressor):
        return model.regressor_
    return model
