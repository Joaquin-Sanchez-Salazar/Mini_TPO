from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class TemporalFold:
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def choose_final_test_start(dates: pd.Series, months: int = 3) -> pd.Timestamp:
    parsed = pd.to_datetime(dates, errors="raise")
    latest_month = parsed.max().to_period("M")
    return (latest_month - (months - 1)).start_time


def development_test_masks(
    dates: pd.Series, test_start: pd.Timestamp
) -> tuple[pd.Series, pd.Series]:
    parsed = pd.to_datetime(dates, errors="raise")
    development = parsed < pd.Timestamp(test_start)
    test = ~development
    if not development.any() or not test.any():
        raise ValueError("Temporal split must contain both development and final test rows")
    return development, test


def expanding_window_splits(
    frame: pd.DataFrame,
    date_column: str = "fecha_inicio_tanda",
    n_splits: int = 4,
) -> list[TemporalFold]:
    dates = pd.to_datetime(frame[date_column], errors="raise")
    unique_dates = np.array(sorted(dates.unique()))
    if len(unique_dates) <= n_splits:
        raise ValueError("Not enough unique dates for requested temporal folds")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[TemporalFold] = []
    for fold_number, (train_date_idx, validation_date_idx) in enumerate(
        splitter.split(unique_dates), start=1
    ):
        train_dates = unique_dates[train_date_idx]
        validation_dates = unique_dates[validation_date_idx]
        train_rows = np.flatnonzero(dates.isin(train_dates).to_numpy())
        validation_rows = np.flatnonzero(dates.isin(validation_dates).to_numpy())
        fold = TemporalFold(
            fold=fold_number,
            train_indices=train_rows,
            validation_indices=validation_rows,
            train_start=pd.Timestamp(train_dates.min()),
            train_end=pd.Timestamp(train_dates.max()),
            validation_start=pd.Timestamp(validation_dates.min()),
            validation_end=pd.Timestamp(validation_dates.max()),
        )
        if fold.train_end >= fold.validation_start:
            raise ValueError(f"Fold {fold_number} is not temporally ordered")
        if set(train_dates).intersection(validation_dates):
            raise ValueError(f"Fold {fold_number} shares dates between train and validation")
        folds.append(fold)
    return folds


def temporal_split_summary(
    development: pd.DataFrame,
    folds: list[TemporalFold],
) -> pd.DataFrame:
    rows = []
    for fold in folds:
        train = development.iloc[fold.train_indices]
        validation = development.iloc[fold.validation_indices]
        train_skus = set(train["id_material"])
        train_chains = set(train["subcadena"])
        train_combinations = set(zip(train["id_material"], train["subcadena"]))
        validation_combinations = set(
            zip(validation["id_material"], validation["subcadena"])
        )
        rows.append(
            {
                "fold": fold.fold,
                "train_fecha_min": fold.train_start,
                "train_fecha_max": fold.train_end,
                "validation_fecha_min": fold.validation_start,
                "validation_fecha_max": fold.validation_end,
                "train_filas": len(train),
                "validation_filas": len(validation),
                "train_skus": train["id_material"].nunique(),
                "validation_skus": validation["id_material"].nunique(),
                "train_cadenas": train["subcadena"].nunique(),
                "validation_cadenas": validation["subcadena"].nunique(),
                "validation_skus_no_vistos": len(set(validation["id_material"]) - train_skus),
                "validation_cadenas_no_vistas": len(set(validation["subcadena"]) - train_chains),
                "validation_sku_cadena_no_vistas": len(
                    validation_combinations - train_combinations
                ),
            }
        )
    return pd.DataFrame(rows)


def final_test_coverage(
    development: pd.DataFrame, test: pd.DataFrame, date_column: str = "fecha_inicio_tanda"
) -> pd.DataFrame:
    development_combinations = set(
        zip(development["id_material"], development["subcadena"])
    )
    test_combinations = set(zip(test["id_material"], test["subcadena"]))
    return pd.DataFrame(
        [
            {
                "periodo": "development",
                "fecha_min": development[date_column].min(),
                "fecha_max": development[date_column].max(),
                "filas": len(development),
                "skus": development["id_material"].nunique(),
                "cadenas": development["subcadena"].nunique(),
                "sku_cadena": len(development_combinations),
                "skus_no_vistos_vs_development": 0,
                "sku_cadena_no_vistas_vs_development": 0,
            },
            {
                "periodo": "final_test",
                "fecha_min": test[date_column].min(),
                "fecha_max": test[date_column].max(),
                "filas": len(test),
                "skus": test["id_material"].nunique(),
                "cadenas": test["subcadena"].nunique(),
                "sku_cadena": len(test_combinations),
                "skus_no_vistos_vs_development": len(
                    set(test["id_material"]) - set(development["id_material"])
                ),
                "sku_cadena_no_vistas_vs_development": len(
                    test_combinations - development_combinations
                ),
            },
        ]
    )
