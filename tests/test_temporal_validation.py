import numpy as np
import pandas as pd

from mini_tpo.temporal_validation import (
    choose_final_test_start,
    development_test_masks,
    expanding_window_splits,
)


def _frame():
    dates = pd.date_range("2024-01-01", periods=12, freq="7D").repeat(3)
    return pd.DataFrame(
        {
            "fecha_inicio_tanda": dates,
            "id_material": np.tile(["A", "B", "C"], 12),
            "subcadena": np.tile(["X", "Y", "X"], 12),
        }
    )


def test_expanding_splits_are_ordered_group_dates_and_deterministic():
    frame = _frame()
    first = expanding_window_splits(frame, n_splits=3)
    second = expanding_window_splits(frame, n_splits=3)
    assert len(first) == len(second) == 3
    for left, right in zip(first, second):
        train_dates = set(frame.iloc[left.train_indices]["fecha_inicio_tanda"])
        validation_dates = set(frame.iloc[left.validation_indices]["fecha_inicio_tanda"])
        assert max(train_dates) < min(validation_dates)
        assert train_dates.isdisjoint(validation_dates)
        assert np.array_equal(left.train_indices, right.train_indices)
        assert np.array_equal(left.validation_indices, right.validation_indices)
        for date in validation_dates:
            all_rows = set(frame.index[frame["fecha_inicio_tanda"].eq(date)])
            assert all_rows.issubset(set(left.validation_indices))


def test_final_test_is_excluded_from_development():
    frame = pd.DataFrame(
        {"fecha_inicio_tanda": pd.date_range("2024-01-01", "2024-12-01", freq="MS")}
    )
    test_start = choose_final_test_start(frame["fecha_inicio_tanda"], months=3)
    development, test = development_test_masks(frame["fecha_inicio_tanda"], test_start)
    assert test_start == pd.Timestamp("2024-10-01")
    assert frame.loc[development, "fecha_inicio_tanda"].max() < test_start
    assert frame.loc[test, "fecha_inicio_tanda"].min() >= test_start
    assert not (development & test).any()
