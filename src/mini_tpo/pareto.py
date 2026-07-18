from __future__ import annotations

import pandas as pd


def pareto_frontier(
    frame: pd.DataFrame,
    roi_column: str = "roi_esperado",
    volume_column: str = "volumen_incremental_esperado",
) -> pd.DataFrame:
    """Return non-dominated rows, maximizing ROI and incremental volume."""
    ordered = frame.sort_values(
        [roi_column, volume_column], ascending=[False, False]
    ).copy()
    best_volume = float("-inf")
    keep = []
    for index, row in ordered.iterrows():
        volume = float(row[volume_column])
        is_nondominated = volume > best_volume
        keep.append((index, is_nondominated))
        best_volume = max(best_volume, volume)
    indices = [index for index, selected in keep if selected]
    return frame.loc[indices].sort_values(
        [volume_column, roi_column], ascending=[True, False]
    ).reset_index(drop=True)


def is_frontier_nondominated(
    frontier: pd.DataFrame,
    universe: pd.DataFrame,
    roi_column: str = "roi_esperado",
    volume_column: str = "volumen_incremental_esperado",
) -> bool:
    for row in frontier.itertuples():
        dominated = universe[
            (universe[roi_column] >= getattr(row, roi_column))
            & (universe[volume_column] >= getattr(row, volume_column))
            & (
                (universe[roi_column] > getattr(row, roi_column))
                | (universe[volume_column] > getattr(row, volume_column))
            )
        ]
        if not dominated.empty:
            return False
    return True
