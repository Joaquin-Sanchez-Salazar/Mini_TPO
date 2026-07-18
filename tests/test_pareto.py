import pandas as pd

from mini_tpo.pareto import is_frontier_nondominated, pareto_frontier


def test_pareto_contains_only_nondominated_scenarios():
    frame = pd.DataFrame({
        "roi_esperado": [3.0, 2.0, 2.5, 1.0],
        "volumen_incremental_esperado": [100, 300, 200, 50],
    })
    frontier = pareto_frontier(frame)
    assert len(frontier) == 3
    assert is_frontier_nondominated(frontier, frame)
