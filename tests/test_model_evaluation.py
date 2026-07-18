import numpy as np

from mini_tpo.model_evaluation import bias, mae, rmse, roi_metrics, uplift_metrics, wape


def test_regression_and_unit_metrics_formulas():
    actual = np.array([1.0, 2.0])
    predicted = np.array([2.0, 0.0])
    assert mae(actual, predicted) == 1.5
    assert np.isclose(rmse(actual, predicted), np.sqrt(2.5))
    assert np.isclose(wape(actual, predicted), 1.0)
    assert bias(actual, predicted) == -0.5
    metrics = uplift_metrics(actual, predicted, np.array([10.0, 20.0]))
    assert metrics["mae_unidades"] == 25.0
    assert metrics["error_abs_total_unidades"] == 50.0
    assert metrics["bias_unidades"] == -15.0
    assert np.isclose(metrics["wape_unidades"], 1.0)


def test_roi_sign_metrics_prioritize_destructive_false_positives():
    actual = np.array([-1.0, -0.2, 0.4, 1.0])
    predicted = np.array([0.3, -0.1, -0.2, 1.1])
    metrics = roi_metrics(actual, predicted)
    assert metrics["falsos_positivos"] == 1
    assert metrics["falsos_negativos"] == 1
    assert metrics["sign_accuracy"] == 0.5
    assert np.isclose(metrics["precision_roi_positivo"], 0.5)
    assert np.isclose(metrics["recall_roi_negativo"], 0.5)
