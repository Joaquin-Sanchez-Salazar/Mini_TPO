import numpy as np
import pandas as pd

from mini_tpo.constants import POST_PROMO_COLUMNS, TARGET_ROI, TARGET_UPLIFT
from mini_tpo.modeling import (
    fit_hierarchical_baseline,
    load_modeling_frame,
    predict_hierarchical_baseline,
)
from mini_tpo.data_loading import load_config
from mini_tpo.paths import PROJECT_ROOT
from mini_tpo.preprocessing import MODELING_FEATURE_SETS, build_model_pipeline


def test_hierarchical_baseline_fallback_order():
    train = pd.DataFrame(
        {
            "id_material": ["A", "A", "B"],
            "subcadena": ["X", "Y", "X"],
            "uplift_real": [1.0, 3.0, 5.0],
        }
    )
    validation = pd.DataFrame(
        {
            "id_material": ["A", "A", "C", "C"],
            "subcadena": ["X", "Z", "X", "Z"],
        }
    )
    model = fit_hierarchical_baseline(train, TARGET_UPLIFT)
    prediction = predict_hierarchical_baseline(model, validation)
    assert np.allclose(prediction, [1.0, 2.0, 3.0, 3.0])


def test_all_feature_sets_exclude_targets_and_postpromotion_columns():
    forbidden = set(POST_PROMO_COLUMNS) | {TARGET_UPLIFT, TARGET_ROI}
    for features in MODELING_FEATURE_SETS.values():
        assert forbidden.isdisjoint(features)
        assert not any(column.startswith("audit_") for column in features)


def test_preprocessing_is_unfitted_and_handles_unseen_category_after_train_fit():
    features = ["id_material", "factor_descuento"]
    model = build_model_pipeline("ridge", TARGET_UPLIFT, features, {"alpha": 1.0})
    assert not hasattr(model.named_steps["preprocessor"], "transformers_")
    train = pd.DataFrame({"id_material": ["A", "B"], "factor_descuento": [0.1, 0.2]})
    model.fit(train, np.array([0.2, 0.4]))
    prediction = model.predict(pd.DataFrame({"id_material": ["UNSEEN"], "factor_descuento": [0.3]}))
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_modeling_inputs_are_one_to_one_ordered_and_leakage_free():
    frame = load_modeling_frame(PROJECT_ROOT, load_config())
    assert frame["row_id"].is_unique
    assert frame["fecha_inicio_tanda"].is_monotonic_increasing
    for features in MODELING_FEATURE_SETS.values():
        assert set(features).issubset(frame.columns)
