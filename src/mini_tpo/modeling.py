from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from mini_tpo.constants import RANDOM_SEED, TARGET_ROI, TARGET_UPLIFT
from mini_tpo.model_evaluation import (
    compare_models_by_date,
    evaluate_prediction_frame,
    metrics_by_group,
    uncertainty_summary,
)
from mini_tpo.model_registry import (
    build_registry_entry,
    load_model,
    save_model,
    write_registry,
)
from mini_tpo.preprocessing import (
    MODELING_FEATURE_SETS,
    build_model_pipeline,
    get_feature_set,
    unwrap_fitted_pipeline,
)
from mini_tpo.temporal_validation import (
    choose_final_test_start,
    development_test_masks,
    expanding_window_splits,
    final_test_coverage,
    temporal_split_summary,
)


@dataclass
class CandidateResult:
    candidate: dict
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: dict


def load_modeling_frame(project_root: Path, config: dict) -> pd.DataFrame:
    project = config["project"]
    features = pd.read_parquet(project_root / project["engineered_optional"])
    targets = pd.read_parquet(project_root / project["safe_targets"])
    index = pd.read_parquet(project_root / project["safe_index"])
    if not features["row_id"].is_unique or not targets["row_id"].is_unique or not index["row_id"].is_unique:
        raise ValueError("row_id must be unique in modeling inputs")
    forbidden = {TARGET_UPLIFT, TARGET_ROI, "volumen_promo", "venta_promo", "inversion_promo"}
    if forbidden.intersection(features.columns) or any(
        column.startswith("audit_") for column in features.columns
    ):
        raise ValueError("Modeling features contain leakage")
    frame = features.merge(targets, on="row_id", validate="one_to_one", sort=False)
    frame = frame.merge(
        index[["row_id", "fecha_inicio_tanda"]],
        on="row_id",
        validate="one_to_one",
        sort=False,
    )
    frame["fecha_inicio_tanda"] = pd.to_datetime(frame["fecha_inicio_tanda"], errors="raise")
    frame["banda_descuento"] = pd.cut(
        frame["factor_descuento"],
        bins=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4000001],
        labels=["05-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40"],
        include_lowest=True,
        right=False,
    ).astype("string")
    support_path = project_root / "reports" / "tables" / "support_sku_chain.csv"
    maturity_path = project_root / "reports" / "tables" / "sku_history_profile.csv"
    if support_path.exists():
        support = pd.read_csv(support_path)[
            ["id_material", "subcadena", "clasificacion_soporte"]
        ]
        frame = frame.merge(
            support, on=["id_material", "subcadena"], how="left", validate="many_to_one"
        )
    else:
        frame["clasificacion_soporte"] = "no_disponible"
    if maturity_path.exists():
        maturity = pd.read_csv(maturity_path)[["id_material", "clasificacion_sku"]]
        frame = frame.merge(maturity, on="id_material", how="left", validate="many_to_one")
    else:
        frame["clasificacion_sku"] = "no_disponible"
    frame["clasificacion_soporte"] = frame["clasificacion_soporte"].fillna("no_disponible")
    frame["clasificacion_sku"] = frame["clasificacion_sku"].fillna("no_disponible")
    return frame.sort_values(["fecha_inicio_tanda", "row_id"]).reset_index(drop=True)


def data_version_hash(project_root: Path, config: dict) -> str:
    digest = hashlib.sha256()
    for key in ["engineered_core", "engineered_optional", "safe_targets", "safe_index"]:
        with (project_root / config["project"][key]).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def fit_hierarchical_baseline(train: pd.DataFrame, target: str) -> dict:
    return {
        "combo": train.groupby(["id_material", "subcadena"], observed=True)[target].median(),
        "sku": train.groupby("id_material", observed=True)[target].median(),
        "chain": train.groupby("subcadena", observed=True)[target].median(),
        "global": float(train[target].median()),
    }


def predict_hierarchical_baseline(model: dict, validation: pd.DataFrame) -> np.ndarray:
    predictions = []
    for row in validation.itertuples():
        combo = (row.id_material, row.subcadena)
        if combo in model["combo"].index:
            value = model["combo"].loc[combo]
        elif row.id_material in model["sku"].index:
            value = model["sku"].loc[row.id_material]
        elif row.subcadena in model["chain"].index:
            value = model["chain"].loc[row.subcadena]
        else:
            value = model["global"]
        predictions.append(value)
    return np.asarray(predictions, dtype=float)


def fit_discount_baseline(train: pd.DataFrame, target: str = TARGET_ROI) -> dict:
    return {
        "band": train.groupby("banda_descuento", observed=True)[target].median(),
        "global": float(train[target].median()),
    }


def predict_discount_baseline(model: dict, validation: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [model["band"].get(value, model["global"]) for value in validation["banda_descuento"]],
        dtype=float,
    )


def _prediction_frame(
    validation: pd.DataFrame,
    target: str,
    prediction: np.ndarray,
    fold: int,
    candidate: dict,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "row_id": validation["row_id"].to_numpy(),
            "fecha_inicio_tanda": validation["fecha_inicio_tanda"].to_numpy(),
            "fecha": validation["fecha_inicio_tanda"].to_numpy(),
            "target_real": validation[target].to_numpy(),
            "prediccion": np.asarray(prediction),
            "fold": fold,
            "modelo": candidate["model_name"],
            "candidate_id": candidate["candidate_id"],
            "familia": candidate["family"],
            "feature_set": candidate["feature_set"],
            "id_material": validation["id_material"].astype("string").to_numpy(),
            "subcadena": validation["subcadena"].astype("string").to_numpy(),
            "factor_descuento": validation["factor_descuento"].to_numpy(),
            "duracion_dias": validation["duracion_dias"].to_numpy(),
            "banda_descuento": validation["banda_descuento"].astype("string").to_numpy(),
            "soporte": validation["clasificacion_soporte"].astype("string").to_numpy(),
            "madurez_sku": validation["clasificacion_sku"].astype("string").to_numpy(),
            "flag_uplift_en_piso": validation["flag_uplift_en_piso"].to_numpy(),
            "flag_roi_extremo_descriptivo": validation["flag_roi_extremo_descriptivo"].to_numpy(),
            "volumen_base_tanda": validation["volumen_base_tanda"].to_numpy(),
        }
    )
    result["residual"] = result["prediccion"] - result["target_real"]
    return result


def evaluate_candidate(
    development: pd.DataFrame,
    folds,
    target: str,
    candidate: dict,
    random_seed: int,
) -> CandidateResult:
    feature_columns = get_feature_set(candidate["feature_set"])
    fold_rows = []
    prediction_frames = []
    for fold in folds:
        train = development.iloc[fold.train_indices]
        validation = development.iloc[fold.validation_indices]
        model = build_model_pipeline(
            candidate["family"],
            target,
            feature_columns,
            candidate["params"],
            candidate.get("target_transform", "original"),
            random_seed,
        )
        started = time.perf_counter()
        model.fit(train[feature_columns], train[target])
        prediction = model.predict(validation[feature_columns])
        elapsed = time.perf_counter() - started
        frame = _prediction_frame(validation, target, prediction, fold.fold, candidate)
        metrics = evaluate_prediction_frame(frame, target)
        metrics.update(
            {
                "target": target,
                "fold": fold.fold,
                "candidate_id": candidate["candidate_id"],
                "modelo": candidate["model_name"],
                "familia": candidate["family"],
                "feature_set": candidate["feature_set"],
                "target_transform": candidate.get("target_transform", "original"),
                "tiempo_entrenamiento_seg": elapsed,
            }
        )
        fold_rows.append(metrics)
        prediction_frames.append(frame)
    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    overall = evaluate_prediction_frame(predictions, target)
    summary = {
        **overall,
        "target": target,
        "candidate_id": candidate["candidate_id"],
        "modelo": candidate["model_name"],
        "familia": candidate["family"],
        "feature_set": candidate["feature_set"],
        "target_transform": candidate.get("target_transform", "original"),
        "mae_fold_std": float(fold_metrics["mae"].std(ddof=0)),
        "tiempo_entrenamiento_seg": float(fold_metrics["tiempo_entrenamiento_seg"].sum()),
        "params": json.dumps(candidate["params"], sort_keys=True),
    }
    return CandidateResult(candidate, fold_metrics, predictions, summary)


def evaluate_baselines(development: pd.DataFrame, folds) -> pd.DataFrame:
    rows = []
    for target in [TARGET_UPLIFT, TARGET_ROI]:
        for fold in folds:
            train = development.iloc[fold.train_indices]
            validation = development.iloc[fold.validation_indices]
            candidates = {
                "global_median": np.repeat(float(train[target].median()), len(validation)),
                "hierarchical_sku_chain": predict_hierarchical_baseline(
                    fit_hierarchical_baseline(train, target), validation
                ),
            }
            if target == TARGET_ROI:
                candidates["roi_discount_band"] = predict_discount_baseline(
                    fit_discount_baseline(train), validation
                )
            for name, prediction in candidates.items():
                candidate = {
                    "model_name": name,
                    "candidate_id": f"baseline__{target}__{name}",
                    "family": "baseline",
                    "feature_set": "historical_train_only",
                }
                frame = _prediction_frame(validation, target, prediction, fold.fold, candidate)
                metrics = evaluate_prediction_frame(frame, target)
                metrics.update({"target": target, "fold": fold.fold, "baseline": name})
                rows.append(metrics)
    return pd.DataFrame(rows)


def _candidate(
    target: str,
    family: str,
    feature_set: str,
    params: dict,
    target_transform: str = "original",
) -> dict:
    param_token = "_".join(f"{key}{value}" for key, value in sorted(params.items()))
    candidate_id = f"{target}__{family}__{feature_set}__{target_transform}__{param_token}"
    return {
        "target": target,
        "family": family,
        "feature_set": feature_set,
        "params": params,
        "target_transform": target_transform,
        "candidate_id": candidate_id,
        "model_name": family,
    }


def ablation_candidates(target: str) -> list[dict]:
    return [
        _candidate(target, "ridge", feature_set, {"alpha": 10.0})
        for feature_set in MODELING_FEATURE_SETS
    ]


def tuning_candidates(target: str, feature_set: str) -> list[dict]:
    candidates = [
        _candidate(target, "ridge", feature_set, {"alpha": alpha})
        for alpha in [1.0, 10.0, 100.0]
    ]
    if target == TARGET_UPLIFT:
        candidates.append(
            _candidate(target, "ridge", feature_set, {"alpha": 10.0}, "log1p")
        )
    hist_configs = [
        {
            "learning_rate": 0.05,
            "max_iter": 220,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.04,
            "max_iter": 300,
            "max_leaf_nodes": 9,
            "min_samples_leaf": 30,
            "l2_regularization": 3.0,
        },
    ]
    for params in hist_configs:
        candidates.append(_candidate(target, "hist_gradient_boosting", feature_set, params))
    if target == TARGET_UPLIFT:
        candidates.append(
            _candidate(
                target,
                "hist_gradient_boosting",
                feature_set,
                hist_configs[0],
                "log1p",
            )
        )
    extra_configs = [
        {"n_estimators": 180, "max_depth": 10, "min_samples_leaf": 5, "max_features": 0.8},
        {"n_estimators": 220, "max_depth": 14, "min_samples_leaf": 8, "max_features": 1.0},
    ]
    candidates.extend(
        _candidate(target, "extra_trees", feature_set, params)
        for params in extra_configs
    )
    return candidates


def _recalculate_scenario(row: pd.Series, discounts=None, durations=None) -> pd.DataFrame:
    values = discounts if discounts is not None else durations
    scenario = pd.DataFrame([row.to_dict() for _ in values])
    if discounts is not None:
        scenario["factor_descuento"] = np.asarray(discounts, dtype=float)
    if durations is not None:
        scenario["duracion_dias"] = np.asarray(durations, dtype=int)
    scenario["duracion_semanas"] = scenario["duracion_dias"] / 7.0
    scenario["volumen_base_tanda"] = (
        scenario["volumen_base_sem"] * scenario["duracion_dias"] / 7.0
    )
    scenario["elasticidad_abs"] = scenario["elasticidad_estimada"].abs()
    scenario["descuento_x_elasticidad"] = (
        scenario["factor_descuento"] * scenario["elasticidad_abs"]
    )
    scenario["descuento_x_duracion"] = (
        scenario["factor_descuento"] * scenario["duracion_dias"]
    )
    scenario["factor_descuento_sq"] = scenario["factor_descuento"] ** 2
    scenario["duracion_dias_sq"] = scenario["duracion_dias"] ** 2
    scenario["log1p_volumen_base_tanda"] = np.log1p(scenario["volumen_base_tanda"])
    return scenario


def select_curve_contexts(development: pd.DataFrame) -> pd.DataFrame:
    ordered = development.sort_values("fecha_inicio_tanda")
    selections = []
    definitions = [
        ("buen_soporte", ordered[ordered["clasificacion_soporte"].isin(["soporte_alto", "soporte_medio"])]),
        ("sku_reciente", ordered[ordered["clasificacion_sku"].eq("sku_reciente")]),
        ("soporte_bajo", ordered[ordered["clasificacion_soporte"].isin(["soporte_bajo", "soporte_insuficiente"])]),
    ]
    used = set()
    for label, subset in definitions:
        if subset.empty:
            subset = ordered
        available = subset[
            ~subset.apply(lambda row: (row["id_material"], row["subcadena"]) in used, axis=1)
        ]
        row = (available if not available.empty else subset).iloc[-1].copy()
        used.add((row["id_material"], row["subcadena"]))
        row["contexto_curva"] = label
        selections.append(row)
    return pd.DataFrame(selections).reset_index(drop=True)


def curve_diagnostics(
    model,
    features: list[str],
    contexts: pd.DataFrame,
    discounts: list[float],
    durations: list[int],
) -> tuple[float, pd.DataFrame]:
    rows = []
    smoothness = []
    for _, context in contexts.iterrows():
        discount_scenario = _recalculate_scenario(context, discounts=discounts)
        discount_predictions = model.predict(discount_scenario[features])
        duration_scenario = _recalculate_scenario(context, durations=durations)
        duration_predictions = model.predict(duration_scenario[features])
        for value, prediction in zip(discounts, discount_predictions):
            rows.append(
                {
                    "contexto_curva": context["contexto_curva"],
                    "id_material": context["id_material"],
                    "subcadena": context["subcadena"],
                    "palanca": "descuento",
                    "valor": value,
                    "prediccion": prediction,
                }
            )
        for value, prediction in zip(durations, duration_predictions):
            rows.append(
                {
                    "contexto_curva": context["contexto_curva"],
                    "id_material": context["id_material"],
                    "subcadena": context["subcadena"],
                    "palanca": "duracion",
                    "valor": value,
                    "prediccion": prediction,
                }
            )
        for predictions in [discount_predictions, duration_predictions]:
            scale = max(float(np.ptp(predictions)), 1e-8)
            smoothness.append(float(np.mean(np.abs(np.diff(predictions, n=2))) / scale))
    return float(np.mean(smoothness)), pd.DataFrame(rows)


def _ordinal_score(series: pd.Series, lower_is_better: bool = True) -> pd.Series:
    rank = series.rank(method="average", ascending=lower_is_better, pct=True)
    return (6 - np.ceil(rank * 5)).clip(1, 5).astype(int)


def build_scorecard(
    results: list[CandidateResult],
    fitted_models: dict,
    development: pd.DataFrame,
    contexts: pd.DataFrame,
    target: str,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    curves = {}
    for result in results:
        predictions = result.predictions
        sku_metrics = metrics_by_group(predictions, target, ["id_material"])
        recent = predictions[predictions["madurez_sku"].eq("sku_reciente")]
        recent_mae = (
            evaluate_prediction_frame(recent, target)["mae"] if not recent.empty else result.summary["mae"]
        )
        features = get_feature_set(result.candidate["feature_set"])
        smoothness, curve = curve_diagnostics(
            fitted_models[result.candidate["candidate_id"]],
            features,
            contexts,
            config["modeling"]["response_discount_grid"],
            config["modeling"]["response_durations"],
        )
        curves[result.candidate["candidate_id"]] = curve
        row = {
            **result.summary,
            "worst_sku_mae": float(sku_metrics["mae"].max()),
            "recent_sku_mae": recent_mae,
            "smoothness_index": smoothness,
            "false_positive_rate": result.summary.get("false_positive_rate", 0.0),
        }
        rows.append(row)
    scorecard = pd.DataFrame(rows)
    scorecard["score_mae"] = _ordinal_score(scorecard["mae"])
    scorecard["score_stability"] = _ordinal_score(scorecard["mae_fold_std"])
    scorecard["score_worst_sku"] = _ordinal_score(scorecard["worst_sku_mae"])
    scorecard["score_recent"] = _ordinal_score(scorecard["recent_sku_mae"])
    scorecard["score_smoothness"] = _ordinal_score(scorecard["smoothness_index"])
    scorecard["score_business_risk"] = _ordinal_score(
        scorecard["false_positive_rate"] if target == TARGET_ROI else scorecard["mae_unidades"]
    )
    scorecard["score_interpretability"] = scorecard["familia"].map(
        {"ridge": 5, "hist_gradient_boosting": 3, "extra_trees": 2}
    )
    scorecard["score_complexity"] = scorecard["familia"].map(
        {"ridge": 5, "hist_gradient_boosting": 3, "extra_trees": 2}
    )
    scorecard["score_training_time"] = scorecard["familia"].map(
        {"ridge": 5, "hist_gradient_boosting": 3, "extra_trees": 2}
    )
    scorecard["score_total"] = (
        0.25 * scorecard["score_mae"]
        + 0.15 * scorecard["score_stability"]
        + 0.10 * scorecard["score_worst_sku"]
        + 0.10 * scorecard["score_recent"]
        + 0.10 * scorecard["score_smoothness"]
        + 0.10 * scorecard["score_business_risk"]
        + 0.08 * scorecard["score_interpretability"]
        + 0.07 * scorecard["score_complexity"]
        + 0.05 * scorecard["score_training_time"]
    )
    scorecard = scorecard.sort_values(
        ["score_total", "mae", "mae_fold_std"], ascending=[False, True, True]
    ).reset_index(drop=True)
    scorecard["rol_seleccion"] = "no_seleccionado"
    scorecard.loc[0, "rol_seleccion"] = "champion"
    if len(scorecard) > 1:
        scorecard.loc[1, "rol_seleccion"] = "challenger"
    scorecard["decision_fijada_antes_test"] = True
    return scorecard, curves


def fit_candidate_model(
    data: pd.DataFrame, target: str, candidate: dict, random_seed: int
):
    features = get_feature_set(candidate["feature_set"])
    model = build_model_pipeline(
        candidate["family"],
        target,
        features,
        candidate["params"],
        candidate.get("target_transform", "original"),
        random_seed,
    )
    model.fit(data[features], data[target])
    return model


def feature_importance_table(
    model,
    candidate: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target: str,
    random_seed: int,
) -> pd.DataFrame:
    features = get_feature_set(candidate["feature_set"])
    fitted = build_model_pipeline(
        candidate["family"], target, features, candidate["params"],
        candidate.get("target_transform", "original"), random_seed
    )
    fitted.fit(train[features], train[target])
    if candidate["family"] == "ridge":
        pipeline = unwrap_fitted_pipeline(fitted)
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        coefficients = pipeline.named_steps["model"].coef_
        return pd.DataFrame(
            {
                "feature": names,
                "importance": np.abs(coefficients),
                "signed_value": coefficients,
                "method": "coeficiente regularizado en espacio preprocesado",
                "warning": "signos pueden variar por colinealidad",
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)
    importance = permutation_importance(
        fitted,
        validation[features],
        validation[target],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=random_seed,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": features,
            "importance": importance.importances_mean,
            "signed_value": importance.importances_mean,
            "importance_std": importance.importances_std,
            "method": "permutation importance en ultima validacion temporal",
            "warning": "asociacion predictiva, no causal",
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)


def _long_segment_metrics(predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for segment in [
        "id_material", "subcadena", "banda_descuento", "duracion_dias",
        "soporte", "madurez_sku", "flag_uplift_en_piso", "flag_roi_extremo_descriptivo"
    ]:
        table = metrics_by_group(predictions, target, ["candidate_id", segment])
        table["segmento"] = segment
        table["valor_segmento"] = table[segment].astype("string")
        rows.append(table.drop(columns=segment))
    combo = metrics_by_group(predictions, target, ["candidate_id", "id_material", "subcadena"])
    combo["segmento"] = "sku_x_cadena"
    combo["valor_segmento"] = combo["id_material"].astype("string") + "__" + combo["subcadena"].astype("string")
    rows.append(combo.drop(columns=["id_material", "subcadena"]))
    return pd.concat(rows, ignore_index=True)


def run_modeling_workflow(project_root: Path, config: dict) -> dict:
    seed = int(config["modeling"].get("random_seed", RANDOM_SEED))
    frame = load_modeling_frame(project_root, config)
    test_start = choose_final_test_start(
        frame["fecha_inicio_tanda"], config["modeling"].get("final_test_months", 3)
    )
    development_mask, test_mask = development_test_masks(frame["fecha_inicio_tanda"], test_start)
    development = frame.loc[development_mask].reset_index(drop=True)
    final_test = frame.loc[test_mask].reset_index(drop=True)
    folds = expanding_window_splits(
        development,
        n_splits=int(config["modeling"].get("n_temporal_splits", 4)),
    )
    split_summary = temporal_split_summary(development, folds)
    test_coverage = final_test_coverage(development, final_test)
    baselines = evaluate_baselines(development, folds)

    ablation_results = []
    selected_sets = {}
    for target in [TARGET_UPLIFT, TARGET_ROI]:
        target_results = [
            evaluate_candidate(development, folds, target, candidate, seed)
            for candidate in ablation_candidates(target)
        ]
        ablation_results.extend(target_results)
        selected_sets[target] = min(target_results, key=lambda result: result.summary["mae"]).candidate["feature_set"]

    tuned_results: dict[str, list[CandidateResult]] = {}
    fitted_development_models: dict[str, object] = {}
    all_fold_metrics = []
    scorecards = []
    curve_cache = {}
    contexts = select_curve_contexts(development)
    for target in [TARGET_UPLIFT, TARGET_ROI]:
        results = [
            evaluate_candidate(development, folds, target, candidate, seed)
            for candidate in tuning_candidates(target, selected_sets[target])
        ]
        tuned_results[target] = results
        for result in results:
            fitted_development_models[result.candidate["candidate_id"]] = fit_candidate_model(
                development, target, result.candidate, seed
            )
            all_fold_metrics.append(result.fold_metrics)
        scorecard, curves = build_scorecard(
            results, fitted_development_models, development, contexts, target, config
        )
        scorecards.append(scorecard)
        curve_cache.update(curves)

    scorecard = pd.concat(scorecards, ignore_index=True)
    selected: dict[str, dict[str, CandidateResult]] = {}
    for target in [TARGET_UPLIFT, TARGET_ROI]:
        target_score = scorecard[scorecard["target"].eq(target)]
        selected[target] = {}
        for role in ["champion", "challenger"]:
            candidate_id = target_score.loc[target_score["rol_seleccion"].eq(role), "candidate_id"].iloc[0]
            selected[target][role] = next(
                result for result in tuned_results[target] if result.candidate["candidate_id"] == candidate_id
            )

    selected_oof = {}
    for target in [TARGET_UPLIFT, TARGET_ROI]:
        selected_oof[target] = pd.concat(
            [selected[target][role].predictions.assign(rol=role) for role in ["champion", "challenger"]],
            ignore_index=True,
        )

    # Direct ROI versus two-stage: OOF uplift is the only uplift signal allowed.
    uplift_champion_oof = selected[TARGET_UPLIFT]["champion"].predictions[
        ["row_id", "prediccion", "fold"]
    ].rename(columns={"prediccion": "uplift_predicho_oof"})
    roi_direct = selected[TARGET_ROI]["champion"]
    two_stage_rows = []
    two_stage_predictions = []
    for fold in folds[1:]:
        train = development.iloc[fold.train_indices].merge(
            uplift_champion_oof[["row_id", "uplift_predicho_oof"]],
            on="row_id", how="inner", validate="one_to_one"
        )
        validation = development.iloc[fold.validation_indices].merge(
            uplift_champion_oof[["row_id", "uplift_predicho_oof"]],
            on="row_id", how="left", validate="one_to_one"
        )
        if train.empty or validation["uplift_predicho_oof"].isna().any():
            continue
        candidate = roi_direct.candidate
        features = [*get_feature_set(candidate["feature_set"]), "uplift_predicho_oof"]
        model = build_model_pipeline(
            candidate["family"], TARGET_ROI, features, candidate["params"], "original", seed
        )
        model.fit(train[features], train[TARGET_ROI])
        pred = model.predict(validation[features])
        stage_candidate = {**candidate, "candidate_id": "roi_two_stage_oof", "model_name": "roi_two_stage_oof"}
        pred_frame = _prediction_frame(validation, TARGET_ROI, pred, fold.fold, stage_candidate)
        metrics = evaluate_prediction_frame(pred_frame, TARGET_ROI)
        direct_same = roi_direct.predictions[roi_direct.predictions["fold"].eq(fold.fold)]
        direct_metrics = evaluate_prediction_frame(direct_same, TARGET_ROI)
        two_stage_rows.append(
            {
                "fold": fold.fold,
                "mae_directo": direct_metrics["mae"],
                "mae_dos_etapas": metrics["mae"],
                "mejora_relativa": (direct_metrics["mae"] - metrics["mae"]) / direct_metrics["mae"],
                "falsos_positivos_directo": direct_metrics["falsos_positivos"],
                "falsos_positivos_dos_etapas": metrics["falsos_positivos"],
                "train_filas_con_uplift_oof": len(train),
                "validation_filas": len(validation),
            }
        )
        two_stage_predictions.append(pred_frame)
    two_stage_comparison = pd.DataFrame(two_stage_rows)
    threshold = config["modeling"].get("two_stage_min_relative_improvement", 0.05)
    two_stage_selected = bool(
        len(two_stage_comparison) >= 2
        and two_stage_comparison["mejora_relativa"].mean() >= threshold
        and (two_stage_comparison["mejora_relativa"] > 0).all()
        and (
            two_stage_comparison["falsos_positivos_dos_etapas"].sum()
            <= two_stage_comparison["falsos_positivos_directo"].sum()
        )
    )
    if not two_stage_comparison.empty:
        two_stage_comparison["decision"] = (
            "seleccionar_dos_etapas" if two_stage_selected else "conservar_roi_directo"
        )
        two_stage_comparison["guardrail"] = "uplift de train y validation generado OOF"

    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    cv_uplift = pd.DataFrame([result.summary for result in tuned_results[TARGET_UPLIFT]])
    cv_roi = pd.DataFrame([result.summary for result in tuned_results[TARGET_ROI]])
    ablation = pd.DataFrame([result.summary for result in ablation_results])

    hyper_search = pd.concat([cv_uplift, cv_roi], ignore_index=True)
    hyper_search["complejidad"] = hyper_search["familia"].map(
        {"ridge": "baja", "hist_gradient_boosting": "media", "extra_trees": "alta"}
    )
    selected_ids = set(scorecard.loc[scorecard["rol_seleccion"].isin(["champion", "challenger"]), "candidate_id"])
    hyper_search["decision"] = np.where(
        hyper_search["candidate_id"].isin(selected_ids), "scorecard_final", "descartado_en_desarrollo"
    )
    hyper_search["justificacion"] = "busqueda acotada con validacion temporal expansiva"

    combined_selected_oof = pd.concat(selected_oof.values(), ignore_index=True)
    segment_metrics = pd.concat(
        [_long_segment_metrics(selected_oof[target], target).assign(target=target) for target in [TARGET_UPLIFT, TARGET_ROI]],
        ignore_index=True,
    )
    by_sku = segment_metrics[segment_metrics["segmento"].eq("id_material")]
    by_chain = segment_metrics[segment_metrics["segmento"].eq("subcadena")]
    by_support = segment_metrics[segment_metrics["segmento"].eq("soporte")]

    uncertainty_tables = []
    uncertainty_radii = {}
    for target in [TARGET_UPLIFT, TARGET_ROI]:
        champion_oof = selected[target]["champion"].predictions
        uncertainty, radius = uncertainty_summary(
            champion_oof,
            target,
            alpha=config["modeling"].get("uncertainty_alpha", 0.10),
            group_columns=["id_material", "soporte"],
        )
        uncertainty_tables.append(uncertainty)
        uncertainty_radii[target] = radius
    uncertainty_table = pd.concat(uncertainty_tables, ignore_index=True)

    statistical = pd.concat(
        [
            compare_models_by_date(
                selected[target]["champion"].predictions,
                selected[target]["challenger"].predictions,
                target,
            )
            for target in [TARGET_UPLIFT, TARGET_ROI]
        ],
        ignore_index=True,
    )

    models_dir = project_root / config["project"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    model_paths = {
        (TARGET_UPLIFT, "champion"): models_dir / "uplift_champion.joblib",
        (TARGET_UPLIFT, "challenger"): models_dir / "uplift_challenger.joblib",
        (TARGET_ROI, "champion"): models_dir / "roi_champion.joblib",
        (TARGET_ROI, "challenger"): models_dir / "roi_challenger.joblib",
    }
    final_metrics = {TARGET_UPLIFT: [], TARGET_ROI: []}
    final_predictions = {}
    registry_entries = []
    fitted_selected = {}
    version = data_version_hash(project_root, config)
    for target in [TARGET_UPLIFT, TARGET_ROI]:
        target_frames = []
        for role in ["champion", "challenger"]:
            result = selected[target][role]
            candidate = result.candidate
            model = fit_candidate_model(development, target, candidate, seed)
            features = get_feature_set(candidate["feature_set"])
            prediction = model.predict(final_test[features])
            pred_frame = _prediction_frame(final_test, target, prediction, 0, candidate).assign(rol=role)
            pred_frame["interval_lower"] = pred_frame["prediccion"] - uncertainty_radii[target]
            pred_frame["interval_upper"] = pred_frame["prediccion"] + uncertainty_radii[target]
            metrics = evaluate_prediction_frame(pred_frame, target)
            metrics.update(
                {
                    "target": target,
                    "rol": role,
                    "candidate_id": candidate["candidate_id"],
                    "modelo": candidate["model_name"],
                    "familia": candidate["family"],
                    "feature_set": candidate["feature_set"],
                    "decision_fijada_antes_test": True,
                }
            )
            final_metrics[target].append(metrics)
            target_frames.append(pred_frame)
            save_model(model, model_paths[(target, role)])
            reloaded = load_model(model_paths[(target, role)])
            reload_prediction = reloaded.predict(final_test.iloc[:5][features])
            if not np.isfinite(reload_prediction).all():
                raise ValueError(f"Reloaded model failed prediction: {target} {role}")
            fitted_selected[(target, role)] = model
            registry_entries.append(
                build_registry_entry(
                    target=target,
                    role=role,
                    model_name=candidate["model_name"],
                    family=candidate["family"],
                    feature_set=candidate["feature_set"],
                    features=features,
                    hyperparameters=candidate["params"],
                    train_period=(str(development["fecha_inicio_tanda"].min().date()), str(development["fecha_inicio_tanda"].max().date())),
                    test_period=(str(final_test["fecha_inicio_tanda"].min().date()), str(final_test["fecha_inicio_tanda"].max().date())),
                    cv_metrics=result.summary,
                    final_test_metrics=metrics,
                    artifact_path=model_paths[(target, role)].relative_to(project_root),
                    data_version=version,
                    random_seed=seed,
                )
            )
        final_predictions[target] = pd.concat(target_frames, ignore_index=True)
    registry_path = project_root / config["project"]["model_registry"]
    write_registry(registry_entries, registry_path)

    last_fold = folds[-1]
    importance_tables = {}
    for target in [TARGET_UPLIFT, TARGET_ROI]:
        candidate = selected[target]["champion"].candidate
        importance_tables[target] = feature_importance_table(
            fitted_selected[(target, "champion")],
            candidate,
            development.iloc[last_fold.train_indices],
            development.iloc[last_fold.validation_indices],
            target,
            seed,
        )

    response_dir = project_root / config["project"]["response_curves_dir"]
    response_dir.mkdir(parents=True, exist_ok=True)
    response_diagnostics = []
    uplift_candidate = selected[TARGET_UPLIFT]["champion"].candidate
    roi_candidate = selected[TARGET_ROI]["champion"].candidate
    _, uplift_curves = curve_diagnostics(
        fitted_selected[(TARGET_UPLIFT, "champion")],
        get_feature_set(uplift_candidate["feature_set"]),
        contexts,
        config["modeling"]["response_discount_grid"],
        config["modeling"]["response_durations"],
    )
    _, roi_curves = curve_diagnostics(
        fitted_selected[(TARGET_ROI, "champion")],
        get_feature_set(roi_candidate["feature_set"]),
        contexts,
        config["modeling"]["response_discount_grid"],
        config["modeling"]["response_durations"],
    )
    for context_name in contexts["contexto_curva"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for target_name, curves, color in [("uplift", uplift_curves, "#4E79A7"), ("roi", roi_curves, "#E15759")]:
            subset = curves[curves["contexto_curva"].eq(context_name)]
            for ax, lever in zip(axes, ["descuento", "duracion"]):
                lever_data = subset[subset["palanca"].eq(lever)]
                ax.plot(lever_data["valor"], lever_data["prediccion"], marker="o", label=target_name, color=color)
                ax.set_xlabel("Factor descuento" if lever == "descuento" else "Duracion (dias)")
                ax.set_ylabel("Prediccion")
                ax.set_title(f"Respuesta por {lever}")
                ax.legend()
        fig.suptitle(f"Curvas descriptivas: {context_name}")
        fig.tight_layout()
        figure_path = response_dir / f"response_curve_{context_name}.png"
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
        context_uplift = uplift_curves[uplift_curves["contexto_curva"].eq(context_name)]
        context_roi = roi_curves[roi_curves["contexto_curva"].eq(context_name)]
        response_diagnostics.append(
            {
                "contexto": context_name,
                "id_material": context_uplift["id_material"].iloc[0],
                "subcadena": context_uplift["subcadena"].iloc[0],
                "uplift_rango_predicho": context_uplift["prediccion"].max() - context_uplift["prediccion"].min(),
                "roi_rango_predicho": context_roi["prediccion"].max() - context_roi["prediccion"].min(),
                "figura": str(figure_path.relative_to(project_root).as_posix()),
                "advertencia": "diagnostico de suavidad; no selecciona optimos",
            }
        )

    return {
        "frame": frame,
        "development": development,
        "final_test": final_test,
        "test_start": test_start,
        "split_summary": split_summary,
        "test_coverage": test_coverage,
        "baseline_metrics": baselines,
        "ablation": ablation,
        "hyperparameter_search": hyper_search,
        "cv_metrics_uplift": cv_uplift,
        "cv_metrics_roi": cv_roi,
        "cv_metrics_by_fold": fold_metrics,
        "selected_oof_uplift": selected_oof[TARGET_UPLIFT],
        "selected_oof_roi": selected_oof[TARGET_ROI],
        "segment_metrics": segment_metrics,
        "metrics_by_sku": by_sku,
        "metrics_by_chain": by_chain,
        "metrics_by_support": by_support,
        "scorecard": scorecard,
        "two_stage_comparison": two_stage_comparison,
        "two_stage_selected": two_stage_selected,
        "uncertainty": uncertainty_table,
        "statistical_comparison": statistical,
        "final_metrics_uplift": pd.DataFrame(final_metrics[TARGET_UPLIFT]),
        "final_metrics_roi": pd.DataFrame(final_metrics[TARGET_ROI]),
        "final_predictions_uplift": final_predictions[TARGET_UPLIFT],
        "final_predictions_roi": final_predictions[TARGET_ROI],
        "importance_uplift": importance_tables[TARGET_UPLIFT],
        "importance_roi": importance_tables[TARGET_ROI],
        "response_diagnostics": pd.DataFrame(response_diagnostics),
        "model_paths": model_paths,
        "registry_path": registry_path,
        "selected": selected,
        "data_version": version,
    }
