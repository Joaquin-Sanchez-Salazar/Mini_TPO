from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mini_tpo.constants import RANDOM_SEED, TARGET_ROI, TARGET_UPLIFT
from mini_tpo.model_registry import build_registry_entry, load_model, save_model, write_registry
from mini_tpo.modeling import data_version_hash, load_modeling_frame
from mini_tpo.optimization_evaluation import (
    add_prediction_outputs,
    apply_guardrails,
    oof_error_quantile,
    recommendation_record,
    select_case_solutions,
)
from mini_tpo.pareto import pareto_frontier
from mini_tpo.preprocessing import build_model_pipeline
from mini_tpo.scenario_generation import (
    add_local_support,
    build_scenario_grid,
    discount_grid,
    next_weekly_date,
)


def load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def registry_model(registry: dict, target: str, role: str) -> dict:
    matches = [m for m in registry["models"] if m["target"] == target and m["role"] == role]
    if len(matches) != 1:
        raise ValueError(f"Expected one registry model for {target}/{role}, got {len(matches)}")
    return matches[0]


def _support_rank(series: pd.Series) -> pd.Series:
    return series.map(
        {"soporte_alto": 4, "soporte_medio": 3, "soporte_bajo": 2, "soporte_insuficiente": 1}
    ).fillna(0)


def select_optimization_cases(history: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    profile = support.copy()
    profile["support_rank"] = _support_rank(profile["clasificacion_soporte"])
    sku_dates = history.groupby("id_material")["fecha_inicio_tanda"].min()
    recent_cutoff = pd.to_datetime(history["fecha_inicio_tanda"]).max() - pd.Timedelta(days=180)
    profile["is_recent"] = profile["id_material"].map(sku_dates).ge(recent_cutoff)

    case_a = profile.sort_values(["support_rank", "promociones"], ascending=False).iloc[0]
    b_pool = profile[
        ~profile["id_material"].eq(case_a["id_material"])
        & ~profile["subcadena"].eq(case_a["subcadena"])
        & ~profile["is_recent"]
        & profile["support_rank"].ge(3)
    ]
    case_b = b_pool.sort_values(["volumen_base_mediano", "promociones"], ascending=False).iloc[0]
    c_pool = profile[
        profile["is_recent"]
        & ~profile["id_material"].isin([case_a["id_material"], case_b["id_material"]])
        & ~profile["subcadena"].isin([case_a["subcadena"], case_b["subcadena"]])
    ]
    if c_pool.empty:
        c_pool = profile[profile["is_recent"]]
    case_c = c_pool.sort_values(["support_rank", "promociones"], ascending=False).iloc[0]

    labels = [
        (case_a, "caso_a_soporte_alto", "SKU maduro con el mejor soporte local; referencia de mayor confianza."),
        (case_b, "caso_b_alto_volumen", "SKU maduro de alto volumen base; hace visible el error y crecimiento en unidades."),
        (case_c, "caso_c_sku_reciente", "SKU reciente con soporte bajo; tensiona guardrails y exige revision humana."),
    ]
    rows = []
    for selected, case_id, reason in labels:
        local = history[
            history["id_material"].eq(selected["id_material"])
            & history["subcadena"].eq(selected["subcadena"])
        ]
        rows.append(
            {
                "case_id": case_id,
                "id_material": selected["id_material"],
                "des_material": local["des_material"].mode().iloc[0],
                "des_marca": local["des_marca"].mode().iloc[0],
                "subcadena": selected["subcadena"],
                "motivo_seleccion": reason,
                "observaciones_historicas": int(selected["promociones"]),
                "nivel_soporte": selected["clasificacion_soporte"],
                "descuento_historico_min": selected["descuento_min"],
                "descuento_historico_max": selected["descuento_max"],
                "duraciones_observadas": selected["duraciones_observadas"],
                "volumen_base_mediano": selected["volumen_base_mediano"],
                "fecha_inicial": selected["fecha_min"],
                "fecha_final": selected["fecha_max"],
            }
        )
    result = pd.DataFrame(rows)
    if result["id_material"].nunique() != 3:
        raise ValueError("Optimization case selection must contain three unique SKUs")
    return result


def build_optimization_contexts(
    history: pd.DataFrame, cases: pd.DataFrame, reference_date: pd.Timestamp, recent_n: int
) -> pd.DataFrame:
    rows = []
    for case in cases.itertuples(index=False):
        local = history[
            history["id_material"].eq(case.id_material)
            & history["subcadena"].eq(case.subcadena)
        ].sort_values(["fecha_inicio_tanda", "row_id"])
        recent = local.tail(recent_n)
        rows.append(
            {
                **case._asdict(),
                "fecha_referencia": reference_date,
                "volumen_base_sem": float(recent["volumen_base_sem"].median()),
                "elasticidad_estimada": float(recent["elasticidad_estimada"].median()),
                "volumen_base_sem_ultimo": float(local["volumen_base_sem"].iloc[-1]),
                "elasticidad_estimada_ultima": float(local["elasticidad_estimada"].iloc[-1]),
                "flag_secundario": "no",
                "regla_contexto": f"mediana de ultimas {recent_n} observaciones; escenario independiente no secundario",
            }
        )
    return pd.DataFrame(rows)


def refit_production_models(project_root: Path, config: dict, registry: dict, frame: pd.DataFrame):
    models = {}
    entries = [m for m in registry["models"] if m.get("role") != "production_refit"]
    version = data_version_hash(project_root, config)
    for target, filename in [(TARGET_UPLIFT, "uplift_production.joblib"), (TARGET_ROI, "roi_production.joblib")]:
        source = registry_model(registry, target, "champion")
        transform = source["cv_metrics"].get("target_transform", "original")
        model = build_model_pipeline(
            source["family"], target, source["features"], source["hyperparameters"],
            transform, int(source["random_seed"]),
        )
        model.fit(frame[source["features"]], frame[target])
        path = project_root / "models" / filename
        save_model(model, path)
        if not np.isfinite(load_model(path).predict(frame.iloc[:3][source["features"]])).all():
            raise ValueError(f"Production model reload failed: {target}")
        entry = build_registry_entry(
            target=target,
            role="production_refit",
            model_name=source["model"],
            family=source["family"],
            feature_set=source["feature_set"],
            features=source["features"],
            hyperparameters=source["hyperparameters"],
            train_period=(str(frame["fecha_inicio_tanda"].min().date()), str(frame["fecha_inicio_tanda"].max().date())),
            test_period=(source["test_period"]["start"], source["test_period"]["end"]),
            cv_metrics=source["cv_metrics"],
            final_test_metrics=source["final_test_metrics"],
            artifact_path=path.relative_to(project_root),
            data_version=version,
            random_seed=int(source["random_seed"]),
        )
        entry["lifecycle_stage"] = "production"
        entry["source_role"] = "champion"
        entry["selection_changed"] = False
        entries.append(entry)
        models[target] = model
    write_registry(entries, project_root / config["project"]["model_registry"])
    return models, entries


def _prepare_scenarios(
    contexts, history, support, discounts, durations, config, uplift_model, roi_model,
    uplift_entry, roi_entry, q_uplift, q_roi, grid_type="principal_soportada",
):
    scenarios = build_scenario_grid(contexts, discounts, durations, config, grid_type)
    scenarios = scenarios.merge(
        support[["id_material", "subcadena", "clasificacion_soporte"]].rename(
            columns={"clasificacion_soporte": "nivel_soporte"}
        ), on=["id_material", "subcadena"], validate="many_to_one"
    )
    scenarios = add_local_support(
        scenarios, history, config["optimization"]["local_discount_tolerance"]
    )
    scenarios = add_prediction_outputs(
        scenarios, uplift_model, roi_model, uplift_entry["features"], roi_entry["features"],
        q_uplift, q_roi, config["optimization"]["support_penalty_uplift"],
        config["optimization"]["support_penalty_roi"],
    )
    return scenarios


def _add_challenger_disagreement(scenarios, frame, registry, config):
    result = scenarios.copy()
    for target, column in [(TARGET_UPLIFT, "uplift_challenger"), (TARGET_ROI, "roi_challenger")]:
        entry = registry_model(registry, target, "challenger")
        transform = entry["cv_metrics"].get("target_transform", "original")
        model = build_model_pipeline(
            entry["family"], target, entry["features"], entry["hyperparameters"],
            transform, int(entry["random_seed"]),
        )
        model.fit(frame[entry["features"]], frame[target])
        result[column] = model.predict(result[entry["features"]])
    result["desacuerdo_uplift_abs"] = (result["uplift_esperado"] - result["uplift_challenger"]).abs()
    result["desacuerdo_roi_abs"] = (result["roi_esperado"] - result["roi_challenger"]).abs()
    result["flag_model_disagreement"] = (
        result["desacuerdo_uplift_abs"].gt(config["optimization"]["material_uplift_disagreement"])
        | result["desacuerdo_roi_abs"].gt(config["optimization"]["material_roi_disagreement"])
    )
    return result


def _solution_table(solutions: dict, kind: str) -> pd.DataFrame:
    rows = []
    for case_id, values in solutions.items():
        solution = values[kind]
        if solution is not None:
            row = solution.to_dict()
            row["solution_type"] = kind
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_case(case_frame: pd.DataFrame, recommendation: pd.Series, output_dir: Path) -> list[Path]:
    case_id = recommendation["case_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, case_frame["duracion_dias"].nunique()))
    for metric, lower, upper, ylabel, name in [
        ("volumen_promo_esperado", "volumen_promo_lower_90", "volumen_promo_upper_90", "Volumen promocional esperado (unidades)", "demand"),
        ("uplift_esperado", "uplift_lower_90", "uplift_upper_90", "Uplift esperado", "uplift"),
        ("roi_esperado", "roi_lower_90", "roi_upper_90", "ROI esperado", "roi"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for color, (duration, group) in zip(colors, case_frame.groupby("duracion_dias")):
            group = group.sort_values("factor_descuento")
            ax.plot(group["factor_descuento"] * 100, group[metric], label=f"{duration} dias", color=color)
            if lower and upper:
                ax.fill_between(group["factor_descuento"] * 100, group[lower], group[upper], color=color, alpha=0.08)
        if metric == "roi_esperado":
            ax.axhline(0, color="black", linewidth=1)
        ax.scatter(
            recommendation["factor_descuento_recomendado"] * 100,
            recommendation[metric], color="#D62728", marker="X", s=90, label="Recomendacion",
        )
        ax.set(title=f"{case_id}: {ylabel}", xlabel="Descuento (%)", ylabel=ylabel)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        path = output_dir / f"{case_id}_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    pivot = case_frame.pivot(index="duracion_dias", columns="factor_descuento", values="roi_robusto")
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(pivot, aspect="auto", cmap="RdYlGn")
    ax.set(title=f"{case_id}: ROI robusto", xlabel="Descuento", ylabel="Duracion (dias)")
    positions = np.arange(0, len(pivot.columns), max(1, len(pivot.columns) // 8))
    ax.set_xticks(positions, [f"{pivot.columns[i]:.0%}" for i in positions])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    fig.colorbar(image, ax=ax, label="ROI robusto")
    fig.tight_layout()
    path = output_dir / f"{case_id}_heatmap.png"
    fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)

    frontier = pareto_frontier(case_frame)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(case_frame["volumen_incremental_esperado"], case_frame["roi_esperado"], c=case_frame["factor_descuento"], cmap="viridis", alpha=.35)
    ax.plot(frontier["volumen_incremental_esperado"], frontier["roi_esperado"], color="#D62728", marker="o", label="Frontera Pareto")
    ax.set(title=f"{case_id}: retorno versus crecimiento", xlabel="Volumen incremental esperado", ylabel="ROI esperado")
    ax.legend(); fig.tight_layout()
    path = output_dir / f"{case_id}_pareto.png"
    fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    return paths


def run_optimization_workflow(project_root: Path, config: dict) -> dict:
    registry_path = project_root / config["project"]["model_registry"]
    registry = load_registry(registry_path)
    uplift_entry = registry_model(registry, TARGET_UPLIFT, "champion")
    roi_entry = registry_model(registry, TARGET_ROI, "champion")
    frame = load_modeling_frame(project_root, config)
    history = pd.read_parquet(project_root / config["project"]["processed_modeling"])
    history["fecha_inicio_tanda"] = pd.to_datetime(history["fecha_inicio_tanda"])
    support = pd.read_csv(project_root / "reports/tables/support_sku_chain.csv")
    for column in ["fecha_min", "fecha_max"]:
        support[column] = pd.to_datetime(support[column])

    production_models, registry_entries = refit_production_models(project_root, config, registry, frame)
    cases = select_optimization_cases(history, support)
    reference_date = next_weekly_date(history["fecha_inicio_tanda"])
    contexts = build_optimization_contexts(
        history, cases, reference_date, config["optimization"]["recent_observations"]
    )
    discounts = discount_grid(
        config["business_rules"]["discount_min"], config["business_rules"]["discount_max"],
        config["optimization"]["discount_step"],
    )
    durations = config["optimization"]["supported_durations"]
    q_uplift = oof_error_quantile(project_root / config["project"]["oof_uplift"])
    q_roi = oof_error_quantile(project_root / config["project"]["oof_roi"])
    scenarios = _prepare_scenarios(
        contexts, history, support, discounts, durations, config,
        production_models[TARGET_UPLIFT], production_models[TARGET_ROI],
        uplift_entry, roi_entry, q_uplift, q_roi,
    )
    scenarios = _add_challenger_disagreement(scenarios, frame, registry, config)
    scenarios = apply_guardrails(scenarios, config["optimization"]["min_local_support"])
    epsilon = max(
        config["optimization"]["roi_epsilon_min"],
        config["optimization"]["roi_epsilon_error_fraction"] * q_roi,
    )
    solutions = {
        case_id: select_case_solutions(group, epsilon)
        for case_id, group in scenarios.groupby("case_id", sort=False)
    }
    recommendations = pd.DataFrame([
        recommendation_record(
            contexts[contexts["case_id"].eq(case_id)].iloc[0], values["robust"], values["mathematical"]
        ) for case_id, values in solutions.items()
    ])
    mathematical = _solution_table(solutions, "mathematical")
    growth = _solution_table(solutions, "growth")
    pareto = pd.concat([
        pareto_frontier(group).assign(case_id=case_id)
        for case_id, group in scenarios.groupby("case_id", sort=False)
    ], ignore_index=True)

    # Sensitivities are diagnostics only; they never replace registered champions.
    sensitivity_rows = []
    for case_id, group in scenarios.groupby("case_id", sort=False):
        for variant, objective, prediction_column in [
            ("champion_principal", "roi_robusto", "roi_esperado"),
            ("roi_challenger", "roi_challenger", "roi_challenger"),
            ("uplift_challenger", "roi_robusto", "uplift_challenger"),
        ]:
            if variant == "champion_principal" and solutions[case_id]["robust"] is not None:
                best = solutions[case_id]["robust"]
            else:
                valid = group[group["flag_guardrails_robustos"]]
                pool = valid if not valid.empty else group
                best = pool.sort_values([objective, "volumen_incremental_esperado"], ascending=False).iloc[0]
            sensitivity_rows.append({
                "case_id": case_id, "variante": variant,
                "factor_descuento": best["factor_descuento"], "duracion_dias": best["duracion_dias"],
                "roi": best["roi_esperado"], "uplift": best["uplift_esperado"],
                "valor_diagnostico": best[prediction_column],
            })
    # Context and duration sensitivities.
    for variant, variant_contexts, variant_durations in [
        ("ultimo_valor_contexto", contexts.assign(volumen_base_sem=contexts["volumen_base_sem_ultimo"], elasticidad_estimada=contexts["elasticidad_estimada_ultima"]), durations),
        ("flag_secundario_si", contexts.assign(flag_secundario="si"), durations),
        ("duraciones_todos_enteros", contexts, config["optimization"]["exploratory_durations"]),
    ]:
        variant_grid = _prepare_scenarios(
            variant_contexts, history, support, discounts, variant_durations, config,
            production_models[TARGET_UPLIFT], production_models[TARGET_ROI],
            uplift_entry, roi_entry, q_uplift, q_roi, variant,
        )
        variant_grid = apply_guardrails(variant_grid, config["optimization"]["min_local_support"])
        for case_id, group in variant_grid.groupby("case_id", sort=False):
            variant_solution = select_case_solutions(group, epsilon)["robust"]
            best = variant_solution if variant_solution is not None else group.sort_values(
                ["roi_esperado", "volumen_incremental_esperado"], ascending=False
            ).iloc[0]
            sensitivity_rows.append({
                "case_id": case_id, "variante": variant,
                "factor_descuento": best["factor_descuento"], "duracion_dias": best["duracion_dias"],
                "roi": best["roi_esperado"], "uplift": best["uplift_esperado"],
                "valor_diagnostico": best["roi_robusto"],
            })
    q80_roi = oof_error_quantile(project_root / config["project"]["oof_roi"], quantile=.80)
    for case_id, group in scenarios.groupby("case_id", sort=False):
        interval80 = group.copy()
        interval80["roi_robusto_80"] = (
            interval80["roi_esperado"] - q80_roi - interval80["penalizacion_soporte_roi"]
        )
        valid = interval80[
            interval80["flag_duracion_observada"]
            & interval80["flag_descuento_en_rango_local"]
            & interval80["local_support_count"].ge(config["optimization"]["min_local_support"])
            & interval80["roi_esperado"].sub(q80_roi).gt(0)
            & ~interval80["nivel_soporte"].eq("soporte_insuficiente")
        ]
        pool = valid if not valid.empty else interval80
        best = pool.sort_values(["roi_robusto_80", "volumen_incremental_esperado"], ascending=False).iloc[0]
        sensitivity_rows.append({
            "case_id": case_id, "variante": "intervalo_80",
            "factor_descuento": best["factor_descuento"], "duracion_dias": best["duracion_dias"],
            "roi": best["roi_esperado"], "uplift": best["uplift_esperado"],
            "valor_diagnostico": best["roi_robusto_80"],
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    main = sensitivity[sensitivity["variante"].eq("champion_principal")][["case_id", "factor_descuento", "duracion_dias"]].rename(columns={"factor_descuento": "descuento_main", "duracion_dias": "duracion_main"})
    sensitivity = sensitivity.merge(main, on="case_id", how="left")
    sensitivity["recomendacion_estable"] = (
        (sensitivity["factor_descuento"] - sensitivity["descuento_main"]).abs().le(0.02)
        & (sensitivity["duracion_dias"] - sensitivity["duracion_main"]).abs().le(4)
    )
    sensitivity["q80_uplift_oof"] = oof_error_quantile(project_root / config["project"]["oof_uplift"], quantile=.80)
    sensitivity["q80_roi_oof"] = q80_roi
    sensitivity["q90_uplift_oof"] = q_uplift
    sensitivity["q90_roi_oof"] = q_roi

    grid_summary = scenarios.groupby("case_id", as_index=False).agg(
        escenarios=("row_id", "size"), escenarios_factibles=("flag_guardrails_robustos", "sum"),
        descuento_min=("factor_descuento", "min"), descuento_max=("factor_descuento", "max"),
        duraciones=("duracion_dias", lambda s: ",".join(map(str, sorted(s.unique())))),
        extrapolativos=("flag_extrapolacion", "sum"),
        desacuerdo_material=("flag_model_disagreement", "sum"),
    )
    figures = []
    figure_dir = project_root / config["project"]["optimization_figures_dir"]
    for _, recommendation in recommendations.iterrows():
        figures.extend(_plot_case(
            scenarios[scenarios["case_id"].eq(recommendation["case_id"])], recommendation, figure_dir
        ))
    return {
        "cases": cases, "contexts": contexts, "scenarios": scenarios,
        "recommendations": recommendations, "mathematical": mathematical,
        "growth": growth, "pareto": pareto, "sensitivity": sensitivity,
        "grid_summary": grid_summary, "figures": figures,
        "reference_date": reference_date, "q90_uplift": q_uplift, "q90_roi": q_roi,
        "epsilon_roi": epsilon, "registry_entries": registry_entries,
        "production_models": production_models,
    }
