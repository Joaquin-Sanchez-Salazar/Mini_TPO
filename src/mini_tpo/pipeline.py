from __future__ import annotations

from pathlib import Path

import pandas as pd

from mini_tpo.data_cleaning import build_cleaning_log, clean_data
from mini_tpo.constants import (
    FEATURES_MODEL_BASE,
    FEATURES_MODEL_MINIMAL,
    FEATURES_MODEL_OPTIONAL,
    LEAKAGE_COLUMNS,
    TARGET_ROI,
    TARGET_UPLIFT,
)
from mini_tpo.data_audit import (
    build_cardinality_profile,
    feature_availability_audit,
    model_role_summary,
    price_variation_by_sku,
    redundancy_audit,
)
from mini_tpo.data_loading import load_config, read_raw_data
from mini_tpo.data_validation import leakage_classification, validation_summary
from mini_tpo.feature_manifest import export_model_artifacts
from mini_tpo.feature_engineering import (
    build_engineered_core,
    build_engineered_optional,
    build_feature_catalog,
    build_optimizer_compatibility,
    create_feature_engineering_manifest,
    engineered_feature_correlation,
    engineered_feature_redundancy,
    engineered_target_association,
    load_feature_engineering_inputs,
    load_optional_safe_context,
    validate_catalog_coverage,
    validate_engineered_dataset,
    validate_input_alignment,
    write_feature_engineering_manifest,
)
from mini_tpo.feature_sets import FEATURES_ENGINEERED_CORE, FEATURES_ENGINEERED_OPTIONAL
from mini_tpo.paths import PROJECT_ROOT
from mini_tpo.modeling import run_modeling_workflow
from mini_tpo.optimization import run_optimization_workflow
from mini_tpo.reporting import executive_findings_table
from mini_tpo.support_analysis import (
    add_discount_band,
    build_discount_support,
    build_duration_support,
    build_sku_chain_support,
    build_temporal_profile,
    baseline_impact_summary,
    discount_roi_summary,
    discount_uplift_summary,
    duration_roi_summary,
    duration_uplift_summary,
    elasticity_discount_uplift_tables,
    floor_summary,
    roi_distribution_audit,
    roi_tail_tables,
    secondary_promo_summary,
    sku_history_profile,
)


def _to_markdown_table(records) -> str:
    if records.empty:
        return ""
    cols = list(records.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in records.iterrows():
        values = [str(row[col]).replace("|", "/") for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_preparation() -> dict:
    cfg = load_config()
    raw = read_raw_data(cfg)
    clean_full, modeling = clean_data(raw, cfg)

    interim_path = PROJECT_ROOT / cfg["project"]["interim_full"]
    processed_path = PROJECT_ROOT / cfg["project"]["processed_modeling"]
    tables_dir = PROJECT_ROOT / cfg["project"]["tables_dir"]
    reports_dir = PROJECT_ROOT / "reports"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    interim_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    clean_full.to_parquet(interim_path, index=False)
    modeling.to_parquet(processed_path, index=False)
    clean_full.to_csv(interim_path.with_suffix(".csv"), index=False)
    modeling.to_csv(processed_path.with_suffix(".csv"), index=False)

    validation = validation_summary(raw, cfg)
    validation.to_csv(tables_dir / "initial_validation_summary.csv", index=False)
    leakage_classification().to_csv(tables_dir / "data_dictionary_enriched.csv", index=False)
    cleaning_log = build_cleaning_log(raw, clean_full, modeling)
    cleaning_log.to_csv(tables_dir / "data_cleaning_log.csv", index=False)

    work = add_discount_band(clean_full)
    build_sku_chain_support(
        work,
        cfg["business_rules"].get("uplift_floor_value", 0.05),
        cfg["business_rules"].get("support_thresholds"),
    ).to_csv(tables_dir / "support_sku_chain.csv", index=False)
    build_discount_support(work).to_csv(tables_dir / "support_discount_by_sku_chain.csv", index=False)
    build_duration_support(work).to_csv(tables_dir / "support_duration_by_sku_chain.csv", index=False)
    roi_distribution_audit(work).to_csv(tables_dir / "roi_distribution_audit.csv", index=False)
    top_roi, bottom_roi = roi_tail_tables(work)
    top_roi.to_csv(tables_dir / "top_10_roi.csv", index=False)
    bottom_roi.to_csv(tables_dir / "bottom_10_roi.csv", index=False)
    feature_availability_audit().to_csv(tables_dir / "feature_availability_audit.csv", index=False)
    floor_summary(work, ["id_material"]).to_csv(tables_dir / "uplift_floor_by_sku.csv", index=False)
    floor_summary(work, ["subcadena"]).to_csv(tables_dir / "uplift_floor_by_chain.csv", index=False)
    floor_summary(work, ["banda_descuento_opt"]).to_csv(tables_dir / "uplift_floor_by_discount_band.csv", index=False)
    month_floor = work.assign(mes=work["fecha_inicio_tanda"].dt.to_period("M").astype(str))
    floor_summary(month_floor, ["mes"]).to_csv(tables_dir / "uplift_floor_by_time.csv", index=False)
    floor_summary(work, ["des_marca"]).to_csv(tables_dir / "uplift_floor_by_brand.csv", index=False)
    floor_summary(work, ["duracion_dias"]).to_csv(tables_dir / "uplift_floor_by_duration.csv", index=False)
    floor_summary(work, ["flag_secundario"]).to_csv(tables_dir / "uplift_floor_by_secondary_flag.csv", index=False)
    build_temporal_profile(work, "M").to_csv(tables_dir / "temporal_profile_monthly.csv", index=False)
    build_temporal_profile(work, "Q").to_csv(tables_dir / "temporal_profile_quarterly.csv", index=False)
    sku_history_profile(work).to_csv(tables_dir / "sku_history_profile.csv", index=False)
    build_cardinality_profile(work, cfg["business_rules"].get("near_constant_threshold", 0.95)).to_csv(tables_dir / "cardinality_and_variance_profile.csv", index=False)
    redundancy_audit(work).to_csv(tables_dir / "redundancy_audit.csv", index=False)
    price_variation_by_sku(work).to_csv(tables_dir / "price_variation_by_sku.csv", index=False)
    model_role_summary().to_csv(tables_dir / "model_role_summary.csv", index=False)

    commercial_tables = {
        "discount_uplift_summary": discount_uplift_summary(modeling),
        "discount_roi_summary": discount_roi_summary(modeling),
        "duration_uplift_summary": duration_uplift_summary(modeling),
        "duration_roi_summary": duration_roi_summary(modeling),
        "secondary_promo_summary": secondary_promo_summary(modeling),
        "baseline_impact_summary": baseline_impact_summary(modeling),
    }
    elasticity_uplift, elasticity_counts = elasticity_discount_uplift_tables(modeling)
    commercial_tables["elasticity_discount_uplift_matrix"] = elasticity_uplift
    commercial_tables["elasticity_discount_count_matrix"] = elasticity_counts
    for name, table in commercial_tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    safe_paths = export_model_artifacts(modeling, processed_dir)

    findings = executive_findings_table(clean_full)
    eda_report_path = PROJECT_ROOT / cfg["project"]["eda_report"]
    eda_report_path.write_text(
        "# EDA Data Quality Report\n\n"
        "Este reporte resume calidad, soporte y relaciones comerciales descriptivas. No establece causalidad.\n\n"
        f"- Registros raw: {len(raw)}\n"
        f"- Registros clean full: {len(clean_full)}\n"
        f"- Registros modeling dentro del dominio: {len(modeling)}\n"
        f"- Registros fuera del dominio operativo: {int(clean_full['flag_fuera_dominio_optimizacion'].sum())}\n"
        f"- Faltantes en flag_secundario: {int(clean_full['flag_secundario_missing'].sum())}\n\n"
        "## Supuesto sobre ROI\n\n"
        "Para esta prueba se asume que `roi` es el KPI oficial calculado por Alicorp y se acepta como target "
        "valido. Como su formula interna no esta disponible, no se realiza una descomposicion contable ni se "
        "atribuyen efectos causales a sus componentes.\n\n"
        "## Outputs comerciales\n\n"
        "Se generaron resumenes de descuento, duracion, elasticidad, volumen base y promocion secundaria en "
        "`reports/tables/`. Los extremos se conservan en las tablas.\n\n"
        "## Hallazgos ejecutivos\n\n"
        + _to_markdown_table(findings)
        + "\n",
        encoding="utf-8",
    )
    preparation_report_path = PROJECT_ROOT / cfg["project"]["preparation_report"]
    preparation_report_path.write_text(
        "# Data Preparation Report\n\n"
        f"- Raw: {len(raw)} filas, {raw.shape[1]} columnas.\n"
        f"- Clean full: {len(clean_full)} filas, {clean_full.shape[1]} columnas.\n"
        f"- Modeling domain: {len(modeling)} filas, {modeling.shape[1]} columnas.\n"
        f"- Features seguras: {len(modeling)} filas y {len(FEATURES_MODEL_BASE) + 1} columnas incluyendo `row_id`.\n"
        f"- Targets: `{TARGET_UPLIFT}` y `{TARGET_ROI}`.\n\n"
        "## Supuestos de disponibilidad\n\n"
        "`volumen_base_sem` y `elasticidad_estimada` se aceptan como estimaciones disponibles antes de la "
        "tanda. `roi` se acepta como KPI oficial y target postpromocion; nunca se usa como predictor.\n\n"
        "## Artifacts\n\n"
        "Se exportaron features, targets e indice separados, junto con un feature manifest y el log de "
        "transformaciones. Las variables de auditoria permanecen fuera de las features candidatas.\n",
        encoding="utf-8",
    )
    paths = {
        "interim_full": interim_path,
        "processed_modeling": processed_path,
        "data_dictionary": tables_dir / "data_dictionary_enriched.csv",
        "cleaning_log": tables_dir / "data_cleaning_log.csv",
        "eda_report": eda_report_path,
        "preparation_report": preparation_report_path,
        **safe_paths,
    }
    paths.update({name: tables_dir / f"{name}.csv" for name in commercial_tables})
    return {
        "paths": paths,
        "counts": {
            "raw_rows": len(raw),
            "clean_full_rows": len(clean_full),
            "modeling_rows": len(modeling),
            "outside_domain_rows": int(clean_full["flag_fuera_dominio_optimizacion"].sum()),
        },
        "dataset_shapes": {
            "raw": raw.shape,
            "clean_full": clean_full.shape,
            "modeling": modeling.shape,
            "model_features_safe": (len(modeling), len(FEATURES_MODEL_BASE) + 1),
        },
        "validation_summary": validation,
        "cleaning_log": cleaning_log,
        "feature_lists": {
            "FEATURES_MODEL_BASE": FEATURES_MODEL_BASE,
            "FEATURES_MODEL_MINIMAL": FEATURES_MODEL_MINIMAL,
            "FEATURES_MODEL_OPTIONAL": FEATURES_MODEL_OPTIONAL,
            "TARGET_UPLIFT": TARGET_UPLIFT,
            "TARGET_ROI": TARGET_ROI,
            "LEAKAGE_COLUMNS": LEAKAGE_COLUMNS,
        },
    }


def run_feature_engineering() -> dict:
    """Build feature artifacts from the immutable safe inputs of phase 02."""
    cfg = load_config()
    project = cfg["project"]
    tables_dir = PROJECT_ROOT / project["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    features, targets, index = load_feature_engineering_inputs(cfg)
    input_validation = validate_input_alignment(features, targets, index)

    core = build_engineered_core(features, index, cfg)
    optional_context = load_optional_safe_context(cfg, index)
    optional = build_engineered_optional(core, features, index, optional_context)
    core_validation = validate_engineered_dataset(
        core, features["row_id"], FEATURES_ENGINEERED_CORE
    )
    optional_validation = validate_engineered_dataset(
        optional,
        features["row_id"],
        [*FEATURES_ENGINEERED_CORE, *FEATURES_ENGINEERED_OPTIONAL],
    )
    validation = pd.concat(
        [
            input_validation.assign(dataset="inputs"),
            core_validation.assign(dataset="core"),
            optional_validation.assign(dataset="optional_extended"),
        ],
        ignore_index=True,
    )[["dataset", "check", "passed", "detail"]]

    catalog = build_feature_catalog()
    validate_catalog_coverage(core, optional, catalog)
    compatibility = build_optimizer_compatibility(catalog)
    correlation = engineered_feature_correlation(optional)
    target_association = engineered_target_association(core, targets)
    redundancy = engineered_feature_redundancy(
        correlation,
        cfg["feature_engineering"].get("high_correlation_threshold", 0.90),
    )

    paths = {
        "engineered_core": PROJECT_ROOT / project["engineered_core"],
        "engineered_optional": PROJECT_ROOT / project["engineered_optional"],
        "feature_engineering_manifest": PROJECT_ROOT / project["feature_engineering_manifest"],
        "feature_engineering_catalog": tables_dir / "feature_engineering_catalog.csv",
        "feature_optimizer_compatibility": tables_dir / "feature_optimizer_compatibility.csv",
        "engineered_feature_correlation": tables_dir / "engineered_feature_correlation.csv",
        "engineered_feature_redundancy": tables_dir / "engineered_feature_redundancy.csv",
        "engineered_feature_target_association": tables_dir / "engineered_feature_target_association.csv",
        "feature_engineering_validation": tables_dir / "feature_engineering_validation.csv",
        "feature_engineering_report": PROJECT_ROOT / project["feature_engineering_report"],
    }
    paths["engineered_core"].parent.mkdir(parents=True, exist_ok=True)
    core.to_parquet(paths["engineered_core"], index=False)
    optional.to_parquet(paths["engineered_optional"], index=False)
    catalog.to_csv(paths["feature_engineering_catalog"], index=False)
    compatibility.to_csv(paths["feature_optimizer_compatibility"], index=False)
    correlation.to_csv(paths["engineered_feature_correlation"], index=True)
    redundancy.to_csv(paths["engineered_feature_redundancy"], index=False)
    target_association.to_csv(paths["engineered_feature_target_association"], index=False)
    validation.to_csv(paths["feature_engineering_validation"], index=False)

    manifest = create_feature_engineering_manifest(core, optional, catalog, validation)
    write_feature_engineering_manifest(manifest, paths["feature_engineering_manifest"])
    paths["feature_engineering_report"].write_text(
        "# Feature Engineering Report\n\n"
        "Fase de construccion de predictores prepromocion. No se entrenaron modelos, encoders ni "
        "escaladores, y no se construyo el optimizador.\n\n"
        f"- Filas de entrada: {len(features)}\n"
        f"- Artifact core: {core.shape[0]} filas y {core.shape[1]} columnas, incluyendo `row_id`.\n"
        f"- Artifact opcional extendido: {optional.shape[0]} filas y {optional.shape[1]} columnas.\n"
        f"- Features core: {len(FEATURES_ENGINEERED_CORE)}\n"
        f"- Features opcionales: {len(FEATURES_ENGINEERED_OPTIONAL)}\n"
        f"- Controles ejecutados: {len(validation)}; fallidos: {int((~validation['passed']).sum())}.\n\n"
        "## Logica de negocio\n\n"
        "`volumen_base_tanda` alinea el baseline semanal con la duracion. Las interacciones de "
        "descuento con elasticidad y duracion representan presion promocional, pero no inversion "
        "real ni causalidad. Las transformaciones logaritmicas reducen asimetria sin eliminar los "
        "valores originales.\n\n"
        "## Guardrails\n\n"
        "- Targets, resultados postpromocion y variables de auditoria estan excluidos.\n"
        "- La fecha original permanece en el indice; solo se exportan derivados de calendario.\n"
        "- `sku_cadena` y tendencia temporal quedan en sensibilidad, no en core.\n"
        "- Promedios historicos de uplift y ROI no se materializan: requeriran shift y tratamiento fold-aware.\n"
        "- Encoders y escaladores deberan ajustarse dentro de cada fold temporal.\n"
        "- Un futuro uplift predicho para ROI debera generarse out-of-fold.\n\n"
        "## Compatibilidad con optimizacion\n\n"
        "Para cada candidato deben recalcularse las features marcadas como dependientes del descuento "
        "o de la duracion en `feature_optimizer_compatibility.csv`.\n",
        encoding="utf-8",
    )
    return {
        "paths": paths,
        "dimensions": {
            "safe_inputs": features.shape,
            "core": core.shape,
            "optional_extended": optional.shape,
        },
        "validation_summary": validation,
        "feature_sets": {
            "core": FEATURES_ENGINEERED_CORE,
            "optional": FEATURES_ENGINEERED_OPTIONAL,
        },
    }


def _write_modeling_report(outputs: dict, path: Path) -> None:
    scorecard = outputs["scorecard"]
    selected = scorecard[scorecard["rol_seleccion"].isin(["champion", "challenger"])]
    final_uplift = outputs["final_metrics_uplift"]
    final_roi = outputs["final_metrics_roi"]
    path.write_text(
        "# Modeling Comparison Report\n\n"
        "La seleccion se fijo con validacion temporal expansiva antes de abrir el test final. "
        "Los resultados son predictivos y descriptivos; no identifican efectos causales.\n\n"
        f"- Desarrollo: {outputs['development']['fecha_inicio_tanda'].min().date()} a "
        f"{outputs['development']['fecha_inicio_tanda'].max().date()} ({len(outputs['development'])} filas).\n"
        f"- Test final aislado: {outputs['final_test']['fecha_inicio_tanda'].min().date()} a "
        f"{outputs['final_test']['fecha_inicio_tanda'].max().date()} ({len(outputs['final_test'])} filas).\n"
        "- Familias: Ridge, HistGradientBoosting y ExtraTrees.\n"
        "- ROI de dos etapas: solo uplift OOF; el artifact productivo conserva ROI directo salvo evidencia "
        "consistente y una implementacion completa del encadenamiento.\n\n"
        "## Seleccion de desarrollo\n\n"
        + _to_markdown_table(selected)
        + "\n\n## Test final uplift\n\n"
        + _to_markdown_table(final_uplift)
        + "\n\n## Test final ROI\n\n"
        + _to_markdown_table(final_roi)
        + "\n\n## Guardrails para optimizacion\n\n"
        "- Recalcular todas las features dependientes de descuento y duracion por escenario.\n"
        "- Restringir recomendaciones al soporte historico local SKU por cadena.\n"
        "- Incorporar intervalos de incertidumbre y evitar explotar picos artificiales.\n"
        "- No cambiar champion o challenger despues de observar este test.\n",
        encoding="utf-8",
    )


def run_modeling_comparison() -> dict:
    """Execute phase 04 once and persist its auditable outputs."""
    cfg = load_config()
    outputs = run_modeling_workflow(PROJECT_ROOT, cfg)
    tables_dir = PROJECT_ROOT / cfg["project"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    table_outputs = {
        "modeling_temporal_splits.csv": outputs["split_summary"],
        "modeling_final_test_coverage.csv": outputs["test_coverage"],
        "baseline_metrics.csv": outputs["baseline_metrics"],
        "modeling_ablation_results.csv": outputs["ablation"],
        "cv_metrics_uplift.csv": outputs["cv_metrics_uplift"],
        "cv_metrics_roi.csv": outputs["cv_metrics_roi"],
        "cv_metrics_by_fold.csv": outputs["cv_metrics_by_fold"],
        "cv_metrics_segmented.csv": outputs["segment_metrics"],
        "cv_metrics_by_sku.csv": outputs["metrics_by_sku"],
        "cv_metrics_by_chain.csv": outputs["metrics_by_chain"],
        "cv_metrics_by_support.csv": outputs["metrics_by_support"],
        "model_hyperparameter_search.csv": outputs["hyperparameter_search"],
        "model_selection_scorecard.csv": outputs["scorecard"],
        "roi_direct_vs_two_stage.csv": outputs["two_stage_comparison"],
        "model_uncertainty_summary.csv": outputs["uncertainty"],
        "model_statistical_comparison.csv": outputs["statistical_comparison"],
        "final_test_metrics_uplift.csv": outputs["final_metrics_uplift"],
        "final_test_metrics_roi.csv": outputs["final_metrics_roi"],
        "model_feature_importance_uplift.csv": outputs["importance_uplift"],
        "model_feature_importance_roi.csv": outputs["importance_roi"],
        "model_response_curve_diagnostics.csv": outputs["response_diagnostics"],
    }
    table_paths = {}
    for filename, table in table_outputs.items():
        table_path = tables_dir / filename
        table.to_csv(table_path, index=False)
        table_paths[filename] = table_path

    parquet_outputs = {
        "oof_uplift": outputs["selected_oof_uplift"],
        "oof_roi": outputs["selected_oof_roi"],
        "final_test_uplift": outputs["final_predictions_uplift"],
        "final_test_roi": outputs["final_predictions_roi"],
    }
    parquet_paths = {}
    for config_key, table in parquet_outputs.items():
        artifact_path = PROJECT_ROOT / cfg["project"][config_key]
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        table.sort_values(["fecha_inicio_tanda", "row_id", "rol"]).to_parquet(
            artifact_path, index=False
        )
        parquet_paths[config_key] = artifact_path

    report_path = PROJECT_ROOT / cfg["project"]["modeling_report"]
    _write_modeling_report(outputs, report_path)
    outputs["paths"] = {
        **table_paths,
        **parquet_paths,
        "modeling_report": report_path,
        "model_registry": outputs["registry_path"],
        **{f"model_{target}_{role}": path for (target, role), path in outputs["model_paths"].items()},
    }
    return outputs


def run_optimization() -> dict:
    """Execute the final Mini-TPO optimization phase and persist its outputs."""
    cfg = load_config()
    outputs = run_optimization_workflow(PROJECT_ROOT, cfg)
    tables_dir = PROJECT_ROOT / cfg["project"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "selected_optimization_cases.csv": outputs["cases"],
        "optimization_contexts.csv": outputs["contexts"],
        "optimal_recommendations.csv": outputs["recommendations"],
        "mathematical_roi_optima.csv": outputs["mathematical"],
        "profitable_growth_alternatives.csv": outputs["growth"],
        "optimization_pareto_frontier.csv": outputs["pareto"],
        "optimization_sensitivity.csv": outputs["sensitivity"],
        "optimization_grid_summary.csv": outputs["grid_summary"],
    }
    paths = {}
    for filename, table in tables.items():
        path = tables_dir / filename
        table.to_csv(path, index=False)
        paths[filename] = path
    scenario_path = PROJECT_ROOT / cfg["project"]["optimization_scenarios"]
    outputs["scenarios"].sort_values(
        ["case_id", "duracion_dias", "factor_descuento"]
    ).to_parquet(scenario_path, index=False)
    paths["optimization_scenarios"] = scenario_path

    report_path = PROJECT_ROOT / cfg["project"]["optimization_report"]
    report_path.write_text(
        "# Optimization Report\n\n"
        "La seleccion de modelos se mantuvo congelada. Se crearon refits de produccion con las "
        "2,048 observaciones despues de completar la evaluacion; no hubo tuning adicional.\n\n"
        f"- Fecha de referencia: {outputs['reference_date'].date()}.\n"
        f"- Error absoluto OOF q90 uplift: {outputs['q90_uplift']:.4f}.\n"
        f"- Error absoluto OOF q90 ROI: {outputs['q90_roi']:.4f}.\n"
        f"- Tolerancia de empate ROI: {outputs['epsilon_roi']:.4f}.\n"
        "- Grilla principal: descuento 5%-40% cada 1 pp y duraciones 5, 7, 10, 14 y 21 dias.\n\n"
        "## Recomendaciones\n\n"
        + _to_markdown_table(outputs["recommendations"])
        + "\n\n## Lectura de negocio\n\n"
        "El optimo matematico maximiza ROI puntual. La recomendacion robusta exige intervalo inferior "
        "positivo, soporte local y ausencia de extrapolacion critica. La alternativa de crecimiento "
        "maximiza unidades incrementales dentro de los mismos guardrails. Ninguna prediccion es una "
        "garantia causal y las restricciones de stock, presupuesto y margen no estan disponibles.\n",
        encoding="utf-8",
    )
    paths["optimization_report"] = report_path
    paths["uplift_production"] = PROJECT_ROOT / "models/uplift_production.joblib"
    paths["roi_production"] = PROJECT_ROOT / "models/roi_production.joblib"
    paths["model_registry"] = PROJECT_ROOT / cfg["project"]["model_registry"]
    outputs["paths"] = paths
    return outputs


if __name__ == "__main__":
    outputs = run_preparation()
    for name, path in outputs["paths"].items():
        print(f"{name}: {path.relative_to(PROJECT_ROOT)}")
