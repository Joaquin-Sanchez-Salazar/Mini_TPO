# Modeling Comparison Report

La seleccion se fijo con validacion temporal expansiva antes de abrir el test final. Los resultados son predictivos y descriptivos; no identifican efectos causales.

- Desarrollo: 2024-07-01 a 2025-09-29 (1691 filas).
- Test final aislado: 2025-10-06 a 2025-12-22 (357 filas).
- Familias: Ridge, HistGradientBoosting y ExtraTrees.
- ROI de dos etapas: solo uplift OOF; el artifact productivo conserva ROI directo salvo evidencia consistente y una implementacion completa del encadenamiento.

## Seleccion de desarrollo

| mae | rmse | wape | smape | bias | r2 | mae_unidades | wape_unidades | error_abs_total_unidades | bias_unidades | n | target | candidate_id | modelo | familia | feature_set | target_transform | mae_fold_std | tiempo_entrenamiento_seg | params | worst_sku_mae | recent_sku_mae | smoothness_index | false_positive_rate | score_mae | score_stability | score_worst_sku | score_recent | score_smoothness | score_business_risk | score_interpretability | score_complexity | score_training_time | score_total | rol_seleccion | decision_fijada_antes_test | median_ae | spearman | sign_accuracy | precision_roi_positivo | recall_roi_positivo | f1_roi_positivo | precision_roi_negativo | recall_roi_negativo | f1_roi_negativo | falsos_positivos | falsos_negativos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.14048454268937957 | 0.1753084302675622 | 0.31035429239928347 | 0.38074942767356335 | 0.004567954162531816 | 0.493099940731324 | 582.0173839377467 | 0.3152330657799772 | 770591.0163335766 | 36.192003198307475 | 1324 | uplift_real | uplift_real__ridge__temporal_weekly__original__alpha100.0 | ridge | ridge | temporal_weekly | original | 0.0008628400762901053 | 0.07649869999977454 | {"alpha": 100.0} | 0.18943284588648907 | 0.1827901638663352 | 0.06511557029382768 | 0.0 | 3 | 5 | 4 | 4 | 5 | 3 | 5 | 5 | 5 | 4.1 | champion | True | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| 0.13749803142590086 | 0.17150844480846883 | 0.3037565801373102 | 0.3762901677381719 | -0.0006031941891606287 | 0.5148369052004523 | 570.4417245064408 | 0.3089634410030207 | 755264.8432465276 | 19.334198704303187 | 1324 | uplift_real | uplift_real__ridge__temporal_weekly__log1p__alpha10.0 | ridge | ridge | temporal_weekly | log1p | 0.0011669989049302428 | 0.08250500000031025 | {"alpha": 10.0} | 0.21375280363980556 | 0.20898826106621524 | 0.0988984953776736 | 0.0 | 5 | 3 | 2 | 2 | 4 | 5 | 5 | 5 | 5 | 4.0 | challenger | True | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| 0.124798203501576 | 1.0483776781646874 | nan | nan | -0.04659522029247078 | 0.5164822995505498 | nan | nan | nan | nan | 1324 | roi | roi__hist_gradient_boosting__nonlinear__original__l2_regularization3.0_learning_rate0.04_max_iter300_max_leaf_nodes9_min_samples_leaf30 | hist_gradient_boosting | hist_gradient_boosting | nonlinear | original | 0.027696668067569943 | 1.661279999999806 | {"l2_regularization": 3.0, "learning_rate": 0.04, "max_iter": 300, "max_leaf_nodes": 9, "min_samples_leaf": 30} | 0.3796083684660621 | 0.02837900222339134 | 0.1830047663757485 | 0.0022658610271903325 | 4 | 5 | 4 | 4 | 2 | 4 | 3 | 3 | 3 | 3.7499999999999996 | champion | True | 0.004671638980403514 | 0.9780953829646805 | 0.9939577039274925 | 0.9961038961038962 | 0.9961038961038962 | 0.9961038961038962 | 0.9909747292418772 | 0.9945652173913043 | 0.9927667269439421 | 3.0 | 3.0 |
| 0.1143027278359598 | 1.0423865832191954 | nan | nan | -0.04168385730223708 | 0.521992762876696 | nan | nan | nan | nan | 1324 | roi | roi__hist_gradient_boosting__nonlinear__original__l2_regularization1.0_learning_rate0.05_max_iter220_max_leaf_nodes15_min_samples_leaf20 | hist_gradient_boosting | hist_gradient_boosting | nonlinear | original | 0.029147813697652857 | 1.8170439000000442 | {"l2_regularization": 1.0, "learning_rate": 0.05, "max_iter": 220, "max_leaf_nodes": 15, "min_samples_leaf": 20} | 0.3700309270442323 | 0.008816652567966004 | 0.3182592143484358 | 0.004531722054380665 | 5 | 3 | 5 | 5 | 1 | 3 | 3 | 3 | 3 | 3.7000000000000006 | challenger | True | 0.00609760109291102 | 0.9783308240543993 | 0.9901812688821753 | 0.9896507115135834 | 0.9935064935064936 | 0.9915748541801686 | 0.9909255898366606 | 0.9891304347826086 | 0.9900271985494107 | 6.0 | 5.0 |

## Test final uplift

| mae | rmse | wape | smape | bias | r2 | mae_unidades | wape_unidades | error_abs_total_unidades | bias_unidades | n | target | rol | candidate_id | modelo | familia | feature_set | decision_fijada_antes_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1479991778832942 | 0.178719657742739 | 0.34482882141934323 | 0.4372054749801502 | 0.019602840592850387 | 0.4778954468414267 | 669.6888696712984 | 0.3575021064446717 | 239078.92647265355 | 118.04940011629617 | 357 | uplift_real | champion | uplift_real__ridge__temporal_weekly__original__alpha100.0 | ridge | ridge | temporal_weekly | True |
| 0.1437589187086655 | 0.1735585764703599 | 0.3349492829339824 | 0.43363886609103847 | 0.0010848797540341112 | 0.5076147992129639 | 643.4045134696845 | 0.3434706462634218 | 229695.41130867737 | 29.069616418165563 | 357 | uplift_real | challenger | uplift_real__ridge__temporal_weekly__log1p__alpha10.0 | ridge | ridge | temporal_weekly | True |

## Test final ROI

| mae | rmse | median_ae | bias | r2 | spearman | sign_accuracy | precision_roi_positivo | recall_roi_positivo | f1_roi_positivo | precision_roi_negativo | recall_roi_negativo | f1_roi_negativo | falsos_positivos | falsos_negativos | false_positive_rate | n | target | rol | candidate_id | modelo | familia | feature_set | decision_fijada_antes_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.07966204610190716 | 0.824642752423264 | 0.0038494463751119756 | -0.015400943737731617 | 0.6040928052045418 | 0.9965620945934415 | 0.9943977591036415 | 1.0 | 0.9946236559139785 | 0.9973045822102425 | 0.9883720930232558 | 1.0 | 0.9941520467836257 | 0 | 1 | 0.0 | 357 | roi | champion | roi__hist_gradient_boosting__nonlinear__original__l2_regularization3.0_learning_rate0.04_max_iter300_max_leaf_nodes9_min_samples_leaf30 | hist_gradient_boosting | hist_gradient_boosting | nonlinear | True |
| 0.07690383651720038 | 0.8236838516699763 | 0.00413552046862703 | -0.013564980259436676 | 0.6050129975777558 | 0.99650103663841 | 0.9915966386554622 | 1.0 | 0.989247311827957 | 0.9945945945945946 | 0.9826589595375722 | 1.0 | 0.9912536443148688 | 0 | 2 | 0.0 | 357 | roi | challenger | roi__hist_gradient_boosting__nonlinear__original__l2_regularization1.0_learning_rate0.05_max_iter220_max_leaf_nodes15_min_samples_leaf20 | hist_gradient_boosting | hist_gradient_boosting | nonlinear | True |

## Guardrails para optimizacion

- Recalcular todas las features dependientes de descuento y duracion por escenario.
- Restringir recomendaciones al soporte historico local SKU por cadena.
- Incorporar intervalos de incertidumbre y evitar explotar picos artificiales.
- No cambiar champion o challenger despues de observar este test.
