# Auditoria pruning SPC — B_SPC_pruned
_Fecha: 2026-04-24 18:03_

## Verificacion anti-leakage
- CorrelationPruner.fit() llamado SOLO con X_train (fold de train).
- val_participates_in_pruning = **False** en todos los folds (verificado).

## Seleccion de features por fold

|   fold |   n_features_before |   n_features_after |   n_features_dropped |
|-------:|--------------------:|-------------------:|---------------------:|
|      1 |                  30 |                 22 |                    8 |
|      2 |                  30 |                 23 |                    7 |
|      3 |                  30 |                 23 |                    7 |
|      4 |                  30 |                 22 |                    8 |
|      5 |                  30 |                 22 |                    8 |

## Estabilidad de seleccion (% folds donde fue seleccionada)

| feature                   |   pct_folds_selected |
|:--------------------------|---------------------:|
| cusum_abs_max             |                  1   |
| cusum_first_alarm_pos     |                  1   |
| cusum_last_neg            |                  1   |
| cusum_neg_max             |                  1   |
| cusum_pos_max             |                  1   |
| ewma_first_alarm_time_sec |                  1   |
| ewma_last                 |                  1   |
| mr_median                 |                  1   |
| ewma_max_abs              |                  1   |
| ewma_slope                |                  1   |
| longest_run_above_median  |                  1   |
| longest_run_below_median  |                  1   |
| mr_mad                    |                  1   |
| mr_max                    |                  1   |
| mr_mean                   |                  1   |
| shewhart_out_rate         |                  1   |
| mr_out_count              |                  1   |
| n_runs_rule_violations    |                  1   |
| shewhart_first_alarm_pos  |                  1   |
| trend_run_up_max          |                  1   |
| trend_run_down_max        |                  1   |
| cusum_last_pos            |                  0.8 |
| shewhart_max_abs_z        |                  0.6 |
