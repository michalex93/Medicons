# v1 — Baseline interpretable congelado
_Fecha: 2026-04-24 18:02_

## Features conservadas por bloque

| Bloque | Features |
|--------|---------|
| hrv_full_actual | rmssd, cv_rr, sd2, sd1, sdsd, ... (12 features) |
| hrv_reduced_interpretable | **rmssd, cv_rr, sd2** |

## Features removidas por colinealidad (|r| >= 0.90)
sdsd, rr_diff_std, rr_diff_mean, sd1 → redundantes con **rmssd**
std_rr → redundante con **cv_rr**

## Resultados AUC baseline

| config                    |   n_features | use_pruning   |    auc |
|:--------------------------|-------------:|:--------------|-------:|
| hrv_reduced_interpretable |            3 | False         | 0.6544 |
| B_SPC                     |           30 | False         | 0.776  |
| B_SPC_pruned              |           30 | True          | 0.752  |
| C_HRV_SPC                 |           33 | False         | 0.7664 |
| D_HRV_SPC_Poincare        |           36 | False         | 0.776  |

## Pairwise Concordance por bloque

| Config | PC medio |
|--------|---------|
| B_SPC | 0.84 |
| B_SPC_pruned | 0.88 |
| C_HRV_SPC | 0.84 |
| D_HRV_SPC_Poincare | 0.84 |
| hrv_reduced_interpretable | 0.72 |

## Análisis delta intra-par (far_baseline vs pre_onset)

> y=0 = **far_baseline** (no 'normal', no 'healthy').
> y=1 = **pre_onset**.

### Interpretación fisiológica:
El pre_onset NO se diferencia principalmente por mean_rr (sin señal, p=0.946),
sino por **incremento de variabilidad RR** (rmssd +35 ms, p=0.004) e
**inestabilidad SPC** (shewhart_out_rate +0.048, p=0.001, r_rb=0.748).
El pre_onset se caracteriza por mayor irregularidad latido a latido,
no por aceleración/desaceleración cardíaca global.

| feature                       |   median_delta |   wilcoxon_p |   rank_biserial_r |   pct_pairs_delta_positive |
|:------------------------------|---------------:|-------------:|------------------:|---------------------------:|
| mr_mad                        |         0      |       0.0396 |             0.857 |                      0.24  |
| n_runs_rule_violations        |         0      |       0.0335 |             0.806 |                      0.28  |
| shewhart_out_rate             |         0.0482 |       0.0011 |             0.748 |                      0.84  |
| shewhart_out_count            |        93      |       0.0013 |             0.735 |                      0.8   |
| mr_median                     |         0      |       0.0303 |             0.692 |                      0.36  |
| rmssd                         |        35.5786 |       0.0042 |             0.637 |                      0.76  |
| sd1                           |        25.1578 |       0.0042 |             0.637 |                      0.76  |
| cv_rr                         |         0.0222 |       0.0061 |             0.612 |                      0.72  |
| sd2                           |        10.5535 |       0.0125 |             0.563 |                      0.68  |
| sd1_sd2_ratio                 |         0.2249 |       0.0173 |             0.538 |                      0.64  |
| mr_max                        |       187.5    |       0.0214 |             0.526 |                      0.64  |
| mr_mean                       |        11.2462 |       0.0203 |             0.526 |                      0.76  |
| ewma_max_abs                  |         2.8276 |       0.0255 |             0.508 |                      0.68  |
| shewhart_max_abs_z            |         4.5856 |       0.0422 |             0.465 |                      0.64  |
| cusum_abs_max                 |       173.491  |       0.0422 |             0.465 |                      0.6   |
| cusum_neg_max                 |        89.3286 |       0.0451 |             0.458 |                      0.64  |
| cusum_last_neg                |         4.8047 |       0.0793 |             0.448 |                      0.56  |
| ewma_first_alarm_time_sec     |       -19.3398 |       0.0646 |            -0.433 |                      0.333 |
| mr_out_count                  |        31      |       0.0596 |             0.431 |                      0.68  |
| cusum_alarm_density           |         0.0983 |       0.0653 |             0.422 |                      0.6   |
| cusum_alarm_count             |        79      |       0.0653 |             0.422 |                      0.64  |
| cusum_time_in_alarm           |       176.779  |       0.0693 |             0.415 |                      0.6   |
| ewma_alarm_count              |        94      |       0.1267 |             0.354 |                      0.6   |
| shewhart_first_alarm_pos      |       -59      |       0.1908 |            -0.305 |                      0.4   |
| shewhart_first_alarm_time_sec |       -49.3164 |       0.2643 |            -0.267 |                      0.417 |
| cusum_last_pos                |         0      |       0.3219 |             0.247 |                      0.48  |
| ewma_last                     |        -0.2287 |       0.4578 |            -0.175 |                      0.44  |
| trend_run_down_max            |         0      |       0.5449 |             0.158 |                      0.4   |
| longest_run_below_median      |       -10      |       0.4926 |            -0.157 |                      0.48  |
| cusum_pos_max                 |         2.3316 |       0.5424 |             0.145 |                      0.52  |
| longest_run_above_median      |         1      |       0.5628 |             0.132 |                      0.52  |
| ewma_slope                    |        -0.0001 |       0.615  |            -0.12  |                      0.44  |
| cusum_first_alarm_pos         |       -40      |       0.893  |            -0.031 |                      0.4   |
| trend_run_up_max              |         0      |       0.9364 |             0.022 |                      0.36  |
| cusum_first_alarm_time_sec    |       -17.6055 |       0.9441 |             0.02  |                      0.458 |

## Nota metodológica
IC 95% pendientes (Fase 2). No declarar mejora definitiva hasta verificar IC.
