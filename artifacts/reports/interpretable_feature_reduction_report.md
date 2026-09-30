# Reporte de reducción de features interpretable — AFPDB/PAF MEDICON 2026
_Generado: 2026-04-24 17:48_

## 1. Features redundantes (|r| >= 0.90 Spearman)

- **Grupo 1**: mean_rr, median_rr
  Conservar: **mean_rr** | Eliminar: median_rr
- **Grupo 2**: rmssd, cv_rr, sd2, sd1, sdsd, rr_diff_std, std_rr
  Conservar: **rmssd** | Eliminar: cv_rr, sd2, sd1, sdsd, rr_diff_std, std_rr

### ¿El análisis delta anterior (AUC=0.397) era metodológicamente válido?

**NO.** El análisis delta intra-par produce 25 vectores Δ(pre−far), uno por par. Todos los vectores corresponden al mismo tipo de evento (el cambio fisiológico que precede al PAF), sin una segunda clase natural. La pseudo-etiqueta 'cambio grande vs cambio pequeño' basada en la norma del vector no representa una hipótesis clínica válida. El AUC resultante (0.397) no tiene interpretación estadística ni clínica clara.

**Corrección aplicada:** solo se reportan estadísticas descriptivas por feature (mediana delta, IQR, Wilcoxon p, rank-biserial r, % pares en dirección esperada).

## 2. Features conservadas por bloque

| Bloque | Features |
|--------|---------|
| `hrv_full_actual` | mean_rr, median_rr, std_rr, mad_rr, rmssd, sdsd, cv_rr, rr_diff_mean, rr_diff_std, sd1, sd2, sd1_sd2_ratio |
| `hrv_reduced_interpretable` | rmssd, cv_rr, sd2 |
| `hrv_minimal` | rmssd, sd2 |

## 3. Impacto en AUC (LogReg, GroupKFold por pair_id)

| block                     |   n_features | use_pruning   |   auc_mean |   auc_std |
|:--------------------------|-------------:|:--------------|-----------:|----------:|
| hrv_full_actual           |           12 | False         |      0.704 |    0.0862 |
| hrv_reduced_interpretable |            3 | False         |      0.744 |    0.0742 |
| hrv_minimal               |            2 | False         |      0.744 |    0.0742 |
| hrv_full_pruned           |           12 | True          |      0.664 |    0.0862 |

## 4. Estabilidad de coeficientes LogReg (% folds con signo positivo)

### hrv_full_actual
| feature       |   mean_coef |   pct_folds_positive | sign_stable   |
|:--------------|------------:|---------------------:|:--------------|
| sd1_sd2_ratio |      0.8373 |                  1   | True          |
| rr_diff_mean  |     -0.5814 |                  0   | True          |
| mad_rr        |      0.5476 |                  1   | True          |
| sd2           |      0.343  |                  1   | True          |
| std_rr        |      0.2381 |                  1   | True          |
| median_rr     |      0.06   |                  1   | True          |
| mean_rr       |     -0.0572 |                  0   | True          |
| sd1           |     -0.0196 |                  0.6 | False         |
| rmssd         |     -0.0196 |                  0.6 | False         |
| rr_diff_std   |     -0.0195 |                  0.6 | False         |
| sdsd          |     -0.0195 |                  0.6 | False         |
| cv_rr         |      0.0085 |                  0.4 | False         |
_Coeficientes con signo estable (>80% o <20% positive): 7/12_

### hrv_reduced_interpretable
| feature   |   mean_coef |   pct_folds_positive | sign_stable   |
|:----------|------------:|---------------------:|:--------------|
| rmssd     |      0.5333 |                  1   | True          |
| sd2       |     -0.1038 |                  0.2 | True          |
| cv_rr     |      0.0478 |                  0.2 | True          |
_Coeficientes con signo estable (>80% o <20% positive): 3/3_

### hrv_full_pruned
| feature       |   mean_coef |   pct_folds_positive | sign_stable   |
|:--------------|------------:|---------------------:|:--------------|
| sd1_sd2_ratio |      0.6556 |                  1   | True          |
| mad_rr        |      0.5569 |                  1   | True          |
| rr_diff_mean  |     -0.556  |                  0   | True          |
| rmssd         |      0.3654 |                  0.8 | True          |
| mean_rr       |      0.0045 |                  0.6 | False         |
_Coeficientes con signo estable (>80% o <20% positive): 4/5_

## 5. Análisis delta intra-par (correcto)

| feature                  |   median_delta |   iqr_delta |   wilcoxon_p |   rank_biserial_r |   pct_pairs_delta_positive | direction_consistent   |
|:-------------------------|---------------:|------------:|-------------:|------------------:|---------------------------:|:-----------------------|
| mr_mad                   |         0      |      0      |       0.0396 |             0.857 |                       0.24 | False                  |
| shewhart_out_rate        |         0.0482 |      0.1339 |       0.0011 |             0.748 |                       0.84 | True                   |
| std_rr                   |        17.4534 |     52.5634 |       0.0038 |             0.643 |                       0.72 | True                   |
| sd1                      |        25.1578 |     62.3463 |       0.0042 |             0.637 |                       0.76 | True                   |
| rr_diff_std              |        35.5849 |     88.1963 |       0.0042 |             0.637 |                       0.76 | True                   |
| rmssd                    |        35.5786 |     88.171  |       0.0042 |             0.637 |                       0.76 | True                   |
| sdsd                     |        35.5849 |     88.1963 |       0.0042 |             0.637 |                       0.76 | True                   |
| cv_rr                    |         0.0222 |      0.0598 |       0.0061 |             0.612 |                       0.72 | True                   |
| sd2                      |        10.5535 |     44.9568 |       0.0125 |             0.563 |                       0.68 | True                   |
| sd1_sd2_ratio            |         0.2249 |      0.4891 |       0.0173 |             0.538 |                       0.64 | True                   |
| rr_diff_mean             |        11.2462 |     57.3361 |       0.0203 |             0.526 |                       0.76 | True                   |
| cusum_abs_max            |       173.491  |    541.701  |       0.0422 |             0.465 |                       0.6  | True                   |
| cusum_alarm_density      |         0.0983 |      0.5393 |       0.0653 |             0.422 |                       0.6  | True                   |
| mad_rr                   |         0      |     15.625  |       0.1636 |             0.373 |                       0.4  | False                  |
| longest_run_above_median |         1      |     80      |       0.5628 |             0.132 |                       0.52 | False                  |
| median_rr                |        23.4375 |     70.3125 |       0.5896 |             0.123 |                       0.52 | False                  |
| ewma_slope               |        -0.0001 |      0.0021 |       0.615  |            -0.12  |                       0.44 | False                  |
| mean_rr                  |        -0.9439 |     66.834  |       0.9464 |            -0.015 |                       0.48 | False                  |

## 6. Pairwise ranking evaluation (PC = score_pre > score_far)

| block                            |   pairwise_concordance |   auc |    f1 |
|:---------------------------------|-----------------------:|------:|------:|
| B_SPC                            |                   0.84 | 0.8   | 0.659 |
| B_SPC_pruned                     |                   0.88 | 0.784 | 0.624 |
| C_HRV_SPC                        |                   0.84 | 0.792 | 0.646 |
| C_HRV_SPC_pruned                 |                   0.84 | 0.76  | 0.613 |
| D_HRV_SPC_Poincare               |                   0.84 | 0.8   | 0.646 |
| D_HRV_SPC_Poincare_pruned        |                   0.84 | 0.776 | 0.599 |
| hrv_reduced_interpretable        |                   0.72 | 0.744 | 0.593 |
| hrv_reduced_interpretable_pruned |                   0.72 | 0.744 | 0.557 |

## 7. Recomendaciones para siguiente paso

1. **Usar `hrv_reduced_interpretable` (rmssd, cv_rr, sd2)** como bloque HRV estándar en publicaciones. Elimina colinealidad severa; coeficientes LogReg deben tener signo estable (+) para las tres features.
2. **El bloque B_SPC sigue siendo el más discriminativo** (AUC ~ 0.66–0.80 según ventana). Combinarlo con `hrv_reduced_interpretable` da C_HRV_SPC.
3. **Pairwise concordance** es la métrica más adecuada para este dataset emparejado. Un PC > 0.7 equivale a que el modelo ordena correctamente > 70% de pares (far < pre-onset en score).
4. **No introducir más features hasta reducir colinealidad** y confirmar estabilidad de signos de coeficientes.
5. **Siguiente módulo recomendado**: tuning de h/k CUSUM dentro del train fold (nested CV) usando ventana de 100 RR, que mostró AUC = 0.80.

## Riesgos metodológicos restantes

- n=25 pares: todas las estimaciones tienen IC amplio.
- CorrelationPruner usa threshold fijo (0.90); sensibilidad al threshold no evaluada.
- `pairwise_concordance` asume que ambos registros de un par siempre van al mismo fold; esto está garantizado por GroupKFold(pair_id) pero debe verificarse.
