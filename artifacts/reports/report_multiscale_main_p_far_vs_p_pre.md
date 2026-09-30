# Reporte metodológico: AFPDB multi-escala (main_p_far_vs_p_pre)
_Generado: 2026-04-24 17:10_

## Pregunta de investigación
¿A partir solo de intervalos RR y derivados explicables se puede distinguir, dentro de sujetos con PAF documentada, el segmento **far** (p impar, y=0) frente al segmento **pre-onset** (p par, y=1)?

## Metodología

### Diseño
- **Tipo**: caso–control emparejado intra-sujeto PAF (par p_impar / p_par).
- **Unidad mínima de agrupación**: `pair_id` (25 pares, n=50 registros).
- **Análisis secundario**: p\* vs n\* (discriminación de grupo, NO pre-onset).

### Ventanas evaluadas
| Tipo | Tamaño |
|------|--------|
| rr_count | 50, 100, 128, 200 RR |
| time | 300 s (5 min) |
| full_record | ~1800 s (30 min) |

**PROHIBIDO**: ventanas de 30 segundos en el análisis principal.

### Cartas SPC
- Shewhart: L = 3.0σ
- CUSUM bilateral: h = 7.0, k = 0.5
- EWMA: λ = 0.2, L = 2.7
- Moving Range: UCL = D4 · mean(MR), D4 = 3.267
- Runs rules (Western Electric, 4 reglas simplificadas)
- Baseline SPC: primeros 200 intervalos RR del registro.
  → Esto garantiza que los límites de control no usen información futura del mismo registro.

### Validación
- `GroupKFold(5)` agrupando por `pair_id`.
- Ningún `pair_id` aparece simultáneamente en train y val.
- Escalado, imputación y selección de features dentro de cada fold.
- Umbral de decisión fijo = 0.5 (sin optimización sobre test).
- `random_state = 42`.

### lead_time (solo y=1)
- Se asume que el PAF onset ocurre al final del registro de 30 min (1800 s).
- `lead_time_sec = 1800 - primera_alarma_absoluta`.
- Para y=0: cualquier alarma es falsa alarma → se reporta `false_alarms_per_hour`.

### Features de ectopia
> **ADVERTENCIA**: Las variables `ectopy_like_*` son patrones COMPATIBLES con ectopia inferidos desde RR únicamente. NO son diagnóstico confirmado de PAC/APC ni análisis morfológico.

## Resumen de resultados (AUC medio por ventana × bloque, LogReg)

| window_type   |   window_size | block                     | model   |   auc_mean |   auc_std |   balanced_accuracy_mean |   balanced_accuracy_std |   sensitivity_mean |   sensitivity_std |   specificity_mean |   specificity_std |   precision_mean |   precision_std |   f1_mean |   f1_std |   mcc_mean |   mcc_std |
|:--------------|--------------:|:--------------------------|:--------|-----------:|----------:|-------------------------:|------------------------:|-------------------:|------------------:|-------------------:|------------------:|-----------------:|----------------:|----------:|---------:|-----------:|----------:|
| full_record   |          1800 | A_HRV                     | LogReg  |      0.656 | 0.118659  |                     0.56 |               0.0547723 |               0.44 |          0.260768 |               0.68 |         0.268328  |         0.647619 |       0.20852   |  0.466234 | 0.156396 |  0.153954  |  0.148189 |
| full_record   |          1800 | B_SPC                     | LogReg  |      0.8   | 0.129615  |                     0.7  |               0.158114  |               0.68 |          0.334664 |               0.72 |         0.109545  |         0.679524 |       0.130753  |  0.658961 | 0.242806 |  0.41588   |  0.330577 |
| full_record   |          1800 | C_HRV_SPC                 | LogReg  |      0.744 | 0.128374  |                     0.68 |               0.148324  |               0.64 |          0.296648 |               0.72 |         0.109545  |         0.67     |       0.129314  |  0.637749 | 0.227942 |  0.366599  |  0.303326 |
| full_record   |          1800 | D_HRV_SPC_Poincare        | LogReg  |      0.76  | 0.126491  |                     0.68 |               0.148324  |               0.64 |          0.296648 |               0.72 |         0.109545  |         0.67     |       0.129314  |  0.637749 | 0.227942 |  0.366599  |  0.303326 |
| full_record   |          1800 | E_HRV_SPC_Poincare_Ectopy | LogReg  |      0.792 | 0.12775   |                     0.66 |               0.151658  |               0.64 |          0.296648 |               0.68 |         0.178885  |         0.650952 |       0.13673   |  0.625628 | 0.223561 |  0.328593  |  0.308661 |
| rr_count      |            50 | A_HRV                     | LogReg  |      0.464 | 0.0963328 |                     0.48 |               0.109545  |               0.48 |          0.228035 |               0.48 |         0.228035  |         0.48     |       0.135794  |  0.463333 | 0.152934 | -0.04      |  0.235837 |
| rr_count      |            50 | B_SPC                     | LogReg  |      0.584 | 0.166373  |                     0.54 |               0.114018  |               0.48 |          0.228035 |               0.6  |         0         |         0.52     |       0.126051  |  0.493232 | 0.180345 |  0.0780061 |  0.23657  |
| rr_count      |            50 | C_HRV_SPC                 | LogReg  |      0.664 | 0.145877  |                     0.56 |               0.194936  |               0.48 |          0.334664 |               0.64 |         0.0894427 |         0.493333 |       0.303132  |  0.483232 | 0.314809 |  0.10165   |  0.425736 |
| rr_count      |            50 | D_HRV_SPC_Poincare        | LogReg  |      0.672 | 0.136821  |                     0.56 |               0.167332  |               0.52 |          0.334664 |               0.6  |         0.244949  |         0.516667 |       0.302765  |  0.501865 | 0.292714 |  0.106943  |  0.378738 |
| rr_count      |            50 | E_HRV_SPC_Poincare_Ectopy | LogReg  |      0.68  | 0.132665  |                     0.6  |               0.122474  |               0.6  |          0.316228 |               0.6  |         0.2       |         0.590476 |       0.0972991 |  0.566234 | 0.203063 |  0.218218  |  0.267261 |
| rr_count      |           100 | A_HRV                     | LogReg  |      0.408 | 0.0521536 |                     0.44 |               0.0547723 |               0.32 |          0.109545 |               0.56 |         0.0894427 |         0.413333 |       0.083666  |  0.357778 | 0.100046 | -0.127287  |  0.116435 |
| rr_count      |           100 | B_SPC                     | LogReg  |      0.688 | 0.240666  |                     0.66 |               0.194936  |               0.64 |          0.296648 |               0.68 |         0.109545  |         0.633333 |       0.2       |  0.631818 | 0.251483 |  0.319656  |  0.401057 |
| rr_count      |           100 | C_HRV_SPC                 | LogReg  |      0.688 | 0.22698   |                     0.64 |               0.240832  |               0.6  |          0.316228 |               0.68 |         0.268328  |         0.646667 |       0.256688  |  0.60583  | 0.283097 |  0.286599  |  0.490571 |
| rr_count      |           100 | D_HRV_SPC_Poincare        | LogReg  |      0.68  | 0.248193  |                     0.58 |               0.248998  |               0.56 |          0.384708 |               0.6  |         0.316228  |         0.508571 |       0.374384  |  0.524444 | 0.360332 |  0.14392   |  0.545528 |
| rr_count      |           100 | E_HRV_SPC_Poincare_Ectopy | LogReg  |      0.704 | 0.240998  |                     0.62 |               0.216795  |               0.64 |          0.328634 |               0.6  |         0.282843  |         0.606667 |       0.196356  |  0.602038 | 0.261977 |  0.243299  |  0.438946 |
| rr_count      |           128 | A_HRV                     | LogReg  |      0.424 | 0.0779744 |                     0.38 |               0.0447214 |               0.24 |          0.167332 |               0.52 |         0.109545  |         0.293333 |       0.167332  |  0.26     | 0.163554 | -0.267287  |  0.130409 |
| rr_count      |           128 | B_SPC                     | LogReg  |      0.736 | 0.115239  |                     0.64 |               0.114018  |               0.68 |          0.178885 |               0.6  |         0.244949  |         0.646667 |       0.109545  |  0.648531 | 0.116896 |  0.285293  |  0.227665 |
| rr_count      |           128 | C_HRV_SPC                 | LogReg  |      0.696 | 0.0669328 |                     0.64 |               0.114018  |               0.72 |          0.228035 |               0.56 |         0.219089  |         0.629524 |       0.083054  |  0.655198 | 0.128107 |  0.296224  |  0.247039 |
| rr_count      |           128 | D_HRV_SPC_Poincare        | LogReg  |      0.688 | 0.0819756 |                     0.62 |               0.130384  |               0.68 |          0.303315 |               0.56 |         0.219089  |         0.59619  |       0.0967382 |  0.612341 | 0.205483 |  0.25258   |  0.281182 |
| rr_count      |           128 | E_HRV_SPC_Poincare_Ectopy | LogReg  |      0.768 | 0.0819756 |                     0.62 |               0.148324  |               0.64 |          0.409878 |               0.6  |         0.244949  |         0.523968 |       0.303576  |  0.562857 | 0.32598  |  0.25258   |  0.366905 |
| rr_count      |           200 | A_HRV                     | LogReg  |      0.448 | 0.0819756 |                     0.44 |               0.0894427 |               0.36 |          0.167332 |               0.52 |         0.109545  |         0.416667 |       0.117851  |  0.381313 | 0.139164 | -0.125293  |  0.184251 |
| rr_count      |           200 | B_SPC                     | LogReg  |      0.648 | 0.158493  |                     0.58 |               0.164317  |               0.52 |          0.268328 |               0.64 |         0.167332  |         0.573333 |       0.158815  |  0.531486 | 0.22119  |  0.16165   |  0.330157 |
| rr_count      |           200 | C_HRV_SPC                 | LogReg  |      0.664 | 0.227332  |                     0.56 |               0.181659  |               0.56 |          0.384708 |               0.56 |         0.219089  |         0.465    |       0.299792  |  0.502937 | 0.325771 |  0.113333  |  0.417399 |
| rr_count      |           200 | D_HRV_SPC_Poincare        | LogReg  |      0.736 | 0.177989  |                     0.58 |               0.130384  |               0.6  |          0.4      |               0.56 |         0.167332  |         0.472619 |       0.271457  |  0.521523 | 0.317288 |  0.158627  |  0.335255 |
| rr_count      |           200 | E_HRV_SPC_Poincare_Ectopy | LogReg  |      0.72  | 0.141421  |                     0.58 |               0.109545  |               0.6  |          0.374166 |               0.56 |         0.167332  |         0.479286 |       0.268599  |  0.527179 | 0.302725 |  0.156977  |  0.302292 |
| time          |           300 | A_HRV                     | LogReg  |      0.472 | 0.0657267 |                     0.46 |               0.151658  |               0.4  |          0.141421 |               0.52 |         0.178885  |         0.46     |       0.185068  |  0.426667 | 0.159009 | -0.08      |  0.308761 |
| time          |           300 | B_SPC                     | LogReg  |      0.584 | 0.212791  |                     0.6  |               0.173205  |               0.64 |          0.219089 |               0.56 |         0.167332  |         0.586667 |       0.14453   |  0.608889 | 0.176982 |  0.203299  |  0.353558 |
| time          |           300 | C_HRV_SPC                 | LogReg  |      0.592 | 0.208614  |                     0.56 |               0.194936  |               0.56 |          0.296648 |               0.56 |         0.219089  |         0.546667 |       0.165999  |  0.537143 | 0.234791 |  0.123299  |  0.397068 |
| time          |           300 | D_HRV_SPC_Poincare        | LogReg  |      0.592 | 0.217991  |                     0.54 |               0.207364  |               0.52 |          0.363318 |               0.56 |         0.219089  |         0.446667 |       0.298701  |  0.48     | 0.327677 |  0.0566326 |  0.4477   |
| time          |           300 | E_HRV_SPC_Poincare_Ectopy | LogReg  |      0.592 | 0.198796  |                     0.58 |               0.192354  |               0.56 |          0.296648 |               0.6  |         0.2       |         0.566667 |       0.164992  |  0.548052 | 0.236535 |  0.163299  |  0.391578 |

## Aporte incremental de bloques (LogReg)

| window_type   |   window_size | block_A            | block_B                   |   delta_auc |   delta_f1 |
|:--------------|--------------:|:-------------------|:--------------------------|------------:|-----------:|
| full_record   |          1800 | A_HRV              | B_SPC                     |       0.144 |     0.1927 |
| full_record   |          1800 | A_HRV              | C_HRV_SPC                 |       0.088 |     0.1715 |
| full_record   |          1800 | B_SPC              | C_HRV_SPC                 |      -0.056 |    -0.0212 |
| full_record   |          1800 | C_HRV_SPC          | D_HRV_SPC_Poincare        |       0.016 |     0      |
| full_record   |          1800 | D_HRV_SPC_Poincare | E_HRV_SPC_Poincare_Ectopy |       0.032 |    -0.0121 |
| rr_count      |            50 | A_HRV              | B_SPC                     |       0.12  |     0.0299 |
| rr_count      |            50 | A_HRV              | C_HRV_SPC                 |       0.2   |     0.0199 |
| rr_count      |            50 | B_SPC              | C_HRV_SPC                 |       0.08  |    -0.01   |
| rr_count      |            50 | C_HRV_SPC          | D_HRV_SPC_Poincare        |       0.008 |     0.0186 |
| rr_count      |            50 | D_HRV_SPC_Poincare | E_HRV_SPC_Poincare_Ectopy |       0.008 |     0.0644 |
| rr_count      |           100 | A_HRV              | B_SPC                     |       0.28  |     0.274  |
| rr_count      |           100 | A_HRV              | C_HRV_SPC                 |       0.28  |     0.2481 |
| rr_count      |           100 | B_SPC              | C_HRV_SPC                 |       0     |    -0.026  |
| rr_count      |           100 | C_HRV_SPC          | D_HRV_SPC_Poincare        |      -0.008 |    -0.0814 |
| rr_count      |           100 | D_HRV_SPC_Poincare | E_HRV_SPC_Poincare_Ectopy |       0.024 |     0.0776 |
| rr_count      |           128 | A_HRV              | B_SPC                     |       0.312 |     0.3885 |
| rr_count      |           128 | A_HRV              | C_HRV_SPC                 |       0.272 |     0.3952 |
| rr_count      |           128 | B_SPC              | C_HRV_SPC                 |      -0.04  |     0.0067 |
| rr_count      |           128 | C_HRV_SPC          | D_HRV_SPC_Poincare        |      -0.008 |    -0.0429 |
| rr_count      |           128 | D_HRV_SPC_Poincare | E_HRV_SPC_Poincare_Ectopy |       0.08  |    -0.0495 |
| rr_count      |           200 | A_HRV              | B_SPC                     |       0.2   |     0.1502 |
| rr_count      |           200 | A_HRV              | C_HRV_SPC                 |       0.216 |     0.1216 |
| rr_count      |           200 | B_SPC              | C_HRV_SPC                 |       0.016 |    -0.0285 |
| rr_count      |           200 | C_HRV_SPC          | D_HRV_SPC_Poincare        |       0.072 |     0.0186 |
| rr_count      |           200 | D_HRV_SPC_Poincare | E_HRV_SPC_Poincare_Ectopy |      -0.016 |     0.0057 |
| time          |           300 | A_HRV              | B_SPC                     |       0.112 |     0.1822 |
| time          |           300 | A_HRV              | C_HRV_SPC                 |       0.12  |     0.1105 |
| time          |           300 | B_SPC              | C_HRV_SPC                 |       0.008 |    -0.0717 |
| time          |           300 | C_HRV_SPC          | D_HRV_SPC_Poincare        |       0     |    -0.0571 |
| time          |           300 | D_HRV_SPC_Poincare | E_HRV_SPC_Poincare_Ectopy |       0     |     0.0681 |

## Top features por importancia (LogReg, coeficientes escalados)

| feature                       |   importance | importance_type    | block                     | window_type   |   window_size |
|:------------------------------|-------------:|:-------------------|:--------------------------|:--------------|--------------:|
| ewma_slope                    |      1.44544 | coefficient_scaled | C_HRV_SPC                 | full_record   |          1800 |
| ewma_slope                    |      1.35554 | coefficient_scaled | B_SPC                     | full_record   |          1800 |
| shewhart_out_count            |      1.34147 | coefficient_scaled | C_HRV_SPC                 | time          |           300 |
| ewma_slope                    |      1.32673 | coefficient_scaled | D_HRV_SPC_Poincare        | full_record   |          1800 |
| shewhart_out_count            |      1.32431 | coefficient_scaled | B_SPC                     | time          |           300 |
| shewhart_out_count            |      1.31619 | coefficient_scaled | D_HRV_SPC_Poincare        | time          |           300 |
| ewma_slope                    |      1.30775 | coefficient_scaled | E_HRV_SPC_Poincare_Ectopy | full_record   |          1800 |
| shewhart_out_count            |      1.29386 | coefficient_scaled | B_SPC                     | rr_count      |            50 |
| mr_max                        |      1.23863 | coefficient_scaled | B_SPC                     | rr_count      |           100 |
| mr_max                        |      1.22907 | coefficient_scaled | E_HRV_SPC_Poincare_Ectopy | rr_count      |           100 |
| mr_max                        |      1.21688 | coefficient_scaled | D_HRV_SPC_Poincare        | rr_count      |           100 |
| mr_max                        |      1.20401 | coefficient_scaled | C_HRV_SPC                 | rr_count      |           100 |
| cv_rr                         |     -1.18771 | coefficient_scaled | C_HRV_SPC                 | rr_count      |            50 |
| shewhart_out_count            |      1.17724 | coefficient_scaled | B_SPC                     | rr_count      |           200 |
| cusum_pos_max                 |     -1.14964 | coefficient_scaled | D_HRV_SPC_Poincare        | rr_count      |           128 |
| cusum_pos_max                 |     -1.142   | coefficient_scaled | C_HRV_SPC                 | rr_count      |           128 |
| cusum_pos_max                 |     -1.13696 | coefficient_scaled | B_SPC                     | rr_count      |           128 |
| shewhart_out_count            |      1.13667 | coefficient_scaled | C_HRV_SPC                 | rr_count      |            50 |
| shewhart_out_count            |      1.10192 | coefficient_scaled | D_HRV_SPC_Poincare        | rr_count      |            50 |
| shewhart_first_alarm_time_sec |     -1.10075 | coefficient_scaled | B_SPC                     | rr_count      |           128 |

## Riesgos metodológicos restantes

1. **Muestra pequeña**: n=25 pares. AUC con IC bootstrap amplio. Resultados exploratorios, no concluyentes clínicamente.
2. **Sin validación externa**: AFPDB es un único dataset de reto; no hay cohorte independiente.
3. **Leakage por diseño de ventanas**: ventanas de un mismo registro comparten el mismo `pair_id` y se tratan como grupo. Si se clasifica a nivel de ventana (no de registro), el riesgo de correlación espuria dentro del registro sigue presente aunque pair_id estén separados.
4. **Ectopy-like no validada**: los patrones inferidos desde RR no sustituyen análisis morfológico de onda P o anotación clínica.
5. **Hiperparámetros SPC fijos**: h, k, λ no se ajustan por fold en este pipeline. Si se tunan, debe ser dentro del train fold via nested CV.
6. **lead_time asume onset al final del registro**: esta asunción simplifica el cálculo pero introduce imprecisión si el PAF arrancó antes.
7. **Ectopy features en bloque E**: solo se incluyen si las medidas son estables (pocas ventanas con todas NaN). Verificar con `feature_audit_table`.
