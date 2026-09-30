# Auditoría de colinealidad HRV
_Generada: 2026-04-24 17:48_

Threshold: |r| >= 0.90 (Spearman)
Features analizadas: 12

## Pares con |r| >= 0.90

| feat_a      | feat_b      |   r_spearman |
|:------------|:------------|-------------:|
| mean_rr     | median_rr   |       0.9853 |
| std_rr      | rmssd       |       0.9584 |
| std_rr      | sdsd        |       0.9584 |
| std_rr      | cv_rr       |       0.9399 |
| std_rr      | rr_diff_std |       0.9584 |
| std_rr      | sd1         |       0.9584 |
| std_rr      | sd2         |       0.9453 |
| rmssd       | sdsd        |       1      |
| rmssd       | cv_rr       |       0.9442 |
| rmssd       | rr_diff_std |       1      |
| rmssd       | sd1         |       1      |
| sdsd        | cv_rr       |       0.9442 |
| sdsd        | rr_diff_std |       1      |
| sdsd        | sd1         |       1      |
| cv_rr       | rr_diff_std |       0.9442 |
| cv_rr       | sd1         |       0.9442 |
| rr_diff_std | sd1         |       1      |

## Grupos colineales identificados

**Grupo 1:** mean_rr, median_rr
  - Feature conservada (mayor prioridad): **mean_rr**
  - Features eliminables: median_rr

**Grupo 2:** rmssd, cv_rr, sd2, sd1, sdsd, rr_diff_std, std_rr
  - Feature conservada (mayor prioridad): **rmssd**
  - Features eliminables: cv_rr, sd2, sd1, sdsd, rr_diff_std, std_rr

## Confirmación de redundancia RMSSD / SDSD / rr_diff_std / SD1

¿RMSSD, SDSD y rr_diff_std están en el mismo grupo colineal? **SÍ — son redundantes**

### Base matemática:
- `sdsd = rmssd` por definición (SDSD = std de diferencias sucesivas = RMSSD, dado que RMSSD = sqrt(mean(diff^2)) y SDSD = std(diff) coinciden con σ=0 en el primer estadístico; en la práctica difieren solo por el denominador n vs n-1).
- `rr_diff_std ≈ sdsd` (implementación equivalente).
- `sd1 = rmssd / sqrt(2)` (definición de Poincaré).
- `sd1_sd2_ratio` está correlacionado con `sd1`.

## Bloques recomendados post-auditoría

| Bloque | Features | Justificación |
|--------|---------|---------------|
| `hrv_reduced_interpretable` | rmssd, cv_rr, sd2 | Una por grupo colineal |
| `hrv_minimal` | rmssd, sd2 | Mínimo interpretable publicable |
