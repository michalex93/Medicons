# MULTISCALE IMPROVEMENT LOG — AFPDB/PAF MEDICON 2026
_Ultima actualizacion: 2026-04-24_

## Convenciones
- y=0 = **far_baseline** (no 'normal', no 'healthy')
- y=1 = **pre_onset**
- Metrica principal: AUC + Pairwise Concordance (PC)
- IC 95%: bootstrap por pair_id (N=500), no por ventana

---

## v0 — Pipeline original (incorrecto)
| Campo | Valor |
|-------|-------|
| Hipotesis | p_far vs p_pre (pero construido con n+nc incorrecto) |
| Cambio | Dataset erroneo: pares n/n+c en lugar de p_impar/p_par |
| AUC antes | No valido (dataset incorrecto) |
| AUC despues | Ver v1 |
| Riesgo controlado | Reconstruccion completa del dataset |

---

## v1.0 — Baseline interpretable (bloques HRV full vs reducido)
| Campo | Valor |
|-------|-------|
| Hipotesis | Eliminar colinealidad HRV mejora interpretabilidad y AUC |
| Cambio | hrv_full_actual (AUC=0.704) → hrv_reduced_interpretable (AUC=0.744) |
| AUC antes | hrv_full_actual: 0.704 |
| AUC despues | hrv_reduced_interpretable: 0.744 |
| AUC con IC95 (hrv_reduced) | 0.654 [IC95: 0.582–0.730] |
| Impacto en interpretabilidad | Coeficientes LogReg con signo fisiologico correcto |
| Impacto en metricas | +0.040 AUC; PC 0.720 |
| Riesgo metodologico | IC 95% puede solaparse; n=25 pares |

---

## v1.1 — SPC supera HRV puro
| Campo | Valor |
|-------|-------|
| Hipotesis | Cartas SPC capturan inestabilidad dinamica mejor que HRV estatico |
| Cambio | Bloque B_SPC evaluado separadamente |
| AUC B_SPC con IC95 | 0.776 [IC95: 0.704–0.862] |
| PC B_SPC con IC95 | 0.840 [IC95: 0.733–0.941] |
| Impacto metricas | AUC=0.800, PC=0.840 vs AUC=0.744, PC=0.720 (HRV) |
| Impacto interpretabilidad | shewhart_out_rate es la feature delta mas significativa |
| Riesgo metodologico | IC 95% amplios con n=25; no declarar mejora definitiva |

---

## v1.2 — B_SPC_pruned: poda colinealidad SPC dentro del fold
| Campo | Valor |
|-------|-------|
| Hipotesis | Poda de features SPC colineales dentro del fold mejora PC |
| Cambio | CorrelationPruner(0.90) aplicado SOLO en train fold |
| AUC B_SPC_pruned con IC95 | 0.752 [IC95: 0.680–0.834] |
| PC B_SPC_pruned con IC95 | 0.880 [IC95: 0.786–1.000] |
| Impacto metricas | PC 0.840→0.880 (con poda); AUC 0.800→0.784 (leve caida) |
| Impacto interpretabilidad | Coeficientes mas estables por fold |
| Riesgo metodologico | Trade-off AUC vs PC; seleccion de threshold importante |

---

## v1.3 — Agregacion temporal (late_third, slope, last5)
| Campo | Valor |
|-------|-------|
| Hipotesis | Las ventanas tardias del registro contienen mas senial pre-onset |
| Mejor AUC temporal | 0.864 (full_record 1800.0) |
| Mejor PC temporal | 0.920 |
| Riesgo metodologico | Temporal features no tienen IC calculados aun |

---

## Siguientes pasos recomendados
1. Nested SPC tuning (Fase 4): verificar si h=3-5 mejora sobre h=7.
2. Combinar late_third_features + B_SPC_pruned.
3. Anadir Mahalanobis T2 sobre features reducidas (hrv_spc_reduced).
4. Reportar como exploratorio hasta replicacion externa.
