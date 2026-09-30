"""
Genera reporte metodológico en Markdown con resultados y limitaciones.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import (
    CUSUM_H, CUSUM_K, DECISION_THRESHOLD,
    EWMA_L, EWMA_LAMBDA,
    GKF_N_SPLITS,
    RANDOM_STATE,
    REPORTS_DIR,
    SHEWHART_L,
)


def generate_report(
    cv_summary: pd.DataFrame,
    incremental: pd.DataFrame,
    importance: pd.DataFrame,
    task_label: str = "main_p_far_vs_p_pre",
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"report_multiscale_{task_label}.md"

    lines = [
        f"# Reporte metodológico: AFPDB multi-escala ({task_label})",
        f"_Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Pregunta de investigación",
        "¿A partir solo de intervalos RR y derivados explicables se puede distinguir, "
        "dentro de sujetos con PAF documentada, el segmento **far** (p impar, y=0) "
        "frente al segmento **pre-onset** (p par, y=1)?",
        "",
        "## Metodología",
        "",
        "### Diseño",
        "- **Tipo**: caso–control emparejado intra-sujeto PAF (par p_impar / p_par).",
        "- **Unidad mínima de agrupación**: `pair_id` (25 pares, n=50 registros).",
        "- **Análisis secundario**: p\\* vs n\\* (discriminación de grupo, NO pre-onset).",
        "",
        "### Ventanas evaluadas",
        "| Tipo | Tamaño |",
        "|------|--------|",
        "| rr_count | 50, 100, 128, 200 RR |",
        "| time | 300 s (5 min) |",
        "| full_record | ~1800 s (30 min) |",
        "",
        "**PROHIBIDO**: ventanas de 30 segundos en el análisis principal.",
        "",
        "### Cartas SPC",
        f"- Shewhart: L = {SHEWHART_L}σ",
        f"- CUSUM bilateral: h = {CUSUM_H}, k = {CUSUM_K}",
        f"- EWMA: λ = {EWMA_LAMBDA}, L = {EWMA_L}",
        "- Moving Range: UCL = D4 · mean(MR), D4 = 3.267",
        "- Runs rules (Western Electric, 4 reglas simplificadas)",
        "- Baseline SPC: primeros 200 intervalos RR del registro.",
        "  → Esto garantiza que los límites de control no usen información futura del mismo registro.",
        "",
        "### Validación",
        f"- `GroupKFold({GKF_N_SPLITS})` agrupando por `pair_id`.",
        "- Ningún `pair_id` aparece simultáneamente en train y val.",
        "- Escalado, imputación y selección de features dentro de cada fold.",
        f"- Umbral de decisión fijo = {DECISION_THRESHOLD} (sin optimización sobre test).",
        f"- `random_state = {RANDOM_STATE}`.",
        "",
        "### lead_time (solo y=1)",
        "- Se asume que el PAF onset ocurre al final del registro de 30 min (1800 s).",
        "- `lead_time_sec = 1800 - primera_alarma_absoluta`.",
        "- Para y=0: cualquier alarma es falsa alarma → se reporta `false_alarms_per_hour`.",
        "",
        "### Features de ectopia",
        "> **ADVERTENCIA**: Las variables `ectopy_like_*` son patrones COMPATIBLES "
        "con ectopia inferidos desde RR únicamente. NO son diagnóstico confirmado "
        "de PAC/APC ni análisis morfológico.",
        "",
        "## Resumen de resultados (AUC medio por ventana × bloque, LogReg)",
        "",
    ]

    if not cv_summary.empty:
        lr = cv_summary[cv_summary.get("model", pd.Series(dtype=str)) == "LogReg"] \
            if "model" in cv_summary.columns else cv_summary
        lines.append(lr.to_markdown(index=False))
    else:
        lines.append("_(sin datos)_")

    lines += [
        "",
        "## Aporte incremental de bloques (LogReg)",
        "",
    ]
    if not incremental.empty:
        lines.append(incremental.to_markdown(index=False))
    else:
        lines.append("_(sin datos)_")

    lines += [
        "",
        "## Top features por importancia (LogReg, coeficientes escalados)",
        "",
    ]
    if not importance.empty:
        top = (
            importance.assign(abs_imp=importance["importance"].abs())
            .sort_values("abs_imp", ascending=False)
            .head(20)[["feature", "importance", "importance_type",
                       "block", "window_type", "window_size"]]
        )
        lines.append(top.to_markdown(index=False))
    else:
        lines.append("_(sin datos)_")

    lines += [
        "",
        "## Riesgos metodológicos restantes",
        "",
        "1. **Muestra pequeña**: n=25 pares. AUC con IC bootstrap amplio. "
           "Resultados exploratorios, no concluyentes clínicamente.",
        "2. **Sin validación externa**: AFPDB es un único dataset de reto; "
           "no hay cohorte independiente.",
        "3. **Leakage por diseño de ventanas**: ventanas de un mismo registro "
           "comparten el mismo `pair_id` y se tratan como grupo. "
           "Si se clasifica a nivel de ventana (no de registro), el riesgo de "
           "correlación espuria dentro del registro sigue presente aunque pair_id estén separados.",
        "4. **Ectopy-like no validada**: los patrones inferidos desde RR no sustituyen "
           "análisis morfológico de onda P o anotación clínica.",
        "5. **Hiperparámetros SPC fijos**: h, k, λ no se ajustan por fold en este pipeline. "
           "Si se tunan, debe ser dentro del train fold via nested CV.",
        "6. **lead_time asume onset al final del registro**: esta asunción simplifica "
           "el cálculo pero introduce imprecisión si el PAF arrancó antes.",
        "7. **Ectopy features en bloque E**: solo se incluyen si las medidas son estables "
           "(pocas ventanas con todas NaN). Verificar con `feature_audit_table`.",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
