"""
Experimento Mahalanobis / Hotelling T² — AFPDB/PAF MEDICON 2026

Ejecutar desde la raíz del proyecto:
    python run_mahal_experiment.py [--alarm-pct 97.5] [--skip-build]

--alarm-pct   Percentil para umbral de alarma (default 95.0)
--skip-build  Reutiliza windows_raw_main.csv si ya existe
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afpdb_multiscale.build_dataset import build_and_save_datasets
from afpdb_multiscale.config import ARTIFACTS_DIR, REPORTS_DIR, TABLES_DIR
from afpdb_multiscale.features_mahalanobis import MAHAL_FEATURE_SETS
from afpdb_multiscale.mahal_evaluate import evaluate_mahalanobis_all


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _savefig(fig, stem: str):
    fig_dir = ARTIFACTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("tif", {"dpi": 300})):
        try:
            fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight", **kw)
        except Exception:
            pass
    plt.close(fig)


def plot_lead_time_distribution(lead_time_df: pd.DataFrame):
    if lead_time_df.empty or "lead_time_sec" not in lead_time_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for fs_name, sub in lead_time_df.groupby("mahal_feature_set"):
        vals = sub["lead_time_sec"].dropna().values / 60.0  # minutos
        if len(vals) > 0:
            ax.hist(vals, bins=10, alpha=0.6, label=fs_name)
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución lead_time (solo y=1, onset asumido al final del registro)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "mahal_lead_time_distribution")


def plot_auc_comparison(cv_summary: pd.DataFrame):
    if cv_summary.empty:
        return
    auc_col = next((c for c in cv_summary.columns if "auc" in c.lower() and "mean" in c.lower()), None)
    if auc_col is None:
        return
    modes = cv_summary["mode"].unique() if "mode" in cv_summary.columns else ["population_fold"]
    for mode in modes:
        sub = cv_summary[cv_summary.get("mode", pd.Series(dtype=str)) == mode] \
            if "mode" in cv_summary.columns else cv_summary
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for mname, msub in sub.groupby("model"):
            if "block" not in msub.columns:
                continue
            ax.scatter(msub["block"], msub[auc_col], label=mname, s=60)
        ax.set_ylabel("AUC medio")
        ax.set_title(f"Mahalanobis AUC por bloque — modo {mode}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        _savefig(fig, f"mahal_auc_blocks_{mode}")


def plot_incremental(incremental: pd.DataFrame):
    if incremental.empty or "delta_auc" not in incremental.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for fs_name, sub in incremental.groupby("mahal_feature_set"):
        models = sub["model"].unique() if "model" in sub.columns else ["LogReg"]
        for mname in models:
            m_sub = sub[sub["model"] == mname] if "model" in sub.columns else sub
            vals = m_sub["delta_auc"].dropna().values
            if len(vals):
                ax.bar(
                    [f"{fs_name}\n{mname}"],
                    [float(np.mean(vals))],
                    yerr=[float(np.std(vals))],
                    capsize=3,
                )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("ΔAUC (E − C)")
    ax.set_title("Aporte incremental: HRV+SPC+Mahal vs HRV+SPC")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "mahal_incremental_auc")


# ---------------------------------------------------------------------------
# Reporte metodológico
# ---------------------------------------------------------------------------

def generate_mahal_report(
    cv_summary: pd.DataFrame,
    incremental: pd.DataFrame,
    lead_time_df: pd.DataFrame,
    alarm_pct: float,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "report_mahalanobis.md"

    lines = [
        "# Reporte Mahalanobis / Hotelling T² — AFPDB PAF MEDICON 2026",
        f"_Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Pregunta",
        "¿La distancia de Mahalanobis / T² calculada sobre features RR/HRV/SPC "
        "separa los registros far (y=0) de los pre-onset (y=1) dentro de sujetos con PAF?",
        "",
        "## Dos modos de estimación",
        "",
        "### 1. `population_fold` (modelo poblacional)",
        "- μ y Σ estimados **exclusivamente** sobre y=0 del train fold.",
        "- Imputer (imputación de NaN) ajustado también solo en y=0 train.",
        f"- Umbral de alarma: percentil {alarm_pct} de D_M en y=0 train.",
        "- Covarianza: LedoitWolf shrinkage; fallback pseudo-inversa con warning.",
        "- Validación: GroupKFold(5) por pair_id. Ningún pair_id en train y val simultáneamente.",
        "",
        "### 2. `pair_baseline` (análisis intra-sujeto)",
        "- μ y Σ estimados con ventanas **far (y=0) del mismo par**.",
        "- D_M de y=0: auto-distancia (desviación interna del registro far).",
        "- D_M de y=1: distancia al basal far del mismo sujeto.",
        "> **ADVERTENCIA**: Este modo asume disponibilidad de un segmento basal far del",
        "> mismo sujeto. No es un modelo poblacional.",
        "",
        "## Feature sets reducidos",
        "| Set | Features |",
        "|-----|---------|",
    ]
    for fs_name, cols in MAHAL_FEATURE_SETS.items():
        lines.append(f"| `{fs_name}` | {', '.join(cols)} |")

    lines += [
        "",
        "## Ventanas evaluadas",
        "50 RR, 100 RR, 128 RR, 200 RR, 5 min (300 s), 30 min completo.",
        "**30 segundos: PROHIBIDO** (ProhibitedWindowError).",
        "",
        "## Bloques de evaluación",
        "| Bloque | Contenido |",
        "|--------|-----------|",
        "| A_HRV | HRV básico |",
        "| B_SPC | SPC puro |",
        "| C_HRV_SPC | HRV + SPC |",
        "| D_Mahal | Solo features D_M agregados |",
        "| E_HRV_SPC_Mahal | HRV + SPC + D_M |",
        "",
        "## Modelos",
        "- `LogReg`: LogisticRegression con StandardScaler (escalado solo dentro de train fold).",
        "- `RandomForest`: árbol de decisión regularizado.",
        "- `Mahal_Threshold`: clasificador puro por umbral en `mahalanobis_max`.",
        "",
        "## Features agregadas por registro desde D_M por ventana",
        "mahalanobis_mean, mahalanobis_median, mahalanobis_max, mahalanobis_p90, "
        "mahalanobis_last, mahalanobis_slope, mahalanobis_alarm_count, "
        "mahalanobis_alarm_density, mahalanobis_time_in_alarm, "
        "mahalanobis_first_alarm_time_sec.",
        "",
        "### lead_time",
        "- **Solo para y=1** (pre-onset).",
        "- Supuesto: PAF onset al final del registro (segundo 1800).",
        "- `lead_time_sec = 1800 − primera_alarma_absoluta_D_M`.",
        "- Para y=0: alarmas = falsas alarmas.",
        "",
    ]

    if not cv_summary.empty:
        lines += ["## Resumen de resultados AUC", ""]
        auc_col = next((c for c in cv_summary.columns if "auc_mean" in c.lower()), None)
        if auc_col:
            disp_cols = [c for c in cv_summary.columns if any(
                k in c for k in ("window_type", "window_size", "mode", "block", "model", auc_col[:6])
            )][:8]
            lines.append(cv_summary[disp_cols].head(30).to_markdown(index=False))
        lines.append("")

    if not incremental.empty:
        lines += ["## Aporte incremental (bloque C → E)", ""]
        lines.append(incremental.to_markdown(index=False))
        lines.append("")

    if not lead_time_df.empty:
        lt_valid = lead_time_df["lead_time_sec"].dropna()
        lines += [
            "## Lead time (y=1)",
            f"- Registros con alarma detectada: {len(lt_valid)} / {len(lead_time_df)}",
            f"- Mediana lead_time: {lt_valid.median() / 60.0:.1f} min" if len(lt_valid) > 0 else "- Sin alarmas detectadas.",
            "",
        ]

    lines += [
        "## Riesgos metodológicos",
        "",
        "1. **n=25 pares**: D_M con p > n o p ≈ n requiere regularización (LedoitWolf); "
           "resultados exploratorios.",
        "2. **Feature sets reducidos ad-hoc**: no se seleccionaron globalmente sobre el dataset; "
           "sin embargo, la selección de subconjunto sigue siendo una decisión a priori "
           "que puede introducir sesgo si no se confirma en validación externa.",
        "3. **pair_baseline asume baseline disponible**: en práctica clínica, puede no existir "
           "un segmento 'far' previo al episodio para el mismo paciente.",
        "4. **lead_time depende de umbral de alarma**: el percentil 95/97.5 es conservador "
           "pero arbitrario; análisis de sensibilidad al percentil recomendado.",
        "5. **Correlación entre features**: features HRV y SPC correlacionadas → "
           "Σ mal condicionada incluso con LedoitWolf si la correlación es muy alta.",
        "6. **30 segundos prohibido**: la prohibición está implementada en código y tests; "
           "no usar ese tamaño en ninguna comparación.",
        "",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(alarm_pct: float = 95.0, skip_build: bool = False):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    raw_path = TABLES_DIR / "windows_raw_main.csv"
    agg_path = TABLES_DIR / "Xy_aggregated_main.csv"

    if not skip_build or not raw_path.is_file():
        print("Extrayendo features multi-escala desde AFPDB…")
        saved = build_and_save_datasets(verbose=True, include_secondary=False)
        print("Datasets guardados:", list(saved.values()))
    else:
        print("Usando datasets existentes.")

    print("Cargando datasets…")
    df_windows = pd.read_csv(raw_path)
    df_agg = pd.read_csv(agg_path)
    print(f"  windows_raw: {len(df_windows)} filas")
    print(f"  Xy_aggregated: {len(df_agg)} filas")

    print(f"\nEvaluando Mahalanobis (alarm_percentile={alarm_pct})…")
    results = evaluate_mahalanobis_all(
        df_windows, df_agg, alarm_percentile=alarm_pct, verbose=True
    )

    cv_detail = results["cv_detail"]
    cv_summary = results["cv_summary"]
    incremental = results["incremental"]
    lead_time = results["lead_time"]

    if not cv_detail.empty:
        cv_detail.to_csv(TABLES_DIR / "mahal_cv_detail.csv", index=False)
    if not cv_summary.empty:
        cv_summary.to_csv(TABLES_DIR / "mahal_cv_summary.csv", index=False)
    if not incremental.empty:
        incremental.to_csv(TABLES_DIR / "mahal_incremental.csv", index=False)
    if not lead_time.empty:
        lead_time.to_csv(TABLES_DIR / "mahal_lead_time.csv", index=False)

    # Figuras
    plot_lead_time_distribution(lead_time)
    plot_auc_comparison(cv_summary)
    plot_incremental(incremental)

    # Reporte
    report_path = generate_mahal_report(cv_summary, incremental, lead_time, alarm_pct)
    print(f"\nReporte guardado: {report_path}")

    # Resumen consola
    if not cv_summary.empty:
        auc_col = next((c for c in cv_summary.columns if "auc_mean" in c.lower()), None)
        if auc_col:
            print(f"\n=== AUC medio (top 10 filas, modo population_fold) ===")
            sub = cv_summary
            if "mode" in sub.columns:
                sub = sub[sub["mode"] == "population_fold"]
            print(sub[[c for c in ["window_type", "window_size", "block", "model", auc_col]
                        if c in sub.columns]].head(10).to_string(index=False))

    print("\nPipeline Mahalanobis completado. Artefactos en:", ARTIFACTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alarm-pct", type=float, default=95.0)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    main(alarm_pct=args.alarm_pct, skip_build=args.skip_build)
