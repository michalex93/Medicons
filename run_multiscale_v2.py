"""
Pipeline multi-escala AFPDB/PAF — MEDICON 2026
Ejecutar desde la raíz del proyecto:
    python run_multiscale_v2.py [--no-secondary] [--skip-build]

--no-secondary   Omite el análisis secundario p* vs n*
--skip-build     Usa Xy_aggregated_main.csv ya existente (evita re-leer AFPDB)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Cambiar al directorio raíz para que wfdb no confunda las rutas con URLs
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from afpdb_multiscale.build_dataset import build_and_save_datasets
from afpdb_multiscale.config import ARTIFACTS_DIR, REPORTS_DIR, TABLES_DIR
from afpdb_multiscale.evaluate import evaluate_all
from afpdb_multiscale.report import generate_report


def savefig(fig, stem: str):
    fig_dir = ARTIFACTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("tif", {"dpi": 300})):
        try:
            fig.savefig(fig_dir / f"{stem}.{ext}", bbox_inches="tight", **kw)
        except Exception:
            pass
    plt.close(fig)


def plot_auc_heatmap(cv_summary: pd.DataFrame):
    """Heatmap AUC medio por (window_size, bloque) para cada modelo."""
    if cv_summary.empty or "auc_mean" not in cv_summary.columns:
        return
    for model in cv_summary.get("model", pd.Series(dtype=str)).unique():
        sub = cv_summary[cv_summary["model"] == model]
        pivot = sub.pivot_table(
            index="block", columns="window_size", values="auc_mean"
        )
        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.9)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                val = pivot.values[r, c]
                if not pd.isna(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, label="AUC medio")
        ax.set_title(f"AUC medio — {model}")
        plt.tight_layout()
        savefig(fig, f"auc_heatmap_{model}")


def main(skip_build: bool = False, include_secondary: bool = True):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Construir / cargar datasets
    # ------------------------------------------------------------------
    agg_main_path = TABLES_DIR / "Xy_aggregated_main.csv"

    if skip_build and agg_main_path.is_file():
        print("Cargando dataset agregado existente…")
        df_agg_main = pd.read_csv(agg_main_path)
    else:
        print("Extrayendo features multi-escala desde AFPDB…")
        saved = build_and_save_datasets(
            verbose=True, include_secondary=include_secondary
        )
        df_agg_main = pd.read_csv(saved["agg_main"])
        print(f"Datasets guardados: {list(saved.values())}")

    print(f"Filas en dataset agregado principal: {len(df_agg_main)}")
    print(df_agg_main[["window_type", "window_size", "y"]].value_counts().head(20))

    # ------------------------------------------------------------------
    # 2. Evaluación: tarea principal
    # ------------------------------------------------------------------
    print("\nEvaluando tarea principal (far vs pre-onset)…")
    results = evaluate_all(df_agg_main, task_label="main")

    cv_detail = results["cv_detail"]
    cv_summary = results["cv_summary"]
    importance = results["importance"]
    incremental = results["incremental"]

    # Guardar tablas
    if not cv_detail.empty:
        cv_detail.to_csv(TABLES_DIR / "cv_detail_main.csv", index=False)
    if not cv_summary.empty:
        cv_summary.to_csv(TABLES_DIR / "cv_summary_main.csv", index=False)
    if not importance.empty:
        importance.to_csv(TABLES_DIR / "feature_importance_main.csv", index=False)
    if not incremental.empty:
        incremental.to_csv(TABLES_DIR / "incremental_blocks_main.csv", index=False)

    # Figura heatmap AUC
    plot_auc_heatmap(cv_summary)

    # ------------------------------------------------------------------
    # 3. Reporte metodológico
    # ------------------------------------------------------------------
    report_path = generate_report(cv_summary, incremental, importance)
    print(f"\nReporte guardado: {report_path}")

    # ------------------------------------------------------------------
    # 4. Impresión de resumen en consola
    # ------------------------------------------------------------------
    if not cv_summary.empty and "auc_mean" in cv_summary.columns:
        print("\n=== Resumen AUC × bloque × modelo (ventana full_record) ===")
        sub = cv_summary[cv_summary.get("window_type", pd.Series(dtype=str)) == "full_record"]
        if not sub.empty:
            print(sub[["block", "model", "auc_mean", "auc_std", "f1_mean"]].to_string(index=False))
        else:
            print(cv_summary[["window_type", "window_size", "block", "model",
                               "auc_mean", "f1_mean"]].head(20).to_string(index=False))

    print("\nPipeline completado. Artefactos en:", ARTIFACTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-secondary", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    main(skip_build=args.skip_build, include_secondary=not args.no_secondary)
