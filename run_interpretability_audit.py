"""
Auditoría de colinealidad HRV e interpretabilidad del pipeline AFPDB/PAF.

Ejecutar:
    python run_interpretability_audit.py

Salidas:
  artifacts/reports/hrv_collinearity_audit.md
  artifacts/reports/interpretable_feature_reduction_report.md
  artifacts/tables/paired_delta_feature_analysis.csv
  artifacts/tables/pairwise_ranking_results.csv
  artifacts/tables/logreg_coef_stability.csv
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from afpdb_multiscale.config import (
    ARTIFACTS_DIR, REPORTS_DIR, TABLES_DIR, RANDOM_STATE, GKF_N_SPLITS,
    DECISION_THRESHOLD,
)
from afpdb_multiscale.feature_blocks import (
    HRV_FULL_ACTUAL, HRV_REDUCED_INTERPRETABLE, HRV_MINIMAL,
    INTERPRETABLE_BLOCKS, CorrelationPruner, compute_collinearity_report,
)
from afpdb_multiscale.paired_analysis import (
    compute_paired_delta_analysis,
    logreg_coef_stability,
    pairwise_ranking_cv,
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

AUC_CI_N = 500
RNG = np.random.default_rng(RANDOM_STATE)


def load_full_record_main() -> pd.DataFrame:
    p = TABLES_DIR / "Xy_aggregated_main.csv"
    if not p.is_file():
        raise FileNotFoundError(
            f"No encontrado: {p}\n"
            "Ejecuta primero: python run_multiscale_v2.py --no-secondary"
        )
    df = pd.read_csv(p)
    return df[df["window_type"] == "full_record"].copy().reset_index(drop=True)


def make_logreg() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
    ])


def quick_auc_cv(df, feats, use_pruning=False) -> tuple[float, float]:
    """AUC medio ± std con GroupKFold(5) por pair_id."""
    avail = [c for c in feats if c in df.columns]
    if not avail:
        return np.nan, np.nan
    X = df[avail]
    y = df["y"].astype(int)
    groups = df["pair_id"].astype(int)
    n_splits = min(GKF_N_SPLITS, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    for tr, va in gkf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr].copy(), X.iloc[va].copy()
        y_tr, y_va = y.iloc[tr].values, y.iloc[va].values
        if use_pruning:
            pruner = CorrelationPruner(threshold=0.90)
            X_tr = pruner.fit(X_tr).transform(X_tr)
            X_va = pruner.transform(X_va)
        pipe = make_logreg()
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_va)[:, 1]
        if len(np.unique(y_va)) > 1:
            aucs.append(roc_auc_score(y_va, prob))
    return (float(np.mean(aucs)), float(np.std(aucs))) if aucs else (np.nan, np.nan)


# ---------------------------------------------------------------------------
# 1. Auditoría de colinealidad
# ---------------------------------------------------------------------------

def run_collinearity_audit(df: pd.DataFrame) -> dict:
    print("1. Auditoría colinealidad HRV...")
    report = compute_collinearity_report(df, HRV_FULL_ACTUAL, threshold=0.90)

    corr_m = report["corr_matrix"]
    pairs = report["high_corr_pairs"]
    groups_col = report["collinear_groups"]

    # Confirmar redundancia RMSSD/SDSD/rr_diff_std/SD1
    key_group = [g for g in groups_col if "rmssd" in g]
    redundant_confirmed = bool(
        key_group and all(f in key_group[0] for f in ["rmssd", "sdsd", "rr_diff_std"])
    )

    lines = [
        "# Auditoría de colinealidad HRV",
        f"_Generada: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"Threshold: |r| >= 0.90 (Spearman)",
        f"Features analizadas: {len(HRV_FULL_ACTUAL)}",
        "",
        "## Pares con |r| >= 0.90",
        "",
    ]
    if pairs:
        pair_df = pd.DataFrame(pairs, columns=["feat_a", "feat_b", "r_spearman"])
        lines.append(pair_df.to_markdown(index=False))
    else:
        lines.append("_Ningún par supera el umbral de colinealidad._")
    lines += [
        "",
        "## Grupos colineales identificados",
        "",
    ]
    for i, g in enumerate(groups_col, 1):
        lines.append(f"**Grupo {i}:** {', '.join(g)}")
        lines.append(f"  - Feature conservada (mayor prioridad): **{g[0]}**")
        lines.append(f"  - Features eliminables: {', '.join(g[1:]) if len(g) > 1 else 'ninguna'}")
        lines.append("")

    lines += [
        "## Confirmación de redundancia RMSSD / SDSD / rr_diff_std / SD1",
        "",
        f"¿RMSSD, SDSD y rr_diff_std están en el mismo grupo colineal? "
        f"{'**SÍ — son redundantes**' if redundant_confirmed else '**NO — revisar manualmente**'}",
        "",
        "### Base matemática:",
        "- `sdsd = rmssd` por definición (SDSD = std de diferencias sucesivas = RMSSD, "
          "dado que RMSSD = sqrt(mean(diff^2)) y SDSD = std(diff) coinciden con σ=0 "
          "en el primer estadístico; en la práctica difieren solo por el denominador n vs n-1).",
        "- `rr_diff_std ≈ sdsd` (implementación equivalente).",
        "- `sd1 = rmssd / sqrt(2)` (definición de Poincaré).",
        "- `sd1_sd2_ratio` está correlacionado con `sd1`.",
        "",
        "## Bloques recomendados post-auditoría",
        "",
        "| Bloque | Features | Justificación |",
        "|--------|---------|---------------|",
        "| `hrv_reduced_interpretable` | rmssd, cv_rr, sd2 | Una por grupo colineal |",
        "| `hrv_minimal` | rmssd, sd2 | Mínimo interpretable publicable |",
        "",
    ]

    out = REPORTS_DIR / "hrv_collinearity_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"   Guardado: {out}")
    print(f"   Grupos colineales: {len(groups_col)}")
    print(f"   Pares |r|>=0.90: {len(pairs)}")

    return {"groups": groups_col, "pairs": pairs, "redundant_confirmed": redundant_confirmed}


# ---------------------------------------------------------------------------
# 2. Análisis delta intra-par
# ---------------------------------------------------------------------------

def run_paired_delta(df: pd.DataFrame) -> pd.DataFrame:
    print("2. Análisis delta intra-par...")
    all_features = HRV_FULL_ACTUAL + [
        "cusum_abs_max", "ewma_slope", "shewhart_out_rate", "mr_mad",
        "longest_run_above_median", "cusum_alarm_density",
    ]
    delta_df = compute_paired_delta_analysis(df, all_features)
    out = TABLES_DIR / "paired_delta_feature_analysis.csv"
    delta_df.to_csv(out, index=False)
    print(f"   Guardado: {out} ({len(delta_df)} features)")
    return delta_df


# ---------------------------------------------------------------------------
# 3. Reentrenamiento con bloques reducidos + coeficientes
# ---------------------------------------------------------------------------

def run_reduced_block_eval(df: pd.DataFrame) -> pd.DataFrame:
    print("3. Evaluación bloques HRV reducidos + coeficientes LogReg...")
    results = []
    blocks_to_test = {
        "hrv_full_actual": HRV_FULL_ACTUAL,
        "hrv_reduced_interpretable": HRV_REDUCED_INTERPRETABLE,
        "hrv_minimal": HRV_MINIMAL,
        "hrv_full_pruned": HRV_FULL_ACTUAL,  # full + pruning
    }
    for bname, feats in blocks_to_test.items():
        use_pruning = bname.endswith("_pruned")
        auc_m, auc_s = quick_auc_cv(df, feats, use_pruning=use_pruning)
        results.append({
            "block": bname,
            "n_features": len([c for c in feats if c in df.columns]),
            "use_pruning": use_pruning,
            "auc_mean": round(auc_m, 4),
            "auc_std": round(auc_s, 4),
        })
        print(f"   {bname}: AUC={auc_m:.3f} ± {auc_s:.3f}")

    coef_rows = []
    for bname, feats in [("hrv_full_actual", HRV_FULL_ACTUAL),
                          ("hrv_reduced_interpretable", HRV_REDUCED_INTERPRETABLE),
                          ("hrv_full_pruned", HRV_FULL_ACTUAL)]:
        use_p = bname.endswith("_pruned")
        stab = logreg_coef_stability(df, feats, use_pruning=use_p)
        stab["block"] = bname
        coef_rows.append(stab)

    coef_df = pd.concat(coef_rows, ignore_index=True)
    coef_df.to_csv(TABLES_DIR / "logreg_coef_stability.csv", index=False)
    print(f"   Coeficientes guardados: {TABLES_DIR / 'logreg_coef_stability.csv'}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 4. Pairwise ranking multi-bloque
# ---------------------------------------------------------------------------

def run_pairwise_ranking(df: pd.DataFrame) -> pd.DataFrame:
    print("4. Pairwise ranking evaluation...")
    blocks = {
        "hrv_reduced_interpretable": HRV_REDUCED_INTERPRETABLE,
        "B_SPC": INTERPRETABLE_BLOCKS["B_SPC"],
        "C_HRV_SPC": INTERPRETABLE_BLOCKS["C_HRV_SPC"],
        "D_HRV_SPC_Poincare": INTERPRETABLE_BLOCKS["D_HRV_SPC_Poincare"],
    }
    model = make_logreg()
    all_rows = []
    for bname, feats in blocks.items():
        for use_pruning in (False, True):
            label = f"{bname}{'_pruned' if use_pruning else ''}"
            res = pairwise_ranking_cv(
                df, feats, model, "LogReg", label, "full_record", 1800.0,
                use_pruning=use_pruning,
            )
            if not res.empty:
                all_rows.append(res)
                pc = res["pairwise_concordance"].mean()
                auc_m = res["auc"].mean()
                print(f"   {label}: PC={pc:.3f} | AUC={auc_m:.3f}")

    pw_df = pd.concat(all_rows, ignore_index=True)
    out = TABLES_DIR / "pairwise_ranking_results.csv"
    pw_df.to_csv(out, index=False)
    print(f"   Guardado: {out}")
    return pw_df


# ---------------------------------------------------------------------------
# 5. Reporte final
# ---------------------------------------------------------------------------

def write_final_report(
    col_audit: dict,
    delta_df: pd.DataFrame,
    block_eval: pd.DataFrame,
    pw_df: pd.DataFrame,
    coef_path: Path,
) -> Path:
    print("5. Generando reporte final...")

    coef_df = pd.read_csv(coef_path) if coef_path.is_file() else pd.DataFrame()

    lines = [
        "# Reporte de reducción de features interpretable — AFPDB/PAF MEDICON 2026",
        f"_Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## 1. Features redundantes (|r| >= 0.90 Spearman)",
        "",
    ]
    for i, g in enumerate(col_audit["groups"], 1):
        lines.append(f"- **Grupo {i}**: {', '.join(g)}")
        lines.append(f"  Conservar: **{g[0]}** | Eliminar: {', '.join(g[1:]) if len(g) > 1 else '—'}")
    lines += [
        "",
        "### ¿El análisis delta anterior (AUC=0.397) era metodológicamente válido?",
        "",
        "**NO.** El análisis delta intra-par produce 25 vectores Δ(pre−far), uno por par. "
        "Todos los vectores corresponden al mismo tipo de evento (el cambio fisiológico "
        "que precede al PAF), sin una segunda clase natural. "
        "La pseudo-etiqueta 'cambio grande vs cambio pequeño' basada en la norma del vector "
        "no representa una hipótesis clínica válida. "
        "El AUC resultante (0.397) no tiene interpretación estadística ni clínica clara.",
        "",
        "**Corrección aplicada:** solo se reportan estadísticas descriptivas por feature "
        "(mediana delta, IQR, Wilcoxon p, rank-biserial r, % pares en dirección esperada).",
        "",
        "## 2. Features conservadas por bloque",
        "",
        "| Bloque | Features |",
        "|--------|---------|",
        f"| `hrv_full_actual` | {', '.join(HRV_FULL_ACTUAL)} |",
        f"| `hrv_reduced_interpretable` | {', '.join(HRV_REDUCED_INTERPRETABLE)} |",
        f"| `hrv_minimal` | {', '.join(HRV_MINIMAL)} |",
        "",
        "## 3. Impacto en AUC (LogReg, GroupKFold por pair_id)",
        "",
    ]
    if not block_eval.empty:
        lines.append(block_eval.round(4).to_markdown(index=False))
    lines += [
        "",
        "## 4. Estabilidad de coeficientes LogReg (% folds con signo positivo)",
        "",
    ]
    if not coef_df.empty:
        for bname in coef_df["block"].unique():
            sub = coef_df[coef_df["block"] == bname][
                ["feature", "mean_coef", "pct_folds_positive", "sign_stable"]
            ].round(4)
            lines.append(f"### {bname}")
            lines.append(sub.to_markdown(index=False))
            n_stable = int(sub["sign_stable"].sum())
            lines.append(f"_Coeficientes con signo estable (>80% o <20% positive): {n_stable}/{len(sub)}_")
            lines.append("")

    lines += [
        "## 5. Análisis delta intra-par (correcto)",
        "",
    ]
    if not delta_df.empty:
        disp = delta_df[
            ["feature", "median_delta", "iqr_delta", "wilcoxon_p",
             "rank_biserial_r", "pct_pairs_delta_positive", "direction_consistent"]
        ].round(4)
        lines.append(disp.to_markdown(index=False))
    lines += [
        "",
        "## 6. Pairwise ranking evaluation (PC = score_pre > score_far)",
        "",
    ]
    if not pw_df.empty:
        summ = (
            pw_df.groupby("block")[["pairwise_concordance", "auc", "f1"]]
            .mean()
            .round(3)
            .reset_index()
        )
        lines.append(summ.to_markdown(index=False))
    lines += [
        "",
        "## 7. Recomendaciones para siguiente paso",
        "",
        "1. **Usar `hrv_reduced_interpretable` (rmssd, cv_rr, sd2)** como bloque HRV "
           "estándar en publicaciones. Elimina colinealidad severa; coeficientes LogReg "
           "deben tener signo estable (+) para las tres features.",
        "2. **El bloque B_SPC sigue siendo el más discriminativo** (AUC ~ 0.66–0.80 "
           "según ventana). Combinarlo con `hrv_reduced_interpretable` da C_HRV_SPC.",
        "3. **Pairwise concordance** es la métrica más adecuada para este dataset "
           "emparejado. Un PC > 0.7 equivale a que el modelo ordena correctamente "
           "> 70% de pares (far < pre-onset en score).",
        "4. **No introducir más features hasta reducir colinealidad** y confirmar "
           "estabilidad de signos de coeficientes.",
        "5. **Siguiente módulo recomendado**: tuning de h/k CUSUM dentro del train fold "
           "(nested CV) usando ventana de 100 RR, que mostró AUC = 0.80.",
        "",
        "## Riesgos metodológicos restantes",
        "",
        "- n=25 pares: todas las estimaciones tienen IC amplio.",
        "- CorrelationPruner usa threshold fijo (0.90); sensibilidad al threshold no evaluada.",
        "- `pairwise_concordance` asume que ambos registros de un par siempre van al mismo "
          "fold; esto está garantizado por GroupKFold(pair_id) pero debe verificarse.",
        "",
    ]

    out = REPORTS_DIR / "interpretable_feature_reduction_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"   Guardado: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Auditoría de interpretabilidad AFPDB/PAF ===")
    df = load_full_record_main()
    print(f"Dataset: {len(df)} registros, {df['pair_id'].nunique()} pares")

    col_audit = run_collinearity_audit(df)
    delta_df = run_paired_delta(df)
    block_eval = run_reduced_block_eval(df)
    pw_df = run_pairwise_ranking(df)
    report = write_final_report(
        col_audit, delta_df, block_eval, pw_df,
        TABLES_DIR / "logreg_coef_stability.csv",
    )
    print(f"\nReporte final: {report}")
    print("Completado.")


if __name__ == "__main__":
    main()
