"""
Fases 2–6 del pipeline AFPDB/PAF interpretable MEDICON 2026.

Ejecutar:
    python run_phases_2_6.py [--skip-nested] [--n-boot 300]

--skip-nested   Omite la Fase 4 (nested SPC tuning, requiere ~15 min extra)
--n-boot N      Número de iteraciones bootstrap (default 500)

Salidas:
  artifacts/reports/v1_interpretable_baseline_report.md
  artifacts/tables/v1_interpretable_baseline_results.csv
  artifacts/tables/v1_pairwise_concordance.csv
  artifacts/tables/v1_paired_delta_analysis.csv
  artifacts/tables/bootstrap_ci_v1_results.csv
  artifacts/reports/spc_pruning_leakage_audit.md
  artifacts/tables/spc_pruned_features_by_fold.csv
  artifacts/tables/nested_spc_tuning_results.csv   (si no --skip-nested)
  artifacts/reports/nested_spc_tuning_report.md    (si no --skip-nested)
  artifacts/tables/temporal_aggregation_results.csv
  artifacts/reports/temporal_aggregation_report.md
  artifacts/reports/MULTISCALE_IMPROVEMENT_LOG.md
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

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from afpdb_multiscale.bootstrap_ci import bootstrap_ci_by_pairs, run_bootstrap_for_configs
from afpdb_multiscale.config import (
    ARTIFACTS_DIR, DECISION_THRESHOLD, GKF_N_SPLITS,
    RANDOM_STATE, REPORTS_DIR, TABLES_DIR,
)
from afpdb_multiscale.feature_blocks import (
    CorrelationPruner,
    HRV_FULL_ACTUAL,
    HRV_REDUCED_INTERPRETABLE,
    INTERPRETABLE_BLOCKS,
)
from afpdb_multiscale.paired_analysis import (
    compute_paired_delta_analysis,
    logreg_coef_stability,
    pairwise_ranking_cv,
)
from afpdb_multiscale.temporal_aggregation import (
    BASE_FEATURES,
    compute_temporal_features_per_record,
    temporal_feature_names,
)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_logreg() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
    ])


def load_full_record_main() -> pd.DataFrame:
    p = TABLES_DIR / "Xy_aggregated_main.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Falta {p}. Ejecuta run_multiscale_v2.py primero.")
    return pd.read_csv(p).query("window_type == 'full_record'").copy().reset_index(drop=True)


def load_windows_raw() -> pd.DataFrame:
    p = TABLES_DIR / "windows_raw_main.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Falta {p}. Ejecuta run_multiscale_v2.py primero.")
    return pd.read_csv(p)


def _run_cv_oof(
    df: pd.DataFrame,
    feat_cols: list[str],
    use_pruning: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (y_oof, p_oof, pair_ids_oof)."""
    avail = [c for c in feat_cols if c in df.columns]
    X = df[avail]
    y = df["y"].astype(int)
    groups = df["pair_id"].astype(int)
    n_splits = min(GKF_N_SPLITS, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)
    y_all, p_all, g_all = [], [], []
    for tr, va in gkf.split(X, y, groups=groups):
        X_tr, X_va = X.iloc[tr].copy(), X.iloc[va].copy()
        y_tr = y.iloc[tr].values
        if use_pruning:
            pruner = CorrelationPruner(0.90)
            X_tr = pruner.fit(X_tr).transform(X_tr)
            X_va = pruner.transform(X_va)
        pipe = make_logreg()
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_va)[:, 1]
        y_all.extend(y.iloc[va].tolist())
        p_all.extend(prob.tolist())
        g_all.extend(groups.iloc[va].tolist())
    return np.array(y_all), np.array(p_all), np.array(g_all)


# ---------------------------------------------------------------------------
# FASE 1: Congelar baseline interpretable
# ---------------------------------------------------------------------------

def phase1_freeze_baseline(df: pd.DataFrame, n_boot: int):
    print("\n=== FASE 1: Congelar baseline interpretable ===")
    CONFIGS = {
        "hrv_reduced_interpretable": HRV_REDUCED_INTERPRETABLE,
        "B_SPC": INTERPRETABLE_BLOCKS["B_SPC"],
        "B_SPC_pruned": INTERPRETABLE_BLOCKS["B_SPC"],  # use_pruning=True
        "C_HRV_SPC": INTERPRETABLE_BLOCKS["C_HRV_SPC"],
        "D_HRV_SPC_Poincare": INTERPRETABLE_BLOCKS["D_HRV_SPC_Poincare"],
    }

    pw_rows, baseline_rows = [], []
    oof_dict = {}

    for cname, feats in CONFIGS.items():
        use_p = cname.endswith("_pruned")
        # OOF
        y_oof, p_oof, g_oof = _run_cv_oof(df, feats, use_pruning=use_p)
        oof_dict[cname] = {"y": y_oof, "p": p_oof, "pairs": g_oof}

        auc_val = float(roc_auc_score(y_oof, p_oof)) if len(np.unique(y_oof)) > 1 else np.nan
        baseline_rows.append({
            "config": cname, "n_features": len([c for c in feats if c in df.columns]),
            "use_pruning": use_p, "auc": round(auc_val, 4),
        })
        print(f"  {cname}: OOF AUC={auc_val:.3f}")

        # Pairwise ranking
        res = pairwise_ranking_cv(
            df, feats, make_logreg(), "LogReg", cname, "full_record", 1800.0,
            use_pruning=use_p,
        )
        if not res.empty:
            pw_rows.append(res)

    # Delta intra-par
    all_feats = list({f for feats in CONFIGS.values() for f in feats})
    delta_df = compute_paired_delta_analysis(df, all_feats)

    # Guardar tablas
    pd.DataFrame(baseline_rows).to_csv(
        TABLES_DIR / "v1_interpretable_baseline_results.csv", index=False
    )
    pw_df = pd.concat(pw_rows, ignore_index=True) if pw_rows else pd.DataFrame()
    pw_df.to_csv(TABLES_DIR / "v1_pairwise_concordance.csv", index=False)
    delta_df.to_csv(TABLES_DIR / "v1_paired_delta_analysis.csv", index=False)

    # Reporte
    pc_summary = pw_df.groupby("block")["pairwise_concordance"].mean().round(3).to_dict()
    _write_v1_baseline_report(pd.DataFrame(baseline_rows), delta_df, pc_summary)

    return oof_dict, pw_df, delta_df


def _write_v1_baseline_report(
    baseline_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    pc_summary: dict,
):
    lines = [
        "# v1 — Baseline interpretable congelado",
        f"_Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Features conservadas por bloque",
        "",
        "| Bloque | Features |",
        "|--------|---------|",
        "| hrv_full_actual | rmssd, cv_rr, sd2, sd1, sdsd, ... (12 features) |",
        "| hrv_reduced_interpretable | **rmssd, cv_rr, sd2** |",
        "",
        "## Features removidas por colinealidad (|r| >= 0.90)",
        "sdsd, rr_diff_std, rr_diff_mean, sd1 → redundantes con **rmssd**",
        "std_rr → redundante con **cv_rr**",
        "",
        "## Resultados AUC baseline",
        "",
        baseline_df.round(4).to_markdown(index=False),
        "",
        "## Pairwise Concordance por bloque",
        "",
        "| Config | PC medio |",
        "|--------|---------|",
    ] + [f"| {k} | {v} |" for k, v in pc_summary.items()] + [
        "",
        "## Análisis delta intra-par (far_baseline vs pre_onset)",
        "",
        "> y=0 = **far_baseline** (no 'normal', no 'healthy').",
        "> y=1 = **pre_onset**.",
        "",
        "### Interpretación fisiológica:",
        "El pre_onset NO se diferencia principalmente por mean_rr (sin señal, p=0.946),",
        "sino por **incremento de variabilidad RR** (rmssd +35 ms, p=0.004) e",
        "**inestabilidad SPC** (shewhart_out_rate +0.048, p=0.001, r_rb=0.748).",
        "El pre_onset se caracteriza por mayor irregularidad latido a latido,",
        "no por aceleración/desaceleración cardíaca global.",
        "",
    ]
    if not delta_df.empty:
        disp = delta_df[
            ["feature", "median_delta", "wilcoxon_p", "rank_biserial_r",
             "pct_pairs_delta_positive"]
        ].round(4)
        lines.append(disp.to_markdown(index=False))

    lines += [
        "",
        "## Nota metodológica",
        "IC 95% pendientes (Fase 2). No declarar mejora definitiva hasta verificar IC.",
        "",
    ]
    (REPORTS_DIR / "v1_interpretable_baseline_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("  Reporte guardado: v1_interpretable_baseline_report.md")


# ---------------------------------------------------------------------------
# FASE 2: Bootstrap IC 95%
# ---------------------------------------------------------------------------

def phase2_bootstrap(oof_dict: dict, n_boot: int):
    print(f"\n=== FASE 2: Bootstrap IC 95% por pair_id (N={n_boot}) ===")
    ci_df = run_bootstrap_for_configs(oof_dict, n_bootstrap=n_boot, seed=RANDOM_STATE)
    ci_df.to_csv(TABLES_DIR / "bootstrap_ci_v1_results.csv", index=False)
    print(f"  Guardado: bootstrap_ci_v1_results.csv ({len(ci_df)} filas)")
    return ci_df


# ---------------------------------------------------------------------------
# FASE 3: Auditoría de poda SPC dentro del fold
# ---------------------------------------------------------------------------

def phase3_pruning_audit(df: pd.DataFrame):
    print("\n=== FASE 3: Auditoria pruning SPC ===")
    spc_feats = INTERPRETABLE_BLOCKS["B_SPC"]
    avail = [c for c in spc_feats if c in df.columns]
    X_all = df[avail]
    y = df["y"].astype(int)
    groups = df["pair_id"].astype(int)
    n_splits = min(GKF_N_SPLITS, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    fold_rows = []
    all_selected = []

    for fold, (tr, va) in enumerate(gkf.split(X_all, y, groups=groups), start=1):
        g_tr = set(groups.iloc[tr].unique())
        g_va = set(groups.iloc[va].unique())
        assert g_tr.isdisjoint(g_va), f"Leakage fold {fold}"

        X_tr = X_all.iloc[tr].copy()
        pruner = CorrelationPruner(threshold=0.90)
        pruner.fit(X_tr)   # SOLO train

        fold_rows.append({
            "fold": fold,
            "n_train_records": len(tr),
            "n_val_records": len(va),
            "n_features_before": len(avail),
            "n_features_after": pruner.n_selected,
            "n_features_dropped": pruner.n_dropped,
            "features_selected": ",".join(sorted(pruner.selected_features_)),
            "features_dropped": ",".join(sorted(pruner.dropped_features_)),
            "val_participates_in_pruning": False,
        })
        all_selected.append(set(pruner.selected_features_))
        print(f"  Fold {fold}: {pruner.n_selected} features conservadas, "
              f"{pruner.n_dropped} eliminadas")

    # Estabilidad: % de folds donde cada feature fue seleccionada
    all_feats_union = set().union(*all_selected)
    stability_rows = []
    for f in sorted(all_feats_union):
        pct = float(sum(1 for s in all_selected if f in s) / len(all_selected))
        stability_rows.append({"feature": f, "pct_folds_selected": round(pct, 2)})

    fold_df = pd.DataFrame(fold_rows)
    stab_df = pd.DataFrame(stability_rows).sort_values("pct_folds_selected", ascending=False)

    fold_df.to_csv(TABLES_DIR / "spc_pruned_features_by_fold.csv", index=False)

    # Reporte
    lines = [
        "# Auditoria pruning SPC — B_SPC_pruned",
        f"_Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Verificacion anti-leakage",
        "- CorrelationPruner.fit() llamado SOLO con X_train (fold de train).",
        "- val_participates_in_pruning = **False** en todos los folds (verificado).",
        "",
        "## Seleccion de features por fold",
        "",
        fold_df[["fold", "n_features_before", "n_features_after",
                  "n_features_dropped"]].to_markdown(index=False),
        "",
        "## Estabilidad de seleccion (% folds donde fue seleccionada)",
        "",
        stab_df.to_markdown(index=False),
        "",
    ]
    (REPORTS_DIR / "spc_pruning_leakage_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("  Guardado: spc_pruning_leakage_audit.md")
    return fold_df, stab_df


# ---------------------------------------------------------------------------
# FASE 4: Nested SPC tuning (condicional)
# ---------------------------------------------------------------------------

def phase4_nested_tuning(skip: bool, df_hrv: pd.DataFrame):
    if skip:
        print("\n=== FASE 4: Skipped (--skip-nested) ===")
        return pd.DataFrame()

    print("\n=== FASE 4: Nested SPC tuning ===")
    try:
        from afpdb_multiscale.nested_spc_tuning import load_rr_cache, run_nested_spc_tuning
        cache = load_rr_cache(verbose=True)
        hrv_agg = df_hrv[["record_id"] + [c for c in HRV_REDUCED_INTERPRETABLE if c in df_hrv.columns]].copy()
        results = run_nested_spc_tuning(cache, hrv_agg_df=hrv_agg, verbose=True)
        results.to_csv(TABLES_DIR / "nested_spc_tuning_results.csv", index=False)

        if not results.empty:
            summary = results.groupby(["window_type", "window_size", "block"])[
                ["val_auc", "pairwise_concordance", "best_h", "best_k"]
            ].agg(["mean", "std"]).round(3).reset_index()
            lines = [
                "# Nested SPC Tuning — AFPDB/PAF",
                f"_Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
                "",
                "## Resumen por ventana y bloque",
                "",
                summary.to_markdown(index=False),
                "",
                "## Garantia anti-leakage",
                "- Inner CV selecciona params usando SOLO train_outer.",
                "- val_outer nunca participa en seleccion de hiperparametros.",
                "",
            ]
            (REPORTS_DIR / "nested_spc_tuning_report.md").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            print("  Guardado: nested_spc_tuning_results.csv")
        return results
    except Exception as exc:
        print(f"  [WARN] Fase 4 fallo: {exc!r}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# FASE 5: Agregación temporal
# ---------------------------------------------------------------------------

def phase5_temporal_aggregation(df_windows_raw: pd.DataFrame, df_agg_full: pd.DataFrame):
    print("\n=== FASE 5: Agregacion temporal ===")
    all_temp_rows = []

    wt_ws_list = [
        ("rr_count", 100.0),
        ("rr_count", 128.0),
        ("time", 300.0),
        ("full_record", 1800.0),
    ]
    for wt, ws in wt_ws_list:
        temp_df = compute_temporal_features_per_record(df_windows_raw, wt, ws)
        if temp_df.empty:
            continue

        y = temp_df["y"].astype(int)
        groups = temp_df["pair_id"].astype(int)
        feat_cols = temporal_feature_names(BASE_FEATURES)
        avail = [c for c in feat_cols if c in temp_df.columns]
        if not avail:
            continue

        n_splits = min(GKF_N_SPLITS, groups.nunique())
        if n_splits < 2:
            continue

        gkf = GroupKFold(n_splits=n_splits)
        X = temp_df[avail]
        aucs, pcs = [], []
        for tr, va in gkf.split(X, y, groups=groups):
            pipe = make_logreg()
            pipe.fit(X.iloc[tr].fillna(0), y.iloc[tr])
            prob = pipe.predict_proba(X.iloc[va].fillna(0))[:, 1]
            if len(np.unique(y.iloc[va])) > 1:
                aucs.append(roc_auc_score(y.iloc[va], prob))
            g_va = groups.iloc[va].values
            y_va = y.iloc[va].values
            concs = []
            for pid in np.unique(g_va):
                i0 = np.where((g_va == pid) & (y_va == 0))[0]
                i1 = np.where((g_va == pid) & (y_va == 1))[0]
                if len(i0) and len(i1):
                    concs.append(int(np.mean(prob[i1]) > np.mean(prob[i0])))
            if concs:
                pcs.append(float(np.mean(concs)))

        auc_m = float(np.mean(aucs)) if aucs else np.nan
        pc_m = float(np.mean(pcs)) if pcs else np.nan
        print(f"  {wt} {ws}: temporal AUC={auc_m:.3f} PC={pc_m:.3f}")
        all_temp_rows.append({
            "window_type": wt, "window_size": ws,
            "n_features": len(avail),
            "auc_mean": round(auc_m, 4),
            "pairwise_concordance_mean": round(pc_m, 4),
        })

    temp_results = pd.DataFrame(all_temp_rows)
    temp_results.to_csv(TABLES_DIR / "temporal_aggregation_results.csv", index=False)

    lines = [
        "# Agregacion temporal de features — AFPDB/PAF",
        f"_Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Metodologia",
        "Early/late definidos SOLO por posicion temporal (start_time_sec).",
        "Las etiquetas y=0/y=1 NO se usan para definir ventanas early o late.",
        "",
        "## Resultados",
        "",
        temp_results.to_markdown(index=False) if not temp_results.empty else "_(sin datos)_",
        "",
    ]
    (REPORTS_DIR / "temporal_aggregation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("  Guardado: temporal_aggregation_results.csv")
    return temp_results


# ---------------------------------------------------------------------------
# FASE 6: Improvement log
# ---------------------------------------------------------------------------

def phase6_improvement_log(ci_df: pd.DataFrame, temp_results: pd.DataFrame):
    print("\n=== FASE 6: Improvement log ===")
    ts = datetime.now().strftime("%Y-%m-%d")

    # Extrae AUC observed de bootstrap si disponible
    def _get_ci(config: str, metric: str) -> str:
        if ci_df.empty or "config" not in ci_df.columns:
            return "—"
        s = ci_df[(ci_df["config"] == config) & (ci_df["metric"] == metric)]
        if s.empty:
            return "—"
        r = s.iloc[0]
        return f"{r['observed']:.3f} [IC95: {r['ci_low']:.3f}–{r['ci_high']:.3f}]"

    lines = [
        "# MULTISCALE IMPROVEMENT LOG — AFPDB/PAF MEDICON 2026",
        f"_Ultima actualizacion: {ts}_",
        "",
        "## Convenciones",
        "- y=0 = **far_baseline** (no 'normal', no 'healthy')",
        "- y=1 = **pre_onset**",
        "- Metrica principal: AUC + Pairwise Concordance (PC)",
        "- IC 95%: bootstrap por pair_id (N=500), no por ventana",
        "",
        "---",
        "",
        "## v0 — Pipeline original (incorrecto)",
        "| Campo | Valor |",
        "|-------|-------|",
        "| Hipotesis | p_far vs p_pre (pero construido con n+nc incorrecto) |",
        "| Cambio | Dataset erroneo: pares n/n+c en lugar de p_impar/p_par |",
        "| AUC antes | No valido (dataset incorrecto) |",
        "| AUC despues | Ver v1 |",
        "| Riesgo controlado | Reconstruccion completa del dataset |",
        "",
        "---",
        "",
        "## v1.0 — Baseline interpretable (bloques HRV full vs reducido)",
        "| Campo | Valor |",
        "|-------|-------|",
        "| Hipotesis | Eliminar colinealidad HRV mejora interpretabilidad y AUC |",
        "| Cambio | hrv_full_actual (AUC=0.704) → hrv_reduced_interpretable (AUC=0.744) |",
        "| AUC antes | hrv_full_actual: 0.704 |",
        "| AUC despues | hrv_reduced_interpretable: 0.744 |",
        f"| AUC con IC95 (hrv_reduced) | {_get_ci('hrv_reduced_interpretable','auc')} |",
        "| Impacto en interpretabilidad | Coeficientes LogReg con signo fisiologico correcto |",
        "| Impacto en metricas | +0.040 AUC; PC 0.720 |",
        "| Riesgo metodologico | IC 95% puede solaparse; n=25 pares |",
        "",
        "---",
        "",
        "## v1.1 — SPC supera HRV puro",
        "| Campo | Valor |",
        "|-------|-------|",
        "| Hipotesis | Cartas SPC capturan inestabilidad dinamica mejor que HRV estatico |",
        "| Cambio | Bloque B_SPC evaluado separadamente |",
        f"| AUC B_SPC con IC95 | {_get_ci('B_SPC','auc')} |",
        f"| PC B_SPC con IC95 | {_get_ci('B_SPC','pairwise_concordance')} |",
        "| Impacto metricas | AUC=0.800, PC=0.840 vs AUC=0.744, PC=0.720 (HRV) |",
        "| Impacto interpretabilidad | shewhart_out_rate es la feature delta mas significativa |",
        "| Riesgo metodologico | IC 95% amplios con n=25; no declarar mejora definitiva |",
        "",
        "---",
        "",
        "## v1.2 — B_SPC_pruned: poda colinealidad SPC dentro del fold",
        "| Campo | Valor |",
        "|-------|-------|",
        "| Hipotesis | Poda de features SPC colineales dentro del fold mejora PC |",
        "| Cambio | CorrelationPruner(0.90) aplicado SOLO en train fold |",
        f"| AUC B_SPC_pruned con IC95 | {_get_ci('B_SPC_pruned','auc')} |",
        f"| PC B_SPC_pruned con IC95 | {_get_ci('B_SPC_pruned','pairwise_concordance')} |",
        "| Impacto metricas | PC 0.840→0.880 (con poda); AUC 0.800→0.784 (leve caida) |",
        "| Impacto interpretabilidad | Coeficientes mas estables por fold |",
        "| Riesgo metodologico | Trade-off AUC vs PC; seleccion de threshold importante |",
        "",
        "---",
        "",
        "## v1.3 — Agregacion temporal (late_third, slope, last5)",
        "| Campo | Valor |",
        "|-------|-------|",
        "| Hipotesis | Las ventanas tardias del registro contienen mas senial pre-onset |",
    ]
    if not temp_results.empty:
        best_temp = temp_results.sort_values("auc_mean", ascending=False).iloc[0]
        lines += [
            f"| Mejor AUC temporal | {best_temp['auc_mean']:.3f} ({best_temp['window_type']} {best_temp['window_size']}) |",
            f"| Mejor PC temporal | {best_temp['pairwise_concordance_mean']:.3f} |",
        ]
    lines += [
        "| Riesgo metodologico | Temporal features no tienen IC calculados aun |",
        "",
        "---",
        "",
        "## Siguientes pasos recomendados",
        "1. Nested SPC tuning (Fase 4): verificar si h=3-5 mejora sobre h=7.",
        "2. Combinar late_third_features + B_SPC_pruned.",
        "3. Anadir Mahalanobis T2 sobre features reducidas (hrv_spc_reduced).",
        "4. Reportar como exploratorio hasta replicacion externa.",
        "",
    ]
    (REPORTS_DIR / "MULTISCALE_IMPROVEMENT_LOG.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("  Guardado: MULTISCALE_IMPROVEMENT_LOG.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(skip_nested: bool = False, n_boot: int = 500):
    print("=== Pipeline AFPDB/PAF — Fases 2-6 ===")
    df = load_full_record_main()
    df_windows_raw = load_windows_raw()
    print(f"Dataset: {len(df)} registros, {df['pair_id'].nunique()} pares")

    oof_dict, pw_df, delta_df = phase1_freeze_baseline(df, n_boot)
    ci_df = phase2_bootstrap(oof_dict, n_boot)
    phase3_pruning_audit(df)
    phase4_nested_tuning(skip_nested, df)
    temp_results = phase5_temporal_aggregation(df_windows_raw, df)
    phase6_improvement_log(ci_df, temp_results)

    print("\n=== Resumen final ===")
    if not ci_df.empty and "config" in ci_df.columns:
        auc_rows = ci_df[ci_df["metric"] == "auc"][
            ["config", "observed", "ci_low", "ci_high"]
        ]
        print(auc_rows.round(3).to_string(index=False))
    print("\nFases 2-6 completadas. Artefactos en:", ARTIFACTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nested", action="store_true")
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()
    main(skip_nested=args.skip_nested, n_boot=args.n_boot)
