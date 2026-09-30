"""
Análisis intra-par y evaluación por ranking pareado.

Módulo 1: compute_paired_delta_analysis
  - delta = feature_pre_onset - feature_far por cada pair_id
  - Estadística descriptiva: median_delta, IQR, Wilcoxon p, rank-biserial r,
    percent_pairs_positive/negative
  - NUNCA calcula AUC sobre vectores delta: con 25 pares y todos con la misma
    "etiqueta" (el pre-onset siempre es la diferencia de un único par), no hay
    dos clases naturales y el AUC sería una pseudométrica sin interpretación válida.

Módulo 2: pairwise_ranking_cv
  - Para cada fold de GroupKFold:
    - Entrena el modelo en pares del train fold.
    - Para cada pair_id de validación: calcula score_pre y score_far.
    - concordant = (score_pre > score_far).
    - pairwise_concordance = proporción de pares correctamente ordenados.
    - margin = score_pre - score_far (magnitud y dirección).
  - Esta métrica es equivalente a P(score_pre > score_far) y relacionada con AUC
    para datos balanceados, pero explota la estructura pareada del dataset AFPDB.

Módulo 3: evaluate_with_correlation_pruning
  - Wrapper que aplica CorrelationPruner dentro de cada fold antes del modelo.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DECISION_THRESHOLD, GKF_N_SPLITS, RANDOM_STATE
from .feature_blocks import CorrelationPruner, INTERPRETABLE_BLOCKS

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. Análisis delta intra-par (SOLO descriptivo — sin AUC inválido)
# ---------------------------------------------------------------------------

def _rank_biserial(d: np.ndarray) -> float:
    """
    Rank-biserial correlation para Wilcoxon de una muestra.
    r = (T+ - T-) / (T+ + T-)
    Rango en [-1, +1]. +1 = todos los deltas positivos.
    """
    dnn = d[d != 0]
    if len(dnn) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(dnn))
    T_plus = float(np.sum(ranks[dnn > 0]))
    T_minus = float(np.sum(ranks[dnn < 0]))
    total = T_plus + T_minus
    return float((T_plus - T_minus) / total) if total > 0 else np.nan


def compute_paired_delta_analysis(
    df_full_record: pd.DataFrame,
    feature_cols: list[str],
    analysis_task: str = "main_p_far_vs_p_pre",
) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas de delta = pre_onset - far para cada feature.

    PRECONDICIÓN:
      df_full_record contiene SOLO registros de analysis_task = 'main_p_far_vs_p_pre'
      (sin mezcla con p* vs n*).

    NO calcula AUC sobre vectores delta: el marco delta no tiene dos clases naturales
    y reportar AUC en ese contexto sería metodológicamente inválido (ver docstring).

    Parámetros
    ----------
    df_full_record : DataFrame con columnas record_id, pair_id, y, + feature_cols.
                     Se espera que pair_id tenga exactamente 1 registro y=0 y 1 y=1.
    """
    # Validar que no se mezclan tareas
    if "analysis_task" in df_full_record.columns:
        tasks = df_full_record["analysis_task"].unique()
        assert len(tasks) == 1 and tasks[0] == analysis_task, (
            f"Se esperaba solo '{analysis_task}', encontrado: {tasks}"
        )

    avail = [c for c in feature_cols if c in df_full_record.columns]
    rows = []

    for feat in avail:
        pivot = df_full_record.pivot_table(index="pair_id", columns="y", values=feat)
        if 0 not in pivot.columns or 1 not in pivot.columns:
            continue

        delta = (pivot[1] - pivot[0]).dropna().values  # pre - far

        if len(delta) == 0:
            continue

        n = len(delta)
        med_delta = float(np.median(delta))
        q25, q75 = float(np.percentile(delta, 25)), float(np.percentile(delta, 75))
        iqr_delta = q75 - q25
        pct_pos = float(np.mean(delta > 0))
        pct_neg = float(np.mean(delta < 0))

        try:
            stat_w, p_wilcoxon = stats.wilcoxon(delta, alternative="two-sided")
        except Exception:
            stat_w, p_wilcoxon = np.nan, np.nan

        r_rb = _rank_biserial(delta)

        rows.append({
            "feature": feat,
            "n_pairs": n,
            "median_delta": round(med_delta, 4),
            "iqr_delta": round(iqr_delta, 4),
            "q25_delta": round(q25, 4),
            "q75_delta": round(q75, 4),
            "pct_pairs_delta_positive": round(pct_pos, 3),
            "pct_pairs_delta_negative": round(pct_neg, 3),
            "wilcoxon_stat": round(float(stat_w), 3) if np.isfinite(stat_w) else np.nan,
            "wilcoxon_p": round(float(p_wilcoxon), 5) if np.isfinite(p_wilcoxon) else np.nan,
            "rank_biserial_r": round(r_rb, 3) if np.isfinite(r_rb) else np.nan,
            "direction_consistent": pct_pos >= 0.60,
            # EXPLÍCITAMENTE sin columna 'auc': el marco delta no tiene dos clases
        })

    df_out = pd.DataFrame(rows).sort_values("rank_biserial_r", key=abs, ascending=False)
    return df_out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Pairwise ranking evaluation
# ---------------------------------------------------------------------------

def _fold_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auc = np.nan
    return {
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": spec,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n_val": len(y_true),
    }


def pairwise_ranking_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    model: Any,
    model_name: str,
    block_name: str,
    window_type: str,
    window_size: Any,
    *,
    use_pruning: bool = False,
    pruning_threshold: float = 0.90,
) -> pd.DataFrame:
    """
    GroupKFold evaluación con métrica de ranking intra-par.

    Para cada par (pair_id) en validation:
      - score_far  = model.predict_proba(X_far)[:,1]
      - score_pre  = model.predict_proba(X_pre)[:,1]
      - concordant = (score_pre > score_far)
      - margin     = score_pre - score_far

    pairwise_concordance = media(concordant) sobre todos los pares de val.

    Si use_pruning=True, aplica CorrelationPruner DENTRO del fold (solo en train).

    Regla anti-leakage verificada:
      - Pruner ajustado solo en X_train.
      - Mismas features seleccionadas aplicadas a X_val.
      - Ningún pair_id en train y val simultáneamente.
    """
    avail = [c for c in feature_cols if c in df.columns]
    if len(avail) < 1:
        return pd.DataFrame()

    X_all = df[avail]
    y_all = df["y"].astype(int)
    groups = df["pair_id"].astype(int)
    n_splits = min(GKF_N_SPLITS, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    fold_rows = []

    for fold, (tr, va) in enumerate(gkf.split(X_all, y_all, groups=groups), start=1):
        g_tr = set(groups.iloc[tr].unique())
        g_va = set(groups.iloc[va].unique())
        assert g_tr.isdisjoint(g_va), f"Leakage pair_id en fold {fold}"

        X_tr_raw = X_all.iloc[tr].copy()
        X_va_raw = X_all.iloc[va].copy()
        y_tr = y_all.iloc[tr].values
        y_va = y_all.iloc[va].values
        g_va_arr = groups.iloc[va].values

        # Correlation pruning DENTRO del fold (solo con datos train)
        if use_pruning:
            pruner = CorrelationPruner(threshold=pruning_threshold)
            X_tr_raw = pruner.fit(X_tr_raw).transform(X_tr_raw)
            X_va_raw = pruner.transform(X_va_raw)

        # Ajustar modelo en train
        pipe = clone(model)
        pipe.fit(X_tr_raw, y_tr)

        # Probabilidades en val
        prob_va = pipe.predict_proba(X_va_raw)[:, 1]
        pred_va = (prob_va >= DECISION_THRESHOLD).astype(int)

        # Métricas estándar
        std_metrics = _fold_metrics(y_va, pred_va, prob_va)

        # ── Pairwise ranking (métrica principal) ──
        pair_concordances = []
        pair_margins = []

        for pid in np.unique(g_va_arr):
            idx_far = np.where((g_va_arr == pid) & (y_va == 0))[0]
            idx_pre = np.where((g_va_arr == pid) & (y_va == 1))[0]
            if len(idx_far) == 0 or len(idx_pre) == 0:
                continue
            s_far = float(np.mean(prob_va[idx_far]))
            s_pre = float(np.mean(prob_va[idx_pre]))
            pair_concordances.append(int(s_pre > s_far))
            pair_margins.append(round(s_pre - s_far, 4))

        pairwise_concordance = float(np.mean(pair_concordances)) if pair_concordances else np.nan
        mean_margin = float(np.mean(pair_margins)) if pair_margins else np.nan
        n_pairs_val = len(pair_concordances)

        row = {
            "fold": fold,
            "model": model_name,
            "block": block_name,
            "window_type": str(window_type),
            "window_size": window_size,
            "use_pruning": use_pruning,
            "pairwise_concordance": round(pairwise_concordance, 4),
            "mean_margin": round(mean_margin, 4),
            "n_pairs_val": n_pairs_val,
        }
        row.update(std_metrics)
        fold_rows.append(row)

    return pd.DataFrame(fold_rows)


# ---------------------------------------------------------------------------
# 3. LogReg con reporte de coeficientes por fold
# ---------------------------------------------------------------------------

def logreg_coef_stability(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    use_pruning: bool = False,
    pruning_threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Entrena LogReg en cada fold de GroupKFold y reporta coeficientes.
    Evalúa estabilidad del signo: % de folds donde el signo es positivo.
    Si use_pruning=True, aplica CorrelationPruner dentro de cada fold.
    """
    avail = [c for c in feature_cols if c in df.columns]
    X_all = df[avail]
    y_all = df["y"].astype(int)
    groups = df["pair_id"].astype(int)
    n_splits = min(GKF_N_SPLITS, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    coef_by_fold: list[dict] = []

    for fold, (tr, va) in enumerate(gkf.split(X_all, y_all, groups=groups), start=1):
        X_tr = X_all.iloc[tr].copy()
        y_tr = y_all.iloc[tr].values

        if use_pruning:
            pruner = CorrelationPruner(threshold=pruning_threshold)
            X_tr = pruner.fit(X_tr).transform(X_tr)
            feat_used = list(X_tr.columns)
        else:
            feat_used = avail

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_tr, y_tr)
        coef = pipe.named_steps["clf"].coef_.ravel()

        for fname, c in zip(feat_used, coef):
            coef_by_fold.append({"fold": fold, "feature": fname, "coef": float(c)})

    if not coef_by_fold:
        return pd.DataFrame()

    coef_df = pd.DataFrame(coef_by_fold)
    summary = (
        coef_df.groupby("feature")["coef"]
        .agg(
            mean_coef="mean",
            std_coef="std",
            n_folds="count",
        )
        .reset_index()
    )
    # Porcentaje de folds con coef positivo
    pct_positive = (
        coef_df.groupby("feature")["coef"]
        .apply(lambda x: float(np.mean(x > 0)))
        .rename("pct_folds_positive")
        .reset_index()
    )
    summary = summary.merge(pct_positive, on="feature")
    summary["sign_stable"] = (
        (summary["pct_folds_positive"] >= 0.80) | (summary["pct_folds_positive"] <= 0.20)
    )
    summary["use_pruning"] = use_pruning
    return summary.sort_values("mean_coef", key=abs, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Evaluación pairwise ranking multi-bloque
# ---------------------------------------------------------------------------

def evaluate_pairwise_ranking_all_blocks(
    df: pd.DataFrame,
    window_type: str,
    window_size: Any,
    *,
    blocks: dict[str, list[str]] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evalúa pairwise_ranking_cv para múltiples bloques de features.
    Compara versiones con y sin CorrelationPruner.
    """
    if blocks is None:
        blocks = {k: v for k, v in INTERPRETABLE_BLOCKS.items()
                  if k in ("hrv_reduced_interpretable", "B_SPC", "C_HRV_SPC", "D_HRV_SPC_Poincare")}

    models = {
        "LogReg": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
    }

    all_rows = []
    for block_name, feat_list in blocks.items():
        for mname, model in models.items():
            for use_pruning in (False, True):
                label = f"{block_name}{'_pruned' if use_pruning else ''}"
                if verbose:
                    print(f"  pairwise ranking | {wt_fmt(window_type, window_size)} | {label}", flush=True)
                res = pairwise_ranking_cv(
                    df, feat_list, model, mname, label,
                    window_type, window_size,
                    use_pruning=use_pruning,
                )
                if not res.empty:
                    all_rows.append(res)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def wt_fmt(wt, ws) -> str:
    return f"{wt} {ws}"
