"""
Bootstrap por bloques de pair_id para IC 95% sobre métricas del pipeline AFPDB/PAF.

Regla crítica:
  El muestreo se hace a nivel de PAR (pair_id), no de ventana ni de registro individual.
  Esto preserva la estructura emparejada (far_baseline / pre_onset) del dataset.

Por qué pair_id y no registro individual:
  - Cada pair_id tiene exactamente 2 registros correlacionados (far y pre_onset).
  - Muestrear registros individuales rompería la estructura de pareo y
    subestimaría la varianza real entre sujetos.
  - El bootstrap por pair_id es un block bootstrap donde cada bloque = un sujeto.

Convención de etiquetas en este módulo:
  - y=0  → far_baseline (no "normal", no "healthy", no "control")
  - y=1  → pre_onset
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import DECISION_THRESHOLD

N_BOOTSTRAP_DEFAULT = 500
CONFIDENCE_DEFAULT = 0.95


# ---------------------------------------------------------------------------
# Métricas punto a punto
# ---------------------------------------------------------------------------

def _metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
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
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def _pairwise_concordance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pair_ids: np.ndarray,
) -> float:
    """
    Proporción de pares donde score_pre_onset > score_far_baseline.
    y=0 → far_baseline, y=1 → pre_onset.
    """
    concordances = []
    for pid in np.unique(pair_ids):
        mask = pair_ids == pid
        y_p = y_true[mask]
        p_p = y_prob[mask]
        far_idx = np.where(y_p == 0)[0]
        pre_idx = np.where(y_p == 1)[0]
        if len(far_idx) == 0 or len(pre_idx) == 0:
            continue
        s_far = float(np.mean(p_p[far_idx]))
        s_pre = float(np.mean(p_p[pre_idx]))
        concordances.append(int(s_pre > s_far))
    return float(np.mean(concordances)) if concordances else np.nan


# ---------------------------------------------------------------------------
# Bootstrap principal
# ---------------------------------------------------------------------------

def bootstrap_ci_by_pairs(
    y_oof: np.ndarray,
    p_oof: np.ndarray,
    pair_ids_oof: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    confidence: float = CONFIDENCE_DEFAULT,
    seed: int = 42,
) -> pd.DataFrame:
    """
    IC bootstrap por pair_id (block bootstrap).

    Parámetros
    ----------
    y_oof        : etiquetas verdaderas (0=far_baseline, 1=pre_onset)
    p_oof        : probabilidades predichas para y=1
    pair_ids_oof : pair_id de cada predicción
    n_bootstrap  : iteraciones bootstrap
    confidence   : nivel de confianza (default 0.95)

    Garantía anti-leakage: el muestreo es por pair_id (bloque), no por muestra.
    Esto respeta que far_baseline y pre_onset del mismo par están correlacionados.

    Devuelve DataFrame con [metric, observed, ci_low, ci_high, mean_boot, std_boot].
    """
    # Verificar etiquetas correctas (no "normal"/"healthy")
    unique_y = set(np.unique(y_oof))
    assert unique_y.issubset({0, 1}), f"Etiquetas inesperadas: {unique_y}"

    unique_pairs = np.unique(pair_ids_oof)
    n_pairs = len(unique_pairs)
    rng = np.random.default_rng(seed)

    alpha = 1.0 - confidence
    lo_pct = (alpha / 2) * 100
    hi_pct = (1.0 - alpha / 2) * 100

    boot_rows: list[dict] = []
    for _ in range(n_bootstrap):
        # Muestrear PARES con reemplazo
        sampled = rng.choice(unique_pairs, size=n_pairs, replace=True)
        idx = np.where(np.isin(pair_ids_oof, sampled))[0]
        y_b = y_oof[idx]
        p_b = p_oof[idx]
        g_b = pair_ids_oof[idx]

        if len(np.unique(y_b)) < 2:
            continue

        m = _metrics_at_threshold(y_b, p_b)
        m["pairwise_concordance"] = _pairwise_concordance(y_b, p_b, g_b)
        boot_rows.append(m)

    if not boot_rows:
        return pd.DataFrame()

    boot_df = pd.DataFrame(boot_rows)

    # Métricas observadas (sin bootstrap)
    obs = _metrics_at_threshold(y_oof, p_oof)
    obs["pairwise_concordance"] = _pairwise_concordance(y_oof, p_oof, pair_ids_oof)

    result_rows = []
    for metric in boot_df.columns:
        vals = boot_df[metric].dropna().values
        result_rows.append({
            "metric": metric,
            "observed": round(float(obs.get(metric, np.nan)), 4),
            "ci_low": round(float(np.percentile(vals, lo_pct)), 4) if len(vals) else np.nan,
            "ci_high": round(float(np.percentile(vals, hi_pct)), 4) if len(vals) else np.nan,
            "mean_boot": round(float(np.mean(vals)), 4) if len(vals) else np.nan,
            "std_boot": round(float(np.std(vals)), 4) if len(vals) else np.nan,
            "n_boot_valid": len(vals),
        })
    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# Bootstrap multi-bloque: recorre dict de OOF por configuración
# ---------------------------------------------------------------------------

def run_bootstrap_for_configs(
    oof_dict: dict[str, dict[str, Any]],
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = 42,
) -> pd.DataFrame:
    """
    oof_dict: {config_name: {'y': array, 'p': array, 'pairs': array}}
    Retorna DataFrame con IC 95% para cada configuración.
    """
    rows = []
    for config_name, oof in oof_dict.items():
        ci_df = bootstrap_ci_by_pairs(
            oof["y"], oof["p"], oof["pairs"],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        if ci_df.empty:
            continue
        ci_df.insert(0, "config", config_name)
        rows.append(ci_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
