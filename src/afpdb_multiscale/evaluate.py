"""
Evaluación: GroupKFold por pair_id, múltiples bloques de features y modelos.

Reglas metodológicas:
  - Ningún pair_id aparece simultáneamente en train y val.
  - StandardScaler, imputer y selección de features solo dentro del fold.
  - Umbral de decisión fijo (no ajustado sobre test pooled).
  - lead_time se reporta solo para y=1; false_alarms_per_hour solo para y=0.
  - random_state fijo para reproducibilidad.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.svm import LinearSVC

from .config import (
    BLOCKS,
    DECISION_THRESHOLD,
    GKF_N_SPLITS,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def make_logreg() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000, random_state=RANDOM_STATE, solver="lbfgs"
        )),
    ])


def make_rf() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=100, max_depth=5,
            random_state=RANDOM_STATE, n_jobs=1,
        )),
    ])


def make_svm_linear() -> Pipeline:
    base = LinearSVC(random_state=RANDOM_STATE, max_iter=8000, dual=False)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, cv=3, method="sigmoid")),
    ])


MODELS: dict[str, Pipeline] = {
    "LogReg": make_logreg(),
    "RandomForest": make_rf(),
    "SVM_linear": make_svm_linear(),
}


# ---------------------------------------------------------------------------
# Métricas por fold
# ---------------------------------------------------------------------------

def _compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    balacc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auc = np.nan

    return {
        "auc": auc,
        "balanced_accuracy": balacc,
        "sensitivity": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "mcc": mcc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_val": n,
    }


# ---------------------------------------------------------------------------
# CV por grupos
# ---------------------------------------------------------------------------

def _run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model: Pipeline,
    model_name: str,
    block_name: str,
    window_type: str,
    window_size: Any,
    feature_cols: list[str],
) -> pd.DataFrame:
    gkf = GroupKFold(n_splits=min(GKF_N_SPLITS, len(groups.unique())))
    fold_rows = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        g_tr = set(groups.iloc[tr].unique())
        g_va = set(groups.iloc[va].unique())
        # Anti-leakage: ningún pair_id puede estar en ambos lados
        assert g_tr.isdisjoint(g_va), (
            f"LEAKAGE detectado: pair_id compartido en fold {fold}."
        )

        # Features disponibles en la ventana actual
        avail = [c for c in feature_cols if c in X.columns]
        if not avail:
            continue

        Xtr = X.iloc[tr][avail]
        Xva = X.iloc[va][avail]
        ytr = y.iloc[tr].to_numpy(dtype=int)
        yva = y.iloc[va].to_numpy(dtype=int)

        pipe = clone(model)
        pipe.fit(Xtr, ytr)

        y_prob = pipe.predict_proba(Xva)[:, 1]
        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

        metrics = _compute_fold_metrics(yva, y_pred, y_prob)
        metrics.update({
            "fold": fold,
            "model": model_name,
            "block": block_name,
            "window_type": window_type,
            "window_size": window_size,
            "n_features_used": len(avail),
        })
        fold_rows.append(metrics)

    return pd.DataFrame(fold_rows)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _get_feature_importance(
    model: Pipeline,
    feature_cols: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    block_name: str,
    window_type: str,
    window_size: Any,
) -> pd.DataFrame:
    avail = [c for c in feature_cols if c in X.columns]
    if not avail:
        return pd.DataFrame()

    pipe = clone(model)
    pipe.fit(X[avail], y)

    clf = pipe.named_steps.get("clf", None)
    rows = []

    if hasattr(clf, "coef_"):
        coefs = np.asarray(clf.coef_).ravel()
        for fname, c in zip(avail, coefs):
            rows.append({
                "feature": fname, "importance": float(c),
                "importance_type": "coefficient_scaled",
                "model": model_name, "block": block_name,
                "window_type": window_type, "window_size": window_size,
            })
    elif hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
        for fname, imp in zip(avail, imps):
            rows.append({
                "feature": fname, "importance": float(imp),
                "importance_type": "gini_importance",
                "model": model_name, "block": block_name,
                "window_type": window_type, "window_size": window_size,
            })
    elif hasattr(clf, "calibrated_classifiers_"):
        # CalibratedClassifierCV sobre LinearSVC
        coefs_list = [
            cc.estimator.coef_.ravel()
            for cc in clf.calibrated_classifiers_
            if hasattr(cc.estimator, "coef_")
        ]
        if coefs_list:
            mean_coef = np.mean(coefs_list, axis=0)
            for fname, c in zip(avail, mean_coef):
                rows.append({
                    "feature": fname, "importance": float(c),
                    "importance_type": "svm_coef_mean",
                    "model": model_name, "block": block_name,
                    "window_type": window_type, "window_size": window_size,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluación completa por ventana × bloque × modelo
# ---------------------------------------------------------------------------

def evaluate_all(df_agg: pd.DataFrame, task_label: str = "main") -> dict[str, pd.DataFrame]:
    """
    Itera sobre (window_type, window_size) × BLOCKS × MODELS.
    Devuelve un dict con DataFrames:
      'cv_detail'      : métricas por fold
      'cv_summary'     : media ± std por (window, bloque, modelo)
      'importance'     : importancia de variables
      'incremental'    : comparación de bloques (A vs C, B vs C, C vs D, etc.)
    """
    cv_rows = []
    imp_rows = []

    window_groups = df_agg.groupby(["window_type", "window_size"])

    for (wtype, wsize), sub in window_groups:
        sub = sub.reset_index(drop=True)
        X = sub.drop(columns=[
            "record_id", "pair_id", "y", "analysis_task",
            "window_type", "window_size", "end_time_sec", "n_rr",
            "lead_time_sec", "false_alarms_per_hour",
        ], errors="ignore")
        y = sub["y"].astype(int)
        groups = sub["pair_id"].astype(int)

        for block_name, feat_list in BLOCKS.items():
            for model_name, model in MODELS.items():
                fold_df = _run_cv(
                    X, y, groups, model, model_name,
                    block_name, str(wtype), wsize, feat_list,
                )
                cv_rows.append(fold_df)

                # Importancia solo con LogReg sobre bloque completo
                if model_name == "LogReg":
                    avail = [c for c in feat_list if c in X.columns]
                    imp_df = _get_feature_importance(
                        model, feat_list, X[avail] if avail else X,
                        y, model_name, block_name, str(wtype), wsize,
                    )
                    imp_rows.append(imp_df)

    cv_detail = pd.concat(cv_rows, ignore_index=True) if cv_rows else pd.DataFrame()
    importance = pd.concat(imp_rows, ignore_index=True) if imp_rows else pd.DataFrame()

    # Resumen: media ± std por (window_type, window_size, block, model)
    if not cv_detail.empty:
        num_cols = ["auc", "balanced_accuracy", "sensitivity",
                    "specificity", "precision", "f1", "mcc"]
        cv_summary = (
            cv_detail.groupby(["window_type", "window_size", "block", "model"])[num_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        cv_summary.columns = [
            "_".join(c).strip("_") for c in cv_summary.columns.values
        ]
    else:
        cv_summary = pd.DataFrame()

    # Tabla incremental (comparación de bloques, solo LogReg)
    incremental_rows = []
    if not cv_detail.empty:
        lr = cv_detail[cv_detail["model"] == "LogReg"]
        comparisons = [
            ("A_HRV", "B_SPC"),
            ("A_HRV", "C_HRV_SPC"),
            ("B_SPC", "C_HRV_SPC"),
            ("C_HRV_SPC", "D_HRV_SPC_Poincare"),
            ("D_HRV_SPC_Poincare", "E_HRV_SPC_Poincare_Ectopy"),
        ]
        for wt, ws in cv_detail[["window_type", "window_size"]].drop_duplicates().itertuples(index=False):
            sub_lr = lr[(lr["window_type"] == str(wt)) & (lr["window_size"] == ws)]
            for b1, b2 in comparisons:
                auc1 = sub_lr[sub_lr["block"] == b1]["auc"].mean()
                auc2 = sub_lr[sub_lr["block"] == b2]["auc"].mean()
                f1_1 = sub_lr[sub_lr["block"] == b1]["f1"].mean()
                f1_2 = sub_lr[sub_lr["block"] == b2]["f1"].mean()
                incremental_rows.append({
                    "window_type": wt,
                    "window_size": ws,
                    "block_A": b1,
                    "block_B": b2,
                    "delta_auc": round(float(auc2 - auc1), 4) if np.isfinite(auc1) and np.isfinite(auc2) else np.nan,
                    "delta_f1": round(float(f1_2 - f1_1), 4) if np.isfinite(f1_1) and np.isfinite(f1_2) else np.nan,
                })
    incremental = pd.DataFrame(incremental_rows)

    return {
        "cv_detail": cv_detail,
        "cv_summary": cv_summary,
        "importance": importance,
        "incremental": incremental,
    }
