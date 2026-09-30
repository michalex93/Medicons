"""
Evaluación del módulo Mahalanobis sobre AFPDB far vs pre-onset.

Dos pipelines de evaluación:
  population_fold  — GroupKFold por pair_id; μ/Σ estimados en y=0 de cada train fold.
  pair_baseline    — Análisis intra-par; μ/Σ del registro far del mismo sujeto.

Bloques evaluados:
  A  HRV básico
  B  SPC puro
  C  HRV + SPC
  D  Mahalanobis solo (agregados D_M)
  E  HRV + SPC + Mahalanobis

Modelos:
  LogisticRegression, RandomForest, MahalanobisThresholdClassifier (umbralizador puro).

Reglas anti-leakage verificadas dentro del código:
  - μ, Σ, imputer, umbral de alarma → solo train y=0.
  - Escalado/imputación de features HRV/SPC → solo train fold.
  - Ningún pair_id aparece simultáneamente en train y val.
  - 30 s no entra en ningún análisis (heredado de config.PROHIBITED_WINDOW_SEC).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
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

from .config import (
    BLOCKS,
    DECISION_THRESHOLD,
    GKF_N_SPLITS,
    HRV_BASIC_FEATURES,
    POINCARE_FEATURES,
    RANDOM_STATE,
    SPC_FEATURES,
)
from .features_mahalanobis import (
    DEFAULT_ALARM_PERCENTILE,
    MAHAL_FEATURE_NAMES,
    MAHAL_FEATURE_SETS,
    MAHAL_LEAD_TIME_FEATURE,
    alarm_threshold_from_ref,
    compute_pair_baseline_dm,
    compute_population_fold_dm,
    mahalanobis_distances,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Bloques de features extendidos con Mahalanobis
# ---------------------------------------------------------------------------

MAHAL_BLOCKS: dict[str, list[str]] = {
    "A_HRV": HRV_BASIC_FEATURES,
    "B_SPC": SPC_FEATURES,
    "C_HRV_SPC": HRV_BASIC_FEATURES + SPC_FEATURES,
    "D_Mahal": MAHAL_FEATURE_NAMES,
    "E_HRV_SPC_Mahal": HRV_BASIC_FEATURES + SPC_FEATURES + MAHAL_FEATURE_NAMES,
}


# ---------------------------------------------------------------------------
# Clasificador puro por umbral Mahalanobis (D_max > thr → y=1)
# ---------------------------------------------------------------------------

class MahalanobisThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador por umbral en `mahalanobis_max` del registro.

    El umbral se define como el percentil `train_percentile` de los valores
    de `mahalanobis_max` en los registros y=0 del train fold.
    No usa datos de validación para definir el umbral.
    """

    def __init__(
        self,
        feature: str = "mahalanobis_max",
        train_percentile: float = DEFAULT_ALARM_PERCENTILE,
    ):
        self.feature = feature
        self.train_percentile = train_percentile
        self.threshold_: float = float("nan")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MahalanobisThresholdClassifier":
        y_arr = np.asarray(y, dtype=int)
        if self.feature not in X.columns:
            raise ValueError(f"Feature '{self.feature}' no está en X.")
        far_vals = X.loc[y_arr == 0, self.feature].dropna().values
        if len(far_vals) == 0:
            self.threshold_ = float("inf")
        else:
            self.threshold_ = float(np.percentile(far_vals, self.train_percentile))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        vals = X[self.feature].fillna(0.0).values.astype(float)
        thr = self.threshold_ if np.isfinite(self.threshold_) else 1.0
        # Probabilidad sigmoide escalada alrededor del umbral
        # p(y=1) ≈ σ(scale * (D - thr))
        scale = 2.0 / max(thr, 1e-6)
        p1 = 1.0 / (1.0 + np.exp(-scale * (vals - thr)))
        p1 = np.clip(p1, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= DECISION_THRESHOLD).astype(int)


# ---------------------------------------------------------------------------
# Métricas por fold
# ---------------------------------------------------------------------------

def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    far_ph = np.nan  # false_alarms_per_hour se calcula post-hoc
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
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_val": len(y_true),
    }


def _make_sklearn_pipe(model_name: str) -> Pipeline:
    if model_name == "LogReg":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ])
    if model_name == "RandomForest":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=6,
                random_state=RANDOM_STATE, n_jobs=1,
            )),
        ])
    raise ValueError(f"Modelo desconocido: {model_name}")


# ---------------------------------------------------------------------------
# Evaluación: population_fold
# ---------------------------------------------------------------------------

def run_population_fold_cv(
    df_windows: pd.DataFrame,
    df_agg: pd.DataFrame,
    mahal_feature_set_name: str,
    alarm_percentile: float = DEFAULT_ALARM_PERCENTILE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Para cada (window_type, window_size):
      1. GroupKFold por pair_id sobre registros.
      2. En cada fold: ajusta CovarianceFit SOLO en y=0 train windows.
      3. Calcula D_M para todos los registros (train + val) con esa covarianza.
      4. Agrega D_M por registro.
      5. Combina con features HRV+SPC del dataset agregado.
      6. Evalúa 5 bloques × 3 modelos.

    Ningún dato de validation participa en ajuste de μ, Σ, umbral ni escalado.
    """
    feat_cols = MAHAL_FEATURE_SETS[mahal_feature_set_name]
    all_rows: list[dict] = []

    for (wtype, wsize), sub_w in df_windows.groupby(["window_type", "window_size"]):
        if verbose:
            print(f"  Mahal population_fold | {wtype} {wsize} | {mahal_feature_set_name}")

        # Registros únicos para este window_type/size
        recs = (
            sub_w[["record_id", "pair_id", "y"]]
            .drop_duplicates("record_id")
            .reset_index(drop=True)
        )
        if recs["pair_id"].nunique() < GKF_N_SPLITS:
            n_splits = max(2, recs["pair_id"].nunique())
        else:
            n_splits = GKF_N_SPLITS

        gkf = GroupKFold(n_splits=n_splits)
        y_by_record = dict(zip(recs["record_id"], recs["y"]))

        for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(recs, recs["y"], recs["pair_id"]), start=1
        ):
            tr_recs = set(recs.iloc[tr_idx]["record_id"])
            va_recs = set(recs.iloc[va_idx]["record_id"])
            tr_pairs = set(recs.iloc[tr_idx]["pair_id"])
            va_pairs = set(recs.iloc[va_idx]["pair_id"])
            assert tr_pairs.isdisjoint(va_pairs), (
                f"LEAKAGE pair_id en fold {fold} | {wtype} {wsize}"
            )

            # Sub-dataframe filtrado a este window_type/size
            sub_win_wtype = df_windows[
                (df_windows["window_type"] == wtype)
                & (df_windows["window_size"] == wsize)
            ]

            try:
                dm_all, cov_fit, imputer, alarm_thr = compute_population_fold_dm(
                    sub_win_wtype,
                    feat_cols,
                    tr_recs,
                    va_recs,
                    y_by_record,
                    alarm_percentile,
                )
            except Exception as exc:
                warnings.warn(f"compute_population_fold_dm fallo: {exc!r}", UserWarning)
                continue

            # DataFrame D_M por registro (solo val)
            dm_val_rows = []
            for rid in va_recs:
                if rid not in dm_all:
                    continue
                row = {"record_id": rid, "y": y_by_record[rid]}
                row.update(dm_all[rid])
                dm_val_rows.append(row)
            if not dm_val_rows:
                continue

            dm_val_df = pd.DataFrame(dm_val_rows)

            # Merge con features agregadas HRV+SPC del val fold
            agg_val = df_agg[
                df_agg["record_id"].isin(va_recs)
                & (df_agg["window_type"] == wtype)
                & (df_agg["window_size"] == wsize)
            ].reset_index(drop=True)
            if agg_val.empty:
                continue
            merged_val = agg_val.merge(
                dm_val_df[["record_id"] + MAHAL_FEATURE_NAMES + [MAHAL_LEAD_TIME_FEATURE]],
                on="record_id", how="left",
            )

            # Mismo merge para train (para ajustar modelos con features unificadas)
            dm_train_rows = []
            for rid in tr_recs:
                if rid not in dm_all:
                    continue
                row = {"record_id": rid, "y": y_by_record[rid]}
                row.update(dm_all[rid])
                dm_train_rows.append(row)
            dm_train_df = pd.DataFrame(dm_train_rows) if dm_train_rows else pd.DataFrame()

            agg_train = df_agg[
                df_agg["record_id"].isin(tr_recs)
                & (df_agg["window_type"] == wtype)
                & (df_agg["window_size"] == wsize)
            ].reset_index(drop=True)

            if not dm_train_df.empty and not agg_train.empty:
                merged_train = agg_train.merge(
                    dm_train_df[["record_id"] + MAHAL_FEATURE_NAMES + [MAHAL_LEAD_TIME_FEATURE]],
                    on="record_id", how="left",
                )
            else:
                merged_train = agg_train.copy()

            y_tr = merged_train["y"].astype(int).values
            y_va = merged_val["y"].astype(int).values

            # Evaluación por bloque × modelo
            for block_name, block_feats in MAHAL_BLOCKS.items():
                avail_tr = [c for c in block_feats if c in merged_train.columns]
                avail_va = [c for c in block_feats if c in merged_val.columns]
                avail = [c for c in avail_tr if c in avail_va]
                if not avail or len(avail) < 1:
                    continue

                Xtr = merged_train[avail]
                Xva = merged_val[avail]

                # sklearn models
                for mname in ("LogReg", "RandomForest"):
                    pipe = _make_sklearn_pipe(mname)
                    try:
                        pipe.fit(Xtr, y_tr)
                        prob = pipe.predict_proba(Xva)[:, 1]
                        pred = (prob >= DECISION_THRESHOLD).astype(int)
                        m = _fold_metrics(y_va, pred, prob)
                        m.update({
                            "fold": fold, "model": mname, "block": block_name,
                            "mode": "population_fold",
                            "mahal_feature_set": mahal_feature_set_name,
                            "window_type": str(wtype), "window_size": wsize,
                            "alarm_percentile": alarm_percentile,
                            "covariance_method": cov_fit.method,
                        })
                        all_rows.append(m)
                    except Exception as exc:
                        warnings.warn(f"{mname} {block_name}: {exc!r}", UserWarning)

                # Threshold classifier (solo para bloques con D_Mahal)
                if "mahalanobis_max" in avail:
                    thr_clf = MahalanobisThresholdClassifier(
                        train_percentile=alarm_percentile
                    )
                    try:
                        thr_clf.fit(Xtr, y_tr)
                        prob_t = thr_clf.predict_proba(Xva)[:, 1]
                        pred_t = (prob_t >= DECISION_THRESHOLD).astype(int)
                        m = _fold_metrics(y_va, pred_t, prob_t)
                        m.update({
                            "fold": fold, "model": "Mahal_Threshold",
                            "block": block_name, "mode": "population_fold",
                            "mahal_feature_set": mahal_feature_set_name,
                            "window_type": str(wtype), "window_size": wsize,
                            "alarm_percentile": alarm_percentile,
                            "covariance_method": cov_fit.method,
                        })
                        all_rows.append(m)
                    except Exception:
                        pass

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Evaluación: pair_baseline
# ---------------------------------------------------------------------------

def run_pair_baseline_analysis(
    df_windows: pd.DataFrame,
    df_agg: pd.DataFrame,
    mahal_feature_set_name: str,
    alarm_percentile: float = DEFAULT_ALARM_PERCENTILE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Para cada par (pair_id):
      - Baseline: ventanas y=0 del par.
      - Evalúa D_M de ventanas y=1 respecto a ese baseline.

    ADVERTENCIA: Asume disponibilidad de segmento basal far del mismo sujeto.
    No es un modelo poblacional; tampoco se mezcla con datos de otros pares
    para estimar la covarianza.

    Evaluación: GroupKFold por pair_id sobre los D_M aggregados.
    Ningún pair_id aparece en train y val simultáneamente.
    """
    feat_cols = MAHAL_FEATURE_SETS[mahal_feature_set_name]
    dm_record_rows: list[dict] = []

    pair_ids = df_windows["pair_id"].unique()
    for pid in sorted(pair_ids):
        try:
            # filter to first window_type/size for pair analysis
            sub = df_windows[df_windows["pair_id"] == pid]
            # compute for each window_type/size combination
            for (wt, ws), sub_sub in sub.groupby(["window_type", "window_size"]):
                dm_results = compute_pair_baseline_dm(
                    sub_sub, feat_cols, pid, alarm_percentile
                )
                for rid, dm_dict in dm_results.items():
                    y_val = int(sub_sub[sub_sub["record_id"] == rid]["y"].iloc[0])
                    row = {
                        "record_id": rid, "pair_id": pid, "y": y_val,
                        "window_type": str(wt), "window_size": ws,
                        "mode": "pair_baseline",
                        "mahal_feature_set": mahal_feature_set_name,
                    }
                    row.update(dm_dict)
                    dm_record_rows.append(row)
        except Exception as exc:
            warnings.warn(
                f"pair_baseline pair_id={pid}: {exc!r}. Saltando.", UserWarning
            )

    if not dm_record_rows:
        return pd.DataFrame()

    dm_df = pd.DataFrame(dm_record_rows)

    # GroupKFold por pair_id para evaluar la separabilidad
    all_rows: list[dict] = []
    for (wt, ws), sub in dm_df.groupby(["window_type", "window_size"]):
        sub = sub.reset_index(drop=True)
        n_splits = min(GKF_N_SPLITS, sub["pair_id"].nunique())
        if n_splits < 2:
            continue
        gkf = GroupKFold(n_splits=n_splits)
        for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(sub, sub["y"], sub["pair_id"]), start=1
        ):
            tr_pairs = set(sub.iloc[tr_idx]["pair_id"])
            va_pairs = set(sub.iloc[va_idx]["pair_id"])
            assert tr_pairs.isdisjoint(va_pairs), f"Leakage pair_baseline fold {fold}"

            Xtr = sub.iloc[tr_idx][MAHAL_FEATURE_NAMES].fillna(0)
            Xva = sub.iloc[va_idx][MAHAL_FEATURE_NAMES].fillna(0)
            y_tr = sub.iloc[tr_idx]["y"].astype(int).values
            y_va = sub.iloc[va_idx]["y"].astype(int).values

            for mname in ("LogReg", "RandomForest"):
                pipe = _make_sklearn_pipe(mname)
                try:
                    pipe.fit(Xtr, y_tr)
                    prob = pipe.predict_proba(Xva)[:, 1]
                    pred = (prob >= DECISION_THRESHOLD).astype(int)
                    m = _fold_metrics(y_va, pred, prob)
                    m.update({
                        "fold": fold, "model": mname,
                        "block": "D_Mahal",
                        "mode": "pair_baseline",
                        "mahal_feature_set": mahal_feature_set_name,
                        "window_type": str(wt), "window_size": ws,
                        "alarm_percentile": alarm_percentile,
                    })
                    all_rows.append(m)
                except Exception:
                    pass

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Evaluación completa Mahalanobis
# ---------------------------------------------------------------------------

def evaluate_mahalanobis_all(
    df_windows: pd.DataFrame,
    df_agg: pd.DataFrame,
    alarm_percentile: float = DEFAULT_ALARM_PERCENTILE,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Ejecuta population_fold y pair_baseline para los 3 feature sets.
    Devuelve dict con DataFrames:
      'cv_detail'    : métricas por fold
      'cv_summary'   : media ± std por (window, mode, block, model, feature_set)
      'incremental'  : delta_AUC y delta_F1 entre bloques C y E
      'lead_time'    : distribución de lead_time para y=1 (solo population_fold)
    """
    pop_rows, pair_rows = [], []

    for fs_name in MAHAL_FEATURE_SETS:
        if verbose:
            print(f"\n--- Feature set: {fs_name} ---")
        pop = run_population_fold_cv(
            df_windows, df_agg, fs_name, alarm_percentile, verbose=verbose
        )
        if not pop.empty:
            pop_rows.append(pop)

        pb = run_pair_baseline_analysis(
            df_windows, df_agg, fs_name, alarm_percentile, verbose=verbose
        )
        if not pb.empty:
            pair_rows.append(pb)

    cv_detail = pd.concat(pop_rows + pair_rows, ignore_index=True) \
        if (pop_rows or pair_rows) else pd.DataFrame()

    if cv_detail.empty:
        return {
            "cv_detail": cv_detail,
            "cv_summary": pd.DataFrame(),
            "incremental": pd.DataFrame(),
            "lead_time": pd.DataFrame(),
        }

    # Resumen
    num_cols = ["auc", "balanced_accuracy", "sensitivity", "specificity",
                "precision", "f1", "mcc"]
    group_cols = ["window_type", "window_size", "mode", "block",
                  "model", "mahal_feature_set"]
    valid_groups = [c for c in group_cols if c in cv_detail.columns]

    cv_summary = (
        cv_detail.groupby(valid_groups)[num_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    cv_summary.columns = [
        "_".join(c).strip("_") for c in cv_summary.columns.values
    ]

    # Incremental: bloque C (HRV+SPC) vs E (HRV+SPC+Mahal)
    inc_rows = []
    for keys, sub in cv_detail.groupby(["window_type", "window_size", "mode", "mahal_feature_set", "model"]):
        auc_C = sub[sub["block"] == "C_HRV_SPC"]["auc"].mean()
        auc_E = sub[sub["block"] == "E_HRV_SPC_Mahal"]["auc"].mean()
        f1_C = sub[sub["block"] == "C_HRV_SPC"]["f1"].mean()
        f1_E = sub[sub["block"] == "E_HRV_SPC_Mahal"]["f1"].mean()
        inc_rows.append({
            "window_type": keys[0], "window_size": keys[1],
            "mode": keys[2], "mahal_feature_set": keys[3], "model": keys[4],
            "block_base": "C_HRV_SPC", "block_new": "E_HRV_SPC_Mahal",
            "delta_auc": round(float(auc_E - auc_C), 4)
                if np.isfinite(auc_C) and np.isfinite(auc_E) else np.nan,
            "delta_f1": round(float(f1_E - f1_C), 4)
                if np.isfinite(f1_C) and np.isfinite(f1_E) else np.nan,
        })
    incremental = pd.DataFrame(inc_rows)

    # Lead time: del campo mahalanobis_lead_time_sec en df_windows aggregado
    lt_rows = []
    for fs_name in MAHAL_FEATURE_SETS:
        feat_cols = MAHAL_FEATURE_SETS[fs_name]
        for (wt, ws), sub_w in df_windows.groupby(["window_type", "window_size"]):
            avail = [c for c in feat_cols if c in sub_w.columns]
            if not avail:
                continue
            recs = sub_w[["record_id", "pair_id", "y"]].drop_duplicates("record_id")
            y0_recs = set(recs[recs["y"] == 0]["record_id"])
            y1_recs = set(recs[recs["y"] == 1]["record_id"])
            if not y0_recs:
                continue
            from sklearn.impute import SimpleImputer as _SI
            X_far = sub_w[sub_w["record_id"].isin(y0_recs)][avail].values.astype(float)
            imp = _SI(strategy="median")
            X_far_imp = imp.fit_transform(X_far)
            from .features_mahalanobis import fit_ledoit_wolf as _flw
            try:
                cov = _flw(X_far_imp)
            except Exception:
                continue
            dm_far = mahalanobis_distances(X_far_imp, cov)
            thr = alarm_threshold_from_ref(dm_far, DEFAULT_ALARM_PERCENTILE)
            for rid in y1_recs:
                rec_w = sub_w[sub_w["record_id"] == rid].sort_values("window_index")
                if rec_w.empty:
                    continue
                X_pre = rec_w[avail].values.astype(float)
                dm_pre = mahalanobis_distances(X_pre, cov, imputer_train=imp)
                alarms = dm_pre > thr
                first_idx = int(np.argmax(alarms)) if alarms.any() else -1
                first_t = float(rec_w["start_time_sec"].iloc[first_idx]) \
                    if first_idx >= 0 else np.nan
                lt = float(1800.0 - first_t) if np.isfinite(first_t) else np.nan
                lt_rows.append({
                    "record_id": rid, "y": 1,
                    "window_type": str(wt), "window_size": ws,
                    "mahal_feature_set": fs_name,
                    "lead_time_sec": lt,
                })
    lead_time_df = pd.DataFrame(lt_rows)

    return {
        "cv_detail": cv_detail,
        "cv_summary": cv_summary,
        "incremental": incremental,
        "lead_time": lead_time_df,
    }
