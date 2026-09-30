"""
MEDICON 2026 — AFPDB tarea principal: clasificación far (impar) vs pre-onset (par)
según PhysioNet AFPDB (solo segmentos p* de 30 min, sin registros *c ni n*).

Ejecutar desde la raíz del proyecto:
  python run_medicon2026_afpdb.py
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.stats import median_abs_deviation
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

# --- Rutas ---
ROOT = Path(__file__).resolve().parent
AFPDB_DIR = ROOT / "afpdb"
OUT_DIR = ROOT / "outputs_medicon2026"
FIG_DIR = OUT_DIR / "figures"

# Umbral de decisión fijo (no ajustado en conjunto de test ni en validación pooled)
DECISION_THRESHOLD = 0.5

H_CUSUM = 7.0
K_CUSUM = 0.5
INNER_WINDOW_SEC = 30.0
BASELINE_WINDOWS = 10
GKF_SPLITS = 5
N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 42
RNG = np.random.default_rng(BOOTSTRAP_SEED)

ORIGINAL_6 = [
    "alarm_events",
    "pct_alarm_sustained",
    "max_cusum",
    "final_cusum",
    "median_rmssd",
    "median_z",
]
SPC_EXTRA = [
    "max_cusum_pos",
    "max_cusum_neg",
    "final_cusum_pos",
    "final_cusum_neg",
    "subwindow_sqi_good_frac",
    "rr_cv",
    "rr_masd_over_mean",
    "segment_rr_mad_ms",
]
SPC_FAST_14 = ORIGINAL_6 + SPC_EXTRA


def load_record_and_qrs(record_name: str, pn_dir: Path | str) -> tuple:
    """
    Solo cabecera + anotaciones QRS (sin cargar .dat): suficiente para RR y mucho más rápido.
    """
    pn_dir = os.fspath(pn_dir)
    header = wfdb.rdheader(record_name, pn_dir=pn_dir)
    fs = int(header.fs)
    ann = wfdb.rdann(record_name, extension="qrs", pn_dir=pn_dir)
    return header, fs, ann.sample


def compute_rr_ms(qrs_samples: np.ndarray, fs: int) -> np.ndarray:
    rr_samples = np.diff(qrs_samples)
    return (rr_samples / fs) * 1000.0


def window_rmssd(
    rr_ms: np.ndarray, rr_time_s: np.ndarray, window_sec: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    rmssd_values = []
    window_center_times = []
    current_window_start_time = 0.0
    max_time = float(rr_time_s[-1]) if len(rr_time_s) > 0 else 0.0
    while current_window_start_time < max_time:
        current_window_end_time = current_window_start_time + window_sec
        indices_in_window = np.where(
            (rr_time_s >= current_window_start_time) & (rr_time_s < current_window_end_time)
        )[0]
        if len(indices_in_window) > 1:
            rr_in_window = rr_ms[indices_in_window]
            diff_rr = np.diff(rr_in_window)
            rmssd = np.sqrt(np.mean(diff_rr**2)) if len(diff_rr) > 0 else np.nan
            rmssd_values.append(rmssd)
        else:
            rmssd_values.append(np.nan)
        window_center_times.append(current_window_start_time + window_sec / 2)
        current_window_start_time = current_window_end_time
    return np.array(rmssd_values), np.array(window_center_times)


def window_subwindow_sqi_gate(
    rr_ms: np.ndarray, rr_time_s: np.ndarray, window_sec: float = 30.0, n_sub: int = 3
) -> np.ndarray:
    """
    Sustituto autocontenido (sin af_predict): misma malla temporal que window_rmssd;
    ventana OK si cada subventana tiene al menos 3 intervalos RR.
    """
    flags = []
    current_window_start_time = 0.0
    max_time = float(rr_time_s[-1]) if len(rr_time_s) > 0 else 0.0
    sub_w = window_sec / float(n_sub)
    while current_window_start_time < max_time:
        current_window_end_time = current_window_start_time + window_sec
        ok_all = True
        for j in range(n_sub):
            ss = current_window_start_time + j * sub_w
            se = ss + sub_w
            idx = np.where((rr_time_s >= ss) & (rr_time_s < se))[0]
            if len(idx) < 3:
                ok_all = False
                break
        flags.append(ok_all)
        current_window_start_time = current_window_end_time
    return np.array(flags, dtype=bool)


def robust_z(rmssd_series: np.ndarray, baseline_windows: int = 10) -> np.ndarray:
    rmssd_series = np.asarray(rmssd_series)
    valid = ~np.isnan(rmssd_series)
    valid_vals = rmssd_series[valid]
    if len(valid_vals) < baseline_windows:
        return np.full_like(rmssd_series, np.nan)
    baseline_values = valid_vals[:baseline_windows]
    baseline_median = np.median(baseline_values)
    baseline_mad = median_abs_deviation(baseline_values)
    if baseline_mad == 0:
        baseline_mad = 1e-6
    z_scores = np.full_like(rmssd_series, np.nan, dtype=float)
    z_scores[valid] = (rmssd_series[valid] - baseline_median) / baseline_mad
    return z_scores


def cusum_bilateral(
    z: np.ndarray, k: float = 0.5, h: float = 7.0, sustained_windows: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(z)
    Splus = np.zeros(n)
    Sminus = np.zeros(n)
    alarms = np.full(n, False, dtype=bool)
    current_Splus = 0.0
    current_Sminus = 0.0
    consecutive_splus_alarms = 0
    consecutive_sminus_alarms = 0
    for i in range(n):
        if np.isnan(z[i]):
            Splus[i] = np.nan
            Sminus[i] = np.nan
            consecutive_splus_alarms = 0
            consecutive_sminus_alarms = 0
            continue
        current_Splus = max(0.0, current_Splus + z[i] - k)
        Splus[i] = current_Splus
        current_Sminus = max(0.0, current_Sminus - z[i] - k)
        Sminus[i] = current_Sminus
        is_splus_alarm = current_Splus > h
        is_sminus_alarm = current_Sminus > h
        consecutive_splus_alarms = consecutive_splus_alarms + 1 if is_splus_alarm else 0
        consecutive_sminus_alarms = consecutive_sminus_alarms + 1 if is_sminus_alarm else 0
        if consecutive_splus_alarms >= sustained_windows:
            for j in range(sustained_windows):
                if i - j >= 0:
                    alarms[i - j] = True
        if consecutive_sminus_alarms >= sustained_windows:
            for j in range(sustained_windows):
                if i - j >= 0:
                    alarms[i - j] = True
    return Splus, Sminus, alarms


def count_alarm_events(alarms: np.ndarray) -> int:
    if len(alarms) == 0:
        return 0
    a = alarms.astype(int)
    num = int(np.sum(np.diff(a) == 1))
    if a[0] == 1:
        num += 1
    return num


def _safe_valid(values):
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def _safe_nanmedian(values):
    arr = _safe_valid(values)
    return float(np.median(arr)) if arr.size else np.nan


def _safe_max(values):
    arr = _safe_valid(values)
    return float(np.max(arr)) if arr.size else np.nan


def _safe_last(values):
    arr = np.asarray(values, dtype=float)
    valid_idx = np.where(~np.isnan(arr))[0]
    return float(arr[valid_idx[-1]]) if valid_idx.size else np.nan


def _safe_max_pair(a, b):
    vals = np.asarray([a, b], dtype=float)
    vals = vals[~np.isnan(vals)]
    return float(np.max(vals)) if vals.size else np.nan


def _rr_irregularity_features(rr_ms):
    rr = _safe_valid(rr_ms)
    if rr.size == 0:
        return {"rr_cv": np.nan, "rr_masd_over_mean": np.nan, "segment_rr_mad_ms": np.nan}
    mean_rr = float(np.mean(rr))
    rr_cv = float(np.std(rr)) / mean_rr if mean_rr > 0 else np.nan
    if rr.size >= 2 and mean_rr > 0:
        rr_masd_over_mean = float(np.mean(np.abs(np.diff(rr)))) / mean_rr
    else:
        rr_masd_over_mean = np.nan
    rr_median = float(np.median(rr))
    segment_rr_mad_ms = float(np.median(np.abs(rr - rr_median)))
    return {
        "rr_cv": rr_cv,
        "rr_masd_over_mean": rr_masd_over_mean,
        "segment_rr_mad_ms": segment_rr_mad_ms,
    }


def _cusum_branch_features(Splus, Sminus):
    return {
        "max_cusum_pos": _safe_max(Splus),
        "max_cusum_neg": _safe_max(Sminus),
        "final_cusum_pos": _safe_last(Splus),
        "final_cusum_neg": _safe_last(Sminus),
    }


def process_record_for_features_spc_plus(
    record_name: str,
    pn_dir: Path | str,
    h_threshold: float,
    k_threshold: float,
    *,
    inner_window_sec: float = INNER_WINDOW_SEC,
    baseline_windows: int = BASELINE_WINDOWS,
    sqi_gating: bool = True,
    _debug: bool = False,
) -> dict | None:
    try:
        pn_dir = str(pn_dir)
        _hdr, fs, qrs_samples = load_record_and_qrs(record_name, pn_dir=pn_dir)
        if len(qrs_samples) < 2:
            out = {f: np.nan for f in SPC_FAST_14}
            out["record_id"] = record_name
            return out

        rr_ms = compute_rr_ms(qrs_samples, fs)
        rr_time_s = np.cumsum(rr_ms / 1000.0)
        rmssd_values, _ = window_rmssd(rr_ms, rr_time_s, window_sec=inner_window_sec)
        z_scores = robust_z(rmssd_values, baseline_windows=baseline_windows)

        if sqi_gating:
            gate_ok = window_subwindow_sqi_gate(rr_ms, rr_time_s, window_sec=inner_window_sec).astype(bool)
        else:
            gate_ok = np.ones(len(z_scores), dtype=bool)

        z_for_cusum = np.asarray(z_scores, dtype=float)
        rmssd_for_summary = np.asarray(rmssd_values, dtype=float)
        common_len = min(len(z_for_cusum), len(gate_ok), len(rmssd_for_summary)) if len(gate_ok) else 0
        if common_len > 0:
            z_for_cusum = z_for_cusum[:common_len].copy()
            rmssd_for_summary = rmssd_for_summary[:common_len].copy()
            gate_ok = gate_ok[:common_len]
            z_for_cusum[~gate_ok] = np.nan
            rmssd_for_summary[~gate_ok] = np.nan
        else:
            z_for_cusum = np.asarray([], dtype=float)
            rmssd_for_summary = np.asarray([], dtype=float)
            gate_ok = np.asarray([], dtype=bool)

        Splus, Sminus, alarms_bool = cusum_bilateral(z_for_cusum, k=k_threshold, h=h_threshold)
        branch_features = _cusum_branch_features(Splus, Sminus)
        rr_features = _rr_irregularity_features(rr_ms)

        alarm_events = count_alarm_events(alarms_bool)
        pct_alarm_sustained = float(np.mean(alarms_bool)) if len(alarms_bool) > 0 else np.nan
        max_cusum = _safe_max_pair(branch_features["max_cusum_pos"], branch_features["max_cusum_neg"])
        final_cusum = _safe_max_pair(branch_features["final_cusum_pos"], branch_features["final_cusum_neg"])
        median_rmssd = _safe_nanmedian(rmssd_for_summary)
        median_z = _safe_nanmedian(z_for_cusum)
        subwindow_sqi_good_frac = float(np.mean(gate_ok)) if len(gate_ok) > 0 else np.nan

        return {
            "record_id": record_name,
            "alarm_events": alarm_events,
            "pct_alarm_sustained": pct_alarm_sustained,
            "max_cusum": max_cusum,
            "final_cusum": final_cusum,
            "median_rmssd": median_rmssd,
            "median_z": median_z,
            "subwindow_sqi_good_frac": subwindow_sqi_good_frac,
            **branch_features,
            **rr_features,
        }
    except Exception as exc:
        if _debug:
            raise
        if os.environ.get("AFPDB_DEBUG_FEATURES", ""):
            print(f"[process_record_for_features_spc_plus] {record_name}: {exc!r}")
        return None


def build_main_task_manifest() -> pd.DataFrame:
    """
    p01,p03,... -> y=0 (far); p02,p04,... -> y=1 (pre-onset).
    pair_id = (n-1)//2 para pn = p01..p50 -> p01,p02 -> 0.
    """
    rows = []
    for n in range(1, 51):
        rid = f"p{n:02d}"
        pair_id = (n - 1) // 2
        y = 0 if n % 2 == 1 else 1
        rows.append({"record_id": rid, "pair_id": pair_id, "y": y, "task": "far_vs_preonset_p30min"})
    return pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)


def audit_main_dataset(Xy: pd.DataFrame) -> pd.DataFrame:
    checks = []
    c_used = Xy["record_id"].astype(str).str.endswith("c").any()
    checks.append(("no_registro_c_en_dataset", not c_used, "Ningún record_id debe terminar en 'c'."))
    n_used = Xy["record_id"].astype(str).str.match(r"^n\d+", case=False).any()
    checks.append(("no_registro_n_en_tarea_principal", not n_used, "Sin registros n* en far vs pre-onset."))
    vc = Xy.groupby("pair_id").size()
    checks.append(("cada_pair_id_tiene_dos_filas", bool((vc == 2).all()), f"Conteos: {vc.to_dict()}"))
    yy = Xy.groupby("pair_id")["y"].agg(["min", "max", "nunique"])
    checks.append(
        (
            "cada_pair_tiene_y0_y_y1",
            bool((yy["min"] == 0).all() and (yy["max"] == 1).all() and (yy["nunique"] == 2).all()),
            yy.head().to_string(),
        )
    )
    return pd.DataFrame(checks, columns=["audit_rule", "passed", "detail"])


def verify_gkf_disjoint(groups_train, groups_val) -> bool:
    return set(np.unique(groups_train)).isdisjoint(set(np.unique(groups_val)))


def export_gkf_fold_audit(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, out_csv: Path
) -> pd.DataFrame:
    """Auditoría: ningún pair_id en train y val del mismo fold."""
    gkf = GroupKFold(n_splits=GKF_SPLITS)
    rows = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        g_tr = set(groups.iloc[tr].unique())
        g_va = set(groups.iloc[va].unique())
        disjoint = g_tr.isdisjoint(g_va)
        rows.append(
            {
                "fold": fold,
                "n_train_rows": len(tr),
                "n_val_rows": len(va),
                "n_train_groups": len(g_tr),
                "n_val_groups": len(g_va),
                "train_pair_ids": ",".join(str(x) for x in sorted(g_tr)),
                "val_pair_ids": ",".join(str(x) for x in sorted(g_va)),
                "train_val_pair_disjoint": disjoint,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def specificity_score(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan


def false_alarm_rate(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (tn + fp)) if (tn + fp) > 0 else np.nan


def classification_metrics_at_threshold(y_true, y_prob, thr: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred),
        "false_alarm_rate": false_alarm_rate(y_true, y_pred),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    return out


def make_logreg():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=42, solver="lbfgs")),
        ]
    )


def make_rf():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_svm():
    # LinearSVC + calibración (predict_proba para Brier; más ligero que RBF+Platt en bucles CV)
    est = LinearSVC(random_state=42, max_iter=8000, dual=False)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(est, cv=min(3, 4), method="sigmoid", n_jobs=1),
            ),
        ]
    )


def _positive_proba(model, X):
    proba = model.predict_proba(X)
    return proba[:, 1].astype(float)


def run_groupkfold_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, feature_cols: list[str], model_name: str, model):
    gkf = GroupKFold(n_splits=GKF_SPLITS)
    rows = []
    fold_leakage_ok = []
    y_all = []
    p_all = []
    g_all = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        g_tr = groups.iloc[tr].to_numpy()
        g_va = groups.iloc[va].to_numpy()
        ok = verify_gkf_disjoint(g_tr, g_va)
        fold_leakage_ok.append(ok)
        assert ok, f"Leakage fold {fold}"
        pipe = clone(model)
        pipe.fit(X.iloc[tr][feature_cols], y.iloc[tr])
        prob = _positive_proba(pipe, X.iloc[va][feature_cols])
        m = classification_metrics_at_threshold(y.iloc[va], prob, DECISION_THRESHOLD)
        m.update({"fold": fold, "model": model_name, "n_val": len(va)})
        rows.append(m)
        y_all.extend(y.iloc[va].tolist())
        p_all.extend(prob.tolist())
        g_all.extend(groups.iloc[va].tolist())
    detail = pd.DataFrame(rows)
    num_cols = [c for c in detail.columns if c not in ("fold", "n_val", "model")]
    summary = detail[num_cols].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    summary["model"] = model_name
    return detail, summary, np.array(y_all), np.array(p_all), np.array(g_all), all(fold_leakage_ok)


def block_bootstrap_metrics(y: np.ndarray, p: np.ndarray, g: np.ndarray, n_boot: int = N_BOOTSTRAP) -> pd.DataFrame:
    """Bootstrap por bloques (pair_id); métricas en muestra bootstrap con umbral fijo."""
    unique_g = np.unique(g)
    n_g = len(unique_g)
    rows = []
    for _ in range(n_boot):
        sampled_groups = RNG.choice(unique_g, size=n_g, replace=True)
        idx = np.where(np.isin(g, sampled_groups))[0]
        yb, pb = y[idx], p[idx]
        rows.append(classification_metrics_at_threshold(yb, pb, DECISION_THRESHOLD))
    df = pd.DataFrame(rows)
    summ = []
    for col in df.columns:
        summ.append(
            {
                "metric": col,
                "mean": float(df[col].mean()),
                "ci_low": float(np.percentile(df[col], 2.5)),
                "ci_high": float(np.percentile(df[col], 97.5)),
            }
        )
    return pd.DataFrame(summ)


def interpretability_top15_logreg(X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> pd.DataFrame:
    pipe = make_logreg()
    pipe.fit(X[feature_cols], y)
    coef = pipe.named_steps["clf"].coef_.ravel()
    intercept = float(pipe.named_steps["clf"].intercept_[0])
    tab = pd.DataFrame({"feature": feature_cols, "coefficient_scaled_space": coef, "abs_coef": np.abs(coef)})
    tab = tab.sort_values("abs_coef", ascending=False).head(15).reset_index(drop=True)
    tab["intercept"] = intercept
    return tab


def feature_nan_audit(Xy: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = Xy[c]
        n_miss = int(s.isna().sum())
        rows.append(
            {
                "feature": c,
                "n_missing": n_miss,
                "pct_missing": float(n_miss / max(len(Xy), 1)),
                "in_original_6": c in ORIGINAL_6,
                "in_spc_fast_14": c in SPC_FAST_14,
            }
        )
    return pd.DataFrame(rows)


def savefig_highres(fig, stem: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    tif_path = FIG_DIR / f"{stem}.tif"
    try:
        fig.savefig(
            tif_path,
            bbox_inches="tight",
            dpi=600,
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except Exception:
        try:
            fig.savefig(tif_path, bbox_inches="tight", dpi=600)
        except Exception:
            pass
    plt.close(fig)


def plot_cv_metric_bars(cv_detail: pd.DataFrame, feature_set_label: str):
    metrics = ["roc_auc", "pr_auc", "f1", "recall", "specificity", "false_alarm_rate", "brier"]
    models = cv_detail["model"].unique()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        sub = cv_detail.groupby("model")[metric].agg(["mean", "std"]).reindex(models)
        x = np.arange(len(models))
        ax.bar(x, sub["mean"], yerr=sub["std"], capsize=3, color="#4c72b0", ecolor="#333")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"GroupKFold ({GKF_SPLITS}) — {feature_set_label} — umbral fijo {DECISION_THRESHOLD}")
    plt.tight_layout()
    savefig_highres(fig, f"cv_metrics_{feature_set_label.replace(' ', '_')}")


def build_secondary_pn_manifest() -> pd.DataFrame:
    """50 registros p* y 50 n* (30 min), y=1 PAF / y=0 sin AF documentada. No es pre-onset."""
    rows = []
    for n in range(1, 51):
        rows.append(
            {
                "record_id": f"p{n:02d}",
                "subject_pair_group": (n - 1) // 2,
                "y": 1,
                "task": "PAF_vs_noPAF_subject_discrimination",
            }
        )
    for n in range(1, 51):
        rows.append(
            {
                "record_id": f"n{n:02d}",
                "subject_pair_group": (n - 1) // 2,
                "y": 0,
                "task": "PAF_vs_noPAF_subject_discrimination",
            }
        )
    return pd.DataFrame(rows)


def main():
    # wfdb en Windows puede fallar con rutas absolutas (interpreta mal como URL PhysioNet).
    os.chdir(ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not AFPDB_DIR.is_dir():
        raise FileNotFoundError(f"No existe la carpeta AFPDB: {AFPDB_DIR}")

    # Siempre ruta relativa al cwd (= ROOT) para wfdb.rdrecord / rdann
    pn_dir_local = "afpdb"

    manifest = build_main_task_manifest()
    manifest_path = OUT_DIR / "manifest_far_preonset_p30min.csv"
    manifest.to_csv(manifest_path, index=False)

    Xy_rows = []
    for n_done, (_, row) in enumerate(manifest.iterrows(), start=1):
        feats = process_record_for_features_spc_plus(
            row["record_id"], pn_dir_local, H_CUSUM, K_CUSUM
        )
        if feats is None:
            raise RuntimeError(f"Fallo al procesar {row['record_id']}")
        feats["pair_id"] = int(row["pair_id"])
        feats["y"] = int(row["y"])
        Xy_rows.append(feats)
        if n_done % 10 == 0 or n_done == len(manifest):
            print(f"  Features extraídas: {n_done}/{len(manifest)}", flush=True)

    Xy_df = pd.DataFrame(Xy_rows)
    Xy_df.to_csv(OUT_DIR / "Xy_far_preonset_main.csv", index=False)

    audit_df = audit_main_dataset(Xy_df)
    if not audit_df["passed"].all():
        raise RuntimeError(f"Auditoría principal fallida:\n{audit_df}")

    # Balance
    balance = Xy_df["y"].value_counts().rename_axis("y").reset_index(name="count")
    balance.to_csv(OUT_DIR / "class_balance_main.csv", index=False)

    X_all = Xy_df.drop(columns=["record_id", "pair_id", "y"])
    y = Xy_df["y"].astype(int)
    groups = Xy_df["pair_id"].astype(int)

    gkf_audit = export_gkf_fold_audit(X_all, y, groups, OUT_DIR / "gkf_fold_audit.csv")
    leak_ok = bool(gkf_audit["train_val_pair_disjoint"].all())
    leak_row = pd.DataFrame(
        [
            {
                "audit_rule": "ningun_pair_id_en_train_y_val_mismo_fold",
                "passed": leak_ok,
                "detail": f"GroupKFold({GKF_SPLITS}); ver gkf_fold_audit.csv",
            }
        ]
    )
    if not leak_ok:
        raise RuntimeError("Auditoría GKF: pair_id compartido entre train y val en algún fold.")
    pd.concat([audit_df, leak_row], ignore_index=True).to_csv(
        OUT_DIR / "dataset_audit_main.csv", index=False
    )

    feature_audit = feature_nan_audit(Xy_df, SPC_FAST_14)
    feature_audit.to_csv(OUT_DIR / "feature_audit_table.csv", index=False)

    models = {
        "LogReg": make_logreg(),
        "RandomForest": make_rf(),
        "SVM_LinearCalib": make_svm(),
    }

    cv_rows_orig = []
    cv_rows_spc = []
    bench_summaries = []
    logreg_oof_y_pg: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for mname, mdl in models.items():
        print(f"GroupKFold CV — {mname} …", flush=True)
        d_o, s_o, yo, po, go, _ = run_groupkfold_cv(X_all, y, groups, ORIGINAL_6, mname, mdl)
        d_o["feature_set"] = "Original_6"
        cv_rows_orig.append(d_o)
        d_s, s_s, ys, ps, gs, _ = run_groupkfold_cv(X_all, y, groups, SPC_FAST_14, mname, mdl)
        d_s["feature_set"] = "SPC_Fast_14"
        cv_rows_spc.append(d_s)
        if mname == "LogReg":
            logreg_oof_y_pg = (ys, ps, gs)
        for d, fs in ((d_o, "Original_6"), (d_s, "SPC_Fast_14")):
            sm = d.drop(columns=["fold", "n_val"], errors="ignore").mean(numeric_only=True)
            sm = sm.to_frame().T
            sm.insert(0, "feature_set", fs)
            sm.insert(0, "model", mname)
            bench_summaries.append(sm)

    cv_detail_all = pd.concat(cv_rows_orig + cv_rows_spc, ignore_index=True)
    cv_detail_all["gkf_train_val_disjoint"] = True
    cv_detail_all.to_csv(OUT_DIR / "cv_summary.csv", index=False)

    benchmark_summary = pd.concat(bench_summaries, ignore_index=True)
    benchmark_summary.to_csv(OUT_DIR / "benchmark_summary.csv", index=False)

    # Ablation: LogReg solamente (comparación explícita de conjuntos de features)
    ab_rows = []
    for fs, sub in cv_detail_all[cv_detail_all["model"] == "LogReg"].groupby("feature_set"):
        for _, r in sub.iterrows():
            ab_rows.append(
                {
                    "model": "LogReg",
                    "feature_set": fs,
                    "fold": r["fold"],
                    **{k: r[k] for k in r.index if k not in ("model", "feature_set", "fold", "n_val")},
                }
            )
    ablation_per_fold = pd.DataFrame(ab_rows)
    agg_cols = [c for c in ablation_per_fold.columns if c not in ("model", "feature_set", "fold")]
    ablation_summary = ablation_per_fold.groupby("feature_set")[agg_cols].agg(["mean", "std"])
    ablation_summary.columns = [f"{a}_{b}" for a, b in ablation_summary.columns]
    ablation_summary = ablation_summary.reset_index()
    ablation_per_fold.to_csv(OUT_DIR / "ablation_per_fold_logreg.csv", index=False)
    ablation_summary.to_csv(OUT_DIR / "ablation_summary.csv", index=False)

    assert logreg_oof_y_pg is not None
    y_oof, p_oof, g_oof = logreg_oof_y_pg
    print(f"Bootstrap por grupos (N={N_BOOTSTRAP}) …", flush=True)
    boot_df = block_bootstrap_metrics(y_oof, p_oof, g_oof)
    boot_df.to_csv(OUT_DIR / "bootstrap_group_metrics_LogReg_SPC.csv", index=False)

    interp = interpretability_top15_logreg(X_all, y, SPC_FAST_14)
    interp.to_csv(OUT_DIR / "interpretability_top15.csv", index=False)

    # Figuras resumen CV (SPC_Fast_14)
    plot_cv_metric_bars(pd.concat(cv_rows_spc, ignore_index=True), "SPC_Fast_14")

    # Curvas ROC/PR pooled (OOF, LogReg + SPC_Fast_14, umbral solo anotado — no optimizado)
    fpr, tpr, _ = roc_curve(y_oof, p_oof)
    prec_c, rec_c, _ = precision_recall_curve(y_oof, p_oof)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC (OOF pooled, LogReg SPC_Fast_14)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)
    axes[1].plot(rec_c, prec_c, lw=2, label=f"AP = {average_precision_score(y_oof, p_oof):.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR (OOF pooled)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Umbral clasificación fijo = {DECISION_THRESHOLD} (no ajustado en test)")
    plt.tight_layout()
    savefig_highres(fig, "oof_roc_pr_LogReg_SPC14")

    # Ablation LogReg: ROC-AUC por fold (Original_6 vs SPC_Fast_14)
    ab_auc = ablation_per_fold.pivot(index="fold", columns="feature_set", values="roc_auc")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for col in ab_auc.columns:
        ax2.plot(ab_auc.index, ab_auc[col], marker="o", label=col)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("ROC-AUC (validación)")
    ax2.set_title("Ablation LogReg: Original_6 vs SPC_Fast_14")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    savefig_highres(fig2, "ablation_logreg_roc_auc_by_fold")

    # Secundario p vs n
    sec_man = build_secondary_pn_manifest()
    sec_man.to_csv(OUT_DIR / "manifest_secondary_PAF_vs_noPAF.csv", index=False)
    sec_rows = []
    for n_sec, (_, row) in enumerate(sec_man.iterrows(), start=1):
        f = process_record_for_features_spc_plus(row["record_id"], pn_dir_local, H_CUSUM, K_CUSUM)
        if f is None:
            raise RuntimeError(f"Fallo al procesar registro secundario {row['record_id']}")
        f["y"] = int(row["y"])
        f["subject_pair_group"] = int(row["subject_pair_group"])
        f["task"] = row["task"]
        sec_rows.append(f)
        if n_sec % 10 == 0 or n_sec == len(sec_man):
            print(f"  Secundario p vs n: {n_sec}/{len(sec_man)} ({row['record_id']})", flush=True)
    sec_xy = pd.DataFrame(sec_rows)
    if len(sec_xy) != 100:
        raise RuntimeError(
            f"Dataset secundario incompleto: {len(sec_xy)} filas (se esperan 100: p01–p50 y n01–n50)."
        )
    sec_xy.to_csv(OUT_DIR / "Xy_secondary_PAF_vs_noPAF.csv", index=False)
    Xs = sec_xy.drop(columns=["record_id", "subject_pair_group", "y", "task"], errors="ignore")
    ys = sec_xy["y"].astype(int)
    gs = sec_xy["subject_pair_group"].astype(int)
    sec_bench = []
    for mname, mdl in models.items():
        d_s, _, _, _, _, _ = run_groupkfold_cv(Xs, ys, gs, SPC_FAST_14, mname, mdl)
        d_s["analysis"] = "secondary_PAF_vs_noPAF_SPC14"
        sec_bench.append(d_s)
    pd.concat(sec_bench, ignore_index=True).to_csv(OUT_DIR / "secondary_pn_cv_summary.csv", index=False)

    meta = {
        "decision_threshold": DECISION_THRESHOLD,
        "threshold_note": "Fijo en 0.5; sin optimización sobre test ni sobre validación pooled.",
        "cusum_h": H_CUSUM,
        "cusum_k": K_CUSUM,
        "gkf_splits": GKF_SPLITS,
        "n_bootstrap_groups": N_BOOTSTRAP,
        "afpdb_path": str(AFPDB_DIR.resolve()),
        "wfdb_pn_dir_used": pn_dir_local,
        "n_main_rows": len(Xy_df),
        "main_task": "far_vs_preonset_p30min_only",
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("OK — salidas en:", OUT_DIR)
    print(manifest.head(4).to_string(index=False))
    print(audit_df.to_string(index=False))


if __name__ == "__main__":
    main()
