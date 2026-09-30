"""
Nested SPC parameter tuning para ventanas candidatas (100 RR, 128 RR, 5 min).

Principio anti-leakage:
  - Outer CV: estima rendimiento del pipeline CON selección de params.
  - Inner CV (dentro de train outer): SELECCIONA mejor combinación de params.
  - Los datos de val outer NUNCA participan en inner CV ni en selección de params.

Flujo:
  1. Cargar RR de los 50 registros en memoria (una vez).
  2. Pre-computar features SPC para todas las combinaciones de params.
  3. Outer GroupKFold(5) por pair_id:
     a. Inner GroupKFold(3) sobre train_outer:
        - Para cada param combo: AUC con el modelo base (LogReg).
        - Seleccionar mejor combo por AUC inner CV promedio.
     b. Evaluar val_outer con features del mejor combo.

Rejilla de parámetros:
  h ∈ {3, 5, 7}         (CUSUM threshold)
  k ∈ {0.25, 0.5, 0.75} (CUSUM drift)
  lam ∈ {0.1, 0.2, 0.3} (EWMA lambda)
  sustained ∈ {1, 2, 3}  (CUSUM sustained windows)
  shewhart_L ∈ {2.5, 3.0}

Total: 3 × 3 × 3 × 3 × 2 = 162 combinaciones.

Ventanas evaluadas: 100 RR, 128 RR, 5 min (300 s).
"""
from __future__ import annotations

import itertools
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BASELINE_N_RR, DECISION_THRESHOLD, GKF_N_SPLITS,
    RANDOM_STATE, ROOT,
)
from .feature_blocks import CorrelationPruner, HRV_REDUCED_INTERPRETABLE
from .features_spc import compute_spc_features
from .loader import RecordData, build_main_manifest, load_record
from .windows import iter_rr_count_windows, iter_time_windows, full_record_window

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Rejilla de parámetros SPC
# ---------------------------------------------------------------------------

SPC_PARAM_GRID: dict[str, list] = {
    "h": [3.0, 5.0, 7.0],
    "k": [0.25, 0.5, 0.75],
    "lam": [0.1, 0.2, 0.3],
    "sustained": [1, 2, 3],
    "shewhart_L": [2.5, 3.0],
}

CANDIDATE_WINDOWS: list[tuple[str, float]] = [
    ("rr_count", 100.0),
    ("rr_count", 128.0),
    ("time", 300.0),
]

SPC_FEATURES_FOR_TUNING: list[str] = [
    "cusum_pos_max", "cusum_neg_max", "cusum_abs_max",
    "cusum_last_pos", "cusum_last_neg",
    "cusum_alarm_count", "cusum_alarm_density",
    "cusum_time_in_alarm",
    "shewhart_out_count", "shewhart_out_rate", "shewhart_max_abs_z",
    "ewma_last", "ewma_max_abs", "ewma_slope", "ewma_alarm_count",
    "mr_mean", "mr_median", "mr_max", "mr_mad", "mr_out_count",
    "longest_run_above_median", "longest_run_below_median",
    "trend_run_up_max", "trend_run_down_max", "n_runs_rule_violations",
]

INNER_SPLITS = 3


# ---------------------------------------------------------------------------
# Carga de datos RR en memoria
# ---------------------------------------------------------------------------

def load_rr_cache(pn_dir: str = "afpdb", verbose: bool = True) -> dict[str, RecordData]:
    """
    Carga todos los registros p01–p50 en memoria (una sola vez).
    Devuelve {record_id: RecordData}.
    """
    os.chdir(ROOT)
    manifest = build_main_manifest()
    cache: dict[str, RecordData] = {}
    for meta in manifest:
        rid = meta["record_id"]
        try:
            cache[rid] = load_record(rid, pn_dir=pn_dir)
        except Exception as exc:
            if verbose:
                print(f"  [WARN] {rid}: {exc!r}")
    if verbose:
        print(f"  Cargados {len(cache)} registros en memoria.")
    return cache


# ---------------------------------------------------------------------------
# Extracción de features SPC para una combinación de params sobre todas las ventanas
# ---------------------------------------------------------------------------

def _extract_spc_for_params(
    cache: dict[str, RecordData],
    manifest: list[dict],
    window_type: str,
    window_size: float,
    h: float,
    k: float,
    lam: float,
    sustained: int,
    shewhart_L: float,
) -> pd.DataFrame:
    """
    Para cada registro, extrae ventanas del tipo indicado y calcula SPC features
    con los parámetros dados. Agrega por registro (mediana para la mayoría).
    """
    rows = []
    for meta in manifest:
        rid = meta["record_id"]
        if rid not in cache:
            continue
        rec = cache[rid]

        if window_type == "rr_count":
            windows = list(iter_rr_count_windows(rec, meta, int(window_size)))
        elif window_type == "time":
            windows = list(iter_time_windows(rec, meta, window_size))
        else:
            windows = [full_record_window(rec, meta)]

        if not windows:
            continue

        win_feat_rows = []
        for w in windows:
            rr = w["_rr_ms"]
            t = w["_rr_time_s"]
            f = compute_spc_features(
                rr, t,
                mu_baseline=rec.mu_baseline,
                sigma_baseline=rec.sigma_baseline,
                cusum_h=h, cusum_k=k,
                cusum_sustained=sustained,
                ewma_lambda=lam,
                ewma_l=shewhart_L * 0.9,   # correlate L with shewhart_L
                shewhart_l=shewhart_L,
            )
            win_feat_rows.append(f)

        if not win_feat_rows:
            continue

        wdf = pd.DataFrame(win_feat_rows)
        agg = {}
        for col in SPC_FEATURES_FOR_TUNING:
            if col not in wdf.columns:
                agg[col] = np.nan
                continue
            vals = wdf[col].dropna()
            agg[col] = float(vals.median()) if len(vals) > 0 else np.nan

        row = {
            "record_id": rid,
            "pair_id": meta["pair_id"],
            "y": meta["y"],
        }
        row.update(agg)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pre-computación de toda la rejilla (memoria, rápido tras carga inicial)
# ---------------------------------------------------------------------------

def precompute_param_grid(
    cache: dict[str, RecordData],
    manifest: list[dict],
    window_type: str,
    window_size: float,
    verbose: bool = True,
) -> dict[tuple, pd.DataFrame]:
    """
    Para cada combinación de (h, k, lam, sustained, shewhart_L),
    pre-computa la feature matrix.
    Devuelve {param_tuple: pd.DataFrame}.
    """
    grid = list(itertools.product(
        SPC_PARAM_GRID["h"],
        SPC_PARAM_GRID["k"],
        SPC_PARAM_GRID["lam"],
        SPC_PARAM_GRID["sustained"],
        SPC_PARAM_GRID["shewhart_L"],
    ))
    results: dict[tuple, pd.DataFrame] = {}
    for i, (h, k, lam, sus, sl) in enumerate(grid):
        key = (h, k, lam, sus, sl)
        results[key] = _extract_spc_for_params(
            cache, manifest, window_type, window_size, h, k, lam, sus, sl
        )
        if verbose and (i + 1) % 20 == 0:
            print(f"    Params pre-computed: {i+1}/{len(grid)}", flush=True)
    if verbose:
        print(f"    Total: {len(results)} combinaciones para {window_type} {window_size}")
    return results


# ---------------------------------------------------------------------------
# Modelo simple para inner CV
# ---------------------------------------------------------------------------

def _make_inner_pipe(use_pruning: bool = False) -> Pipeline:
    steps = [
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=1.0)),
    ]
    return Pipeline(steps)


def _inner_cv_auc(
    df: pd.DataFrame,
    feat_cols: list[str],
    n_inner: int = INNER_SPLITS,
) -> float:
    """AUC promedio en inner GroupKFold sobre df (train_outer subset)."""
    avail = [c for c in feat_cols if c in df.columns]
    if len(avail) < 1:
        return 0.0
    X = df[avail].fillna(0)
    y = df["y"].astype(int)
    groups = df["pair_id"].astype(int)
    n_splits = min(n_inner, groups.nunique())
    if n_splits < 2:
        return 0.0
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    pipe = _make_inner_pipe()
    for tr, va in gkf.split(X, y, groups=groups):
        p = clone(pipe)
        try:
            p.fit(X.iloc[tr], y.iloc[tr])
            prob = p.predict_proba(X.iloc[va])[:, 1]
            if len(np.unique(y.iloc[va])) > 1:
                aucs.append(roc_auc_score(y.iloc[va], prob))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


# ---------------------------------------------------------------------------
# Nested CV principal
# ---------------------------------------------------------------------------

def nested_spc_tuning_cv(
    param_grid_cache: dict[tuple, pd.DataFrame],
    manifest: list[dict],
    window_type: str,
    window_size: float,
    block_name: str,
    hrv_df: pd.DataFrame | None = None,
    use_pruning: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Outer GroupKFold(5) por pair_id. En cada outer fold:
      - Inner GroupKFold(3) sobre train_outer → selecciona mejor param combo.
      - Evalúa val_outer con features del mejor combo.

    hrv_df: si se proporciona, las features HRV se concatenan con SPC para block C.
    """
    # Construir DataFrame base con pair_id/y info
    any_df = next(iter(param_grid_cache.values()))
    meta_df = any_df[["record_id", "pair_id", "y"]].copy()
    groups = meta_df["pair_id"].astype(int)
    y_all = meta_df["y"].astype(int)

    n_outer = min(GKF_N_SPLITS, groups.nunique())
    outer_gkf = GroupKFold(n_splits=n_outer)

    fold_rows = []

    for outer_fold, (tr_outer, va_outer) in enumerate(
        outer_gkf.split(meta_df, y_all, groups=groups), start=1
    ):
        g_tr = set(groups.iloc[tr_outer].unique())
        g_va = set(groups.iloc[va_outer].unique())
        assert g_tr.isdisjoint(g_va), f"Leakage pair_id en outer fold {outer_fold}"

        tr_rids = set(meta_df.iloc[tr_outer]["record_id"])
        va_rids = set(meta_df.iloc[va_outer]["record_id"])

        # Inner CV: seleccionar mejor param combo usando SOLO train_outer
        best_params = None
        best_inner_auc = -1.0

        for params, df_params in param_grid_cache.items():
            df_tr = df_params[df_params["record_id"].isin(tr_rids)].copy()
            if hrv_df is not None:
                hrv_tr = hrv_df[hrv_df["record_id"].isin(tr_rids)].copy()
                df_tr = df_tr.merge(hrv_tr.drop(columns=["pair_id","y"], errors="ignore"),
                                    on="record_id", how="left")
            feat_cols = SPC_FEATURES_FOR_TUNING[:]
            if hrv_df is not None:
                hrv_feats = [c for c in HRV_REDUCED_INTERPRETABLE if c in df_tr.columns]
                feat_cols = hrv_feats + feat_cols

            auc_inner = _inner_cv_auc(df_tr, feat_cols)
            if auc_inner > best_inner_auc:
                best_inner_auc = auc_inner
                best_params = params

        if best_params is None:
            continue

        # Eval outer val con el mejor combo
        df_best = param_grid_cache[best_params].copy()
        df_val = df_best[df_best["record_id"].isin(va_rids)].copy()
        df_train = df_best[df_best["record_id"].isin(tr_rids)].copy()

        if hrv_df is not None:
            for dset_name, dset in (("val", df_val), ("train", df_train)):
                hrv_sub = hrv_df[hrv_df["record_id"].isin(
                    va_rids if dset_name == "val" else tr_rids
                )].copy()
                if dset_name == "val":
                    df_val = dset.merge(hrv_sub.drop(columns=["pair_id","y"], errors="ignore"),
                                        on="record_id", how="left")
                else:
                    df_train = dset.merge(hrv_sub.drop(columns=["pair_id","y"], errors="ignore"),
                                          on="record_id", how="left")

        feat_cols = SPC_FEATURES_FOR_TUNING[:]
        if hrv_df is not None:
            hrv_feats = [c for c in HRV_REDUCED_INTERPRETABLE if c in df_train.columns]
            feat_cols = hrv_feats + feat_cols

        avail = [c for c in feat_cols if c in df_train.columns and c in df_val.columns]

        if use_pruning:
            pruner = CorrelationPruner(threshold=0.90)
            X_tr_raw = df_train[avail]
            pruner.fit(X_tr_raw)
            X_tr = pruner.transform(X_tr_raw).values
            X_va = pruner.transform(df_val[avail]).values
            features_used = pruner.selected_features_
        else:
            X_tr = df_train[avail].fillna(0).values
            X_va = df_val[avail].fillna(0).values
            features_used = avail

        y_tr = df_train["y"].astype(int).values
        y_va = df_val["y"].astype(int).values
        g_va_arr = df_val["pair_id"].astype(int).values

        pipe = _make_inner_pipe()
        try:
            pipe.fit(X_tr, y_tr)
            prob = pipe.predict_proba(X_va)[:, 1]
            pred = (prob >= DECISION_THRESHOLD).astype(int)
        except Exception as exc:
            warnings.warn(f"Outer fold {outer_fold}: {exc!r}", UserWarning)
            continue

        auc_val = float(roc_auc_score(y_va, prob)) if len(np.unique(y_va)) > 1 else np.nan

        # Pairwise concordance
        concordances = []
        for pid in np.unique(g_va_arr):
            idx_far = np.where((g_va_arr == pid) & (y_va == 0))[0]
            idx_pre = np.where((g_va_arr == pid) & (y_va == 1))[0]
            if len(idx_far) and len(idx_pre):
                concordances.append(int(np.mean(prob[idx_pre]) > np.mean(prob[idx_far])))
        pc = float(np.mean(concordances)) if concordances else np.nan

        h_, k_, lam_, sus_, sl_ = best_params
        fold_rows.append({
            "outer_fold": outer_fold,
            "window_type": window_type,
            "window_size": window_size,
            "block": block_name,
            "use_pruning": use_pruning,
            "best_h": h_, "best_k": k_, "best_lam": lam_,
            "best_sustained": sus_, "best_shewhart_L": sl_,
            "best_inner_auc": round(best_inner_auc, 4),
            "val_auc": round(auc_val, 4),
            "pairwise_concordance": round(pc, 4),
            "n_val": len(y_va),
            "n_features_used": len(features_used),
        })
        if verbose:
            print(
                f"    fold {outer_fold}: best_params=({h_},{k_},{lam_},{sus_},{sl_}) "
                f"inner_AUC={best_inner_auc:.3f} val_AUC={auc_val:.3f} PC={pc:.3f}",
                flush=True,
            )

    return pd.DataFrame(fold_rows)


# ---------------------------------------------------------------------------
# Función de entrada principal
# ---------------------------------------------------------------------------

def run_nested_spc_tuning(
    cache: dict[str, RecordData] | None = None,
    hrv_agg_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Ejecuta nested tuning para todos los bloques y ventanas candidatas.

    Si cache=None, carga los datos desde AFPDB (requiere os.chdir(ROOT)).
    Si hrv_agg_df se proporciona, incluye bloques hrv_reduced + SPC.
    """
    if cache is None:
        cache = load_rr_cache(verbose=verbose)

    manifest = build_main_manifest()
    all_results = []

    for window_type, window_size in CANDIDATE_WINDOWS:
        if verbose:
            print(f"\n  Ventana {window_type} {window_size}: pre-computando {162} combos…")
        param_cache = precompute_param_grid(cache, manifest, window_type, window_size, verbose)

        configs: list[tuple[str, pd.DataFrame | None, bool]] = [
            ("B_SPC", None, False),
            ("B_SPC_pruned", None, True),
        ]
        if hrv_agg_df is not None:
            configs += [
                ("C_hrv_SPC", hrv_agg_df, False),
                ("C_hrv_SPC_pruned", hrv_agg_df, True),
            ]

        for block_name, hrv_df, use_pruning in configs:
            if verbose:
                print(f"  {block_name}…", flush=True)
            res = nested_spc_tuning_cv(
                param_cache, manifest, window_type, window_size,
                block_name, hrv_df=hrv_df, use_pruning=use_pruning,
                verbose=verbose,
            )
            if not res.empty:
                all_results.append(res)

    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
