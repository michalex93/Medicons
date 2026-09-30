"""
Agregaciones temporales de features por registro.

Principio crítico:
  Early/late se define SOLO por posición temporal (start_time_sec o window_index).
  Las etiquetas y=0 (far_baseline) y y=1 (pre_onset) NO se usan para definir
  qué ventanas son tempranas o tardías.

Base: para cada registro de 30 min, las ventanas se ordenan por start_time_sec.
  - "early" = primer tercio de ventanas
  - "late"  = último tercio de ventanas
  - "last_5" = últimas 5 ventanas por start_time_sec

Features base que se agregan temporalmente:
  - rmssd, cv_rr, sd2
  - shewhart_out_rate
  - cusum_abs_max
  - ewma_slope
  - mr_mad

Agregaciones producidas (sufijos):
  - _global_median
  - _global_p90
  - _global_max
  - _late_third_median
  - _late_third_p90
  - _late_third_max
  - _last5_mean
  - _last5_max
  - _slope_over_time   (regresión lineal de la feature vs time index)
  - _late_minus_early  (mean_late - mean_early, sin usar labels)

Nota: para registros con pocas ventanas (< 6), algunos valores pueden ser NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES: list[str] = [
    "rmssd", "cv_rr", "sd2",
    "shewhart_out_rate", "cusum_abs_max", "ewma_slope", "mr_mad",
]

AGG_SUFFIXES: list[str] = [
    "_global_median", "_global_p90", "_global_max",
    "_late_third_median", "_late_third_p90", "_late_third_max",
    "_last5_mean", "_last5_max",
    "_slope_over_time",
    "_late_minus_early",
]


def _compute_temporal_aggs_for_record(
    windows_df: pd.DataFrame,
    feature: str,
) -> dict[str, float]:
    """
    Calcula todas las agregaciones temporales para una feature de un registro.
    windows_df debe estar ordenado por start_time_sec o window_index.
    """
    NAN = {f"{feature}{s}": np.nan for s in AGG_SUFFIXES}

    vals = windows_df[feature].values.astype(float)
    n = len(vals)
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return NAN

    # Global
    g_med = float(np.median(finite))
    g_p90 = float(np.percentile(finite, 90)) if len(finite) >= 2 else g_med
    g_max = float(np.max(finite))

    # Early / late partitions (puramente temporales, sin usar labels)
    third = max(1, n // 3)
    early_vals = vals[:third][np.isfinite(vals[:third])]
    late_vals = vals[n - third:][np.isfinite(vals[n - third:])]

    lt_med = float(np.median(late_vals)) if len(late_vals) > 0 else np.nan
    lt_p90 = float(np.percentile(late_vals, 90)) if len(late_vals) >= 2 else (
        lt_med if not np.isnan(lt_med) else np.nan
    )
    lt_max = float(np.max(late_vals)) if len(late_vals) > 0 else np.nan
    lt_minus_early = (
        float(np.mean(late_vals) - np.mean(early_vals))
        if (len(late_vals) > 0 and len(early_vals) > 0)
        else np.nan
    )

    # Last 5 windows
    last5_vals = vals[max(0, n - 5):][np.isfinite(vals[max(0, n - 5):])]
    l5_mean = float(np.mean(last5_vals)) if len(last5_vals) > 0 else np.nan
    l5_max = float(np.max(last5_vals)) if len(last5_vals) > 0 else np.nan

    # Slope (lineal, usando índice de ventana como x)
    x_idx = np.where(np.isfinite(vals))[0].astype(float)
    y_fin = vals[np.isfinite(vals)]
    if len(x_idx) >= 2:
        slope = float(np.polyfit(x_idx, y_fin, 1)[0])
    else:
        slope = np.nan

    return {
        f"{feature}_global_median": g_med,
        f"{feature}_global_p90": g_p90,
        f"{feature}_global_max": g_max,
        f"{feature}_late_third_median": lt_med,
        f"{feature}_late_third_p90": lt_p90,
        f"{feature}_late_third_max": lt_max,
        f"{feature}_last5_mean": l5_mean,
        f"{feature}_last5_max": l5_max,
        f"{feature}_slope_over_time": slope,
        f"{feature}_late_minus_early": lt_minus_early,
    }


def compute_temporal_features_per_record(
    df_windows: pd.DataFrame,
    window_type: str,
    window_size: float,
    base_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Para cada registro, calcula todas las agregaciones temporales.

    GARANTÍA: las particiones early/late se basan únicamente en
    start_time_sec / window_index. La columna 'y' NO se usa para definirlas.

    Devuelve un DataFrame con una fila por registro y columnas:
      record_id, pair_id, y, analysis_task, window_type, window_size,
      <feature>_<agg_suffix>, ...
    """
    if base_features is None:
        base_features = BASE_FEATURES

    sub = df_windows[
        (df_windows["window_type"] == window_type)
        & (df_windows["window_size"] == window_size)
    ].copy()

    avail_feats = [f for f in base_features if f in sub.columns]
    rows = []

    for rid, rec_df in sub.groupby("record_id"):
        # Ordenar por posición temporal — sin usar y
        rec_df = rec_df.sort_values("start_time_sec").reset_index(drop=True)

        row = {
            "record_id": rid,
            "pair_id": int(rec_df["pair_id"].iloc[0]),
            "y": int(rec_df["y"].iloc[0]),
            "analysis_task": str(rec_df["analysis_task"].iloc[0])
                             if "analysis_task" in rec_df.columns else "unknown",
            "window_type": window_type,
            "window_size": window_size,
            "n_windows": len(rec_df),
        }

        for feat in avail_feats:
            aggs = _compute_temporal_aggs_for_record(rec_df, feat)
            row.update(aggs)

        rows.append(row)

    return pd.DataFrame(rows)


def temporal_feature_names(base_features: list[str] | None = None) -> list[str]:
    """Lista de todas las features temporales generadas."""
    if base_features is None:
        base_features = BASE_FEATURES
    return [f"{feat}{suf}" for feat in base_features for suf in AGG_SUFFIXES]


def verify_temporal_agg_does_not_use_labels(
    df_windows: pd.DataFrame,
    window_type: str,
    window_size: float,
) -> bool:
    """
    Test de integridad: verifica que early/late no dependen de y.
    Computa temporal features con y shuffle. Si el resultado cambia → hay leakage.
    Devuelve True si no hay leakage (el resultado es idéntico con y original y y shuffled).
    """
    df_orig = compute_temporal_features_per_record(df_windows, window_type, window_size)

    # Shuffle y sin tocar nada más
    df_shuffled = df_windows.copy()
    df_shuffled["y"] = np.random.RandomState(0).permutation(df_shuffled["y"].values)
    df_shuffle_res = compute_temporal_features_per_record(
        df_shuffled, window_type, window_size
    )

    # Las features temporales deben ser idénticas (no dependen de y)
    feat_cols = [c for c in df_orig.columns if any(s in c for s in AGG_SUFFIXES)]
    if not feat_cols:
        return True

    orig_vals = df_orig[feat_cols].values
    shuf_vals = df_shuffle_res[feat_cols].values
    return bool(np.allclose(orig_vals, shuf_vals, equal_nan=True))
