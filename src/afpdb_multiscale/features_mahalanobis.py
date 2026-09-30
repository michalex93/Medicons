"""
Distancia de Mahalanobis / Hotelling T² para detección de pre-onset PAF.

Dos modos de estimación:
  population_fold  — μ y Σ estimados SOLO en y=0 del train fold (no validation).
  pair_baseline    — μ y Σ estimados SOLO en ventanas far (y=0) del mismo par.
                     REQUIERE segmento basal del mismo sujeto; no es modelo poblacional.

Garantías anti-leakage:
  - fit_ledoit_wolf() recibe ÚNICAMENTE filas del train fold con y=0.
  - alarm_threshold_from_ref() recibe ÚNICAMENTE D_M de esas mismas filas.
  - compute_pair_baseline_dm() usa SOLO el registro y=0 del par para estimar baseline.
  - Los datos de validación no participan en ningún ajuste.

Referencia covarianza robusta:
  Ledoit & Wolf (2004). A well-conditioned estimator for large-dimensional
  covariance matrices. J. Multivar. Anal. 88(2), 365-411.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer

from .config import FULL_RECORD_SEC

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MAHAL_FEATURE_NAMES: list[str] = [
    "mahalanobis_mean", "mahalanobis_median", "mahalanobis_max",
    "mahalanobis_p90", "mahalanobis_last", "mahalanobis_slope",
    "mahalanobis_alarm_count", "mahalanobis_alarm_density",
    "mahalanobis_time_in_alarm", "mahalanobis_first_alarm_time_sec",
]
# lead_time se agrega por separado (solo y=1)
MAHAL_LEAD_TIME_FEATURE = "mahalanobis_lead_time_sec"

MAHAL_FEATURE_SETS: dict[str, list[str]] = {
    "hrv_reduced": ["rmssd", "mad_rr", "sd1", "sd2", "sd1_sd2_ratio"],
    "spc_reduced": [
        "cusum_abs_max", "ewma_slope", "shewhart_out_rate",
        "mr_mad", "longest_run_above_median",
    ],
    "hrv_spc_reduced": [
        "rmssd", "mad_rr", "sd1",
        "cusum_abs_max", "ewma_slope", "mr_mad",
    ],
}

DEFAULT_ALARM_PERCENTILE: float = 95.0
MIN_SAMPLES_FOR_COVARIANCE: int = 5   # mínimo filas y=0 para ajustar covarianza


# ---------------------------------------------------------------------------
# Resultado de ajuste de covarianza
# ---------------------------------------------------------------------------

@dataclass
class CovarianceFit:
    mu: np.ndarray
    sigma_inv: np.ndarray
    method: str          # 'ledoit_wolf' | 'pseudo_inverse'
    n_samples: int
    n_features: int
    covariance_ok: bool  # False si se usó fallback pseudo-inversa


# ---------------------------------------------------------------------------
# Ajuste de covarianza (LedoitWolf + fallback)
# ---------------------------------------------------------------------------

def fit_ledoit_wolf(
    X_train_far: np.ndarray,
    *,
    assume_centered: bool = False,
) -> CovarianceFit:
    """
    Ajusta covarianza LedoitWolf sobre X_train_far (SOLO y=0 del train fold).

    Si LedoitWolf falla, usa pseudo-inversa como fallback y emite UserWarning.
    Imputa NaN con mediana de columna antes de ajustar.

    Parámetro
    ---------
    X_train_far : matriz (n_samples, n_features) con y=0 del train fold.
                  NUNCA debe incluir datos de validation ni de y=1.
    """
    n, p = X_train_far.shape
    if n < MIN_SAMPLES_FOR_COVARIANCE:
        raise ValueError(
            f"Se necesitan al menos {MIN_SAMPLES_FOR_COVARIANCE} muestras y=0 "
            f"para ajustar la covarianza, recibidas: {n}."
        )

    # Imputar NaN con mediana de columna (calculada sobre train_far únicamente)
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_train_far.astype(float))
    mu = np.mean(X, axis=0)

    try:
        lw = LedoitWolf(assume_centered=assume_centered)
        lw.fit(X)
        sigma_inv = lw.precision_       # Σ^{-1} regularizado
        return CovarianceFit(
            mu=lw.location_,
            sigma_inv=sigma_inv,
            method="ledoit_wolf",
            n_samples=n,
            n_features=p,
            covariance_ok=True,
        )
    except Exception as exc:
        warnings.warn(
            f"LedoitWolf falló (n={n}, p={p}): {exc!r}. "
            "Usando pseudo-inversa como fallback. "
            "Resultado menos robusto con p > n.",
            UserWarning,
            stacklevel=2,
        )
        Sigma = np.cov(X, rowvar=False) if n > 1 else np.eye(p)
        sigma_inv = np.linalg.pinv(Sigma)
        return CovarianceFit(
            mu=mu,
            sigma_inv=sigma_inv,
            method="pseudo_inverse",
            n_samples=n,
            n_features=p,
            covariance_ok=False,
        )


# ---------------------------------------------------------------------------
# Cálculo de distancias D_M
# ---------------------------------------------------------------------------

def mahalanobis_distances(
    X: np.ndarray,
    fit: CovarianceFit,
    *,
    imputer_train: SimpleImputer | None = None,
) -> np.ndarray:
    """
    Calcula D_M(x) = sqrt((x-μ)^T Σ^{-1} (x-μ)) para cada fila de X.

    Si X contiene NaN, se imputan con la mediana del train_far (imputer_train).
    Si imputer_train es None, se imputa con la mediana de X (solo para uso interno).
    """
    X_arr = np.asarray(X, dtype=float)
    if imputer_train is not None:
        X_imp = imputer_train.transform(X_arr)
    else:
        # Fallback: mediana de X (no debe usarse en producción con val data)
        col_medians = np.nanmedian(X_arr, axis=0)
        nan_mask = np.isnan(X_arr)
        X_imp = X_arr.copy()
        X_imp[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    delta = X_imp - fit.mu           # (n, p)
    # D²_M = diag( delta @ Σ^{-1} @ delta.T )
    dm_sq = np.einsum("ij,jk,ik->i", delta, fit.sigma_inv, delta)
    dm_sq = np.clip(dm_sq, 0.0, None)   # guard float imprecision
    return np.sqrt(dm_sq)


def alarm_threshold_from_ref(
    dm_train_far: np.ndarray,
    percentile: float = DEFAULT_ALARM_PERCENTILE,
) -> float:
    """
    Calcula umbral de alarma como percentil de D_M en y=0 del train fold.
    SOLO debe llamarse con D_M calculadas sobre y=0 train data.
    """
    valid = dm_train_far[np.isfinite(dm_train_far)]
    if len(valid) == 0:
        return float("inf")
    return float(np.percentile(valid, percentile))


# ---------------------------------------------------------------------------
# Agregación de D_M por registro
# ---------------------------------------------------------------------------

def aggregate_dm_per_record(
    dm_values: np.ndarray,
    start_times: np.ndarray,
    alarm_threshold: float,
    y: int,
) -> dict:
    """
    Transforma la serie temporal de D_M (una por ventana) en features de registro.

    Regla lead_time:
      - Solo calculado para y=1 (pre-onset).
      - Asume PAF onset al final del registro (FULL_RECORD_SEC = 1800 s).
    Para y=0, cualquier alarma = false alarm → se contabiliza en alarm_count/density.
    """
    NAN_DICT = {k: np.nan for k in MAHAL_FEATURE_NAMES + [MAHAL_LEAD_TIME_FEATURE]}

    dm = np.asarray(dm_values, dtype=float)
    n = len(dm)
    if n == 0:
        return NAN_DICT

    finite_mask = np.isfinite(dm)
    dm_finite = dm[finite_mask]
    if len(dm_finite) == 0:
        return NAN_DICT

    # Estadísticas básicas
    dm_mean = float(np.mean(dm_finite))
    dm_median = float(np.median(dm_finite))
    dm_max = float(np.max(dm_finite))
    dm_p90 = float(np.percentile(dm_finite, 90))
    dm_last = float(dm[-1]) if np.isfinite(dm[-1]) else float(dm_finite[-1])

    # Pendiente (tendencia temporal de D_M en la segunda mitad del registro)
    half = max(n // 2, 1)
    tail = dm[max(0, n - half):]
    tail_f = tail[np.isfinite(tail)]
    if len(tail_f) >= 2:
        x_t = np.arange(len(tail_f), dtype=float)
        slope = float(np.polyfit(x_t, tail_f, 1)[0])
    else:
        slope = np.nan

    # Alarmas
    alarms = dm > alarm_threshold
    alarm_count = int(alarms.sum())
    alarm_density = float(alarm_count / n)
    time_in_alarm = float(alarm_count / n) * float(np.nanmax(start_times)) \
        if (n > 0 and np.any(np.isfinite(start_times))) else np.nan

    first_alarm_idx = int(np.argmax(alarms)) if alarm_count > 0 else -1
    first_alarm_time = (
        float(start_times[first_alarm_idx])
        if (first_alarm_idx >= 0 and np.isfinite(start_times[first_alarm_idx]))
        else np.nan
    )

    # lead_time: SOLO y=1
    lead_time = np.nan
    if y == 1 and np.isfinite(first_alarm_time):
        lead_time = max(0.0, float(FULL_RECORD_SEC) - first_alarm_time)

    return {
        "mahalanobis_mean": dm_mean,
        "mahalanobis_median": dm_median,
        "mahalanobis_max": dm_max,
        "mahalanobis_p90": dm_p90,
        "mahalanobis_last": dm_last,
        "mahalanobis_slope": slope,
        "mahalanobis_alarm_count": alarm_count,
        "mahalanobis_alarm_density": alarm_density,
        "mahalanobis_time_in_alarm": time_in_alarm,
        "mahalanobis_first_alarm_time_sec": first_alarm_time,
        MAHAL_LEAD_TIME_FEATURE: lead_time,
    }


# ---------------------------------------------------------------------------
# Modo 1: population_fold  — cálculo dentro de un fold
# ---------------------------------------------------------------------------

def compute_population_fold_dm(
    df_windows: pd.DataFrame,
    feature_cols: list[str],
    train_record_ids: set[str],
    val_record_ids: set[str],
    y_by_record: dict[str, int],
    alarm_percentile: float = DEFAULT_ALARM_PERCENTILE,
) -> tuple[dict[str, dict], CovarianceFit, SimpleImputer, float]:
    """
    Calcula D_M para todos los registros de train+val usando covarianza
    estimada ÚNICAMENTE en y=0 del train fold.

    Retorna
    -------
    dm_records  : {record_id: agregated_dm_dict}
    cov_fit     : resultado del ajuste de covarianza
    imputer     : SimpleImputer ajustado sobre y=0 train (para test anti-leakage)
    alarm_thr   : umbral calculado sobre y=0 train D_M
    """
    avail = [c for c in feature_cols if c in df_windows.columns]
    if not avail:
        raise ValueError(f"Ninguna de las features {feature_cols} está en df_windows.")

    # Ventanas y=0 del train fold ÚNICAMENTE para ajuste de covarianza
    train_far_mask = (
        df_windows["record_id"].isin(train_record_ids)
        & (df_windows["y"] == 0)
    )
    X_train_far_raw = df_windows.loc[train_far_mask, avail].values

    # Imputador ajustado SOLO sobre train_far
    imputer = SimpleImputer(strategy="median")
    X_train_far_imp = imputer.fit_transform(X_train_far_raw.astype(float))

    cov_fit = fit_ledoit_wolf(X_train_far_imp)

    # Threshold sobre D_M de y=0 train
    dm_train_far = mahalanobis_distances(X_train_far_imp, cov_fit)
    alarm_thr = alarm_threshold_from_ref(dm_train_far, alarm_percentile)

    # D_M para TODOS los registros (train + val) usando la covarianza ya ajustada
    all_record_ids = train_record_ids | val_record_ids
    dm_records: dict[str, dict] = {}

    for rid in all_record_ids:
        rec_mask = df_windows["record_id"] == rid
        rec_wins = df_windows[rec_mask].sort_values("window_index")
        if rec_wins.empty:
            continue
        X_rec = rec_wins[avail].values.astype(float)
        dm = mahalanobis_distances(X_rec, cov_fit, imputer_train=imputer)
        starts = rec_wins["start_time_sec"].values
        y_rec = y_by_record.get(rid, int(rec_wins["y"].iloc[0]))
        dm_records[rid] = aggregate_dm_per_record(dm, starts, alarm_thr, y_rec)

    return dm_records, cov_fit, imputer, alarm_thr


# ---------------------------------------------------------------------------
# Modo 2: pair_baseline  — intra-sujeto
# ---------------------------------------------------------------------------

def compute_pair_baseline_dm(
    df_windows: pd.DataFrame,
    feature_cols: list[str],
    pair_id: int,
    alarm_percentile: float = DEFAULT_ALARM_PERCENTILE,
) -> dict[str, dict]:
    """
    Estima μ y Σ usando SOLO ventanas far (y=0) del par dado.
    Calcula D_M tanto para y=0 (auto-distancia) como para y=1 (distancia al basal far).

    ADVERTENCIA: Este modo asume que un segmento basal far del mismo sujeto está
    disponible para cada registro pre-onset. No es un modelo poblacional.
    NO usar datos del registro y=1 para estimar μ ni Σ.

    El y=0 del par sirve como "estado normal" del sujeto;
    el y=1 es evaluado respecto a ese estado.
    """
    avail = [c for c in feature_cols if c in df_windows.columns]
    pair_mask = df_windows["pair_id"] == pair_id
    df_pair = df_windows[pair_mask].copy()

    far_wins = df_pair[df_pair["y"] == 0].sort_values("window_index")
    pre_wins = df_pair[df_pair["y"] == 1].sort_values("window_index")

    if far_wins.empty:
        raise ValueError(
            f"pair_id={pair_id}: no hay ventanas y=0 para estimar baseline."
        )

    X_far_raw = far_wins[avail].values.astype(float)
    imputer = SimpleImputer(strategy="median")
    X_far_imp = imputer.fit_transform(X_far_raw)
    cov_fit = fit_ledoit_wolf(X_far_imp)

    # Threshold desde D_M del propio registro far
    dm_far_self = mahalanobis_distances(X_far_imp, cov_fit)
    alarm_thr = alarm_threshold_from_ref(dm_far_self, alarm_percentile)

    dm_results: dict[str, dict] = {}

    # y=0: auto-distancia (comparación interna del registro far)
    far_rid = far_wins["record_id"].iloc[0]
    dm_results[far_rid] = aggregate_dm_per_record(
        dm_far_self, far_wins["start_time_sec"].values, alarm_thr, 0
    )

    # y=1: distancia respecto al basal far del mismo par
    if not pre_wins.empty:
        pre_rid = pre_wins["record_id"].iloc[0]
        X_pre = pre_wins[avail].values.astype(float)
        # El imputer usa estadísticas del far (solo y=0), no del pre-onset
        dm_pre = mahalanobis_distances(X_pre, cov_fit, imputer_train=imputer)
        dm_results[pre_rid] = aggregate_dm_per_record(
            dm_pre, pre_wins["start_time_sec"].values, alarm_thr, 1
        )

    return dm_results
