"""
Cartas de control SPC aplicadas a la serie z-scored de intervalos RR de una ventana.

Cartas implementadas:
  1. Shewhart (individuos, límites ±L·σ)
  2. CUSUM bilateral
  3. EWMA
  4. Moving Range (MR)
  5. Runs rules (Western Electric simplificadas)

El z-scoring se hace fuera de este módulo (windows.zscore_with_record_baseline)
para garantizar que μ/σ provienen del baseline del registro, no de la ventana misma.

NOTA sobre lead_time:
  Para registros y=1 (pre-onset), el tiempo de primera alarma es relevante.
  cusum_first_alarm_time_sec es RELATIVO al inicio de la ventana.
  El pipeline de evaluación calcula la alarma ABSOLUTA como:
      start_time_sec + cusum_first_alarm_time_sec
  y después lead_time = FULL_RECORD_SEC - alarm_time_absolute (solo y=1).
  Para y=0, cualquier alarma es falsa alarma (alarm_density, cusum_time_in_alarm).
"""
from __future__ import annotations

import numpy as np

from .config import (
    CUSUM_H, CUSUM_K, CUSUM_SUSTAINED,
    EWMA_L, EWMA_LAMBDA,
    MR_D4,
    SHEWHART_L,
)

MIN_N_FOR_SPC = 6  # mínimo de puntos para cartas SPC


def _nan_spc_shewhart() -> dict:
    return {k: np.nan for k in [
        "shewhart_out_count", "shewhart_out_rate",
        "shewhart_max_abs_z", "shewhart_first_alarm_pos",
        "shewhart_first_alarm_time_sec",
    ]}


def _nan_spc_cusum() -> dict:
    return {k: np.nan for k in [
        "cusum_pos_max", "cusum_neg_max", "cusum_abs_max",
        "cusum_last_pos", "cusum_last_neg",
        "cusum_alarm_count", "cusum_first_alarm_pos",
        "cusum_first_alarm_time_sec", "cusum_time_in_alarm",
        "cusum_alarm_density",
    ]}


def _nan_spc_ewma() -> dict:
    return {k: np.nan for k in [
        "ewma_last", "ewma_max_abs", "ewma_slope",
        "ewma_alarm_count", "ewma_first_alarm_time_sec",
    ]}


def _nan_spc_mr() -> dict:
    return {k: np.nan for k in [
        "mr_mean", "mr_median", "mr_max", "mr_mad", "mr_out_count",
    ]}


def _nan_spc_runs() -> dict:
    return {k: np.nan for k in [
        "longest_run_above_median", "longest_run_below_median",
        "trend_run_up_max", "trend_run_down_max",
        "n_runs_rule_violations",
    ]}


# ---------------------------------------------------------------------------
# 1. Shewhart
# ---------------------------------------------------------------------------

def _shewhart(z: np.ndarray, rr_time_s: np.ndarray, L: float) -> dict:
    alarm = np.abs(z) > L
    out_count = int(alarm.sum())
    out_rate = float(out_count / len(z))
    max_abs_z = float(np.max(np.abs(z)))

    first_pos = int(np.argmax(alarm)) if out_count > 0 else -1
    first_time = (
        float(rr_time_s[first_pos]) if (out_count > 0 and first_pos >= 0) else np.nan
    )

    return {
        "shewhart_out_count": out_count,
        "shewhart_out_rate": out_rate,
        "shewhart_max_abs_z": max_abs_z,
        "shewhart_first_alarm_pos": float(first_pos),
        "shewhart_first_alarm_time_sec": first_time,
    }


# ---------------------------------------------------------------------------
# 2. CUSUM bilateral
# ---------------------------------------------------------------------------

def _cusum_bilateral(
    z: np.ndarray,
    rr_time_s: np.ndarray,
    h: float,
    k: float,
    sustained: int,
) -> dict:
    n = len(z)
    Sp = np.zeros(n)
    Sm = np.zeros(n)
    alarms = np.zeros(n, dtype=bool)

    cur_p = 0.0
    cur_m = 0.0
    consec_p = 0
    consec_m = 0

    for i in range(n):
        cur_p = max(0.0, cur_p + z[i] - k)
        cur_m = max(0.0, cur_m - z[i] - k)
        Sp[i] = cur_p
        Sm[i] = cur_m

        consec_p = consec_p + 1 if cur_p > h else 0
        consec_m = consec_m + 1 if cur_m > h else 0
        if consec_p >= sustained:
            for j in range(sustained):
                if i - j >= 0:
                    alarms[i - j] = True
        if consec_m >= sustained:
            for j in range(sustained):
                if i - j >= 0:
                    alarms[i - j] = True

    alarm_count = int(alarms.sum())
    first_alarm_idx = int(np.argmax(alarms)) if alarm_count > 0 else -1
    first_alarm_time = (
        float(rr_time_s[first_alarm_idx]) if (alarm_count > 0 and first_alarm_idx >= 0)
        else np.nan
    )
    time_in_alarm = (
        float(np.sum(alarms) / n) * float(rr_time_s[-1]) if n > 0 else np.nan
    )
    alarm_density = float(alarm_count / n)

    return {
        "cusum_pos_max": float(np.max(Sp)),
        "cusum_neg_max": float(np.max(Sm)),
        "cusum_abs_max": float(max(np.max(Sp), np.max(Sm))),
        "cusum_last_pos": float(Sp[-1]),
        "cusum_last_neg": float(Sm[-1]),
        "cusum_alarm_count": alarm_count,
        "cusum_first_alarm_pos": float(first_alarm_idx),
        "cusum_first_alarm_time_sec": first_alarm_time,
        "cusum_time_in_alarm": time_in_alarm,
        "cusum_alarm_density": alarm_density,
    }


# ---------------------------------------------------------------------------
# 3. EWMA
# ---------------------------------------------------------------------------

def _ewma_chart(
    z: np.ndarray,
    rr_time_s: np.ndarray,
    lam: float,
    L: float,
) -> dict:
    """
    E_t = λ·z_t + (1-λ)·E_{t-1}
    UCL = L · σ · sqrt(λ/(2-λ))   [límite estacionario]
    """
    n = len(z)
    E = np.zeros(n)
    E[0] = lam * z[0]
    for i in range(1, n):
        E[i] = lam * z[i] + (1 - lam) * E[i - 1]

    # Límite estacionario (asintótico)
    ucl = L * np.sqrt(lam / (2.0 - lam))
    alarms = np.abs(E) > ucl

    alarm_count = int(alarms.sum())
    first_alarm_idx = int(np.argmax(alarms)) if alarm_count > 0 else -1
    first_alarm_time = (
        float(rr_time_s[first_alarm_idx]) if (alarm_count > 0 and first_alarm_idx >= 0)
        else np.nan
    )

    # Pendiente de EWMA (regresión lineal simple sobre últimos n//2 puntos)
    tail = E[n // 2:] if n >= 4 else E
    x = np.arange(len(tail), dtype=float)
    if len(tail) >= 2:
        slope = float(np.polyfit(x, tail, 1)[0])
    else:
        slope = np.nan

    return {
        "ewma_last": float(E[-1]),
        "ewma_max_abs": float(np.max(np.abs(E))),
        "ewma_slope": slope,
        "ewma_alarm_count": alarm_count,
        "ewma_first_alarm_time_sec": first_alarm_time,
    }


# ---------------------------------------------------------------------------
# 4. Moving Range
# ---------------------------------------------------------------------------

def _moving_range(rr_ms: np.ndarray) -> dict:
    """
    MR_t = |RR_t - RR_{t-1}|
    UCL_MR = D4 · mean(MR) donde D4 ≈ 3.267 para n=2.
    """
    mr = np.abs(np.diff(rr_ms))
    if len(mr) == 0:
        return _nan_spc_mr()

    mr_mean = float(np.mean(mr))
    mr_median = float(np.median(mr))
    mr_max = float(np.max(mr))
    mr_mad = float(np.median(np.abs(mr - mr_median)))
    ucl = MR_D4 * mr_mean
    mr_out = int(np.sum(mr > ucl))

    return {
        "mr_mean": mr_mean,
        "mr_median": mr_median,
        "mr_max": mr_max,
        "mr_mad": mr_mad,
        "mr_out_count": mr_out,
    }


# ---------------------------------------------------------------------------
# 5. Runs rules (Western Electric simplificadas)
# ---------------------------------------------------------------------------

def _longest_run(arr: np.ndarray, condition: np.ndarray) -> int:
    max_run = 0
    cur = 0
    for v in condition:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def _longest_trend(arr: np.ndarray, increasing: bool) -> int:
    max_run = 0
    cur = 1
    for i in range(1, len(arr)):
        diff = arr[i] - arr[i - 1]
        if (increasing and diff > 0) or (not increasing and diff < 0):
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return max_run


def _runs_rules(z: np.ndarray) -> dict:
    """
    Versión simplificada de las reglas Western Electric:
      WE1: 1 punto > 3σ  (capturada por Shewhart, no repetida)
      WE2: 8+ consecutivos del mismo lado de la línea central
      WE3: 6+ consecutivos aumentando o disminuyendo
      WE4: 2 de 3 consecutivos > 2σ mismo lado
      WE5: 4 de 5 consecutivos > 1σ mismo lado
    """
    if len(z) < 2:
        return _nan_spc_runs()

    med = float(np.median(z))
    above = z > med
    below = z < med

    run_above = _longest_run(z, above)
    run_below = _longest_run(z, below)
    trend_up = _longest_trend(z, increasing=True)
    trend_down = _longest_trend(z, increasing=False)

    violations = 0
    n = len(z)

    # WE2: 8+ consecutivos mismo lado
    if run_above >= 8 or run_below >= 8:
        violations += 1

    # WE3: 6+ tendencia monótona
    if trend_up >= 6 or trend_down >= 6:
        violations += 1

    # WE4: 2 de 3 consecutivos > 2σ mismo lado
    for i in range(2, n):
        window3 = z[i - 2:i + 1]
        if np.sum(window3 > 2.0) >= 2 or np.sum(window3 < -2.0) >= 2:
            violations += 1
            break  # contar una vez

    # WE5: 4 de 5 consecutivos > 1σ mismo lado
    for i in range(4, n):
        window5 = z[i - 4:i + 1]
        if np.sum(window5 > 1.0) >= 4 or np.sum(window5 < -1.0) >= 4:
            violations += 1
            break

    return {
        "longest_run_above_median": run_above,
        "longest_run_below_median": run_below,
        "trend_run_up_max": trend_up,
        "trend_run_down_max": trend_down,
        "n_runs_rule_violations": violations,
    }


# ---------------------------------------------------------------------------
# Función de entrada principal
# ---------------------------------------------------------------------------

def compute_spc_features(
    rr_ms: np.ndarray,
    rr_time_s: np.ndarray,
    mu_baseline: float,
    sigma_baseline: float,
    *,
    cusum_h: float = CUSUM_H,
    cusum_k: float = CUSUM_K,
    cusum_sustained: int = CUSUM_SUSTAINED,
    ewma_lambda: float = EWMA_LAMBDA,
    ewma_l: float = EWMA_L,
    shewhart_l: float = SHEWHART_L,
) -> dict:
    """
    Computa todas las cartas SPC sobre los intervalos RR de una ventana.

    Parámetros
    ----------
    rr_ms        : intervalos RR en ms (dentro de la ventana)
    rr_time_s    : tiempo acumulado relativo al inicio de la ventana (s)
    mu_baseline  : media del baseline del registro (primeros BASELINE_N_RR RR)
    sigma_baseline: std  del baseline del registro
    Los hiperparámetros cusum_h, cusum_k, ewma_lambda, etc. se pasan desde
    fuera para permitir tuning solo dentro del train fold.
    """
    rr = np.asarray(rr_ms, dtype=float)
    t = np.asarray(rr_time_s, dtype=float)
    n = len(rr)

    if n < MIN_N_FOR_SPC:
        return {
            **_nan_spc_shewhart(),
            **_nan_spc_cusum(),
            **_nan_spc_ewma(),
            **_nan_spc_mr(),
            **_nan_spc_runs(),
        }

    # z-score con baseline del registro (evita fugas de información)
    z = (rr - mu_baseline) / sigma_baseline

    return {
        **_shewhart(z, t, shewhart_l),
        **_cusum_bilateral(z, t, cusum_h, cusum_k, cusum_sustained),
        **_ewma_chart(z, t, ewma_lambda, ewma_l),
        **_moving_range(rr),
        **_runs_rules(z),
    }
