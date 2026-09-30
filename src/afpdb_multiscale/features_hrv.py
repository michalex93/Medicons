"""
Features HRV básicas + Poincaré calculadas sobre un array de intervalos RR (ms).
Todas las funciones son puras: no acceden a disco ni a estado global.
"""
from __future__ import annotations

import numpy as np


MIN_RR_FOR_HRV = 4   # mínimo de intervalos RR para calcular métricas


def _nanval(v):
    return float(v) if np.isfinite(v) else np.nan


# ---------------------------------------------------------------------------
# HRV básico (dominio temporal)
# ---------------------------------------------------------------------------

def compute_hrv_basic(rr_ms: np.ndarray) -> dict:
    """
    mean_rr, median_rr, std_rr, mad_rr, rmssd, sdsd, cv_rr,
    rr_diff_mean, rr_diff_std.
    Devuelve NaN para todo si n_rr < MIN_RR_FOR_HRV.
    """
    NAN = {
        "mean_rr": np.nan, "median_rr": np.nan, "std_rr": np.nan,
        "mad_rr": np.nan, "rmssd": np.nan, "sdsd": np.nan,
        "cv_rr": np.nan, "rr_diff_mean": np.nan, "rr_diff_std": np.nan,
    }
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    if len(rr) < MIN_RR_FOR_HRV:
        return NAN

    drr = np.diff(rr)
    mean_rr = float(np.mean(rr))
    med_rr = float(np.median(rr))
    std_rr = float(np.std(rr, ddof=1)) if len(rr) > 1 else np.nan
    mad_rr = float(np.median(np.abs(rr - med_rr)))

    rmssd = _nanval(np.sqrt(np.mean(drr**2))) if len(drr) > 0 else np.nan
    sdsd = _nanval(np.std(drr, ddof=1)) if len(drr) > 1 else np.nan
    cv_rr = _nanval(std_rr / mean_rr) if mean_rr > 0 else np.nan
    rr_diff_mean = _nanval(float(np.mean(np.abs(drr)))) if len(drr) > 0 else np.nan
    rr_diff_std = _nanval(float(np.std(drr, ddof=1))) if len(drr) > 1 else np.nan

    return {
        "mean_rr": mean_rr,
        "median_rr": med_rr,
        "std_rr": std_rr,
        "mad_rr": mad_rr,
        "rmssd": rmssd,
        "sdsd": sdsd,
        "cv_rr": cv_rr,
        "rr_diff_mean": rr_diff_mean,
        "rr_diff_std": rr_diff_std,
    }


# ---------------------------------------------------------------------------
# Poincaré (SD1, SD2, ratio)
# ---------------------------------------------------------------------------

def compute_poincare(rr_ms: np.ndarray) -> dict:
    """
    SD1 = RMSSD / sqrt(2)  → variabilidad rápida latido a latido
    SD2 = sqrt(2*SDNN² - SD1²)  → variabilidad a largo plazo
    sd1_sd2_ratio = SD1 / SD2

    Referencia: Brennan et al. (2001), IEEE Trans. Biomed. Eng.
    """
    NAN = {"sd1": np.nan, "sd2": np.nan, "sd1_sd2_ratio": np.nan}
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    if len(rr) < 3:
        return NAN

    drr = np.diff(rr)
    sdnn = float(np.std(rr, ddof=1))
    rmssd_val = float(np.sqrt(np.mean(drr**2)))

    sd1 = rmssd_val / np.sqrt(2.0)
    # SD2² = 2*SDNN² - SD1²; guard for negative due to float imprecision
    sd2_sq = max(0.0, 2.0 * sdnn**2 - sd1**2)
    sd2 = float(np.sqrt(sd2_sq))

    ratio = _nanval(sd1 / sd2) if sd2 > 0 else np.nan

    return {"sd1": _nanval(sd1), "sd2": _nanval(sd2), "sd1_sd2_ratio": ratio}
