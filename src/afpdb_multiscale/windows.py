"""
Extracción de ventanas multi-escala sobre la serie RR de un registro.

Tipos:
  rr_count  — ventanas de N intervalos RR consecutivos (N ∈ {50,100,128,200})
  time      — ventanas temporales (por defecto 5 min = 300 s)
  full_record — registro completo como ventana única

PROHIBIDO: ventanas de 30 segundos en el análisis principal.

Cada ventana devuelve un dict de provenance + los arrays rr_ms y rr_time_s_local
que se pasan a las funciones de extracción de features.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .config import FULL_RECORD_SEC, PROHIBITED_WINDOW_SEC
from .loader import RecordData


class ProhibitedWindowError(ValueError):
    """Lanzada si se intenta crear ventanas de 30 s en el análisis principal."""


# ---------------------------------------------------------------------------
# Provenance dict
# ---------------------------------------------------------------------------

def _provenance(
    record_id: str,
    pair_id: int,
    y: int,
    analysis_task: str,
    window_type: str,
    window_size: float | int,
    window_index: int,
    start_time_sec: float,
    end_time_sec: float,
    n_rr: int,
) -> dict:
    return {
        "record_id": record_id,
        "pair_id": pair_id,
        "y": y,
        "analysis_task": analysis_task,
        "window_type": window_type,
        "window_size": window_size,
        "window_index": window_index,
        "start_time_sec": round(start_time_sec, 4),
        "end_time_sec": round(end_time_sec, 4),
        "n_rr": n_rr,
    }


# ---------------------------------------------------------------------------
# RR-count windows
# ---------------------------------------------------------------------------

def iter_rr_count_windows(
    rec: RecordData,
    meta: dict,
    n_rr_window: int,
) -> Iterator[dict]:
    """
    Ventanas no solapadas de exactamente n_rr_window intervalos RR.
    Devuelve también rr_ms_local y rr_time_s_local (tiempo relativo al inicio de ventana).
    """
    rr = rec.rr_ms
    t = rec.rr_time_s
    n_total = len(rr)

    if n_rr_window > n_total:
        return

    n_windows = n_total // n_rr_window
    for i in range(n_windows):
        start_idx = i * n_rr_window
        end_idx = start_idx + n_rr_window
        rr_local = rr[start_idx:end_idx]
        # Tiempo acumulado local desde 0 dentro de la ventana
        rr_time_local = np.cumsum(rr_local / 1000.0)

        start_t = float(t[start_idx - 1]) if start_idx > 0 else 0.0
        end_t = float(t[end_idx - 1])

        prov = _provenance(
            record_id=meta["record_id"],
            pair_id=meta["pair_id"],
            y=meta["y"],
            analysis_task=meta["analysis_task"],
            window_type="rr_count",
            window_size=n_rr_window,
            window_index=i,
            start_time_sec=start_t,
            end_time_sec=end_t,
            n_rr=len(rr_local),
        )
        prov["_rr_ms"] = rr_local
        prov["_rr_time_s"] = rr_time_local
        yield prov


# ---------------------------------------------------------------------------
# Time-based windows
# ---------------------------------------------------------------------------

def iter_time_windows(
    rec: RecordData,
    meta: dict,
    window_sec: float,
    *,
    analysis_task_label: str | None = None,
) -> Iterator[dict]:
    """
    Ventanas temporales no solapadas de duración window_sec.

    Raises ProhibitedWindowError si window_sec == PROHIBITED_WINDOW_SEC
    y analysis_task indica tarea principal.
    """
    task = analysis_task_label or meta.get("analysis_task", "")
    if window_sec == PROHIBITED_WINDOW_SEC and "main" in task:
        raise ProhibitedWindowError(
            f"Ventana de {PROHIBITED_WINDOW_SEC} s prohibida en el análisis principal."
        )

    rr = rec.rr_ms
    t = rec.rr_time_s  # tiempo acumulado al final de cada intervalo RR
    max_time = float(t[-1]) if len(t) > 0 else 0.0

    win_idx = 0
    current_start = 0.0
    while current_start < max_time:
        current_end = current_start + window_sec
        mask = (t >= current_start) & (t < current_end)
        idx = np.where(mask)[0]
        if len(idx) < 2:
            current_start = current_end
            win_idx += 1
            continue

        rr_local = rr[idx]
        rr_time_local = np.cumsum(rr_local / 1000.0)

        prov = _provenance(
            record_id=meta["record_id"],
            pair_id=meta["pair_id"],
            y=meta["y"],
            analysis_task=meta["analysis_task"],
            window_type="time",
            window_size=window_sec,
            window_index=win_idx,
            start_time_sec=current_start,
            end_time_sec=min(current_end, max_time),
            n_rr=len(rr_local),
        )
        prov["_rr_ms"] = rr_local
        prov["_rr_time_s"] = rr_time_local
        yield prov

        current_start = current_end
        win_idx += 1


# ---------------------------------------------------------------------------
# Full-record window
# ---------------------------------------------------------------------------

def full_record_window(rec: RecordData, meta: dict) -> dict:
    """
    Registro completo como ventana única (~30 min).
    """
    rr = rec.rr_ms
    rr_time_local = np.cumsum(rr / 1000.0)
    end_t = float(rr_time_local[-1]) if len(rr_time_local) > 0 else 0.0

    prov = _provenance(
        record_id=meta["record_id"],
        pair_id=meta["pair_id"],
        y=meta["y"],
        analysis_task=meta["analysis_task"],
        window_type="full_record",
        window_size=FULL_RECORD_SEC,
        window_index=0,
        start_time_sec=0.0,
        end_time_sec=end_t,
        n_rr=len(rr),
    )
    prov["_rr_ms"] = rr
    prov["_rr_time_s"] = rr_time_local
    return prov


# ---------------------------------------------------------------------------
# Helper: z-score usando baseline del registro
# ---------------------------------------------------------------------------

def zscore_with_record_baseline(
    rr_ms: np.ndarray,
    mu_baseline: float,
    sigma_baseline: float,
) -> np.ndarray:
    """
    z_t = (RR_t - mu_baseline) / sigma_baseline
    Usa el baseline del registro (primeros BASELINE_N_RR RR), no el de la ventana.
    Garantiza que la información futura del mismo registro no defina los límites de control.
    """
    return (rr_ms - mu_baseline) / sigma_baseline
