"""
Carga de registros AFPDB (solo cabecera + anotaciones QRS).
Calcula el baseline por registro para las cartas SPC.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from .config import BASELINE_N_RR, ROOT


@dataclass(frozen=True)
class RecordData:
    record_id: str
    fs: int
    qrs_samples: np.ndarray      # índices QRS en muestras
    rr_ms: np.ndarray            # intervalos RR en ms
    rr_time_s: np.ndarray        # tiempo acumulado al final de cada RR (s)
    mu_baseline: float           # media baseline (primeros BASELINE_N_RR RR)
    sigma_baseline: float        # std  baseline
    n_rr_total: int


def load_record(record_id: str, pn_dir: str = "afpdb") -> RecordData:
    """
    Carga cabecera + anotaciones QRS; calcula RR y baseline para SPC.
    Debe llamarse con os.getcwd() == ROOT para que wfdb no confunda la ruta con URL.
    """
    header = wfdb.rdheader(record_id, pn_dir=pn_dir)
    fs = int(header.fs)
    ann = wfdb.rdann(record_id, extension="qrs", pn_dir=pn_dir)
    qrs = ann.sample

    if len(qrs) < 2:
        raise ValueError(f"Registro '{record_id}' tiene < 2 muestras QRS.")

    rr_samples = np.diff(qrs)
    rr_ms = (rr_samples / fs) * 1000.0
    rr_time_s = np.cumsum(rr_ms / 1000.0)

    # Baseline: primeros min(BASELINE_N_RR, len) intervalos RR
    n_base = min(BASELINE_N_RR, len(rr_ms))
    baseline = rr_ms[:n_base]
    mu = float(np.mean(baseline))
    sigma = float(np.std(baseline, ddof=1)) if n_base > 1 else 1.0
    if sigma <= 0:
        sigma = 1.0  # fallback para std cero

    return RecordData(
        record_id=record_id,
        fs=fs,
        qrs_samples=qrs,
        rr_ms=rr_ms,
        rr_time_s=rr_time_s,
        mu_baseline=mu,
        sigma_baseline=sigma,
        n_rr_total=len(rr_ms),
    )


def build_main_manifest() -> list[dict]:
    """
    Devuelve lista de dicts con record_id, pair_id, y, analysis_task.
    Tarea principal: p01 (impar)→y=0, p02 (par)→y=1, ..., p50→y=1.
    EXCLUYE completamente registros *c y n*.
    """
    rows = []
    for n in range(1, 51):
        rid = f"p{n:02d}"
        pair_id = (n - 1) // 2
        y = 0 if n % 2 == 1 else 1
        rows.append(
            {
                "record_id": rid,
                "pair_id": pair_id,
                "y": y,
                "analysis_task": "main_p_far_vs_p_pre",
            }
        )
    return rows


def build_secondary_manifest() -> list[dict]:
    """
    Análisis secundario: p* vs n* (discriminación de grupo, NO pre-onset).
    Etiqueta separada: analysis_task='secondary_p_vs_n'.
    """
    rows = []
    for n in range(1, 51):
        rows.append(
            {
                "record_id": f"p{n:02d}",
                "pair_id": (n - 1) // 2,
                "y": 1,
                "analysis_task": "secondary_p_vs_n",
            }
        )
    for n in range(1, 51):
        rows.append(
            {
                "record_id": f"n{n:02d}",
                "pair_id": (n - 1) // 2,
                "y": 0,
                "analysis_task": "secondary_p_vs_n",
            }
        )
    return rows
