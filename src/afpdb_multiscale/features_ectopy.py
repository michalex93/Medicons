"""
Patrones compatibles con ectopia inferidos SOLO desde intervalos RR.

ADVERTENCIA METODOLÓGICA EXPLÍCITA:
  Estas variables son patrones COMPATIBLES con ectopia inferidos desde RR.
  NO son diagnóstico confirmado de PAC/APC ni resultado de análisis morfológico de onda P.
  No usar la palabra "PAC" ni "latido ectópico confirmado" en publicaciones
  sin anotación clínica o detector morfológico explícito.

Definiciones de "corto" y "largo" (ver config.py):
  ECTOPY_SHORT_THRESH = 0.75  → RR < 0.75·mediana  = "corto/prematuro-like"
  ECTOPY_LONG_THRESH  = 1.25  → RR > 1.25·mediana  = "largo/post-pausa-like"
"""
from __future__ import annotations

import numpy as np

from .config import (
    ATRIAL_TACHY_MIN_RUN,
    ECTOPY_LONG_THRESH,
    ECTOPY_SHORT_THRESH,
)

MIN_RR_FOR_ECTOPY = 5


def _nan_ectopy() -> dict:
    return {k: np.nan for k in [
        "ectopy_like_count", "ectopy_like_density",
        "short_long_sequence_count", "compensatory_pause_like_count",
        "bigeminy_like_pattern_count", "trigeminy_like_pattern_count",
        "atrial_tachy_like_run_count",
    ]}


def compute_ectopy_like_features(rr_ms: np.ndarray) -> dict:
    """
    Detecta patrones RR compatibles con ectopia sin acceder a morfología ECG.

    Devuelve:
      ectopy_like_count           : RR fuera del rango [0.75·med, 1.25·med]
      ectopy_like_density         : ectopy_like_count / n
      short_long_sequence_count   : pares (corto, largo) consecutivos
      compensatory_pause_like_count : RR largo que sigue a un RR corto
      bigeminy_like_pattern_count : rachas de ≥4 intervalos alternando corto-normal
      trigeminy_like_pattern_count : rachas de ≥6 con patrón corto-normal-normal
      atrial_tachy_like_run_count  : rachas de ≥3 RR cortos consecutivos (PAT-like)
    """
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    n = len(rr)
    if n < MIN_RR_FOR_ECTOPY:
        return _nan_ectopy()

    med = float(np.median(rr))
    if med <= 0:
        return _nan_ectopy()

    short_thr = ECTOPY_SHORT_THRESH * med   # < umbral → corto
    long_thr = ECTOPY_LONG_THRESH * med     # > umbral → largo

    is_short = rr < short_thr
    is_long = rr > long_thr
    is_normal = ~is_short & ~is_long

    # 1 & 2: conteo y densidad ectopy-like
    ectopy_like = is_short | is_long
    ectopy_count = int(ectopy_like.sum())
    ectopy_density = float(ectopy_count / n)

    # 3: pares (corto, largo) consecutivos
    short_long_seq = 0
    for i in range(n - 1):
        if is_short[i] and is_long[i + 1]:
            short_long_seq += 1

    # 4: pausa compensatoria-like = RR largo después de RR corto (igual que short_long_seq)
    compensatory = short_long_seq  # es el mismo patrón

    # 5: bigeminismo-like: alternancia corto-normal de longitud ≥ 4
    bigeminy_count = 0
    i = 0
    while i < n - 3:
        run_len = 0
        j = i
        while j < n - 1 and is_short[j] and is_normal[j + 1]:
            run_len += 2
            j += 2
        if run_len >= 4:
            bigeminy_count += 1
            i = j
        else:
            i += 1

    # 6: trigeminismo-like: patrón corto-normal-normal de longitud ≥ 6
    trigeminy_count = 0
    i = 0
    while i < n - 5:
        run_len = 0
        j = i
        while j + 2 < n and is_short[j] and is_normal[j + 1] and is_normal[j + 2]:
            run_len += 3
            j += 3
        if run_len >= 6:
            trigeminy_count += 1
            i = j
        else:
            i += 1

    # 7: taquicardia auricular paroxística-like: ≥ ATRIAL_TACHY_MIN_RUN RR cortos consecutivos
    tachy_runs = 0
    i = 0
    while i < n:
        if is_short[i]:
            run_len = 1
            while i + run_len < n and is_short[i + run_len]:
                run_len += 1
            if run_len >= ATRIAL_TACHY_MIN_RUN:
                tachy_runs += 1
            i += run_len
        else:
            i += 1

    return {
        "ectopy_like_count": ectopy_count,
        "ectopy_like_density": ectopy_density,
        "short_long_sequence_count": short_long_seq,
        "compensatory_pause_like_count": compensatory,
        "bigeminy_like_pattern_count": bigeminy_count,
        "trigeminy_like_pattern_count": trigeminy_count,
        "atrial_tachy_like_run_count": tachy_runs,
    }
