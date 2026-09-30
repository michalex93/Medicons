"""
Orquesta la extracción de features multi-escala sobre todos los registros.

Flujo:
  1. Construir manifesto (p01-p50 para tarea principal).
  2. Para cada registro: cargar QRS, calcular baseline.
  3. Para cada tipo/tamaño de ventana: iterar ventanas, extraer HRV+SPC+Ectopy.
  4. Agregar ventanas → 1 fila por (registro, window_type, window_size).
  5. Añadir campos de lead_time (solo y=1) y false_alarm_info (solo y=0).
  6. Guardar CSV raw (ventanas individuales) y CSV agregado.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    FULL_RECORD_SEC,
    RANDOM_STATE,
    RR_COUNT_WINDOWS,
    ROOT,
    TABLES_DIR,
    TIME_WINDOWS_SEC,
    ALL_FEATURES,
    METADATA_FIELDS,
)
from .features_ectopy import compute_ectopy_like_features
from .features_hrv import compute_hrv_basic, compute_poincare
from .features_spc import compute_spc_features
from .loader import RecordData, build_main_manifest, build_secondary_manifest, load_record
from .windows import (
    full_record_window,
    iter_rr_count_windows,
    iter_time_windows,
)


# ---------------------------------------------------------------------------
# Extracción de features de una ventana
# ---------------------------------------------------------------------------

def _extract_window_features(
    window_dict: dict,
    rec: RecordData,
) -> dict:
    """Calcula HRV + SPC + Ectopy sobre una ventana ya extraída."""
    rr = window_dict["_rr_ms"]
    rr_t = window_dict["_rr_time_s"]

    hrv = compute_hrv_basic(rr)
    poi = compute_poincare(rr)
    spc = compute_spc_features(
        rr, rr_t,
        mu_baseline=rec.mu_baseline,
        sigma_baseline=rec.sigma_baseline,
    )
    ect = compute_ectopy_like_features(rr)

    # Copia solo campos de provenance (sin arrays internos)
    row = {k: v for k, v in window_dict.items() if not k.startswith("_")}
    row.update(hrv)
    row.update(poi)
    row.update(spc)
    row.update(ect)
    return row


# ---------------------------------------------------------------------------
# Extracción para un registro completo (todas las ventanas)
# ---------------------------------------------------------------------------

def extract_all_windows_for_record(
    rec: RecordData,
    meta: dict,
) -> list[dict]:
    rows = []

    # A) RR-count windows
    for n_rr in RR_COUNT_WINDOWS:
        for win in iter_rr_count_windows(rec, meta, n_rr):
            rows.append(_extract_window_features(win, rec))

    # B) Time windows (5 min)
    for w_sec in TIME_WINDOWS_SEC:
        for win in iter_time_windows(rec, meta, w_sec):
            rows.append(_extract_window_features(win, rec))

    # C) Full-record
    win = full_record_window(rec, meta)
    rows.append(_extract_window_features(win, rec))

    return rows


# ---------------------------------------------------------------------------
# Construcción del dataset completo
# ---------------------------------------------------------------------------

def build_raw_windows_dataframe(
    manifest: list[dict],
    pn_dir: str = "afpdb",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Devuelve DataFrame con UNA FILA POR VENTANA de todos los registros
    del manifest.
    """
    all_rows: list[dict] = []
    n_total = len(manifest)

    for i, meta in enumerate(manifest, start=1):
        rid = meta["record_id"]
        try:
            rec = load_record(rid, pn_dir=pn_dir)
        except Exception as exc:
            if verbose:
                print(f"  [WARN] {rid}: {exc!r}")
            continue
        rows = extract_all_windows_for_record(rec, meta)
        all_rows.extend(rows)
        if verbose and (i % 10 == 0 or i == n_total):
            print(f"  Procesados: {i}/{n_total} registros ({rid})", flush=True)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Agregación: 1 fila por (registro, window_type, window_size)
# ---------------------------------------------------------------------------

# Features cuyo mejor agregado es la suma
_SUM_FEATURES = {
    "shewhart_out_count", "cusum_alarm_count", "ewma_alarm_count",
    "mr_out_count", "n_runs_rule_violations",
    "ectopy_like_count", "short_long_sequence_count",
    "compensatory_pause_like_count", "bigeminy_like_pattern_count",
    "trigeminy_like_pattern_count", "atrial_tachy_like_run_count",
}
# Features cuyo mejor agregado es el mínimo (primera alarma)
_MIN_FEATURES = {
    "shewhart_first_alarm_time_sec", "cusum_first_alarm_time_sec",
    "ewma_first_alarm_time_sec",
}
# Features cuyo mejor agregado es el máximo
_MAX_FEATURES = {
    "shewhart_max_abs_z", "cusum_pos_max", "cusum_neg_max",
    "cusum_abs_max", "ewma_max_abs", "mr_max",
    "longest_run_above_median", "longest_run_below_median",
    "trend_run_up_max", "trend_run_down_max",
    "atrial_tachy_like_run_count",
}


def _agg_func(col: str) -> str:
    if col in _SUM_FEATURES:
        return "sum"
    if col in _MIN_FEATURES:
        return "min"
    if col in _MAX_FEATURES:
        return "max"
    return "median"


def aggregate_to_record_level(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega las filas de ventanas → 1 fila por (record_id, window_type, window_size).
    También calcula:
      cusum_first_alarm_abs_sec  = start_time_sec + cusum_first_alarm_time_sec
      lead_time_sec              = FULL_RECORD_SEC - cusum_first_alarm_abs_sec  (solo y=1)
      false_alarms_per_hour      = (cusum_alarm_count / record_duration_h)      (solo y=0)
    """
    # Calcular alarma absoluta antes de agregar
    df = df_raw.copy()
    df["_cusum_alarm_abs"] = df["start_time_sec"] + df["cusum_first_alarm_time_sec"]

    feature_cols = [c for c in df.columns if c in set(ALL_FEATURES)]
    group_keys = ["record_id", "pair_id", "y", "analysis_task",
                  "window_type", "window_size"]

    agg_dict: dict[str, str] = {}
    for col in feature_cols:
        agg_dict[col] = _agg_func(col)

    # Campos adicionales de provenance
    agg_dict["n_rr"] = "sum"
    agg_dict["end_time_sec"] = "max"
    agg_dict["_cusum_alarm_abs"] = "min"  # primera alarma absoluta

    df_agg = df.groupby(group_keys, as_index=False).agg(agg_dict)

    # lead_time: solo y=1; asumiendo PAF onset al final del registro (1800 s)
    df_agg["lead_time_sec"] = np.where(
        (df_agg["y"] == 1) & df_agg["_cusum_alarm_abs"].notna(),
        FULL_RECORD_SEC - df_agg["_cusum_alarm_abs"],
        np.nan,
    )
    # false_alarms_per_hour: solo y=0
    record_duration_h = df_agg["end_time_sec"] / 3600.0
    df_agg["false_alarms_per_hour"] = np.where(
        (df_agg["y"] == 0) & (record_duration_h > 0),
        df_agg.get("cusum_alarm_count", np.nan) / record_duration_h,
        np.nan,
    )
    df_agg.drop(columns=["_cusum_alarm_abs"], inplace=True)

    return df_agg


# ---------------------------------------------------------------------------
# Función principal del módulo
# ---------------------------------------------------------------------------

def build_and_save_datasets(
    verbose: bool = True,
    include_secondary: bool = True,
) -> dict[str, Path]:
    """
    Construye y guarda los datasets (raw + agregado) para tarea principal y secundaria.
    Llama siempre desde ROOT para que wfdb no interprete rutas como URLs.
    """
    os.chdir(ROOT)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # --- Tarea principal ---
    if verbose:
        print("=== Tarea principal: p_far vs p_pre ===")
    main_manifest = build_main_manifest()
    df_raw_main = build_raw_windows_dataframe(main_manifest, verbose=verbose)
    p_raw_main = TABLES_DIR / "windows_raw_main.csv"
    df_raw_main.to_csv(p_raw_main, index=False)
    saved["raw_main"] = p_raw_main

    df_agg_main = aggregate_to_record_level(df_raw_main)
    p_agg_main = TABLES_DIR / "Xy_aggregated_main.csv"
    df_agg_main.to_csv(p_agg_main, index=False)
    saved["agg_main"] = p_agg_main
    if verbose:
        print(f"  Raw: {len(df_raw_main)} filas -> Agregado: {len(df_agg_main)} filas")

    # --- Tarea secundaria ---
    if include_secondary:
        if verbose:
            print("=== Análisis secundario: p* vs n* ===")
        sec_manifest = build_secondary_manifest()
        df_raw_sec = build_raw_windows_dataframe(sec_manifest, verbose=verbose)
        p_raw_sec = TABLES_DIR / "windows_raw_secondary.csv"
        df_raw_sec.to_csv(p_raw_sec, index=False)
        saved["raw_secondary"] = p_raw_sec

        df_agg_sec = aggregate_to_record_level(df_raw_sec)
        p_agg_sec = TABLES_DIR / "Xy_aggregated_secondary.csv"
        df_agg_sec.to_csv(p_agg_sec, index=False)
        saved["agg_secondary"] = p_agg_sec

    return saved
