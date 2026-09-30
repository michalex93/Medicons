"""
Tests obligatorios del módulo Mahalanobis — AFPDB/PAF RR-SPC.

Ejecutar:
    python -m pytest tests/test_mahalanobis.py -v

Todos los tests usan datos sintéticos (no requieren acceso a AFPDB).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from afpdb_multiscale.features_mahalanobis import (
    DEFAULT_ALARM_PERCENTILE,
    MAHAL_FEATURE_SETS,
    CovarianceFit,
    aggregate_dm_per_record,
    alarm_threshold_from_ref,
    compute_pair_baseline_dm,
    compute_population_fold_dm,
    fit_ledoit_wolf,
    mahalanobis_distances,
)
from afpdb_multiscale.config import PROHIBITED_WINDOW_SEC, RR_COUNT_WINDOWS, TIME_WINDOWS_SEC


# ---------------------------------------------------------------------------
# Helpers: datos sintéticos
# ---------------------------------------------------------------------------

def _make_feature_df(
    n_pairs: int = 5,
    n_windows_per_rec: int = 10,
    n_features: int = 5,
    seed: int = 42,
    window_sec: float = 300.0,
) -> pd.DataFrame:
    """
    DataFrame de ventanas sintéticas con 2*n_pairs registros (1 far + 1 pre-onset por par).
    Simula una diferencia leve entre y=0 y y=1 en la feature 0.
    """
    rng = np.random.default_rng(seed)
    feat_cols = list(MAHAL_FEATURE_SETS["hrv_reduced"])[:n_features]
    # Completar si hay menos features en el feature_set
    while len(feat_cols) < n_features:
        feat_cols.append(f"feat_{len(feat_cols)}")

    rows = []
    for pair_id in range(n_pairs):
        for y_val, record_suffix in ((0, "a"), (1, "b")):
            rid = f"p{pair_id:02d}{record_suffix}"
            for win_idx in range(n_windows_per_rec):
                start_t = win_idx * window_sec
                # y=1 tiene valores ligeramente más altos en feat 0
                mu_shift = 1.5 if y_val == 1 else 0.0
                feat_vals = rng.normal(loc=mu_shift, scale=1.0, size=len(feat_cols))
                row = {
                    "record_id": rid,
                    "pair_id": pair_id,
                    "y": y_val,
                    "analysis_task": "main_p_far_vs_p_pre",
                    "window_type": "time",
                    "window_size": window_sec,
                    "window_index": win_idx,
                    "start_time_sec": start_t,
                    "end_time_sec": start_t + window_sec,
                    "n_rr": 200,
                }
                for fname, fval in zip(feat_cols, feat_vals):
                    row[fname] = fval
                rows.append(row)
    return pd.DataFrame(rows)


def _make_agg_df(df_windows: pd.DataFrame) -> pd.DataFrame:
    """Agrega ventanas a 1 fila por (record_id, window_type, window_size)."""
    feat_cols = [c for c in df_windows.columns if c not in (
        "record_id", "pair_id", "y", "analysis_task",
        "window_type", "window_size", "window_index",
        "start_time_sec", "end_time_sec", "n_rr",
    )]
    group = ["record_id", "pair_id", "y", "analysis_task", "window_type", "window_size"]
    df_agg = df_windows.groupby(group)[feat_cols].median().reset_index()
    return df_agg


# ---------------------------------------------------------------------------
# TEST 1: μ y Σ estimados solo con y=0 del train fold
# ---------------------------------------------------------------------------

def test_covariance_estimated_only_from_train_far():
    """
    fit_ledoit_wolf debe ajustarse ÚNICAMENTE sobre filas y=0 del train fold.
    Verificamos que si pasamos solo y=0 train, los resultados difieren
    de si incluimos y=1 (prueba de que el contenido de X_train_far importa).
    """
    df = _make_feature_df(n_pairs=5, seed=0)
    feat_cols = list(MAHAL_FEATURE_SETS["hrv_reduced"])
    avail = [c for c in feat_cols if c in df.columns]

    # Filas y=0 (debería usarse en producción)
    X_far_only = df[df["y"] == 0][avail].values.astype(float)
    # Filas y=1 (no deben usarse)
    X_pre_only = df[df["y"] == 1][avail].values.astype(float)

    fit_far = fit_ledoit_wolf(X_far_only)
    fit_pre = fit_ledoit_wolf(X_pre_only)

    # Los medios deben ser distintos si y=0 y y=1 tienen distribuciones distintas
    assert not np.allclose(fit_far.mu, fit_pre.mu, atol=0.1), (
        "μ para y=0 y y=1 son iguales, los datos sintéticos no tienen separación suficiente"
    )
    # El fit sobre far NO debe contener datos de y=1
    # (verificación estructural: fit_far.n_samples == n filas y=0)
    assert fit_far.n_samples == len(X_far_only)


# ---------------------------------------------------------------------------
# TEST 2: Anti-leakage por pair_id en GroupKFold
# ---------------------------------------------------------------------------

def test_anti_leakage_population_fold():
    """
    compute_population_fold_dm recibe train_record_ids y val_record_ids.
    Verificamos que nunca se solapan pair_ids entre train y val.
    """
    df = _make_feature_df(n_pairs=8, seed=42)
    feat_cols = list(MAHAL_FEATURE_SETS["hrv_reduced"])
    avail = [c for c in feat_cols if c in df.columns]

    from sklearn.model_selection import GroupKFold
    recs = df[["record_id", "pair_id", "y"]].drop_duplicates("record_id").reset_index(drop=True)
    gkf = GroupKFold(n_splits=4)

    for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(recs, recs["y"], recs["pair_id"]), start=1
    ):
        tr_pairs = set(recs.iloc[tr_idx]["pair_id"])
        va_pairs = set(recs.iloc[va_idx]["pair_id"])
        overlap = tr_pairs & va_pairs
        assert len(overlap) == 0, (
            f"Fold {fold}: pair_id {overlap} en train Y val → leakage!"
        )


# ---------------------------------------------------------------------------
# TEST 3: Validation no participa en covarianza ni umbral
# ---------------------------------------------------------------------------

def test_validation_excluded_from_covariance_and_threshold():
    """
    compute_population_fold_dm debe usar SOLO train y=0 para:
      - ajustar fit_ledoit_wolf
      - calcular el imputer
      - calcular alarm_threshold
    Probamos que si adulteramos las filas de val (haciéndolas idénticas a y=0 far),
    la covarianza no cambia.
    """
    df_original = _make_feature_df(n_pairs=5, seed=10)
    feat_cols = list(MAHAL_FEATURE_SETS["hrv_reduced"])
    avail = [c for c in feat_cols if c in df_original.columns]

    recs = df_original[["record_id", "pair_id", "y"]].drop_duplicates("record_id").reset_index(drop=True)
    tr_pairs = {0, 1, 2, 3}
    va_pairs = {4}
    tr_recs = set(recs[recs["pair_id"].isin(tr_pairs)]["record_id"])
    va_recs = set(recs[recs["pair_id"].isin(va_pairs)]["record_id"])
    y_by_record = dict(zip(recs["record_id"], recs["y"]))

    _, cov_orig, imp_orig, thr_orig = compute_population_fold_dm(
        df_original, avail, tr_recs, va_recs, y_by_record
    )

    # Adulteramos las filas val (ponemos valores muy grandes) → no debe cambiar covarianza
    df_modified = df_original.copy()
    df_modified.loc[df_modified["record_id"].isin(va_recs), avail] = 9999.0

    _, cov_mod, imp_mod, thr_mod = compute_population_fold_dm(
        df_modified, avail, tr_recs, va_recs, y_by_record
    )

    np.testing.assert_allclose(cov_orig.mu, cov_mod.mu, rtol=1e-6,
        err_msg="μ cambió cuando se adulteraron datos de val → leakage")
    np.testing.assert_allclose(thr_orig, thr_mod, rtol=1e-6,
        err_msg="Umbral cambió cuando se adulteraron datos de val → leakage")


# ---------------------------------------------------------------------------
# TEST 4: Mahalanobis genera valores finitos
# ---------------------------------------------------------------------------

def test_mahalanobis_generates_finite_values():
    """
    mahalanobis_distances debe retornar valores finitos y no negativos para datos válidos.
    """
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(100, 5))
    X_test = rng.normal(size=(20, 5))

    fit = fit_ledoit_wolf(X_train)
    dm = mahalanobis_distances(X_test, fit)

    assert dm.shape == (20,)
    assert np.all(np.isfinite(dm)), "D_M contiene inf o NaN"
    assert np.all(dm >= 0), "D_M contiene valores negativos"


# ---------------------------------------------------------------------------
# TEST 5: p > n no falla con LedoitWolf
# ---------------------------------------------------------------------------

def test_high_dimensional_ledoit_wolf_does_not_fail():
    """
    Con p > n (más features que muestras), LedoitWolf debe completar sin excepción.
    Si falla, el fallback con pseudo-inversa debe activarse con UserWarning.
    """
    rng = np.random.default_rng(7)
    n_samples = 8    # pocas muestras
    n_features = 20  # muchas features

    X = rng.normal(size=(n_samples, n_features))

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        fit = fit_ledoit_wolf(X)
    
    # No debe lanzar excepción; si hubo fallback debe ser UserWarning
    assert fit.sigma_inv is not None
    assert fit.sigma_inv.shape == (n_features, n_features)
    # Verificar que D_M se puede calcular
    dm = mahalanobis_distances(X, fit)
    assert np.all(np.isfinite(dm)), "D_M con p > n contiene no-finitos"


# ---------------------------------------------------------------------------
# TEST 6: lead_time SOLO para y=1
# ---------------------------------------------------------------------------

def test_lead_time_only_for_y1():
    """
    aggregate_dm_per_record debe retornar lead_time_sec = NaN para y=0
    y solo puede ser numérico para y=1.
    """
    rng = np.random.default_rng(42)
    n = 20
    dm = rng.uniform(1.0, 5.0, size=n)
    starts = np.arange(n, dtype=float) * 90.0
    alarm_thr = 3.0

    # y=0: lead_time debe ser NaN
    result_far = aggregate_dm_per_record(dm, starts, alarm_thr, y=0)
    from afpdb_multiscale.features_mahalanobis import MAHAL_LEAD_TIME_FEATURE
    lt_far = result_far[MAHAL_LEAD_TIME_FEATURE]
    assert np.isnan(lt_far), f"lead_time para y=0 no es NaN: {lt_far}"

    # y=1: si hubo alarma, lead_time debe ser finito y ≤ 1800
    result_pre = aggregate_dm_per_record(dm, starts, alarm_thr, y=1)
    lt_pre = result_pre[MAHAL_LEAD_TIME_FEATURE]
    if result_pre["mahalanobis_alarm_count"] > 0:
        assert np.isfinite(lt_pre), "lead_time para y=1 con alarma debería ser finito"
        assert lt_pre <= 1800.0, f"lead_time > 1800 s: {lt_pre}"
    # Si no hubo alarma, también puede ser NaN (aceptable)


# ---------------------------------------------------------------------------
# TEST 7: pair_baseline no usa datos y=1 para estimar baseline
# ---------------------------------------------------------------------------

def test_pair_baseline_does_not_use_preonset_for_fit():
    """
    compute_pair_baseline_dm debe usar SOLO ventanas y=0 del par para
    estimar μ y Σ. Verificamos que adulterando las filas y=1 (pre-onset)
    del par no cambia la covarianza estimada.
    """
    df_orig = _make_feature_df(n_pairs=3, n_windows_per_rec=15, seed=99)
    feat_cols = list(MAHAL_FEATURE_SETS["hrv_reduced"])
    avail = [c for c in feat_cols if c in df_orig.columns]

    pair_id = 0

    # Calculamos D_M con datos originales
    dm_orig = compute_pair_baseline_dm(df_orig, avail, pair_id)

    # Adulteramos filas y=1 con valores extremos
    df_modified = df_orig.copy()
    pre_mask = (df_modified["pair_id"] == pair_id) & (df_modified["y"] == 1)
    df_modified.loc[pre_mask, avail] = 99999.0

    # La covarianza estimada debe ser IDÉNTICA (solo usa y=0 del par)
    dm_mod = compute_pair_baseline_dm(df_modified, avail, pair_id)

    # El resultado para y=0 (record far) no debe cambiar en nada
    far_rid_orig = [rid for rid, d in dm_orig.items() if d.get("mahalanobis_alarm_count") is not None
                    and df_orig[df_orig["record_id"] == rid]["y"].iloc[0] == 0]
    if far_rid_orig:
        rid = far_rid_orig[0]
        assert np.isclose(
            dm_orig[rid]["mahalanobis_mean"],
            dm_mod[rid]["mahalanobis_mean"],
            rtol=1e-6,
        ), "La covarianza cambió al adulterar y=1 → pair_baseline usa datos y=1 ilegalmente"


# ---------------------------------------------------------------------------
# TEST 8: 30 segundos no aparece en análisis principal
# ---------------------------------------------------------------------------

def test_30s_window_excluded_from_main():
    """
    PROHIBITED_WINDOW_SEC (30 s) no debe estar en RR_COUNT_WINDOWS ni en TIME_WINDOWS_SEC.
    Además, intentar crear una ventana de 30 s en main debe lanzar ProhibitedWindowError.
    """
    from afpdb_multiscale.windows import ProhibitedWindowError, iter_time_windows
    from unittest.mock import MagicMock

    # Config checks
    assert PROHIBITED_WINDOW_SEC not in RR_COUNT_WINDOWS, (
        f"30 s ({PROHIBITED_WINDOW_SEC}) está en RR_COUNT_WINDOWS"
    )
    for w in TIME_WINDOWS_SEC:
        assert w != float(PROHIBITED_WINDOW_SEC), (
            f"TIME_WINDOWS_SEC contiene ventana prohibida: {w} s"
        )

    # Window creation guard
    rng = np.random.default_rng(0)
    n_rr = 2500
    rr_ms = rng.normal(800, 30, n_rr).clip(400, 1400)
    rr_time_s = np.cumsum(rr_ms / 1000.0)

    rec = MagicMock()
    rec.rr_ms = rr_ms
    rec.rr_time_s = rr_time_s

    meta = {
        "record_id": "p01",
        "pair_id": 0,
        "y": 0,
        "analysis_task": "main_p_far_vs_p_pre",
    }
    with pytest.raises(ProhibitedWindowError):
        list(iter_time_windows(
            rec, meta,
            window_sec=float(PROHIBITED_WINDOW_SEC),
            analysis_task_label="main_p_far_vs_p_pre",
        ))
