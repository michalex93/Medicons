"""
Tests obligatorios del pipeline multi-escala AFPDB/PAF.

Ejecutar:
    python -m pytest tests/test_multiscale.py -v

No requieren acceso a AFPDB real; usan datos sintéticos reproducibles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from afpdb_multiscale.config import (
    PROHIBITED_WINDOW_SEC,
    RANDOM_STATE,
    RR_COUNT_WINDOWS,
    TIME_WINDOWS_SEC,
)
from afpdb_multiscale.features_ectopy import compute_ectopy_like_features
from afpdb_multiscale.features_hrv import compute_hrv_basic, compute_poincare
from afpdb_multiscale.features_spc import compute_spc_features
from afpdb_multiscale.loader import build_main_manifest, build_secondary_manifest
from afpdb_multiscale.windows import (
    ProhibitedWindowError,
    full_record_window,
    iter_rr_count_windows,
    iter_time_windows,
    zscore_with_record_baseline,
)


# ── fixtures sintéticos ──────────────────────────────────────────────────────

def _make_rr(n: int = 2500, mean_ms: float = 800.0, seed: int = 0) -> np.ndarray:
    """~2500 intervalos RR ≈ 30 min a ~80 bpm."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean_ms, scale=30.0, size=n).clip(300, 1800)


def _make_rr_time(rr_ms: np.ndarray) -> np.ndarray:
    return np.cumsum(rr_ms / 1000.0)


def _make_record_data(n: int = 2500, seed: int = 42):
    """Simula un RecordData sin acceder a disco."""
    from unittest.mock import MagicMock
    from afpdb_multiscale.config import BASELINE_N_RR

    rr = _make_rr(n=n, seed=seed)
    t = _make_rr_time(rr)
    n_base = min(BASELINE_N_RR, len(rr))
    baseline = rr[:n_base]
    mu = float(np.mean(baseline))
    sigma = float(np.std(baseline, ddof=1))

    rec = MagicMock()
    rec.rr_ms = rr
    rec.rr_time_s = t
    rec.mu_baseline = mu
    rec.sigma_baseline = sigma
    rec.n_rr_total = len(rr)
    return rec


def _make_meta(record_id: str = "p01", pair_id: int = 0, y: int = 0,
               task: str = "main_p_far_vs_p_pre") -> dict:
    return {
        "record_id": record_id,
        "pair_id": pair_id,
        "y": y,
        "analysis_task": task,
    }


# ── TEST 1: Ventanas RR-count (50, 100, 128, 200) ────────────────────────────

@pytest.mark.parametrize("n_rr_win", RR_COUNT_WINDOWS)
def test_rr_count_windows_correctness(n_rr_win):
    """Cada ventana rr_count debe tener exactamente n_rr_win intervalos."""
    rec = _make_record_data(n=2500)
    meta = _make_meta()
    wins = list(iter_rr_count_windows(rec, meta, n_rr_win))

    assert len(wins) > 0, f"No se generaron ventanas para n_rr={n_rr_win}"
    for w in wins:
        assert w["n_rr"] == n_rr_win, f"Ventana con n_rr={w['n_rr']} ≠ {n_rr_win}"
        assert len(w["_rr_ms"]) == n_rr_win
        assert w["window_type"] == "rr_count"
        assert w["window_size"] == n_rr_win

    # No solapamiento: start del siguiente ≥ end del anterior
    starts = [w["start_time_sec"] for w in wins]
    ends = [w["end_time_sec"] for w in wins]
    for i in range(len(wins) - 1):
        assert starts[i + 1] >= ends[i] - 1e-6, "Ventanas solapadas en rr_count"


# ── TEST 2: Ventana temporal de 5 minutos ────────────────────────────────────

def test_time_window_5min():
    """Ventanas de 5 min sobre registro de 30 min → ~6 ventanas, cada una ≤ 300 s."""
    rec = _make_record_data(n=2500)
    meta = _make_meta()
    wins = list(iter_time_windows(rec, meta, window_sec=300.0))

    assert len(wins) >= 5, f"Se esperaban ≥5 ventanas de 5 min, obtenidas: {len(wins)}"
    for w in wins:
        duration = w["end_time_sec"] - w["start_time_sec"]
        assert duration <= 300.1, f"Ventana con duración {duration:.1f} s > 300 s"
        assert w["window_type"] == "time"
        assert w["window_size"] == 300.0


# ── TEST 3: Registro completo (full_record) ───────────────────────────────────

def test_full_record_window():
    """full_record debe devolver una única ventana con todos los RR."""
    rec = _make_record_data(n=2500)
    meta = _make_meta()
    win = full_record_window(rec, meta)

    assert win["window_type"] == "full_record"
    assert win["n_rr"] == len(rec.rr_ms)
    assert win["window_index"] == 0
    assert len(win["_rr_ms"]) == len(rec.rr_ms)


# ── TEST 4: Ventana de 30 s PROHIBIDA en análisis principal ──────────────────

def test_30s_window_prohibited_in_main():
    """Intentar ventana de 30 s en tarea principal debe lanzar ProhibitedWindowError."""
    rec = _make_record_data(n=2500)
    meta = _make_meta(task="main_p_far_vs_p_pre")
    with pytest.raises(ProhibitedWindowError):
        list(iter_time_windows(
            rec, meta,
            window_sec=float(PROHIBITED_WINDOW_SEC),
            analysis_task_label="main_p_far_vs_p_pre",
        ))


def test_30s_window_not_in_rr_count_sizes():
    """30 s no debe estar en RR_COUNT_WINDOWS ni TIME_WINDOWS_SEC de config."""
    # 30 s en RR ≈ ~40 RR a 80 bpm; verificamos que 30 no está explícitamente
    assert PROHIBITED_WINDOW_SEC not in RR_COUNT_WINDOWS
    # TIME_WINDOWS_SEC solo debe contener 300 s
    for w in TIME_WINDOWS_SEC:
        assert w != float(PROHIBITED_WINDOW_SEC), (
            f"TIME_WINDOWS_SEC contiene ventana prohibida: {w} s"
        )


# ── TEST 5: Anti-leakage por pair_id ─────────────────────────────────────────

def test_anti_leakage_pair_id():
    """En GroupKFold por pair_id, ningún pair_id puede estar en train y val simultáneamente."""
    from sklearn.model_selection import GroupKFold
    from afpdb_multiscale.config import GKF_N_SPLITS

    # Simula un DataFrame con 50 registros (25 pares)
    rng = np.random.default_rng(RANDOM_STATE)
    n_records = 50
    df = pd.DataFrame({
        "pair_id": np.repeat(np.arange(25), 2),
        "y": np.tile([0, 1], 25),
        "feature_x": rng.normal(size=n_records),
    })

    gkf = GroupKFold(n_splits=GKF_N_SPLITS)
    groups = df["pair_id"]
    X = df[["feature_x"]]
    y = df["y"]

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), start=1):
        g_tr = set(groups.iloc[tr].unique())
        g_va = set(groups.iloc[va].unique())
        overlap = g_tr & g_va
        assert len(overlap) == 0, (
            f"Fold {fold}: pair_id {overlap} aparece en train Y val → leakage!"
        )


# ── TEST 6: p_far vs p_pre NO mezclado con p* vs n* ─────────────────────────

def test_tasks_not_mixed():
    """Los dos manifiestos deben tener analysis_task distintos y no mezclarse."""
    main_manifest = build_main_manifest()
    sec_manifest = build_secondary_manifest()

    main_tasks = {r["analysis_task"] for r in main_manifest}
    sec_tasks = {r["analysis_task"] for r in sec_manifest}

    assert main_tasks == {"main_p_far_vs_p_pre"}, f"Task inesperada en main: {main_tasks}"
    assert sec_tasks == {"secondary_p_vs_n"}, f"Task inesperada en secondary: {sec_tasks}"
    assert main_tasks.isdisjoint(sec_tasks), "Tareas mezcladas entre manifiestos"

    # En la tarea principal NO debe haber registros n*
    main_ids = {r["record_id"] for r in main_manifest}
    n_in_main = [rid for rid in main_ids if rid.startswith("n")]
    assert len(n_in_main) == 0, f"Registros n* en tarea principal: {n_in_main}"

    # En la tarea principal NO debe haber registros *c
    c_in_main = [rid for rid in main_ids if rid.endswith("c")]
    assert len(c_in_main) == 0, f"Registros *c en tarea principal: {c_in_main}"


# ── TEST 7: Features numéricas sin NaN/inf silenciosos ───────────────────────

def test_features_no_silent_nan_inf():
    """
    Para RR sintético bien formado, las features principales no deben ser
    todas NaN ni contener inf silencioso.
    """
    rng = np.random.default_rng(42)
    rr = rng.normal(800, 30, size=200).clip(400, 1400)
    t = np.cumsum(rr / 1000.0)

    # HRV
    hrv = compute_hrv_basic(rr)
    for k, v in hrv.items():
        assert not np.isinf(v), f"HRV: {k} = inf"
        assert v is not None, f"HRV: {k} es None"

    # Poincaré
    poi = compute_poincare(rr)
    for k, v in poi.items():
        assert not np.isinf(v), f"Poincaré: {k} = inf"

    # SPC (con baseline artificial)
    mu = float(np.mean(rr))
    sigma = float(np.std(rr, ddof=1))
    spc = compute_spc_features(rr, t, mu, sigma)
    for k, v in spc.items():
        if v is not None and not np.isnan(float(v)):
            assert not np.isinf(float(v)), f"SPC: {k} = inf"

    # Ectopy
    ect = compute_ectopy_like_features(rr)
    for k, v in ect.items():
        if v is not None and not np.isnan(float(v)):
            assert not np.isinf(float(v)), f"Ectopy: {k} = inf"
            assert float(v) >= 0, f"Ectopy: {k} es negativo"


# ── TEST 8: Reproducibilidad con random_state fijo ───────────────────────────

def test_reproducibility_fixed_random_state():
    """
    Dos corridas del mismo modelo con random_state fijo deben producir
    exactamente los mismos coeficientes.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(50, 5))
    y = rng.integers(0, 2, size=50)

    def _fit_coef(seed):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=seed)),
        ])
        pipe.fit(X, y)
        return pipe.named_steps["clf"].coef_.ravel().copy()

    coef1 = _fit_coef(RANDOM_STATE)
    coef2 = _fit_coef(RANDOM_STATE)
    np.testing.assert_array_equal(coef1, coef2, err_msg="Coeficientes no reproducibles")


# ── TEST 9: lead_time solo para y=1 ──────────────────────────────────────────

def test_lead_time_only_for_preonset():
    """
    En el dataset agregado, lead_time_sec debe ser NaN para todos los y=0
    y solo puede ser numérico (o NaN) para y=1.
    """
    from afpdb_multiscale.build_dataset import aggregate_to_record_level

    rng = np.random.default_rng(42)
    n_rr_win = 100
    n_wins_per_rec = 20

    # Simula un pequeño DataFrame de ventanas
    rows = []
    for pair_id in range(5):
        for y_val, record_id in ((0, f"p{pair_id*2+1:02d}"), (1, f"p{pair_id*2+2:02d}")):
            for win_idx in range(n_wins_per_rec):
                start_t = win_idx * 90.0
                # CUSUM alarma simulada solo en y=1, win_idx > 10
                cusum_alarm_t = (
                    float(rng.uniform(5, 85)) if (y_val == 1 and win_idx > 10) else np.nan
                )
                rows.append({
                    "record_id": record_id,
                    "pair_id": pair_id,
                    "y": y_val,
                    "analysis_task": "main_p_far_vs_p_pre",
                    "window_type": "rr_count",
                    "window_size": n_rr_win,
                    "window_index": win_idx,
                    "start_time_sec": start_t,
                    "end_time_sec": start_t + 90.0,
                    "n_rr": n_rr_win,
                    "cusum_first_alarm_time_sec": cusum_alarm_t,
                    "cusum_alarm_count": 0 if np.isnan(cusum_alarm_t) else 3,
                    # Rellenar demás features con 0 para que no falle el agg
                    **{k: 0.0 for k in [
                        "mean_rr", "median_rr", "std_rr", "mad_rr",
                        "rmssd", "sdsd", "cv_rr", "rr_diff_mean", "rr_diff_std",
                        "sd1", "sd2", "sd1_sd2_ratio",
                        "shewhart_out_count", "shewhart_out_rate", "shewhart_max_abs_z",
                        "shewhart_first_alarm_pos", "shewhart_first_alarm_time_sec",
                        "cusum_pos_max", "cusum_neg_max", "cusum_abs_max",
                        "cusum_last_pos", "cusum_last_neg", "cusum_first_alarm_pos",
                        "cusum_time_in_alarm", "cusum_alarm_density",
                        "ewma_last", "ewma_max_abs", "ewma_slope",
                        "ewma_alarm_count", "ewma_first_alarm_time_sec",
                        "mr_mean", "mr_median", "mr_max", "mr_mad", "mr_out_count",
                        "longest_run_above_median", "longest_run_below_median",
                        "trend_run_up_max", "trend_run_down_max", "n_runs_rule_violations",
                        "ectopy_like_count", "ectopy_like_density",
                        "short_long_sequence_count", "compensatory_pause_like_count",
                        "bigeminy_like_pattern_count", "trigeminy_like_pattern_count",
                        "atrial_tachy_like_run_count",
                    ]},
                })

    df_raw = pd.DataFrame(rows)
    df_agg = aggregate_to_record_level(df_raw)

    # Regla crítica: lead_time_sec debe ser NaN para y=0
    y0_rows = df_agg[df_agg["y"] == 0]
    assert y0_rows["lead_time_sec"].isna().all(), (
        "lead_time_sec no debería tener valores para y=0 (far from PAF)"
    )

    # Para y=1: puede ser NaN (si no hubo alarma) o un número ≤ 1800
    y1_rows = df_agg[df_agg["y"] == 1]
    valid_lt = y1_rows["lead_time_sec"].dropna()
    assert (valid_lt <= 1800.0).all(), "lead_time_sec > 1800 s en y=1"
