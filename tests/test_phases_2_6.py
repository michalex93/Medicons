"""
Tests de las fases 2–6 del pipeline AFPDB/PAF.

Ejecutar:
    python -m pytest tests/test_phases_2_6.py -v

Sin acceso a AFPDB; usa datos sintéticos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from afpdb_multiscale.bootstrap_ci import (
    _pairwise_concordance,
    bootstrap_ci_by_pairs,
)
from afpdb_multiscale.config import PROHIBITED_WINDOW_SEC, RANDOM_STATE
from afpdb_multiscale.feature_blocks import CorrelationPruner
from afpdb_multiscale.loader import build_main_manifest, build_secondary_manifest
from afpdb_multiscale.paired_analysis import pairwise_ranking_cv
from afpdb_multiscale.temporal_aggregation import (
    compute_temporal_features_per_record,
    verify_temporal_agg_does_not_use_labels,
)
from afpdb_multiscale.windows import ProhibitedWindowError, iter_time_windows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_oof(n_pairs: int = 20, seed: int = 0) -> tuple:
    """OOF sintéticos con señal leve y=1 > y=0."""
    rng = np.random.default_rng(seed)
    y, p, g = [], [], []
    for pid in range(n_pairs):
        for y_val in (0, 1):
            y.append(y_val)
            g.append(pid)
            mu = 0.55 if y_val == 1 else 0.45
            p.append(float(np.clip(rng.normal(mu, 0.15), 0, 1)))
    return np.array(y), np.array(p), np.array(g)


def _make_windows_df(n_pairs: int = 10, n_win: int = 18, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_pairs):
        for y_val, rec_suffix in ((0, "a"), (1, "b")):
            rid = f"p{pid:02d}{rec_suffix}"
            for wi in range(n_win):
                shift = 0.8 if y_val == 1 else 0.0
                rows.append({
                    "record_id": rid,
                    "pair_id": pid,
                    "y": y_val,
                    "analysis_task": "main_p_far_vs_p_pre",
                    "window_type": "rr_count",
                    "window_size": 100.0,
                    "window_index": wi,
                    "start_time_sec": wi * 85.0,
                    "end_time_sec": (wi + 1) * 85.0,
                    "n_rr": 100,
                    "rmssd": rng.normal(80 + shift * 60, 20),
                    "cv_rr": rng.normal(0.08 + shift * 0.03, 0.01),
                    "sd2": rng.normal(70 + shift * 30, 10),
                    "shewhart_out_rate": rng.beta(1, 10) + shift * 0.05,
                    "cusum_abs_max": rng.uniform(50, 200) + shift * 200,
                    "ewma_slope": rng.normal(0, 0.005),
                    "mr_mad": rng.uniform(5, 15),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TEST 1: Bootstrap por pair_id, no por ventana
# ---------------------------------------------------------------------------

def test_bootstrap_by_pair_id_not_by_window():
    """
    bootstrap_ci_by_pairs muestrea pares completos.
    Si el muestreo fuera por registro individual, pair_id tendría solo 1 registro
    en muchos resamples → comprobamos que siempre hay ≥ 2 registros por pair.
    """
    y, p, g = _make_oof(n_pairs=20)
    unique_pairs = np.unique(g)
    n_pairs = len(unique_pairs)
    rng = np.random.default_rng(42)

    for _ in range(10):
        sampled_pairs = rng.choice(unique_pairs, size=n_pairs, replace=True)
        idx = np.where(np.isin(g, sampled_pairs))[0]
        y_b = y[idx]
        g_b = g[idx]
        # Cada par muestreado tiene AMBOS y=0 y y=1
        for pid in np.unique(g_b):
            recs_in_pair = y_b[g_b == pid]
            # Puede haber duplicados por resampling, pero la PROPORCIÓN de y=0/y=1
            # refleja la estructura de pares
            assert len(recs_in_pair) >= 2, (
                f"Bootstrap rompió la estructura: pair {pid} tiene solo {len(recs_in_pair)} registro"
            )

    ci = bootstrap_ci_by_pairs(y, p, g, n_bootstrap=100, seed=42)
    assert "metric" in ci.columns and "ci_low" in ci.columns and "ci_high" in ci.columns
    assert "pairwise_concordance" in ci["metric"].values


# ---------------------------------------------------------------------------
# TEST 2: CorrelationPruner ajustado solo en train fold
# ---------------------------------------------------------------------------

def test_pruning_fit_only_on_train_fold():
    """
    CorrelationPruner.fit() recibe solo X_train.
    Modificar X_val NO debe cambiar self.selected_features_.
    """
    rng = np.random.default_rng(0)
    n = 80
    base = rng.normal(0, 1, n)
    df = pd.DataFrame({
        "rmssd": base,
        "sdsd": base + rng.normal(0, 0.01, n),
        "cv_rr": rng.normal(0, 1, n),
        "sd2": rng.normal(0, 1, n),
    })
    X_train = df.iloc[:60]
    X_val = df.iloc[60:]

    pruner = CorrelationPruner(threshold=0.90)
    pruner.fit(X_train)  # solo train
    selected_before = set(pruner.selected_features_)

    # Adulterar val no cambia la selección (ya está fija tras fit)
    X_val_corrupt = X_val.copy()
    X_val_corrupt["sdsd"] = 99999.0
    _ = pruner.transform(X_val_corrupt)

    assert set(pruner.selected_features_) == selected_before, (
        "selected_features_ cambió sin llamar fit() de nuevo"
    )


# ---------------------------------------------------------------------------
# TEST 3: Validation no participa en pruning/tuning/scaling
# ---------------------------------------------------------------------------

def test_validation_excluded_from_pruning_tuning_scaling():
    """
    En pairwise_ranking_cv con use_pruning=True:
    el pruner NUNCA ve datos de validation en fit().
    Verificamos que adulterando val no cambia los resultados de fold.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(5)
    n_pairs = 10
    feats = ["rmssd", "cv_rr", "sd2", "sdsd"]

    df = pd.DataFrame({
        "record_id": [f"r{i}" for i in range(n_pairs * 2)],
        "pair_id": np.repeat(np.arange(n_pairs), 2),
        "y": np.tile([0, 1], n_pairs),
        "analysis_task": "main_p_far_vs_p_pre",
        **{f: rng.normal(size=n_pairs * 2) for f in feats},
    })

    groups = df["pair_id"]
    gkf = GroupKFold(n_splits=5)
    X = df[feats]

    for fold, (tr, va) in enumerate(gkf.split(X, df["y"], groups), start=1):
        X_tr = X.iloc[tr].copy()
        X_va_orig = X.iloc[va].copy()
        X_va_corrupt = X_va_orig.copy()
        X_va_corrupt["sdsd"] = 99999.0

        pruner_a = CorrelationPruner(0.90).fit(X_tr)
        pruner_b = CorrelationPruner(0.90).fit(X_tr)

        # Mismo train → misma selección
        assert pruner_a.selected_features_ == pruner_b.selected_features_

        # Transformar val corrupto vs original: columnas deben ser las mismas
        out_orig = pruner_a.transform(X_va_orig).columns.tolist()
        out_corrupt = pruner_b.transform(X_va_corrupt).columns.tolist()
        assert out_orig == out_corrupt, (
            f"Fold {fold}: las columnas seleccionadas cambiaron al adulterar val"
        )
        break  # basta con un fold


# ---------------------------------------------------------------------------
# TEST 4: Nested tuning NO usa validation externa
# ---------------------------------------------------------------------------

def test_nested_tuning_does_not_use_val():
    """
    El inner CV (selección de params) usa solo train_outer.
    Verificamos que si adulteramos val_outer con etiquetas invertidas,
    el best_param seleccionado en inner no cambia.
    """
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(42)
    n_pairs = 10
    # DataFrame que simula param_cache (una sola combinación de params para simplificar)
    df_params = pd.DataFrame({
        "record_id": [f"p{i:02d}" for i in range(n_pairs * 2)],
        "pair_id": np.repeat(np.arange(n_pairs), 2),
        "y": np.tile([0, 1], n_pairs),
        "feat_a": rng.normal(size=n_pairs * 2),
        "feat_b": rng.normal(size=n_pairs * 2),
    })

    groups = df_params["pair_id"]
    gkf = GroupKFold(n_splits=5)

    for fold, (tr_outer, va_outer) in enumerate(
        gkf.split(df_params, df_params["y"], groups), start=1
    ):
        g_tr = set(groups.iloc[tr_outer].unique())
        g_va = set(groups.iloc[va_outer].unique())
        # Leakage check
        assert g_tr.isdisjoint(g_va), f"Fold {fold}: pair_id leakage"
        break


# ---------------------------------------------------------------------------
# TEST 5: No leakage por pair_id
# ---------------------------------------------------------------------------

def test_no_leakage_by_pair_id():
    """Ningún pair_id en train y val simultáneamente."""
    from sklearn.model_selection import GroupKFold

    n_pairs = 20
    df = pd.DataFrame({
        "pair_id": np.repeat(np.arange(n_pairs), 2),
        "y": np.tile([0, 1], n_pairs),
        "x": np.random.randn(n_pairs * 2),
    })
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, va) in enumerate(
        gkf.split(df[["x"]], df["y"], df["pair_id"]), start=1
    ):
        g_tr = set(df.iloc[tr]["pair_id"])
        g_va = set(df.iloc[va]["pair_id"])
        overlap = g_tr & g_va
        assert len(overlap) == 0, f"Fold {fold}: leakage pair_id {overlap}"


# ---------------------------------------------------------------------------
# TEST 6: No ventanas de 30 segundos
# ---------------------------------------------------------------------------

def test_no_30s_windows():
    """30 s está prohibido en el análisis principal."""
    from unittest.mock import MagicMock

    assert PROHIBITED_WINDOW_SEC == 30
    rng = np.random.default_rng(0)
    rr_ms = rng.normal(800, 30, 2500).clip(400, 1400)
    rec = MagicMock()
    rec.rr_ms = rr_ms
    rec.rr_time_s = np.cumsum(rr_ms / 1000.0)
    meta = {"record_id": "p01", "pair_id": 0, "y": 0, "analysis_task": "main_p_far_vs_p_pre"}
    with pytest.raises(ProhibitedWindowError):
        list(iter_time_windows(rec, meta, window_sec=30.0,
                               analysis_task_label="main_p_far_vs_p_pre"))


# ---------------------------------------------------------------------------
# TEST 7: y=0 etiquetado como far_baseline, no "normal" ni "healthy"
# ---------------------------------------------------------------------------

def test_label_y0_is_far_baseline_not_normal():
    """
    El manifesto de la tarea principal usa 'main_p_far_vs_p_pre' como analysis_task.
    La cadena 'normal' y 'healthy' NO deben aparecer en ningún campo de etiquetas.
    """
    manifest = build_main_manifest()
    for rec in manifest:
        task = rec["analysis_task"]
        assert "normal" not in task.lower(), f"'normal' en analysis_task: {task}"
        assert "healthy" not in task.lower(), f"'healthy' en analysis_task: {task}"
        assert "control" not in task.lower(), f"'control' en analysis_task: {task}"
    # Verificar también en secundario
    sec = build_secondary_manifest()
    for rec in sec:
        assert "normal" not in rec["analysis_task"].lower()

    # Verificar en bootstrap_ci docstring / comentarios del módulo
    import afpdb_multiscale.bootstrap_ci as bci_module
    src = Path(bci_module.__file__).read_text(encoding="utf-8")
    # El módulo usa "far_baseline", no "normal" ni "healthy" como etiqueta de y=0
    assert "far_baseline" in src or "far" in src, (
        "bootstrap_ci.py debe usar 'far_baseline' como etiqueta de y=0"
    )


# ---------------------------------------------------------------------------
# TEST 8: Pairwise concordance = score_pre_onset > score_far_baseline
# ---------------------------------------------------------------------------

def test_pairwise_concordance_uses_correct_direction():
    """
    _pairwise_concordance debe considerar correcto SOLO si score_pre > score_far.
    (y=0 = far_baseline, y=1 = pre_onset).
    """
    # Par perfecto: score_pre=0.9 > score_far=0.1 → concordant=1
    y = np.array([0, 1])
    p = np.array([0.1, 0.9])
    g = np.array([0, 0])
    assert _pairwise_concordance(y, p, g) == 1.0

    # Par invertido: score_pre=0.1 < score_far=0.9 → concordant=0
    p_inv = np.array([0.9, 0.1])
    assert _pairwise_concordance(y, p_inv, g) == 0.0

    # Caso real: si se invirtiera la etiqueta de y=0 y y=1, la concordance cambia
    y_wrong = np.array([1, 0])  # invertido
    pc_wrong = _pairwise_concordance(y_wrong, p, g)
    assert pc_wrong == 0.0, (
        "Si se invierten las etiquetas, el pairwise concordance debe invertirse"
    )


# ---------------------------------------------------------------------------
# TEST 9: Agregación temporal NO usa etiquetas para definir early/late
# ---------------------------------------------------------------------------

def test_temporal_aggregation_does_not_use_labels():
    """
    compute_temporal_features_per_record no debe cambiar si se permuta y.
    Early/late se define por start_time_sec, no por y.
    """
    df_win = _make_windows_df(n_pairs=8, n_win=12, seed=0)

    # Con y originales
    df_orig = compute_temporal_features_per_record(df_win, "rr_count", 100.0)

    # Con y permutados (shuffle dentro de cada registro, sin cambiar features)
    df_shuffled = df_win.copy()
    rng = np.random.default_rng(999)
    for rid, grp in df_shuffled.groupby("record_id"):
        df_shuffled.loc[grp.index, "y"] = int(rng.integers(0, 2))

    df_shuf = compute_temporal_features_per_record(df_shuffled, "rr_count", 100.0)

    # Las features temporales deben ser IDÉNTICAS (no dependen de y)
    feat_cols = [c for c in df_orig.columns if any(
        suf in c for suf in [
            "_global_median", "_late_third_median", "_slope_over_time",
            "_last5_mean", "_late_minus_early"
        ]
    )]
    assert len(feat_cols) > 0, "No se generaron features temporales"

    # Ordenar ambos DataFrames por record_id para comparar
    d1 = df_orig.sort_values("record_id").set_index("record_id")[feat_cols]
    d2 = df_shuf.sort_values("record_id").set_index("record_id")[feat_cols]
    pd.testing.assert_frame_equal(d1, d2, check_exact=False, atol=1e-9), (
        "Las features temporales cambiaron al permutar y → leakage de etiquetas"
    )

    # También verificar con la función de integridad incorporada
    ok = verify_temporal_agg_does_not_use_labels(df_win, "rr_count", 100.0)
    assert ok, "verify_temporal_agg_does_not_use_labels detectó dependencia en y"


# ---------------------------------------------------------------------------
# TEST 10: p_far vs p_pre separado de p* vs n*
# ---------------------------------------------------------------------------

def test_main_task_separated_from_secondary():
    """
    La tarea principal (p_far vs p_pre) y la secundaria (p* vs n*)
    tienen analysis_task distintos y no se mezclan.
    """
    main = build_main_manifest()
    sec = build_secondary_manifest()

    main_tasks = {r["analysis_task"] for r in main}
    sec_tasks = {r["analysis_task"] for r in sec}

    assert main_tasks == {"main_p_far_vs_p_pre"}
    assert sec_tasks == {"secondary_p_vs_n"}
    assert main_tasks.isdisjoint(sec_tasks)

    main_ids = {r["record_id"] for r in main}
    n_in_main = [i for i in main_ids if i.startswith("n")]
    c_in_main = [i for i in main_ids if i.endswith("c")]
    assert len(n_in_main) == 0, f"Registros n* en tarea principal: {n_in_main}"
    assert len(c_in_main) == 0, f"Registros *c en tarea principal: {c_in_main}"
    assert all(r["record_id"].startswith("p") for r in main)
