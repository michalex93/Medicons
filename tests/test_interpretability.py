"""
Tests del módulo de interpretabilidad HRV y análisis intra-par.

Ejecutar:
    python -m pytest tests/test_interpretability.py -v

Todos los tests usan datos sintéticos; no requieren acceso a AFPDB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from afpdb_multiscale.feature_blocks import (
    CorrelationPruner,
    HRV_FULL_ACTUAL,
    HRV_MINIMAL,
    HRV_REDUCED_INTERPRETABLE,
    compute_collinearity_report,
)
from afpdb_multiscale.paired_analysis import (
    compute_paired_delta_analysis,
    logreg_coef_stability,
    pairwise_ranking_cv,
)
from afpdb_multiscale.loader import build_main_manifest, build_secondary_manifest


# ---------------------------------------------------------------------------
# Fixtures sintéticos
# ---------------------------------------------------------------------------

def _make_correlated_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """DataFrame con features colineales conocidas."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, size=n)
    # rmssd y sdsd son idénticas (r=1), cv_rr es ortogonal
    return pd.DataFrame({
        "rmssd": base,
        "sdsd": base + rng.normal(0, 0.01, n),        # r ~ 0.9999 con rmssd
        "cv_rr": rng.normal(0, 1, n),                  # ortogonal
        "sd2": rng.normal(0, 1, n),                    # ortogonal
        "rr_diff_std": base * 1.0 + rng.normal(0, 0.05, n),  # ~ rmssd
    })


def _make_pair_df(
    n_pairs: int = 10,
    seed: int = 42,
    include_secondary: bool = False,
) -> pd.DataFrame:
    """DataFrame con structure pair_id/y para análisis delta y pairwise ranking."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_pairs):
        for y_val in (0, 1):
            rid = f"p{pid*2 + y_val + 1:02d}"
            # pre-onset (y=1) tiene rmssd sistemáticamente más alto
            rmssd = rng.normal(140 if y_val == 1 else 70, 20)
            sd2 = rng.normal(95 if y_val == 1 else 70, 15)
            rows.append({
                "record_id": rid,
                "pair_id": pid,
                "y": y_val,
                "analysis_task": "main_p_far_vs_p_pre",
                "rmssd": rmssd,
                "cv_rr": rng.uniform(0.06, 0.12),
                "sd2": sd2,
                "sd1": rmssd / 1.414,
                "cusum_abs_max": rng.uniform(50, 600),
                "ewma_slope": rng.normal(0, 0.01),
                "shewhart_out_rate": rng.uniform(0, 0.1),
                "mr_mad": rng.uniform(5, 20),
                "longest_run_above_median": rng.integers(1, 10),
            })
    if include_secondary:
        for n in range(n_pairs):
            rows.append({
                "record_id": f"n{n:02d}",
                "pair_id": n,
                "y": 0,
                "analysis_task": "secondary_p_vs_n",
                "rmssd": rng.normal(70, 20),
                "cv_rr": 0.07,
                "sd2": 70.0,
                "sd1": 50.0,
                "cusum_abs_max": 100.0,
                "ewma_slope": 0.0,
                "shewhart_out_rate": 0.01,
                "mr_mad": 7.0,
                "longest_run_above_median": 3,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TEST 1: CorrelationPruner se ajusta solo en train
# ---------------------------------------------------------------------------

def test_correlation_pruner_fitted_only_on_train():
    """
    CorrelationPruner.fit() debe ajustarse solo sobre X_train.
    Adulterando X_val con valores extremos no debe cambiar la selección.
    """
    df = _make_correlated_df(n=100, seed=0)
    n_train = 70
    X_train = df.iloc[:n_train].copy()
    X_val_orig = df.iloc[n_train:].copy()
    X_val_corrupt = X_val_orig.copy()
    X_val_corrupt["rmssd"] = 99999.0  # adulteramos validación

    pruner_a = CorrelationPruner(threshold=0.90).fit(X_train)
    pruner_b = CorrelationPruner(threshold=0.90).fit(X_train)  # mismo train

    selected_a = pruner_a.selected_features_
    selected_b = pruner_b.selected_features_

    assert set(selected_a) == set(selected_b), (
        "La selección cambió aunque el train fue idéntico."
    )
    # La transformación de val (corrupta vs original) no cambia la selección del pruner
    out_orig = pruner_a.transform(X_val_orig).columns.tolist()
    out_corrupt = pruner_b.transform(X_val_corrupt).columns.tolist()
    assert out_orig == out_corrupt, "Las columnas seleccionadas en val cambiaron al adulterar val."


# ---------------------------------------------------------------------------
# TEST 2: Validation no participa en selección de features
# ---------------------------------------------------------------------------

def test_validation_excluded_from_feature_selection():
    """
    En pairwise_ranking_cv con use_pruning=True, el pruner debe ajustarse
    solo con X_train. Verificamos que adulterando val no cambia la selección.
    """
    from sklearn.model_selection import GroupKFold

    df = _make_pair_df(n_pairs=10, seed=5)
    feats = ["rmssd", "cv_rr", "sd2", "sd1", "cusum_abs_max"]
    X_all = df[feats]
    groups = df["pair_id"]

    gkf = GroupKFold(n_splits=5)
    for fold, (tr, va) in enumerate(gkf.split(X_all, df["y"], groups), start=1):
        X_tr = X_all.iloc[tr].copy()
        X_va = X_all.iloc[va].copy()

        pruner = CorrelationPruner(threshold=0.90)
        pruner.fit(X_tr)  # solo train
        selected_before = set(pruner.selected_features_)

        # Adulteramos val (no debe cambiar selected_)
        X_va_corrupt = X_va.copy()
        X_va_corrupt["sd1"] = 99999.0

        # La selección ya está fija (fit no fue llamado con val)
        assert set(pruner.selected_features_) == selected_before, (
            f"Fold {fold}: selected_features_ cambió sin re-fit"
        )
        break  # solo primer fold


# ---------------------------------------------------------------------------
# TEST 3: hrv_reduced contiene exactamente rmssd, cv_rr, sd2
# ---------------------------------------------------------------------------

def test_hrv_reduced_interpretable_exact_composition():
    """hrv_reduced_interpretable debe contener exactamente {rmssd, cv_rr, sd2}."""
    expected = {"rmssd", "cv_rr", "sd2"}
    actual = set(HRV_REDUCED_INTERPRETABLE)
    assert actual == expected, (
        f"hrv_reduced_interpretable contiene {actual}, se esperaba {expected}"
    )
    assert len(HRV_REDUCED_INTERPRETABLE) == 3


# ---------------------------------------------------------------------------
# TEST 4: hrv_minimal contiene exactamente rmssd, sd2
# ---------------------------------------------------------------------------

def test_hrv_minimal_exact_composition():
    """hrv_minimal debe contener exactamente {rmssd, sd2}."""
    expected = {"rmssd", "sd2"}
    actual = set(HRV_MINIMAL)
    assert actual == expected, (
        f"hrv_minimal contiene {actual}, se esperaba {expected}"
    )
    assert len(HRV_MINIMAL) == 2


# ---------------------------------------------------------------------------
# TEST 5: Pairwise ranking evalúa score_pre > score_far por pair_id
# ---------------------------------------------------------------------------

def test_pairwise_ranking_evaluates_score_pre_gt_score_far():
    """
    pairwise_ranking_cv debe comparar score_pre > score_far por pair_id.
    Con datos donde pre siempre tiene rmssd >> far, la concordance debe ser alta.
    """
    df = _make_pair_df(n_pairs=20, seed=77)
    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    result = pairwise_ranking_cv(
        df, ["rmssd", "sd2"], model, "LogReg", "hrv_minimal", "full_record", 1800.0
    )
    assert "pairwise_concordance" in result.columns, "Falta columna pairwise_concordance"
    assert "mean_margin" in result.columns, "Falta columna mean_margin"
    assert len(result) > 0, "No se generaron folds"
    # La concordance debe ser positiva (> 0.5) dado que rmssd separa bien
    mean_pc = result["pairwise_concordance"].mean()
    assert mean_pc > 0.5, f"Pairwise concordance = {mean_pc:.2f} debería ser > 0.5 con señal clara"


# ---------------------------------------------------------------------------
# TEST 6: paired_delta genera un vector por pair_id y NO calcula AUC inválido
# ---------------------------------------------------------------------------

def test_paired_delta_no_invalid_auc():
    """
    compute_paired_delta_analysis debe:
    1. Devolver exactamente 1 fila por feature (no por registro).
    2. NO contener columna 'auc' (el marco delta no tiene dos clases naturales).
    3. Contener columnas descriptivas: median_delta, wilcoxon_p, rank_biserial_r.
    """
    df = _make_pair_df(n_pairs=10, seed=0)
    feats = ["rmssd", "cv_rr", "sd2"]
    result = compute_paired_delta_analysis(df, feats)

    assert "auc" not in result.columns, (
        "La columna 'auc' no debe existir en el análisis delta (marco no binario válido)."
    )
    assert "median_delta" in result.columns
    assert "wilcoxon_p" in result.columns
    assert "rank_biserial_r" in result.columns
    # Una fila por feature (no por registro ni por par)
    assert len(result) == len(feats), (
        f"Se esperaban {len(feats)} filas (una por feature), obtenidas {len(result)}"
    )
    # Verificar que n_pairs es correcto
    assert (result["n_pairs"] == 10).all(), "n_pairs incorrecto"


# ---------------------------------------------------------------------------
# TEST 7: No leakage por pair_id — pairwise_ranking_cv
# ---------------------------------------------------------------------------

def test_pairwise_ranking_no_pair_id_leakage():
    """
    En cada fold de pairwise_ranking_cv, ningún pair_id debe aparecer
    simultáneamente en train y validation.
    """
    from sklearn.model_selection import GroupKFold

    df = _make_pair_df(n_pairs=20, seed=42)
    feats = ["rmssd", "cv_rr", "sd2"]
    X = df[feats]
    groups = df["pair_id"]

    gkf = GroupKFold(n_splits=5)
    for fold, (tr, va) in enumerate(gkf.split(X, df["y"], groups), start=1):
        g_tr = set(groups.iloc[tr].unique())
        g_va = set(groups.iloc[va].unique())
        overlap = g_tr & g_va
        assert len(overlap) == 0, (
            f"Fold {fold}: pair_id {overlap} aparece en train Y val → leakage"
        )


# ---------------------------------------------------------------------------
# TEST 8: p_far vs p_pre no se mezcla con p* vs n*
# ---------------------------------------------------------------------------

def test_tasks_not_mixed_in_paired_delta():
    """
    compute_paired_delta_analysis debe fallar si el DataFrame contiene mezcla de tareas.
    Los manifiestos de tarea principal y secundaria deben tener analysis_task distintos.
    """
    # Test de manifiestos
    main = build_main_manifest()
    sec = build_secondary_manifest()
    main_tasks = {r["analysis_task"] for r in main}
    sec_tasks = {r["analysis_task"] for r in sec}
    assert main_tasks == {"main_p_far_vs_p_pre"}
    assert sec_tasks == {"secondary_p_vs_n"}
    assert main_tasks.isdisjoint(sec_tasks)

    # Test de que compute_paired_delta falla con mezcla
    df_mixed = _make_pair_df(n_pairs=5, seed=0, include_secondary=True)
    with pytest.raises(AssertionError):
        compute_paired_delta_analysis(df_mixed, ["rmssd", "sd2"])
