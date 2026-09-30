"""
Bloques de features HRV con control de colinealidad.

Bloques definidos:
  hrv_full_actual          — todas las features HRV extraídas (con redundancia)
  hrv_reduced_interpretable — rmssd, cv_rr, sd2  (una por grupo colineal)
  hrv_minimal              — rmssd, sd2           (mínimo interpretable)

CorrelationPruner:
  Elimina features colineales DENTRO de cada train fold.
  NUNCA usa datos de validación para decidir qué features eliminar.
  Prioridad declarativa: la feature más interpretable/citada en literatura se conserva.

Regla de prioridad para colinealidad HRV:
  rmssd > sd1 > sdsd > rr_diff_std > rr_diff_mean
  cv_rr > std_rr
  sd2  (retener, no colineal con rmssd/sd1)
  cusum_abs_max > cusum_pos_max, cusum_neg_max (si se incluyen)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from .config import (
    HRV_BASIC_FEATURES,
    POINCARE_FEATURES,
    SPC_FEATURES,
    ECTOPY_FEATURES,
)

# ---------------------------------------------------------------------------
# Definición de bloques HRV
# ---------------------------------------------------------------------------

HRV_FULL_ACTUAL: list[str] = HRV_BASIC_FEATURES + POINCARE_FEATURES

HRV_REDUCED_INTERPRETABLE: list[str] = ["rmssd", "cv_rr", "sd2"]

HRV_MINIMAL: list[str] = ["rmssd", "sd2"]

# Bloques de evaluación extendidos
INTERPRETABLE_BLOCKS: dict[str, list[str]] = {
    "hrv_full_actual": HRV_FULL_ACTUAL,
    "hrv_reduced_interpretable": HRV_REDUCED_INTERPRETABLE,
    "hrv_minimal": HRV_MINIMAL,
    "B_SPC": SPC_FEATURES,
    "C_HRV_SPC": HRV_REDUCED_INTERPRETABLE + SPC_FEATURES,
    "D_HRV_SPC_Poincare": HRV_REDUCED_INTERPRETABLE + SPC_FEATURES + POINCARE_FEATURES,
}

# ---------------------------------------------------------------------------
# Orden de prioridad para poda por colinealidad
# (primero = más interpretable / más citado en literatura HRV)
# ---------------------------------------------------------------------------

PRIORITY_ORDER: list[str] = [
    # HRV dominante
    "rmssd",        # más citado, gold standard HRV corto plazo
    "cv_rr",        # escala-invariante, no correlacionado con rmssd
    "sd2",          # variabilidad largo plazo, ortogonal a sd1
    "sd1",          # = rmssd/sqrt(2) → redundante con rmssd
    "sdsd",         # = rmssd → estrictamente redundante
    "mad_rr",       # dispersión robusta de RR
    "mean_rr",      # HR basal
    "median_rr",    # correlacionado con mean_rr
    "rr_diff_mean", # = sdsd/media → derivado de sdsd
    "rr_diff_std",  # = sdsd → estrictamente redundante
    "std_rr",       # = sd_rr → correlacionado con cv_rr y sd2
    "sd1_sd2_ratio",
    # SPC
    "cusum_abs_max",
    "cusum_pos_max",
    "cusum_neg_max",
    "cusum_last_pos",
    "cusum_last_neg",
    "ewma_slope",
    "ewma_last",
    "ewma_max_abs",
    "shewhart_out_rate",
    "shewhart_out_count",
    "shewhart_max_abs_z",
    "mr_mad",
    "mr_median",
    "mr_mean",
    "mr_max",
    "mr_out_count",
    "longest_run_above_median",
    "longest_run_below_median",
    "trend_run_up_max",
    "trend_run_down_max",
    "n_runs_rule_violations",
]

_PRIORITY_INDEX = {f: i for i, f in enumerate(PRIORITY_ORDER)}


def _priority(feature: str) -> int:
    return _PRIORITY_INDEX.get(feature, len(PRIORITY_ORDER))


# ---------------------------------------------------------------------------
# Análisis de colinealidad (solo descriptivo, no para selección global)
# ---------------------------------------------------------------------------

def compute_collinearity_report(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.90,
    method: str = "spearman",
) -> dict:
    """
    Calcula matriz de correlación y detecta grupos colineales.
    USO DESCRIPTIVO ÚNICAMENTE — no usar para selección global antes de CV.

    Devuelve dict con:
      corr_matrix  : pd.DataFrame
      high_corr_pairs : lista de (feat_a, feat_b, r)
      collinear_groups : lista de listas (grupos con |r| >= threshold)
    """
    avail = [c for c in feature_cols if c in df.columns]
    X = df[avail].dropna(how="all")
    corr = X.corr(method=method)

    pairs = []
    for i, fa in enumerate(avail):
        for j, fb in enumerate(avail):
            if j <= i:
                continue
            try:
                r_val = corr.loc[fa, fb]
                r = float(r_val.iloc[0]) if hasattr(r_val, "iloc") else float(r_val)
            except Exception:
                r = np.nan
            if np.isfinite(r) and abs(r) >= threshold:
                pairs.append((fa, fb, round(r, 4)))

    # Componentes conexos
    parent = {f: f for f in avail}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for a, b, _ in pairs:
        union(a, b)

    components: dict[str, list[str]] = defaultdict(list)
    for f in avail:
        components[find(f)].append(f)

    groups = [sorted(v, key=_priority) for v in components.values() if len(v) > 1]

    return {
        "corr_matrix": corr,
        "high_corr_pairs": pairs,
        "collinear_groups": groups,
        "threshold": threshold,
        "method": method,
    }


# ---------------------------------------------------------------------------
# CorrelationPruner — poda SOLO dentro del fold
# ---------------------------------------------------------------------------

class CorrelationPruner:
    """
    Selecciona, DENTRO de cada fold, las features a conservar eliminando
    aquellas con |r| >= threshold respecto a una feature de mayor prioridad.

    Garantías anti-leakage:
      - fit()      : solo recibe X_train
      - transform(): aplica la selección aprendida de train a cualquier split
      - NUNCA fit() sobre datos de validación

    Uso en CV:
        pruner = CorrelationPruner(threshold=0.90)
        pruner.fit(X_train)
        X_train_pruned = pruner.transform(X_train)
        X_val_pruned   = pruner.transform(X_val)   # mismo set de features
    """

    def __init__(
        self,
        threshold: float = 0.90,
        priority_order: Sequence[str] | None = None,
        method: str = "spearman",
    ):
        self.threshold = threshold
        self.priority_order = list(priority_order) if priority_order else PRIORITY_ORDER
        self.method = method
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, y=None) -> "CorrelationPruner":
        """Ajusta sobre X_train únicamente."""
        cols = list(X.columns)
        report = compute_collinearity_report(
            X, cols, threshold=self.threshold, method=self.method
        )

        to_drop: set[str] = set()
        for group in report["collinear_groups"]:
            # Ordenar por prioridad; conservar el primero, eliminar el resto
            ordered = sorted(group, key=_priority)
            to_drop.update(ordered[1:])  # drop todos menos el de mayor prioridad

        self.selected_features_ = [c for c in cols if c not in to_drop]
        self.dropped_features_ = list(to_drop)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("CorrelationPruner no fue ajustado (llama fit primero).")
        avail = [c for c in self.selected_features_ if c in X.columns]
        return X[avail]

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def n_selected(self) -> int:
        return len(self.selected_features_)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_features_)
