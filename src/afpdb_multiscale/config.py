"""
Constantes globales del pipeline multi-escala AFPDB/PAF RR-SPC.
Ningún hiperparámetro de SPC ni de modelos debe ajustarse fuera de train fold.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # .../Medicons
AFPDB_DIR = ROOT / "afpdb"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
TABLES_DIR = ARTIFACTS_DIR / "tables"

# ---------------------------------------------------------------------------
# SPC hyperparameters (valores iniciales; solo tunable dentro de train fold)
# ---------------------------------------------------------------------------
CUSUM_H: float = 7.0       # umbral de alarma
CUSUM_K: float = 0.5       # drift
EWMA_LAMBDA: float = 0.2   # factor de suavizado
EWMA_L: float = 2.7        # límite de control EWMA (en σ efectivas)
SHEWHART_L: float = 3.0    # límite Shewhart (3σ)
MR_D4: float = 3.267       # factor UCL para carta de rango móvil (n=2)
CUSUM_SUSTAINED: int = 2   # ventanas consecutivas para alarma sostenida

# ---------------------------------------------------------------------------
# Baseline por registro para SPC
# ---------------------------------------------------------------------------
BASELINE_N_RR: int = 200   # primeros N intervalos RR para calcular μ/σ de baseline

# ---------------------------------------------------------------------------
# Tamaños de ventana
# ---------------------------------------------------------------------------
RR_COUNT_WINDOWS: list[int] = [50, 100, 128, 200]
TIME_WINDOWS_SEC: list[float] = [300.0]          # 5 minutos
FULL_RECORD_SEC: float = 1800.0                  # 30 minutos
PROHIBITED_WINDOW_SEC: int = 30                  # PROHIBIDO en análisis principal

# ---------------------------------------------------------------------------
# Umbrales ectopia inferida desde RR (sin morfología)
# ---------------------------------------------------------------------------
ECTOPY_SHORT_THRESH: float = 0.75  # RR < 0.75 * mediana → "corto"
ECTOPY_LONG_THRESH: float = 1.25   # RR > 1.25 * mediana → "largo"
ATRIAL_TACHY_MIN_RUN: int = 3      # mínimo de RR cortos consecutivos

# ---------------------------------------------------------------------------
# Validación y modelos
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
GKF_N_SPLITS: int = 5
DECISION_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Definición de features
# ---------------------------------------------------------------------------
HRV_BASIC_FEATURES: list[str] = [
    "mean_rr", "median_rr", "std_rr", "mad_rr",
    "rmssd", "sdsd", "cv_rr", "rr_diff_mean", "rr_diff_std",
]
POINCARE_FEATURES: list[str] = ["sd1", "sd2", "sd1_sd2_ratio"]
SHEWHART_FEATURES: list[str] = [
    "shewhart_out_count", "shewhart_out_rate",
    "shewhart_max_abs_z", "shewhart_first_alarm_pos",
    "shewhart_first_alarm_time_sec",
]
CUSUM_FEATURES: list[str] = [
    "cusum_pos_max", "cusum_neg_max", "cusum_abs_max",
    "cusum_last_pos", "cusum_last_neg",
    "cusum_alarm_count", "cusum_first_alarm_pos",
    "cusum_first_alarm_time_sec", "cusum_time_in_alarm",
    "cusum_alarm_density",
]
EWMA_FEATURES: list[str] = [
    "ewma_last", "ewma_max_abs", "ewma_slope",
    "ewma_alarm_count", "ewma_first_alarm_time_sec",
]
MR_FEATURES: list[str] = [
    "mr_mean", "mr_median", "mr_max", "mr_mad", "mr_out_count",
]
RUNS_FEATURES: list[str] = [
    "longest_run_above_median", "longest_run_below_median",
    "trend_run_up_max", "trend_run_down_max",
    "n_runs_rule_violations",
]
SPC_FEATURES: list[str] = (
    SHEWHART_FEATURES + CUSUM_FEATURES + EWMA_FEATURES + MR_FEATURES + RUNS_FEATURES
)
ECTOPY_FEATURES: list[str] = [
    "ectopy_like_count", "ectopy_like_density",
    "short_long_sequence_count", "compensatory_pause_like_count",
    "bigeminy_like_pattern_count", "trigeminy_like_pattern_count",
    "atrial_tachy_like_run_count",
]

ALL_FEATURES: list[str] = (
    HRV_BASIC_FEATURES + POINCARE_FEATURES + SPC_FEATURES + ECTOPY_FEATURES
)

# Bloques de evaluación
BLOCKS: dict[str, list[str]] = {
    "A_HRV": HRV_BASIC_FEATURES,
    "B_SPC": SPC_FEATURES,
    "C_HRV_SPC": HRV_BASIC_FEATURES + SPC_FEATURES,
    "D_HRV_SPC_Poincare": HRV_BASIC_FEATURES + SPC_FEATURES + POINCARE_FEATURES,
    "E_HRV_SPC_Poincare_Ectopy": (
        HRV_BASIC_FEATURES + SPC_FEATURES + POINCARE_FEATURES + ECTOPY_FEATURES
    ),
}

# Campos de provenance / metadata por fila
METADATA_FIELDS: list[str] = [
    "record_id", "pair_id", "y",
    "analysis_task", "window_type", "window_size",
    "window_index", "start_time_sec", "end_time_sec", "n_rr",
]
