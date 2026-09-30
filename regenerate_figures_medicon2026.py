"""Regenera figuras 600 dpi a partir de outputs_medicon2026/ (sin re-leer AFPDB)."""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_medicon2026"
FIG = OUT / "figures"
GKF_SPLITS = 5
DECISION_THRESHOLD = 0.5

SPC_FAST_14 = [
    "alarm_events",
    "pct_alarm_sustained",
    "max_cusum",
    "final_cusum",
    "median_rmssd",
    "median_z",
    "max_cusum_pos",
    "max_cusum_neg",
    "final_cusum_pos",
    "final_cusum_neg",
    "subwindow_sqi_good_frac",
    "rr_cv",
    "rr_masd_over_mean",
    "segment_rr_mad_ms",
]


def savefig_highres(fig, stem: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=600)
    try:
        fig.savefig(FIG / f"{stem}.tif", bbox_inches="tight", dpi=600)
    except Exception:
        pass
    plt.close(fig)


def plot_cv_metric_bars(cv_detail: pd.DataFrame, label: str):
    metrics = ["roc_auc", "pr_auc", "f1", "recall", "specificity", "false_alarm_rate", "brier"]
    models = cv_detail["model"].unique()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, metric in zip(axes.ravel(), metrics):
        sub = cv_detail.groupby("model")[metric].agg(["mean", "std"]).reindex(models)
        x = range(len(models))
        ax.bar(x, sub["mean"], yerr=sub["std"], capsize=3, color="#4c72b0", ecolor="#333")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"GroupKFold ({GKF_SPLITS}) — {label} — umbral fijo {DECISION_THRESHOLD}")
    plt.tight_layout()
    savefig_highres(fig, f"cv_metrics_{label.replace(' ', '_')}")


def main():
    os.chdir(ROOT)
    cv = pd.read_csv(OUT / "cv_summary.csv")
    spc = cv[cv["feature_set"] == "SPC_Fast_14"]
    plot_cv_metric_bars(spc, "SPC_Fast_14")
    lr = cv[(cv["model"] == "LogReg") & (cv["feature_set"].isin(["Original_6", "SPC_Fast_14"]))]
    ab = lr.pivot_table(index="fold", columns="feature_set", values="roc_auc")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for col in ab.columns:
        ax2.plot(ab.index, ab[col], marker="o", label=col)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_title("Ablation LogReg")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    savefig_highres(fig2, "ablation_logreg_roc_auc_by_fold")

    xy_path = OUT / "Xy_far_preonset_main.csv"
    if xy_path.is_file():
        Xy = pd.read_csv(xy_path)
        X = Xy[SPC_FAST_14]
        y = Xy["y"].astype(int)
        groups = Xy["pair_id"].astype(int)
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, random_state=42, solver="lbfgs")),
            ]
        )
        gkf = GroupKFold(n_splits=GKF_SPLITS)
        y_oof, p_oof = [], []
        for tr, va in gkf.split(X, y, groups=groups):
            m = clone(pipe)
            m.fit(X.iloc[tr], y.iloc[tr])
            p_oof.extend(m.predict_proba(X.iloc[va])[:, 1].tolist())
            y_oof.extend(y.iloc[va].tolist())
        y_oof = np.asarray(y_oof, dtype=int)
        p_oof = np.asarray(p_oof, dtype=float)
        fpr, tpr, _ = roc_curve(y_oof, p_oof)
        prec_c, rec_c, _ = precision_recall_curve(y_oof, p_oof)
        fig3, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].set_title("ROC (OOF pooled, LogReg SPC_Fast_14)")
        axes[0].legend(loc="lower right")
        axes[0].grid(alpha=0.3)
        axes[1].plot(rec_c, prec_c, lw=2, label=f"AP = {average_precision_score(y_oof, p_oof):.3f}")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("PR (OOF pooled)")
        axes[1].legend(loc="upper right")
        axes[1].grid(alpha=0.3)
        fig3.suptitle(f"Umbral clasificación fijo = {DECISION_THRESHOLD}")
        plt.tight_layout()
        savefig_highres(fig3, "oof_roc_pr_LogReg_SPC14")

    print("Figuras en", FIG)


if __name__ == "__main__":
    main()
