"""Reemplaza la celda de coeficientes con una que calcula en vuelo B_SPC/C/D."""
import json
from pathlib import Path

NB = Path('PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.ipynb')
nb = json.load(open(NB, encoding='utf-8'))

NEW_SOURCE = '''\
# ── Coefficient stability: B_SPC, C_HRV_SPC, D_HRV_SPC_Poincare ──────────
# Computed in-notebook from Xy_aggregated_main.csv (full_record window)
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from IPython.display import display

TABLES  = Path('artifacts/tables')
FIG_DIR = Path('artifacts/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load full-record dataset
Xy = pd.read_csv(TABLES / 'Xy_aggregated_main.csv')
full_rec = Xy[Xy.window_type == 'full_record'].copy().reset_index(drop=True)

# Feature block definitions
HRV_RED  = ['rmssd', 'cv_rr', 'sd2']
SPC_BASE = [c for c in full_rec.columns if c in [
    'shewhart_out_count','shewhart_out_rate','shewhart_max_abs_z',
    'cusum_pos_max','cusum_neg_max','cusum_abs_max',
    'cusum_last_pos','cusum_last_neg','cusum_alarm_count',
    'cusum_alarm_density','cusum_time_in_alarm',
    'ewma_last','ewma_max_abs','ewma_slope','ewma_alarm_count',
    'mr_mean','mr_median','mr_max','mr_mad','mr_out_count',
    'longest_run_above_median','longest_run_below_median',
    'trend_run_up_max','trend_run_down_max','n_runs_rule_violations',
]]
POI_FEATS = [c for c in ['sd1','sd2','sd1_sd2_ratio'] if c in full_rec.columns]

EVAL_BLOCKS = {
    'B_SPC'          : SPC_BASE,
    'C_HRV_SPC'      : HRV_RED + SPC_BASE,
    'D_HRV_SPC_Poi'  : HRV_RED + SPC_BASE + POI_FEATS,
}

def coef_stability(df, feats):
    avail = [c for c in feats if c in df.columns]
    X = df[avail]; y = df.y.astype(int); groups = df.pair_id.astype(int)
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler()),
                     ('clf', LogisticRegression(max_iter=3000, random_state=42))])
    coef_list = []
    for tr, _ in GroupKFold(min(5, groups.nunique())).split(X, y, groups):
        p = clone(pipe); p.fit(X.iloc[tr], y.iloc[tr])
        coef_list.append(p.named_steps['clf'].coef_.ravel())
    arr = np.array(coef_list)
    out = pd.DataFrame({
        'feature'           : avail,
        'mean_coef'         : arr.mean(0).round(4),
        'std_coef'          : arr.std(0).round(4),
        'pct_folds_positive': (arr > 0).mean(0).round(2),
    })
    out['sign_stable'] = (out.pct_folds_positive >= 0.80) | (out.pct_folds_positive <= 0.20)
    out['direction']   = out.mean_coef.apply(lambda c: '+' if c > 0 else '-')
    return out.sort_values('mean_coef', key=abs, ascending=False)

# Compute and display
all_tables = []
for bname, feats in EVAL_BLOCKS.items():
    tbl = coef_stability(full_rec, feats)
    stable_n = int(tbl.sign_stable.sum())
    print(f"\\n{'='*60}")
    print(f"Table: LogReg coefficient stability — {bname}")
    print(f"  n_features used  : {len(tbl)}")
    print(f"  Stable sign (>80% or <20% folds positive): {stable_n}/{len(tbl)}")
    display(tbl[['feature','mean_coef','std_coef','pct_folds_positive','sign_stable','direction']].round(4))
    all_tables.append(tbl.assign(block=bname))

# Bar chart: B_SPC
spc_t = all_tables[0].sort_values('mean_coef', ascending=True)
fig, ax = plt.subplots(figsize=(7, 6))
colors = ['#0072B2' if c > 0 else '#E69F00' for c in spc_t.mean_coef]
ax.barh(spc_t.feature, spc_t.mean_coef, color=colors,
        xerr=spc_t.std_coef, capsize=3, edgecolor='k', linewidth=0.5)
for i, row in enumerate(spc_t.itertuples()):
    if row.sign_stable:
        ax.text(row.mean_coef + (0.03 if row.mean_coef >= 0 else -0.03),
                i, '*', va='center', ha='left', color='red', fontsize=11)
ax.axvline(0, color='k', linewidth=0.8)
ax.set_xlabel('Mean coefficient (standardised space)')
ax.set_title('LogReg coefficients — B_SPC\\n(* = stable sign, >80% or <20% folds positive)')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
for ext, kw in (('png',{'dpi':600}),('tif',{'dpi':600}),('svg',{}),('pdf',{})):
    try: fig.savefig(FIG_DIR / f'logreg_coef_B_SPC.{ext}', bbox_inches='tight', **kw)
    except Exception: pass
plt.show(); plt.close(fig)

# Export combined table
if all_tables:
    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(TABLES / 'coef_stability_key_blocks.csv', index=False)
    print("\\nExported: artifacts/tables/coef_stability_key_blocks.csv")
'''

# Find the cell with id coef-stable-table
for i, c in enumerate(nb['cells']):
    if c.get('id') == 'coef-stable-table':
        nb['cells'][i]['source'] = NEW_SOURCE.splitlines(keepends=True)
        print(f'Patched cell {i} (id=coef-stable-table)')
        break
else:
    print('Cell coef-stable-table not found, appending...')
    nb['cells'].append({
        "cell_type": "code", "id": "coef-stable-table",
        "metadata": {}, "execution_count": None, "outputs": [],
        "source": NEW_SOURCE.splitlines(keepends=True)
    })

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Saved: {NB} ({len(nb["cells"])} cells)')
