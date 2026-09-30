"""
Limpia el notebook paper-ready:
  - Elimina referencias al pipeline viejo (outputs_medicon2026, ablation_summary, etc.)
  - Añade tabla de coeficientes estables para B_SPC, C_HRV_SPC, D_HRV_SPC_Poincare
  - Corrige setup para no definir OUT2026
"""
import json, copy, re
from pathlib import Path

ROOT = Path('.')
NB   = ROOT / 'PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.ipynb'
OUT  = NB   # overwrite in place

nb = json.load(open(NB, encoding='utf-8'))

def src(cell):
    s = cell.get('source', [])
    return ''.join(s) if isinstance(s, list) else s

def set_src(cell, text):
    cell['source'] = text.splitlines(keepends=True)

# ─── CELL 1 (setup): remove OUT2026 lines ────────────────────────────────
c1 = nb['cells'][1]
lines = src(c1).splitlines(keepends=True)
new_lines = [l for l in lines
             if 'OUT2026' not in l and 'outputs_medicon2026' not in l]
set_src(c1, ''.join(new_lines))
print('Cell 1 cleaned.')

# ─── CELL 13 (block comparison): replace entire cell source ───────────────
NEW_CELL13 = '''\
# ── Feature block comparison — results from artifacts/tables/ ─────────────
import pandas as pd, numpy as np
from pathlib import Path
from IPython.display import display

TABLES = Path('artifacts/tables')

# Baseline AUC results (v1 — GroupKFold by pair_id)
bl_csv = TABLES / 'v1_interpretable_baseline_results.csv'
if bl_csv.is_file():
    bl_df = pd.read_csv(bl_csv)
    print('Baseline AUC results (full_record window, LogReg):')
    display(bl_df.round(4))
else:
    print(f'{bl_csv.name} not found.')

# Bootstrap 95% CI for AUC
ci_csv = TABLES / 'bootstrap_ci_v1_results.csv'
if ci_csv.is_file():
    ci_df = pd.read_csv(ci_csv)
    auc_ci = ci_df[ci_df.metric == 'auc'][['config','observed','ci_low','ci_high']]
    print('\\nAUC 95% CI (bootstrap by pair_id):')
    display(auc_ci.round(4))

# Incremental contribution
inc_csv = TABLES / 'incremental_blocks_main.csv'
if inc_csv.is_file():
    inc_df = pd.read_csv(inc_csv)
    fr_inc = inc_df[inc_df.window_type == 'full_record'] if 'window_type' in inc_df.columns else inc_df
    if not fr_inc.empty:
        print('\\nIncremental AUC by block (full_record):')
        display(fr_inc.round(4))

# Paper-ready results summary
pr_csv = TABLES / 'paper_ready_results.csv'
if pr_csv.is_file():
    pr_df = pd.read_csv(pr_csv)
    print('\\nPaper-ready results summary:')
    display(pr_df.round(4))
'''

set_src(nb['cells'][13], NEW_CELL13)
print('Cell 13 replaced.')

# ─── CELL 21 (feature importance): replace entire cell source ─────────────
NEW_CELL21 = '''\
# ── Feature importance and coefficient stability (from artifacts/tables/) ─
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TABLES  = Path('artifacts/tables')
FIG_DIR = Path('artifacts/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

coef_csv = TABLES / 'logreg_coef_stability.csv'
if coef_csv.is_file():
    coef_df = pd.read_csv(coef_csv)
    key_blocks = ['B_SPC', 'C_HRV_SPC', 'D_HRV_SPC_Poincare']

    for bname in key_blocks:
        sub = coef_df[coef_df.block == bname].sort_values('mean_coef', key=abs, ascending=False)
        if sub.empty:
            continue
        stable = int(sub.sign_stable.sum())
        print(f"\\n--- {bname}: {stable}/{len(sub)} features with stable sign (>80% or <20% folds positive) ---")
        display(sub[['feature','mean_coef','std_coef','pct_folds_positive','sign_stable']].head(10).round(4))

    # Figure: coefficient stability for B_SPC
    spc_coef = coef_df[coef_df.block == 'B_SPC'].sort_values('mean_coef', ascending=True)
    if not spc_coef.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ['#0072B2' if c > 0 else '#E69F00' for c in spc_coef.mean_coef]
        ax.barh(spc_coef.feature, spc_coef.mean_coef, color=colors,
                xerr=spc_coef.std_coef, capsize=3, edgecolor='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.8)
        stable_sym = spc_coef.sign_stable.map({True: ' *', False: ''})
        for i, (_, row) in enumerate(spc_coef.iterrows()):
            ax.text(0, i, ' *' if row.sign_stable else '', va='center', color='red', fontsize=9)
        ax.set_xlabel('Mean coefficient (standardised space)')
        ax.set_title('LogReg coefficients — B_SPC block\\n(* = stable sign, >80% or <20% folds positive)')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        for ext, kw in (('png', {'dpi':600}), ('tif', {'dpi':600}), ('svg', {}), ('pdf', {})):
            try: fig.savefig(FIG_DIR / f'logreg_coef_B_SPC.{ext}', bbox_inches='tight', **kw)
            except Exception: pass
        plt.show(); plt.close(fig)
else:
    print(f'{coef_csv.name} not found — run run_interpretability_audit.py first.')
'''

set_src(nb['cells'][21], NEW_CELL21)
print('Cell 21 replaced.')

# ─── Remove any remaining bad terms (text replacement in ALL cells) ────────
BAD_TERMS = [
    ('from older pipeline', 'from the corrected pipeline'),
    ('old pipeline',        'previous pipeline version (corrected)'),
    ('n+nc pairs',          'p01-p50 pairs (far_baseline / pre_onset)'),
    ('SPC_Fast-14',         'B_SPC block'),
    ('SPC_Fast_14',         'B_SPC block'),
    ('SPC_Fast14',          'B_SPC block'),
    ('Original-6',          'HRV_reduced block'),
    ('Original_6',          'HRV_reduced block'),
]
for old, new in BAD_TERMS:
    for c in nb['cells']:
        s = src(c)
        if old in s:
            set_src(c, s.replace(old, new))
            print(f'  Replaced "{old}" with "{new}"')

# ─── Insert stable coefficient table cell (after cell 21) ─────────────────
NEW_COEF_TABLE_CELL = {
    "cell_type": "code",
    "id": "coef-stable-table",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": '''\
# ── Stable coefficient table: B_SPC, C_HRV_SPC, D_HRV_SPC_Poincare ────────
import pandas as pd
from pathlib import Path

coef_df = pd.read_csv(Path('artifacts/tables/logreg_coef_stability.csv'))

key_blocks = ['B_SPC', 'C_HRV_SPC', 'D_HRV_SPC_Poincare']
tables = {}
for bname in key_blocks:
    sub = coef_df[coef_df.block == bname].sort_values('mean_coef', key=abs, ascending=False)
    if sub.empty:
        continue
    sub = sub[['feature','mean_coef','std_coef','pct_folds_positive','sign_stable']].copy()
    sub['direction'] = sub.mean_coef.apply(lambda c: '+' if c > 0 else '-')
    sub['interpretation'] = sub.sign_stable.map(
        {True: 'Stable: consistent direction across folds',
         False: 'Unstable: sign varies across folds'})
    tables[bname] = sub
    print(f"\\n{'='*60}")
    print(f"Table: LogReg coefficient stability — {bname}")
    print(f"  Features with stable sign: {sub.sign_stable.sum()}/{len(sub)}")
    print(f"  Threshold: GroupKFold(5) by pair_id; sign stable if >80% or <20% folds positive")
    display(sub.round(4))

# Export to CSV
combined = pd.concat([t.assign(block=b) for b, t in tables.items()], ignore_index=True)
combined.to_csv(Path('artifacts/tables/coef_stability_key_blocks.csv'), index=False)
print("\\nExported: artifacts/tables/coef_stability_key_blocks.csv")
'''.splitlines(keepends=True)
}

# Insert after cell 21 (index 21)
nb['cells'].insert(22, NEW_COEF_TABLE_CELL)
print('New coefficient table cell inserted at position 22.')

# ─── Verify no bad terms remain ────────────────────────────────────────────
still_bad = ['ablation_summary','interpretability_top15','n+nc','SPC_Fast','Original_6']
issues = []
for i, c in enumerate(nb['cells']):
    s = src(c)
    hits = [t for t in still_bad if t in s]
    if hits:
        issues.append(f'Cell {i}: {hits}')
if issues:
    print('\nRemaining issues:')
    for iss in issues:
        print(f'  {iss}')
else:
    print('\nAudit passed — no bad references remaining.')

# ─── Save ──────────────────────────────────────────────────────────────────
out_txt = json.dumps(nb, indent=1, ensure_ascii=False)
NB.write_text(out_txt, encoding='utf-8')
print(f'\nSaved: {NB}  (cells: {len(nb["cells"])})')
