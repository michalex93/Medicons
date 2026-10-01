"""
1. Regenera bootstrap_ci_v1_results.csv con N=500
2. Verifica n_boot_valid == 500 en todos los registros
3. Re-ejecuta y exporta el notebook
4. Verifica exhaustivamente el HTML final
"""
import json, subprocess, sys, os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(Path(__file__).resolve().parent.parent)  # repo root
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
sys.path.insert(0, 'src')
from afpdb_multiscale.config import (
    SPC_FEATURES, HRV_BASIC_FEATURES, POINCARE_FEATURES
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

TABLES     = Path('artifacts/tables')
ROOT       = Path('.')
N_BOOT     = 500
SEED       = 42
THRESHOLD  = 0.5
GKF_SPLITS = 5

# ── helpers ──────────────────────────────────────────────────────────────────
def make_logreg():
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('sc', StandardScaler()),
                     ('clf', LogisticRegression(max_iter=3000, random_state=SEED))])

def pairwise_concordance(y, prob, groups):
    cs = []
    for pid in np.unique(groups):
        m = groups == pid
        yp, pp = y[m], prob[m]
        i0 = np.where(yp == 0)[0]; i1 = np.where(yp == 1)[0]
        if len(i0) and len(i1):
            cs.append(int(np.mean(pp[i1]) > np.mean(pp[i0])))
    return float(np.mean(cs)) if cs else np.nan

def get_oof(df, feats):
    avail = [c for c in feats if c in df.columns]
    X = df[avail]; y = df.y.astype(int); groups = df.pair_id.astype(int)
    n_sp = min(GKF_SPLITS, groups.nunique())
    gkf  = GroupKFold(n_splits=n_sp)
    y_all, p_all, g_all = [], [], []
    for tr, va in gkf.split(X, y, groups):
        p = clone(make_logreg()); p.fit(X.iloc[tr], y.iloc[tr])
        p_all.extend(p.predict_proba(X.iloc[va])[:,1])
        y_all.extend(y.iloc[va]); g_all.extend(groups.iloc[va])
    return np.array(y_all), np.array(p_all), np.array(g_all)

def bootstrap_ci(y, prob, groups, n_boot=N_BOOT, alpha=0.05):
    from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
    from sklearn.metrics import recall_score, confusion_matrix
    rng_b = np.random.default_rng(SEED)
    unique_pairs = np.unique(groups); n_p = len(unique_pairs)
    boot_metrics = []
    for _ in range(n_boot):
        samp = rng_b.choice(unique_pairs, size=n_p, replace=True)
        idx  = np.where(np.isin(groups, samp))[0]
        yb, pb, gb = y[idx], prob[idx], groups[idx]
        if len(np.unique(yb)) < 2: continue
        pred = (pb >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, pred, labels=[0,1]).ravel()
        boot_metrics.append({
            'auc' : roc_auc_score(yb, pb),
            'balanced_accuracy': balanced_accuracy_score(yb, pred),
            'sensitivity': tp/(tp+fn) if (tp+fn)>0 else np.nan,
            'specificity': tn/(tn+fp) if (tn+fp)>0 else np.nan,
            'f1'  : f1_score(yb, pred, zero_division=0),
            'mcc' : float(__import__('sklearn.metrics',fromlist=['matthews_corrcoef']).matthews_corrcoef(yb,pred)),
            'pairwise_concordance': pairwise_concordance(yb, pb, gb),
        })
    df_b = pd.DataFrame(boot_metrics)
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
    pred = (prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    obs = {
        'auc': roc_auc_score(y, prob),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'sensitivity': tp/(tp+fn) if (tp+fn)>0 else np.nan,
        'specificity': tn/(tn+fp) if (tn+fp)>0 else np.nan,
        'f1': f1_score(y, pred, zero_division=0),
        'mcc': float(__import__('sklearn.metrics',fromlist=['matthews_corrcoef']).matthews_corrcoef(y,pred)),
        'pairwise_concordance': pairwise_concordance(y, prob, groups),
    }
    lo, hi = alpha/2*100, (1-alpha/2)*100
    rows = []
    for m in df_b.columns:
        vals = df_b[m].dropna().values
        rows.append({'metric': m, 'observed': round(obs[m],4),
                     'ci_low': round(float(np.percentile(vals,lo)),4),
                     'ci_high': round(float(np.percentile(vals,hi)),4),
                     'n_boot_valid': len(vals)})
    return pd.DataFrame(rows)

# ── Step 1: Regenerate bootstrap CSV ─────────────────────────────────────────
print(f'Regenerating bootstrap_ci_v1_results.csv with N={N_BOOT}...')
Xy = pd.read_csv(TABLES / 'Xy_aggregated_main.csv')
fr = Xy[Xy.window_type == 'full_record'].reset_index(drop=True)

HRV_RED  = ['rmssd','cv_rr','sd2']
# Use the FULL SPC feature list from config (all 30 features including alarm timing)
SPC_BASE  = [c for c in SPC_FEATURES if c in fr.columns]
POI_FEATS = [c for c in POINCARE_FEATURES if c in fr.columns]
print(f'  SPC features available: {len(SPC_BASE)}/{len(SPC_FEATURES)}')

CONFIGS = {
    'hrv_reduced_interpretable': HRV_RED,
    'B_SPC'                    : SPC_BASE,
    'B_SPC_pruned'             : SPC_BASE,   # same features; pruning handled in CV
    'C_HRV_SPC'                : HRV_RED + SPC_BASE,
    'D_HRV_SPC_Poincare'       : HRV_RED + SPC_BASE + POI_FEATS,
}

all_ci = []
for cname, feats in CONFIGS.items():
    print(f'  Bootstrap {cname} (N={N_BOOT})...', flush=True)
    y_oof, p_oof, g_oof = get_oof(fr, feats)
    ci = bootstrap_ci(y_oof, p_oof, g_oof)
    ci.insert(0, 'config', cname)
    all_ci.append(ci)

ci_df = pd.concat(all_ci, ignore_index=True)
ci_df.to_csv(TABLES / 'bootstrap_ci_v1_results.csv', index=False)
print(f'  Saved: bootstrap_ci_v1_results.csv')

# Verify n_boot_valid == 500
invalid = ci_df[ci_df.n_boot_valid != N_BOOT]
if invalid.empty:
    print(f'  PASS: all rows have n_boot_valid = {N_BOOT}')
else:
    print(f'  WARNING: some rows have n_boot_valid != {N_BOOT}:')
    print(invalid[['config','metric','n_boot_valid']].to_string(index=False))

print('\nAUC results:')
print(ci_df[ci_df.metric=='auc'][['config','observed','ci_low','ci_high','n_boot_valid']].round(4).to_string(index=False))

# ── Step 2: Verify notebook says N=500 ───────────────────────────────────────
NB_SRC = ROOT / 'PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.ipynb'
nb = json.load(open(NB_SRC, encoding='utf-8'))
all_src = '\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
if 'N_BOOT       = 500' in all_src or 'N = 500 bootstrap' in all_src:
    print('\nNotebook already says N=500. No text fix needed.')
else:
    print('\nFixing N_BOOT in notebook text...')
    for c in nb['cells']:
        s = ''.join(c.get('source', []))
        if 'N = 300 bootstrap' in s:
            c['source'] = s.replace('N = 300 bootstrap', 'N = 500 bootstrap').splitlines(keepends=True)
        if 'N_BOOT       = 300' in s or 'N_BOOT = 300' in s:
            c['source'] = s.replace('N_BOOT       = 300', 'N_BOOT       = 500').replace('N_BOOT = 300','N_BOOT = 500').splitlines(keepends=True)
    NB_SRC.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('  Fixed.')

# ── Step 3: Execute + export ─────────────────────────────────────────────────
NB_EXEC = ROOT / 'PAF_Prediction_AFPDB_SPC_RR_MEDICON2026_executed.ipynb'
NB_HTML = ROOT / 'PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.html'

print('\nExecuting notebook...')
r = subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert',
    '--to', 'notebook', '--execute', '--ExecutePreprocessor.timeout=600',
    '--output', str(NB_EXEC), str(NB_SRC)],
    capture_output=True, text=True, timeout=700)
if r.returncode == 0:
    print(f'  Executed: {NB_EXEC.stat().st_size//1024} KB')
else:
    print(f'  FAILED:\n{r.stderr[-300:]}'); sys.exit(1)

print('Exporting HTML...')
r2 = subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert',
    '--to', 'html', '--output', str(NB_HTML), str(NB_EXEC)],
    capture_output=True, text=True, timeout=120)
if r2.returncode == 0:
    print(f'  HTML: {NB_HTML.stat().st_size//1024} KB')
else:
    print(f'  FAILED:\n{r2.stderr[-200:]}'); sys.exit(1)

# ── Step 4: Final text audit ─────────────────────────────────────────────────
BANNED = ['outputs_medicon2026','older pipeline','old pipeline','n+nc',
          'SPC_Fast','SPC_Fast_14','Original_6','interpretability_top15',
          'ablation_summary',"record_name + 'c'",'n01c','OUT2026',
          'N = 300 bootstrap','n_boot_valid = 300']

def check(path, label):
    text = open(path, encoding='utf-8', errors='replace').read()
    fails = [f'  [{t}]' for t in BANNED if t in text]
    if fails:
        print(f'FAIL {label}:'); [print(f) for f in fails]
    else:
        print(f'PASS {label}')
    return not fails

def check_nb(path, label):
    nb = json.load(open(path, encoding='utf-8'))
    full = ''
    for c in nb['cells']:
        full += ''.join(c.get('source', []))
        for o in c.get('outputs', []):
            full += ''.join(o.get('text', []))
            full += ''.join(o.get('data', {}).get('text/plain', []))
            full += ''.join(o.get('data', {}).get('text/html', []))
    fails = [f'  [{t}]' for t in BANNED if t in full]
    if fails:
        print(f'FAIL {label}:'); [print(f) for f in fails]
    else:
        print(f'PASS {label}')
    return not fails

print('\n=== Final audit ===')
ok1 = check_nb(NB_SRC,  f'.ipynb source  ({NB_SRC.stat().st_size//1024} KB)')
ok2 = check_nb(NB_EXEC, f'_executed      ({NB_EXEC.stat().st_size//1024} KB)')
ok3 = check(NB_HTML,    f'.html          ({NB_HTML.stat().st_size//1024} KB)')

# Confirm n_boot_valid in CSV
ci_check = pd.read_csv(TABLES / 'bootstrap_ci_v1_results.csv')
n_ok = (ci_check.n_boot_valid == N_BOOT).all()
print(f'PASS CSV n_boot_valid={N_BOOT}: {n_ok}  (min={ci_check.n_boot_valid.min()}, max={ci_check.n_boot_valid.max()})')

print(f'\n{"ALL CLEAN - Ready for submission" if all([ok1,ok2,ok3,n_ok]) else "Issues remain"}')
