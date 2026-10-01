"""
1. Corrige primera celda markdown (etiquetas y=0/y=1)
2. Re-ejecuta y exporta a HTML
3. Verifica por búsqueda textual exhaustiva
"""
import json, subprocess, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('.')
NB_SRC  = ROOT / 'PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.ipynb'
NB_EXEC = ROOT / 'PAF_Prediction_AFPDB_SPC_RR_MEDICON2026_executed.ipynb'
NB_HTML = ROOT / 'PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.html'

# ── 1. Fix first markdown cell ──────────────────────────────────────────────
nb = json.load(open(NB_SRC, encoding='utf-8'))
src0 = ''.join(nb['cells'][0].get('source', []))

OLD_LABEL = '- `y=0` → `far_baseline` (not "far_baseline", not "far_baseline")  \n- `y=1` → `pre_onset`'
NEW_LABEL  = '- `y=0` → **far_baseline** (not "normal", not "healthy", not "control")\n- `y=1` → **pre_onset**'

if OLD_LABEL in src0:
    src0 = src0.replace(OLD_LABEL, NEW_LABEL)
    nb['cells'][0]['source'] = src0.splitlines(keepends=True)
    print('Fixed: y=0/y=1 label note corrected.')
else:
    # Try a broader replacement
    import re
    pattern = r'- `y=0` → `far_baseline`[^\n]*\n- `y=1` → `pre_onset`'
    replacement = '- `y=0` → **far_baseline** (not "normal", not "healthy", not "control")\n- `y=1` → **pre_onset**'
    new_src = re.sub(pattern, replacement, src0)
    if new_src != src0:
        nb['cells'][0]['source'] = new_src.splitlines(keepends=True)
        print('Fixed (regex): y=0/y=1 label note corrected.')
    else:
        print('WARNING: label pattern not found — check manually.')
        # Show current label lines
        for line in src0.splitlines():
            if 'y=0' in line or 'y=1' in line:
                print(f'  Current: {repr(line)}')

NB_SRC.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Saved: {NB_SRC.name}')

# ── 2. Execute notebook ─────────────────────────────────────────────────────
print('\nExecuting notebook...')
r = subprocess.run([
    sys.executable, '-m', 'jupyter', 'nbconvert',
    '--to', 'notebook',
    '--execute',
    '--ExecutePreprocessor.timeout=600',
    '--output', str(NB_EXEC),
    str(NB_SRC),
], capture_output=True, text=True, timeout=700)

if r.returncode == 0:
    print(f'  Executed: {NB_EXEC.name}  ({NB_EXEC.stat().st_size//1024} KB)')
else:
    print(f'  EXECUTE FAILED:\n{r.stderr[-500:]}')
    sys.exit(1)

# ── 3. Export HTML ───────────────────────────────────────────────────────────
print('\nExporting HTML...')
r2 = subprocess.run([
    sys.executable, '-m', 'jupyter', 'nbconvert',
    '--to', 'html',
    '--output', str(NB_HTML),
    str(NB_EXEC),
], capture_output=True, text=True, timeout=120)

if r2.returncode == 0:
    print(f'  HTML: {NB_HTML.name}  ({NB_HTML.stat().st_size//1024} KB)')
else:
    print(f'  HTML FAILED:\n{r2.stderr[-300:]}')
    sys.exit(1)

# ── 4. Exhaustive text search in .ipynb + _executed.ipynb + .html ───────────
BANNED = [
    'outputs_medicon2026',
    'older pipeline',
    'old pipeline',
    'n+nc',
    'SPC_Fast',
    'SPC_Fast_14',
    'Original_6',
    'interpretability_top15',
    'ablation_summary',
    "record_name + 'c'",
    "record_name + \"c\"",
    'n01c',
    'OUT2026',
]

def check_file(path, label):
    text = open(path, encoding='utf-8', errors='replace').read()
    fails = []
    for term in BANNED:
        if term in text:
            idx = text.find(term)
            ctx = text[max(0,idx-40):idx+80].replace('\n',' ')
            fails.append(f'  FOUND [{term}]: ...{ctx}...')
    if fails:
        print(f'\nFAIL {label}:')
        for f in fails: print(f)
    else:
        print(f'PASS {label}')
    return len(fails) == 0

def check_nb_deep(path, label):
    """Check .ipynb source + outputs."""
    nb = json.load(open(path, encoding='utf-8'))
    fails = []
    for i, c in enumerate(nb['cells']):
        full = ''.join(c.get('source', []))
        for o in c.get('outputs', []):
            full += ''.join(o.get('text', []))
            full += ''.join(o.get('data', {}).get('text/plain', []))
            full += ''.join(o.get('data', {}).get('text/html', []))
        for term in BANNED:
            if term in full:
                ctx = full[full.find(term)-30:full.find(term)+70].replace('\n',' ')
                fails.append(f'  Cell {i:2d}: [{term}] ...{ctx}...')
    if fails:
        print(f'\nFAIL {label}:')
        for f in fails: print(f)
    else:
        print(f'PASS {label}')
    return len(fails) == 0

print('\n=== Exhaustive text audit ===')
ok1 = check_nb_deep(NB_SRC,  f'.ipynb source        ({NB_SRC.stat().st_size//1024} KB)')
ok2 = check_nb_deep(NB_EXEC, f'_executed.ipynb     ({NB_EXEC.stat().st_size//1024} KB)')
ok3 = check_file(NB_HTML,    f'.html               ({NB_HTML.stat().st_size//1024} KB)')

print(f'\n=== Summary ===')
print(f'  .ipynb source    : {"CLEAN" if ok1 else "HAS ISSUES"}')
print(f'  _executed.ipynb  : {"CLEAN" if ok2 else "HAS ISSUES"}')
print(f'  .html            : {"CLEAN" if ok3 else "HAS ISSUES"}')

if ok1 and ok2 and ok3:
    print('\nAll files CLEAN. Ready for paper submission.')
else:
    print('\nIssues remain — investigate above.')
