"""
Verificación final completa de todos los artefactos paper-ready.
Busca en: .ipynb fuente, _executed.ipynb, .html
"""
import json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path('.')
NB_SRC  = ROOT / 'PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.ipynb'
NB_EXEC = ROOT / 'PAF_Prediction_AFPDB_SPC_RR_MEDICON2026_executed.ipynb'
NB_HTML = ROOT / 'PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.html'

BANNED = [
    'outputs_medicon2026',
    'ablation_summary',
    'interpretability_top15',
    'older pipeline',
    'old pipeline',
    'n+nc',
    'SPC_Fast',
    'Original_6',
    'SPC_Fast-14',
    'Original-6',
    'OUT2026',
]

def audit_json(path, label):
    nb = json.load(open(path, encoding='utf-8'))
    issues = []
    for i, c in enumerate(nb['cells']):
        full = ''.join(c.get('source', []))
        for o in c.get('outputs', []):
            full += ''.join(o.get('text', []))
            full += ''.join(o.get('data', {}).get('text/plain', []))
            full += ''.join(o.get('data', {}).get('text/html', []))
        for term in BANNED:
            if term in full:
                ctx = full[full.find(term)-30:full.find(term)+60].replace('\n',' ')
                issues.append(f'  Cell {i:2d}: [{term}] ...{ctx}...')
    if issues:
        print(f'FAIL {label}:')
        for iss in issues: print(iss)
    else:
        print(f'PASS {label}: no banned terms found.')

def audit_html(path, label):
    text = open(path, encoding='utf-8', errors='replace').read()
    issues = []
    for term in BANNED:
        idx = text.find(term)
        if idx >= 0:
            ctx = text[max(0,idx-40):idx+80].replace('\n',' ')
            issues.append(f'  [{term}] at pos {idx}: ...{ctx}...')
    if issues:
        print(f'FAIL {label}:')
        for iss in issues: print(iss)
    else:
        print(f'PASS {label}: no banned terms found.')

print('=== Final paper-ready audit ===\n')
audit_json(NB_SRC,  '.ipynb source')
audit_json(NB_EXEC, '_executed.ipynb')
audit_html(NB_HTML, '.html')

# File sizes
print('\n=== Artifact sizes ===')
for p in [NB_SRC, NB_EXEC, NB_HTML]:
    if p.is_file():
        sz = p.stat().st_size / 1024
        print(f'  {p.name}: {sz:.0f} KB')

# Figures
fig_dir = ROOT / 'artifacts' / 'figures'
pngs = list(fig_dir.glob('*.png'))
tifs = list(fig_dir.glob('*.tif'))
svgs = list(fig_dir.glob('*.svg'))
pdfs = list(fig_dir.glob('*.pdf'))
print(f'\n=== Figures in artifacts/figures/ ===')
print(f'  .png: {len(pngs)}')
print(f'  .tif: {len(tifs)}')
print(f'  .svg: {len(svgs)}')
print(f'  .pdf: {len(pdfs)}')

# CSV sources used
print('\n=== CSV sources (only artifacts/ expected) ===')
nb = json.load(open(NB_SRC, encoding='utf-8'))
all_src = '\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
import re
csv_refs = re.findall(r"['\"]([^'\"]*\.csv)['\"]", all_src)
for r in sorted(set(csv_refs)):
    marker = 'EXTERNAL-OLD' if 'outputs_medicon2026' in r else 'OK'
    print(f'  {marker}: {r}')
