"""Auditoría del notebook _executed.ipynb (source + outputs)."""
import json
from pathlib import Path

NB = Path('PAF_Prediction_AFPDB_SPC_RR_MEDICON2026_executed.ipynb')
nb = json.load(open(NB, encoding='utf-8'))

BAD = ['outputs_medicon2026','ablation_summary','interpretability_top15',
       'older pipeline','old pipeline','n+nc','SPC_Fast','Original_6','SPC_Fast-14',
       'Original-6','OUT2026']

for i, c in enumerate(nb['cells']):
    all_text = ''.join(c.get('source', []))
    # Also check outputs
    for o in c.get('outputs', []):
        all_text += ''.join(o.get('text', []))
        all_text += ''.join(o.get('data', {}).get('text/plain', []))
        all_text += ''.join(o.get('data', {}).get('text/html', []))
        all_text += str(o.get('evalue', ''))
        all_text += str(o.get('traceback', []))
    hits = [t for t in BAD if t in all_text]
    if hits:
        src_preview = ''.join(c.get('source', []))[:200].replace('\n', ' ')
        print(f'Cell {i:2d} [{c["cell_type"]}] HITS={hits}')
        print(f'  src: {src_preview}')
        print()
