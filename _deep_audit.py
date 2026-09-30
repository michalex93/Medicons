"""Auditoría profunda: muestra el contenido completo de cada celda problemática."""
import json
from pathlib import Path

NB = Path('PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.ipynb')
nb = json.load(open(NB, encoding='utf-8'))

BAD = ['outputs_medicon2026','ablation_summary','interpretability_top15',
       'older pipeline','old pipeline','n+nc','SPC_Fast','Original_6','SPC_Fast-14',
       'Original-6','OUT2026']

for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    hits = [t for t in BAD if t in src]
    if hits:
        print(f'\n{"="*70}')
        print(f'CELL {i:2d}  type={c["cell_type"]}  id={c.get("id","")}')
        print(f'HITS: {hits}')
        print('SOURCE:')
        print(src[:600])
        print('...' if len(src) > 600 else '')
