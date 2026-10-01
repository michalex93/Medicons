import json
nb = json.load(open('PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.ipynb', encoding='utf-8'))
bad = ['outputs_medicon2026','ablation_summary','interpretability_top15',
       'old pipeline','from older pipeline','n+nc','SPC_Fast','Original_6','OUT2026']
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    hits = [t for t in bad if t in src]
    if hits:
        preview = src[:120].replace('\n',' ')
        ctype = c['cell_type']
        print(f'Cell {i:2d} [{ctype:8s}] ISSUES={hits}')
        print(f'         {preview}')
        print()
