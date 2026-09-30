# Agregacion temporal de features — AFPDB/PAF
_Fecha: 2026-04-24 18:03_

## Metodologia
Early/late definidos SOLO por posicion temporal (start_time_sec).
Las etiquetas y=0/y=1 NO se usan para definir ventanas early o late.

## Resultados

| window_type   |   window_size |   n_features |   auc_mean |   pairwise_concordance_mean |
|:--------------|--------------:|-------------:|-----------:|----------------------------:|
| rr_count      |           100 |           70 |      0.792 |                        0.76 |
| rr_count      |           128 |           70 |      0.744 |                        0.76 |
| time          |           300 |           70 |      0.816 |                        0.84 |
| full_record   |          1800 |           70 |      0.864 |                        0.92 |
