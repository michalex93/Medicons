# Pipeline multi-escala RR/HRV/SPC — MEDICON 2026

## Pregunta de investigación

> ¿A partir solo de intervalos RR y derivados explicables se puede distinguir,
> dentro de sujetos con PAF documentada, el segmento de 30 min **lejos** del episodio
> (p impar, y=0) frente al segmento de 30 min **inmediatamente previo** al PAF (p par, y=1)?

**NO confundir** con el análisis secundario: p\* vs n\* es discriminación de cohorte (sujeto con PAF vs sin AF documentada), no proximidad temporal al episodio.

## Estructura del proyecto

```
src/
  afpdb_multiscale/
    config.py          # constantes, umbrales SPC, listas de features, bloques
    loader.py          # carga QRS desde AFPDB local + baseline por registro
    windows.py         # ventanas rr_count, time, full_record
    features_hrv.py    # HRV básico (RMSSD, SDSD, CV, etc.) + Poincaré (SD1, SD2)
    features_spc.py    # Shewhart, CUSUM bilateral, EWMA, MR, Runs rules WE
    features_ectopy.py # patrones ectopia-like inferidos solo desde RR
    build_dataset.py   # orquesta extracción y agregación por registro
    evaluate.py        # GroupKFold × bloques × modelos, métricas, importancia
    report.py          # reporte Markdown con limitaciones
artifacts/
  figures/             # heatmaps AUC por ventana × bloque
  reports/             # reporte metodológico .md
  tables/              # CSVs: windows_raw, Xy_aggregated, cv_detail, etc.
tests/
  test_multiscale.py   # 13 tests: correctness, antialeakage, prohibición 30s, etc.
run_multiscale_v2.py   # script de entrada
```

## Ejecución

```powershell
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo (extrae features + evalúa + genera reporte)
python run_multiscale_v2.py

# Omitir análisis secundario (más rápido)
python run_multiscale_v2.py --no-secondary

# Saltar re-extracción si los CSV ya existen
python run_multiscale_v2.py --skip-build

# Tests
python -m pytest tests/test_multiscale.py -v
```

## Ventanas evaluadas

| Tipo | Tamaño | Nota |
|------|--------|------|
| `rr_count` | 50 RR | ~35–45 s |
| `rr_count` | 100 RR | ~70–90 s |
| `rr_count` | 128 RR | ~90–110 s |
| `rr_count` | 200 RR | ~140–180 s |
| `time` | 300 s (5 min) | 6 ventanas por registro |
| `full_record` | ~1800 s (30 min) | 1 ventana por registro |

**PROHIBIDO**: ventanas de 30 segundos en el análisis principal. El código lanza `ProhibitedWindowError` si se intenta.

## Bloques de features

| Bloque | Contenido |
|--------|-----------|
| A\_HRV | mean\_rr, median\_rr, std\_rr, mad\_rr, rmssd, sdsd, cv\_rr, rr\_diff\_mean, rr\_diff\_std |
| B\_SPC | Shewhart + CUSUM + EWMA + MR + Runs rules |
| C\_HRV\_SPC | A + B |
| D\_HRV\_SPC\_Poincaré | A + B + sd1, sd2, sd1\_sd2\_ratio |
| E\_HRV\_SPC\_Poincaré\_Ectopy | A + B + Poincaré + ectopy\_like\_\* |

## Cartas SPC

- **Shewhart individuos**: |z| > 3σ → alarma puntual.
- **CUSUM bilateral**: acumula desviaciones positivas (C+) y negativas (C−) con k=0.5, h=7. C+ captura exceso simpático (taquicardia-like); C− captura exceso parasimpático (bradicardia-like).
- **EWMA** (λ=0.2): detecta cambios graduales; límite estacionario L·√(λ/(2−λ)).
- **Moving Range**: volatilidad local entre RR consecutivos.
- **Runs rules WE** (4 reglas simplificadas): tendencias y rachas.

**Baseline por registro**: se calculan μ y σ sobre los primeros 200 RR del registro. Esto garantiza que los límites de control no usen información futura del mismo registro (`lead_time` seguro).

## lead\_time

- Solo calculado para **y=1** (pre-onset).
- Supuesto: PAF onset al final del registro (segundo 1800).
- `lead_time_sec = 1800 − primera_alarma_absoluta_cusum`.
- Para **y=0**: alarmas = falsas alarmas → se reporta `false_alarms_per_hour`.

## Ectopia-like (advertencia metodológica)

Las variables `ectopy_like_*` son **patrones compatibles con ectopia inferidos solo desde RR**. No son diagnóstico confirmado de PAC/APC ni análisis morfológico de onda P.  
No usar la palabra "PAC confirmado" sin anotación clínica o detector morfológico explícito.

## Tests (13 en total)

1. Ventanas rr\_count = {50, 100, 128, 200} tienen el tamaño exacto correcto.
2. Ventanas de 5 min duran ≤ 300 s.
3. `full_record` devuelve 1 ventana con todos los RR.
4. Ventana de 30 s lanza `ProhibitedWindowError` en tarea principal.
5. 30 s no está en las listas de ventanas de config.
6. GroupKFold por `pair_id`: ningún par aparece en train y val simultáneamente.
7. `main_p_far_vs_p_pre` y `secondary_p_vs_n` no se mezclan; sin registros n\*/\*c en main.
8. Features con RR bien formado: sin inf silencioso, sin None.
9. Reproducibilidad: coeficientes idénticos con random\_state fijo.
10. (Incluido en test 9 + otros) lead\_time\_sec = NaN para y=0; ≤ 1800 para y=1.

## Riesgos metodológicos

1. **Muestra pequeña** (n=25 pares): AUC con IC bootstrap amplio. Resultados exploratorios.
2. **Sin validación externa**: un único dataset de reto.
3. **Hiperparámetros SPC fijos**: h, k, λ no se ajustan por fold. Si se tunan, debe ser dentro del train fold via nested CV.
4. **lead\_time asume onset al final del registro**: simplificación que puede introducir imprecisión.
5. **Ectopy-like no validada morfológicamente**.
6. **Posible correlación entre ventanas del mismo registro**: la agregación por mediana mitiga pero no elimina este efecto.
