# AFPDB (PhysioNet) — metodología MEDICON 2026

Este repositorio incluye el script `run_medicon2026_afpdb.py`, que reconstruye el dataset y las evaluaciones según la definición oficial del **PAF Prediction Challenge Database (AFPDB)** en PhysioNet.

## Tipos de registro

| Prefijo / forma | Duración típica | Rol |
|------------------|-----------------|-----|
| **p\*** (p01…p50, sin sufijo `c`) | Segmentos principales de **30 minutos** | Sujetos **con** episodios de PAF documentados. En cada par consecutivo **(impar, par)**, el segmento **par** precede de inmediato al PAF; el **impar** está alejado del episodio (contexto “far”). |
| **\*c** (p01c, n01c, …) | Continuaciones cortas (~5 min) de **verificación** | No son segmentos principales de entrenamiento para la tarea far vs pre-onset. |
| **n\*** (n01…n50, sin `c`) | Segmentos principales de **30 minutos** | Sujetos **sin** FA documentada (control fisiológico a nivel de sujeto/registro, no “pre-onset”). |

## Tarea principal: far vs pre-onset (solo **p\*** de 30 min)

- **Entrada**: exactamente los 50 registros `p01` … `p50` (ninguno termina en `c`).
- **Etiqueta** `y`:
  - `p01, p03, …, p49` → `y = 0` (lejos de PAF).
  - `p02, p04, …, p50` → `y = 1` (pre-onset: segmento de 30 min inmediatamente anterior al PAF).
- **`pair_id`**: cada par (impar, par) comparte el mismo id: `(p01,p02)→0`, `(p03,p04)→1`, …, `(p49,p50)→24`.
- **Excluidos**: todos los `*c` y todos los `n*` **no** entran en esta tarea (evita mezclar “sin PAF documentada” con pre-onset).
- **Validación**: `GroupKFold` por `pair_id` (sin solapamiento de pares entre train y validación en un mismo fold). El umbral de decisión fijado en el script es **0.5** sobre la probabilidad de la clase positiva; **no** se optimiza en test ni en un hold-out global.
- **Modelos**: regresión logística, bosque aleatorio y **SVM lineal** con calibración sigmoide (`CalibratedClassifierCV` sobre `LinearSVC`) para obtener `predict_proba` y métricas como Brier sin usar el modo costoso `probability=True` de `SVC` RBF en bucles de CV repetidos.

## Análisis secundario (opcional en el mismo script)

Archivos `manifest_secondary_PAF_vs_noPAF.csv`, `Xy_secondary_PAF_vs_noPAF.csv` y `secondary_pn_cv_summary.csv`:

- **100 filas**: `p01…p50` con `y=1` frente a `n01…n50` con `y=0` (mismas ventanas de 30 min, sin sufijos `c`).
- **Interpretación**: discriminación **PAF vs no PAF** a nivel de registro de sujeto (o de par n/p agrupado), **no** equivalencia con predicción de pre-onset temporal. No debe citarse como la misma tarea clínica que far vs pre-onset.

## Características (RR → RMSSD por ventanas → z robusto → CUSUM)

Se conservan las seis características “originales” del notebook previo y el conjunto ampliado **SPC_Fast_14** (CUSUM por ramas, fracción de subventanas que pasan un SQI simple, métricas de irregularidad de RR). La función de puerta SQI por subventanas está **implementada en el script** (no requiere el paquete externo `af_predict`).

Para acelerar la lectura, solo se cargan **cabecera + anotaciones QRS** (`rdheader` + `rdann`), no la señal ECG completa, porque las métricas dependen únicamente de intervalos RR.

## Salidas en `outputs_medicon2026/`

| Archivo | Contenido |
|---------|-----------|
| `manifest_far_preonset_p30min.csv` | Lista oficial de registros y etiquetas de la tarea principal. |
| `Xy_far_preonset_main.csv` | Dataset completo con features. |
| `dataset_audit_main.csv` + `gkf_fold_audit.csv` | Comprobaciones de integridad y folds. |
| `cv_summary.csv`, `benchmark_summary.csv`, `ablation_*.csv` | Resultados de CV y ablación Original_6 vs SPC_Fast_14. |
| `bootstrap_group_metrics_LogReg_SPC.csv` | IC bootstrap por **bloques** (`pair_id`), umbral fijo 0.5. |
| `feature_audit_table.csv`, `interpretability_top15.csv` | Auditoría de NaN y coeficientes (LogReg, espacio escalado). |
| `figures/*.png`, `figures/*.tif` | Figuras a **600 dpi** (TIFF con compresión si el entorno lo permite). |

## Ejecución

```bash
cd ruta/al/proyecto/Medicons
pip install -r requirements.txt
python run_medicon2026_afpdb.py
```

Tiempo aproximado: varios minutos en SSD local; puede ser **mucho mayor** si la carpeta `afpdb` está en OneDrive u otro almacenamiento sincronizado (muchas lecturas de cabecera/anotaciones).

Solo figuras (600 dpi PNG/TIFF) a partir de CSV ya generados:

```bash
python regenerate_figures_medicon2026.py
```

Los resultados numéricos de ejecuciones anteriores basadas en pares `n` + `n+c` **no** son comparables con esta definición de tarea y no deben reportarse como válidos para el mismo endpoint.
