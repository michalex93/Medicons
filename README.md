# Multiscale Statistical Process Control of RR Interval Dynamics for Explainable Pre-Onset Characterization of Paroxysmal Atrial Fibrillation

**Paper (MEDICON 2026 / IFMBE Proceedings, vol. 145)**  
Springer chapter: https://link.springer.com/chapter/10.1007/978-3-032-37739-5_32  
**DOI:** [10.1007/978-3-032-37739-5_32](https://doi.org/10.1007/978-3-032-37739-5_32)

This repository contains the analysis code, notebooks, and small derived result tables used for the MEDICON 2026 study on **explainable pre-onset characterization of paroxysmal atrial fibrillation (PAF)** from **RR-interval dynamics** using **multiscale statistical process control (SPC)**.

> **Data availability (as stated in the paper):** code and workflows are provided in this public repository. The PhysioNet AFPDB waveforms are **not** redistributed here; download them separately (see below).

## Method (brief)

Within AFPDB subjects with documented PAF, we contrast **30-minute far-from-onset** segments (`p` odd) vs **immediately pre-onset** segments (`p` even) using only RR / QRS annotations (no full ECG waveform required for feature extraction).

Pipeline highlights:

- Multiscale RR windows (`rr_count`, 5 min time windows, full 30 min record)
- Explainable blocks: HRV, SPC charts (Shewhart, CUSUM, EWMA, moving range, Western Electric runs), Poincaré, RR-inferred ectopy-like patterns
- Subject-pair aware validation (`GroupKFold` on `pair_id`)
- Logistic regression / RF / calibrated linear SVM reporting with interpretability audits

**Do not confuse** the main far vs pre-onset task with the optional secondary cohort contrast (`p*` vs `n*`), which is not temporal pre-onset prediction.

## Citation

**APA**

Castañeda Alarcón, M. A., et al. (2026). Multiscale statistical process control of RR interval dynamics for explainable pre-onset characterization of paroxysmal atrial fibrillation. In *IFMBE Proceedings* (Vol. 145). Springer. https://doi.org/10.1007/978-3-032-37739-5_32

**BibTeX**

```bibtex
@InProceedings{CastanedaAlarcon2026MediconPAFSPC,
  author    = {Casta{\~n}eda Alarc{\'o}n, Michel Alexander and others},
  title     = {Multiscale Statistical Process Control of {RR} Interval Dynamics for Explainable Pre-Onset Characterization of Paroxysmal Atrial Fibrillation},
  booktitle = {IFMBE Proceedings},
  volume    = {145},
  year      = {2026},
  publisher = {Springer},
  doi       = {10.1007/978-3-032-37739-5_32},
  url       = {https://link.springer.com/chapter/10.1007/978-3-032-37739-5_32}
}
```

Please replace `and others` with the full co-author list from the Springer chapter when citing formally.

## Authors / contact

- **Michel Alexander Castañeda Alarcón** (corresponding / first author)  
  ORCID: [0009-0006-9860-0865](https://orcid.org/0009-0006-9860-0865)

## Requirements

- Python 3.10+ recommended
- Dependencies: see `requirements.txt` (`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `wfdb`)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest   # for tests
```

## Obtaining AFPDB (PhysioNet)

We **do not** ship the AFPDB database in this repo (~200+ MB locally; licensing belongs to PhysioNet).

1. Create a PhysioNet account if needed and accept the data use agreement.  
2. Download **AF Termination Challenge Database / PAF Prediction Challenge Database (AFPDB) v1.0.0**:  
   https://physionet.org/content/afpdb/1.0.0/
3. Place the records under `./afpdb/` at the repository root (headers + annotations are sufficient for the RR pipeline; full `.dat` waveforms are optional for this feature set).

Example with the PhysioNet CLI / `wfdb` tooling (adjust to your preferred download method):

```powershell
# After installing wfdb / physionet tools, download into ./afpdb
# Official page: https://physionet.org/content/afpdb/1.0.0/
```

AFPDB remains under its **PhysioNet / original database license**; this repository does not grant rights to redistribute those files.

## Reproduce main analyses

From the repository root (with `./afpdb` present):

```powershell
# Classic MEDICON AFPDB far vs pre-onset pipeline + tables under outputs_medicon2026/
python run_medicon2026_afpdb.py

# Multiscale RR / HRV / SPC pipeline (package under src/afpdb_multiscale)
python run_multiscale_v2.py
python run_multiscale_v2.py --no-secondary
python run_multiscale_v2.py --skip-build

# Optional experiments / audits
python run_mahal_experiment.py
python run_interpretability_audit.py
python run_phases_2_6.py
python regenerate_figures_medicon2026.py

# Tests
python -m pytest tests -v
```

Primary notebooks:

- `PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.ipynb` (+ executed companion)
- `PAF_Prediction_AFPDB_Standalone_MEDICON2026.ipynb`

Additional notes:

- `README_MEDICON2026_AFPDB.md` — AFPDB task definition (far vs pre-onset)
- `README_MULTISCALE.md` — multiscale feature blocks and windows

Small derived CSVs / PNGs used for paper tables and figures are tracked under `outputs_medicon2026/` and `artifacts/{tables,figures,reports}/` (raw window dumps and `.tif` intermediates are excluded).

## Repository layout

```
src/afpdb_multiscale/   # feature extraction, SPC, evaluation, reporting
tests/                  # unit / anti-leakage tests
run_*.py                # entry-point scripts
outputs_medicon2026/    # small summary CSVs + key PNG figures
artifacts/              # multiscale tables, PNG figures, markdown reports
```

## License

| Asset | License / rights |
|-------|------------------|
| **Code** in this repository | [MIT](./LICENSE) |
| **PhysioNet AFPDB** data | PhysioNet / database terms (not redistributed here) |
| **Published paper PDF / chapter** | Springer copyright — **not** included in this repository |

Do not commit publisher PDFs, Licence-to-Publish forms, or submission packages.

## Disclaimer

This software is for research reproducibility of the published MEDICON 2026 analysis. It is **not** a medical device and must not be used for clinical decision-making.
