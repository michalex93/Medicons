# MEDICON 2026 PAF-SPC-RR Full Paper v2 - Author, ORCID, Bibliography and Claim Audit

## Author/ORCID updates
- Michel Alexander Castañeda Alarcón: ORCID 0009-0006-9860-0865 (provided by author).
- José Javier Díaz-Carmona: ORCID 0000-0002-7363-3349. Public evidence: TeCEO author metadata and SCIRP editorial profile.
- Horacio Rostro González: ORCID 0000-0001-7530-9027. Public evidence: IQS profile ORCID link and IntechOpen author profile. Primary affiliation in the manuscript was set to Universidad de Guanajuato as requested by the corresponding author; IQS/Universitat Ramon Llull was retained as secondary affiliation.
- Alejandro Espinosa-Calderón: ORCID 0000-0002-1758-4212. Public evidence: SciProfiles and Wiley article metadata associated with Tecnológico Nacional de México / CRODE.

## Bibliographic audit
The reference list was cleaned to include only sources used by the manuscript and to remove previously uncited or unnecessary references. The final reference list supports the following claims:

1. PhysioNet and AFPDB provenance:
   - Goldberger et al. (2000) supports PhysioNet as the physiological signal resource.
   - Moody et al. (2001) supports the PAF Prediction Challenge formulation.
   - PhysioNet AFPDB database page supports the p/n/c record structure and official citation.

2. PAF pre-onset and prior methods:
   - Hnatkova et al. (1998) supports rhythm changes preceding spontaneous PAF.
   - Zong et al. (2001) supports ECG arrhythmia/APC-based PAF prediction in the CinC challenge.
   - Maier et al. (2001) supports HRV-based screening/prediction work on PAF.
   - Thong et al. (2004) supports APC-based PAF prediction.
   - Narin et al. (2018) supports short-term HRV-based early PAF prediction.
   - Castro et al. (2021) supports a modern HRV feature-analysis methodology using AFPDB.

3. SPC and healthcare monitoring:
   - Montgomery (2020) supports classical SPC concepts.
   - Woodall (2006) supports use of control charts in healthcare and public-health monitoring.

4. HRV definitions:
   - Task Force (1996) supports formal HRV measurement standards.
   - Shaffer and Ginsberg (2017) supports HRV metric overview and interpretation.

5. Software and reproducibility:
   - WFDB Software Package and WFDB Python support waveform/QRS annotation handling.
   - Pedregosa et al. (2011) supports scikit-learn model implementation.

6. Interpretability:
   - Rudin (2019) supports the claim that interpretable models are preferable to post-hoc black-box explanations in high-stakes decisions.

## Scientific claim audit
The manuscript intentionally uses "pre-onset characterization" rather than "clinical prediction" or "real-time detection." This is justified because:
- the primary dataset has 25 matched pairs;
- the task is retrospective;
- QRS annotations are machine-generated and unaudited;
- full-record aggregation uses the complete 30-min record;
- no external validation dataset is included.

## Figure audit
The v2 figure set was regenerated with simplified layouts, larger fonts, and reduced internal text to avoid overlapping labels. All figures are available in PNG and TIFF formats; TIFF files were exported at 600 dpi.

## Final caution before submission
Before actual MEDICON submission, author names, final institutional wording, and ORCID display style should be confirmed with all coauthors.
