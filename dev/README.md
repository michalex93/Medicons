# dev/ — maintenance utilities

Scripts used while preparing the MEDICON 2026 submission: notebook clean-up, audits of executed notebooks,
re-export of figures/tables and bootstrap regeneration. They are **not** needed to reproduce the analysis;
use the `run_*.py` entry points in the repository root for that.

Run them from the repository root, e.g. `python dev/_regenerate_bootstrap_and_export.py`.

| Script | Purpose |
|---|---|
| `apply_paper_changes.py` | Applies paper-ready edits to the notebooks and exports `artifacts/tables/paper_ready_results.csv` |
| `_regenerate_bootstrap_and_export.py` | Regenerates `bootstrap_ci_v1_results.csv` (500 pair-bootstrap replicates) and re-exports the notebook |
| `_fix_nb.py`, `_patch_coef_cell.py` | One-off notebook edits |
| `_audit_nb.py`, `_audit_executed.py`, `_deep_audit.py`, `_final_verify.py`, `_final_clean_export.py` | Consistency audits of notebooks / HTML exports |
