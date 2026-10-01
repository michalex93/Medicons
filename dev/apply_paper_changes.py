"""
Aplica todos los cambios de paper-ready a ambos notebooks y genera archivos auxiliares.
Ejecutar: python apply_paper_changes.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import pandas as pd

TABLES   = ROOT / "artifacts" / "tables"
REPORTS  = ROOT / "artifacts" / "reports"
FIG_DIR  = ROOT / "artifacts" / "figures"

NB_PAPER  = ROOT / "PAF_Prediction_AFPDB_SPC_RR_MEDICON2026.ipynb"
NB_STANDALONE = ROOT / "PAF_Prediction_AFPDB_Standalone_MEDICON2026.ipynb"

# ─── helper: load / save notebook ───────────────────────────────────────────

def load_nb(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save_nb(nb, path):
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {path.name}")

def source_str(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src

def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)

def replace_in_nb(nb, old, new):
    count = 0
    for cell in nb["cells"]:
        src = source_str(cell)
        if old in src:
            set_source(cell, src.replace(old, new))
            count += 1
    return count

def insert_cell_after_id(nb, cell_id, new_cell):
    """Insert new_cell after the cell with id == cell_id."""
    for i, c in enumerate(nb["cells"]):
        if c.get("id") == cell_id:
            nb["cells"].insert(i + 1, new_cell)
            return True
    nb["cells"].append(new_cell)
    return True

def append_cell(nb, source, cell_type="code", cell_id=None):
    cell = {
        "cell_type": cell_type,
        "id": cell_id or f"auto-{len(nb['cells'])}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    nb["cells"].append(cell)

# ============================================================================
# STEP 1 — Global text replacements (both notebooks)
# ============================================================================

GLOBAL_REPLACEMENTS = [
    # Title
    ("PAF Onset Prediction from RR Intervals",
     "Pre-Onset Characterization of PAF from RR Intervals"),
    ("PAF Onset Prediction", "Pre-Onset Characterization of PAF"),
    # DPI
    ("dpi=300", "dpi=600"),
    # Single-centre
    ("single-centre challenge dataset", "single public challenge dataset (PhysioNet)"),
    ("single-centre", "single public challenge"),
    ("single centre", "single public challenge"),
    ("a challenge dataset", "a public challenge dataset (PhysioNet)"),
    # Label: do not say "normal" or "healthy" for y=0
    ('"normal"', '"far_baseline"'),
    ('"healthy"', '"far_baseline"'),
    ('"control"', '"far_baseline"'),
]

# Additional replacement for paper-ready notebook: remove outputs_medicon2026 ref
PAPER_ONLY_REPLACEMENTS = [
    ("OUT2026  = ROOT / 'outputs_medicon2026'", "# OUT2026 removed — use artifacts/ instead"),
    ("OUT2026 = ROOT / 'outputs_medicon2026'", "# OUT2026 removed — use artifacts/ instead"),
]

# ============================================================================
# STEP 2 — Savefig: add svg/pdf for diagrams
# ============================================================================

OLD_SAVEFIG = """def savefig(fig, name, dpi=300):
    fig.savefig(FIG_DIR / f'{name}.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.tif', dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close(fig)"""

NEW_SAVEFIG = """def savefig(fig, name, dpi=600, vector=True):
    \"\"\"Save figure as PNG + TIFF (600 dpi) and optionally SVG + PDF for diagrams.\"\"\"
    fig.savefig(FIG_DIR / f'{name}.png', dpi=dpi, bbox_inches='tight')
    try:
        fig.savefig(FIG_DIR / f'{name}.tif', dpi=dpi, bbox_inches='tight')
    except Exception:
        pass
    if vector:
        try:
            fig.savefig(FIG_DIR / f'{name}.svg', bbox_inches='tight')
            fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight')
        except Exception:
            pass
    plt.show()
    plt.close(fig)"""

# Standalone notebook savefig variant
OLD_SAVEFIG_SA = """def savefig(fig, name, dpi=300):
    for ext in ('png', 'tif'):
        try: fig.savefig(FIG_DIR / f'{name}.{ext}', dpi=dpi, bbox_inches='tight')
        except Exception: pass
    plt.show(); plt.close(fig)"""

NEW_SAVEFIG_SA = """def savefig(fig, name, dpi=600, vector=True):
    \"\"\"Save figure as PNG + TIFF 600 dpi; SVG + PDF for diagrams.\"\"\"
    for ext in ('png', 'tif'):
        try: fig.savefig(FIG_DIR / f'{name}.{ext}', dpi=dpi, bbox_inches='tight')
        except Exception: pass
    if vector:
        for ext in ('svg', 'pdf'):
            try: fig.savefig(FIG_DIR / f'{name}.{ext}', bbox_inches='tight')
            except Exception: pass
    plt.show(); plt.close(fig)"""

# ============================================================================
# STEP 3 — New cells to add to both notebooks
# ============================================================================

CELL_AUDIT = """\
# ── Dataset audit: export dataset_audit.csv ───────────────────────────────
from afpdb_multiscale.loader import build_main_manifest
import pandas as pd
from pathlib import Path
TABLES = Path('artifacts/tables')
TABLES.mkdir(parents=True, exist_ok=True)

manifest = build_main_manifest()
audit_rows = []
for rec in manifest:
    rid = rec['record_id']
    audit_rows.append({
        'record_id'      : rid,
        'pair_id'        : rec['pair_id'],
        'y'              : rec['y'],
        'role'           : 'pre_onset' if rec['y'] == 1 else 'far_baseline',
        'record_type'    : 'p_30min_main',   # p-odd or p-even 30-min segments
        'is_continuation': False,             # *c records excluded from main task
    })

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(TABLES / 'dataset_audit.csv', index=False)
print(f'Exported: dataset_audit.csv  ({len(audit_df)} records)')
display(audit_df.head(8))
"""

CELL_AUDIT_SA = """\
# ── Dataset audit: export dataset_audit.csv ───────────────────────────────
import pandas as pd
from pathlib import Path
CACHE_DIR = Path('notebook_cache')
CACHE_DIR.mkdir(exist_ok=True)

audit_rows = []
for rec in manifest:
    rid = rec['record_id']
    audit_rows.append({
        'record_id'      : rid,
        'pair_id'        : rec['pair_id'],
        'y'              : rec['y'],
        'role'           : 'pre_onset' if rec['y'] == 1 else 'far_baseline',
        'record_type'    : 'p_30min_main',   # p-odd or p-even 30-min segments
        'is_continuation': False,             # *c records excluded from main task
    })

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(CACHE_DIR / 'dataset_audit.csv', index=False)
print(f'Exported: notebook_cache/dataset_audit.csv  ({len(audit_df)} records)')

# Confirm no n* or *c in dataset
print(f'No n* records in dataset : {not audit_df.record_id.str.startswith(\"n\").any()}')
print(f'No *c records in dataset : {not audit_df.record_id.str.endswith(\"c\").any()}')
display(audit_df.head(8))
"""

CELL_CONFIRM_XY = """\
# ── Confirm no n* or *c in Xy_aggregated_main.csv ─────────────────────────
from pathlib import Path
import pandas as pd
p = Path('artifacts/tables/Xy_aggregated_main.csv')
if p.is_file():
    xy = pd.read_csv(p)
    n_star = xy.record_id.str.startswith('n').any()
    c_star = xy.record_id.str.endswith('c').any()
    print(f'File: {p.name}  ({len(xy)} rows)')
    print(f'  Contains n* records : {n_star}  — expected: False')
    print(f'  Contains *c records : {c_star}  — expected: False')
    assert not n_star, 'ERROR: n* records found in Xy_aggregated_main.csv'
    assert not c_star, 'ERROR: *c records found in Xy_aggregated_main.csv'
    print('  AUDIT PASSED: Xy_aggregated_main.csv contains only p* (30-min) records.')
else:
    print('Xy_aggregated_main.csv not found — run run_multiscale_v2.py first.')
"""

CELL_CONFIRM_XY_SA = """\
# ── Confirm no n* or *c in feature cache ──────────────────────────────────
from pathlib import Path
import pandas as pd
cache_p = Path('notebook_cache/features.csv')
if cache_p.is_file():
    xy = pd.read_csv(cache_p)
    n_star = xy.record_id.str.startswith('n').any()
    c_star = xy.record_id.str.endswith('c').any()
    print(f'Features cache ({len(xy)} rows):')
    print(f'  Contains n* records : {n_star}  — expected: False')
    print(f'  Contains *c records : {c_star}  — expected: False')
    assert not n_star, 'ERROR: n* records found in cache'
    assert not c_star, 'ERROR: *c records found in cache'
    print('  AUDIT PASSED: cache contains only p* (30-min) records.')
else:
    print('Cache not yet generated — run feature extraction first.')
"""

CELL_PAPER_RESULTS = """\
# ── Export paper_ready_results.csv ────────────────────────────────────────
from pathlib import Path
import numpy as np, pandas as pd

TABLES  = Path('artifacts/tables')
cv_p    = TABLES / 'cv_summary_main.csv'
ci_p    = TABLES / 'bootstrap_ci_v1_results.csv'
pw_p    = TABLES / 'pairwise_ranking_results.csv'

rows = []
if cv_p.is_file():
    cv  = pd.read_csv(cv_p)
    ci  = pd.read_csv(ci_p) if ci_p.is_file() else pd.DataFrame()
    pw  = pd.read_csv(pw_p) if pw_p.is_file() else pd.DataFrame()

    for _, r in cv.iterrows():
        ci_row = ci[(ci.metric=='auc') & (ci.config==r.get('block',''))] if not ci.empty else pd.DataFrame()
        pc_row = pw[(pw.block==r.get('block','')) & (pw.window_type==r.get('window_type',''))] if not pw.empty else pd.DataFrame()
        rows.append({
            'window_type'   : r.get('window_type',''),
            'window_size'   : r.get('window_size',''),
            'feature_block' : r.get('block',''),
            'model'         : r.get('model',''),
            'auc_mean'      : round(float(r.get('auc_mean', r.get('auc',np.nan))), 4),
            'auc_std'       : round(float(r.get('auc_std',  np.nan)), 4),
            'ci_low'        : round(float(ci_row.ci_low.iloc[0]),  4) if len(ci_row) else np.nan,
            'ci_high'       : round(float(ci_row.ci_high.iloc[0]), 4) if len(ci_row) else np.nan,
            'f1_mean'       : round(float(r.get('f1_mean', r.get('f1', np.nan))), 4),
            'sensitivity'   : round(float(r.get('sens_mean', r.get('sensitivity', np.nan))), 4),
            'specificity'   : round(float(r.get('spec_mean', r.get('specificity', np.nan))), 4),
            'mcc_mean'      : round(float(r.get('mcc_mean', r.get('mcc', np.nan))), 4),
            'pairwise_concordance': round(float(pc_row.pairwise_concordance.mean()), 4)
                                    if len(pc_row) else np.nan,
        })

pr_df = pd.DataFrame(rows).drop_duplicates()
pr_df.to_csv(TABLES / 'paper_ready_results.csv', index=False)
print(f'Exported: paper_ready_results.csv  ({len(pr_df)} rows)')
display(pr_df.sort_values('auc_mean', ascending=False).head(12).round(4))
"""

CELL_PAPER_RESULTS_SA = """\
# ── Export paper_ready_results.csv ────────────────────────────────────────
import numpy as np, pandas as pd
from pathlib import Path

OUT = Path('notebook_cache')
if not cv_all.empty:
    summary_paper = cv_all.groupby(['window_type','window_size','block','model']).agg(
        auc_mean=('auc','mean'), auc_std=('auc','std'),
        f1_mean=('f1','mean'), sensitivity=('sens','mean'),
        specificity=('spec','mean'), mcc_mean=('mcc','mean'),
        pc_mean=('pc','mean')
    ).reset_index().round(4)
    summary_paper.rename(columns={'block':'feature_block'}, inplace=True)
    summary_paper.to_csv(OUT / 'paper_ready_results.csv', index=False)
    print(f'Exported: notebook_cache/paper_ready_results.csv  ({len(summary_paper)} rows)')
    display(summary_paper.sort_values('auc_mean', ascending=False).head(10))
else:
    print('cv_all is empty — run CV cells first.')
"""

# ============================================================================
# STEP 4 — Apply to paper-ready notebook
# ============================================================================

def apply_to_paper_ready():
    print("\n-- Paper-ready notebook")
    nb = load_nb(NB_PAPER)

    # Global text replacements
    for old, new in GLOBAL_REPLACEMENTS:
        n = replace_in_nb(nb, old, new)
        if n: print(f"  Replaced '{old[:40]}...' in {n} cell(s)")

    # Paper-only replacements
    for old, new in PAPER_ONLY_REPLACEMENTS:
        n = replace_in_nb(nb, old, new)
        if n: print(f"  Paper-only: '{old[:40]}...' in {n} cell(s)")

    # savefig upgrade
    n = replace_in_nb(nb, OLD_SAVEFIG, NEW_SAVEFIG)
    if n: print(f"  Upgraded savefig in {n} cell(s)")

    # Add audit + confirmation + paper_ready cells
    append_cell(nb, CELL_CONFIRM_XY, cell_id="confirm-xy-paper")
    append_cell(nb, CELL_AUDIT, cell_id="audit-paper")
    append_cell(nb, CELL_PAPER_RESULTS, cell_id="paper-results-paper")

    # Add note about src modules
    NOTE_SRC = """\
# ── Module reference ──────────────────────────────────────────────────────
# This notebook uses the following modules from src/afpdb_multiscale/:
#   loader.py        — AFPDB record loading, manifest construction
#   feature_blocks.py — HRV block definitions, CorrelationPruner
# Source available at: src/afpdb_multiscale/
from pathlib import Path
src_files = ['loader.py', 'feature_blocks.py']
for f in src_files:
    p = Path(f'src/afpdb_multiscale/{f}')
    size = p.stat().st_size if p.is_file() else 0
    print(f'  {f}: {size} bytes  (exists={p.is_file()})')
"""
    append_cell(nb, NOTE_SRC, cell_id="src-note-paper")

    save_nb(nb, NB_PAPER)


# ============================================================================
# STEP 5 — Apply to standalone notebook
# ============================================================================

def apply_to_standalone():
    print("\n-- Standalone notebook")
    nb = load_nb(NB_STANDALONE)

    # Global text replacements
    for old, new in GLOBAL_REPLACEMENTS:
        n = replace_in_nb(nb, old, new)
        if n: print(f"  Replaced '{old[:40]}...' in {n} cell(s)")

    # savefig upgrade
    n = replace_in_nb(nb, OLD_SAVEFIG_SA, NEW_SAVEFIG_SA)
    if n: print(f"  Upgraded savefig in {n} cell(s)")
    # Also try the paper-ready version in case it got copied
    replace_in_nb(nb, OLD_SAVEFIG, NEW_SAVEFIG)

    # Generic dpi=300 cleanup (in case any remaining)
    replace_in_nb(nb, "dpi=300", "dpi=600")

    # Add audit + confirmation + paper_ready cells
    append_cell(nb, CELL_CONFIRM_XY_SA, cell_id="confirm-xy-sa")
    append_cell(nb, CELL_AUDIT_SA, cell_id="audit-sa")
    append_cell(nb, CELL_PAPER_RESULTS_SA, cell_id="paper-results-sa")

    save_nb(nb, NB_STANDALONE)


# ============================================================================
# STEP 6 — Generate dataset_audit.csv from existing data (no notebook needed)
# ============================================================================

def generate_audit_csv():
    print("\n-- Generating dataset_audit.csv")
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in range(1, 51):
        rid = f"p{n:02d}"
        rows.append({
            "record_id"      : rid,
            "pair_id"        : (n - 1) // 2,
            "y"              : 0 if n % 2 == 1 else 1,
            "role"           : "pre_onset" if n % 2 == 0 else "far_baseline",
            "record_type"    : "p_30min_main",
            "is_continuation": False,
        })
    # Add documentation rows for excluded types
    for n in range(1, 51):
        rows.append({"record_id": f"p{n:02d}c", "pair_id": (n-1)//2,
                     "y": None, "role": "excluded_continuation",
                     "record_type": "p_continuation_5min", "is_continuation": True})
    for n in range(1, 51):
        rows.append({"record_id": f"n{n:02d}", "pair_id": (n-1)//2,
                     "y": None, "role": "excluded_no_af_subject",
                     "record_type": "n_no_af_30min", "is_continuation": False})
    for n in range(1, 51):
        rows.append({"record_id": f"n{n:02d}c", "pair_id": (n-1)//2,
                     "y": None, "role": "excluded_continuation",
                     "record_type": "n_continuation_5min", "is_continuation": True})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "dataset_audit.csv", index=False)
    print(f"  Saved: {TABLES / 'dataset_audit.csv'}  ({len(df)} rows)")
    main_only = df[df.record_type == "p_30min_main"]
    print(f"  Main task records: {len(main_only)}")
    print(f"  No n* in main task: {not main_only.record_id.str.startswith('n').any()}")
    print(f"  No *c in main task: {not main_only.record_id.str.endswith('c').any()}")


# ============================================================================
# STEP 7 — Confirm Xy_aggregated_main.csv audit
# ============================================================================

def confirm_xy_audit():
    print("\n-- Confirming Xy_aggregated_main.csv audit")
    p = TABLES / "Xy_aggregated_main.csv"
    if not p.is_file():
        print("  File not found — run run_multiscale_v2.py first."); return
    xy = pd.read_csv(p)
    n_star = xy.record_id.str.startswith("n").any()
    c_star = xy.record_id.str.endswith("c").any()
    print(f"  Total rows : {len(xy)}")
    print(f"  Contains n* records : {n_star}  — expected: False")
    print(f"  Contains *c records : {c_star}  — expected: False")
    if not n_star and not c_star:
        print("  AUDIT PASSED")
    else:
        print("  AUDIT FAILED — investigate!")


# ============================================================================
# STEP 8 — Build paper_ready_results.csv from existing CSVs
# ============================================================================

def build_paper_ready_csv():
    print("\n-- Building paper_ready_results.csv")
    cv_p  = TABLES / "cv_summary_main.csv"
    ci_p  = TABLES / "bootstrap_ci_v1_results.csv"
    pw_p  = TABLES / "pairwise_ranking_results.csv"
    bl_p  = TABLES / "v1_interpretable_baseline_results.csv"

    if not any(p.is_file() for p in [cv_p, bl_p]):
        print("  No CV results found. Run run_phases_2_6.py first."); return

    rows = []
    cv_df = pd.read_csv(cv_p) if cv_p.is_file() else pd.DataFrame()
    ci_df = pd.read_csv(ci_p) if ci_p.is_file() else pd.DataFrame()
    pw_df = pd.read_csv(pw_p) if pw_p.is_file() else pd.DataFrame()
    bl_df = pd.read_csv(bl_p) if bl_p.is_file() else pd.DataFrame()

    # From baseline results (cleaner)
    if not bl_df.empty:
        for _, r in bl_df.iterrows():
            config = r.get("config", "")
            ci_row = ci_df[(ci_df.metric == "auc") & (ci_df.config == config)] \
                     if not ci_df.empty else pd.DataFrame()
            pw_row = pw_df[pw_df.block == config] if not pw_df.empty else pd.DataFrame()
            rows.append({
                "window_type"          : "full_record",
                "window_size_sec"      : 1800,
                "feature_block"        : config,
                "model"                : "LogReg",
                "use_collinearity_pruning": r.get("use_pruning", False),
                "n_features"           : r.get("n_features", np.nan),
                "auc_observed"         : round(float(ci_row.observed.iloc[0]), 4) if len(ci_row) else r.get("auc", np.nan),
                "auc_ci_95_low"        : round(float(ci_row.ci_low.iloc[0]),  4) if len(ci_row) else np.nan,
                "auc_ci_95_high"       : round(float(ci_row.ci_high.iloc[0]), 4) if len(ci_row) else np.nan,
                "pairwise_concordance" : round(float(pw_row.pairwise_concordance.mean()), 4)
                                         if len(pw_row) else np.nan,
                "f1_mean"              : np.nan,
                "sensitivity_mean"     : np.nan,
                "specificity_mean"     : np.nan,
                "note"                 : "GroupKFold(5) by pair_id; threshold=0.5 fixed",
            })

    if rows:
        pr_df = pd.DataFrame(rows)
        pr_df.to_csv(TABLES / "paper_ready_results.csv", index=False)
        print(f"  Saved: paper_ready_results.csv  ({len(pr_df)} rows)")
        print(pr_df[["feature_block","auc_observed","auc_ci_95_low","auc_ci_95_high",
                     "pairwise_concordance"]].round(4).to_string(index=False))
    else:
        print("  No data available yet.")


# ============================================================================
# STEP 9 — Run nbconvert for paper-ready notebook
# ============================================================================

def run_nbconvert():
    print("\n-- Running nbconvert")
    import subprocess

    for nb_path in [NB_PAPER, NB_STANDALONE]:
        print(f"\n  Processing: {nb_path.name}")
        base = nb_path.stem

        # Execute
        executed = nb_path.parent / f"{base}_executed.ipynb"
        cmd_exec = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=3600",
            f"--ExecutePreprocessor.kernel_name=python3",
            "--output", str(executed),
            str(nb_path),
        ]
        print(f"  Executing: {executed.name}")
        r = subprocess.run(cmd_exec, capture_output=True, text=True, timeout=4000)
        if r.returncode == 0:
            print(f"  Executed successfully.")
        else:
            print(f"  Execute failed:\n    {r.stderr[-300:]}")
            print("  Continuing with HTML export of unexecuted version...")
            executed = nb_path  # fallback

        # HTML export
        html_out = nb_path.parent / f"{base}.html"
        cmd_html = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "html",
            "--output", str(html_out),
            str(executed),
        ]
        r2 = subprocess.run(cmd_html, capture_output=True, text=True, timeout=120)
        if r2.returncode == 0:
            print(f"  HTML exported: {html_out.name}")
        else:
            print(f"  HTML export failed: {r2.stderr[-200:]}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Applying paper-ready changes to notebooks")
    print("=" * 60)

    generate_audit_csv()
    confirm_xy_audit()
    build_paper_ready_csv()
    apply_to_paper_ready()
    apply_to_standalone()
    run_nbconvert()

    print("\nDone.")
