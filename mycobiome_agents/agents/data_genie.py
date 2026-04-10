"""
Agent 1 · DataGenie
────────────────────
Responsibilities:
  - Download all Final_files from knightlab-analyses/mycobiome
  - Validate files and detect schema (cancer-type column, sample count)
  - Update PipelineState with file paths and dataset summary
  - Report back to the orchestrator with a structured message
"""
from __future__ import annotations
import requests, os
from pathlib import Path
from langchain_core.messages import AIMessage
from mycobiome_agents.state import PipelineState

BASE_URL = (
    "https://raw.githubusercontent.com/knightlab-analyses/"
    "mycobiome/master/Final_files"
)

CORE_FILES = {
    "counts_raw":         "count_data_fungi_decontaminated_raw.tsv",
    "counts_corrected": "count_data_fungi_decontaminated_voom_snm_corrected.tsv",
    "metadata":           "metadata_fungi_14495samples.tsv",
    "taxonomy":           "taxonomy_table_rep200.tsv",
}

OPTIONAL_FILES = {
    "bacteria_raw":       "count_data_bacteria_WIS_overlapping_raw.tsv",
    "bacteria_corrected": "count_data_bacteria_WIS_overlapping_voom_snm_corrected.tsv",
}

DATA_DIR = Path(__file__).parent.parent / "final_files_data"
DATA_DIR.mkdir(exist_ok=True)


def _download(filename: str) -> tuple[Path, bool]:
    dest = DATA_DIR / filename
    if dest.exists() and dest.stat().st_size > 1000:
        return dest, True
    url = f"{BASE_URL}/{filename}"
    try:
        r = requests.get(url, timeout=180, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        return dest, True
    except Exception as e:
        return dest, False


def _detect_ct_col(meta_path: str) -> tuple[str | None, list[str], int]:
    """Return (cancer_type_col, cancer_type_list, n_samples)."""
    import pandas as pd
    meta = pd.read_csv(meta_path, sep="\t", index_col=0, nrows=5000)
    candidates = ["disease", "cancer_type", "tumor_type", "cohort", "TCGA_project"]
    ct_col = next(
        (c for c in candidates if c in meta.columns),
        next(
            (c for c in meta.columns
             if meta[c].dtype == object and 2 < meta[c].nunique() < 45),
            None,
        ),
    )
    cancer_types = sorted(meta[ct_col].dropna().unique().tolist()) if ct_col else []
    # Full row count
    full = pd.read_csv(meta_path, sep="\t", index_col=0, usecols=[0])
    return ct_col, cancer_types, len(full)


# ─── LangGraph node ───────────────────────────────────────────────────────
def data_genie_node(state: PipelineState) -> PipelineState:
    """Download Final_files, validate, update state."""

    log = ["🟢 DataGenie starting..."]
    paths: dict[str, str] = {}
    ok:    dict[str, bool] = {}

    # Download core files
    for key, fname in CORE_FILES.items():
        p, success = _download(fname)
        paths[key] = str(p)
        ok[key]    = success
        status     = f"✓ {fname}  ({p.stat().st_size/1e6:.1f} MB)" if success else f"✗ FAILED: {fname}"
        log.append(f"  {status}")

    # Download optional bacterial files
    for key, fname in OPTIONAL_FILES.items():
        p, success = _download(fname)
        paths[key] = str(p)
        ok[key]    = success
        log.append(f"  {'✓' if success else '⚠ optional missing'} {fname}")

    # Detect schema from metadata
    ct_col, cancer_types, n_samples = None, [], 0
    if ok.get("metadata"):
        try:
            ct_col, cancer_types, n_samples = _detect_ct_col(paths["metadata"])
        except Exception as e:
            log.append(f"  ⚠ schema detection failed: {e}")

    # Count fungal genera
    n_fungi = 0
    if ok.get("counts_corrected"):
        import pandas as pd
        hdr = pd.read_csv(paths["counts_corrected"], sep="\t", index_col=0, nrows=0)
        n_fungi = len(hdr.columns)

    summary = (
        f"DataGenie complete — {n_samples:,} samples · "
        f"{n_fungi} fungal genera · {len(cancer_types)} cancer types"
    )
    log.append(f"\n  {summary}")

    message = AIMessage(
        content="\n".join(log),
        name="DataGenie",
    )

    return {
        **state,
        "messages":              state["messages"] + [message.dict()],
        "next_agent":            "omics_genie",
        "status":                {**state.get("status", {}), "data_genie": "complete"},
        "counts_raw_path":       paths.get("counts_raw"),
        "counts_corrected_path": paths.get("counts_corrected"),
        "metadata_path":         paths.get("metadata"),
        "taxonomy_path":         paths.get("taxonomy"),
        "bacteria_raw_path":     paths.get("bacteria_raw")       if ok.get("bacteria_raw")       else None,
        "bacteria_corrected_path": paths.get("bacteria_corrected") if ok.get("bacteria_corrected") else None,
        "n_samples":             n_samples,
        "n_fungi":               n_fungi,
        "cancer_types":          cancer_types,
        "ct_col":                ct_col,
    }
