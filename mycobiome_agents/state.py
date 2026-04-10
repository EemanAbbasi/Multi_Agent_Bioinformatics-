"""
Shared pipeline state for the mycobiome multi-agent system.
Every agent reads from and writes to this TypedDict.
LangGraph passes this state between nodes automatically.
"""
from __future__ import annotations
from typing import TypedDict, Annotated, Optional
import operator
import pandas as pd


class PipelineState(TypedDict):
    # ── Conversation / orchestration ─────────────────────────────────────
    messages: Annotated[list[dict], operator.add]   # full message history
    next_agent: str                                  # which agent runs next
    status: dict[str, str]                          # per-agent status log

    # ── Agent 1 · DataGenie outputs ──────────────────────────────────────
    counts_raw_path: Optional[str]
    counts_corrected_path: Optional[str]
    metadata_path: Optional[str]
    taxonomy_path: Optional[str]
    bacteria_raw_path: Optional[str]
    bacteria_corrected_path: Optional[str]
    n_samples: Optional[int]
    n_fungi: Optional[int]
    cancer_types: Optional[list[str]]
    ct_col: Optional[str]                           # detected cancer-type column name

    # ── Agent 2 · OmicsGenie outputs ─────────────────────────────────────
    mycotypes: Optional[dict]                       # {sample_id: "F1"|"F2"|"F3"}
    alpha_diversity: Optional[dict]                 # {sample_id: float}
    pca_variance_explained: Optional[float]
    top_variable_genera: Optional[list[str]]

    # ── Agent 3A · MLGenie classifier outputs ────────────────────────────
    auroc_per_cancer: Optional[dict[str, float]]    # {cancer_type: auroc}
    mean_auroc: Optional[float]
    top_biomarkers: Optional[list[str]]             # ranked genera by importance

    # ── Agent 3B · MLGenie survival outputs ──────────────────────────────
    survival_available: Optional[bool]
    logrank_pvalue: Optional[float]
    cox_hazard_ratio: Optional[float]

    # ── Agent 3C · MLGenie synergy outputs ───────────────────────────────
    auc_fungi_only: Optional[float]
    auc_bacteria_only: Optional[float]
    auc_combined: Optional[float]
    synergy_delta: Optional[float]

    # ── Agent 4 · MarkerGenie outputs ────────────────────────────────────
    literature_evidence: Optional[list[dict]]       # [{fungus, cancer, evidence, mechanism}]

    # ── Agent 5 · ReportGenie outputs ────────────────────────────────────
    figure_paths: Optional[list[str]]
    report_path: Optional[str]
    biomarker_table_path: Optional[str]
