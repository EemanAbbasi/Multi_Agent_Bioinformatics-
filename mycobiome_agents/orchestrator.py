"""
Orchestrator — LangGraph StateGraph
─────────────────────────────────────
langgraph.json points to this file:
    "mycobiome_pipeline": "./mycobiome_agents/orchestrator.py:graph"

The `graph` variable at module level is what LangGraph Studio loads.
"""
from __future__ import annotations
import sys, os

# Make the package importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langgraph.graph import StateGraph, END
from mycobiome_agents.state import PipelineState
from mycobiome_agents.agents.data_genie   import data_genie_node
from mycobiome_agents.agents.omics_genie  import omics_genie_node
from mycobiome_agents.agents.ml_genie     import (
    ml_genie_classifier_node,
    ml_genie_survival_node,
    ml_genie_synergy_node,
)
from mycobiome_agents.agents.marker_genie  import marker_genie_node
from mycobiome_agents.agents.report_genie  import report_genie_node


def _route(state: PipelineState) -> str:
    """Read state['next_agent'] and return the next node name."""
    return state.get("next_agent", "END")


_DESTINATIONS = {
    "omics_genie":         "omics_genie",
    "ml_genie_classifier": "ml_genie_classifier",
    "ml_genie_survival":   "ml_genie_survival",
    "ml_genie_synergy":    "ml_genie_synergy",
    "marker_genie":        "marker_genie",
    "report_genie":        "report_genie",
    "END":                 END,
}


def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("data_genie",           data_genie_node)
    g.add_node("omics_genie",          omics_genie_node)
    g.add_node("ml_genie_classifier",  ml_genie_classifier_node)
    g.add_node("ml_genie_survival",    ml_genie_survival_node)
    g.add_node("ml_genie_synergy",     ml_genie_synergy_node)
    g.add_node("marker_genie",         marker_genie_node)
    g.add_node("report_genie",         report_genie_node)

    g.set_entry_point("data_genie")

    # Every agent node needs conditional edges — including data_genie
    ALL_AGENT_NODES = [
        "data_genie",
        "omics_genie",
        "ml_genie_classifier",
        "ml_genie_survival",
        "ml_genie_synergy",
        "marker_genie",
        "report_genie",
    ]
    for src in ALL_AGENT_NODES:
        g.add_conditional_edges(src, _route, _DESTINATIONS)

    return g.compile()


# ── Module-level `graph` — this is what langgraph.json imports ───────────
graph = build_graph()


# ── Convenience runner for terminal use ──────────────────────────────────
def run_pipeline(openai_api_key: str = "") -> PipelineState:
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    initial: PipelineState = {
        "messages":   [{"role": "user",
                        "content": "Run the full mycobiome biomarker discovery pipeline."}],
        "next_agent": "data_genie",
        "status":     {},
        "counts_raw_path": None, "counts_corrected_path": None,
        "metadata_path": None,  "taxonomy_path": None,
        "bacteria_raw_path": None, "bacteria_corrected_path": None,
        "n_samples": None, "n_fungi": None,
        "cancer_types": None, "ct_col": None,
        "mycotypes": None, "alpha_diversity": None,
        "pca_variance_explained": None, "top_variable_genera": None,
        "auroc_per_cancer": None, "mean_auroc": None, "top_biomarkers": None,
        "survival_available": None, "logrank_pvalue": None, "cox_hazard_ratio": None,
        "auc_fungi_only": None, "auc_bacteria_only": None,
        "auc_combined": None, "synergy_delta": None,
        "literature_evidence": None,
        "figure_paths": None, "report_path": None, "biomarker_table_path": None,
    }

    print("=" * 60)
    print("🍄  MYCOBIOME MULTI-AGENT PIPELINE  (LangGraph)")
    print("=" * 60)
    final = graph.invoke(initial)
    print("\n✅  PIPELINE COMPLETE")
    for agent, status in (final.get("status") or {}).items():
        print(f"   {agent:<26} {status}")
    return final


if __name__ == "__main__":
    run_pipeline()
