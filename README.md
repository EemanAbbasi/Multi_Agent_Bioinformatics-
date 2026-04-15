# Multi-Agent Bioinformatics Pipeline

A production-grade multi-agent system for pan-cancer mycobiome biomarker discovery, built with **LangGraph**. Seven autonomous agents orchestrate an end-to-end workflow, from raw TCGA data ingestion through ML-based biomarker discovery, LLM-powered literature validation, and automated report synthesis.

> **Reference:** Narunsky-Haziza, Sepich-Poore, Livyatan et al. (2022). *Pan-cancer analyses reveal cancer-type-specific fungal ecologies and bacteriome interactions.* **Cell.** [doi:10.1016/j.cell.2022.09.005](https://doi.org/10.1016/j.cell.2022.09.005)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     LangGraph StateGraph                 │
│                                                         │
│  ┌───────────┐   ┌───────────┐   ┌───────────────────┐ │
│  │ DataGenie │──▶│OmicsGenie │──▶│ MLGenie Classifier│ │
│  │  Agent 1  │   │  Agent 2  │   │     Agent 3A      │ │
│  └───────────┘   └───────────┘   └────────┬──────────┘ │
│                                           │             │
│                            ┌──────────────▼──────────┐  │
│                            │  MLGenie Survival        │  │
│                            │     Agent 3B             │  │
│                            └──────────────┬──────────┘  │
│                                           │             │
│                            ┌──────────────▼──────────┐  │
│                            │  MLGenie Synergy         │  │
│                            │     Agent 3C             │  │
│                            └──────────────┬──────────┘  │
│                                           │             │
│                            ┌──────────────▼──────────┐  │
│                            │   MarkerGenie            │  │
│                            │     Agent 4              │  │
│                            └──────────────┬──────────┘  │
│                                           │             │
│                            ┌──────────────▼──────────┐  │
│                            │   ReportGenie            │  │
│                            │     Agent 5              │  │
│                            └─────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

Each agent is an autonomous LangGraph node that reads from and writes to a shared typed `PipelineState`. Routing is driven entirely by `state["next_agent"]` — agents decide where the pipeline goes next.

---

## Agents

| Agent | File | Responsibility |
|-------|------|----------------|
| **DataGenie** | `agents/data_genie.py` | Downloads TCGA `Final_files` from GitHub, detects metadata schema, validates file integrity |
| **OmicsGenie** | `agents/omics_genie.py` | CLR normalization, Shannon alpha diversity, Aitchison PCA, K-means mycotype clustering (F1/F2/F3) |
| **MLGenie 3A** | `agents/ml_genie.py` | Random Forest pan-cancer classifier, one-vs-all AUROC across 30+ cancer types, 5-fold CV |
| **MLGenie 3B** | `agents/ml_genie.py` | Kaplan-Meier survival curves + Cox PH model,  Candida abundance vs. GI cancer prognosis |
| **MLGenie 3C** | `agents/ml_genie.py` | Multi-domain synergy, fungi-only vs. bacteria-only vs. combined AUROC comparison |
| **MarkerGenie** | `agents/marker_genie.py` | PubMed query via Entrez E-utilities + GPT-4o-mini structured evidence extraction |
| **ReportGenie** | `agents/report_genie.py` | Heatmaps, 6-panel summary figure, biomarker evidence CSV, Markdown report |

---

## Dataset

All data comes from the official `Final_files/` release of [knightlab-analyses/mycobiome](https://github.com/knightlab-analyses/mycobiome):

| File | Description |
|------|-------------|
| `count_data_fungi_decontaminated_raw.tsv` | Raw counts — 224 decontaminated fungal genera × 14,495 samples |
| `count_data_fungi_decontaminated_voom_snm_corrected.tsv.zip` | Voom-SNM batch-corrected counts — used for all ML |
| `metadata_fungi_14495samples.tsv` | TCGA clinical metadata — cancer type, tissue, batch |
| `taxonomy_table_rep200.tsv` | Taxonomy table mapping feature IDs to genus names |

Data files are excluded from this repository via `.gitignore`. DataGenie downloads them automatically on first run.

---

## Project Structure

```
mycobiome_studio/
├── langgraph.json                  # LangGraph Studio config
├── pyproject.toml                  # Package definition + dependencies
├── .env                            # API keys (not committed)
└── mycobiome_agents/
    ├── state.py                    # Shared PipelineState TypedDict
    ├── orchestrator.py             # StateGraph wiring + module-level graph
    ├── agents/
    │   ├── data_genie.py
    │   ├── omics_genie.py
    │   ├── ml_genie.py
    │   ├── marker_genie.py
    │   └── report_genie.py
    └── tools/
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/EemanAbbasi/Multi_Agent_Bioinformatics-.git
cd Multi_Agent_Bioinformatics-

pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your OpenAI API key to .env:
# OPENAI_API_KEY=sk-...
```

> MarkerGenie (Agent 4) requires an OpenAI API key. All other agents run without one.

### 3. Run via LangGraph Studio

```bash
pip install "langgraph-cli[inmem]"
langgraph dev --port 8080 --debug-port 5678
```

Then open [http://localhost:8080](http://localhost:8080) in Chrome. Use `--tunnel` flag for Safari.

### 4. Run via Python

```python
from mycobiome_agents.orchestrator import run_pipeline

final_state = run_pipeline(openai_api_key="sk-...")
```

---

## Key Design Decisions

**Typed shared state over direct agent coupling**
All agents communicate exclusively through `PipelineState` — a TypedDict defining every field each agent can read or write. No agent imports another agent directly. This makes individual agents independently testable and swappable.

**File paths in state, not DataFrames**
Large DataFrames (14,000+ rows) are never stored in the state. Agents store file paths and reload data as needed. This keeps checkpoints lightweight and allows agents to run in separate processes.

**Conditional routing via `next_agent`**
Each agent sets `state["next_agent"]` before returning. The orchestrator's router reads this field to decide the next node. This means routing logic lives in the agents themselves — an agent can conditionally skip downstream steps based on its results.

**Voom-SNM corrected data for ML, raw counts for diversity**
Following the original paper's methodology: Voom-SNM batch-corrected data is used for all machine learning tasks; raw counts are used for alpha diversity (Shannon entropy requires compositional proportions).

---

## Outputs

After a full run, the `figures/` directory contains:

| File | Description |
|------|-------------|
| `01_sample_distribution.png` | TCGA samples per cancer type |
| `02_alpha_diversity.png` | Shannon entropy by cancer type |
| `03_pcoa_beta_diversity.png` | Aitchison PCA of CLR-normalized genera |
| `04_mycotypes.png` | t-SNE + mycotype cluster composition |
| `05_auroc_per_cancer.png` | One-vs-all AUROC per cancer type |
| `06_feature_importance.png` | Top fungal genera by RF importance |
| `07_survival_candida.png` | Kaplan-Meier: Candida × GI cancer survival |
| `08_multi_domain_synergy.png` | Fungi vs. bacteria vs. combined AUROC |
| `09_fungi_heatmap.png` | Genus × cancer type abundance heatmap |
| `10_master_summary.png` | 6-panel publication-ready summary |
| `biomarker_evidence_table.csv` | Ranked biomarkers with literature evidence |
| `mycobiome_pipeline_report.md` | Full Markdown report with results |

---

## Requirements

- Python ≥ 3.10
- See `pyproject.toml` for full dependency list

Core dependencies:
```
langgraph>=0.2.0
langchain>=0.2.0
langchain-openai>=0.1.0
scikit-learn>=1.3.0
lifelines>=0.27.0
openai>=1.0.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

---

## License

MIT
