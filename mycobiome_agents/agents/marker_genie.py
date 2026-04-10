"""
Agent 4 · MarkerGenie
──────────────────────
Uses LangChain tool-calling to:
  1. Query PubMed (Entrez E-utilities) for each top fungus × cancer pair
  2. Call GPT-4o-mini to extract structured evidence from abstracts
  3. Build a literature evidence table in the state
"""
from __future__ import annotations
import json, re, time
import requests
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mycobiome_agents.state import PipelineState


# ── Tools (LangChain @tool decorated functions) ───────────────────────────

@tool
def pubmed_search(query: str, max_results: int = 5) -> list[str]:
    """Search PubMed and return up to max_results abstracts for the query."""
    base_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    base_fetch  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    try:
        r = requests.get(base_search, params={
            "db": "pubmed", "term": query, "retmax": max_results,
            "retmode": "json", "sort": "relevance",
        }, timeout=15)
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return ["No PubMed results found."]
        r2 = requests.get(base_fetch, params={
            "db": "pubmed", "id": ",".join(ids),
            "rettype": "abstract", "retmode": "text",
        }, timeout=20)
        parts = [p.strip() for p in r2.text.split("\n\n") if len(p.strip()) > 80]
        return parts[:max_results]
    except Exception as e:
        return [f"PubMed error: {e}"]


@tool
def extract_biomarker_evidence(
    fungus: str,
    cancer: str,
    abstracts: list[str],
) -> dict:
    """
    Use an LLM to extract structured evidence about a fungus-cancer association
    from PubMed abstracts.
    Returns a dict with keys: evidence_strength, direction, mechanism,
    key_finding, confidence.
    """
    # This tool is called by the agent's LLM, not directly
    return {}


# ── Core GPT-4 extraction logic ───────────────────────────────────────────

def _gpt_extract(client, fungus: str, cancer: str, abstracts: list[str]) -> dict:
    abstract_text = "\n\n---\n\n".join(abstracts[:3])[:3500]
    prompt = f"""You are a cancer bioinformatics expert. Analyze these PubMed abstracts
about {fungus} in {cancer}.

ABSTRACTS:
{abstract_text}

Return ONLY a JSON object (no markdown, no extra text):
{{
  "evidence_strength": "strong|moderate|weak|none",
  "direction": "deleterious|protective|unclear",
  "mechanism": "<one sentence — biological mechanism>",
  "key_finding": "<one sentence — most relevant result>",
  "confidence": <0.0-1.0>
}}"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "evidence_strength": "error",
            "direction": "unclear",
            "mechanism": str(e),
            "key_finding": "API extraction failed",
            "confidence": 0.0,
        }


# ─── LangGraph node ───────────────────────────────────────────────────────

def marker_genie_node(state: PipelineState) -> PipelineState:
    log = ["🔴 MarkerGenie starting — PubMed + GPT-4 evidence extraction..."]

    # Build validation pairs from paper findings + our ML top features
    PAPER_PAIRS = [
        ("Candida",     "colorectal cancer"),
        ("Candida",     "gastric cancer"),
        ("Malassezia",  "breast cancer"),
        ("Aspergillus", "lung cancer"),
        ("Blastomyces", "lung cancer"),
        ("Saccharomyces", "gastrointestinal cancer"),
    ]

    # Add top ML genera
    ml_pairs = []
    for genus in (state.get("top_biomarkers") or [])[:5]:
        clean = genus.replace("g__", "").split(";")[-1].strip().capitalize()
        if len(clean) > 3 and (clean, "cancer") not in PAPER_PAIRS:
            ml_pairs.append((clean, "cancer"))

    all_pairs = PAPER_PAIRS + ml_pairs
    log.append(f"  Validating {len(all_pairs)} fungus × cancer pairs")

    # Check for OpenAI key
    import os
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-..."):
        log.append("  ⚠  OPENAI_API_KEY not set — skipping GPT-4 extraction")
        log.append("     Set os.environ['OPENAI_API_KEY'] = 'sk-...' before running")
        message = AIMessage(content="\n".join(log), name="MarkerGenie")
        return {
            **state,
            "messages":           state["messages"] + [message.dict()],
            "next_agent":         "report_genie",
            "status":             {**state.get("status", {}), "marker_genie": "skipped"},
            "literature_evidence": [],
        }

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    results = []
    for fungus, cancer in all_pairs[:8]:   # cap at 8 for rate limits
        log.append(f"  Querying: {fungus} × {cancer} ...", )
        query     = f"{fungus} {cancer} tumor microbiome mycobiome"
        abstracts = pubmed_search.invoke({"query": query, "max_results": 5})
        evidence  = _gpt_extract(client, fungus, cancer, abstracts)
        evidence.update({
            "fungus":       fungus,
            "cancer":       cancer,
            "n_abstracts":  len(abstracts),
        })
        results.append(evidence)
        log[-1] += f" [{evidence.get('evidence_strength','?')}]"
        time.sleep(0.5)

    log.append(f"\n  ✅ MarkerGenie complete — {len(results)} pairs validated")

    message = AIMessage(content="\n".join(log), name="MarkerGenie")

    return {
        **state,
        "messages":            state["messages"] + [message.dict()],
        "next_agent":          "report_genie",
        "status":              {**state.get("status", {}), "marker_genie": "complete"},
        "literature_evidence": results,
    }
