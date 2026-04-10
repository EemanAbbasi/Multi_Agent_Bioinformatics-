"""
Agent 2 · OmicsGenie
─────────────────────
Responsibilities:
  - Load Voom-SNM corrected table + taxonomy
  - Compute alpha diversity (Shannon) on raw counts
  - Run Aitchison PCA (CLR transform) for beta diversity
  - Cluster samples into mycotypes F1 / F2 / F3 (K-means k=3)
  - Save figures, update PipelineState
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from langchain_core.messages import AIMessage
from mycobiome_agents.state import PipelineState

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

MT_COLORS = {"F1": "#1D9E75", "F2": "#7F77DD", "F3": "#D85A30"}

# ── Helpers ───────────────────────────────────────────────────────────────

def _clr(df: pd.DataFrame, pseudo: float = 0.5) -> pd.DataFrame:
    X = df.values.astype(float) + pseudo
    lx = np.log(X)
    return pd.DataFrame(lx - lx.mean(axis=1, keepdims=True),
                        index=df.index, columns=df.columns)


def _shannon(row: pd.Series) -> float:
    c = row[row > 0]
    if len(c) == 0:
        return 0.0
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)))


def _genus_name(col: str) -> str:
    """Extract clean genus name from taxonomy string."""
    name = col.split(";")[-1].replace("g__", "").strip()
    return name if name else col[:25]


def _name_columns(df: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Rename feature IDs to genus names using taxonomy table."""
    mapping = {}
    genus_cols = [c for c in taxonomy.columns
                  if any(k in c.lower() for k in ["genus", "level_6", "6"])]
    for fid in df.columns:
        if fid in taxonomy.index and genus_cols:
            val = str(taxonomy.loc[fid, genus_cols[0]]).replace("g__", "").strip()
            mapping[fid] = val if val and val != "nan" else _genus_name(fid)
        else:
            mapping[fid] = _genus_name(fid)
    renamed = df.rename(columns=mapping)
    # Deduplicate
    seen: dict[str, int] = {}
    new_cols = []
    for c in renamed.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    renamed.columns = new_cols
    return renamed


def _assign_mycotype_names(km: KMeans, columns: list[str]) -> dict[int, str]:
    """Map cluster indices to F1/F2/F3 based on dominant genera."""
    centroid_df = pd.DataFrame(km.cluster_centers_, columns=columns)
    MARKERS = {
        "F1": ["malassezia", "ramularia", "trichosporon"],
        "F2": ["aspergillus", "candida"],
        "F3": [],
    }
    assigned: dict[int, str] = {}
    used = set()
    for k in range(3):
        top_genera = centroid_df.iloc[k].sort_values(ascending=False).index[:8]
        top_lower  = [g.lower() for g in top_genera]
        match = None
        for label, markers in MARKERS.items():
            if label in used:
                continue
            if any(m in g for m in markers for g in top_lower):
                match = label
                break
        if match is None:
            match = next(f for f in ["F1", "F2", "F3"] if f not in used)
        assigned[k] = match
        used.add(match)
    return assigned


# ─── LangGraph node ───────────────────────────────────────────────────────

def omics_genie_node(state: PipelineState) -> PipelineState:
    log = ["🟣 OmicsGenie starting..."]

    # ── Load data ─────────────────────────────────────────────────────────
    counts_raw = pd.read_csv(state["counts_raw_path"],       sep="\t", index_col=0)
    counts_vsn = pd.read_csv(state["counts_corrected_path"], sep="\t", index_col=0)
    metadata   = pd.read_csv(state["metadata_path"],          sep="\t", index_col=0)
    taxonomy   = pd.read_csv(state["taxonomy_path"],          sep="\t", index_col=0)

    shared = counts_raw.index.intersection(counts_vsn.index).intersection(metadata.index)
    counts_raw = counts_raw.loc[shared]
    counts_vsn = counts_vsn.loc[shared]
    metadata   = metadata.loc[shared]

    ct_col = state["ct_col"]
    log.append(f"  Loaded {len(shared):,} samples · {counts_vsn.shape[1]} genera")

    # ── Rename columns to genus names ─────────────────────────────────────
    counts_raw_g = _name_columns(counts_raw, taxonomy)
    counts_vsn_g = _name_columns(counts_vsn, taxonomy)
    log.append(f"  Taxonomy mapped — sample genera: {list(counts_vsn_g.columns[:4])}")

    # ── Alpha diversity ───────────────────────────────────────────────────
    alpha = counts_raw_g.apply(_shannon, axis=1)
    metadata["shannon"] = alpha
    log.append(f"  Alpha diversity — mean Shannon: {alpha.mean():.3f}")

    top_cancers = metadata[ct_col].value_counts().head(12).index.tolist()
    meta_top = metadata[metadata[ct_col].isin(top_cancers)]
    order = meta_top.groupby(ct_col)["shannon"].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(data=meta_top, x=ct_col, y="shannon", order=order,
                palette="tab10", linewidth=0.7, fliersize=1.5, ax=ax)
    ax.set_xlabel("Cancer type"); ax.set_ylabel("Shannon entropy")
    ax.set_title("Mycobiome alpha diversity across TCGA cancer types\n"
                 "(224 decontaminated fungal genera · raw counts)")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    p_alpha = str(FIGURE_DIR / "02_alpha_diversity.png")
    plt.savefig(p_alpha, dpi=150, bbox_inches="tight"); plt.close()
    log.append("  ✓ Alpha diversity figure saved")

    # ── CLR + PCA ─────────────────────────────────────────────────────────
    X_clr = _clr(counts_vsn_g)
    pca   = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_clr.values)
    var_exp = float(pca.explained_variance_ratio_[:2].sum())
    cancer_labels = metadata.loc[X_clr.index, ct_col].values
    uniq = sorted(set(cancer_labels))
    pal  = plt.cm.tab20(np.linspace(0, 1, len(uniq)))
    cmap_c = {c: pal[i] for i, c in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(11, 8))
    for ct in uniq:
        m = cancer_labels == ct
        ax.scatter(coords[m, 0], coords[m, 1], color=cmap_c[ct],
                   s=7, alpha=0.55, label=ct, rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Aitchison PCA — Voom-SNM corrected fungal genera\n"
                 f"({len(shared):,} TCGA samples · {counts_vsn.shape[1]} genera)")
    ax.legend(fontsize=6, ncol=3, markerscale=2,
              bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.4)
    plt.tight_layout()
    p_pca = str(FIGURE_DIR / "03_pcoa_beta_diversity.png")
    plt.savefig(p_pca, dpi=150, bbox_inches="tight"); plt.close()
    log.append(f"  ✓ PCA done — PC1+PC2 = {var_exp*100:.1f}% variance")

    # ── Mycotype clustering ───────────────────────────────────────────────
    Xs = StandardScaler().fit_transform(X_clr.values)
    km = KMeans(n_clusters=3, n_init=30, random_state=42)
    raw_labels = km.fit_predict(Xs)
    cluster_map = _assign_mycotype_names(km, list(X_clr.columns))
    mycotype_series = pd.Series(
        [cluster_map[l] for l in raw_labels],
        index=X_clr.index
    )
    metadata.loc[X_clr.index, "mycotype"] = mycotype_series

    counts_mt = mycotype_series.value_counts().to_dict()
    log.append(f"  ✓ Mycotypes: F1={counts_mt.get('F1',0):,}  "
               f"F2={counts_mt.get('F2',0):,}  F3={counts_mt.get('F3',0):,}")

    # t-SNE visualisation (subsample for speed)
    n_tsne = min(3000, len(Xs))
    rng    = np.random.RandomState(42)
    idx    = rng.choice(len(Xs), n_tsne, replace=False)
    tsne   = TSNE(n_components=2, perplexity=40, n_iter=500, random_state=42)
    tc     = tsne.fit_transform(Xs[idx])
    mt_sub = [mycotype_series.iloc[i] for i in idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for mt, col in MT_COLORS.items():
        m = np.array(mt_sub) == mt
        ax1.scatter(tc[m, 0], tc[m, 1], color=col, s=7, alpha=0.6,
                    label=f"{mt} (n={m.sum():,})", rasterized=True)
    ax1.set_title("t-SNE: Mycotype clusters (k=3)\nF1=Malassezia · F2=Aspergillus/Candida · F3=multi")
    ax1.legend(title="Mycotype", fontsize=9, markerscale=2)
    ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")

    mt_ct = pd.crosstab(
        metadata.loc[X_clr.index, ct_col],
        metadata.loc[X_clr.index, "mycotype"],
        normalize="index"
    ).reindex(columns=["F1", "F2", "F3"], fill_value=0)
    mt_ct_top = mt_ct.loc[mt_ct.index.isin(top_cancers)]
    mt_ct_top.plot(kind="bar", stacked=True, ax=ax2,
                   color=[MT_COLORS[c] for c in ["F1", "F2", "F3"]],
                   edgecolor="white", linewidth=0.4)
    ax2.set_title("Mycotype composition per cancer type")
    ax2.set_ylabel("Proportion"); ax2.set_xlabel("Cancer type")
    ax2.legend(title="Mycotype", fontsize=9)
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    p_mt = str(FIGURE_DIR / "04_mycotypes.png")
    plt.savefig(p_mt, dpi=150, bbox_inches="tight"); plt.close()
    log.append("  ✓ Mycotype figure saved")

    # Top variable genera
    top_var = X_clr.var(axis=0).sort_values(ascending=False).head(20).index.tolist()

    summary = (
        f"OmicsGenie complete — alpha diversity computed, "
        f"PCA ({var_exp*100:.1f}% var), mycotypes F1/F2/F3 assigned"
    )
    log.append(f"\n  {summary}")

    message = AIMessage(content="\n".join(log), name="OmicsGenie")

    return {
        **state,
        "messages":              state["messages"] + [message.dict()],
        "next_agent":            "ml_genie_classifier",
        "status":                {**state.get("status", {}), "omics_genie": "complete"},
        "mycotypes":             mycotype_series.to_dict(),
        "alpha_diversity":       alpha.to_dict(),
        "pca_variance_explained": var_exp,
        "top_variable_genera":   top_var,
        "figure_paths":          (state.get("figure_paths") or []) + [p_alpha, p_pca, p_mt],
    }
