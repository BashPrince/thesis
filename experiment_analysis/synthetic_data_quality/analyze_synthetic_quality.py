#!/usr/bin/env python3
"""
Synthetic data quality characterization.

Downloads training data artifacts from WandB for each (sequence, augmentation method)
combination, separates real from synthetic samples by diffing against the baseline
(none) artifact, then computes:
  1. Class balance of the synthetic pool
  2. Embedding similarity between synthetic and real samples (per class)
  3. Lexical diversity (type-token ratio)
  4. Sentence length statistics

Usage:
    python experiment_analysis/synthetic_data_quality/analyze_synthetic_quality.py \
        --group v7_poolfilter --seqs 5
"""

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed.", file=sys.stderr)
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Artifact discovery and download
# ---------------------------------------------------------------------------

def discover_artifacts(group, entity="redstag", project="thesis"):
    """Find one dataset artifact per (seq, aug) from a WandB group.

    Returns dict[(seq, aug)] -> artifact_name.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    seen = set()
    art_map = {}
    for run in runs:
        if run.state != "finished":
            continue
        try:
            seq = int(run.name.split("seq_")[1].split("_")[0])
            aug = run.name.split("aug_")[1].split("_seed")[0]
        except (IndexError, ValueError):
            continue
        key = (seq, aug)
        if key in seen:
            continue
        seen.add(key)
        arts = list(run.used_artifacts())
        if arts:
            art_map[key] = arts[0].name
    return art_map


def download_artifacts(art_map, cache_dir, entity="redstag", project="thesis",
                       max_workers=16):
    """Download all artifacts in parallel. Returns dict[(seq, aug)] -> DataFrame."""
    api = wandb.Api()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _download(key, art_name):
        dest = cache_dir / art_name.replace(":", "_")
        art = api.artifact(f"{entity}/{project}/{art_name}")
        art.download(str(dest))
        return key, pd.read_csv(dest / "train.csv")

    dfs = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download, k, v): k for k, v in art_map.items()}
        for f in as_completed(futures):
            key, df = f.result()
            dfs[key] = df
    return dfs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_ttr(texts):
    """Type-token ratio for a list of texts."""
    tokens = " ".join(texts).lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def analyze_sequence(seq, dfs, model, methods):
    """Analyze synthetic data quality for one sequence.

    Returns list of result dicts (one per method).
    """
    real_df = dfs[(seq, "none")]
    real_texts = set(real_df["Text"])

    # Encode real samples
    real_pos_texts = real_df[real_df["class_label"] == "Yes"]["Text"].tolist()
    real_neg_texts = real_df[real_df["class_label"] == "No"]["Text"].tolist()
    real_pos_emb = model.encode(real_pos_texts, normalize_embeddings=True)
    real_neg_emb = model.encode(real_neg_texts, normalize_embeddings=True)

    # Real-vs-real reference
    rp_rp = real_pos_emb @ real_pos_emb.T
    np.fill_diagonal(rp_rp, 0)
    n = rp_rp.shape[0]
    rp_rp_mean = rp_rp.sum() / (n * (n - 1))
    rp_rn_mean = float((real_pos_emb @ real_neg_emb.T).mean())

    results = []
    for method in methods:
        if (seq, method) not in dfs:
            continue
        df = dfs[(seq, method)]
        synth = df[~df["Text"].isin(real_texts)]

        if len(synth) == 0:
            continue

        synth_pos = synth[synth["class_label"] == "Yes"]["Text"].tolist()
        synth_neg = synth[synth["class_label"] == "No"]["Text"].tolist()

        sp_emb = model.encode(synth_pos, normalize_embeddings=True)
        sn_emb = model.encode(synth_neg, normalize_embeddings=True)

        sp_rp = sp_emb @ real_pos_emb.T
        sp_rn = sp_emb @ real_neg_emb.T
        sn_rn = sn_emb @ real_neg_emb.T
        sn_rp = sn_emb @ real_pos_emb.T

        results.append({
            "seq": seq,
            "method": method,
            "n_synth": len(synth),
            "n_synth_pos": len(synth_pos),
            "n_synth_neg": len(synth_neg),
            "n_real": len(real_df),
            # Embedding similarity
            "sp_rp": float(sp_rp.mean()),
            "sp_rn": float(sp_rn.mean()),
            "sn_rn": float(sn_rn.mean()),
            "sn_rp": float(sn_rp.mean()),
            "pos_gap": float(sp_rp.mean() - sp_rn.mean()),
            "neg_gap": float(sn_rn.mean() - sn_rp.mean()),
            "sp_rp_max": float(sp_rp.max(axis=1).mean()),
            "sn_rn_max": float(sn_rn.max(axis=1).mean()),
            # Per-sample max similarities (for distribution plots)
            "sp_rp_max_all": sp_rp.max(axis=1).tolist(),
            "sn_rn_max_all": sn_rn.max(axis=1).tolist(),
            # Reference
            "rp_rp_ref": float(rp_rp_mean),
            "rp_rn_ref": float(rp_rn_mean),
            # Lexical
            "synth_ttr": compute_ttr(synth["Text"].tolist()),
            "synth_ttr_pos": compute_ttr(synth_pos),
            "synth_ttr_neg": compute_ttr(synth_neg),
            "real_ttr": compute_ttr(real_df["Text"].tolist()),
            "synth_avg_len": float(synth["Text"].str.split().str.len().mean()),
            "synth_std_len": float(synth["Text"].str.split().str.len().std()),
            "real_avg_len": float(real_df["Text"].str.split().str.len().mean()),
            "real_std_len": float(real_df["Text"].str.split().str.len().std()),
        })

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_results(rdf):
    """Print aggregated results."""
    methods = rdf["method"].unique()

    def fmt(series):
        m = series.mean()
        se = series.std() / np.sqrt(len(series))
        return f"{m:.3f}+/-{se:.3f}"

    print("--- Embedding Similarity (mean +/- SE across sequences) ---")
    print(f"{'':>12} | {'SP-RP':>12} | {'SP-RP max':>12} | {'SN-RN':>12} "
          f"| {'pos_gap':>12} | {'neg_gap':>12}")
    print("-" * 80)
    for method in methods:
        sub = rdf[rdf["method"] == method]
        print(f"{method:>12} | {fmt(sub['sp_rp']):>12} | {fmt(sub['sp_rp_max']):>12} "
              f"| {fmt(sub['sn_rn']):>12} | {fmt(sub['pos_gap']):>12} "
              f"| {fmt(sub['neg_gap']):>12}")

    ref_rp = rdf.groupby("seq")["rp_rp_ref"].first()
    ref_rn = rdf.groupby("seq")["rp_rn_ref"].first()
    print(f"  real-real ref: RP-RP={ref_rp.mean():.3f}+/-{ref_rp.std()/np.sqrt(len(ref_rp)):.3f}"
          f"  RP-RN={ref_rn.mean():.3f}+/-{ref_rn.std()/np.sqrt(len(ref_rn)):.3f}")

    print()
    print("--- Lexical Metrics (mean +/- SE) ---")
    print(f"{'':>12} | {'Synth TTR':>12} | {'Real TTR':>12} | {'Synth avg len':>14}")
    print("-" * 60)
    for method in methods:
        sub = rdf[rdf["method"] == method]
        print(f"{method:>12} | {fmt(sub['synth_ttr']):>12} | {fmt(sub['real_ttr']):>12} "
              f"| {fmt(sub['synth_avg_len']):>14}")

    print()
    print("--- Class Balance ---")
    print(f"{'':>12} | {'Synth pos%':>12} | {'n_synth':>8}")
    print("-" * 40)
    for method in methods:
        sub = rdf[rdf["method"] == method]
        pos_pct = (sub["n_synth_pos"] / sub["n_synth"]).mean()
        n = int(sub["n_synth"].mean())
        print(f"{method:>12} | {pos_pct:>11.1%} | {n:>8}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

METHOD_COLORS = {
    "embed": "#0072b2",
    "tfidf": "#56b4e9",
    "unfiltered": "#e69f00",
    "free": "#cc79a7",
    "genetic": "#d55e00",
}
METHOD_ORDER = ["embed", "tfidf", "unfiltered", "free", "genetic"]

# F1 gains from v7_poolfilter analysis (Δ vs none, 5 sequences)
F1_GAINS = {
    "embed": 0.030,
    "tfidf": 0.021,
    "unfiltered": 0.021,
    "free": 0.001,
    "genetic": -0.040,
}


def _method_agg(rdf, col):
    """Return (mean, se) per method for a column."""
    out = {}
    for m in METHOD_ORDER:
        sub = rdf[rdf["method"] == m][col]
        if len(sub) == 0:
            continue
        out[m] = (sub.mean(), sub.std() / np.sqrt(len(sub)))
    return out


def plot_heatmap(rdf, output_dir):
    """Fig 1: Embedding similarity heatmap (methods × similarity type)."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    cols = ["sp_rp", "sp_rn", "sn_rp", "sn_rn"]
    col_labels = ["Synth+ → Real+", "Synth+ → Real−",
                   "Synth− → Real+", "Synth− → Real−"]
    methods = [m for m in METHOD_ORDER if m in rdf["method"].values]

    data = np.zeros((len(methods), len(cols)))
    for i, m in enumerate(methods):
        sub = rdf[rdf["method"] == m]
        for j, c in enumerate(cols):
            data[i, j] = sub[c].mean()

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.05, vmax=0.20)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=25, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)

    for i in range(len(methods)):
        for j in range(len(cols)):
            ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if data[i, j] > 0.14 else "black")

    fig.colorbar(im, ax=ax, label="Mean cosine similarity", shrink=0.8)
    ax.set_title("Embedding similarity: synthetic vs real samples", fontsize=11)
    fig.tight_layout()
    path = output_dir / "synth_similarity_heatmap.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_similarity_violins(rdf, output_dir):
    """Fig 2: Violin plot of per-sample max cosine similarity to same-class real."""
    methods = [m for m in METHOD_ORDER if m in rdf["method"].values]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, class_label, col in zip(axes,
                                     ["Positives (check-worthy)", "Negatives (not check-worthy)"],
                                     ["sp_rp_max_all", "sn_rn_max_all"]):
        all_data = []
        positions = []
        colors = []
        for i, m in enumerate(methods):
            sub = rdf[rdf["method"] == m]
            # Pool all per-sample values across sequences
            vals = []
            for _, row in sub.iterrows():
                vals.extend(row[col])
            all_data.append(vals)
            positions.append(i)
            colors.append(METHOD_COLORS.get(m, "gray"))

        parts = ax.violinplot(all_data, positions=positions, showmeans=True,
                              showextrema=False)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmeans"].set_color("black")

        ax.set_xticks(positions)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_title(class_label, fontsize=10)
        ax.set_ylabel("Max cosine sim to same-class real" if ax == axes[0] else "")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Per-sample similarity to nearest real same-class sample", fontsize=11)
    fig.tight_layout()
    path = output_dir / "synth_similarity_violins.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_class_separation(rdf, output_dir):
    """Fig 3: Class separation scatter (same-class vs cross-class similarity)."""
    fig, ax = plt.subplots(figsize=(5, 5))

    methods = [m for m in METHOD_ORDER if m in rdf["method"].values]
    for m in methods:
        sub = rdf[rdf["method"] == m]
        x = sub["sp_rp"].mean()
        y = sub["sp_rn"].mean()
        xe = sub["sp_rp"].std() / np.sqrt(len(sub))
        ye = sub["sp_rn"].std() / np.sqrt(len(sub))
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", markersize=10,
                    color=METHOD_COLORS.get(m, "gray"), label=m,
                    capsize=3, capthick=1.2, zorder=5)

    # Diagonal line (no class separation)
    lims = [0.08, 0.20]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Reference point for real-real
    ref_rp = rdf.groupby("seq")["rp_rp_ref"].first().mean()
    ref_rn = rdf.groupby("seq")["rp_rn_ref"].first().mean()
    ax.plot(ref_rp, ref_rn, "s", markersize=10, color="black", label="real-real ref",
            zorder=5, markerfacecolor="none", markeredgewidth=1.5)

    ax.set_xlabel("Similarity to same class (SP-RP)", fontsize=10)
    ax.set_ylabel("Similarity to opposite class (SP-RN)", fontsize=10)
    ax.set_title("Class separation of synthetic positives", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "synth_class_separation.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_lexical_bars(rdf, output_dir):
    """Fig 4: TTR and sentence length bar chart (synthetic vs real reference line)."""
    methods = [m for m in METHOD_ORDER if m in rdf["method"].values]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Real reference values (same across methods)
    real_ttr_mean = rdf.groupby("seq")["real_ttr"].first().mean()
    real_ttr_se = rdf.groupby("seq")["real_ttr"].first().std() / np.sqrt(5)
    real_len_mean = rdf.groupby("seq")["real_avg_len"].first().mean()
    real_len_se = rdf.groupby("seq")["real_avg_len"].first().std() / np.sqrt(5)

    # TTR
    ax = axes[0]
    x = np.arange(len(methods))
    synth_ttr = [rdf[rdf["method"] == m]["synth_ttr"].mean() for m in methods]
    synth_ttr_se = [rdf[rdf["method"] == m]["synth_ttr"].std() / np.sqrt(5) for m in methods]

    ax.bar(x, synth_ttr, yerr=synth_ttr_se, capsize=3,
           color=[METHOD_COLORS.get(m, "gray") for m in methods], alpha=0.8)
    ax.axhline(real_ttr_mean, color="black", ls="--", linewidth=1.2, label="Real (128 samples)")
    ax.axhspan(real_ttr_mean - real_ttr_se, real_ttr_mean + real_ttr_se,
               alpha=0.15, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Type-token ratio", fontsize=10)
    ax.set_title("Lexical diversity", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Sentence length
    ax = axes[1]
    synth_len = [rdf[rdf["method"] == m]["synth_avg_len"].mean() for m in methods]
    synth_len_se = [rdf[rdf["method"] == m]["synth_avg_len"].std() / np.sqrt(5) for m in methods]

    ax.bar(x, synth_len, yerr=synth_len_se, capsize=3,
           color=[METHOD_COLORS.get(m, "gray") for m in methods], alpha=0.8)
    ax.axhline(real_len_mean, color="black", ls="--", linewidth=1.2, label="Real (128 samples)")
    ax.axhspan(real_len_mean - real_len_se, real_len_mean + real_len_se,
               alpha=0.15, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Mean sentence length (words)", fontsize=10)
    ax.set_title("Sentence length", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = output_dir / "synth_lexical_comparison.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_gap_vs_f1(rdf, output_dir):
    """Fig 5: Class separation gap vs F1 gain scatter."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    methods = [m for m in METHOD_ORDER if m in rdf["method"].values and m in F1_GAINS]
    for m in methods:
        sub = rdf[rdf["method"] == m]
        gap = sub["pos_gap"].mean()
        gap_se = sub["pos_gap"].std() / np.sqrt(len(sub))
        f1 = F1_GAINS[m]
        ax.errorbar(gap, f1, xerr=gap_se, fmt="o", markersize=11,
                    color=METHOD_COLORS.get(m, "gray"), capsize=3, capthick=1.2,
                    zorder=5, label=m)

    # Correlation
    gaps = np.array([rdf[rdf["method"] == m]["pos_gap"].mean() for m in methods])
    f1s = np.array([F1_GAINS[m] for m in methods])
    from scipy import stats
    r, p = stats.pearsonr(gaps, f1s)
    ax.text(0.05, 0.95, f"r = {r:.2f}, p = {p:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.axhline(0, color="gray", ls="--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Class separation gap (pos_gap)", fontsize=10)
    ax.set_ylabel("F1 gain over baseline", fontsize=10)
    ax.set_title("Synthetic data quality vs downstream performance", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "synth_gap_vs_f1.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def generate_plots(rdf, output_dir):
    """Generate all figures."""
    if not HAS_MPL:
        print("matplotlib not installed — skipping plots.", file=sys.stderr)
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmap(rdf, output_dir)
    plot_similarity_violins(rdf, output_dir)
    plot_class_separation(rdf, output_dir)
    plot_lexical_bars(rdf, output_dir)
    plot_gap_vs_f1(rdf, output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic data quality analysis")
    parser.add_argument("--group", default="v7_poolfilter",
                        help="WandB group name (default: v7_poolfilter)")
    parser.add_argument("--entity", default="redstag")
    parser.add_argument("--project", default="thesis")
    parser.add_argument("--methods", nargs="+",
                        default=["embed", "tfidf", "unfiltered", "free", "genetic"],
                        help="Augmentation methods to analyze")
    parser.add_argument("--cache-dir",
                        default=str(Path.home() / ".cache/thesis_preds/artifacts"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--fig-dir", default=None,
                        help="Output directory for figures (default: same dir as script)")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    # 1. Discover and download artifacts
    print(f"Discovering artifacts for group: {args.group}")
    art_map = discover_artifacts(args.group, args.entity, args.project)
    print(f"Found {len(art_map)} (seq, aug) combinations")

    print(f"Downloading artifacts...")
    dfs = download_artifacts(art_map, args.cache_dir, args.entity, args.project,
                             max_workers=args.workers)
    print(f"Downloaded {len(dfs)} artifacts")

    # 2. Load embedding model
    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)

    # 3. Analyze each sequence
    seqs = sorted(set(k[0] for k in dfs if k[1] == "none"))
    all_results = []
    for seq in seqs:
        results = analyze_sequence(seq, dfs, model, args.methods)
        all_results.extend(results)
        print(f"  Seq {seq} done")

    rdf = pd.DataFrame(all_results)

    # 4. Print results
    print()
    print(f"=== Synthetic Data Quality ({args.group}, {len(seqs)} sequences) ===")
    print()
    print_results(rdf)

    # 5. Generate figures
    if not args.no_plots:
        fig_dir = Path(args.fig_dir) if args.fig_dir else Path(__file__).resolve().parent
        print()
        generate_plots(rdf, fig_dir)


if __name__ == "__main__":
    main()
