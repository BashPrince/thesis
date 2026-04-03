#!/usr/bin/env python3
"""
Sequence-level covariate analysis for v7_poolfilter_extend.

Downloads the baseline (none) training artifacts for each sequence, computes
embedding-space and lexical covariates, and correlates them with baseline F1
and augmentation gain.

Usage:
    python experiment_analysis/sequence_covariates/analyze_sequence_covariates.py
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
# Baseline F1 and embed F1 per sequence (v7_poolfilter_extend, mean over seeds)
# ---------------------------------------------------------------------------

BASELINE_F1 = {
    0: 0.6055, 1: 0.5672, 2: 0.6838, 3: 0.6865, 4: 0.6022,
    5: 0.6909, 6: 0.6206, 7: 0.5978, 8: 0.6748, 9: 0.5781,
    10: 0.6451, 11: 0.6344, 12: 0.6772, 13: 0.6087, 14: 0.5706,
}
EMBED_F1 = {
    0: 0.6508, 1: 0.6204, 2: 0.6760, 3: 0.6737, 4: 0.6607,
    5: 0.7106, 6: 0.6374, 7: 0.6274, 8: 0.6739, 9: 0.6309,
    10: 0.6591, 11: 0.6134, 12: 0.6584, 13: 0.6606, 14: 0.6140,
}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def download_artifacts(n_seqs, group, entity, project, cache_dir, max_workers=16):
    """Download none artifacts for all sequences. Returns dict[seq] -> (train_df, test_df)."""
    api = wandb.Api()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Discover artifact names
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    art_names = {}
    seen = set()
    for run in runs:
        if run.state != "finished" or "aug_none" not in run.name:
            continue
        try:
            seq = int(run.name.split("seq_")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        if seq in seen:
            continue
        seen.add(seq)
        arts = list(run.used_artifacts())
        if arts:
            art_names[seq] = arts[0].name

    def _download(seq, art_name):
        dest = cache_dir / art_name.replace(":", "_")
        art = api.artifact(f"{entity}/{project}/{art_name}")
        art.download(str(dest))
        train = pd.read_csv(dest / "train.csv")
        test = pd.read_csv(dest / "test.csv")
        return seq, train, test

    print(f"Downloading {len(art_names)} artifacts...")
    data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download, s, a): s for s, a in art_names.items()}
        for f in as_completed(futures):
            seq, train, test = f.result()
            data[seq] = (train, test)
    print("Done.")
    return data


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_covariates(data, model):
    """Compute covariates for each sequence.

    Args:
        data: dict[seq] -> (train_df, test_df)
        model: SentenceTransformer

    Returns:
        DataFrame with one row per sequence.
    """
    # Encode test set once (same across sequences)
    _, test_df = next(iter(data.values()))
    test_emb = model.encode(test_df["Text"].tolist(), normalize_embeddings=True)
    test_labels = (test_df["class_label"] == "Yes").values
    test_pos_emb = test_emb[test_labels]
    test_neg_emb = test_emb[~test_labels]

    results = []
    for seq in sorted(data.keys()):
        train_df, _ = data[seq]
        texts = train_df["Text"].tolist()
        labels = (train_df["class_label"] == "Yes").values

        emb = model.encode(texts, normalize_embeddings=True)
        pos_emb = emb[labels]
        neg_emb = emb[~labels]

        # Mean pairwise similarity
        sim_all = emb @ emb.T
        np.fill_diagonal(sim_all, 0)
        n = len(emb)
        mean_sim = sim_all.sum() / (n * (n - 1))

        # Embedding spread
        emb_spread = emb.std(axis=0).mean()
        pos_spread = pos_emb.std(axis=0).mean()
        neg_spread = neg_emb.std(axis=0).mean()

        # Class separability
        pp = pos_emb @ pos_emb.T
        np.fill_diagonal(pp, 0)
        np_ = pos_emb.shape[0]
        pp_mean = pp.sum() / (np_ * (np_ - 1))
        pn_mean = (pos_emb @ neg_emb.T).mean()
        class_sep = pp_mean - pn_mean

        # Test overlap
        train_pos_test_pos = pos_emb @ test_pos_emb.T
        train_neg_test_neg = neg_emb @ test_neg_emb.T
        test_overlap_pos = train_pos_test_pos.max(axis=1).mean()
        test_overlap_neg = train_neg_test_neg.max(axis=1).mean()
        test_overlap = (test_overlap_pos + test_overlap_neg) / 2
        test_sim = (train_pos_test_pos.mean() + train_neg_test_neg.mean()) / 2

        # Sentence length
        lens = train_df["Text"].str.split().str.len()

        results.append({
            "seq": seq,
            "baseline_f1": BASELINE_F1.get(seq, float("nan")),
            "embed_gain": EMBED_F1.get(seq, float("nan")) - BASELINE_F1.get(seq, float("nan")),
            "mean_sim": float(mean_sim),
            "emb_spread": float(emb_spread),
            "pos_spread": float(pos_spread),
            "neg_spread": float(neg_spread),
            "class_sep": float(class_sep),
            "test_overlap": float(test_overlap),
            "test_overlap_pos": float(test_overlap_pos),
            "test_overlap_neg": float(test_overlap_neg),
            "test_sim": float(test_sim),
            "mean_len": float(lens.mean()),
            "std_len": float(lens.std()),
        })
        print(f"  Seq {seq} done")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

COVARIATES = [
    ("mean_sim", "Mean pairwise sim (lower=more diverse)"),
    ("emb_spread", "Embedding spread"),
    ("pos_spread", "Positive class spread"),
    ("neg_spread", "Negative class spread"),
    ("class_sep", "Class separability"),
    ("test_overlap", "Test overlap (max sim)"),
    ("test_sim", "Test similarity (mean)"),
    ("mean_len", "Mean sentence length"),
    ("std_len", "Sentence length std"),
]


def print_correlations(rdf):
    """Print correlation tables."""
    for target, label in [("baseline_f1", "Baseline F1"), ("embed_gain", "Embed Gain")]:
        print(f"\n=== Correlation with {label} ({len(rdf)} sequences) ===")
        print(f"{'Covariate':>20} | {'r':>6} | {'p':>6}")
        print("-" * 40)
        rows = []
        for col, desc in COVARIATES:
            r, p = stats.pearsonr(rdf[col], rdf[target])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            rows.append((abs(r), col, r, p, sig))
        rows.sort(reverse=True)
        for _, col, r, p, sig in rows:
            print(f"{col:>20} | {r:>+6.3f} | {p:>6.3f} {sig}")

    print("\n=== Per-Sequence Data ===")
    cols = ["seq", "baseline_f1", "embed_gain", "class_sep", "test_overlap",
            "test_sim", "mean_sim"]
    print(rdf[cols].sort_values("baseline_f1").to_string(
        index=False, float_format="{:.4f}".format))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sequence-level covariate analysis")
    parser.add_argument("--group", default="v7_poolfilter_extend")
    parser.add_argument("--entity", default="redstag")
    parser.add_argument("--project", default="thesis")
    parser.add_argument("--cache-dir",
                        default=str(Path.home() / ".cache/thesis_preds/artifacts"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    data = download_artifacts(
        n_seqs=15, group=args.group, entity=args.entity, project=args.project,
        cache_dir=args.cache_dir, max_workers=args.workers)

    print(f"Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)

    print("Computing covariates...")
    rdf = compute_covariates(data, model)

    print_correlations(rdf)


if __name__ == "__main__":
    main()
