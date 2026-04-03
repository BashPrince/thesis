#!/usr/bin/env python3
"""
Mean precision-recall curves per augmentation method.

Downloads prediction artifacts, computes per-run PR curves, interpolates onto
a common recall grid, and plots mean precision ± SE per method.

Usage:
    python experiment_analysis/pr_curves/plot_pr_curves.py <group> [options]

Examples:
    python experiment_analysis/pr_curves/plot_pr_curves.py v7_poolfilter
    python experiment_analysis/pr_curves/plot_pr_curves.py v7_poolfilter_extend --methods embed none unfiltered
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add analysis dir to path
THESIS_CODE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(THESIS_CODE / "analysis"))

from analyze_predictions import (
    fetch_predictions,
    logits_to_probs,
)

try:
    from sklearn.metrics import precision_recall_curve, average_precision_score
except ImportError:
    print("ERROR: scikit-learn not installed.", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed.", file=sys.stderr)
    sys.exit(1)


METHOD_STYLES = {
    "none": {"color": "#888888", "ls": "--", "label": "none (baseline)"},
    "real": {"color": "#009e73", "ls": "-", "label": "real"},
    "embed": {"color": "#0072b2", "ls": "-", "label": "embed"},
    "tfidf": {"color": "#56b4e9", "ls": "-", "label": "tfidf"},
    "unfiltered": {"color": "#e69f00", "ls": "-", "label": "unfiltered"},
    "free": {"color": "#cc79a7", "ls": "-", "label": "free"},
    "genetic": {"color": "#d55e00", "ls": "-", "label": "genetic"},
    "embed-multi": {"color": "#d55e00", "ls": "-", "label": "embed-multi"},
}

METHOD_ORDER = ["real", "embed", "embed-multi", "tfidf", "unfiltered", "none", "free", "genetic"]


def interpolate_pr_curve(labels, probs, recall_grid):
    """Compute PR curve and interpolate precision onto a common recall grid.

    Returns interpolated precision array aligned with recall_grid.
    Precision is interpolated using the "step" convention: at each recall
    level, precision is the maximum precision at any recall >= that level.
    """
    precision, recall, _ = precision_recall_curve(labels, probs)

    # PR curves go from high recall to low recall; flip for interpolation
    # Remove duplicates in recall for interpolation stability
    recall_sorted = recall[::-1]
    precision_sorted = precision[::-1]

    # Make precision monotonically decreasing with recall (standard PR interpolation)
    for i in range(len(precision_sorted) - 2, -1, -1):
        precision_sorted[i] = max(precision_sorted[i], precision_sorted[i + 1])

    interp_precision = np.interp(recall_grid, recall_sorted, precision_sorted,
                                  left=1.0, right=0.0)
    return interp_precision


def compute_mean_pr_curves(pred_data, recall_grid, methods=None):
    """Compute mean PR curves per method.

    Averages over seeds per sequence, then computes mean ± SE across sequences.

    Returns:
        dict[aug] -> {"mean": array, "se": array, "ap_mean": float, "ap_se": float}
    """
    if methods is None:
        methods = sorted(pred_data.keys(), key=str)

    results = {}
    for aug in methods:
        if aug not in pred_data:
            continue

        # Per-sequence mean PR curve (averaged over seeds)
        seq_curves = {}
        seq_aps = {}
        for seq in sorted(pred_data[aug].keys()):
            seed_curves = []
            seed_aps = []
            for preds, labels in pred_data[aug][seq]:
                probs = logits_to_probs(preds)
                interp_p = interpolate_pr_curve(labels, probs, recall_grid)
                seed_curves.append(interp_p)
                seed_aps.append(average_precision_score(labels, probs))
            seq_curves[seq] = np.mean(seed_curves, axis=0)
            seq_aps[seq] = np.mean(seed_aps)

        all_curves = np.array(list(seq_curves.values()))
        all_aps = np.array(list(seq_aps.values()))
        n = len(all_curves)

        results[aug] = {
            "mean": all_curves.mean(axis=0),
            "se": all_curves.std(axis=0, ddof=1) / np.sqrt(n),
            "ap_mean": all_aps.mean(),
            "ap_se": all_aps.std(ddof=1) / np.sqrt(n),
        }

    return results


def plot_pr_curves(results, recall_grid, title=None, output_path=None):
    """Plot mean PR curves with SE bands."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot in method order
    ordered = [m for m in METHOD_ORDER if m in results]
    # Add any methods not in the default order
    ordered += [m for m in results if m not in ordered]

    for aug in ordered:
        r = results[aug]
        style = METHOD_STYLES.get(aug, {"color": "gray", "ls": "-", "label": str(aug)})
        label = f"{style['label']} (AP={r['ap_mean']:.3f})"

        ax.plot(recall_grid, r["mean"], ls=style["ls"], color=style["color"],
                label=label, linewidth=1.8)
        ax.fill_between(recall_grid, r["mean"] - r["se"], r["mean"] + r["se"],
                        alpha=0.15, color=style["color"])

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot mean PR curves per augmentation method")
    parser.add_argument("group", help="WandB run group name")
    parser.add_argument("--entity", default="redstag")
    parser.add_argument("--project", default="thesis")
    parser.add_argument("--artifact-split", default="test")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/thesis_preds"))
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to include (default: all)")
    parser.add_argument("--n-points", type=int, default=200,
                        help="Number of recall grid points (default: 200)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) / args.group
    cache_dir.mkdir(parents=True, exist_ok=True)

    pred_data, _ = fetch_predictions(
        args.group, args.entity, args.project,
        args.artifact_split, cache_dir, expected_seeds=3,
        max_workers=args.workers,
    )
    if not pred_data:
        print("No prediction data found.", file=sys.stderr)
        sys.exit(1)

    recall_grid = np.linspace(0, 1, args.n_points)

    print("Computing mean PR curves...")
    results = compute_mean_pr_curves(pred_data, recall_grid, methods=args.methods)

    for aug in results:
        r = results[aug]
        print(f"  {aug:>12}: AP = {r['ap_mean']:.4f} +/- {r['ap_se']:.4f}")

    output = args.output
    if output is None:
        output = str(Path(__file__).resolve().parent / f"{args.group}_pr_curves.pdf")

    title = args.title or f"Mean PR curves ({args.group})"
    plot_pr_curves(results, recall_grid, title=title, output_path=output)


if __name__ == "__main__":
    main()
