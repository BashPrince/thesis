#!/usr/bin/env python3
"""
Scatter plot of per-dataset baseline F1 vs augmentation gain.

For each (method, test set), plots one point per sub-sampled dataset:
    x = baseline F1 on that dataset (seed-averaged, 'none' condition)
    y = augmentation gain (method F1 minus baseline F1, seed-averaged)

Overlays an OLS regression line per method and annotates the Pearson r.

Usage:
    python experiment_analysis/gain_baseline/plot_gain_baseline.py \
        --groups v7_poolfilter_extend_ct24_eval v7_poolfilter_extend_eval \
        --labels "CT24 test (26%)" "CT24 holdout (13%)" \
        --methods embed unfiltered \
        --output figures/gain_baseline_confirmatory.pdf
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    print("ERROR: matplotlib not installed.", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
    from scipy.stats import pearsonr
    import wandb
except ImportError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)


METHOD_STYLES = {
    "embed":       {"color": "#0072b2", "marker": "o", "label": "embed"},
    "unfiltered":  {"color": "#e69f00", "marker": "s", "label": "unfiltered"},
    "tfidf":       {"color": "#56b4e9", "marker": "^", "label": "tfidf"},
    "free":        {"color": "#cc79a7", "marker": "v", "label": "free"},
    "genetic":     {"color": "#d55e00", "marker": "D", "label": "genetic"},
    "real":        {"color": "#009e73", "marker": "P", "label": "real"},
    "embed-multi": {"color": "#9467bd", "marker": "X", "label": "embed-multi"},
    "embed-multi-gunel": {"color": "#9467bd", "marker": "X", "label": "embed-multi"},
}


def load_group(group, entity="redstag", project="thesis", exclude_augs=()):
    """Return DataFrame with columns: seq, aug, f1 (one row per finished run)."""
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}))
    rows = []
    for r in runs:
        if r.state != "finished":
            continue
        m = re.match(r"seq_(\d+)_aug_([^_]+(?:-[^_]+)*)_seed", r.name)
        if not m:
            continue
        seq = int(m.group(1))
        aug = m.group(2)
        if aug in exclude_augs:
            continue
        f1 = r.summary.get("test/f1")
        if f1 is None:
            continue
        rows.append({"seq": seq, "aug": aug, "f1": f1})
    return pd.DataFrame(rows)


def seq_table(df):
    """Seed-average: one F1 per (seq, aug)."""
    return df.groupby(["seq", "aug"])["f1"].mean().unstack()


def plot_panel(ax, seq_df, methods, title):
    base = seq_df["none"]
    handles = []
    for method in methods:
        if method not in seq_df.columns:
            continue
        style = METHOD_STYLES.get(method, {"color": "gray", "marker": "o", "label": method})
        gain = seq_df[method] - base
        r, p = pearsonr(base, gain)

        ax.scatter(base, gain, color=style["color"], marker=style["marker"],
                   s=55, alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)

        # OLS fit
        slope, intercept = np.polyfit(base, gain, 1)
        x = np.linspace(base.min(), base.max(), 50)
        ax.plot(x, slope * x + intercept, color=style["color"], linewidth=1.6,
                alpha=0.75, zorder=2)

        handles.append(Line2D([0], [0], color=style["color"],
                              marker=style["marker"], linestyle="-",
                              markersize=7, markeredgecolor="white",
                              markeredgewidth=0.8,
                              label=f"{style['label']} ($r={r:+.2f}$, $p={p:.3f}$)"))

    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xlabel("Baseline F1 (seed-averaged)", fontsize=10)
    ax.set_ylabel("Augmentation gain (F1)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=handles, fontsize=8, loc="upper right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", required=True,
                        help="WandB group names (one panel per group)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Panel titles (default: group names)")
    parser.add_argument("--methods", nargs="+", default=["embed", "unfiltered"])
    parser.add_argument("--exclude-augs", nargs="+",
                        default=["embed-multi", "embed-multi-gunel"])
    parser.add_argument("--entity", default="redstag")
    parser.add_argument("--project", default="thesis")
    parser.add_argument("--output", default=None,
                        help="Output PDF path")
    args = parser.parse_args()

    labels = args.labels or args.groups
    if len(labels) != len(args.groups):
        print("--labels must match --groups length", file=sys.stderr)
        sys.exit(1)

    n = len(args.groups)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, group, label in zip(axes, args.groups, labels):
        print(f"Loading {group}...")
        df = load_group(group, args.entity, args.project, tuple(args.exclude_augs))
        sdf = seq_table(df)
        plot_panel(ax, sdf, args.methods, label)

    fig.tight_layout()

    out = args.output
    if out is None:
        out = str(Path(__file__).resolve().parent / "gain_baseline.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
