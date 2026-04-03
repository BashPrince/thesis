#!/usr/bin/env python3
"""
Plot F1 vs base rate for augmentation methods using saved prediction artifacts.

Resamples the test set to a range of positive rates and computes optimal-threshold
F1 at each rate, producing one curve per augmentation method. Overlays the actual
measured F1 at the empirical base rate as markers.

Usage:
    python figures/f1_base_rate.py <group> [options]

Examples:
    python figures/f1_base_rate.py v7_poolfilter_extend
    python figures/f1_base_rate.py v7_poolfilter_extend_eval --output figures/f1_base_rate_eval.pdf
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Add analysis dir to path so we can reuse its infrastructure
THESIS_CODE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(THESIS_CODE / "analysis"))

from analyze_predictions import (
    fetch_predictions,
    logits_to_probs,
    optimal_threshold_f1,
    f1_at_threshold,
    resample_to_positive_rate,
)


def compute_f1_at_base_rates(pred_data, base_rates, n_resamples=10, seed=42,
                             fixed_threshold=None):
    """For each method, compute mean F1 at each base rate.

    If fixed_threshold is None, uses optimal-threshold F1.
    If fixed_threshold is a float, uses F1 at that threshold.

    Averages over seeds within a sequence, then over sequences.
    Resamples n_resamples times per run to reduce resampling noise.

    Returns:
        dict[aug] -> {
            "mean": array of mean F1 per base rate,
            "se": array of standard error per base rate,
        }
    """
    rng = np.random.default_rng(seed)
    results = {}

    for aug in sorted(pred_data.keys(), key=str):
        # seq_f1s[rate_idx][seq] -> mean F1 across seeds and resamples
        seq_f1s = {i: {} for i in range(len(base_rates))}

        for seq in sorted(pred_data[aug].keys()):
            for rate_idx, rate in enumerate(base_rates):
                seed_vals = []
                for preds, labels in pred_data[aug][seq]:
                    probs = logits_to_probs(preds)
                    resample_vals = []
                    for _ in range(n_resamples):
                        rl, rp = resample_to_positive_rate(labels, probs, rate, rng)
                        if fixed_threshold is not None:
                            f1 = f1_at_threshold(rl, rp, fixed_threshold)
                        else:
                            f1, _ = optimal_threshold_f1(rl, rp)
                        resample_vals.append(f1)
                    seed_vals.append(np.mean(resample_vals))
                seq_f1s[rate_idx][seq] = np.mean(seed_vals)

        # Aggregate across sequences
        means = []
        ses = []
        for rate_idx in range(len(base_rates)):
            vals = np.array(list(seq_f1s[rate_idx].values()))
            means.append(vals.mean())
            ses.append(vals.std(ddof=1) / np.sqrt(len(vals)))

        results[aug] = {"mean": np.array(means), "se": np.array(ses)}

    return results


def plot_f1_base_rate(results, base_rates, title=None, output_path=None,
                      ylabel="Optimal-threshold F1"):
    """Plot F1 vs base rate curves.

    Args:
        results: dict from compute_f1_at_base_rates
        base_rates: array of base rates
        title: plot title
        output_path: if given, save figure here
        ylabel: y-axis label
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    method_styles = {
        "none": {"color": "#888888", "ls": "--", "label": "none (baseline)"},
        "unfiltered": {"color": "#e69f00", "ls": "-", "label": "unfiltered"},
        "embed": {"color": "#0072b2", "ls": "-", "label": "embed"},
        "embed-multi": {"color": "#d55e00", "ls": "-", "label": "embed-multi"},
        "real": {"color": "#009e73", "ls": "-", "label": "real"},
    }

    for aug in results:
        style = method_styles.get(aug, {"color": "gray", "ls": "-", "label": str(aug)})
        m = results[aug]["mean"]
        se = results[aug]["se"]
        ax.plot(base_rates * 100, m, ls=style["ls"], color=style["color"],
                label=style["label"], linewidth=1.8)
        ax.fill_between(base_rates * 100, m - se, m + se,
                        alpha=0.15, color=style["color"])

    ax.set_xlabel("Positive rate (%)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot F1 vs base rate from prediction artifacts")
    parser.add_argument("group", help="WandB run group name")
    parser.add_argument("--entity", default="redstag")
    parser.add_argument("--project", default="thesis")
    parser.add_argument("--baseline-aug", default="none")
    parser.add_argument("--artifact-split", default="test")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/thesis_preds"))
    parser.add_argument("--output", default=None, help="Output file path (default: figures/<group>_f1_base_rate.pdf)")
    parser.add_argument("--min-rate", type=float, default=0.05)
    parser.add_argument("--max-rate", type=float, default=0.50)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--n-resamples", type=int, default=20,
                        help="Resamples per run per base rate to reduce noise (default: 20)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--title", default=None)
    parser.add_argument("--fixed-threshold", type=float, default=None,
                        help="Use F1 at this fixed threshold instead of optimal-threshold F1")
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

    base_rates = np.linspace(args.min_rate, args.max_rate, args.n_steps)

    thr_label = f"F1 at threshold {args.fixed_threshold}" if args.fixed_threshold else "optimal-threshold F1"
    print(f"Computing {thr_label} at {args.n_steps} base rates "
          f"({args.min_rate:.0%}–{args.max_rate:.0%}), "
          f"{args.n_resamples} resamples per run...")
    results = compute_f1_at_base_rates(pred_data, base_rates,
                                       n_resamples=args.n_resamples,
                                       fixed_threshold=args.fixed_threshold)

    output = args.output
    if output is None:
        suffix = f"_thr{args.fixed_threshold:.1f}" if args.fixed_threshold else ""
        output = str(Path(__file__).resolve().parent / f"{args.group}_f1_base_rate{suffix}.pdf")

    ylabel = f"F1 (threshold={args.fixed_threshold})" if args.fixed_threshold else "Optimal-threshold F1"
    plot_f1_base_rate(results, base_rates, title=args.title,
                      output_path=output, ylabel=ylabel)


if __name__ == "__main__":
    main()
