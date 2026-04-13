#!/usr/bin/env python3
"""
Precision-recall and threshold analysis using saved prediction artifacts.

Downloads predictions.npy / labels.npy artifacts from WandB for every run in a
group, then reports:
  1. Average Precision (AUC-PR) per aug method
  2. Optimal-threshold F1 per aug method (paired t-test vs baseline)
  3. Method ranking at a range of fixed thresholds
  4. Class-balance sensitivity: optimal-threshold F1 at resampled positive rates

Artifacts are cached locally to avoid re-downloading on repeated runs.

Usage:
    python analysis/analyze_predictions.py <group> [options]

Options:
    --entity          WandB entity (default: redstag)
    --project         WandB project (default: thesis)
    --baseline-aug    Aug level used as comparison baseline (default: none)
    --artifact-split  Prediction artifact prefix, e.g. "test" -> "test_predictions" (default: test)
    --cache-dir       Local directory for cached artifacts (default: ~/.cache/thesis_preds)
    --thresholds      Space-separated fixed thresholds to evaluate (default: 0.1 0.2 0.3 0.4 0.5)
    --positive-rates  Space-separated target positive rates for sensitivity analysis
                      (default: 0.13 0.25 0.50); use "none" to skip
    --expected-seeds  Expected seed runs per (seq, aug) combo (default: 3)
"""

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run: pip install wandb", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Artifact downloading / caching
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, run_id: str) -> Path:
    return cache_dir / run_id


def download_artifact(run, artifact_split: str, cache_dir: Path):
    """Download prediction artifact for a run; return (predictions, labels) arrays.

    predictions: (n_samples, 2) float32 raw logits
    labels:      (n_samples,)   int64

    Returns None if the artifact is not found.
    """
    dest = _cache_path(cache_dir, run.id)
    pred_file = dest / "predictions.npy"
    label_file = dest / "labels.npy"

    if pred_file.exists() and label_file.exists():
        return np.load(pred_file), np.load(label_file)

    artifact_name = f"{artifact_split}_predictions"
    artifacts = [a for a in run.logged_artifacts() if a.type == "prediction"
                 and a.name.startswith(artifact_name)]
    if not artifacts:
        return None

    dest.mkdir(parents=True, exist_ok=True)
    artifacts[0].download(str(dest))

    if not pred_file.exists() or not label_file.exists():
        return None

    return np.load(pred_file), np.load(label_file)


def _parse_run_name(name):
    """Parse seq and aug from a run name. Returns (seq, aug) or raises ValueError."""
    seq = int(name.split("seq_")[1].split("_")[0])
    aug_str = name.split("aug_")[1]
    if "_seed" in aug_str:
        aug_str = aug_str[:aug_str.index("_seed")]
    elif aug_str.endswith(tuple(f"seed{d}" for d in "0123456789")):
        aug_str = aug_str.rsplit("seed", 1)[0].rstrip("_")
    try:
        return seq, int(aug_str)
    except ValueError:
        return seq, aug_str


def _download_one(run, artifact_split, cache_dir):
    """Worker: parse name, download artifact. Returns (seq, aug, result, name)."""
    try:
        seq, aug = _parse_run_name(run.name)
    except (IndexError, ValueError):
        return None
    result = download_artifact(run, artifact_split, cache_dir)
    return seq, aug, result, run.name


def fetch_predictions(group, entity, project, artifact_split, cache_dir, expected_seeds,
                      max_workers=16):
    """Fetch all prediction artifacts for a group in parallel.

    Returns:
        pred_data: dict[aug][seq] -> list of (predictions, labels) tuples
        all_runs:  list of dicts with run metadata
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    runs = list(runs)

    pred_data = defaultdict(lambda: defaultdict(list))
    all_runs = []
    missing = []
    state_counts = defaultdict(int)

    finished_runs = []
    for run in runs:
        state_counts[run.state] += 1
        all_runs.append({"id": run.id, "name": run.name, "state": run.state})
        if run.state == "finished":
            finished_runs.append(run)

    total = len(runs)
    print(f"Group: {group}")
    print(f"Total runs: {total}  ({', '.join(f'{s}={n}' for s, n in sorted(state_counts.items()))})")
    print(f"Artifact split: {artifact_split}_predictions")
    print(f"Downloading {len(finished_runs)} artifacts (workers={max_workers})...", flush=True)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_one, run, artifact_split, cache_dir): run
            for run in finished_runs
        }
        for future in as_completed(futures):
            completed += 1
            print(f"  {completed}/{len(finished_runs)}", end="\r", flush=True)
            out = future.result()
            if out is None:
                continue
            seq, aug, result, name = out
            if result is None:
                missing.append(name)
            else:
                pred_data[aug][seq].append(result)

    print()  # newline after \r progress

    aug_levels = sorted(pred_data.keys())
    for aug in aug_levels:
        n_runs = sum(len(v) for v in pred_data[aug].values())
        print(f"  aug={aug:>12}: {len(pred_data[aug]):>3} sequences, {n_runs:>3} runs downloaded")

    if missing:
        print(f"\nWARNING: artifact missing for {len(missing)} run(s):")
        for name in missing:
            print(f"  {name}")

    return pred_data, all_runs


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def logits_to_probs(logits):
    """Convert (n, 2) logits to (n,) positive-class probabilities."""
    return softmax(logits, axis=1)[:, 1]


def optimal_threshold_f1(labels, probs):
    """Return (best_f1, best_threshold) maximising F1 over the PR curve."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    # precision_recall_curve appends a sentinel at the end; thresholds is one shorter
    f1s = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
        0.0,
    )
    idx = np.argmax(f1s)
    return float(f1s[idx]), float(thresholds[idx])


def f1_at_threshold(labels, probs, threshold):
    """F1 for the positive class at a fixed probability threshold."""
    preds = (probs >= threshold).astype(int)
    return f1_score(labels, preds, zero_division=0)


def resample_to_positive_rate(labels, probs, target_rate, rng):
    """Resample (labels, probs) to achieve target positive rate.

    Keeps all positives; downsamples or upsamples negatives accordingly.
    Returns (resampled_labels, resampled_probs).
    """
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = len(pos_idx)

    if n_pos == 0:
        return labels, probs

    # n_pos / (n_pos + n_neg_target) = target_rate
    n_neg_target = int(round(n_pos * (1 - target_rate) / target_rate))
    if n_neg_target <= 0:
        n_neg_target = 1

    replace = n_neg_target > len(neg_idx)
    neg_sample = rng.choice(neg_idx, size=n_neg_target, replace=replace)
    idx = np.concatenate([pos_idx, neg_sample])
    return labels[idx], probs[idx]


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------

def compute_run_metrics(pred_data, thresholds, positive_rates):
    """Compute per-run metrics for all aug/seq/seed combinations.

    Returns a DataFrame with one row per run:
        aug, seq, seed_idx, ap, opt_f1, opt_threshold,
        f1_at_{t} for each t in thresholds,
        opt_f1_at_rate_{r} and ap_at_rate_{r} for each r in positive_rates
    """
    rng = np.random.default_rng(42)
    rows = []
    for aug in sorted(pred_data.keys()):
        for seq in sorted(pred_data[aug].keys()):
            for seed_idx, (preds, labels) in enumerate(pred_data[aug][seq]):
                probs = logits_to_probs(preds)
                ap = average_precision_score(labels, probs)
                opt_f1, opt_thr = optimal_threshold_f1(labels, probs)
                row = {
                    "aug": aug,
                    "seq": seq,
                    "seed_idx": seed_idx,
                    "ap": ap,
                    "opt_f1": opt_f1,
                    "opt_threshold": opt_thr,
                }
                for t in thresholds:
                    row[f"f1_at_{t:.2f}"] = f1_at_threshold(labels, probs, t)
                for rate in positive_rates:
                    rl, rp = resample_to_positive_rate(labels, probs, rate, rng)
                    rf1, _ = optimal_threshold_f1(rl, rp)
                    row[f"opt_f1_at_rate_{rate:.2f}"] = rf1
                    row[f"ap_at_rate_{rate:.2f}"] = average_precision_score(rl, rp)
                rows.append(row)

    return pd.DataFrame(rows)


def pool_predictions(pred_data, aug):
    """Concatenate all (predictions, labels) arrays for a given aug method."""
    all_probs, all_labels = [], []
    for seq in pred_data[aug]:
        for preds, labels in pred_data[aug][seq]:
            all_probs.append(logits_to_probs(preds))
            all_labels.append(labels)
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Aggregation and statistics
# ---------------------------------------------------------------------------

def aggregate_by_aug(run_df, metric, baseline_aug):
    """For a given metric, compute per-aug stats and Holm-corrected paired t-tests vs baseline.

    Averages over seeds first (per seq), then tests across sequences.
    Returns a DataFrame sorted by aug.
    """
    from statsmodels.stats.multitest import multipletests

    seq_means = run_df.groupby(["aug", "seq"])[metric].mean().reset_index()
    augs = sorted(seq_means["aug"].unique(), key=lambda a: (str(a) != str(baseline_aug), str(a)))

    baseline_vals = seq_means[seq_means["aug"] == baseline_aug].set_index("seq")[metric]
    rows = []
    for aug in augs:
        vals = seq_means[seq_means["aug"] == aug].set_index("seq")[metric]
        aligned = pd.concat([baseline_vals, vals], axis=1, keys=["base", "aug"]).dropna()
        diff = aligned["aug"] - aligned["base"]
        if aug != baseline_aug and len(aligned) >= 2:
            t, p = stats.ttest_rel(aligned["aug"], aligned["base"])
            wins = int((diff > 1e-6).sum())
        else:
            t, p, wins = float("nan"), float("nan"), None
        rows.append({
            "aug": aug,
            "n_seq": len(aligned),
            "mean": aligned["aug"].mean(),
            "std": aligned["aug"].std(),
            "mean_diff": diff.mean() if aug != baseline_aug else float("nan"),
            "t": t,
            "p-unc": p,
            "wins": f"{wins}/{len(aligned)}" if wins is not None else "—",
        })
    df = pd.DataFrame(rows)

    # Apply Holm correction across vs-baseline comparisons
    mask = df["aug"] != baseline_aug
    p_unc = df.loc[mask, "p-unc"].values
    if len(p_unc) > 0 and not all(np.isnan(p_unc)):
        _, p_holm, _, _ = multipletests(p_unc, method="holm")
        df.loc[mask, "p-holm"] = p_holm
    else:
        df["p-holm"] = float("nan")
    # Baseline row gets NaN
    df.loc[~mask, "p-holm"] = float("nan")

    # Ensure consistent column order
    df = df[["aug", "n_seq", "mean", "std", "mean_diff", "t", "p-unc", "p-holm", "wins"]]
    return df


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_section(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print('=' * 64)


def fmt_float(x, digits=4, sign=False):
    if isinstance(x, float) and np.isnan(x):
        return "—"
    fmt = f"{{:+.{digits}f}}" if sign else f"{{:.{digits}f}}"
    return fmt.format(x)


def print_metric_table(df, metric_label, baseline_aug):
    display = df.copy()
    display["mean"] = display["mean"].map(lambda x: fmt_float(x, 4))
    display["std"] = display["std"].map(lambda x: fmt_float(x, 4))
    display["mean_diff"] = display["mean_diff"].map(lambda x: fmt_float(x, 4, sign=True))
    display["t"] = display["t"].map(lambda x: fmt_float(x, 3))
    display["p-unc"] = display["p-unc"].map(lambda x: fmt_float(x, 4))
    display["p-holm"] = display["p-holm"].map(lambda x: fmt_float(x, 4))
    display.columns = ["aug", "n_seq", metric_label, "std", f"diff_vs_{baseline_aug}",
                        "t", "p-unc", "p-holm", "wins"]
    print(display.to_string(index=False))

    p_col = "p-holm" if "p-holm" in df.columns else "p-unc"
    sig = df[(~df[p_col].isna()) & (df[p_col] < 0.05)]
    if not sig.empty:
        print(f"\n  ** Significant vs {baseline_aug} (Holm p<0.05): {sig['aug'].tolist()}")
    else:
        print(f"\n  No significant differences vs {baseline_aug} (all Holm p >= 0.05)")


def print_threshold_ranking(run_df, thresholds, baseline_aug):
    print_section("Method ranking at fixed thresholds")
    seq_means = run_df.groupby(["aug", "seq"])[[f"f1_at_{t:.2f}" for t in thresholds]].mean().reset_index()

    header = f"  {'aug':>14}" + "".join(f"  thr={t:.2f}" for t in thresholds)
    print(header)
    augs = sorted(seq_means["aug"].unique(), key=str)
    for aug in augs:
        sub = seq_means[seq_means["aug"] == aug]
        vals = "".join(f"  {sub[f'f1_at_{t:.2f}'].mean():.4f}" for t in thresholds)
        marker = " *" if aug == baseline_aug else "  "
        print(f"{marker} {str(aug):>14}{vals}")

    print("\n  Rank (1=best) at each threshold:")
    means = {aug: {f"f1_at_{t:.2f}": seq_means[seq_means["aug"] == aug][f"f1_at_{t:.2f}"].mean()
                   for t in thresholds}
             for aug in augs}
    for t in thresholds:
        col = f"f1_at_{t:.2f}"
        ranked = sorted(augs, key=lambda a: means[a][col], reverse=True)
        print(f"  thr={t:.2f}: {' > '.join(str(a) for a in ranked)}")


def print_positive_rate_sensitivity(run_df, positive_rates, baseline_aug, orig_pos_rate=None):
    print_section("Class-balance sensitivity: optimal-threshold F1 at resampled positive rates")

    rate_cols = [f"opt_f1_at_rate_{r:.2f}" for r in positive_rates]
    seq_means = run_df.groupby(["aug", "seq"])[["opt_f1"] + rate_cols].mean().reset_index()
    augs = sorted(seq_means["aug"].unique(), key=str)

    orig_label = f"orig ({orig_pos_rate:.1%})" if orig_pos_rate is not None else "orig"
    rate_labels = [orig_label] + [f"{r:.0%}" for r in positive_rates]
    cols = ["opt_f1"] + rate_cols
    header = f"  {'aug':>14}" + "".join(f"  {lbl:>9}" for lbl in rate_labels)
    print(header)
    for aug in augs:
        sub = seq_means[seq_means["aug"] == aug]
        vals = "".join(f"  {sub[c].mean():>9.4f}" for c in cols)
        marker = " *" if aug == baseline_aug else "  "
        print(f"{marker} {str(aug):>14}{vals}")

    print("\n  Rank (1=best) at each positive rate:")
    means = {aug: {c: seq_means[seq_means["aug"] == aug][c].mean() for c in cols}
             for aug in augs}
    for lbl, col in zip(rate_labels, cols):
        ranked = sorted(augs, key=lambda a: means[a][col], reverse=True)
        print(f"  {lbl:>5}: {' > '.join(str(a) for a in ranked)}")


def print_ap_rate_sensitivity(run_df, positive_rates, baseline_aug, orig_pos_rate=None):
    # Mirrors the F1 class-balance sensitivity but uses AP (threshold-free).
    # AP is not class-balance invariant: resampling negatives raises precision at every threshold,
    # so AP increases with higher positive rates. Recall is unaffected. This section checks whether
    # the method ranking under AP remains stable across positive rates, complementing the F1 view.
    print_section("Class-balance sensitivity: AP at resampled positive rates")

    rate_cols = [f"ap_at_rate_{r:.2f}" for r in positive_rates]
    seq_means = run_df.groupby(["aug", "seq"])[["ap"] + rate_cols].mean().reset_index()
    augs = sorted(seq_means["aug"].unique(), key=str)

    orig_label = f"orig ({orig_pos_rate:.1%})" if orig_pos_rate is not None else "orig"
    rate_labels = [orig_label] + [f"{r:.0%}" for r in positive_rates]
    cols = ["ap"] + rate_cols
    header = f"  {'aug':>14}" + "".join(f"  {lbl:>9}" for lbl in rate_labels)
    print(header)
    for aug in augs:
        sub = seq_means[seq_means["aug"] == aug]
        vals = "".join(f"  {sub[c].mean():>9.4f}" for c in cols)
        marker = " *" if aug == baseline_aug else "  "
        print(f"{marker} {str(aug):>14}{vals}")

    print("\n  Rank (1=best) at each positive rate:")
    means = {aug: {c: seq_means[seq_means["aug"] == aug][c].mean() for c in cols}
             for aug in augs}
    for lbl, col in zip(rate_labels, cols):
        ranked = sorted(augs, key=lambda a: means[a][col], reverse=True)
        print(f"  {lbl:>5}: {' > '.join(str(a) for a in ranked)}")


def print_pooled_pr_summary(pred_data, baseline_aug):
    # Concatenates all predictions from all runs (all seqs × all seeds) for each method into
    # one array, then computes a single PR curve on the pool. opt_F1 and opt_threshold are the
    # best F1 and the threshold that achieves it on this pooled curve.
    # Note: probabilities are not comparable across models trained on different sequences, so
    # treat this as a rough illustration rather than a number to report.
    print_section("Pooled PR curve summary (all runs concatenated per method)")
    augs = sorted(pred_data.keys(), key=str)
    rows = []
    for aug in augs:
        probs, labels = pool_predictions(pred_data, aug)
        ap = average_precision_score(labels, probs)
        opt_f1, opt_thr = optimal_threshold_f1(labels, probs)
        pos_rate = labels.mean()
        rows.append({
            "aug": aug,
            "n_samples": len(labels),
            "pos_rate": pos_rate,
            "AP": ap,
            "opt_F1": opt_f1,
            "opt_threshold": opt_thr,
        })
    df = pd.DataFrame(rows)
    display = df.copy()
    display["pos_rate"] = display["pos_rate"].map(lambda x: f"{x:.3f}")
    display["AP"] = display["AP"].map(lambda x: f"{x:.4f}")
    display["opt_F1"] = display["opt_F1"].map(lambda x: f"{x:.4f}")
    display["opt_threshold"] = display["opt_threshold"].map(lambda x: f"{x:.3f}")
    print(display.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Precision-recall and threshold analysis for a WandB group")
    parser.add_argument("group", help="WandB run group name")
    parser.add_argument("--entity", default="redstag")
    parser.add_argument("--project", default="thesis")
    parser.add_argument("--baseline-aug", default="none",
                        help="Aug level used as comparison baseline (default: none)")
    parser.add_argument("--artifact-split", default="test",
                        help="Prediction artifact prefix: 'test' -> 'test_predictions' (default: test)")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/thesis_preds"),
                        help="Local directory for cached artifact files")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help="Fixed probability thresholds to evaluate (default: 0.1 0.2 0.3 0.4 0.5)")
    parser.add_argument("--positive-rates", nargs="+", default=["0.13", "0.25", "0.50"],
                        help="Target positive rates for sensitivity analysis, or 'none' to skip "
                             "(default: 0.13 0.25 0.50)")
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel download threads (default: 16)")
    parser.add_argument("--exclude-aug", nargs="+", default=[],
                        help="Aug levels/techniques to exclude from analysis (e.g. --exclude-aug real)")
    parser.add_argument("--allow-multi", action="store_true",
                        help="Allow embed-multi runs (their training predictions have a known bug)")
    args = parser.parse_args()

    try:
        args.baseline_aug = int(args.baseline_aug)
    except ValueError:
        pass

    exclude = set()
    for v in args.exclude_aug:
        try:
            exclude.add(int(v))
        except ValueError:
            exclude.add(v)
    args.exclude_aug = exclude

    if args.positive_rates == ["none"]:
        positive_rates = []
    else:
        positive_rates = [float(r) for r in args.positive_rates]

    cache_dir = Path(args.cache_dir) / args.group
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download artifacts
    pred_data, _ = fetch_predictions(
        args.group, args.entity, args.project,
        args.artifact_split, cache_dir, args.expected_seeds,
        max_workers=args.workers,
    )
    if not pred_data:
        print("No prediction data found.", file=sys.stderr)
        sys.exit(1)

    # Safety: embed-multi training runs have buggy predictions (dataloader_drop_last)
    if "embed-multi" in pred_data and not args.allow_multi:
        print("ERROR: embed-multi detected in group. These training runs have buggy "
              "predictions (dataloader_drop_last). Use corrected ct24_eval groups "
              "or --allow-multi to override.",
              file=sys.stderr)
        sys.exit(1)

    if args.exclude_aug:
        for aug in args.exclude_aug:
            pred_data.pop(aug, None)
        print(f"Excluded aug levels: {args.exclude_aug}")

    # Compute actual positive rate from first available run's labels
    orig_pos_rate = None
    for aug in pred_data:
        for seq in pred_data[aug]:
            _, labels = pred_data[aug][seq][0]
            orig_pos_rate = labels.mean()
            break
        break

    # 2. Compute per-run metrics
    print("\nComputing per-run metrics...")
    run_df = compute_run_metrics(pred_data, args.thresholds, positive_rates)

    # 3. Pooled PR summary
    print_pooled_pr_summary(pred_data, args.baseline_aug)

    # 4. Average Precision
    # AP is computed per run on its own predictions, then seeds are averaged to get one value
    # per (aug, seq), and a paired t-test is run across the 15 sequence means vs baseline.
    # AP is threshold-free: it summarises the full PR curve as the area under it, rewarding
    # models that rank positives highly across all operating points.
    print_section("Average Precision (AUC-PR) — paired t-test vs baseline")
    ap_df = aggregate_by_aug(run_df, "ap", args.baseline_aug)
    print_metric_table(ap_df, "mean_AP", args.baseline_aug)

    # 5. Optimal-threshold F1
    # For each run, sweeps all thresholds and records the F1 at the one that maximises it.
    # Seeds are averaged per (aug, seq), then paired t-test across 15 sequence means vs baseline.
    # This is an oracle metric — the threshold is chosen with knowledge of the labels, giving an
    # upper bound on achievable F1. Useful for "how good could this model be with threshold tuning"
    # but not directly comparable to a deployment scenario.
    print_section("Optimal-threshold F1 — paired t-test vs baseline")
    opt_df = aggregate_by_aug(run_df, "opt_f1", args.baseline_aug)
    print_metric_table(opt_df, "mean_opt_F1", args.baseline_aug)

    # The threshold value itself at the oracle optimum. All methods peak near ~0.97 rather than
    # 0.5, reflecting that at 13% positive rate the model needs high confidence to predict positive.
    print_section("Optimal threshold values by aug method")
    thr_df = aggregate_by_aug(run_df, "opt_threshold", args.baseline_aug)
    print_metric_table(thr_df, "mean_opt_thr", args.baseline_aug)

    # 6. Fixed-threshold ranking
    # Applies a fixed probability cutoff to every run and computes F1, then averages over seeds
    # and sequences. Shows whether method rankings are stable across different operating points.
    # The logged test/f1 metric corresponds exactly to thr=0.50.
    print_threshold_ranking(run_df, args.thresholds, args.baseline_aug)

    # 7. Class-balance sensitivity
    # For each run, resamples the test set to a target positive rate (keeping all positives,
    # resampling negatives) and runs the oracle threshold sweep on the resampled set. Shows what
    # optimal-threshold F1 would look like on a more balanced evaluation set, and whether the
    # method ranking would change.
    if positive_rates:
        print_positive_rate_sensitivity(run_df, positive_rates, args.baseline_aug, orig_pos_rate)
        print_ap_rate_sensitivity(run_df, positive_rates, args.baseline_aug, orig_pos_rate)

    print("\n")


if __name__ == "__main__":
    main()
