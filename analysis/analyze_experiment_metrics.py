#!/usr/bin/env python3
"""
Default analysis for a WandB augmentation experiment group.

Usage:
    python analysis/analyze_experiment_metrics.py <group> [--entity redstag] [--project thesis]
                                                         [--expected-seeds 3] [--baseline-aug 0]
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run: pip install wandb", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_runs(group, entity="redstag", project="thesis"):
    """Fetch finished runs from a WandB group.

    Supports run name formats:
        seq_{i}_aug_{j}             (old, random seeds)
        seq_{i}_aug_{j}_seed{s}     (new, fixed init seeds)

    Returns:
        data: dict[aug][seq] -> list of f1 scores
        all_runs: list of dicts with raw run metadata
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    data = defaultdict(lambda: defaultdict(list))
    # metric_data[metric][aug][seq] -> list of values
    metric_data = {m: defaultdict(lambda: defaultdict(list))
                   for m in ("precision", "recall", "accuracy")}
    all_runs = []

    state_counts = defaultdict(int)
    finished = 0
    for run in runs:
        state_counts[run.state] += 1
        meta = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "f1": run.summary.get("test/f1"),
        }
        all_runs.append(meta)

        if run.state != "finished":
            continue
        f1 = run.summary.get("test/f1")
        if f1 is None:
            continue
        finished += 1
        try:
            seq = int(run.name.split("seq_")[1].split("_")[0])
            aug_str = run.name.split("aug_")[1].split("_")[0]
        except IndexError:
            continue
        try:
            aug = int(aug_str)
        except ValueError:
            aug = aug_str
        data[aug][seq].append(f1)
        for m in metric_data:
            v = run.summary.get(f"test/{m}")
            if v is not None:
                metric_data[m][aug][seq].append(v)

    total = sum(state_counts.values())
    print(f"Group: {group}")
    state_summary = "  ".join(f"{s}={n}" for s, n in sorted(state_counts.items()))
    print(f"Total runs: {total}  ({state_summary})")
    print(f"Finished with test/f1: {finished}")
    aug_levels = sorted(data.keys())
    for aug in aug_levels:
        n_runs = sum(len(v) for v in data[aug].values())
        print(f"  aug={aug:>5}: {len(data[aug]):>3} sequences, {n_runs:>3} runs")

    return data, metric_data, all_runs


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def check_completion(data, expected_seeds=3):
    """Print diagnostic report of incomplete or missing runs."""
    all_seqs = set(seq for seqs in data.values() for seq in seqs)

    issues = []
    for aug in sorted(data.keys()):
        for seq in sorted(data[aug].keys()):
            n = len(data[aug][seq])
            if n < expected_seeds:
                issues.append(f"  aug={aug:>5}, seq={seq}: {n}/{expected_seeds} runs")

    print(f"\n--- Completion check (expected {expected_seeds} seeds per combo) ---")
    if issues:
        print("\n".join(issues))
    else:
        print("  All combos complete.")

    missing_issues = []
    for aug in sorted(data.keys()):
        missing = all_seqs - set(data[aug].keys())
        if missing:
            missing_issues.append(f"  aug={aug}: seqs {sorted(missing)} missing entirely")

    if missing_issues:
        print("Missing sequences:")
        print("\n".join(missing_issues))


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def build_summary_df(data, metric_data=None, baseline_aug=0):
    """Compute per-aug-level descriptive statistics (no significance tests).

    Returns a DataFrame with columns:
        aug, n_seq, mean_precision, mean_recall, mean_accuracy,
        mean_f1, std_f1, mean_diff, std_diff, seq_means
    """
    aug_levels = sorted(data.keys())
    if baseline_aug not in data:
        print(f"WARNING: baseline aug={baseline_aug} not found in data.", file=sys.stderr)

    results = []
    for aug in aug_levels:
        common_seqs = sorted(set(data[aug].keys()) & set(data[baseline_aug].keys()))
        a0 = np.array([np.mean(data[baseline_aug][s]) for s in common_seqs])
        a1 = np.array([np.mean(data[aug][s]) for s in common_seqs])
        diff = a1 - a0
        row = {
            "aug": aug,
            "n_seq": len(common_seqs),
            "mean_f1": a1.mean(),
            "std_f1": a1.std(),
            "mean_diff": diff.mean(),
            "std_diff": diff.std(),
            "seq_means": a1,
        }
        if metric_data:
            for m in ("precision", "recall", "accuracy"):
                vals = [np.mean(metric_data[m][aug][s])
                        for s in common_seqs if metric_data[m][aug].get(s)]
                row[f"mean_{m}"] = np.mean(vals) if vals else float("nan")
        results.append(row)
    return pd.DataFrame(results)


def build_seq_df(data):
    """Build long-format DataFrame: aug, seq, f1 (mean across seeds)."""
    aug_levels = sorted(data.keys())
    rows = []
    for aug in aug_levels:
        for seq in sorted(data[aug].keys()):
            rows.append({"aug": aug, "seq": seq, "f1": np.mean(data[aug][seq])})
    df = pd.DataFrame(rows)
    if pd.api.types.is_numeric_dtype(df["aug"]):
        df["log_aug"] = np.log1p(df["aug"])
    return df


def run_pairwise_vs_baseline(seq_df, baseline_aug=0):
    """Holm-corrected paired t-tests: each method vs baseline.

    Returns a DataFrame with columns: aug, T, dof, p-unc, p-corr (Holm), hedges g.
    """
    try:
        import pingouin as pg
    except ImportError:
        print("  (pingouin not installed — skipping paired t-tests)", file=sys.stderr)
        return None

    # Run uncorrected pairwise tests via pingouin
    posthoc = pg.pairwise_tests(
        data=seq_df, dv="f1", within="aug", subject="seq",
        parametric=True, padjust="none"
    )
    # Filter to vs-baseline comparisons only
    mask = (posthoc["A"] == baseline_aug) | (posthoc["B"] == baseline_aug)
    vs_baseline = posthoc[mask].copy()
    # normalise so baseline is always in column A
    swap = vs_baseline["B"] == baseline_aug
    vs_baseline.loc[swap, ["A", "B"]] = vs_baseline.loc[swap, ["B", "A"]].values

    # Apply Holm correction across only the vs-baseline family
    from statsmodels.stats.multitest import multipletests
    reject, p_holm, _, _ = multipletests(vs_baseline["p_unc"].values, method="holm")
    vs_baseline["p_holm"] = p_holm

    keep = ["B", "T", "dof", "p_unc", "p_holm", "hedges"]
    keep = [c for c in keep if c in vs_baseline.columns]
    vs_baseline = vs_baseline[keep]
    rename = {"B": "aug", "p_unc": "p-unc", "p_holm": "p-corr (Holm)", "hedges": "hedges g"}
    vs_baseline = vs_baseline.rename(columns=rename)
    return vs_baseline


def run_rm_anova(seq_df):
    """Repeated-measures ANOVA (diagnostic: effect size and sphericity check).

    Returns the ANOVA table DataFrame, or None if pingouin is unavailable.
    """
    try:
        import pingouin as pg
    except ImportError:
        print("  (pingouin not installed — skipping RM-ANOVA)", file=sys.stderr)
        return None

    rm = pg.rm_anova(data=seq_df, dv="f1", within="aug", subject="seq", detailed=True)
    return rm


def win_rate_table(data, baseline_aug=0):
    """For each aug level, count sequences that improved / tied / regressed vs baseline.

    Returns a DataFrame with columns: aug, wins, ties, losses, win_rate.
    """
    if baseline_aug not in data:
        return None
    aug_levels = [a for a in sorted(data.keys()) if a != baseline_aug]
    rows = []
    for aug in aug_levels:
        common_seqs = sorted(set(data[aug].keys()) & set(data[baseline_aug].keys()))
        wins = ties = losses = 0
        for s in common_seqs:
            diff = np.mean(data[aug][s]) - np.mean(data[baseline_aug][s])
            if diff > 1e-6:
                wins += 1
            elif diff < -1e-6:
                losses += 1
            else:
                ties += 1
        n = len(common_seqs)
        rows.append({
            "aug": aug,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": wins / n if n else float("nan"),
        })
    return pd.DataFrame(rows)


def best_aug_per_seq(data, baseline_aug=0):
    """For each sequence, return the aug level with the highest mean F1."""
    aug_levels = sorted(data.keys())
    all_seqs = sorted(set(seq for seqs in data.values() for seq in seqs))
    rows = []
    for seq in all_seqs:
        aug_f1 = {a: np.mean(data[a][seq]) for a in aug_levels if seq in data[a]}
        if not aug_f1:
            continue
        best_aug = max(aug_f1, key=aug_f1.get)
        baseline_f1 = aug_f1.get(baseline_aug, float("nan"))
        rows.append({
            "seq": seq,
            "best_aug": best_aug,
            "best_f1": aug_f1[best_aug],
            "baseline_f1": baseline_f1,
            "gain": aug_f1[best_aug] - baseline_f1,
        })
    return pd.DataFrame(rows)


def gain_baseline_correlation(data, baseline_aug=0):
    """Pearson correlation between sequence baseline F1 and augmentation gain.

    For each method, computes r(baseline_f1, method_f1 - baseline_f1) across sequences.
    A strong negative correlation means augmentation helps weak sequences most.

    Returns a DataFrame with columns: aug, r, p, n_seq.
    """
    if baseline_aug not in data:
        return None
    aug_levels = [a for a in sorted(data.keys()) if a != baseline_aug]
    rows = []
    for aug in aug_levels:
        common_seqs = sorted(set(data[aug].keys()) & set(data[baseline_aug].keys()))
        if len(common_seqs) < 3:
            continue
        baseline_f1 = np.array([np.mean(data[baseline_aug][s]) for s in common_seqs])
        aug_f1 = np.array([np.mean(data[aug][s]) for s in common_seqs])
        gain = aug_f1 - baseline_f1
        r, p = stats.pearsonr(baseline_f1, gain)
        rows.append({"aug": aug, "r": r, "p": p, "n_seq": len(common_seqs)})
    return pd.DataFrame(rows) if rows else None


def print_gain_baseline(gain_df):
    print_section("Gain-baseline correlation (r between seq baseline F1 and gain)")
    if gain_df is None or gain_df.empty:
        print("  Not enough data.")
        return
    fmt = gain_df.copy()
    fmt["r"] = fmt["r"].map("{:+.3f}".format)
    fmt["p"] = fmt["p"].map("{:.4f}".format)
    print(fmt.to_string(index=False))


def seed_variance_table(data):
    """Compute per-(seq, aug) std of F1 across seeds."""
    rows = [
        {"seq": seq, "aug": aug, "std": np.std(data[aug][seq], ddof=1)}
        for aug in sorted(data.keys())
        for seq in sorted(data[aug].keys())
        if len(data[aug][seq]) > 1
    ]
    if not rows:
        return None
    var_df = pd.DataFrame(rows)
    std_table = var_df.pivot(index="seq", columns="aug", values="std")
    std_table.loc["mean"] = std_table.mean()
    return std_table


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def print_summary(df, baseline_aug=0):
    print_section("Descriptive statistics by augmentation level")
    cols = ["aug", "n_seq"]
    for m in ("mean_precision", "mean_recall", "mean_accuracy"):
        if m in df.columns:
            cols.append(m)
    cols += ["mean_f1", "std_f1", "mean_diff", "std_diff"]

    display = df[cols].copy()
    for col in ("mean_precision", "mean_recall", "mean_accuracy", "mean_f1", "std_f1", "std_diff"):
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "—")
    display["mean_diff"] = display["mean_diff"].map("{:+.4f}".format)
    print(display.to_string(index=False))


def print_pairwise(vs_baseline):
    print_section("Paired t-tests vs baseline (Holm-corrected)")
    if vs_baseline is None:
        return
    fmt = vs_baseline.copy()
    for col in ["T", "dof", "p-unc", "p-corr (Holm)", "hedges g"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "—")
    print(fmt.to_string(index=False))

    # Flag significant results
    p_col = "p-corr (Holm)" if "p-corr (Holm)" in vs_baseline.columns else "p-unc"
    sig = vs_baseline[vs_baseline[p_col] < 0.05]
    if not sig.empty:
        print(f"\n  ** Significant vs baseline (Holm-corrected p<0.05): {sig['aug'].tolist()}")
    else:
        print(f"\n  No significant differences vs baseline (all Holm-corrected p >= 0.05)")


def print_rm_anova(rm):
    print_section("RM-ANOVA (diagnostic)")
    if rm is None:
        return
    cols = ["Source", "SS", "DF", "MS", "F", "p_unc", "p_GG_corr", "ng2", "eps", "sphericity"]
    available = [c for c in cols if c in rm.columns]
    print(rm[available].to_string(index=False))


def print_seed_variance(std_table):
    print_section("Seed variance: std(F1) across seeds per (seq, aug)")
    if std_table is None:
        print("  Not enough seeds per run to compute variance.")
        return
    print(std_table.map(lambda x: f"{x:.4f}").to_string())


def print_win_rate(win_df):
    print_section("Win rate vs baseline (sequences that improved)")
    if win_df is None or win_df.empty:
        print("  Not enough data.")
        return
    fmt = win_df.copy()
    fmt["win_rate"] = fmt["win_rate"].map("{:.0%}".format)
    print(fmt.to_string(index=False))


def print_best_aug(best_df):
    print_section("Best aug level per sequence")
    if best_df is None or best_df.empty:
        print("  Not enough data.")
        return
    fmt = best_df.copy()
    fmt["best_f1"] = fmt["best_f1"].map("{:.4f}".format)
    fmt["baseline_f1"] = fmt["baseline_f1"].map("{:.4f}".format)
    fmt["gain"] = fmt["gain"].map("{:+.4f}".format)
    print(fmt.to_string(index=False))

    counts = best_df["best_aug"].value_counts().sort_index()
    print(f"\n  Best-aug distribution: { {a: int(c) for a, c in counts.items()} }")


def print_per_seq_summary(data):
    """Show per-sequence F1 at each aug level to spot outliers."""
    print_section("Per-sequence mean F1 at each aug level")
    aug_levels = sorted(data.keys())
    all_seqs = sorted(set(seq for seqs in data.values() for seq in seqs))

    rows = {}
    for seq in all_seqs:
        rows[seq] = {f"aug={a}": f"{np.mean(data[a][seq]):.4f}" if seq in data[a] else "—"
                     for a in aug_levels}

    table = pd.DataFrame(rows).T
    table.index.name = "seq"
    print(table.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Default WandB experiment analysis")
    parser.add_argument("group", help="WandB run group name")
    parser.add_argument("--entity", default="redstag")
    parser.add_argument("--project", default="thesis")
    parser.add_argument("--expected-seeds", type=int, default=3,
                        help="Expected number of seed runs per (seq, aug) combo")
    parser.add_argument("--baseline-aug", default="none",
                        help="Aug level/technique to use as baseline for comparisons (default: 'none')")
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

    # 1. Fetch
    data, metric_data, _ = fetch_runs(args.group, entity=args.entity, project=args.project)
    if not data:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    if args.exclude_aug:
        for aug in args.exclude_aug:
            data.pop(aug, None)
            for m in metric_data:
                metric_data[m].pop(aug, None)
        print(f"Excluded aug levels: {args.exclude_aug}")

    # Safety: embed-multi training runs have buggy predictions (dataloader_drop_last)
    if "embed-multi" in data and not args.allow_multi:
        print("ERROR: embed-multi detected in group. These training runs have buggy "
              "predictions (dataloader_drop_last). Use corrected ct24_eval groups, "
              "--exclude-aug embed-multi, or --allow-multi to override.",
              file=sys.stderr)
        sys.exit(1)

    # 2. Completion check
    check_completion(data, expected_seeds=args.expected_seeds)

    # 3. Per-sequence summary
    print_per_seq_summary(data)

    # 4. Descriptive statistics
    df = build_summary_df(data, metric_data=metric_data, baseline_aug=args.baseline_aug)
    print_summary(df, baseline_aug=args.baseline_aug)

    # 5. Paired t-tests vs baseline (Holm-corrected) — primary inference
    seq_df = build_seq_df(data)
    vs_baseline = run_pairwise_vs_baseline(seq_df, baseline_aug=args.baseline_aug)
    print_pairwise(vs_baseline)

    # 6. RM-ANOVA — diagnostic (effect size, sphericity)
    rm = run_rm_anova(seq_df)
    print_rm_anova(rm)

    # 7. Win rate & best aug per sequence
    win_df = win_rate_table(data, baseline_aug=args.baseline_aug)
    print_win_rate(win_df)

    best_df = best_aug_per_seq(data, baseline_aug=args.baseline_aug)
    print_best_aug(best_df)

    # 8. Gain-baseline correlation
    gain_df = gain_baseline_correlation(data, baseline_aug=args.baseline_aug)
    print_gain_baseline(gain_df)

    # 9. Seed variance
    std_table = seed_variance_table(data)
    print_seed_variance(std_table)

    print("\n")


if __name__ == "__main__":
    main()
