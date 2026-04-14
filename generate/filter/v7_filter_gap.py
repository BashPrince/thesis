"""
Pos-gap filtering over stored v7 pools.

Reprocesses the unfiltered candidate pools under v7_output/run_*/ using a
class-separation-gap criterion instead of the existing same-class nearest-
neighbour score. For each candidate the score is

    gap = mean(cos(cand, same_class_refs)) - mean(cos(cand, opp_class_refs))

Top-k by gap feeds the usual greedy-maximin diversity filter (same embedding
backend) down to the target size. Outputs samples_embedding_gap.csv and
augmented_embedding_gap.csv alongside the existing files in each run_dir.

After processing every run, reports pos_gap / neg_gap of the resulting
synthetic set against the full 128 real samples (same measurement used by
experiment_analysis/synthetic_data_quality/analyze_synthetic_quality.py).

Usage:
    python generate/filter/v7_filter_gap.py
    python generate/filter/v7_filter_gap.py --root generate/filter/v7_output
    python generate/filter/v7_filter_gap.py --agg max --lambda_ 1.0 --seed 0
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


TARGET_PER_CLASS = 512
METHOD_NAME = "embedding_gap"


# ---------------------------------------------------------------------------
# Filtering primitives
# ---------------------------------------------------------------------------

def gap_filter(cand_emb: np.ndarray,
               same_emb: np.ndarray,
               opp_emb: np.ndarray,
               keep_n: int,
               agg: str = "mean",
               lam: float = 1.0) -> np.ndarray:
    """Return indices of the top-keep_n candidates by class-separation gap.

    score = agg(cos(cand, same)) - lam * agg(cos(cand, opp))
    Embeddings are assumed L2-normalised, so cos = dot.
    """
    if len(cand_emb) <= keep_n:
        return np.arange(len(cand_emb))
    same = cand_emb @ same_emb.T            # (n_cand, n_same)
    opp = cand_emb @ opp_emb.T              # (n_cand, n_opp)
    if agg == "max":
        scores = same.max(axis=1) - lam * opp.max(axis=1)
    elif agg == "mean":
        scores = same.mean(axis=1) - lam * opp.mean(axis=1)
    else:
        raise ValueError(f"unknown agg: {agg}")
    return np.argpartition(scores, -keep_n)[-keep_n:]


def diversity_filter(emb: np.ndarray, keep_n: int, rng: random.Random) -> list[int]:
    """Greedy maximin on cosine distance. Matches v7_generate.diversity_filter."""
    if len(emb) <= keep_n:
        return list(range(len(emb)))
    sim = emb @ emb.T
    n = sim.shape[0]
    selected = [rng.randrange(n)]
    min_dists = np.ones(n)
    for _ in range(keep_n - 1):
        last = selected[-1]
        dists = 1.0 - sim[last]
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1.0
        selected.append(int(np.argmax(min_dists)))
    return selected


# ---------------------------------------------------------------------------
# Per-run pipeline
# ---------------------------------------------------------------------------

@dataclass
class GapResult:
    run: str
    pos_gap: float
    neg_gap: float
    sp_rp: float
    sp_rn: float
    sn_rn: float
    sn_rp: float


def process_run(run_dir: Path,
                model: SentenceTransformer,
                agg: str,
                lam: float,
                seed: int) -> GapResult:
    """Filter the stored pool with gap scoring and save outputs.

    Uses real_reference.csv (32/class) as the filter reference — matching the
    existing pipeline — but reports the final gap against real.csv (128).
    """
    pool = pd.read_csv(run_dir / "pool_unfiltered.csv")
    ref = pd.read_csv(run_dir / "real_reference.csv")
    real_all = pd.read_csv(run_dir / "real.csv")

    def encode(texts):
        return model.encode(list(texts), normalize_embeddings=True,
                            show_progress_bar=False)

    # --- Encode references and candidates ----------------------------------
    ref_yes_emb = encode(ref[ref["class_label"] == "Yes"]["Text"])
    ref_no_emb = encode(ref[ref["class_label"] == "No"]["Text"])

    pool_yes = pool[pool["class_label"] == "Yes"].reset_index(drop=True)
    pool_no = pool[pool["class_label"] == "No"].reset_index(drop=True)
    pool_yes_emb = encode(pool_yes["Text"])
    pool_no_emb = encode(pool_no["Text"])

    rng = random.Random(seed)

    def select(cand_emb, same_emb, opp_emb):
        # Stage 1: gap-based distribution filter to midpoint between pool and target
        keep_dist = TARGET_PER_CLASS + (len(cand_emb) - TARGET_PER_CLASS) // 2
        dist_idx = gap_filter(cand_emb, same_emb, opp_emb, keep_dist,
                              agg=agg, lam=lam)
        surv_emb = cand_emb[dist_idx]
        # Stage 2: diversity filter to exactly TARGET_PER_CLASS
        div_idx = diversity_filter(surv_emb, TARGET_PER_CLASS, rng)
        return dist_idx[div_idx]

    yes_idx = select(pool_yes_emb, ref_yes_emb, ref_no_emb)
    no_idx = select(pool_no_emb, ref_no_emb, ref_yes_emb)

    sel_yes = pool_yes.iloc[yes_idx].reset_index(drop=True)
    sel_no = pool_no.iloc[no_idx].reset_index(drop=True)
    samples = pd.concat([sel_yes, sel_no], ignore_index=True)
    samples.to_csv(run_dir / f"samples_{METHOD_NAME}.csv", index=False)

    # Augmented = real (128) + synthetic (1024); topic column for real = ""
    real_rows = real_all.copy()
    real_rows["topic"] = ""
    real_rows = real_rows[["Text", "class_label", "topic"]]
    aug = pd.concat([real_rows, samples], ignore_index=True)
    aug.to_csv(run_dir / f"augmented_{METHOD_NAME}.csv", index=False)

    # --- Report pos_gap / neg_gap against the full 128 real ---------------
    real_yes_emb = encode(real_all[real_all["class_label"] == "Yes"]["Text"])
    real_no_emb = encode(real_all[real_all["class_label"] == "No"]["Text"])
    sel_yes_emb = pool_yes_emb[yes_idx]
    sel_no_emb = pool_no_emb[no_idx]

    sp_rp = float((sel_yes_emb @ real_yes_emb.T).mean())
    sp_rn = float((sel_yes_emb @ real_no_emb.T).mean())
    sn_rn = float((sel_no_emb @ real_no_emb.T).mean())
    sn_rp = float((sel_no_emb @ real_yes_emb.T).mean())

    return GapResult(
        run=run_dir.name,
        pos_gap=sp_rp - sp_rn,
        neg_gap=sn_rn - sn_rp,
        sp_rp=sp_rp, sp_rn=sp_rn, sn_rn=sn_rn, sn_rp=sn_rp,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="generate/filter/v7_output",
                    help="Directory containing run_* subdirs (default: v7_output)")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--agg", choices=["mean", "max"], default="mean",
                    help="Aggregation over references for gap score")
    ap.add_argument("--lambda_", type=float, default=1.0,
                    help="Weight on opposite-class term: score = same - lambda*opp")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the diversity filter's greedy start")
    args = ap.parse_args()

    root = Path(args.root)
    runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
    if not runs:
        raise SystemExit(f"No run_* directories found under {root}")

    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"Processing {len(runs)} runs "
          f"(agg={args.agg}, lambda={args.lambda_}, seed={args.seed})")
    results = []
    for run_dir in runs:
        print(f"  {run_dir.name}...", end=" ", flush=True)
        res = process_run(run_dir, model,
                          agg=args.agg, lam=args.lambda_, seed=args.seed)
        print(f"pos_gap={res.pos_gap:+.4f}  neg_gap={res.neg_gap:+.4f}")
        results.append(res)

    df = pd.DataFrame([r.__dict__ for r in results])
    print("\nPer-run results:")
    print(df.to_string(index=False,
                       float_format=lambda x: f"{x:+.4f}"))

    print("\nAggregate (mean +/- SE across runs):")
    for col in ["pos_gap", "neg_gap", "sp_rp", "sp_rn", "sn_rn", "sn_rp"]:
        vals = df[col].to_numpy()
        m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        print(f"  {col:<8s}  {m:+.4f} +/- {se:.4f}")

    print(f"\nReference (embed method from quality analysis): pos_gap=+0.045, neg_gap=+0.009")


if __name__ == "__main__":
    main()
