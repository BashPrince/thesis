"""
v5: Adaptive generation with diversity filtering and distribution matching.

Moves away from hand-crafted pattern rules (v3/v4) toward statistical quality control:
- Dynamic few-shots: each batch gets a fresh random sample of real CT24 examples in the
  prompt, anchoring generation style to the real distribution without fixed templates.
- Diversity filter: greedy maximin selection on TF-IDF bigrams — keeps the subset that
  maximises minimum pairwise distance (i.e. maximises coverage of the semantic space).
- Distribution filter: nearest-neighbour cosine similarity to real reference data —
  discards synthetic samples unlike any real example.
- Global pool dedup: rejects candidates too similar to already-accepted pool members.
- Iterates until TARGET_N samples are collected.

Usage:
    python v5_generate.py                     # full run to TARGET_N
    python v5_generate.py --dry-run           # 3 batches, report metrics only
"""
import argparse
import asyncio
import csv
import json
import random
import re
from pathlib import Path

import litellm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

litellm.api_key = Path(__file__).parent.parent / "secrets" / "openai_api_key.txt"
litellm.api_key = litellm.api_key.read_text().strip()

# ── Config ────────────────────────────────────────────────────────────────────
REAL_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "CT24_checkworthy_english" / "CT24_checkworthy_english_train.csv"
OUTPUT_FILE    = Path(__file__).parent / "v5_samples.csv"

TARGET_N          = 128   # final dataset size
OVERSHOOT_FACTOR  = 2.5    # generate this many × what's still needed each iteration
BATCH_SIZE        = 10     # sentences per API call
NUM_WORKERS       = 15     # concurrent API calls

FEW_SHOT_N        = 8      # real examples per batch prompt (stratified by label)
REFERENCE_N       = 128    # real examples held out for distribution scoring (per class)

DIVERSITY_KEEP    = 0.55   # fraction to keep after diversity filter
DIST_KEEP         = 0.60   # fraction to keep after distribution filter
POOL_SIM_THRESH   = 0.82   # reject new samples with cosine sim > this to any pool member

TOPICS = [
    "healthcare", "tax policy", "the economy", "employment", "education",
    "energy", "crime", "the military", "trade", "reproductive rights",
    "gun control", "the environment", "climate change", "vaccines",
    "elections", "immigration", "foreign policy", "social security",
    "infrastructure", "housing",
]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are generating training data for a political fact-checking research project.
Generate sentences in the style of US presidential or congressional debate transcripts.

CHECKWORTHY (Yes): the sentence makes a specific verifiable claim — specific numbers,
past votes/records, attributed findings, directional facts about named subjects,
comparative or superlative assertions.

NOT CHECKWORTHY (No): vague opinions, future promises, emotional appeals, rhetorical
questions, vague attacks, overly general normative statements.

Write in spoken debate style: contractions, filler words, mid-sentence pivots,
references to "he/she/my opponent/the Senator". Vary sentence length.\
"""

def build_user_prompt(few_shots, topic):
    """Build a user prompt with dynamically sampled real examples."""
    lines = [f"Here are {len(few_shots)} real examples from the target dataset:\n"]
    for text, label in few_shots:
        lines.append(f'  [{label}] "{text}"')
    lines.append(f"""
Generate exactly {BATCH_SIZE} sentences from a political debate on the topic of {topic}.
Match the style and difficulty of the examples above.
Aim for roughly {sum(1 for _, l in few_shots if l == 'Yes')} Yes and \
{sum(1 for _, l in few_shots if l == 'No')} No sentences.

Return ONLY a JSON object:
{{"sentences": [{{"text": "...", "label": "Yes"}}, ...]}}""")
    return "\n".join(lines)


# ── Real data loading ─────────────────────────────────────────────────────────
def load_real_data():
    df = pd.read_csv(REAL_DATA_PATH)
    yes = df[df["class_label"] == "Yes"]["Text"].dropna().tolist()
    no  = df[df["class_label"] == "No"]["Text"].dropna().tolist()
    return yes, no


def split_real_data(yes_texts, no_texts, seed=42):
    """Split real data into few-shot pool and reference set."""
    rng = random.Random(seed)
    yes_shuf = yes_texts[:]
    no_shuf  = no_texts[:]
    rng.shuffle(yes_shuf)
    rng.shuffle(no_shuf)

    # Reference: first REFERENCE_N of each class
    ref_yes = yes_shuf[:REFERENCE_N]
    ref_no  = no_shuf[:REFERENCE_N]

    # Few-shot pool: the rest
    pool_yes = yes_shuf[REFERENCE_N:]
    pool_no  = no_shuf[REFERENCE_N:]

    return (pool_yes, pool_no), (ref_yes, ref_no)


def sample_few_shots(pool_yes, pool_no, rng):
    """Sample FEW_SHOT_N examples, stratified ~25% Yes / 75% No to match CT24."""
    n_yes = max(1, round(FEW_SHOT_N * 0.25))
    n_no  = FEW_SHOT_N - n_yes
    shots = (
        [(t, "Yes") for t in rng.sample(pool_yes, min(n_yes, len(pool_yes)))] +
        [(t, "No")  for t in rng.sample(pool_no,  min(n_no,  len(pool_no)))]
    )
    rng.shuffle(shots)
    return shots


# ── TF-IDF vectorizer (fit once on reference) ─────────────────────────────────
def fit_vectorizer(ref_texts):
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    vect.fit(ref_texts)
    return vect


# ── Diversity filter: greedy maximin ──────────────────────────────────────────
def diversity_filter(texts, keep_n, vectorizer):
    """
    Greedy maximin: iteratively add the sample furthest from the already-selected set.
    O(n * keep_n) — feasible for batches of a few thousand.
    """
    if len(texts) <= keep_n:
        return list(range(len(texts)))

    vecs = vectorizer.transform(texts)
    n = vecs.shape[0]
    selected = [random.randrange(n)]
    # min distance from each point to the nearest selected point
    min_dists = np.ones(n)

    while len(selected) < keep_n:
        last = selected[-1]
        sims = cosine_similarity(vecs[last], vecs).flatten()
        dists = 1.0 - sims
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1.0  # exclude already selected

        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

    return selected


# ── Distribution filter: nearest-neighbour to real reference ─────────────────
def distribution_filter(texts, ref_vectors, keep_n, vectorizer):
    """
    Score each synthetic candidate by its cosine similarity to its nearest
    real-data neighbour. Keep the top keep_n.
    """
    if len(texts) <= keep_n:
        return list(range(len(texts)))

    cand_vecs = vectorizer.transform(texts)
    # max sim to any reference example
    sims = cosine_similarity(cand_vecs, ref_vectors)
    nn_scores = sims.max(axis=1)

    top_indices = np.argpartition(nn_scores, -keep_n)[-keep_n:]
    return top_indices.tolist()


# ── Global pool dedup ─────────────────────────────────────────────────────────
def pool_dedup_filter(new_texts, pool_texts, vectorizer, threshold=POOL_SIM_THRESH):
    """Reject new samples with cosine sim > threshold to any pool member."""
    if not pool_texts:
        return list(range(len(new_texts)))

    new_vecs  = vectorizer.transform(new_texts)
    pool_vecs = vectorizer.transform(pool_texts)
    sims = cosine_similarity(new_vecs, pool_vecs)
    max_sims = sims.max(axis=1)
    return [i for i, s in enumerate(max_sims) if s <= threshold]


# ── Generation ────────────────────────────────────────────────────────────────
def is_valid_batch(items):
    if not isinstance(items, list) or len(items) != BATCH_SIZE:
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        if "text" not in item or "label" not in item:
            return False
        if item["label"] not in ("Yes", "No"):
            return False
        text = item["text"]
        if not text or len(text) < 8 or "\n" in text:
            return False
    return True


async def generate_batch(semaphore, few_shots, topic, rng):
    async with semaphore:
        prompt = build_user_prompt(few_shots, topic)
        while True:
            response = await litellm.acompletion(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=1.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(raw)
                items = parsed if isinstance(parsed, list) else (
                    parsed.get("sentences") or
                    next((v for v in parsed.values() if isinstance(v, list)), None)
                )
                if items and is_valid_batch(items):
                    return [{"Text": it["text"], "class_label": it["label"], "topic": topic}
                            for it in items]
            except (json.JSONDecodeError, StopIteration, TypeError):
                pass


async def generate_candidates(n_needed, pool_yes, pool_no):
    """Generate ~n_needed * OVERSHOOT_FACTOR candidates."""
    n_batches = max(1, round(n_needed * OVERSHOOT_FACTOR / BATCH_SIZE))
    semaphore = asyncio.Semaphore(NUM_WORKERS)
    rng = random.Random()

    topics = random.choices(TOPICS, k=n_batches)
    few_shots_list = [sample_few_shots(pool_yes, pool_no, rng) for _ in range(n_batches)]

    from tqdm import tqdm
    pbar = tqdm(total=n_batches, desc=f"  Generating ({n_batches} batches)")

    results = []
    async def tracked(fs, t):
        result = await generate_batch(semaphore, fs, t, rng)
        pbar.update(1)
        return result

    results = await asyncio.gather(
        *[tracked(fs, t) for fs, t in zip(few_shots_list, topics)]
    )
    pbar.close()
    return [row for batch in results if batch for row in batch]


# ── Diversity report ─────────────────────────────────────────────────────────
def report_diversity(rows, ref_texts, vectorizer, label="pool"):
    yes = [r["Text"] for r in rows if r["class_label"] == "Yes"]
    no  = [r["Text"] for r in rows if r["class_label"] == "No"]
    total = len(rows)

    if len(yes) >= 2:
        vecs = vectorizer.transform(yes)
        sims = cosine_similarity(vecs)
        np.fill_diagonal(sims, 0)
        avg_sim = float(sims.mean())
    else:
        avg_sim = float("nan")

    # Distribution score: avg NN sim to reference
    if yes:
        ref_vecs  = vectorizer.transform(ref_texts)
        cand_vecs = vectorizer.transform(yes)
        nn_scores = cosine_similarity(cand_vecs, ref_vecs).max(axis=1)
        avg_dist  = float(nn_scores.mean())
    else:
        avg_dist = float("nan")

    def top_ngrams(texts, n=2, top=8):
        from collections import Counter
        counts = Counter()
        for t in texts:
            words = re.findall(r"\b\w+\b", t.lower())
            for i in range(len(words) - n + 1):
                counts[" ".join(words[i:i+n])] += 1
        return counts.most_common(top)

    yes_pct = len(yes) / total if total else 0
    print(f"\n[{label}] n={total}  Yes={yes_pct:.1%}")
    print(f"  Yes avg pairwise sim (TF-IDF): {avg_sim:.4f}  (CT24≈0.021)")
    print(f"  Yes avg NN sim to real:        {avg_dist:.4f}")
    print(f"  Top Yes 2-grams: {top_ngrams(yes)}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
async def main(dry_run=False):
    print("Loading real data…")
    yes_texts, no_texts = load_real_data()
    (pool_yes, pool_no), (ref_yes, ref_no) = split_real_data(yes_texts, no_texts)
    ref_texts = ref_yes + ref_no

    print(f"  Few-shot pool: {len(pool_yes)} Yes, {len(pool_no)} No")
    print(f"  Reference set: {len(ref_yes)} Yes, {len(ref_no)} No")

    print("Fitting TF-IDF vectorizer on reference data…")
    vectorizer = fit_vectorizer(ref_texts + pool_yes + pool_no)
    ref_vecs   = vectorizer.transform(ref_texts)

    target = 30 if dry_run else TARGET_N
    pool: list[dict] = []
    iteration = 0

    # Check for existing output to resume
    if OUTPUT_FILE.exists() and not dry_run:
        existing = pd.read_csv(OUTPUT_FILE).to_dict("records")
        pool.extend(existing)
        print(f"Resuming: loaded {len(pool)} existing samples from {OUTPUT_FILE.name}")

    while len(pool) < target:
        iteration += 1
        n_needed = target - len(pool)
        print(f"\n── Iteration {iteration}  (have {len(pool)}/{target}) ──")

        # 1. Generate candidates
        candidates = await generate_candidates(n_needed, pool_yes, pool_no)
        print(f"  Generated: {len(candidates)} candidates")

        # 2. Diversity filter within candidates
        texts = [r["Text"] for r in candidates]
        keep_n_div = max(1, round(len(candidates) * DIVERSITY_KEEP))
        div_idx = diversity_filter(texts, keep_n_div, vectorizer)
        candidates = [candidates[i] for i in div_idx]
        print(f"  After diversity filter: {len(candidates)}")

        # 3. Distribution filter (how similar to real data)
        texts = [r["Text"] for r in candidates]
        keep_n_dist = max(1, round(len(candidates) * DIST_KEEP))
        dist_idx = distribution_filter(texts, ref_vecs, keep_n_dist, vectorizer)
        candidates = [candidates[i] for i in dist_idx]
        print(f"  After distribution filter: {len(candidates)}")

        # 4. Global pool dedup
        pool_texts = [r["Text"] for r in pool]
        texts = [r["Text"] for r in candidates]
        dedup_idx = pool_dedup_filter(texts, pool_texts, vectorizer)
        candidates = [candidates[i] for i in dedup_idx]
        print(f"  After pool dedup: {len(candidates)}")

        if not candidates:
            print("  No candidates survived — consider loosening thresholds")
            break

        pool.extend(candidates)

        # Write / overwrite output
        if not dry_run:
            with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Text", "class_label", "topic"])
                writer.writeheader()
                writer.writerows(pool[:target])

        report_diversity(pool[:target], ref_texts, vectorizer, label=f"iter{iteration}")

        if dry_run:
            break

    print(f"\nDone. Final pool: {min(len(pool), target)} samples")
    if dry_run:
        report_diversity(pool, ref_texts, vectorizer, label="final-dry-run")


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 1 iteration with small target for testing")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(dry_run=args.dry_run))
    finally:
        # Cancel lingering tasks (e.g. litellm connection pools) to prevent hang
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
    os._exit(0)
