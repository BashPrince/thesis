"""
v7: Pool-and-filter generation — generate all candidates upfront, then filter.

Changes from v6:
- No iterative generate→filter loop; instead generate POOL_SIZE candidates
  per class in one pass, then apply filtering to the full pool
- Filtering is two-stage: (1) distribution filter keeps samples closest to
  real data, (2) diversity filter maximises spread among survivors
- Filter backend is pluggable via --filter-method: "tfidf" (TF-IDF cosine)
  or "embedding" (sentence-transformers cosine)
- Supports --runs N to run the full pipeline N times sequentially, each with
  a different real-data split seed. Outputs are saved in a structured directory.

Usage:
    python v7_generate.py                                  # default tfidf
    python v7_generate.py --filter-method embedding        # use embeddings
    python v7_generate.py --runs 5                         # 5 sequential runs
    python v7_generate.py --dry-run                        # small test run
"""
import abc
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

litellm.api_key = Path(__file__).parent.parent / "secrets" / "openai_api_key.txt"
litellm.api_key = litellm.api_key.read_text().strip()

# ── Config ────────────────────────────────────────────────────────────────────
REAL_DATA_PATH = (
    Path(__file__).parent.parent.parent
    / "data" / "CT24_checkworthy_english" / "CT24_checkworthy_english_train.csv"
)
OUTPUT_DIR  = Path(__file__).parent / "v7_output_temp_1"

TARGET_N         = 1024   # total samples; split evenly between classes
POOL_SIZE        = 5      # multiplier: generate POOL_SIZE × TARGET_N/2 per class
BATCH_SIZE       = 8      # sentences per API call (all same class)
NUM_WORKERS      = 15     # concurrent API calls

FEW_SHOT_N            = 4    # real same-class examples shown per batch prompt

FEW_SHOT_REFERENCE_N  = 32   # pool size per class for few-shot sampling
REFERENCE_N           = 32   # held-out real examples per class for distribution scoring

TOPICS = [
    "healthcare", "tax policy", "the economy", "employment", "education",
    "energy", "crime", "the military", "trade", "reproductive rights",
    "gun control", "the environment", "climate change", "vaccines",
    "elections", "immigration", "foreign policy", "social security",
    "infrastructure", "housing",
]

# ── Prompts (unchanged from v6) ──────────────────────────────────────────────
SYSTEM_YES = """\
You are generating training data for a political fact-checking research project.
Generate CHECKWORTHY sentences in the style of US presidential or congressional debate transcripts.

A checkworthy sentence makes a SPECIFIC VERIFIABLE CLAIM:
- Specific numbers or statistics
- Claims about someone's past votes, positions, or actions (even brief: "He was for the invasion of Iraq.")
- Claims about one's own record ("I've appointed more judges than any president before me.")
- Directional facts with a named subject, even without numbers ("They didn't have the right body armor.")
- Claims attributed to named organisations, studies, or people ("The CBO estimated this adds $2 trillion.")
- Comparative or superlative assertions ("The second biggest surplus next to Japan.")
- Short speech fragments asserting an implicit fact ("And more importantly, that was after getting a subpoena.")

Write in spoken debate style: contractions, filler words ("uh", "look,", "I mean,"), mid-sentence pivots,
references to the opponent as "he/she/my opponent/the Senator/the President". Vary sentence length:
include very short fragments (5–10 words) as well as longer rambling sentences.\
"""

SYSTEM_NO = """\
You are generating training data for a political fact-checking research project.
Generate NOT CHECKWORTHY sentences in the style of US presidential or congressional debate transcripts.

A not-checkworthy sentence does NOT make a specific falsifiable claim:
- Vague normative statements ("we need to do better", "we have to work together")
- Future intentions or promises ("I will fight for you", "we're going to fix this")
- Emotional appeals or hyperbole ("this weakens the chances of civilization to survive")
- Vague attacks without a specific claim ("he doesn't understand the impact of this")
- Rhetorical questions ("where's the plan?", "can she name one time?")
- Vague references to experts without stating what they say ("the experts are clear on this")
- General character statements, praise, or scene-setting ("I've visited communities across this country")
- Opinions and interpretations ("the African-American community has been let down")

Write in spoken debate style: contractions, filler words ("uh", "look,", "I mean,"), mid-sentence pivots,
references to the opponent as "he/she/my opponent/the Senator/the President". Vary sentence length.\
"""


def build_prompt_yes(few_shots, topic):
    lines = [f"Here are {len(few_shots)} real checkworthy sentences from political debates:\n"]
    for text in few_shots:
        lines.append(f'  "{text}"')
    lines.append(f"""
Generate exactly {BATCH_SIZE} checkworthy sentences from a debate on the topic of {topic}.
Match the style and variety of the examples — include both short and long sentences,
and avoid repeating the same structural pattern (e.g. don't make all sentences start with "According to" or all reference a vote).

Return ONLY a JSON object:
{{"sentences": [{{"text": "...", "label": "Yes"}}, ...]}}""")
    return "\n".join(lines)


def build_prompt_no(few_shots, topic):
    lines = [f"Here are {len(few_shots)} real not-checkworthy sentences from political debates:\n"]
    for text in few_shots:
        lines.append(f'  "{text}"')
    lines.append(f"""
Generate exactly {BATCH_SIZE} not-checkworthy sentences from a debate on the topic of {topic}.
Match the style and variety of the examples — opinions, vague claims, future promises, emotional appeals,
rhetorical questions. Vary length and sentence type.

Return ONLY a JSON object:
{{"sentences": [{{"text": "...", "label": "No"}}, ...]}}""")
    return "\n".join(lines)


# ── Real data loading ─────────────────────────────────────────────────────────
def load_real_data():
    df = pd.read_csv(REAL_DATA_PATH)
    yes = df[df["class_label"] == "Yes"]["Text"].dropna().tolist()
    no  = df[df["class_label"] == "No"]["Text"].dropna().tolist()
    return yes, no


def split_real_data(yes_texts, no_texts, seed=42):
    """Split into per-class few-shot pool, reference set, and remainder."""
    rng = random.Random(seed)
    yes_shuf, no_shuf = yes_texts[:], no_texts[:]
    rng.shuffle(yes_shuf)
    rng.shuffle(no_shuf)
    used = REFERENCE_N + FEW_SHOT_REFERENCE_N
    ref_yes  = yes_shuf[:REFERENCE_N]
    pool_yes = yes_shuf[REFERENCE_N:used]
    rest_yes = yes_shuf[used:]
    ref_no   = no_shuf[:REFERENCE_N]
    pool_no  = no_shuf[REFERENCE_N:used]
    rest_no  = no_shuf[used:]
    return (pool_yes, pool_no), (ref_yes, ref_no), (rest_yes, rest_no)


# ── Filter backends ──────────────────────────────────────────────────────────
class FilterBackend(abc.ABC):
    """Vectorises texts and provides similarity scores for filtering."""

    @abc.abstractmethod
    def fit(self, texts: list[str]) -> None: ...

    @abc.abstractmethod
    def similarity_matrix(self, texts: list[str]) -> np.ndarray:
        """Return (n, n) pairwise cosine similarity matrix."""

    @abc.abstractmethod
    def cross_similarity(self, candidates: list[str], references: list[str]) -> np.ndarray:
        """Return (len(candidates), len(references)) cosine similarity matrix."""


class TfidfBackend(FilterBackend):
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=20000, sublinear_tf=True,
        )

    def fit(self, texts):
        self._vectorizer.fit(texts)

    def similarity_matrix(self, texts):
        from sklearn.metrics.pairwise import cosine_similarity
        vecs = self._vectorizer.transform(texts)
        return cosine_similarity(vecs)

    def cross_similarity(self, candidates, references):
        from sklearn.metrics.pairwise import cosine_similarity
        cand_vecs = self._vectorizer.transform(candidates)
        ref_vecs = self._vectorizer.transform(references)
        return cosine_similarity(cand_vecs, ref_vecs)


class EmbeddingBackend(FilterBackend):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def fit(self, texts):
        # Embedding models don't need fitting, but we warm up the model
        pass

    def _encode(self, texts):
        return self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def similarity_matrix(self, texts):
        embs = self._encode(texts)
        return embs @ embs.T

    def cross_similarity(self, candidates, references):
        cand_embs = self._encode(candidates)
        ref_embs = self._encode(references)
        return cand_embs @ ref_embs.T


FILTER_BACKENDS = {
    "tfidf": TfidfBackend,
    "embedding": EmbeddingBackend,
}


# ── Filters ──────────────────────────────────────────────────────────────────
def distribution_filter(texts, ref_texts, keep_n, backend: FilterBackend):
    """Keep samples with highest cosine similarity to their nearest real neighbour."""
    if len(texts) <= keep_n:
        return list(range(len(texts)))
    sims = backend.cross_similarity(texts, ref_texts)
    nn_scores = sims.max(axis=1)
    return np.argpartition(nn_scores, -keep_n)[-keep_n:].tolist()


def diversity_filter(texts, keep_n, backend: FilterBackend):
    """Greedy maximin: select keep_n samples maximising minimum pairwise distance."""
    if len(texts) <= keep_n:
        return list(range(len(texts)))
    sim_matrix = backend.similarity_matrix(texts)
    n = sim_matrix.shape[0]
    selected = [random.randrange(n)]
    min_dists = np.ones(n)
    for _ in range(keep_n - 1):
        last = selected[-1]
        dists = 1.0 - sim_matrix[last]
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1.0
        selected.append(int(np.argmax(min_dists)))
    return selected


# ── Generation ────────────────────────────────────────────────────────────────
def is_valid_batch(items, expected_label):
    if not isinstance(items, list) or len(items) != BATCH_SIZE:
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        if "text" not in item or "label" not in item:
            return False
        if item["label"] != expected_label:
            return False
        text = item["text"]
        if not text or len(text) < 8 or "\n" in text:
            return False
    return True


MAX_RETRIES = 5
RETRY_BACKOFF = 2.0  # seconds; doubled each retry


async def generate_batch(semaphore, system_prompt, user_prompt, expected_label, topic):
    async with semaphore:
        retries = 0
        while True:
            try:
                response = await litellm.acompletion(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=1.0,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                retries += 1
                if retries > MAX_RETRIES:
                    raise
                wait = RETRY_BACKOFF * (2 ** (retries - 1))
                print(f"\n  [WARN] API error (attempt {retries}/{MAX_RETRIES}), "
                      f"retrying in {wait:.0f}s: {e}")
                await asyncio.sleep(wait)
                continue
            retries = 0  # reset on successful API call
            raw = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(raw)
                items = parsed if isinstance(parsed, list) else (
                    parsed.get("sentences") or
                    next((v for v in parsed.values() if isinstance(v, list)), None)
                )
                if items and is_valid_batch(items, expected_label):
                    return [{"Text": it["text"], "class_label": it["label"], "topic": topic}
                            for it in items]
            except (json.JSONDecodeError, StopIteration, TypeError):
                pass


async def generate_class_pool(n_total, few_shot_pool, label, semaphore):
    """Generate n_total candidates for a single class in one pass."""
    n_batches = max(1, -(-n_total // BATCH_SIZE))
    topics = random.choices(TOPICS, k=n_batches)

    system = SYSTEM_YES if label == "Yes" else SYSTEM_NO
    build_prompt = build_prompt_yes if label == "Yes" else build_prompt_no

    from tqdm import tqdm
    pbar = tqdm(total=n_batches, desc=f"  [{label}] Generating ({n_batches} batches)")

    async def tracked(topic):
        shots = random.sample(few_shot_pool, min(FEW_SHOT_N, len(few_shot_pool)))
        prompt = build_prompt(shots, topic)
        result = await generate_batch(semaphore, system, prompt, label, topic)
        pbar.update(1)
        return result

    results = await asyncio.gather(*[tracked(t) for t in topics])
    pbar.close()
    return [row for batch in results if batch for row in batch]


# ── Per-class filter pipeline ────────────────────────────────────────────────
def filter_pipeline(candidates, ref_texts, backend: FilterBackend, label, target):
    """
    Filter candidates down to `target` in two steps:
    1. Distribution filter: keep the most realistic samples (midpoint between pool and target)
    2. Diversity filter: greedy maximin to select exactly `target`
    """
    if not candidates:
        return []

    texts = [r["Text"] for r in candidates]
    print(f"  [{label}] pool size: {len(candidates)}")

    # Step 1: distribution filter — reduce to midpoint between pool size and target
    keep_dist = min(len(candidates), target + (len(candidates) - target) // 2)
    dist_idx = distribution_filter(texts, ref_texts, keep_dist, backend)
    candidates = [candidates[i] for i in dist_idx]
    texts = [r["Text"] for r in candidates]
    print(f"  [{label}] after distribution filter: {len(candidates)}")

    # Step 2: diversity filter — select exactly target
    keep_div = min(target, len(candidates))
    div_idx = diversity_filter(texts, keep_div, backend)
    candidates = [candidates[i] for i in div_idx]
    print(f"  [{label}] after diversity filter:    {len(candidates)}")

    return candidates


# ── Diversity report ─────────────────────────────────────────────────────────
def report_diversity(pool_yes, pool_no, ref_yes, ref_no, backend: FilterBackend):
    def class_report(texts, ref_texts, cls):
        if len(texts) < 2:
            print(f"  [{cls}] n={len(texts)} — too few to report")
            return
        sims = backend.similarity_matrix(texts)
        np.fill_diagonal(sims, 0)
        avg_sim = float(sims.mean())

        nn_scores = backend.cross_similarity(texts, ref_texts).max(axis=1)
        avg_nn = float(nn_scores.mean())

        from collections import Counter
        ngrams = Counter()
        for t in texts:
            words = re.findall(r"\b\w+\b", t.lower())
            for i in range(len(words) - 1):
                ngrams[f"{words[i]} {words[i+1]}"] += 1
        top = ngrams.most_common(6)

        print(f"  [{cls}] n={len(texts)}  pairwise_sim={avg_sim:.4f}  nn_real={avg_nn:.4f}")
        print(f"         top 2-grams: {top}")

    print("\n── Diversity report ──")
    class_report([r["Text"] for r in pool_yes], ref_yes, "Yes")
    class_report([r["Text"] for r in pool_no],  ref_no,  "No")


# ── I/O helpers ──────────────────────────────────────────────────────────────
def save_csv(rows, path, fieldnames=("Text", "class_label", "topic")):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=fieldnames)
    df.to_csv(path, index=False)


def save_labeled_texts(yes_texts, no_texts, path):
    """Save labeled text lists as a CSV with Text and class_label columns."""
    rows = [{"Text": t, "class_label": "Yes"} for t in yes_texts] + \
           [{"Text": t, "class_label": "No"} for t in no_texts]
    save_csv(rows, path, fieldnames=("Text", "class_label"))


# ── Single run ───────────────────────────────────────────────────────────────
async def run_once(run_idx, run_dir, yes_texts, no_texts, backends, dry_run, semaphore):
    """Execute one full generate→filter pipeline, saving all artifacts to run_dir.

    backends is a dict of {name: FilterBackend}. Each backend filters the same
    pool independently, producing samples_{name}.csv (or samples.csv if only one).
    """
    seed = random.randint(0, 2**32 - 1)
    (fs_yes, fs_no), (ref_yes, ref_no), (rest_yes, rest_no) = split_real_data(
        yes_texts, no_texts, seed=seed,
    )
    print(f"  Few-shot pool: {len(fs_yes)} Yes, {len(fs_no)} No  (seed={seed})")
    print(f"  Reference set: {len(ref_yes)} Yes, {len(ref_no)} No")
    print(f"  Remaining real: {len(rest_yes)} Yes, {len(rest_no)} No")

    # Save the sampled real data
    save_labeled_texts(fs_yes, fs_no, run_dir / "real_fewshot.csv")
    save_labeled_texts(ref_yes, ref_no, run_dir / "real_reference.csv")
    save_labeled_texts(fs_yes + ref_yes, fs_no + ref_no, run_dir / "real.csv")

    target_each = 15 if dry_run else TARGET_N // 2
    pool_mult = 1.5 if dry_run else POOL_SIZE
    gen_each = round(target_each * pool_mult)

    # ── Generate full pools upfront ──────────────────────────────────────
    print(f"\n  Generating {gen_each} candidates per class ({gen_each * 2} total)…")
    cands_yes, cands_no = await asyncio.gather(
        generate_class_pool(gen_each, fs_yes, "Yes", semaphore),
        generate_class_pool(gen_each, fs_no,  "No",  semaphore),
    )
    print(f"  Generated: {len(cands_yes)} Yes, {len(cands_no)} No")

    # Save unfiltered pool
    save_csv(cands_yes + cands_no, run_dir / "pool_unfiltered.csv")

    # ── Unfiltered augmentation (first TARGET_N/2 per class) ────────────
    unf_yes = cands_yes[:target_each]
    unf_no  = cands_no[:target_each]
    real_rows_unf = [{"Text": t, "class_label": "Yes", "topic": ""} for t in fs_yes + ref_yes] + \
                    [{"Text": t, "class_label": "No",  "topic": ""} for t in fs_no + ref_no]
    unf_rows = unf_yes + unf_no
    save_csv(real_rows_unf + unf_rows, run_dir / "augmented_unfiltered.csv")
    print(f"  Unfiltered augmentation: {len(real_rows_unf)} real + {len(unf_rows)} synthetic = {len(real_rows_unf) + len(unf_rows)}")

    # ── Real-only augmentation (baseline) ────────────────────────────────
    rng = random.Random(seed)
    extra_yes = rng.sample(rest_yes, min(target_each, len(rest_yes)))
    extra_no  = rng.sample(rest_no,  min(target_each, len(rest_no)))
    base_rows = [{"Text": t, "class_label": "Yes", "topic": ""} for t in fs_yes + ref_yes] + \
                [{"Text": t, "class_label": "No",  "topic": ""} for t in fs_no + ref_no]
    extra_rows = [{"Text": t, "class_label": "Yes", "topic": ""} for t in extra_yes] + \
                 [{"Text": t, "class_label": "No",  "topic": ""} for t in extra_no]
    save_csv(base_rows + extra_rows, run_dir / "augmented_real.csv")
    print(f"  Real augmentation: {len(base_rows)} base + {len(extra_rows)} extra = {len(base_rows) + len(extra_rows)}")

    # ── Filter with each backend ─────────────────────────────────────────
    use_suffix = len(backends) > 1
    for name, backend in backends.items():
        print(f"\n  Filtering to {target_each} per class [{name}]…")
        pool_yes = filter_pipeline(cands_yes, ref_yes, backend, "Yes", target_each)
        pool_no  = filter_pipeline(cands_no,  ref_no,  backend, "No",  target_each)

        syn_suffix = f"samples_{name}.csv" if use_suffix else "samples.csv"
        aug_suffix = f"augmented_{name}.csv" if use_suffix else "augmented.csv"
        all_rows = pool_yes + pool_no
        save_csv(all_rows, run_dir / syn_suffix)

        # Augmented = sampled real + filtered synthetic
        real_rows = [{"Text": t, "class_label": "Yes", "topic": ""} for t in fs_yes + ref_yes] + \
                    [{"Text": t, "class_label": "No",  "topic": ""} for t in fs_no + ref_no]
        save_csv(real_rows + all_rows, run_dir / aug_suffix)
        print(f"\n  Saved {len(all_rows)} synthetic to {syn_suffix}, "
              f"{len(real_rows) + len(all_rows)} augmented to {aug_suffix}")

        report_diversity(pool_yes, pool_no, ref_yes, ref_no, backend)
        print(f"  Done [{name}]. Yes={len(pool_yes)}, No={len(pool_no)}, Total={len(all_rows)}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
async def main(dry_run=False, filter_method="tfidf", n_runs=1, seed=None):
    if seed is not None:
        print(f"Seeding RNG with {seed}")
        random.seed(seed)
        np.random.seed(seed)
    print("Loading real data…")
    yes_texts, no_texts = load_real_data()
    all_texts = yes_texts + no_texts

    methods = list(FILTER_BACKENDS.keys()) if filter_method == "all" else [filter_method]
    backends = {}
    for m in methods:
        print(f"Initialising filter backend: {m}")
        b = FILTER_BACKENDS[m]()
        b.fit(all_texts)
        backends[m] = b

    semaphore = asyncio.Semaphore(NUM_WORKERS)

    for run_idx in range(n_runs):
        run_dir = OUTPUT_DIR / f"run_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        header = f"Run {run_idx + 1}/{n_runs}"
        print(f"\n{'=' * 60}\n{header}\n{'=' * 60}")
        await run_once(run_idx, run_dir, yes_texts, no_texts, backends, dry_run, semaphore)


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--filter-method",
        choices=list(FILTER_BACKENDS.keys()) + ["all"],
        default="tfidf",
        help="Filtering backend: 'tfidf' (default), 'embedding', or 'all' (both)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of sequential augmentation runs (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: None, fully random)",
    )
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(
            dry_run=args.dry_run,
            filter_method=args.filter_method,
            n_runs=args.runs,
            seed=args.seed,
        ))
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
    os._exit(0)
