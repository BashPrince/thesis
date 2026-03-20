import asyncio
import csv
import itertools
import random
import re
from pathlib import Path

import litellm
from tqdm.asyncio import tqdm

litellm.api_key = Path(__file__).parent.parent / "secrets" / "openai_api_key.txt"
litellm.api_key = litellm.api_key.read_text().strip()

pos_qualities = ["a concern raising", "an interesting", "a likely", "a consequential", "a high stakes", "a surprising"]
neg_qualities = ["a trivial", "a benign", "an inconsequential", "a low stakes", "a true", "an obvious", "an unsurprising", "an outrageously false"]
num_qualities = [("numerical", 0.15), ("quantitative", 0.15), ("non-numerical", 0.7)]
pos_lengths = [(" of short length ", 0.2), (" of very short length ", 0.1), (" ", 0.7)]
neg_lengths = [(" of short length ", 0.05), (" of very short length ", 0.025), (" ", 0.85)]
topics = ["healthcare", "tax", "the economy", "employment", "education", "energy", "crime", "the military", "trade", "reproductive rights", "guns", "the environment", "society", "climate change", "vaccines", "elections", "home ownership", "fuel prices", "international relations", "Europe", "Asia", "the U.S.", "the middle east", "space", "epidemics", "religion"]


NUM_SAMPLES = 2425
NUM_WORKERS = 10
OUTPUT_FILE = "free_generations_25k_neg.csv"
WHICH_CLASS = "neg"  # "pos", "neg", or "both"


def shuffled_cycle(lst):
    while True:
        shuffled = lst[:]
        random.shuffle(shuffled)
        yield from shuffled


def build_tasks(num_samples, which="both"):
    """Build (prompt, class_label, topic, quality) tuples, cycling through qualities/topics.

    Args:
        num_samples: total number of samples to generate
        which: "pos", "neg", or "both" (default: equal split)
    """
    nq_values, nq_weights = zip(*num_qualities)
    pos_len_values, pos_len_weights = zip(*pos_lengths)
    neg_len_values, neg_len_weights = zip(*neg_lengths)

    n_pos = num_samples if which == "pos" else (0 if which == "neg" else num_samples // 2)
    n_neg = num_samples if which == "neg" else (0 if which == "pos" else num_samples - n_pos)

    tasks = []
    pos_cycle = zip(shuffled_cycle(pos_qualities), shuffled_cycle(topics))
    for quality, topic in itertools.islice(pos_cycle, n_pos):
        num_quality = random.choices(nq_values, weights=nq_weights, k=1)[0]
        length = random.choices(pos_len_values, weights=pos_len_weights, k=1)[0]
        prompt = f"Make {quality}, {num_quality}, single-sentence, political claim{length}about {topic} that should be fact checked. Your response should only contain the claim and nothing else."
        length_tag = f" | {length.strip()}" if length.strip() else ""
        tasks.append((prompt, "Yes", topic, f"{quality} | {num_quality}{length_tag}"))
    neg_cycle = zip(shuffled_cycle(neg_qualities), shuffled_cycle(topics))
    for quality, topic in itertools.islice(neg_cycle, n_neg):
        length = random.choices(neg_len_values, weights=neg_len_weights, k=1)[0]
        prompt = f"Make {quality} single-sentence political claim{length}about {topic} that does not require fact checking. Your response should only contain the claim and nothing else."
        length_tag = f" | {length.strip()}" if length.strip() else ""
        tasks.append((prompt, "No", topic, f"{quality}{length_tag}"))
    return tasks


def is_valid(text):
    return text and len(text) >= 10 and "\n" not in text and not re.search(r'\s[A-Z]\s', text)


async def generate_one(semaphore, prompt, class_label, topic, quality, pbar, writer, f):
    async with semaphore:
        while True:
            response = await litellm.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )
            text = response.choices[0].message.content.strip()
            if is_valid(text):
                break
        row = {"Text": text, "class_label": class_label, "topic": topic, "quality": quality}
        writer.writerow(row)
        f.flush()
        pbar.update(1)
        return row


async def main():
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        answer = input(f"{OUTPUT_FILE} already exists. Delete it? [y/N] ").strip().lower()
        if answer == "y":
            output_path.unlink()
        else:
            print("Aborting.")
            return

    tasks = build_tasks(NUM_SAMPLES, WHICH_CLASS)
    semaphore = asyncio.Semaphore(NUM_WORKERS)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Text", "class_label", "topic", "quality"])
        writer.writeheader()

        with tqdm(total=NUM_SAMPLES, desc="Generating") as pbar:
            coros = [generate_one(semaphore, *t, pbar, writer, f) for t in tasks]
            results = await asyncio.gather(*coros)

    print(f"Saved {len(results)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
