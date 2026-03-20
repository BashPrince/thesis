"""
v1: Batch generation approach.
Generate 10 sentences per call, simulating political debate speech.
Target: ~25% Yes / 75% No to match CT24 distribution.
Spoken language style with context-dependent references.
"""
import asyncio
import csv
import json
import random
import re
from pathlib import Path

import litellm

litellm.api_key = Path(__file__).parent.parent / "secrets" / "openai_api_key.txt"
litellm.api_key = litellm.api_key.read_text().strip()

TOPICS = [
    "healthcare", "tax policy", "the economy", "employment",
    "education", "energy", "crime", "the military", "trade",
    "reproductive rights", "gun control", "the environment",
    "climate change", "vaccines", "elections", "immigration",
    "foreign policy", "social security", "infrastructure", "housing",
]

SYSTEM_PROMPT = """You are generating training data for a fact-checking research project.
Your task: generate sentences in the style of a political debate transcript.

CHECKWORTHY (label: Yes) — sentences that contain a specific verifiable factual claim:
- Specific statistics or numbers ("crime is down 40%", "$17 billion in spending")
- Claims about someone's past record or actions ("I voted against...", "he voted for...")
- Historical assertions ("we've done X since Y year")
- Comparative claims ("more than any other president")
- Verifiable claims about policies, events, people

NOT CHECKWORTHY (label: No) — sentences that lack specific verifiable content:
- Vague opinions or analysis ("we need to do better", "this is very important")
- Future intentions or promises ("I will make sure...", "we're going to fix this")
- Rhetorical questions or emotional appeals
- General characterizations without specifics ("my opponent has bad judgment")
- Statements about one's own feelings or beliefs

Style requirements:
- First-person speech from a politician in a live debate
- Spoken language: contractions, filler words ("uh", "look,", "I mean,"), mid-sentence pivots
- Often references the other speaker as "he", "she", "my opponent"
- Sentences can be incomplete or run-on, as in real transcripts
- Mix of short (5 words) and long (40+ words) sentences
"""

BATCH_PROMPT = """Generate exactly 10 sentences from a political debate on the topic of {topic}.

Requirements:
- Exactly 2-3 should be labeled "Yes" (checkworthy), the rest "No"
- Vary sentence length: some very short, some long and rambling
- Include spoken language markers in some sentences
- Make the Yes/No boundary realistic and sometimes fuzzy

Return ONLY a JSON array, no explanation. Format:
[
  {{"text": "...", "label": "Yes"}},
  {{"text": "...", "label": "No"}},
  ...
]"""


NUM_BATCHES = 5  # small run to inspect quality; 10 sentences each = 50 samples
NUM_WORKERS = 5
OUTPUT_FILE = Path(__file__).parent / "v1_samples.csv"


def is_valid_batch(items):
    if not isinstance(items, list) or len(items) != 10:
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


async def generate_batch(semaphore, topic, pbar, writer, f):
    async with semaphore:
        prompt = BATCH_PROMPT.format(topic=topic)
        while True:
            response = await litellm.acompletion(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(raw)
                # handle both {"items": [...]} and direct [...]
                if isinstance(parsed, dict):
                    items = next(iter(parsed.values()))
                else:
                    items = parsed
                if is_valid_batch(items):
                    break
            except (json.JSONDecodeError, StopIteration):
                pass

        rows = []
        for item in items:
            row = {"Text": item["text"], "class_label": item["label"], "topic": topic}
            writer.writerow(row)
            rows.append(row)
        f.flush()
        pbar.update(1)
        return rows


async def main():
    from tqdm.asyncio import tqdm

    if OUTPUT_FILE.exists():
        answer = input(f"{OUTPUT_FILE.name} already exists. Delete it? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborting.")
            return
        OUTPUT_FILE.unlink()

    topics = random.choices(TOPICS, k=NUM_BATCHES)
    semaphore = asyncio.Semaphore(NUM_WORKERS)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Text", "class_label", "topic"])
        writer.writeheader()

        with tqdm(total=NUM_BATCHES, desc="Batches") as pbar:
            coros = [generate_batch(semaphore, topic, pbar, writer, f) for topic in topics]
            results = await asyncio.gather(*coros)

    total = sum(len(r) for r in results)
    print(f"\nSaved {total} samples to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    asyncio.run(main())
