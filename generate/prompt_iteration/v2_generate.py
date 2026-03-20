"""
v2: Few-shot examples anchored to real CT24 data.
Fixes v1 issues:
- "Yes" class was too number-heavy; added directional/non-numerical checkworthy types
- Some "No" borderline claims were mislabeled; added contrast examples showing the distinction
- Few-shot examples drawn from actual CT24 training data
"""
import asyncio
import csv
import json
import random
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

# Few-shot examples drawn from actual CT24 training data
# Chosen to show the full range, including the hard/fuzzy boundary cases
FEW_SHOT_YES = [
    "I mean, we've cut defense spending too much in the first place.",
    "Since I've been President, I've appointed, for instance, more than twice as many black Federal judges as all previous presidents in the history of this country.",
    "They didn't have the right body armor.",
    "The National Federation of Independent Businesses has said that will cost 700,000 jobs.",
    "Our dollar's at an all time low against some foreign currencies.",
    "He was for the invasion of Iraq.",
    "The resolutions passed by the Congress have added about $17 billion in more spending over the budget that I recommended.",
    "I have a long record of reform and fighting through on the floor of the United States Senate.",
]

FEW_SHOT_NO = [
    "We have to work more closely with our allies.",
    "This weakens the chances of civilization to survive.",
    "Should I be fortunate enough to become your president, when I put my hand on the Bible, I will swear to not only uphold the laws of the land.",
    "Look, the African-American community has been let down by our politicians.",
    "And thanks to this great general, David Petraeus, and the troops who serve under him, they have succeeded.",
    "But it's not going to solve the problem.",
    "My background, my experience, my knowledge of the people of this country — those are the best bases to restore our leadership in the world.",
    "I think of people in law enforcement who are taking their lives in their hands everyday.",
]

SYSTEM_PROMPT = """You are generating training data for a political fact-checking research project.
Your task: generate sentences in the style of a US presidential or congressional debate transcript.

CHECKWORTHY (label: Yes) — the sentence makes a specific claim that could be investigated:
- Specific numbers or statistics ("cut 700,000 jobs", "$17 billion over budget", "more than twice as many")
- Claims about someone's past votes, positions, or actions ("he voted for the invasion of Iraq")
- Claims about one's own record ("I've appointed more judges than...", "I have a long record of reform")
- Directional factual claims, even without numbers ("we've cut defense spending too much", "they didn't have the right body armor")
- Comparative or superlative claims ("at an all-time low", "the second biggest surplus")
- Claims about what experts or organizations have said ("The NFIB has said this will cost 700,000 jobs")

NOT CHECKWORTHY (label: No) — the sentence lacks a specific verifiable claim:
- Vague normative statements ("we need to work more closely with allies", "it's not going to solve the problem")
- Future intentions or promises ("I will swear to uphold the laws", "we're going to fix this")
- Emotional appeals or hyperbole ("this weakens the chances of civilization to survive")
- General character attacks without specific claims ("my opponent has bad judgment")
- Rhetorical questions ("where's the plan?", "can he name one time?")
- Vague references to events or people that don't make a falsifiable claim ("thanks to General Petraeus, they succeeded")
- Opinions and interpretations ("the African-American community has been let down")

CRITICAL DISTINCTION: A sentence can mention real people, numbers, or events and still be "No" if it doesn't make a specific falsifiable claim. A sentence can be "Yes" with no numbers if it asserts a specific directional fact.

Style requirements:
- First-person speech from a politician in a live debate
- Spoken language: "uh", "look,", "I mean,", contractions, mid-sentence pivots
- Often references the other speaker as "he", "she", "my opponent", "the Senator", "the President"
- Sentences can be incomplete or run-on, as in real transcripts
- Vary length: some very short fragments (5–10 words), some long rambling sentences (30–50 words)

Real examples of CHECKWORTHY sentences (Yes):
""" + "\n".join(f'- "{s}"' for s in FEW_SHOT_YES) + """

Real examples of NOT CHECKWORTHY sentences (No):
""" + "\n".join(f'- "{s}"' for s in FEW_SHOT_NO)

BATCH_PROMPT = """Generate exactly 10 sentences from a political debate on the topic of {topic}.

Requirements:
- Exactly 2-3 should be labeled "Yes" (checkworthy), the rest "No"
- Include at least one "Yes" that is NOT purely numerical — a directional or record-based claim
- Include at least one short sentence (under 12 words) and one long one (over 30 words)
- Do NOT always start with "Look," — vary the sentence openings
- Make some "No" examples that reference specific events/people but still don't make a verifiable claim

Return ONLY a JSON array, no explanation. Format:
[
  {{"text": "...", "label": "Yes"}},
  {{"text": "...", "label": "No"}},
  ...
]"""


NUM_BATCHES = 8
NUM_WORKERS = 8
OUTPUT_FILE = Path(__file__).parent / "v2_samples.csv"


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
                items = parsed if isinstance(parsed, list) else next(iter(parsed.values()))
                if is_valid_batch(items):
                    break
            except (json.JSONDecodeError, StopIteration, TypeError):
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
    all_rows = [r for batch in results for r in batch]
    yes_pct = sum(1 for r in all_rows if r["class_label"] == "Yes") / total
    print(f"\nSaved {total} samples | Yes: {yes_pct:.1%} / No: {1-yes_pct:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
