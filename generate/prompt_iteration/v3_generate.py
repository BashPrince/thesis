"""
v3: Fixes v2 residual mislabels.
Key issues fixed:
- Voting records ("he has consistently voted against X") must be Yes
- Claims attributed to studies/reports/organizations must be Yes
- Directional claims ("costs have skyrocketed under your watch") must be Yes
- Yes% reduced to ~25% (was ~30% in v2): ask for 2 Yes per batch of 10, allow 3
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

FEW_SHOT_YES = [
    "I mean, we've cut defense spending too much in the first place.",
    "Since I've been President, I've appointed more than twice as many black Federal judges as all previous presidents in the history of this country.",
    "They didn't have the right body armor.",
    "The National Federation of Independent Businesses has said that will cost 700,000 jobs.",
    "Our dollar's at an all time low against some foreign currencies.",
    "He was for the invasion of Iraq.",
    "The resolutions passed by the Congress have added about $17 billion in more spending over the budget that I recommended.",
    "He has consistently voted against any background check legislation.",  # voting record = Yes
    "According to the CDC, gun deaths have risen every year since 2015.",  # attributed claim = Yes
    "Energy costs have gone up 30 percent since this administration took office.",  # directional with number = Yes
]

FEW_SHOT_NO = [
    "We have to work more closely with our allies.",
    "This weakens the chances of civilization to survive.",
    "Should I be fortunate enough to become your president, when I put my hand on the Bible, I will swear to not only uphold the laws of the land.",
    "Look, the African-American community has been let down by our politicians.",
    "And thanks to this great general, David Petraeus, and the troops who serve under him, they have succeeded.",
    "But it's not going to solve the problem.",
    "He doesn't understand the impact of tariffs on American farmers.",  # vague attack without specific claim
    "Mr. President, under your watch, things have gotten much worse for families.",  # directional but too vague = No
    "The experts are clear on this issue.",  # vague reference to experts without a specific claim = No
    "We need to trust the science and work together for the common good.",
]

SYSTEM_PROMPT = """You are generating training data for a political fact-checking research project.
Your task: generate sentences in the style of a US presidential or congressional debate transcript.

CHECKWORTHY (label: Yes) — the sentence makes a specific claim that could be fact-checked:
- Specific numbers or statistics ("cut 700,000 jobs", "$17 billion over budget", "30% increase")
- Claims about someone's VOTING RECORD or past positions ("he voted for/against X", "she has consistently opposed Y")
- Claims about one's own record ("I've appointed more judges than...", "I've backed background checks")
- Directional factual claims with a named subject, even without numbers ("they didn't have the right body armor", "our dollar is at an all-time low")
- Claims attributed to a specific organization, study, or expert ("the CDC says...", "according to the NFIB...")
- Comparative or superlative claims ("more than any president", "the worst deal ever signed")

NOT CHECKWORTHY (label: No) — the sentence does NOT make a specific falsifiable claim:
- Vague normative statements ("we need to do better", "we have to work together")
- Future intentions or promises ("I will fight for you", "we're going to fix this")
- Emotional appeals or hyperbole ("this weakens the chances of civilization")
- Vague attacks without a specific claim ("he doesn't understand the impact", "she's been wrong on everything")
- Rhetorical questions ("where's the plan?", "can he name one time?")
- Vague references to studies/experts without stating what they say ("the experts are clear on this", "studies show we need to do more")
- Directional claims that are too vague ("things have gotten much worse", "this administration has failed on every front")
- Acknowledgments or praise that don't assert a falsifiable fact ("thanks to General Petraeus, they succeeded")

CRITICAL TESTS — ask yourself:
1. Could a fact-checker look up specific records or data to verify this claim? → Yes
2. Is this a claim about someone's specific past vote or stated position? → Yes
3. Does this cite a specific number, statistic, or source? → Yes
4. Is this just an opinion, vague criticism, or future promise? → No
5. Is this directional but too vague to verify (e.g., "got worse", "increased a lot")? → No

Style requirements:
- First-person speech from a politician in a live debate
- Spoken language: "uh", "look,", "I mean,", contractions, mid-sentence pivots
- References the other speaker as "he", "she", "my opponent", "the Senator", "the President"
- Sentences can be incomplete or run-on, as in real transcripts
- Vary openings: don't always start with "Look," or "I"
- Mix short fragments (5–10 words) with long rambling sentences (30–50 words)

Real CHECKWORTHY examples (Yes):
""" + "\n".join(f'- "{s}"' for s in FEW_SHOT_YES) + """

Real NOT CHECKWORTHY examples (No):
""" + "\n".join(f'- "{s}"' for s in FEW_SHOT_NO)

BATCH_PROMPT = """Generate exactly 10 sentences from a political debate on the topic of {topic}.

Requirements:
- Exactly 2 or 3 should be labeled "Yes" (checkworthy); the rest "No"
- At least one "Yes" must involve a voting record or attributed-to-a-source claim
- At least one "Yes" must NOT use specific numbers (directional or record-based)
- At least one sentence must be short (under 12 words), at least one must be long (over 35 words)
- Some "No" sentences should look superficially specific (referencing events/people) but still lack a falsifiable claim

Return ONLY a JSON object with a single key "sentences" containing the array, no explanation:
{{"sentences": [
  {{"text": "...", "label": "Yes"}},
  {{"text": "...", "label": "No"}},
  ...
]}}"""


NUM_BATCHES = 10
NUM_WORKERS = 10
OUTPUT_FILE = Path(__file__).parent / "v3_samples.csv"


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
                if isinstance(parsed, list):
                    items = parsed
                elif isinstance(parsed, dict):
                    # expect {"sentences": [...]} but handle any single-key wrapper
                    items = parsed.get("sentences") or next(
                        (v for v in parsed.values() if isinstance(v, list)), None
                    )
                else:
                    items = None
                if items and is_valid_batch(items):
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
    print(f"Topics used: {set(r['topic'] for r in all_rows)}")


if __name__ == "__main__":
    asyncio.run(main())
