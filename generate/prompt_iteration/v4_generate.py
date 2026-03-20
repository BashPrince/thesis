"""
v4: Fixes structural repetition from v3.

Root cause: batch prompt requirements ("at least one must involve a voting record") caused
the model to treat these as a per-batch checklist, producing 27% 'voted_against/for' and
20% 'according_to' in Yes samples (vs CT24 Yes which is 90% structurally diverse "other").

Fixes:
- Removed structural requirements from batch prompt entirely
- Expanded few-shot Yes examples to cover the full range of CT24 "other" patterns:
  short fragments, mid-speech references, indirect attribution, historical claims, etc.
- Added explicit anti-repetition instruction: no structural pattern may repeat in Yes sentences
- Added post-generation diversity metrics (TF-IDF similarity, structure breakdown)
"""
import asyncio
import csv
import json
import random
import re
from collections import Counter
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

# Drawn from CT24 training data — heavily biased toward "other" structural types
# to counter the model's tendency to overuse 'voted against' and 'according to'
FEW_SHOT_YES = [
    # Short fragments / implicit claims
    "They didn't have the right body armor.",
    "Secretary Clinton also fought it.",
    "The other called it untrue.",
    "That happens to be a fact.",
    "The second biggest surplus next to Japan.",
    # Mid-speech contextual references
    "And more importantly, that was after getting a subpoena.",
    "He had a Democrat House, a Democrat Senate, super majority in both Houses.",
    "Now, Governor Romney actually wants to expand those tax breaks.",
    "They assumed that the inflation rate was because of excessive demand.",
    # Directional claims without numbers
    "I mean, we've cut defense spending too much in the first place.",
    "Our dollar's at an all time low against some foreign currencies.",
    "They're devaluing their currency, and there's nobody in our government to fight them.",
    "We're doing better than a lot of the countries in the world.",
    # Attribution WITHOUT "according to" — vary the phrase
    "Mr. Bush says we are going to put the IRS on every taxpayer.",
    "The National Federation of Independent Businesses has said that will cost 700,000 jobs.",
    "The CBO has estimated it would add $2 trillion to the deficit.",
    "The Pentagon's own report found equipment shortages in three divisions.",
    "Economists at the Fed warned this would trigger a recession.",
    # With specific numbers
    "Since I've been President, I've appointed more than twice as many black Federal judges as all previous presidents in the history of this country.",
    "The resolutions passed by the Congress have added about $17 billion in more spending over the budget that I recommended.",
    # Indirect claims about opponent's past
    "He was for the invasion of Iraq.",
    "All of these are programs that Senator Dole opposed that I supported.",
    "What I criticized were the policies that made it necessary to go to Korea.",
    "She supported the bill before she opposed it.",
]

FEW_SHOT_NO = [
    "We have to work more closely with our allies.",
    "This weakens the chances of civilization to survive.",
    "Should I be fortunate enough to become your president, when I put my hand on the Bible, I will swear to uphold the laws of the land.",
    "Look, the African-American community has been let down by our politicians.",
    "And thanks to this great general, David Petraeus, and the troops who serve under him, they have succeeded.",
    "But it's not going to solve the problem.",
    "He doesn't understand the impact of tariffs on American farmers.",
    "Mr. President, under your watch, things have gotten much worse for families.",
    "The experts are clear on this issue.",
    "We need to trust the science and work together for the common good.",
    "I've visited communities across this country and people are hurting.",
    "She just doesn't get it — she's never had to make a real decision under pressure.",
    "And we ought to go back to a little bit of Ronald Reagan's trust, but verify.",
    "I think of people in law enforcement who are taking their lives in their hands everyday.",
    "Look, my opponent has been wrong on this issue time and time again.",
]

SYSTEM_PROMPT = """You are generating training data for a political fact-checking research project.
Your task: generate sentences in the style of a US presidential or congressional debate transcript.

CHECKWORTHY (label: Yes) — the sentence makes a specific claim that could be investigated:
- Specific numbers or statistics
- Claims about someone's past votes, positions, or actions (can be brief and indirect)
- Claims about one's own record
- Directional factual claims with a named subject, even without numbers
- Claims attributed to a named organization, person, or study
- Comparative or superlative claims
- Short fragments that assert a verifiable fact

NOT CHECKWORTHY (label: No) — the sentence does NOT make a specific falsifiable claim:
- Vague normative statements, future intentions, promises
- Emotional appeals, hyperbole, rhetorical questions
- Vague attacks without a specific claim
- Vague references to experts/studies without stating what they say
- Directional claims that are too vague to verify
- Acknowledgments or praise without a falsifiable assertion

CRITICAL TESTS:
1. Could a fact-checker look up specific records or data to verify this? → Yes
2. Is this a claim about a specific past action, vote, or stated position? → Yes
3. Does it cite a specific number, statistic, or named source? → Yes
4. Is it an opinion, vague criticism, future promise, or rhetorical device? → No
5. Directional but too vague to verify ("got worse", "failed on every front")? → No

Style requirements:
- First-person speech from a politician in a live debate
- Spoken language: "uh", "look,", "I mean,", contractions, mid-sentence pivots
- References to other speaker: "he", "she", "my opponent", "the Senator", "the President"
- Sentences can be incomplete or run-on, as in real transcripts
- Vary openings and length: include short fragments (5–10 words) and long rambling sentences

Real CHECKWORTHY examples (Yes):
""" + "\n".join(f'- "{s}"' for s in FEW_SHOT_YES) + """

Real NOT CHECKWORTHY examples (No):
""" + "\n".join(f'- "{s}"' for s in FEW_SHOT_NO)

# Three batch types to force structural diversity across batches:
# A (free): any structure allowed, just hard-cap at 1 per pattern
# B (restricted): voting-record and "according to" phrases completely forbidden
# C (fragment): Yes claims must be short, contextual, or speech-like — no templates at all
BATCH_PROMPT_A = """Generate exactly 10 sentences from a political debate on the topic of {topic}.

- 2 or 3 should be "Yes" (checkworthy), the rest "No"
- Hard limits on Yes sentences: at most ONE may start with "According to"; \
at most ONE may contain "voted against" or "voted for"; at most ONE may contain "last year"
- Vary subjects across Yes sentences: I, he, she, they, named organisations, fragments
- Include at least one short sentence (under 12 words) and one long sentence (over 35 words)
- Some "No" sentences may reference specific events or people but still lack a falsifiable claim

Return ONLY a JSON object:
{{"sentences": [{{"text": "...", "label": "Yes"}}, ...]}}\
"""

BATCH_PROMPT_B = """Generate exactly 10 sentences from a political debate on the topic of {topic}.

- 2 or 3 should be "Yes" (checkworthy), the rest "No"
- STRICT: Yes sentences must NOT contain "voted against", "voted for", or start with "According to"
- Make checkworthy claims through: specific numbers, named records, past actions/events, \
comparative claims, short fragments asserting facts, indirect attributions ("X says", "X has warned", "X found")
- Include at least one short sentence (under 12 words) and one long sentence (over 35 words)

Return ONLY a JSON object:
{{"sentences": [{{"text": "...", "label": "Yes"}}, ...]}}\
"""

BATCH_PROMPT_C = """Generate exactly 10 sentences from a political debate on the topic of {topic}.

- 2 or 3 should be "Yes" (checkworthy), the rest "No"
- STRICT: Yes sentences must NOT contain "voted against", "voted for", or start with "According to"
- Yes sentences must be SHORT (under 15 words), contextual, and speech-like — fragments, mid-speech \
assertions, implicit claims that assume prior context, e.g.: "They didn't have the right body armor.", \
"Secretary Clinton also fought it.", "He had a Democrat House, a Democrat Senate.", \
"They're devaluing their currency.", "That happens to be a fact."
- No templates. No "last year". Vary subjects and claim types.

Return ONLY a JSON object:
{{"sentences": [{{"text": "...", "label": "Yes"}}, ...]}}\
"""

BATCH_PROMPTS = [BATCH_PROMPT_A, BATCH_PROMPT_B, BATCH_PROMPT_C]


NUM_BATCHES = 10
NUM_WORKERS = 10
OUTPUT_FILE = Path(__file__).parent / "v4_samples.csv"


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

    yes_texts = [item["text"] for item in items if item["label"] == "Yes"]

    # Reject if any structural pattern is overused across Yes sentences
    according_to = sum(1 for t in yes_texts if t.lower().startswith("according to"))
    if according_to > 1:
        return False, False

    voted = sum(1 for t in yes_texts if re.search(r"\bvoted (against|for)\b", t.lower()))
    if voted > 1:
        return False, False

    last_year = sum(1 for t in yes_texts if "last year" in t.lower())
    if last_year > 1:
        return False, False

    # For type B/C: these patterns must be zero
    has_restricted = according_to > 0 or voted > 0
    return True, has_restricted


async def generate_batch(semaphore, topic, batch_type, pbar, writer, f):
    async with semaphore:
        prompt = BATCH_PROMPTS[batch_type].format(topic=topic)
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
                    items = parsed.get("sentences") or next(
                        (v for v in parsed.values() if isinstance(v, list)), None
                    )
                else:
                    items = None
                if items:
                    valid, has_restricted = is_valid_batch(items)
                    if valid and (batch_type == 0 or not has_restricted):
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


def classify_structure(text):
    t = text.lower()
    if re.search(r"\bvoted (against|for)\b", t):
        return "voted_against/for"
    if t.startswith("according to"):
        return "according_to"
    if re.search(r"\b(i've|i have) (consistently|backed|supported|decreased|appointed|been)\b", t):
        return "ive_record"
    if re.search(r"\bsince \d{4}\b", t):
        return "since_year"
    if re.search(r"\b\d+\s*(%|percent|billion|million|thousand)\b", t):
        return "number_stat"
    return "other"


def report_diversity(rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    yes_texts = [r["Text"] for r in rows if r["class_label"] == "Yes"]
    no_texts = [r["Text"] for r in rows if r["class_label"] == "No"]
    total = len(rows)
    yes_pct = len(yes_texts) / total

    print(f"\nTotal: {total} | Yes: {yes_pct:.1%} / No: {1-yes_pct:.1%}")

    if len(yes_texts) >= 2:
        vect = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(yes_texts)
        sims = cosine_similarity(vect)
        np.fill_diagonal(sims, 0)
        avg_sim = sims.mean()
        print(f"Yes avg pairwise TF-IDF similarity: {avg_sim:.4f}  (CT24 baseline: 0.0214, v3: 0.0316)")

    print("\nYes structure breakdown:")
    structs = Counter(classify_structure(t) for t in yes_texts)
    for s, c in structs.most_common():
        print(f"  {c:3d}/{len(yes_texts)}  {s}  {'⚠' if s != 'other' and c / len(yes_texts) > 0.25 else ''}")

    print("\nTop Yes 2-grams:")
    import collections
    ngram_counts = collections.Counter()
    for t in yes_texts:
        words = re.findall(r"\b\w+\b", t.lower())
        for i in range(len(words) - 1):
            ngram_counts[f"{words[i]} {words[i+1]}"] += 1
    for ng, c in ngram_counts.most_common(10):
        flag = " ⚠" if c >= 3 else ""
        print(f"  {c:3d}x  {ng}{flag}")


async def main():
    from tqdm.asyncio import tqdm

    if OUTPUT_FILE.exists():
        answer = input(f"{OUTPUT_FILE.name} already exists. Delete it? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborting.")
            return
        OUTPUT_FILE.unlink()

    topics = random.choices(TOPICS, k=NUM_BATCHES)
    # Rotate batch types: A, B, C, A, B, C, ... so each type gets equal representation
    batch_types = [i % 3 for i in range(NUM_BATCHES)]
    semaphore = asyncio.Semaphore(NUM_WORKERS)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Text", "class_label", "topic"])
        writer.writeheader()

        with tqdm(total=NUM_BATCHES, desc="Batches") as pbar:
            coros = [generate_batch(semaphore, topic, bt, pbar, writer, f)
                     for topic, bt in zip(topics, batch_types)]
            results = await asyncio.gather(*coros)

    all_rows = [r for batch in results for r in batch]
    report_diversity(all_rows)


if __name__ == "__main__":
    asyncio.run(main())
