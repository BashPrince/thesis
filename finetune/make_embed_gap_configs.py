"""Generate per-run training configs for the embed-gap experiment.

Mirrors the embed runs in wandb group `v7_poolfilter_extend`: for each
(seq, seed) pair used there, emits one JSON config with
  - run_name: seq_{i}_aug_embed-gap_seed{seed}
  - data_artifact: v7_output_extend_seq_{i}_aug_embed-gap:latest
  - seed / shuffle_seed set to the embed run's seed
All other fields copied verbatim from the embed template.

Usage:
    python finetune/make_embed_gap_configs.py
"""
import json
import random
from pathlib import Path

import wandb

# Master seed for generating this experiment's training seeds (distinct from
# the embed runs'). Keeping it constant lets this script be re-run to produce
# identical configs.
MASTER_SEED = 20260414
SEEDS_PER_SEQ = 3

PROJECT = "redstag/thesis"
SRC_GROUP = "v7_poolfilter_extend"
SRC_AUG = "embed"
NEW_AUG = "embed-gap"
DATA_ARTIFACT_BASE = "v7_output_extend"

CONFIG_DIR = Path(__file__).parent / "configs"


def collect_embed_seqs(api):
    """Return the sorted list of seq indices that have embed runs."""
    runs = api.runs(PROJECT, filters={"group": SRC_GROUP,
                                      "config.wandb_job_type": "train"})
    seqs = set()
    for r in runs:
        name = r.name
        if f"_aug_{SRC_AUG}_seed" not in name:
            continue
        try:
            seqs.add(int(name.split("seq_")[1].split("_aug_")[0]))
        except (IndexError, ValueError):
            continue
    return sorted(seqs)


def sample_fresh_seeds(seqs, embed_seeds_used):
    """Generate SEEDS_PER_SEQ new seeds per seq, avoiding any already-used value."""
    rng = random.Random(MASTER_SEED)
    out = {}
    for seq in seqs:
        seeds = []
        while len(seeds) < SEEDS_PER_SEQ:
            s = rng.randint(1, 99999)
            if s in embed_seeds_used or s in seeds:
                continue
            seeds.append(s)
        out[seq] = seeds
    return out


def collect_all_embed_seeds(api):
    """All embed seeds across seqs, for exclusion when sampling new ones."""
    runs = api.runs(PROJECT, filters={"group": SRC_GROUP,
                                      "config.wandb_job_type": "train"})
    used = set()
    for r in runs:
        name = r.name
        if f"_aug_{SRC_AUG}_seed" not in name:
            continue
        try:
            used.add(int(name.split("_seed")[1]))
        except (IndexError, ValueError):
            continue
    return used


def fetch_template(api, seq0):
    """Download one embed run's config from seq `seq0` to use as a template."""
    runs = api.runs(PROJECT, filters={"group": SRC_GROUP,
                                      "config.wandb_job_type": "train"})
    for r in runs:
        if r.name.startswith(f"seq_{seq0}_aug_{SRC_AUG}_seed"):
            for f in r.files():
                if f.name.startswith("configs/") and f.name.endswith(".json"):
                    import tempfile
                    tmp = tempfile.mkdtemp()
                    f.download(root=tmp, replace=True)
                    with open(Path(tmp) / f.name) as fh:
                        return json.load(fh)
    raise RuntimeError(f"No embed config found for seq_{seq0}")


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    print("Collecting embed sequences and existing seeds...")
    seqs = collect_embed_seqs(api)
    used_seeds = collect_all_embed_seeds(api)
    print(f"  {len(seqs)} sequences, {len(used_seeds)} embed seeds to avoid")

    print(f"Sampling fresh seeds (master_seed={MASTER_SEED})...")
    seqs_seeds = sample_fresh_seeds(seqs, used_seeds)
    total = sum(len(v) for v in seqs_seeds.values())
    print(f"  {total} new (seq, seed) pairs")

    print("Fetching embed config template...")
    tmpl = fetch_template(api, seqs[0])

    written = 0
    for seq, seeds in seqs_seeds.items():
        for seed in seeds:
            cfg = dict(tmpl)
            cfg["seed"] = seed
            cfg["shuffle_seed"] = seed
            cfg["run_name"] = f"seq_{seq}_aug_{NEW_AUG}_seed{seed}"
            cfg["data_artifact"] = f"{DATA_ARTIFACT_BASE}_seq_{seq}_aug_{NEW_AUG}:latest"
            cfg["wandb_group_name"] = SRC_GROUP
            cfg["wandb_job_type"] = "train"

            out = CONFIG_DIR / f"{cfg['run_name']}.json"
            with open(out, "w") as fh:
                json.dump(cfg, fh, indent=4)
            written += 1

    print(f"Wrote {written} configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
