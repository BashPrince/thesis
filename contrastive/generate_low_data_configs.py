import random
import subprocess
from pathlib import Path

script = Path(__file__).parent / "generate_train_config.py"

sequences = 15
n_runs = 3
aug_values = [{"name": "v7_output_extend", "size": "embed"}]
model = "roberta-base"

# Fixed init seeds per sequence, constant across aug levels so that differences
# between aug conditions are attributable to data, not initialisation.
# Derived deterministically from a fixed master seed.
_rng = random.Random()
init_seeds_by_seq = [
    _rng.sample(range(100_000), n_runs) for _ in range(sequences)
]

for j in aug_values:
    # num_samples = 128 + j
    num_epochs = 1000
    # effective_batch_size = 25 + (j // 100) * 25
    # batch_size = min(effective_batch_size, 50) # limit to 64 (memory)
    # grad_accum = effective_batch_size // batch_size # carry over to gradient accumulation steps (ignore remainder)
    mode = "multi"

    for i in range(sequences):
        run_name = f"seq_{i}_aug_{j['name']}"

        if mode == "contrastive":
            run_name += "-contrastive"
        elif mode == "multi":
            run_name += "-multi"
            
        args =  [
                    "python", str(script),
                    "--data-artifact", f"{j['name']}_seq_{i}_aug_{j['size']}",
                    "--group", "v7_poolfilter_extend",
                    "--n-runs", str(n_runs),
                    "--name", run_name,
                    "--seeds", *[str(s) for s in init_seeds_by_seq[i]],
                    "--epochs-contrastive", str(num_epochs),
                    "--epochs-classify", str(num_epochs),
                    "--batch-size-contrastive", str(64),
                    "--grad-accum-contrastive", str(1),
                    "--eval-steps-contrastive", str(64),
                    "--eval-steps-classify", str(64),
                        "--patience-contrastive", str(20),
                        "--patience-classify", str(20),
                        "--model", model,
                        "--append",
                    ]
        
        if mode == "contrastive":
            args.append("--contrastive")
        elif mode == "multi":
            args.append("--multi")
            args.append("--batch-size-multi")
            args.append(str(16))
            args.append("--multi-alpha")
            args.append(str(0.2))

        subprocess.run(args, check=True)
