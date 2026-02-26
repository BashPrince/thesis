import subprocess
from pathlib import Path

script = Path(__file__).parent / "generate_configs.py"

sequences = range(5)
aug_values = [5000]

for j in aug_values:
    num_samples = 100 + j
    num_epochs = 45000 // num_samples
    # effective_batch_size = 25 + (j // 100) * 25
    # batch_size = min(effective_batch_size, 50) # limit to 64 (memory)
    # grad_accum = effective_batch_size // batch_size # carry over to gradient accumulation steps (ignore remainder)
    for i in sequences:
        subprocess.run(
            [
                "python", str(script),
                "--data-artifact", f"experiment_014_seq_{i}_aug_{j}",
                "--group", "contrastive_low_data_large_augment",
                "--n-runs", "1",
                "--name", f"seq_{i}_aug_{j}_contrastive",
                "--epochs-contrastive", str(100),
                "--epochs-classify", str(num_epochs),
                "--batch-size-contrastive", str(64),
                "--grad-accum-contrastive", str(4),
                "--contrastive",
                "--append",
            ],
            check=True,
        )
