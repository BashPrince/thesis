import subprocess
from pathlib import Path

script = Path(__file__).parent / "generate_configs.py"

sequences = range(5)
aug_values = [0, 100, 200, 400, 800]

for j in aug_values:
    num_samples = 100 + j
    num_epochs = 45000 // num_samples
    effective_batch_size = 25 + (j // 100) * 25
    batch_size = min(effective_batch_size, 50) # limit to 64 (memory)
    grad_accum = effective_batch_size // batch_size # carry over to gradient accumulation steps (ignore remainder)
    for i in sequences:
        subprocess.run(
            [
                "python", str(script),
                "--data-artifact", f"experiment_001_seq_{i}_aug_{j}",
                "--group", "contrastive_low_data_longer_train",
                "--n-runs", "3",
                "--name", f"seq_{i}_aug_{j}_contrastive",
                "--epochs-contrastive", str(num_epochs),
                "--epochs-classify", str(num_epochs),
                "--batch-size-contrastive", str(batch_size),
                "--grad-accum-contrastive", str(grad_accum),
                "--contrastive",
                "--append",
            ],
            check=True,
        )
