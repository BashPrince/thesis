import subprocess
from pathlib import Path

script = Path(__file__).parent / "generate_train_config.py"

sequences = range(10)
aug_values = [0, 128, 256, 512, 1024, 2048, 4096]
model = "roberta-base"

for j in aug_values:
    num_samples = 128 + j
    num_epochs = 120000 // num_samples
    # effective_batch_size = 25 + (j // 100) * 25
    # batch_size = min(effective_batch_size, 50) # limit to 64 (memory)
    # grad_accum = effective_batch_size // batch_size # carry over to gradient accumulation steps (ignore remainder)
    mode = "classify"

    for i in sequences:
        args =  [
                    "python", str(script),
                    "--data-artifact", f"unrestricted_wrup_seq_{i}_aug_{j}",
                    "--group", "unrestricted_wrup_extended_roberta",
                    "--n-runs", "3",
                    "--name", f"seq_{i}_aug_{j}" + ("_contrastive" if mode == "contrastive" else ""),
                    "--epochs-contrastive", str(num_epochs),
                    "--epochs-classify", str(num_epochs),
                    "--batch-size-contrastive", str(64),
                    "--grad-accum-contrastive", str(1),
                    "--eval-steps-contrastive", str(64),
                    "--eval-steps-classify", str(64),
                    "--patience-contrastive", str(10),
                    "--patience-classify", str(15),
                    "--model", model,
                    "--append",
                ]
        
        if mode == "contrastive":
            args.append("--contrastive")
        elif mode == "multi":
            args.append("--multi")
            args.append("--batch-size-multi")
            args.append(str(32))
            args.append("--multi-alpha")
            args.append(str(0.1))

        subprocess.run(args, check=True)
