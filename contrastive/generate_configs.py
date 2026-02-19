#!/usr/bin/env python3
"""Generate experiment configs for a seed sweep.

Without --contrastive: generates N standard classification runs.
With    --contrastive: generates N contrastive pre-training runs each paired
                       with a classification fine-tuning run, with dependencies
                       set up so classification starts only after its paired
                       contrastive run completes.

Usage:
    # 8 standard classification runs
    python generate_configs.py --data-artifact mydata:latest --group exp1 --name baseline

    # 8 contrastive pre-training + classification runs
    python generate_configs.py --data-artifact mydata:latest --group exp1 --name baseline --contrastive

Seeds are sampled randomly. The chosen seeds are printed and saved to
seeds.json in the output directory for reproducibility.
"""

import argparse
import glob
import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, '..', 'finetune', 'config_templates')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'finetune', 'configs')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data-artifact', required=True,
        help='WandB data artifact name, e.g. "my_dataset:latest"',
    )
    parser.add_argument(
        '--group', required=True,
        help='WandB group name for this experiment sweep',
    )
    parser.add_argument(
        '--name', required=True,
        help='Base name for runs, e.g. "baseline" produces run names like "baseline_01"',
    )
    parser.add_argument(
        '--n-runs', type=int, default=8,
        help='Number of runs to generate (default: 8)',
    )
    parser.add_argument(
        '--contrastive', action='store_true',
        help='Generate contrastive pre-training + classification pairs '
             'instead of classification-only runs',
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help='Directory to write configs into (default: ../finetune/configs/)',
    )
    args = parser.parse_args()

    if ':' not in args.data_artifact:
        args.data_artifact += ':latest'

    os.makedirs(args.output_dir, exist_ok=True)

    existing = [f for f in glob.glob(os.path.join(args.output_dir, '*.json'))
                if os.path.isfile(f)]
    if existing:
        print(f'Output directory already contains {len(existing)} JSON file(s):')
        for f in sorted(existing):
            print(f'  {os.path.basename(f)}')
        answer = input('Delete existing files and continue? [y/N] ').strip().lower()
        if answer == 'y':
            for f in existing:
                os.remove(f)
            print('Deleted.')
        else:
            print('Aborted.')
            return

    seeds = random.sample(range(100_000), args.n_runs)
    print(f'Sampled seeds: {seeds}')

    # Load templates
    with open(os.path.join(TEMPLATE_DIR, 'train.json')) as f:
        classify_only_template = json.load(f)
    if args.contrastive:
        with open(os.path.join(TEMPLATE_DIR, 'contrastive.json')) as f:
            contrastive_template = json.load(f)
        with open(os.path.join(TEMPLATE_DIR, 'train_from_contrastive.json')) as f:
            classify_from_contrastive_template = json.load(f)

    dependencies = {}

    for i, seed in enumerate(seeds, start=1):
        idx = f'{i:02d}'

        if args.contrastive:
            contrastive_run_name = f'{args.name}_contrastive_{idx}'
            classify_run_name    = f'{args.name}_{idx}'

            contrastive_filename = f'contrastive_{idx}.json'
            classify_filename    = f'classify_{idx}.json'

            # Contrastive pre-training config
            contrastive_cfg = dict(contrastive_template)
            contrastive_cfg['seed']             = seed
            contrastive_cfg['shuffle_seed']     = seed
            contrastive_cfg['run_name']         = contrastive_run_name
            contrastive_cfg['wandb_group_name'] = args.group
            contrastive_cfg['data_artifact']    = args.data_artifact

            contrastive_path = os.path.join(args.output_dir, contrastive_filename)
            with open(contrastive_path, 'w') as f:
                json.dump(contrastive_cfg, f, indent=4)
            print(f'Wrote {contrastive_path}  (seed {seed})')

            # Classification fine-tuning config
            classify_cfg = dict(classify_from_contrastive_template)
            classify_cfg['seed']                       = seed
            classify_cfg['shuffle_seed']               = seed
            classify_cfg['run_name']                   = classify_run_name
            classify_cfg['wandb_group_name']           = args.group
            classify_cfg['data_artifact']              = args.data_artifact
            classify_cfg['contrastive_model_artifact'] = f'{contrastive_run_name}:latest'

            classify_path = os.path.join(args.output_dir, classify_filename)
            with open(classify_path, 'w') as f:
                json.dump(classify_cfg, f, indent=4)
            print(f'Wrote {classify_path}  (seed {seed})')

            dependencies[classify_filename] = [contrastive_filename]

        else:
            classify_run_name = f'{args.name}_{idx}'
            classify_filename = f'classify_{idx}.json'

            classify_cfg = dict(classify_only_template)
            classify_cfg['seed']             = seed
            classify_cfg['shuffle_seed']     = seed
            classify_cfg['run_name']         = classify_run_name
            classify_cfg['wandb_group_name'] = args.group
            classify_cfg['data_artifact']    = args.data_artifact

            classify_path = os.path.join(args.output_dir, classify_filename)
            with open(classify_path, 'w') as f:
                json.dump(classify_cfg, f, indent=4)
            print(f'Wrote {classify_path}  (seed {seed})')

    # Always write dependencies.json so a stale one from a previous run is overwritten
    dep_path = os.path.join(args.output_dir, 'dependencies.json')
    with open(dep_path, 'w') as f:
        json.dump(dependencies, f, indent=4)
    print(f'Wrote {dep_path}')

    n_configs = args.n_runs * (2 if args.contrastive else 1)
    print(f'\nGenerated {n_configs} configs for {args.n_runs} runs')


if __name__ == '__main__':
    main()
