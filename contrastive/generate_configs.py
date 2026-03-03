#!/usr/bin/env python3
"""Generate experiment configs for a seed sweep.

Without --contrastive: generates N standard classification runs.
With    --contrastive: generates N contrastive pre-training runs each paired
                       with a classification fine-tuning run, with dependencies
                       set up so classification starts only after its paired
                       contrastive run completes.

Usage:
    # N standard classification runs
    python generate_configs.py --data-artifact mydata:latest --group exp1 --name baseline

    # N contrastive pre-training + classification runs
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
        '--epochs-contrastive', type=int, default=None,
        help='If set, override the "num_train_epochs" field for contrastive pretraining configs',
    )
    parser.add_argument(
        '--epochs-classify', type=int, default=None,
        help='If set, override the "num_train_epochs" field for classification/fine-tuning configs',
    )
    parser.add_argument(
        '--batch-size-contrastive', type=int, default=None,
        help='If set, override the "per_device_train_batch_size" field for contrastive pre-training configs',
    )
    parser.add_argument(
        '--grad-accum-contrastive', type=int, default=None,
        help='If set, override the "gradient_accumulation_steps" field for contrastive pre-training configs',
    )
    parser.add_argument(
        '--eval-steps-contrastive', type=int, default=None,
        help='If set, override the "eval_steps" field for contrastive pre-training configs',
    )
    parser.add_argument(
        '--eval-steps-classify', type=int, default=None,
        help='If set, override the "eval_steps" field for classification/fine-tuning configs',
    )
    parser.add_argument(
        '--patience-contrastive', type=int, default=None,
        help='If set, override the "early_stopping_patience" field for contrastive pre-training configs',
    )
    parser.add_argument(
        '--patience-classify', type=int, default=None,
        help='If set, override the "early_stopping_patience" field for classification/fine-tuning configs',
    )
    parser.add_argument(
        '--contrastive', action='store_true',
        help='Generate contrastive pre-training + classification pairs '
             'instead of classification-only runs',
    )
    parser.add_argument(
        '-a', '--append', action='store_true',
        help='Append new configs to an existing output directory, '
             'continuing index numbering from where the previous run left off',
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
    if existing and not args.append:
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

    # In append mode, continue numbering from the highest existing index
    if args.append:
        indices = []
        for f in existing:
            stem = os.path.splitext(os.path.basename(f))[0]  # e.g. classify_03
            try:
                indices.append(int(stem.split('_')[-1]))
            except ValueError:
                pass
        start_idx = max(indices) + 1 if indices else 1
    else:
        start_idx = 1

    seeds = random.sample(range(100_000), args.n_runs)

    # Load templates
    with open(os.path.join(TEMPLATE_DIR, 'train.json')) as f:
        classify_only_template = json.load(f)
    if args.contrastive:
        with open(os.path.join(TEMPLATE_DIR, 'contrastive.json')) as f:
            contrastive_template = json.load(f)

    dep_path = os.path.join(args.output_dir, 'dependencies.json')
    if args.append and os.path.exists(dep_path):
        with open(dep_path) as f:
            dependencies = json.load(f)
    else:
        dependencies = {}

    for i, seed in enumerate(seeds, start=start_idx):
        idx = f'{i:02d}'

        if args.contrastive:
            contrastive_run_name = f'{args.name}_pretrain_{idx}'
            classify_run_name    = f'{args.name}' # Do not append index to name of classification runs

            contrastive_filename = f'pretrain_{idx}.json'
            classify_filename    = f'classify_{idx}.json'

            # Contrastive pre-training config
            contrastive_cfg = dict(contrastive_template)
            contrastive_cfg['seed']             = seed
            contrastive_cfg['shuffle_seed']     = seed
            contrastive_cfg['run_name']         = contrastive_run_name
            contrastive_cfg['wandb_group_name'] = args.group
            contrastive_cfg['data_artifact']    = args.data_artifact
            if args.epochs_contrastive is not None:
                contrastive_cfg['num_train_epochs'] = args.epochs_contrastive
            if args.batch_size_contrastive is not None:
                contrastive_cfg['per_device_train_batch_size'] = args.batch_size_contrastive
            if args.grad_accum_contrastive is not None:
                contrastive_cfg['gradient_accumulation_steps'] = args.grad_accum_contrastive
            if args.eval_steps_contrastive is not None:
                contrastive_cfg['eval_steps'] = args.eval_steps_contrastive
                contrastive_cfg['save_steps'] = args.eval_steps_contrastive
            if args.patience_contrastive is not None:
                contrastive_cfg['early_stopping_patience'] = args.patience_contrastive

            contrastive_path = os.path.join(args.output_dir, contrastive_filename)
            with open(contrastive_path, 'w') as f:
                json.dump(contrastive_cfg, f, indent=4)

            # Classification fine-tuning config
            classify_cfg = dict(classify_only_template)
            classify_cfg['seed']                       = seed
            classify_cfg['shuffle_seed']               = seed
            classify_cfg['run_name']                   = classify_run_name
            classify_cfg['wandb_group_name']           = args.group
            classify_cfg['data_artifact']              = args.data_artifact
            classify_cfg['contrastive_model_artifact'] = f'{contrastive_run_name}:latest'
            if args.epochs_classify is not None:
                classify_cfg['num_train_epochs'] = args.epochs_classify
            if args.eval_steps_classify is not None:
                classify_cfg['eval_steps'] = args.eval_steps_classify
                classify_cfg['save_steps'] = args.eval_steps_classify
            if args.patience_classify is not None:
                classify_cfg['early_stopping_patience'] = args.patience_classify

            classify_path = os.path.join(args.output_dir, classify_filename)
            with open(classify_path, 'w') as f:
                json.dump(classify_cfg, f, indent=4)

            dependencies[classify_filename] = [contrastive_filename]

        else:
            classify_run_name = f'{args.name}' # Do not append index to names of classification runs
            classify_filename = f'classify_{idx}.json'

            classify_cfg = dict(classify_only_template)
            classify_cfg['seed']             = seed
            classify_cfg['shuffle_seed']     = seed
            classify_cfg['run_name']         = classify_run_name
            classify_cfg['wandb_group_name'] = args.group
            classify_cfg['data_artifact']    = args.data_artifact
            if args.epochs_classify is not None:
                classify_cfg['num_train_epochs'] = args.epochs_classify
            if args.eval_steps_classify is not None:
                classify_cfg['eval_steps'] = args.eval_steps_classify
                classify_cfg['save_steps'] = args.eval_steps_classify
            if args.patience_classify is not None:
                classify_cfg['early_stopping_patience'] = args.patience_classify

            classify_path = os.path.join(args.output_dir, classify_filename)
            with open(classify_path, 'w') as f:
                json.dump(classify_cfg, f, indent=4)

    with open(dep_path, 'w') as f:
        json.dump(dependencies, f, indent=4)

    n_configs = args.n_runs * (2 if args.contrastive else 1)
    print(f'\nGenerated {n_configs} configs for {args.n_runs} runs')


if __name__ == '__main__':
    main()
