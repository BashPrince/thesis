#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional
import jsonlines

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from datasets import Value, load_dataset

import wandb
import transformers
import adapters as adapters_lib
from adapters import AdapterArguments, AdapterTrainer, setup_adapter_training
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.47.1")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)
os.environ["WANDB_PROJECT"]="thesis"

class DynamicTrackingTrainer(AdapterTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits = []
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        if self.model.training:
            logits = outputs.get("logits")
            self.logits.extend(logits.tolist())

        return (loss, outputs) if return_outputs else loss


class SupervisedContrastiveTrainer(AdapterTrainer):
    """Trainer that optimises a supervised InfoNCE (SupCon) loss.

    For every anchor in the batch, same-class samples are treated as positives
    and different-class samples as negatives.  The loss is the mean over all
    anchors that have at least one in-batch positive:

        L = mean_i [ -1/|P(i)| * sum_{p in P(i)} log(
                exp(z_i · z_p / τ) / sum_{j≠i} exp(z_i · z_j / τ) ) ]
    """

    def __init__(self, *args, temperature: float = 0.05, pooling: str = "cls",
                 proj_dim: int = 128, proj_type: str = "mlp", balanced_sampling: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.pooling = pooling
        self.balanced_sampling = balanced_sampling

        hidden = self.model.config.hidden_size
        if proj_type == "mlp":
            proj_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, proj_dim),
            )
        else:
            proj_head = nn.Linear(hidden, proj_dim)
        # Attach to model so its parameters are included in the optimizer alongside
        # the adapter parameters.  model.save_adapter() will not persist this, so
        # the projection head is naturally discarded at the end of contrastive training.
        self.model.contrastive_proj = proj_head

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        # mean pooling over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def _supcon_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(embeddings, dim=-1)       # (N, D)
        N = z.shape[0]

        sim = torch.matmul(z, z.T) / self.temperature  # (N, N)

        self_mask = torch.eye(N, dtype=torch.bool, device=z.device)
        # same label, excluding self
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask  # (N, N)

        # denominator: sum over all j ≠ i  (mask diagonal before logsumexp)
        log_denom = torch.logsumexp(sim.masked_fill(self_mask, float("-inf")), dim=1)  # (N,)

        # log p(positive | anchor) for every pair
        log_prob = sim - log_denom.unsqueeze(1)  # (N, N)

        num_positives = pos_mask.float().sum(dim=1)  # (N,)
        loss_per_anchor = -(pos_mask.float() * log_prob).sum(dim=1) / num_positives.clamp(min=1)

        has_positives = num_positives > 0
        if not has_positives.any():
            return torch.tensor(0.0, device=z.device, requires_grad=True)

        loss = loss_per_anchor[has_positives].mean()

        # Diagnostic metrics (detached — no gradient cost)
        neg_mask = ~pos_mask & ~self_mask
        cosim = torch.matmul(z, z.T)  # raw cosine similarities, unscaled
        with torch.no_grad():
            cosim_pos = cosim[pos_mask].mean() if pos_mask.any() else cosim.new_tensor(0.0)
            cosim_neg = cosim[neg_mask].mean() if neg_mask.any() else cosim.new_tensor(0.0)
            self._contrastive_metrics = {
                "cosim_pos": cosim_pos.item(),
                "cosim_neg": cosim_neg.item(),
                "cosim_gap": (cosim_pos - cosim_neg).item(),
                "num_positives_mean": num_positives.mean().item(),
                "frac_anchors_with_positives": has_positives.float().mean().item(),
                "loss_std": loss_per_anchor[has_positives].std().item(),
            }

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True)
        # Use hidden_states[-1] rather than last_hidden_state: works for any output
        # type (e.g. MaskedLMOutput returned by AutoAdapterModel on encoder models)
        embeddings = self._pool(outputs.hidden_states[-1], inputs["attention_mask"])
        r = F.normalize(embeddings, dim=-1)          # normalize to unit hypersphere before projection
        z = model.contrastive_proj(r)                # project to lower-dim space
        loss = self._supcon_loss(z, labels)          # _supcon_loss normalizes z internally
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self.model.training and hasattr(self, "_contrastive_metrics"):
            logs.update(self._contrastive_metrics)
        super().log(logs, *args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return {}

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        accum = {k: 0.0 for k in ("cosim_pos", "cosim_neg", "cosim_gap",
                                   "num_positives_mean", "frac_anchors_with_positives", "loss_std")}
        n = 0
        with torch.no_grad():
            for inputs in eval_dataloader:
                inputs = self._prepare_inputs(inputs)
                self.compute_loss(self.model, {**inputs})
                if hasattr(self, "_contrastive_metrics"):
                    for k in accum:
                        accum[k] += self._contrastive_metrics.get(k, 0.0)
                    n += 1

        metrics = {f"{metric_key_prefix}_{k}": v / n for k, v in accum.items()} if n > 0 else {}
        metrics[f"{metric_key_prefix}_samples"] = len(eval_dataset)

        # Log while model is still in eval mode so our log() override doesn't
        # inject training-time _contrastive_metrics into the eval log entry.
        self.log(metrics)
        self.model.train()
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def _get_train_sampler(self):
        if not self.balanced_sampling:
            return super()._get_train_sampler()
        labels = [int(self.train_dataset[i]["label"]) for i in range(len(self.train_dataset))]
        counts = Counter(labels)
        weights = [1.0 / counts[l] for l in labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class MultiTaskTrainer(SupervisedContrastiveTrainer):
    """Joint CE + SupCon trainer. loss = alpha * supcon + (1-alpha) * ce."""

    def __init__(self, *args, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=model.training)

        # CE loss (computed manually since labels were popped)
        ce_loss = F.cross_entropy(outputs.logits, labels)

        if model.training:
            # SupCon loss (only during training; hidden states not available at inference)
            embeddings = self._pool(outputs.hidden_states[-1], inputs["attention_mask"])
            r = F.normalize(embeddings, dim=-1)
            z = model.contrastive_proj(r)
            supcon_loss = self._supcon_loss(z, labels)
            loss = self.alpha * supcon_loss + (1.0 - self.alpha) * ce_loss
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Skip SupervisedContrastiveTrainer.evaluate (contrastive-only loop);
        # use AdapterTrainer.evaluate (standard HF path with compute_metrics).
        return super(SupervisedContrastiveTrainer, self).evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on"},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classification task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "The delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    data_artifact: str = field(default=None, metadata={"help": "The name of the dataset artifact to download from wandb."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

@dataclass
class MyTrainingArguments(TrainingArguments):
    record_dynamics: bool = field(
        default=False,
        metadata={"help": "Whether to record the dynamics of the model for dataset cartography."}
    )
    record_prediction_split: bool = field(
        default=None,
        metadata={"help": "Optional dataset split for recording predictions on."}
    )
    wandb_group_name: Optional[str] = field(
        default=None,
        metadata={"help": "The group name to use for wandb."},
    )
    wandb_job_type: Optional[str] = field(
        default=None,
        metadata={"help": "The job type to use for wandb."},
    )
    prediction_model_artifact: Optional[str] = field(
        default=None,
        metadata={"help": "Optional model checkpoint artifact used for recording predictions on a dataset."},
    )
    training_mode: str = field(
        default="classification",
        metadata={"help": "Training mode: 'classification' (default), 'contrastive' (supervised SupCon pre-training), "
                          "or 'multi' (joint CE + SupCon multi-task learning)."},
    )
    contrastive_temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature τ for the SupCon InfoNCE loss (used when training_mode='contrastive')."},
    )
    contrastive_pooling: str = field(
        default="cls",
        metadata={"help": "Sentence pooling for contrastive embeddings: 'cls' (default) or 'mean' (used when training_mode='contrastive')."},
    )
    contrastive_model_artifact: Optional[str] = field(
        default=None,
        metadata={"help": "WandB artifact path of a contrastive-pretrained adapter to load before classification fine-tuning."},
    )
    contrastive_proj_dim: int = field(
        default=128,
        metadata={"help": "Output dimension of the contrastive projection head (used when training_mode='contrastive')."},
    )
    contrastive_proj_type: str = field(
        default="mlp",
        metadata={"help": "Projection head type: 'mlp' (2-layer MLP with ReLU, default) or 'linear' (single linear layer)."},
    )
    contrastive_balanced_sampling: bool = field(
        default=False,
        metadata={"help": "Use class-balanced batch sampling during contrastive pre-training (WeightedRandomSampler)."},
    )
    multi_alpha: float = field(
        default=0.5,
        metadata={"help": "SupCon loss weight in multi-task mode. "
                          "loss = alpha * supcon + (1 - alpha) * ce. "
                          "Only used when training_mode='multi'."},
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "Stop training when the monitored metric has not improved for this many evaluations. Requires load_best_model_at_end=True."},
    )


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments, AdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    
    # Some config sanity checks
    if training_args.prediction_model_artifact:
        if training_args.do_train or training_args.do_predict:
            raise ValueError("Evaluation of model checkpoint does not permit --do_train or --do_predict")
        if not training_args.record_prediction_split:
            raise ValueError("Evaluation of model checkpoint requires that --record_prediction_split is specified.")

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Setup wandb
    run = wandb.init(
        project="thesis",
        name=training_args.run_name,
        group=training_args.wandb_group_name,
        job_type=training_args.wandb_job_type
    )
    # Save the json config
    wandb.save(os.path.relpath(sys.argv[1]))

    # Download a model checkpoint for eval if given
    prediction_model_path = None
    if training_args.prediction_model_artifact:
        artifact = run.use_artifact(training_args.prediction_model_artifact)
        prediction_model_path = artifact.download()

    # Download a contrastive-pretrained adapter for classification fine-tuning if given
    contrastive_adapter_path = None
    if training_args.contrastive_model_artifact:
        artifact = run.use_artifact(training_args.contrastive_model_artifact)
        contrastive_adapter_path = artifact.download()


    # Download datasets from wandb and use the downloaded files in the remaining script
    artifact = run.use_artifact(data_args.data_artifact)
    artifact_dir = artifact.download()
    train_file = os.path.join(artifact_dir, "train.csv")
    validation_file = os.path.join(artifact_dir, "dev.csv")
    test_file = os.path.join(artifact_dir, data_args.test_file if data_args.test_file else "test.csv")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: uses the wandb downloaded files

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {"train": train_file, "validation": validation_file}

    # use the test dataset
    if training_args.do_predict or training_args.record_prediction_split:
        data_files["test"] = test_file

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    # Strip all CSV files to only Text and class_label so columns are consistent
    _tmp_dir = tempfile.mkdtemp()
    for key, path in data_files.items():
        df = pd.read_csv(path)
        df = df[[c for c in ["Text", "class_label"] if c in df.columns]]
        tmp_path = os.path.join(_tmp_dir, f"{key}.csv")
        df.to_csv(tmp_path, index=False)
        data_files[key] = tmp_path

    # Loading a dataset from local csv files
    raw_datasets = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    if data_args.remove_splits is not None:
        for split in data_args.remove_splits.split(","):
            logger.info(f"removing split {split}")
            raw_datasets.pop(split)

    if data_args.train_split_name is not None:
        logger.info(f"using {data_args.train_split_name} as train set")
        raw_datasets["train"] = raw_datasets[data_args.train_split_name]
        raw_datasets.pop(data_args.train_split_name)

    if data_args.validation_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
        raw_datasets.pop(data_args.validation_split_name)

    if data_args.test_split_name is not None:
        logger.info(f"using {data_args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[data_args.test_split_name]
        raw_datasets.pop(data_args.test_split_name)

    if data_args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                try:
                    raw_datasets[split] = raw_datasets[split].remove_columns(column)
                except ValueError:
                    logger.warning(f"Column {column} not found in split {split}, skipping removal.")

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    # Trying to have good defaults here, don't hesitate to tweak to your needs.

    is_regression = (
        raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if data_args.do_regression is None
        else data_args.do_regression
    )

    if training_args.training_mode in ("contrastive", "multi") and is_regression:
        raise ValueError("training_mode='contrastive' and 'multi' require classification labels, not regression targets.")

    is_multi_label = False
    if is_regression:
        label_list = None
        num_labels = 1
        # regression requires float as label type, let's cast it if needed
        for split in raw_datasets.keys():
            if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                logger.warning(
                    f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                )
                features = raw_datasets[split].features
                features.update({"label": Value("float32")})
                try:
                    raw_datasets[split] = raw_datasets[split].cast(features)
                except TypeError as error:
                    logger.error(
                        f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                    )
                    raise error

    else:  # classification
        if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
            is_multi_label = True
            logger.info("Label type is list, doing multi-label classification")
        # Trying to find the number of labels in a multi-label classification task
        # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
        # So we build the label list from the union of labels in train/val/test.
        label_list = get_label_list(raw_datasets, split="train")
        for split in ["validation", "test"]:
            if split in raw_datasets:
                val_or_test_labels = get_label_list(raw_datasets, split=split)
                diff = set(val_or_test_labels).difference(set(label_list))
                if len(diff) > 0:
                    # add the labels that appear in val/test but not in train, throw a warning
                    logger.warning(
                        f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                    )
                    label_list += list(diff)
        # if label is -1, we throw a warning and remove it from the label list
        for label in label_list:
            if label == -1:
                logger.warning("Label -1 found in label list, removing it.")
                label_list.remove(label)

        label_list.sort()
        num_labels = len(label_list)
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if is_regression:
        config.problem_type = "regression"
        logger.info("setting problem type to regression")
    elif is_multi_label:
        config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    auto_model_cls = (
        AutoModelForMaskedLM
        if training_args.training_mode == "contrastive"
        else AutoModelForSequenceClassification
    )
    model = auto_model_cls.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    adapters_lib.init(model)

    adapter_name, lang_adapter_name = None, None
    if training_args.training_mode == "contrastive":
        # Contrastive pre-training: train adapter without a classification head
        adapter_name, lang_adapter_name = setup_adapter_training(model=model, adapter_args=adapter_args, adapter_name=data_args.task_name)
    elif prediction_model_path:
        # Eval-only with a pre-loaded model checkpoint
        model.load_adapter(prediction_model_path + "/checkworthy")
        model.set_active_adapters("checkworthy")
        logger.info("Loaded adapter and head from " + prediction_model_path)
    elif contrastive_adapter_path:
        # Classification fine-tuning initialised from a contrastive-pretrained adapter
        # If the artifact contains a task-named subdirectory (new format) use it,
        # otherwise fall back to the artifact root (old format, pre-add_dir name fix)
        adapter_load_path = os.path.join(contrastive_adapter_path, data_args.task_name)
        if not os.path.isdir(adapter_load_path):
            adapter_load_path = contrastive_adapter_path
        model.load_adapter(adapter_load_path)
        model.set_active_adapters(data_args.task_name)
        model.train_adapter(data_args.task_name)
        # train_adapter() freezes all params including the classifier head; unfreeze it explicitly
        for param in model.classifier.parameters():
            param.requires_grad = True
        logger.info("Loaded contrastive adapter from " + contrastive_adapter_path)
    elif training_args.training_mode in ("classification", "multi"):
        # Standard classification training (or multi-task: always fresh adapter)
        adapter_name, lang_adapter_name = setup_adapter_training(model=model, adapter_args=adapter_args, adapter_name=data_args.task_name)

    if adapter_name is not None:
        logger.info("Added adapters to the model: {}".format(adapter_name))
    if lang_adapter_name is not None:
        logger.info("Added language adapter to the model: {}".format(lang_adapter_name))


    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # always use the labels from the training set
    label_to_id = {v: i for i, v in enumerate(label_list)}
    # update config with label infos
    if model.config.label2id != label_to_id:
        logger.warning(
            "The label2id key in the model config.json is not equal to the label2id key of this "
            "run. You can ignore this if you are doing finetuning."
        )
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in label_to_id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label_to_id[label]] = 1.0
        return ids

    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
        # Tokenize the texts
        result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        # Convert the string "0", "1" labels to integers
        for split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].cast_column("label", Value("int64"))

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = raw_datasets["test"]
        else:
            eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or training_args.record_prediction_split:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = raw_datasets["test"]

        if data_args.max_test_samples is not None:
            max_test_samples = min(len(test_dataset), data_args.max_test_samples)
            test_dataset = test_dataset.select(range(max_test_samples))

    # Log a few random samples from the training set:
    # if training_args.do_train:
    #     for index in random.sample(range(len(train_dataset)), 3):
    #         logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if data_args.metric_name is not None:
        if isinstance(data_args.metric_name, str):
            metric = (
                evaluate.load(data_args.metric_name, config_name="multilabel", cache_dir=model_args.cache_dir)
                if is_multi_label
                else evaluate.load(data_args.metric_name, cache_dir=model_args.cache_dir)
            )
            logger.info(f"Using metric {data_args.metric_name} for evaluation.")
        else:
            metric = evaluate.combine(data_args.metric_name)
            metric_string = ", ".join(data_args.metric_name)
            logger.info(f"Using metrics {metric_string} for evaluation.")
    else:
        if is_regression:
            metric = evaluate.load("mse", cache_dir=model_args.cache_dir)
            logger.info("Using mean squared error (mse) as regression score, you can use --metric_name to overwrite.")
        else:
            if is_multi_label:
                metric = evaluate.load("f1", config_name="multilabel", cache_dir=model_args.cache_dir)
                logger.info(
                    "Using multilabel F1 for multi-label classification task, you can use --metric_name to overwrite."
                )
            else:
                metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
                logger.info("Using accuracy as classification score, you can use --metric_name to overwrite.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
        elif is_multi_label:
            preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding
            # Micro F1 is commonly used in multi-label classification
            result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
        else:
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
        # if len(result) > 1:
        #     result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    callbacks = []
    if training_args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))

    if training_args.training_mode == "contrastive":
        trainer = SupervisedContrastiveTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=None,
            processing_class=tokenizer,
            data_collator=data_collator,
            temperature=training_args.contrastive_temperature,
            pooling=training_args.contrastive_pooling,
            proj_dim=training_args.contrastive_proj_dim,
            proj_type=training_args.contrastive_proj_type,
            balanced_sampling=training_args.contrastive_balanced_sampling,
            callbacks=callbacks if callbacks else None,
        )
    elif training_args.training_mode == "multi":
        trainer = MultiTaskTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
            data_collator=data_collator,
            temperature=training_args.contrastive_temperature,
            pooling=training_args.contrastive_pooling,
            proj_dim=training_args.contrastive_proj_dim,
            proj_type=training_args.contrastive_proj_type,
            balanced_sampling=training_args.contrastive_balanced_sampling,
            alpha=training_args.multi_alpha,
            callbacks=callbacks if callbacks else None,
        )
    else:
        trainer = DynamicTrackingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks if callbacks else None,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        if training_args.training_mode == "contrastive":
            # Save adapter weights as a standalone wandb artifact for use in downstream classification runs
            adapter_save_path = os.path.join(training_args.output_dir, data_args.task_name)
            os.makedirs(adapter_save_path, exist_ok=True)
            model.save_adapter(adapter_save_path, data_args.task_name)
            artifact = wandb.Artifact(name=training_args.run_name, type="model")
            artifact.add_dir(adapter_save_path, name=data_args.task_name)
            run.log_artifact(artifact)
            logger.info("Saved contrastive adapter to " + adapter_save_path)
        elif training_args.load_best_model_at_end:
            if trainer.state.best_model_checkpoint:
                # Create a wandb artifact and upload the best model checkpoint
                logger.info("Logging best model at " + trainer.state.best_model_checkpoint)
                artifact = wandb.Artifact(name="best_model", type="model", metadata={'original_path': trainer.state.best_model_checkpoint})
                artifact.add_dir(trainer.state.best_model_checkpoint)
                run.log_artifact(artifact)
        else:
            # Find the checkpoint with the highest number in output_dir
            checkpoint_dirs = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-") and d[len("checkpoint-"):].isdigit()]
            if checkpoint_dirs:
                last_checkpoint_dir = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
                last_checkpoint_path = os.path.join(training_args.output_dir, last_checkpoint_dir)

                # Create a wandb artifact and upload the last model checkpoint
                logger.info("Logging last model at " + last_checkpoint_path)
                artifact = wandb.Artifact(name="last_model", type="model", metadata={'original_path': last_checkpoint_path})
                artifact.add_dir(last_checkpoint_path)
                run.log_artifact(artifact)


        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Test performance
    if training_args.do_predict or training_args.record_prediction_split:
        logger.info("*** Test ***")
        
        if training_args.record_prediction_split == 'train':
            predict_dataset = train_dataset
        elif training_args.record_prediction_split == 'eval':
            predict_dataset = eval_dataset
        else:
            predict_dataset = test_dataset
        
        prediction_output = trainer.predict(test_dataset=predict_dataset)
        metrics = prediction_output.metrics
        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(predict_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Report to wandb
        if training_args.do_predict or training_args.record_prediction_split:
            for key, value in metrics.items():
                key = key.removeprefix("predict_")
                wandb.log({f"test/{key}": value})
        
        if training_args.record_prediction_split:
            predictions_path = os.path.join(training_args.output_dir, "predictions.npy")
            labels_path = os.path.join(training_args.output_dir, "labels.npy")

            np.save(predictions_path, prediction_output.predictions)
            np.save(labels_path, prediction_output.label_ids)
            artifact = wandb.Artifact(name=f"{training_args.record_prediction_split}_predictions", type="prediction")
            artifact.add_file(predictions_path)
            artifact.add_file(labels_path)
            run.log_artifact(artifact)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    
    if training_args.record_dynamics and training_args.training_mode != "contrastive":
        logger.info("Recording dynamics of the model for dataset cartography ...")
        for e in range(training_args.num_train_epochs):
            logger.info(f"Collecting dynamics for epoch {e + 1} ...")
            epoch_logits = trainer.logits[e * len(train_dataset):(e + 1) * len(train_dataset)]

            epoch_dynamics = []

            for sample, logits in zip(train_dataset, epoch_logits):
                epoch_dynamics.append({
                    "guid": sample["Sentence_id"],
                    f"logits_epoch_{e}": logits,
                    "gold": sample["label"],
                })
            
            dynamics_dir = os.path.join(training_args.output_dir, "dynamics")
            os.makedirs(dynamics_dir, exist_ok=True)
            with jsonlines.open(f"{dynamics_dir}/dynamics_epoch_{e}.jsonl", "w") as writer:
                writer.write_all(epoch_dynamics)
            
        # Create a wandb artifact and upload dynamics
        artifact = wandb.Artifact(name=f"{training_args.run_name}_dynamics", type="dynamics")
        artifact.add_dir(dynamics_dir)
        run.log_artifact(artifact)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()