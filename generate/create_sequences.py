from abc import ABC, abstractmethod
from typing import Iterable
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import os
import shutil
from litellm import completion
import wandb
from tqdm import tqdm
import json

# Create and upload augmented sequences
NUM_API_WORKERS = 5

def upload_dataset(
    dataset_name: str,
    description: str,
    files: dict,
    metadata: dict = None,
):
    with wandb.init(project="thesis", job_type="generate-data", group='datagen') as run:

        # 🏺 create our Artifact
        data = wandb.Artifact(
            dataset_name,
            type="dataset",
            description=description,
            metadata=metadata)

        # 📦 Add files to the Artifact
        for name, file in files.items():
            suffix = file.split(".")[-1]
            data.add_file(file, name=f"{name}.{suffix}")

        # ✍️ Save the artifact to W&B.
        run.log_artifact(data)

def delete_configs():
    # Delete existing config files before creating new ones
    config_dir = os.path.join(os.path.dirname(__file__), "../finetune/configs")
    if os.path.exists(config_dir):
        for f in os.listdir(config_dir):
            file_path = os.path.join(config_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

def make_config(model_name: str, data_artifact_name: str, group_name: str, seed: int, batch_size: int, file_num: int, num_epochs: int, load_best_model: bool):
        # Load template
        template_path = os.path.join(os.path.dirname(__file__), "../finetune/config_templates/train.json")
        with open(template_path, "r") as f:
            config = json.load(f)

        # Update fields
        config["model_name_or_path"] = model_name
        config["run_name"] = data_artifact_name
        config["data_artifact"] = data_artifact_name + ":latest"
        config["wandb_group_name"] = group_name
        config["seed"] = seed
        config["per_device_train_batch_size"] = batch_size
        config["per_device_eval_batch_size"] = batch_size
        config["num_train_epochs"] = num_epochs
        config["load_best_model_at_end"] = load_best_model

        # Prepare output path
        out_dir = os.path.join(os.path.dirname(__file__), "../finetune/configs")
        out_path = os.path.join(out_dir, f"train_{file_num:02d}.json")

        # Save new config
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)

class DatasetProvider(ABC):
    @abstractmethod
    def get_datasets(self) -> list[pd.DataFrame]:
        pass

class SubsetDatasetProvider(DatasetProvider):
    """Sample non-overlapping subsets from a source file"""
    def __init__(self, source_path: str, num_subsets: int, subset_size: int, balance_classes: bool):
        self.source_path = source_path
        self.num_subsets = num_subsets
        self.subset_size = subset_size
        self.balance_classes = balance_classes

        # Initialize data
        self.subsets = []
        df = pd.read_csv(self.source_path)
        df = df[["Text", "class_label"]]
        used_indices = set()

        for _ in range(self.num_subsets):
            available_indices = list(set(df.index) - used_indices)
            if len(available_indices) < self.subset_size:
                break
            if self.balance_classes:
                # ...balance classes logic...
                df_available = df.loc[available_indices]
                yes_df = df_available[df_available["class_label"] == "Yes"]
                no_df = df_available[df_available["class_label"] == "No"]
                selected_yes = yes_df.sample(self.subset_size // 2, replace=False)
                selected_no = no_df.sample(self.subset_size // 2, replace=False)
                subset = pd.concat([selected_yes, selected_no]).sample(frac=1).reset_index(drop=True)
                selected_indices = subset.index.tolist()
                used_indices.update(selected_yes.index.tolist())
                used_indices.update(selected_no.index.tolist())
            else:
                selected_indices = pd.Series(available_indices).sample(self.subset_size, replace=False).tolist()
                subset = df.loc[selected_indices].reset_index(drop=True)
                used_indices.update(selected_indices)
            self.subsets.append(subset)


    def get_datasets(self) -> list[pd.DataFrame]:
        return self.subsets

class FileDatasetProvider(DatasetProvider):
    """Return datasets from a list of files"""

    def __init__(self, files: list[str]):
        self.datasets = [pd.read_csv(f) for f in files]
    
    def get_datasets(self):
        return self.datasets

class PromptProvider(ABC):
    @abstractmethod
    def get_pos_args_iterator(self):
        pass

    @abstractmethod
    def get_neg_args_iterator(self):
        pass

class ExamplePromptProvider(PromptProvider):
    """Fill templates with examples from the source data"""
    def __init__(self, source_data: pd.DataFrame, num_per_turn: int):
        self.source_data = source_data
        self.num_per_turn = num_per_turn
        self.pos_template = """### Instruction:
We are working on a natural language processing task to identify check-worthy claims in statements.
Your task is to generate synthetic data samples, i.e. check-worthy claims, to augment a dataset which will be used to finetune a model to automatically detect such claims.
The claims you generate must be instances of the positive (check-worthy) class.
The new claims may or may not contain false information.
A check-worthy claim has the following attributes:
- It contains a verifiable factual claim.
- It is of interest to the general public.
- It might be harmful to society.
- It contains information relevant to political discourse or policy decisions.
- It should be checked by a professional fact checker.

Below are some examples of check-worthy claims taken from U.S. general election presidential debates, you must generate one new claim for each given example.
The claims you generate must also fit into a debate context.
Each generated claim should match the style (lentgh, grammatical tense, sentiment, filler words, presence/absence of numbers, insults, missing context, etc.) of the correpsonding example.
Each claim you generate must not match the topic of the corresponding example.
Be creative in the choice of topics and style since the generated dataset should be varied.

### Examples:
{examples}

Now generate a new sample for each of the given examples. Put the samples in a list formatted like the examples above.

### Response: """
        self.neg_template = """### Instruction:
We are working on a natural language processing task to identify check-worthy claims in statements.
Your task is to generate synthetic data samples to augment a dataset which will be used to finetune a model to automatically detect such claims.
The claims you generate must be instances of the negative (non check-worthy) class.
A check-worthy claim has the following attributes:
- It contains a verifiable factual claim.
- It is of interest to the general public.
- It might be harmful to society.
- It contains information relevant to political discourse or policy decisions.
- It should be checked by a professional fact checker.
Thus the claims you generate should go against one or multiple of these attributes.

Below are some examples of non check-worthy claims taken from U.S. general election presidential debates, you must generate one new claim for each given example.
The claims you generate must also fit into a debate context.
Each generated claim should match the style (lentgh, grammatical tense, sentiment, filler words, presence/absence of numbers, insults, missing context, etc.) of the correpsonding example.
Each claim you generate must not match the topic of the corresponding example.
Be creative in the choice of topics and style since the generated dataset should be varied.

### Examples:
{examples}

Now generate a new sample for each of the given examples. Put the samples in a list formatted like the examples above.

### Response: """

    def get_pos_args_iterator(self):
        return self._get_iterator(positive_class=True)

    def get_neg_args_iterator(self):
        return self._get_iterator(positive_class=False)
    
    def _add_args(self, prompt_args: dict[str], metadata: dict[str]) -> dict[str]:
        """Method for sub-classes to fill args with additional parameters"""
        return prompt_args, metadata

    def _get_iterator(self, positive_class: bool):
        label = "Yes" if positive_class else "No"
        template = self.pos_template if positive_class else self.neg_template
        matching_data = self.source_data[self.source_data["class_label"]==label]
        shuffled_data = matching_data.sample(frac=1).reset_index(drop=True)

        while True:
            if len(shuffled_data) < self.num_per_turn:
                shuffled_data = matching_data.sample(frac=1).reset_index(drop=True)
            
            batch = shuffled_data.iloc[:self.num_per_turn]
            shuffled_data = shuffled_data.iloc[self.num_per_turn:]

            # Prepare prompt
            example_list = batch["Text"].tolist()
            examples_str = "\n".join(["- " + e for e in example_list])

            prompt_args = {"examples": examples_str}
            metadata = {"examples": example_list}
            
            # Let sub-classes add args and metadata
            prompt_args, metadata = self._add_args(prompt_args=prompt_args, metadata=metadata)

            # Create prompt with complete prompt_args
            prompt = template.format(**prompt_args)

            yield prompt, metadata, self.num_per_turn

class ExampleTopicPromptProvider(ExamplePromptProvider):
    """Fill templates with examples from the source data and randomly sampled topic"""
    def __init__(self, source_data: pd.DataFrame, num_per_turn: int, topics: list[str]):
        super().__init__(source_data, num_per_turn)
        self.topics = topics
        self.pos_template = """### Instruction:
We are working on a natural language processing task to identify check-worthy claims in statements.
Your task is to generate synthetic data samples, i.e. check-worthy claims, to augment a dataset which will be used to finetune a model to automatically detect such claims.
The claims you generate must be instances of the positive (check-worthy) class.
The new claims may or may not contain false information.
A check-worthy claim has the following attributes:
- It contains a verifiable factual claim.
- It is of interest to the general public.
- It might be harmful to society.
- It contains information relevant to political discourse or policy decisions.
- It should be checked by a professional fact checker.

Below are some examples of check-worthy claims taken from U.S. general election presidential debates, you must generate one new claim for each given example.
The claims you generate must also fit into a debate context.
Each generated claim should match the style (lentgh, grammatical tense, sentiment, filler words, presence/absence of numbers, insults, missing context, etc.) of the correpsonding example.
Your are provided a topic that all generated claims should follow.

### Examples:
{examples}

### Topic:
{topic}

Now generate a new sample for each of the given examples. Put the samples in a list formatted like the examples above.

### Response: """
        self.neg_template = """### Instruction:
We are working on a natural language processing task to identify check-worthy claims in statements.
Your task is to generate synthetic data samples to augment a dataset which will be used to finetune a model to automatically detect such claims.
The claims you generate must be instances of the negative (non check-worthy) class.
A check-worthy claim has the following attributes:
- It contains a verifiable factual claim.
- It is of interest to the general public.
- It might be harmful to society.
- It contains information relevant to political discourse or policy decisions.
- It should be checked by a professional fact checker.
Thus the claims you generate should go against one or multiple of these attributes.

Below are some examples of non check-worthy claims taken from U.S. general election presidential debates, you must generate one new claim for each given example.
The claims you generate must also fit into a debate context.
Each generated claim should match the style (lentgh, grammatical tense, sentiment, filler words, presence/absence of numbers, insults, missing context, etc.) of the correpsonding example.
Your are provided a topic that all generated claims should follow.

### Examples:
{examples}

### Topic:
{topic}

Now generate a new sample for each of the given examples. Put the samples in a list formatted like the examples above.

### Response: """
    
    def _add_args(self, prompt_args, metadata):
        topic = random.choice(self.topics)
        prompt_args["topic"] = topic
        metadata["topic"] = [topic] * self.num_per_turn

        return prompt_args, metadata


class SampleGenerator():
    def __init__(self, model: str):
        self.model = model

    def generate(self, prompt: str, num_samples: int) -> str|None:
        # Get completion
        response = completion(
            model=self.model,
            messages=[{"content": prompt, "role": "user"}]
        )

        # Parse the individual samples from the completion
        samples = response.choices[0].message.content
        samples = samples.split("\n")
        samples = [s.removeprefix('- ') for s in samples]

        if len(samples) == 0:
            return None

        if len(samples) != num_samples:
            return None
        
        return samples

class GenerationPipeline():
    def __init__(
            self,
            datasets: list[pd.DataFrame],
            template_arg_providers: list[PromptProvider],
            generator: SampleGenerator,
            augment_sizes: list[int],
            balance_classes: bool,
            results_dir: str):
        
        self.datasets = datasets
        self.arg_providers = template_arg_providers
        self.generator = generator
        self.augment_sizes = augment_sizes
        self.balance_classes = balance_classes
        self.results_dir = results_dir

    def _get_gen_sample_counts(self) ->  list[list[dict[str]]]:
        # For each dataset and augment size determine the number of positive and negative samples to generate
        num_gen_samples = []

        for d in self.datasets:
            num_pos_dataset = len(d[d["class_label"] == "Yes"])
            num_neg_dataset = len(d[d["class_label"] == "No"])
            pos_ratio = num_pos_dataset / len(d)
            diff_neg_pos = num_neg_dataset - num_pos_dataset
            num_gen_samples.append([])

            running_pos = 0
            running_neg = 0
            augment_size_diffs = [next_sz - sz for next_sz, sz in zip(self.augment_sizes, [0] + self.augment_sizes[:-1])]
            for sz_diff in augment_size_diffs:
                if self.balance_classes:
                    if abs(diff_neg_pos) > sz_diff:
                        raise ValueError(f"Augment size {sz_diff} does not suffice to balance dataset.")
                    
                    # Fill up less represented class until balance is reached
                    if diff_neg_pos < 0:
                        running_neg += abs(diff_neg_pos)
                        sz_diff -= abs(diff_neg_pos)
                        diff_neg_pos = 0
                    elif diff_neg_pos > 0:
                        running_pos += diff_neg_pos
                        sz_diff -= diff_neg_pos
                        diff_neg_pos = 0
                    
                    # Remaining samples are balanced
                    running_pos += sz_diff // 2
                    running_neg += sz_diff // 2
                else:
                    # Recreate pos/neg ratio in generated data
                    pos_add = round(sz_diff * pos_ratio) # Positive samples for this sequence step
                    running_pos += pos_add
                    running_neg += sz_diff - pos_add

                num_gen_samples[-1].append({"num_pos": running_pos, "num_neg": running_neg})
        
        return num_gen_samples
    
    def _assemble_sequences(self, components: dict[str, pd.DataFrame], counts: list[list[dict[str]]]):
        sequences = []

        for job_id, partial_dfs in components.items():
            real_df = partial_dfs["real"]
            synth_pos = partial_dfs["synth_pos"]
            synth_neg = partial_dfs["synth_neg"]
            # Start the sequence with the original dataset
            sequences.append([real_df])

            for step_cnt in counts[job_id]:
                mix = pd.concat([real_df, synth_pos.iloc[:step_cnt["num_pos"]], synth_neg.iloc[:step_cnt["num_neg"]]])
                sequences[-1].append(mix)
        
        return sequences
    
    def save_sequences(self, seqs: list[list[pd.DataFrame]]):
        # Remove all contents in results_dir
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

        # Save all sequences as CSVs
        for seq_idx, seq in enumerate(seqs):
            seq_dir = os.path.join(self.results_dir, f"sequence_{seq_idx}")
            os.makedirs(seq_dir, exist_ok=True)
            for aug_size, df in zip([0] + self.augment_sizes, seq):
                df.to_csv(os.path.join(seq_dir, f"seq_{seq_idx}_aug_{aug_size}.csv"), index=False)

    def _job(self, job_id:int, arg_iter: Iterable[dict[str]], num_samples: int, is_pos: bool) -> pd.DataFrame:
        synth_data = pd.DataFrame()
        pbar = tqdm(total=num_samples, desc=f"Dataset {job_id} {'pos' if is_pos else 'neg'}", leave=False)

        for prompt, metadata, num_per_turn in arg_iter:
            samples = self.generator.generate(prompt=prompt, num_samples=num_per_turn)

            if samples is None:
                continue
            
            out_dict = {
                "Text": samples,
                "class_label": ["Yes" if is_pos else "No"] * len(samples)
            }
            out_dict |= metadata
            out_df = pd.DataFrame(out_dict)

            synth_data = pd.concat([synth_data, out_df])
            pbar.n = min(len(synth_data), num_samples)
            pbar.last_print_n = pbar.n
            pbar.update(0)  # force refresh

            if len(synth_data) >= num_samples:
                synth_data = synth_data[:num_samples]
                pbar.n = num_samples
                pbar.last_print_n = num_samples
                pbar.update(0)
                pbar.close()
                return job_id, is_pos, synth_data
        pbar.close()
        
    def generate(self):
        num_gen_samples = self._get_gen_sample_counts()
        max_augment_per_sequence = [seq_sizes[-1] for seq_sizes in num_gen_samples]

        with ThreadPoolExecutor(max_workers=NUM_API_WORKERS) as executor:
            # Submit jobs with ascending id
            # Generate the max augment size for each dataset
            futures = [executor.submit(self._job, id, params[0].get_pos_args_iterator(), params[1]['num_pos'], True) for id, params in enumerate(zip(self.arg_providers, max_augment_per_sequence))]
            futures += [executor.submit(self._job, id, params[0].get_neg_args_iterator(), params[1]['num_neg'], False) for id, params in enumerate(zip(self.arg_providers, max_augment_per_sequence))]
            results = [future.result() for future in as_completed(futures)]

            # Create dict with positive and negative results and original data indexed by job_id
            results_dict = {i: {"real": ds} for i, ds in enumerate(self.datasets)}
            
            for job_id, is_pos, df in results:
                df["class_label"] = "Yes" if is_pos else "No"
                results_dict[job_id]["synth_pos" if is_pos else "synth_neg"] = df.reset_index(drop=True)

            seqs = self._assemble_sequences(components=results_dict, counts=num_gen_samples)
            self.save_sequences(seqs=seqs)


# Dataset params
source = "../data/CT24_checkworthy_english/train.csv"
dev_file = "../data/CT24_checkworthy_english/dev-wo-id.csv"
dev_test_file = "../data/CT24_checkworthy_english/dev-test-wo-id.csv"
test_file = "../data/CT24_checkworthy_english/test-combined-wo-id.csv"
gen_model = "openai/gpt-4o"
num_seq = 5
num_source_samples = 100
augment_sizes = sorted([100, 200, 400, 800])
topics = ["Healthcare", "Tax", "Economy", "Employment", "Education", "Energy", "Crime", "Military", "Trade", "Reproductive rights", "Guns", "Environment"]
num_examples_per_turn = 5
balance_source_classes = True
balance_gen_classes = False

# Upload params
artifact_base_name = "experiment_002"
artifact_description = f"Sequence created from {num_source_samples} CT24 samples using example prompting and topic guiding with {gen_model}."

# Train config params
num_seeds = 3
batch_size = 64
train_model = "roberta-base"
total_train_samples = 45000

# Enable/disable steps
do_generate = True
do_upload = True
make_configs = True
load_best_model = True

results_dir = "./sequences/" + artifact_base_name

if __name__ == "__main__":
        ## set ENV variables
    with open('secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()

    # Load and segment source data
    # subsets = SubsetDatasetProvider(
    #     source_path=source,
    #     num_subsets=num_seq,
    #     subset_size=num_source_samples,
    #     balance_classes=balance_source_classes).get_datasets()
    
    subsets = FileDatasetProvider(
        [
            'sequences/experiment_001/sequence_0/seq_0_aug_0.csv',
            'sequences/experiment_001/sequence_1/seq_1_aug_0.csv',
            'sequences/experiment_001/sequence_2/seq_2_aug_0.csv',
            'sequences/experiment_001/sequence_3/seq_3_aug_0.csv',
            'sequences/experiment_001/sequence_4/seq_4_aug_0.csv',
        ]
    ).get_datasets()

    # Setup pipeline
    arg_providers = [ExampleTopicPromptProvider(source_data=s, num_per_turn=num_examples_per_turn, topics=topics) for s in subsets]
    generator = SampleGenerator(model=gen_model)
    gen_strategy = GenerationPipeline(datasets=subsets, template_arg_providers=arg_providers, generator=generator, augment_sizes=augment_sizes, balance_classes=balance_gen_classes, results_dir=results_dir)

    # Generate
    if do_generate:
        gen_strategy.generate()

    # Upload resulting datasets and create train configs
    config_file_suffix = 0

    if make_configs:
        delete_configs()

    for root, _, files in os.walk(results_dir):
        for file in files:
            file_path = os.path.join(root, file)
            files = {
                "train": file_path,
                "dev": dev_file,
                "dev-test": dev_test_file,
                "test": test_file,
            }

            dataset_name = f"{artifact_base_name}_{file.replace('.csv', '')}"

            if do_upload:
                upload_dataset(
                    dataset_name=dataset_name,
                    description=artifact_description,
                    files=files,
                    metadata={
                        "model": gen_model,
                        "augment_sizes": augment_sizes,
                        "num_source_samples": num_source_samples,
                        "balanced_gen": balance_gen_classes,
                        "balanced_source": balance_source_classes,
                        "arg_provider": arg_providers[0].__class__.__name__}
                )

            if make_configs:
                dataset_size = len(pd.read_csv(file_path))
                num_epochs = total_train_samples // dataset_size

                for _ in range(num_seeds):
                    seed = random.randint(0, 2**16)
                    make_config(
                        model_name=train_model,
                        data_artifact_name=dataset_name,
                        group_name=artifact_base_name,
                        seed=seed,
                        batch_size=batch_size,
                        file_num=config_file_suffix,
                        num_epochs=num_epochs,
                        load_best_model=load_best_model)
                    
                    config_file_suffix += 1