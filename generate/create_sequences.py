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
    def get_train_datasets(self) -> list[pd.DataFrame]:
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
    
    def get_train_datasets(self):
        return self.datasets

class TopicDatasetProvider(DatasetProvider):
    """Split datasets by topic"""
    def __init__(self, train_path: str, dev_path: str, test_path: str):
        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)

        # We'll use the same test set for all later training runs
        self.test_df = pd.read_csv(test_path)
        
        # Only keep columns of interest
        train_df = train_df[["Text", "class_label", "topic"]]
        dev_df = dev_df[["Text", "class_label", "topic"]]
        self.test_df = self.test_df[["Text", "class_label", "topic"]]

        # Split topics for train and dev
        self.topic_dfs_train = {topic: train_df[train_df["topic"] == topic].reset_index(drop=True) for topic in train_df["topic"].unique()}
        self.topic_dfs_dev = {topic: dev_df[dev_df["topic"] == topic].reset_index(drop=True) for topic in dev_df["topic"].unique()}
        self.topics_sorted = sorted(self.topic_dfs_train.keys())

    def get_topics(self) -> list[str]:
        return self.topics_sorted
    
    def get_train_datasets(self) -> list[pd.DataFrame]:
        return [self.topic_dfs_train[t] for t in self.topics_sorted]
    
    def get_train_dataset(self, topic: str) -> pd.DataFrame:
        return self.topic_dfs_train[topic]
    
    def get_dev_dataset(self, topic: str) -> pd.DataFrame:
        return self.topic_dfs_dev[topic]
    
    def get_test_dataset(self) -> pd.DataFrame:
        return self.test_df
    
class DuplicateTopicDatasetProvider(TopicDatasetProvider):
    def __init__(self, train_path: str, dev_path: str, test_path: str):
        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)

        # We'll use the same test set for all later training runs
        self.test_df = pd.read_csv(test_path)
        
        # Only keep columns of interest
        train_df = train_df[["Text", "class_label", "topic"]]
        dev_df = dev_df[["Text", "class_label", "topic"]]
        self.test_df = self.test_df[["Text", "class_label", "topic"]]

        self.unique_topics = list(train_df["topic"].unique())
        self.duplicated_train_datasets = []
        self.topic_dfs_train = {}
        self.topic_dfs_dev = {}

        # Provide base topic for each combo of base and augment topic
        for base_topic in self.unique_topics:
            base_topic_train_data = train_df[train_df["topic"] == base_topic].reset_index(drop=True)
            base_topic_dev_data = dev_df[dev_df["topic"] == base_topic].reset_index(drop=True)
            self.topic_dfs_train[base_topic] = base_topic_train_data
            self.topic_dfs_dev[base_topic] = base_topic_dev_data

            for augment_topic in self.unique_topics:
                self.duplicated_train_datasets.append(base_topic_train_data)
        
    def get_train_datasets(self):
        return self.duplicated_train_datasets
    
    def get_topics(self):
        return self.unique_topics
    
    def get_train_dataset(self, topic: str) -> pd.DataFrame:
        return self.topic_dfs_train[topic]
    
    def get_dev_dataset(self, topic: str) -> pd.DataFrame:
        return self.topic_dfs_dev[topic]
    
    def get_test_dataset(self) -> pd.DataFrame:
        return self.test_df
    
    def get_augment_topics(self):
        # Return repeating topics
        return self.unique_topics * len(self.unique_topics)

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

class RephrasePromptProvider(ExamplePromptProvider):
    def __init__(self, source_data: pd.DataFrame, num_per_turn: int):
        super().__init__(source_data=source_data, num_per_turn=num_per_turn)
        self.pos_template = """Rephrase the following sentences while keeping the meaning of each sentence the same.
Be creative.
Put the rephrased samples in a list formatted like the input below. Do not produce additional output.
{examples}"""
        self.neg_template = self.pos_template

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

class TopicPromptProvider(PromptProvider):
    """Fill templates with fixed topic and randomly sampled properties"""
    def __init__(self, properties_path: str, num_per_turn: int, num_properties:int, topic: str):
        self.topic = topic
        self.num_per_turn = num_per_turn
        self.num_properties = num_properties
        with open(properties_path, 'r') as properties_file:
            self.properties = json.load(properties_file)

        self.pos_template = """### Instruction:
We are working on a natural language processing task to identify check-worthy claims in statements.
Your task is to generate synthetic data samples, i.e. check-worthy claims.
The claims you generate must be instances of the positive (check-worthy) class.
The new claims may or may not contain false information, i.e. you can make up facts.
When generating a new sample make it different from already generated samples.

### Properties:
A check-worthy claim has the following general attributes, when generating a claim make sure it does not violate these attributes, this is VERY IMPORTANT:
1. It makes a consequential assertion that is of interest to the general public.
2. It contains a factual statement which can be verified or disproved using publicly accessible sources.
3. It would take an average person substantial time and effort to fact check the statement.

In addition, the claims you generate should follow these thematic and stylistic properties:
- Do not include claims about the future.
- Do not wrap the entire sentence in quotation marks.
{properties}

### Topic:
The claim should be about {topic}.

### Formatting:
Each claim should be preceded by a few sentences of context (e.g. made up dialog or sentences in an article leading up to a claim) and the delimiter symbol ">>". The final sentence should then contain the claim.
Format your response as a list following the given structure:
{format_example}

Now generate {num_examples} checkworthy claims.

### Response: """
        self.neg_template = """### Instruction:
We are working on a natural language processing task to identify check-worthy claims in statements.
Your task is to generate synthetic counter examples, i.e. non-check-worthy claims.
The claims you generate must be instances of the negative (non-check-worthy) class.
The new claims may or may not contain false information, i.e. you can make up facts.
When generating a new sample make it different from already generated samples.

### Properties:
A check-worthy claim has the following general attributes:
1. It makes a consequential assertion that is of interest to the general public.
2. It contains a factual statement which can be verified or disproved using publicly accessible sources.
3. It would take an average person substantial time and effort to fact check the statement.

In addition, the claims you generate should follow these thematic and stylistic properties:
- Do not include claims about the future.
- Do not wrap the entire sentence in quotation marks.
- Do not cite authoritative sounding sources.
{properties}

### Topic:
The claim should be about {topic}.

### Formatting:
Each claim should be preceded by a few sentences of context (e.g. made up dialog or sentences in an article leading up to a claim) and the delimiter symbol ">>". The final sentence should then contain the claim.
Format your response as a list following the given structure:
{format_example}

Now generate {num_examples} non-check-worthy claims.

### Response: """

    def get_pos_args_iterator(self):
        return self._get_iterator(positive_class=True)

    def get_neg_args_iterator(self):
        return self._get_iterator(positive_class=False)
    
    def _get_iterator(self, positive_class: bool):
        template = self.pos_template if positive_class else self.neg_template

        while True:
            property_categories = list(self.properties.keys())
            # Do not allow sampling topic and violation category
            property_categories.remove("topic")
            property_categories.remove("violation")
            property_keys = random.sample(property_categories, k=self.num_properties)
            sampled_properties = []
            for k in property_keys:
                prop_category = self.properties[k]
                prop_template = prop_category["template"]
                prop_feature = random.choice(prop_category["features"])
                sampled_properties.append(prop_template.format(prop_feature))

            properties_joined = "\n".join([f"- {s}" for s in sampled_properties])
            
            prompt_args = {
                "properties": properties_joined,
                "num_examples": self.num_per_turn,
                "topic": self.topic
            }
            
            format_example = "\n".join([f"- Context sentences. >> Claim {i}." for i in range(1, self.num_per_turn + 1)])
            prompt_args["format_example"] = format_example

            # Create prompt with complete prompt_args
            prompt = template.format(**prompt_args)
            metadata = {
                #"properties": property_keys,
                "topic": [self.topic] * self.num_per_turn
            }

            yield prompt, metadata, self.num_per_turn


class SampleGenerator():
    def __init__(self, model: str):
        self.model = model
    
    def parse(self, response: str, num_samples: int) -> dict[str, list[str]]|None:
        samples = response.split("\n")
        samples = [s.removeprefix('- ') for s in samples]

        if len(samples) == 0:
            return None

        if len(samples) != num_samples:
            return None
        
        return {"Text": samples}


    def generate(self, prompt: str, num_samples: int) -> dict[str, list[str]]|None:
        # Get completion
        response = completion(
            model=self.model,
            messages=[{"content": prompt, "role": "user"}]
        )

        # Parse the individual samples from the completion
        response = response.choices[0].message.content

        return self.parse(response=response, num_samples=num_samples)

class TopicSampleGenerator(SampleGenerator):
    def parse(self, response, num_samples):
        samples = response.split("\n")
        samples = [s.removeprefix('- ') for s in samples]
        samples = [s for s in samples if s]

        if len(samples) == 0:
            return None

        # Strip whitespace
        samples = [s for s in samples if s.strip() != ""]

        if len(samples) != num_samples:
            return None
        
        # Split each sample along delimiter.
        samples = [s.split(">>") for s in samples]
        # Include only samples that have one occurence of the delimiter.
        samples = [s for s in samples if len(s) == 2]

        sample_context = [s[0].strip() for s in samples]
        sample_text = [s[1].strip() for s in samples]

        return {"Text": sample_text, "context": sample_context}

class ExampleGenerationPipeline():
    def __init__(
            self,
            dataset_provider: DatasetProvider,
            template_arg_providers: list[PromptProvider],
            generator: SampleGenerator,
            augment_sizes: list[int],
            balance_classes: bool,
            results_dir: str):
        
        self.dataset_provider = dataset_provider
        self.datasets = dataset_provider.get_train_datasets()
        self.arg_providers = template_arg_providers
        self.generator = generator
        self.augment_sizes = augment_sizes
        self.balance_classes = balance_classes
        self.results_dir = results_dir
        self.gen_counts = self._get_gen_sample_counts()

    def _get_gen_sample_counts(self) ->  list[list[dict[str]]]:
        """For each dataset and augment size determine the number of positive and negative samples to generate"""

        augment_size_diffs = [next_sz - sz for next_sz, sz in zip(self.augment_sizes, [0] + self.augment_sizes[:-1])]
        num_gen_samples = []

        for d in self.datasets:
            num_pos_dataset = len(d[d["class_label"] == "Yes"])
            num_neg_dataset = len(d[d["class_label"] == "No"])
            pos_ratio = num_pos_dataset / len(d)
            diff_neg_pos = num_neg_dataset - num_pos_dataset
            num_gen_samples.append([])

            running_pos = 0
            running_neg = 0
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
    
    def _get_max_gen_sample_counts(self):
        """Get the maximum sample counts to generate for each sequence"""
        return [seq_sizes[-1] for seq_sizes in self.gen_counts]
    
    def _assemble_sequences(self, components: dict[int, dict[str, pd.DataFrame]]):
        sequences = []

        for job_id, partial_dfs in components.items():
            real_df = partial_dfs["real"]
            synth_pos = partial_dfs["synth_pos"]
            synth_neg = partial_dfs["synth_neg"]
            # Start the sequence with the original dataset
            sequences.append([real_df])

            for step_cnt in self.gen_counts[job_id]:
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
            out_dict = self.generator.generate(prompt=prompt, num_samples=num_per_turn)

            if out_dict is None:
                continue
            
            out_dict["class_label"] = ["Yes" if is_pos else "No"] * len(out_dict["Text"])
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
        max_augment_per_sequence = self._get_max_gen_sample_counts()

        with ThreadPoolExecutor(max_workers=NUM_API_WORKERS) as executor:
            # Submit jobs with ascending id
            # Generate the max augment size for each dataset
            futures = [executor.submit(self._job, id, params[0].get_pos_args_iterator(), params[1]['num_pos'], True) for id, params in enumerate(zip(self.arg_providers, max_augment_per_sequence))]
            futures += [executor.submit(self._job, id, params[0].get_neg_args_iterator(), params[1]['num_neg'], False) for id, params in enumerate(zip(self.arg_providers, max_augment_per_sequence))]
            results = [future.result() for future in as_completed(futures)]

            # Create dict with positive and negative results and original data indexed by job_id
            results_dict = {i: {"real": ds} for i, ds in enumerate(self.datasets)}
            
            for job_id, is_pos, df in results:
                results_dict[job_id]["synth_pos" if is_pos else "synth_neg"] = df.reset_index(drop=True)

            seqs = self._assemble_sequences(components=results_dict)
            self.save_sequences(seqs=seqs)
    
    def upload(self, artifact_base_name: str, artifact_description: str, metadata: dict[str], dev_file: str, dev_test_file: str, test_file: str):
        for root, _, files in os.walk(self.results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                files = {
                    "train": file_path,
                    "dev": dev_file,
                    "dev-test": dev_test_file,
                    "test": test_file,
                }

                dataset_name = f"{artifact_base_name}_{file.replace('.csv', '')}"

                upload_dataset(
                    dataset_name=dataset_name,
                    description=artifact_description,
                    files=files,
                    metadata=metadata)
    
    def make_config(self, artifact_base_name: str, total_train_samples: int, train_model: str, batch_size: int, load_best_model: bool):
        delete_configs()
        config_file_suffix = 0

        for root, _, files in os.walk(self.results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                dataset_name = f"{artifact_base_name}_{file.replace('.csv', '')}"

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

class CorrelationGenerationPipeline(ExampleGenerationPipeline):
    """Create augmented and non-augmented datasets for cross-topic training"""
    def __init__(
            self,
            dataset_provider: TopicDatasetProvider,
            template_arg_providers: list[PromptProvider],
            generator: SampleGenerator,
            augment_size: int,
            results_dir: str):
        
        self.dataset_provider = dataset_provider
        self.datasets = dataset_provider.get_train_datasets()
        self.arg_providers = template_arg_providers
        self.generator = generator
        self.augment_sizes = [augment_size] # Extend to array to make this work with superclass
        self.balance_classes = False
        self.results_dir = results_dir
        self.gen_counts = self._get_gen_sample_counts()

    def _assemble_sequences(self, components: dict[int, dict[str, pd.DataFrame]]):
        sequences = []

        for topic in self.dataset_provider.get_topics():
            train_data = self.dataset_provider.get_train_dataset(topic=topic)
            dev_data = self.dataset_provider.get_dev_dataset(topic=topic)
            # For each dataset we want to save an unchanged version and one augmented version for each topic
            # Unchanged
            sequences.append([{
                "train": train_data,
                "dev": dev_data,
                "test": self.dataset_provider.get_test_dataset(),
                "name": topic
            }])

            # Augmentation for each topic
            for j, partial_dfs in components.items():
                augment_topic = self.dataset_provider.get_topics()[j]
                synth_pos = partial_dfs["synth_pos"]
                synth_neg = partial_dfs["synth_neg"]
                augmented_dataset = pd.concat([train_data, synth_pos, synth_neg])

                sequences[-1].append({
                    "train": augmented_dataset,
                    "dev": dev_data, # Use topic of real dataset for dev
                    "test": self.dataset_provider.get_test_dataset(),
                    "name": f"{topic}_{augment_topic}"
                })
        
        return sequences

    
    def save_sequences(self, seqs: list[list[dict[str]]]):
        # Remove all contents in results_dir
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

        # Save train set once
        test_set = seqs[0][0]["test"]
        test_set.to_csv(os.path.join(self.results_dir, "test.csv"), index=False)

        # Save all topics as CSVs
        for topic_seq in seqs:
            # Create a directory and save the dev set
            data_dict = topic_seq[0]
            topic_dir = os.path.join(self.results_dir, f"{data_dict['name']}")
            os.makedirs(topic_dir, exist_ok=True)
            dev_set = topic_seq[0]["dev"]
            dev_set.to_csv(os.path.join(topic_dir, "dev.csv"), index=False)
            for data_dict in topic_seq:
                train_set = data_dict["train"]
                train_set.to_csv(os.path.join(topic_dir, f"{data_dict['name']}.csv"), index=False)

    def upload(self, artifact_base_name: str, artifact_description: str, metadata: dict[str]):
        for root, _, files in os.walk(self.results_dir):
            for file in files:
                # Skip the dev and test files
                if file == "test.csv" or file == "dev.csv":
                    continue

                train_path = os.path.join(root, file)
                dev_path = os.path.join(root, "dev.csv")
                test_path = os.path.join(self.results_dir, "test.csv")
                files = {
                    "train": train_path,
                    "dev": dev_path,
                    "dev-test": test_path,
                    "test": test_path,
                }

                dataset_name = f"{artifact_base_name}_{file.replace('.csv', '')}"

                upload_dataset(
                    dataset_name=dataset_name,
                    description=artifact_description,
                    files=files,
                    metadata=metadata)
                
    def make_config(self, artifact_base_name: str, total_train_samples: int, train_model: str, batch_size: int, load_best_model: bool):
        delete_configs()
        config_file_suffix = 0

        for root, _, files in os.walk(self.results_dir):
            for file in files:
                # Skip the dev and test files
                if file == "test.csv" or file == "dev.csv":
                    continue
                
                file_path = os.path.join(root, file)
                dataset_name = f"{artifact_base_name}_{file.replace('_train', '').replace('.csv', '')}"

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

class ExampleCorrelationGenerationPipeline(CorrelationGenerationPipeline):
    def __init__(
            self,
            dataset_provider: DuplicateTopicDatasetProvider,
            template_arg_providers: list[PromptProvider],
            generator: SampleGenerator,
            augment_size: int,
            results_dir: str):
        
        self.dataset_provider = dataset_provider
        self.datasets = dataset_provider.get_train_datasets()
        self.arg_providers = template_arg_providers
        self.generator = generator
        self.augment_sizes = [augment_size] # Extend to array to make this work with superclass
        self.balance_classes = False
        self.results_dir = results_dir
        self.gen_counts = self._get_gen_sample_counts()

    def _assemble_sequences(self, components: dict[int, dict[str, pd.DataFrame]]):
        base_topics_extended = []
        num_topics = self.dataset_provider.get_topics()

        # Repeat-interleave base topics
        for base_topic in self.dataset_provider.get_topics():
            base_topics_extended += [base_topic] * len(num_topics)

        sequences = {}

        for topic in self.dataset_provider.get_topics():
            train_data = self.dataset_provider.get_train_dataset(topic=topic)
            dev_data = self.dataset_provider.get_dev_dataset(topic=topic)
            # For each dataset we want to save a version without augmentation
            sequences[topic] = [{
                "train": train_data,
                "dev": dev_data,
                "test": self.dataset_provider.get_test_dataset(),
                "name": topic
            }]

        # Augmentation for each topic
        for i, partial_dfs in components.items():
            base_topic = base_topics_extended[i]
            train_data = self.dataset_provider.get_train_dataset(topic=base_topic)
            dev_data = self.dataset_provider.get_dev_dataset(topic=base_topic)
            augment_topic = self.dataset_provider.get_augment_topics()[i]
            synth_pos = partial_dfs["synth_pos"]
            synth_neg = partial_dfs["synth_neg"]
            augmented_dataset = pd.concat([train_data, synth_pos, synth_neg])

            sequences[base_topic].append({
                "train": augmented_dataset,
                "dev": dev_data, # Use topic of real dataset for dev
                "test": self.dataset_provider.get_test_dataset(),
                "name": f"{base_topic}_{augment_topic}"
            })
        
        return [topic_seq for _, topic_seq in sequences.items()]

# Dataset params
source = "../data/CT24_checkworthy_english/train.csv"
dev_file = "../data/CT24_checkworthy_english/dev-wo-id.csv"
dev_test_file = "../data/CT24_checkworthy_english/dev-test-wo-id.csv"
test_file = "../data/CT24_checkworthy_english/test-combined-wo-id.csv"
gen_model = "openai/gpt-4o"
num_seq = 5
num_source_samples = 100
augment_sizes = sorted([100, 200, 400, 800]) # For examples
augment_size = 314 # For correlations
topics = ["Healthcare", "Tax", "Economy", "Employment", "Education", "Energy", "Crime", "Military", "Trade", "Reproductive rights", "Guns", "Environment"]
num_examples_per_turn = 5
num_properties = 3
balance_source_classes = True
balance_gen_classes = False

# Upload params
artifact_base_name = "experiment_007"
artifact_description = f"Sequence for back-translation experiment with {gen_model}."

# Train config params
num_seeds = 3
batch_size = 64
train_model = "roberta-base"
total_train_samples = 45000
load_best_model = True

# Enable/disable steps
do_generate = False
do_upload = True
make_configs = True

results_dir = "./sequences/" + artifact_base_name

if __name__ == "__main__":
        ## set ENV variables
    with open('secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()

    # Setup pipeline

    # dataset_provider = SubsetDatasetProvider(
    #     source_path=source,
    #     num_subsets=num_seq,
    #     subset_size=num_source_samples,
    #     balance_classes=balance_source_classes)
    
    dataset_provider = FileDatasetProvider(
        [
            'sequences/experiment_001/sequence_0/seq_0_aug_0.csv',
            'sequences/experiment_001/sequence_1/seq_1_aug_0.csv',
            'sequences/experiment_001/sequence_2/seq_2_aug_0.csv',
            'sequences/experiment_001/sequence_3/seq_3_aug_0.csv',
            'sequences/experiment_001/sequence_4/seq_4_aug_0.csv',
        ]
    )
    # dataset_provider = TopicDatasetProvider(
    #     train_path="../data/CT24_checkworthy_english/topic_correlation/train.csv",
    #     dev_path="../data/CT24_checkworthy_english/topic_correlation/dev.csv",
    #     test_path="../data/CT24_checkworthy_english/topic_correlation/test.csv",
    # )
    # dataset_provider = DuplicateTopicDatasetProvider(
    #     train_path="../data/CT24_checkworthy_english/topic_correlation/train.csv",
    #     dev_path="../data/CT24_checkworthy_english/topic_correlation/dev.csv",
    #     test_path="../data/CT24_checkworthy_english/topic_correlation/test.csv",
    # )

    # arg_providers = [ExamplePromptProvider(source_data=s, num_per_turn=num_examples_per_turn, topics=topics) for s in dataset_provider.get_train_datasets()]
    #arg_providers = [TopicPromptProvider(properties_path="./templates/properties.json", num_per_turn=num_examples_per_turn, num_properties=num_properties, topic=t) for t in dataset_provider.get_topics()]

    arg_providers = []
    for data in dataset_provider.get_train_datasets():
        # Create arg provider with dataset as example source and single augment topic
        arg_providers.append(RephrasePromptProvider(source_data=data, num_per_turn=num_examples_per_turn))

    generator = SampleGenerator(model=gen_model)
    gen_strategy = ExampleGenerationPipeline(dataset_provider=dataset_provider, template_arg_providers=arg_providers, generator=generator, augment_sizes=augment_sizes, balance_classes=balance_gen_classes, results_dir=results_dir)
    #gen_strategy = CorrelationGenerationPipeline(dataset_provider=dataset_provider, template_arg_providers=arg_providers, generator=generator, augment_size=augment_size, results_dir=results_dir)
    # gen_strategy = ExampleCorrelationGenerationPipeline(dataset_provider=dataset_provider, template_arg_providers=arg_providers, generator=generator, augment_size=augment_size, results_dir=results_dir)

    # Generate
    if do_generate:
        if os.path.exists(results_dir):
            proceed = input(f"Warning: results_dir '{results_dir}' already exists and will be overwritten. Proceed? (y/n): ")
            if proceed.lower() != 'y':
                print("Aborting.")
                exit(0)

        gen_strategy.generate()

    if do_upload:
        metadata = {
            "model": gen_model,
            "augment_sizes": augment_sizes,
            "num_source_samples": num_source_samples,
            "balanced_gen": balance_gen_classes,
            "balanced_source": balance_source_classes,
            "arg_provider": arg_providers[0].__class__.__name__}
        
        gen_strategy.upload(
            artifact_base_name=artifact_base_name,
            artifact_description=artifact_description,
            metadata=metadata,
            dev_file=dev_file,
            dev_test_file=dev_test_file,
            test_file=test_file
        )
    
    if make_configs:
        gen_strategy.make_config(
            artifact_base_name=artifact_base_name,
            total_train_samples=total_train_samples,
            train_model=train_model,
            batch_size=batch_size,
            load_best_model=load_best_model
        )