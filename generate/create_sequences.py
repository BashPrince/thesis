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

# Create and upload augmented sequences
NUM_API_WORKERS = 5

def select_subsets(source_path: str, num_subsets: int, subset_size: int) -> list[pd.DataFrame]:
    df = pd.read_csv(source_path)
    df = df.drop(columns=["Sentence_id"], errors="ignore")
    subsets = []
    used_indices = set()
    for _ in range(num_subsets):
        available_indices = list(set(df.index) - used_indices)
        if len(available_indices) < subset_size:
            break
        selected_indices = pd.Series(available_indices).sample(subset_size, replace=False).tolist()
        used_indices.update(selected_indices)
        subset = df.loc[selected_indices].reset_index(drop=True)
        subsets.append(subset)

    return subsets

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
    
    def _add_args(self, prompt_args: dict[str], return_args: dict[str]) -> dict[str]:
        """Method for sub-classes to fill args with additional parameters"""
        return prompt_args, return_args

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
            args = {
                "examples": example_list,
                "num_samples" : self.num_per_turn
                }
            
            # Let sub-classes add args
            prompt_args, args = self._add_args(prompt_args=prompt_args, return_args=args)

            # Create prompt with complete prompt_args
            prompt = template.format(**prompt_args)
            args["prompt"] = prompt

            yield args

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
    
    def _add_args(self, prompt_args, return_args):
        topic = random.choice(self.topics)
        prompt_args["topic"] = topic
        return_args["topic"] = topic

        return prompt_args, return_args


class SampleGenerator():
    def __init__(self, model: str):
        self.model = model

    def generate(self, args: dict[str]) -> str|None:
        # Get completion
        response = completion(
            model=self.model,
            messages=[{"content": args["prompt"], "role": "user"}]
        )

        # Parse the individual samples from the completion
        samples = response.choices[0].message.content
        samples = samples.split("\n")
        samples = [s.removeprefix('- ') for s in samples]

        if len(samples) == 0:
            return None

        if len(samples) != args["num_samples"]:
            return None
        
        # Insert the generated samples into the args dict and return
        args["samples"] = samples

        return args

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
        pbar = tqdm(total=num_samples, desc=f"Job {job_id} {'pos' if is_pos else 'neg'}", leave=False)

        for args in arg_iter:
            out = self.generator.generate(args=args)

            if out is None:
                continue
            
            out_df = pd.DataFrame({
                "Text": out["samples"],
                "examples": out["examples"],
                "class_label": ["Yes" if is_pos else "No"] * len(out["samples"])
            })

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



source = "../data/CT24_checkworthy_english/train.csv"
results_dir = "./sequences"
dev_file = "../data/CT24_checkworthy_english/dev-wo-id.csv"
dev_test_file = "../data/CT24_checkworthy_english/dev-test-wo-id.csv"
test_file = "../data/CT24_checkworthy_english/test-combined-wo-id.csv"
model = "openai/gpt-4o"
num_seq = 5
num_source_samples = 100
augment_sizes = sorted([100, 200, 400, 800])
num_examples_per_turn = 5
balance_classes = True

artifact_base_name = "experiment_001"
artifact_description = ("Class-balanced sequence" if balance_classes else "Sequence") + f" created from {num_source_samples} CT24 samples using example prompting with {model}."

if __name__ == "__main__":
        ## set ENV variables
    with open('secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()

    subsets = select_subsets(source_path=source, num_subsets=num_seq, subset_size=num_source_samples)
    arg_providers = [ExamplePromptProvider(source_data=s, num_per_turn=num_examples_per_turn) for s in subsets]
    generator = SampleGenerator(model=model)
    gen_strategy = GenerationPipeline(datasets=subsets, template_arg_providers=arg_providers, generator=generator, augment_sizes=augment_sizes, balance_classes=balance_classes, results_dir=results_dir)
    #gen_strategy.generate()

    for root, _, files in os.walk(results_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, results_dir)
            files = {
                "train": rel_path,
                "dev": dev_file,
                "dev-test": dev_test_file,
                "test": test_file,
            }

            dataset_name = f"{artifact_base_name}_{file.replace('.csv', '')}"
            upload_dataset(
                dataset_name=dataset_name,
                description=artifact_description,
                files={rel_path: file_path},
                metadata={
                    "model": model,
                    "augment_sizes": augment_sizes,
                    "num_source_samples": num_source_samples,
                    "balanced": balance_classes,
                    "arg_provider": arg_providers[0].__class__.__name__}
            )
