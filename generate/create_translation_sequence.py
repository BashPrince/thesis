import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import litellm
import random
import shutil
from tqdm import tqdm

NUM_WORKERS = 5  # ThreadPoolExecutor max_workers
# Helper: LLM translation prompt
TRANSLATE_PROMPT = (
    "Translate the following sentence from {source} to {target}. Only output the translation, nothing else.\n\nSentence: {text}"
)

def call_llm(prompt: str, temperature: float, model: str) -> str:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response["choices"][0]["message"]["content"].strip()

def back_translate(sample_id: int, text: str, source_lang: str, intermediate_lang: str, temperature: float, model: str) -> str:
    prompt1 = TRANSLATE_PROMPT.format(source=source_lang, target=intermediate_lang, text=text)
    intermediate = call_llm(prompt1, temperature, model)
    prompt2 = TRANSLATE_PROMPT.format(source=intermediate_lang, target=source_lang, text=intermediate)
    back_translated = call_llm(prompt2, temperature, model)

    return sample_id, back_translated

def process_dataset(file_path: str, output_dir: str, sequence: int, aug_factors: List[int], method: int, source_lang: str, intermediate_langs: str|list[str], temperature: float, model: str):
    if isinstance(intermediate_langs, list):
        assert len(intermediate_langs) == aug_factors[-1]
    else:
        intermediate_langs = [intermediate_langs] * aug_factors[-1]

    df = pd.read_csv(file_path)
    max_augment = max(aug_factors)

    # Extract "Text" column as a list of strings
    text_list = df["Text"].astype(str).tolist()
    augmentations = []
    
    pbar = tqdm(desc=f"Dataset {sequence}", total=len(text_list) * aug_factors[-1])

    for a in range(max_augment):
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for i, sample in enumerate(text_list):
                if len(augmentations) <= i:
                    augmentations.append([])
                    prev_augment = sample
                else:
                    prev_augment = augmentations[i][-1]
                
                futures.append(
                    executor.submit(
                        back_translate,
                        i,
                        prev_augment if method == 2 else sample,
                        source_lang,
                        intermediate_langs[a],
                        temperature,
                        model
                    ))

            for future in as_completed(futures):
                try:
                    i, translation = future.result(timeout=10)
                except TimeoutError:
                    print(f"Sample {i} timed out")
                augmentations[i].append(translation)
                pbar.update(1)
        
    pbar.close()

    
    augmented_dfs = []
    for _ in aug_factors:
        df_copy = df.copy()
        df_copy["source"] = ""
        augmented_dfs.append(df_copy)

    labels_list = df["class_label"].astype(str).tolist()

    for sample, label, translations in zip(text_list, labels_list, augmentations):
        random.shuffle(translations)
        for factor, aug_df in zip(aug_factors, augmented_dfs):
            for t in translations[:factor]:
                aug_df.loc[len(aug_df)] = [t, label, sample]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    for factor, aug_df in zip([0] + aug_factors, [df] + augmented_dfs):
        output_path = os.path.join(output_dir, f"seq_{sequence}_aug_{factor * len(df)}.csv")
        aug_df.to_csv(output_path, index=False)



def main():
    ## set ENV variables
    with open('secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()

    # Parameters (set these as needed)
    output_dir = "sequences/experiment_007"  # Directory to save augmented datasets
    aug_factors = [1, 2, 4, 8]  # List of augmentation factors for partial saves
    source_lang = "English"  # Source language for back-translation
    # intermediate_lang = "German"  # Intermediate language for back-translation
    intermediate_langs = ["German", "German", "Spanish", "Spanish", "French", "French", "Dutch", "Dutch"]
    method = 1  # 1: always real sample, 2: chain augmentations
    temperature = 1.0  # LLM temperature
    model = "gpt-4o"  # LLM model name
    input_files = [
        "sequences/experiment_001/sequence_0/seq_0_aug_0.csv",
        "sequences/experiment_001/sequence_1/seq_1_aug_0.csv",
        "sequences/experiment_001/sequence_2/seq_2_aug_0.csv",
        "sequences/experiment_001/sequence_3/seq_3_aug_0.csv",
        "sequences/experiment_001/sequence_4/seq_4_aug_0.csv",
    ]

    # Check if output_dir exists and prompt for deletion
    if os.path.exists(output_dir):
        response = input(f"Output directory '{output_dir}' already exists. Delete it? (y/n): ").strip().lower()
        if response == "y":
            shutil.rmtree(output_dir)
        else:
            print("Aborting. Please choose a different output directory or delete it manually.")
            return
        
    for i, fname in enumerate(input_files):
        process_dataset(
            fname,
            os.path.join(output_dir, f"sequence_{i}"),
            i,
            aug_factors,
            method,
            source_lang,
            intermediate_langs,
            temperature,
            model
        )

if __name__ == "__main__":
    main()
