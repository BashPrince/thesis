from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig
import pandas as pd
import gc
from collections import Counter
import random
import numpy as np

gc.collect()


def generate_test_prompt(sample):
    system_message = ("Read the statement provided below."
                      "Your task is to evaluate whether the statement contains information or claims"
                      "that are worthy to be verified through fact-checking. "
                      "If the statement presents assertions, facts, or claims that would "
                      "benefit from verification, respond with 'Yes'. "
                      "If the statement is purely opinion-based, trivial, or does"
                      "not contain any verifiable claims, respond with 'No'.")
    input_text = sample["Text"]

    full_prompt = ""
    full_prompt += "### Instruction:"
    full_prompt += "\n" + system_message
    full_prompt += "\n\n### Input Sentence:"
    full_prompt += "\n" + input_text
    full_prompt += "\n\n### Response:"

    sample['prompt'] = full_prompt
    return sample


def generate_response(sample, model, j):
    prompt = sample['prompt']
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')
    output = model.generate(**model_inputs, max_new_tokens=3,
                            pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.3)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded_output.replace(prompt, "").strip()

    if "No" in response:
        sample["prediction" + str(j)] = "No"
    elif "Yes" in response:
        sample["prediction" + str(j)] = "Yes"
    else:
        print("Error: ", response)
        exit(0)
    return sample


def get_majority(sample, iterations):
    predictions = []

    for i in range(iterations):
        predictions.append(sample["prediction" + str(i)])

    counter = Counter(predictions)
    majority, count = counter.most_common()[0]
    sample['prediction'] = majority
    return sample


def get_consistency(predictions):
    transpose = list(zip(*predictions))

    consistent_count = 0
    for sub_list in transpose:
        if all(sub_list[0] == element for element in sub_list):
            consistent_count += 1

    consistency = consistent_count / (len(predictions[0]))
    return consistency


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

test_data = pd.read_csv("../data/CT24_checkworthy_english_test_gold.tsv", sep='\t', on_bad_lines='skip')

print("Testing dataset size: ", test_data.shape)

# add the "prompt" column in the dataset
test_data = test_data.apply(generate_test_prompt, axis=1)

# Load model and config
checkpoint_path = "../model/checkpoint"

# 4-bit Quantization Configuration
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=compute_dtype
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)


tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, cache_dir='../hf_cache')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

set_random_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    device_map='auto',
    quantization_config=quant_config,
    cache_dir='../hf_cache'
)
model.config.use_cache = False
model.config.pretraining_tp = 1

model.eval()

for j in range(5):
    with torch.no_grad():
        test_data = test_data.apply(lambda sample: generate_response(sample, model, j), axis=1)

test_data = test_data.apply(lambda sample: get_majority(sample, 5), axis=1)
prediction = test_data['prediction']

predictions_t = []
for j in range(5):
    predictions_t.append(test_data['prediction' + str(j)].tolist())
test_data.to_csv('../results/predict.csv', index=False)

