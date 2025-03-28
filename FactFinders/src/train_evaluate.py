from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import TrainingArguments
from trl import SFTTrainer
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import gc
import os.path
import statistics
import csv
from collections import Counter
import random
import numpy as np
import argparse

gc.collect()


def generate_prompt(sample):
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
    full_prompt += "\n" + sample["class_label"]

    sample['prompt'] = full_prompt
    return sample


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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate a fine-tuned language model.')
parser.add_argument('--model_id', type=str, required=True, help='The model id of a pre-trained model.')
args = parser.parse_args()


train_data = pd.read_csv("../data/CT24_checkworthy_english_train.tsv", sep='\t', on_bad_lines='skip')
validation_data = pd.read_csv("../data/CT24_checkworthy_english_dev.tsv", sep='\t', on_bad_lines='skip')
dev_test_data = pd.read_csv("../data/CT24_checkworthy_english_dev-test.tsv", sep='\t', on_bad_lines='skip')
test_data = pd.read_csv("../data/CT24_checkworthy_english_test_gold.tsv", sep='\t', on_bad_lines='skip')

training_data_name = "CT24_checkworthy_english_train.tsv"
print("Training dataset size: ", train_data.shape)
print("Validation dataset size: ", validation_data.shape)
print("Dev Testing dataset size: ", dev_test_data.shape)
print("Testing dataset size: ", test_data.shape)

# add the "prompt" column in the dataset
train_data = Dataset.from_pandas(train_data.apply(generate_prompt, axis=1))
validation_data = Dataset.from_pandas(validation_data.apply(generate_prompt, axis=1))
dev_test_data = dev_test_data.apply(generate_test_prompt, axis=1)
test_data = test_data.apply(generate_test_prompt, axis=1)

# Load model and config
model_id = args.model_id
run_name = model_id.split("/")[-1].replace("-hf", "").replace("-", "_") + "_evaluation"
print(f'Model: {args.model_id}, run_name: {run_name}')

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

args = TrainingArguments(
    output_dir="../models/" + run_name,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=25,
    optim="paged_adamw_32bit",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    bf16=False,
    fp16=False,
    weight_decay=0.001,
    max_grad_norm=0.3, max_steps=-1, warmup_ratio=0.03, group_by_length=True,
    run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    lr_scheduler_type='constant'
)


tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='../hf_cache')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

train_data = train_data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
validation_data = validation_data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
print("Training dataset size: ", train_data.shape)
print("validation dataset size: ", validation_data.shape)


evaluation_file = run_name + ".csv"
columns = ['Model', 'Dataset', 'Dataset Size', 'Running Time', 'Accuracy-T', 'Accuracy-STD-T', 'Precision-T',
           'Precision-STD-T', 'Recall-T', 'Recall-STD-T', 'F1-T', 'F1-STD-T', 'Consistency-T', 'Consistency-STD-T',
           'Accuracy-DT', 'Accuracy-STD-DT', 'Precision-DT', 'Precision-STD-DT', 'Recall-DT', 'Recall-STD-DT', 'F1-DT',
           'F1-STD-DT', 'Consistency-DT', 'Consistency-STD-DT']

if not os.path.exists(evaluation_file):
    with open(evaluation_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

accuracy_t = []
precision_t = []
recall_t = []
f1_score_t = []
consistency_t = []
accuracy_dt = []
precision_dt = []
recall_dt = []
f1_score_dt = []
consistency_dt = []
running_time = []

# Only one training iteration (authors did 3)
for i in range(1):
    print("Iteration - ", i)
    set_random_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        quantization_config=quant_config,
        cache_dir='../hf_cache'
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=None,
        tokenizer=tokenizer,
        packing=False,
        args=args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        dataset_text_field='prompt'
    )

    gc.collect()
    torch.cuda.empty_cache()

    start_time = datetime.now()
    print("Training started at : ", start_time.strftime('%Y-%m-%d-%H-%M'))
    trainer.train()
    end_time = datetime.now()
    print("Training ended at: ", end_time.strftime('%Y-%m-%d-%H-%M'))
    time_taken = (end_time - start_time).total_seconds() / 3600
    running_time.append(time_taken)

    model.eval()

    for j in range(5):
        with torch.no_grad():
            test_data = test_data.apply(lambda sample: generate_response(sample, model, j), axis=1)

    test_data = test_data.apply(lambda sample: get_majority(sample, 5), axis=1)
    actual = test_data['class_label']
    prediction = test_data['prediction']

    accuracy_t.append(accuracy_score(actual, prediction))
    precision_t.append(precision_score(actual, prediction, pos_label='Yes'))
    recall_t.append(recall_score(actual, prediction, pos_label='Yes'))
    f1_score_t.append(f1_score(actual, prediction, pos_label='Yes'))

    predictions_t = []
    for j in range(5):
        predictions_t.append(test_data['prediction' + str(j)].tolist())
    consistency_t.append(get_consistency(predictions_t))
    test_data.to_csv('../results/test_' + run_name + "_" + str(i) + ".csv", index=False)

    for j in range(5):
        with torch.no_grad():
            dev_test_data = dev_test_data.apply(lambda sample: generate_response(sample, model, j), axis=1)

    dev_test_data = dev_test_data.apply(lambda sample: get_majority(sample, 5), axis=1)
    actual = dev_test_data['class_label']
    prediction = dev_test_data['prediction']

    accuracy_dt.append(accuracy_score(actual, prediction))
    precision_dt.append(precision_score(actual, prediction, pos_label='Yes'))
    recall_dt.append(recall_score(actual, prediction, pos_label='Yes'))
    f1_score_dt.append(f1_score(actual, prediction, pos_label='Yes'))

    predictions_dt = []
    for j in range(5):
        predictions_dt.append(dev_test_data['prediction' + str(j)].tolist())
    consistency_dt.append(get_consistency(predictions_dt))
    dev_test_data.to_csv('../results/dev_test_' + run_name + "_" + str(i) + ".csv", index=False)

    model = None
    trainer = None
    gc.collect()
    torch.cuda.empty_cache()

train_size = train_data.shape[0]
mean_running_time = statistics.mean(running_time)

mean_accuracy_t = statistics.mean(accuracy_t)
stdev_accuracy_t = statistics.stdev(accuracy_t)
mean_precision_t = statistics.mean(precision_t)
stdev_precision_t = statistics.stdev(precision_t)
mean_recall_t = statistics.mean(recall_t)
stdev_recall_t = statistics.stdev(recall_t)
mean_f1_score_t = statistics.mean(f1_score_t)
stdev_f1_score_t = statistics.stdev(f1_score_t)
mean_consistency_t = statistics.mean(consistency_t)
stdev_consistency_t = statistics.stdev(consistency_t)

mean_accuracy_dt = statistics.mean(accuracy_dt)
stdev_accuracy_dt = statistics.stdev(accuracy_dt)
mean_precision_dt = statistics.mean(precision_dt)
stdev_precision_dt = statistics.stdev(precision_dt)
mean_recall_dt = statistics.mean(recall_dt)
stdev_recall_dt = statistics.stdev(recall_dt)
mean_f1_score_dt = statistics.mean(f1_score_dt)
stdev_f1_score_dt = statistics.stdev(f1_score_dt)
mean_consistency_dt = statistics.mean(consistency_dt)
stdev_consistency_dt = statistics.stdev(consistency_dt)

print(f"Model: {model_id}, Dataset: {training_data_name}, Size: {train_size}, Runtime: {mean_running_time}")
print(f"Test: Accuracy: {mean_accuracy_t:.4f} ± {stdev_accuracy_t:.4f}, "
      f"Precision: {mean_precision_t:.4f} ± {stdev_precision_t:.4f}, "
      f"Recall: {mean_recall_t:.4f} ± {stdev_recall_t:.4f}, "
      f"F1 Score: {mean_f1_score_t:.4f} ± {stdev_f1_score_t:.4f}, "
      f"Consistency: {mean_consistency_t:.4f} ± {stdev_consistency_t:.4f}")

print(f"Dev test: Accuracy: {mean_accuracy_dt:.4f} ± {stdev_accuracy_dt:.4f}, "
      f"Precision: {mean_precision_dt:.4f} ± {stdev_precision_dt:.4f}, "
      f"Recall: {mean_recall_dt:.4f} ± {stdev_recall_dt:.4f}, "
      f"F1 Score: {mean_f1_score_dt:.4f} ± {stdev_f1_score_dt:.4f}, "
      f"Consistency: {mean_consistency_dt:.4f} ± {stdev_consistency_dt:.4f}")

with open(evaluation_file, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([model_id,
                     training_data_name,
                     train_size,
                     mean_running_time,
                     mean_accuracy_t, stdev_accuracy_t,
                     mean_precision_t, stdev_precision_t,
                     mean_recall_t, stdev_recall_t,
                     mean_f1_score_t, stdev_f1_score_t,
                     mean_consistency_t, stdev_consistency_t,
                     mean_accuracy_dt, stdev_accuracy_dt,
                     mean_precision_dt, stdev_precision_dt,
                     mean_recall_dt, stdev_recall_dt,
                     mean_f1_score_dt, stdev_f1_score_dt,
                     mean_consistency_dt, stdev_consistency_dt])


