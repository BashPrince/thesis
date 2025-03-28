import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def generate_prompt(sample):
    verb = sample["Verb"]

    full_prompt = (f"Classify the verb '{verb}' into one of the following types:"
                   f"\n1. Physical Actions"
                   f"\n2. Mental Actions"
                   f"\n3. Changes in State "
                   f"\n4. Creation or Destruction"
                   f"\n5. Communication"
                   f"\n6. Movement"
                   f"\n7. Emotion or Feeling"
                   f"\n8. Perception"
                   f"\n9. Linking verb"
                   f"\n10. Other"
                   f"\n Generate only the verb type.")
    full_prompt += "Verb type: "
    sample['prompt'] = full_prompt
    return sample


def generate_response(sample, model):
    prompt = sample['prompt']
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')
    output = model.generate(**model_inputs, max_new_tokens=5,
                            pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.3)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded_output.replace(prompt, "").strip()
    sample['Response'] = response

    verb_types = ["Physical Actions", "Mental Actions", "Changes in State", "Creation or Destruction", "Communication",
                  "Movement", "Emotion or Feeling", "Perception", "Linking verb", "Other"]

    verb_found = None
    for verb in verb_types:
        if verb in response:
            verb_found = verb
            break

    if verb_found is None:
        sample["Verb_Type"] = "Error"
    else:
        sample["Verb_Type"] = verb_found

    return sample


data = pd.read_csv("../data/train_verbs.csv")
print("Dataset size: ", data.shape)

data = data.apply(generate_prompt, axis=1)

# Load model and config
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    load_in_4bit=True,
    torch_dtype=torch.float16,
    cache_dir='../hf_cache'
)

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='../hf_cache')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.eval()

with torch.no_grad():
    data = data.apply(lambda sample: generate_response(sample, model), axis=1)

data.drop(['prompt'], inplace=True, axis=1)
data.to_csv("../data/train_verbs_types.csv", index=False)