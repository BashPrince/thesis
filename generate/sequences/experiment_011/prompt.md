Implement a python script that augments data as described in the pseudo code. I want to try multiple starting population sizes and augmentation sizes.
In the script I want to be able to set the following parameters as variables (illustrated with example values):

- `BASE_SIZES = [50, 100, 200]` the starting population sizes for real data subsets (base sets). The base sets should be class balanced e.g. size 50 should result in 25 positive and 25 negative samples.
- `AUG_SIZES = [400, 800]` the number of synthetic samples to be generated from each base set of a specific size. Should result in class balanced synthetic data (e.g. 400 -> 200 positive + 200 negative)
- `NUM_RESAMPLE` how many augmented datasets each combination of base size and aug size should result in. The base set will be sampled and subsequently augmented this many times.
- `MODEL = "gpt-4o"`

Use litellm for the communication with the LLM. Include
```
# Set ENV variables
with open('../../secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()
```
in `main`.

Store each base set with augmentations appended in a file named `real_{real_count}_aug_{aug_count}_v{i}.csv` with columns "Text" and "class_label" where i is the resampling index.

Make the functionality modular and create a file test.py where you create pytest unit tests.