import argparse
import csv
import re
from pathlib import Path
import litellm
import os

def strip_quotes(text):
    # Remove all surrounding quotes (single, double, or unicode)
    return re.sub(r'^[\'"“”‘’]+|[\'"“”‘’]+$', '', text.strip())

def main():
    # set ENV variables
    with open('../../generate/secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()

    # Specify models here
    models = [
        "gpt-4o",
        "gpt-4.1",
        # Add more model names as needed, e.g. "gpt-3.5-turbo"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, required=True, help='Row index (0-based)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', action='store_true', help='Use positive template')
    group.add_argument('-n', action='store_true', help='Use negative template')
    args = parser.parse_args()

    csv_path = Path(__file__).parent / "FactSpan_en.csv"
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        idx = args.i - 2
        if idx < 0 or idx >= len(reader):
            raise IndexError(f"Row index {idx} out of range (0-{len(reader)-1})")
        row = reader[idx]
        claim = strip_quotes(row["claim"])

    template_file = "pos_template.txt" if args.p else "neg_template.txt"
    template_path = Path(__file__).parent / template_file
    with open(template_path, encoding='utf-8') as f:
        template = f.read()

    prompt = template.format(statement=claim)

    print("Claim:", row["claim"])

    for model in models:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"Model: {model}")
        print("Model completion:")
        print(response['choices'][0]['message']['content'])
        print("-" * 40)

if __name__ == "__main__":
    main()
