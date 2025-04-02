from litellm import completion
import os
import csv  # Import csv for handling CSV files

## set ENV variables
with open('secrets/openai_api_key.txt', 'r') as key_file:
    os.environ["OPENAI_API_KEY"] = key_file.read().strip()

with open('templates/template.txt', 'r') as file:
    template = file.read()

response = completion(
  model="openai/gpt-4o",
  messages=[{ "content": template, "role": "user"}]
)

sample = response.choices[0].message.content

# Append the generated sample to data/samples.csv
output_path = 'data/samples.csv'
file_exists = os.path.exists(output_path)

with open(output_path, mode='a', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["sample"])
    if not file_exists:
        writer.writeheader()  # Write header if file doesn't exist
    writer.writerow({"sample": sample})  # Append the sample

