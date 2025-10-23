from litellm import completion
import os
import pandas as pd
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples using a template and GPT model.")
    parser.add_argument("-o", action="store_true", help="Overwrite the existing output file if it exists.")
    parser.add_argument("-a", action="store_true", help="Append to the existing output file if it exists.")
    parser.add_argument("--num_pos", type=int, help="Number of positive samples to generate.")
    parser.add_argument("--num_neg", type=int, help="Number of negative samples to generate.")
    parser.add_argument("--example_source", required=True, type=str, help="Path to the optional example source file.")
    parser.add_argument("--model", type=str, default="openai/gpt-4o", help="Model to use for generation.")
    parser.add_argument("--template", type=str, help="Path to the template file.")
    parser.add_argument("--out_file", required=True, type=str, help="Output file path.")
    parser.add_argument("--seed", required=False, type=int, help="Seed for random module.")

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    ## set ENV variables
    with open('secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()

    with open(args.template, 'r') as file:
        template = file.read()

    all_examples = pd.read_csv(args.example_source)

    # Assert that the output file does not exist, unless overwrite/append flag is set
    if os.path.exists(args.out_file) and not args.o and not args.a:
        raise RuntimeError(f"Sample output file {args.out_file} already exists. Use -o to overwrite or -a to append.")
    
    # Empty DataFrame to store generated samples
    if args.a:
        gen_samples = pd.read_csv(args.out_file)
    else:
        gen_samples = pd.DataFrame(columns=['example_Text', 'class_label', 'topic', 'Text'])

    for num_samples, label in [(args.num_pos, 'Yes'), (args.num_neg, 'No')]:
        num_samples_generated = 0
        examples = all_examples[all_examples['class_label'] == label]

        while num_samples_generated < num_samples:
            # Reload examples if none are left
            if examples.empty:
                examples = all_examples[all_examples['class_label'] == label]
            
            # Sample num_samples_per_prompt from the dataframe and remove
            selected_example = examples.sample(n=1)
            examples = examples.drop(selected_example.index)
            selected_example = selected_example['Text']

            check_worthy_str = 'check-worthy' if label == 'Yes' else 'non-check-worthy'

            topics = ['education', 'the military', 'the economy', 'crime']
            topic = random.choice(topics)

            prompt = template.format(example=selected_example, label=check_worthy_str, topic=topic)
            
            response = completion(
                model=args.model,
                messages=[{ "content": prompt, "role": "user"}]
            )

            response_str = response.choices[0].message.content
            split_1 = response_str.split(">>")
            last_part = split_1[-1]
            if len(split_1) != 2 or response_str.count(">><<") == 1 or last_part.count('<') != 2 or last_part.count('<<') != 1:
                print("Misformatted output, skipping ...")
                continue
            sample = last_part.split("<<")[0]
            
            print(response_str)

            temp_df = pd.DataFrame({
                'example_Text': selected_example,
                'class_label': label,
                'topic': topic,
                'Text': sample,
            })

            # Append the sample_lines and ids_list to the existing dataframe
            gen_samples = pd.concat([gen_samples, temp_df], ignore_index=True)

            # Save the generated samples to the output file
            gen_samples.to_csv(args.out_file, index=False)

            num_samples_generated += 1
