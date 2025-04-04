from litellm import completion
import os
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples using a template and GPT model.")
    parser.add_argument("-o", action="store_true", help="Overwrite the existing output file if it exists.")
    parser.add_argument("--num_samples", type=int, help="Number of samples to generate.")
    parser.add_argument("--num_examples", default=5, type=int, help="Number of examples to include in template.")
    parser.add_argument("--example_source", type=str, help="Path to the example source file.")
    parser.add_argument("--model", type=str, default="openai/gpt-4o", help="Model to use for generation.")
    parser.add_argument("--template", type=str, help="Path to the template file.")
    parser.add_argument("--out", type=str, help="Output file path for generated samples.")

    args = parser.parse_args()

    ## set ENV variables
    with open('secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()

    with open(args.template, 'r') as file:
        template = file.read()

    # Assert that the output file does not exist, unless overwrite flag is set
    if os.path.exists(args.out) and not args.o:
        raise RuntimeError(f"Sample output file {args.out} already exists. Use -o to overwrite.")

    num_samples_generated = 0
    example_data_frame = pd.read_csv(args.example_source)
    example_data_frame = example_data_frame[example_data_frame['class_label'] == 'Yes']

    if example_data_frame.empty:
        raise RuntimeError(f"No examples found in {args.example_source} with class_label 'Yes'.")
    
    gen_sample_data_frame = pd.DataFrame(columns=['example_Sentence_id', 'example_Text' 'Text'])

    while num_samples_generated < args.num_samples:
        # Reload all examples if none are left
        if example_data_frame.empty:
            example_data_frame = pd.read_csv(args.example_source)
            example_data_frame = example_data_frame[example_data_frame['class_label'] == 'Yes']
        
        # Sample num_examples from the dataframe and remove
        n = min(args.num_examples, len(example_data_frame))
        selected_examples = example_data_frame.sample(n=n)
        example_data_frame = example_data_frame.drop(selected_examples.index)

        example_list = selected_examples['Text'].tolist()

        # Prepare prompt
        examples_str = "\n".join(["- " + e for e in example_list])

        prompt = template.format(examples_str)

        response = completion(
            model=args.model,
            messages=[{ "content": prompt, "role": "user"}]
        )

        samples = response.choices[0].message.content

        # Parse the individual samples from the completion
        sample_lines = samples.split("\n")
        
        if len(sample_lines) != args.num_examples:
            print(f"Sampled {len(sample_lines)} lines, expected {args.num_examples}. Skipping generated sample.")
            continue

        max_samples_to_add = min(args.num_samples - num_samples_generated, len(sample_lines))
        sample_lines = sample_lines[:max_samples_to_add]
        sample_lines = [s.removeprefix('- ') for s in sample_lines]

        # Retrieve the Sentence_id for the selected examples
        ids_list = selected_examples['Sentence_id'].tolist()
        ids_list = ids_list[:max_samples_to_add]
        example_list = example_list[:max_samples_to_add]

        # Append the sample_lines and ids_list to the existing dataframe
        temp_df = pd.DataFrame({
            'example_Sentence_id': ids_list,
            'example_Text': example_list,
            'Text': sample_lines
        })
        gen_sample_data_frame = pd.concat([gen_sample_data_frame, temp_df], ignore_index=True)

        # Save the generated samples to the output file
        gen_sample_data_frame.to_csv(args.out, index=False)

        num_samples_generated += len(sample_lines)
