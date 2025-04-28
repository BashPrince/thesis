from litellm import completion
import os
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples using a template and GPT model.")
    parser.add_argument("-o", action="store_true", help="Overwrite the existing output file if it exists.")
    parser.add_argument("-n", action="store_true", help="Generate negative samples")
    parser.add_argument("--num_samples", type=int, help="Number of samples to generate.")
    parser.add_argument("--num_samples_per_prompt", default=5, type=int, help="Number of generations per prompt.")
    parser.add_argument("--example_source", required=False, type=str, help="Path to the optional example source file.")
    parser.add_argument("--model", type=str, default="openai/gpt-4o", help="Model to use for generation.")
    parser.add_argument("--template", type=str, help="Path to the template file.")
    parser.add_argument("--relaxed_count", action="store_true", help="Use relaxed count for sampling.")

    args = parser.parse_args()

    ## set ENV variables
    with open('secrets/openai_api_key.txt', 'r') as key_file:
        os.environ["OPENAI_API_KEY"] = key_file.read().strip()

    with open(args.template, 'r') as file:
        template = file.read()

    # Expect template of format template_{info}.txt
    # Name out file samples_{info}.csv

    if "template_" not in args.template or not args.template.endswith(".txt"):
        raise ValueError("Template file name must be of the format 'template_{info}.txt'.")

    # Extract the {info} part from the template file name
    info = args.template.split("template_")[1].rsplit(".txt", 1)[0]

    # Construct the output file name
    model_name = args.model.split("/")[-1]
    out_file = f"data/samples_{info}_{model_name}.csv"

    # Assert that the output file does not exist, unless overwrite flag is set
    if os.path.exists(out_file) and not args.o:
        raise RuntimeError(f"Sample output file {out_file} already exists. Use -o to overwrite.")

    target_class = "Yes" if not args.n else "No"

    if args.example_source:
        example_data_frame = pd.read_csv(args.example_source)
        example_data_frame = example_data_frame[example_data_frame['class_label'] == target_class]


        if example_data_frame.empty:
            raise RuntimeError(f"No examples found in {args.example_source} with class_label '{target_class}'.")
    
        print(f"Found {len(example_data_frame)} examples")
    else:
        example_data_frame = None
    
    # Empty DataFrame to store generated samples
    if args.example_source:
        gen_sample_data_frame = pd.DataFrame(columns=['example_Sentence_id', 'example_Text', 'Text'])
    else:
        gen_sample_data_frame = pd.DataFrame(columns=['Text'])
        
    num_samples_generated = 0

    while num_samples_generated < args.num_samples:
        if args.example_source:
            # Reload all examples if none are left
            if example_data_frame.empty:
                example_data_frame = pd.read_csv(args.example_source)
                example_data_frame = example_data_frame[example_data_frame['class_label'] == target_class]
            
            # Sample num_samples_per_prompt from the dataframe and remove
            n = min(args.num_samples_per_prompt, len(example_data_frame))
            selected_examples = example_data_frame.sample(n=n)
            example_data_frame = example_data_frame.drop(selected_examples.index)

            example_list = selected_examples['Text'].tolist()

            # Prepare prompt
            examples_str = "\n".join(["- " + e for e in example_list])

            prompt = template.format(examples=examples_str)
        else:
            prompt = template.format(num_examples=args.num_samples_per_prompt)

        response = completion(
            model=args.model,
            messages=[{ "content": prompt, "role": "user"}]
        )

        samples = response.choices[0].message.content

        # Parse the individual samples from the completion
        sample_lines = samples.split("\n")
        sample_lines = [s.removeprefix('- ') for s in sample_lines]

        if len(sample_lines) == 0:
            print("No samples generated. Skipping this iteration.")
            continue
        
        if not args.relaxed_count and len(sample_lines) != args.num_samples_per_prompt:
            print(f"Sampled {len(sample_lines)} lines, expected {args.num_samples_per_prompt}. Skipping generated sample.")
            continue

        max_samples_to_add = min(args.num_samples - num_samples_generated, len(sample_lines))
        sample_lines = sample_lines[:max_samples_to_add]

        # Retrieve the Sentence_id for the selected examples
        if args.example_source:
            ids_list = selected_examples['Sentence_id'].tolist()
            ids_list = ids_list[:max_samples_to_add]
            example_list = example_list[:max_samples_to_add]

            # Create a temporary dataframe for this iteration's samples
            # I'm assuming that sample i of the model's generation is based on example i (which seems to always be true)
            # i.e. the order of samples and examples should correspond
            temp_df = pd.DataFrame({
                'example_Sentence_id': ids_list,
                'example_Text': example_list,
                'Text': sample_lines
            })
        else:
            temp_df = pd.DataFrame({
                'Text': sample_lines
            })
        # Append the sample_lines and ids_list to the existing dataframe
        gen_sample_data_frame = pd.concat([gen_sample_data_frame, temp_df], ignore_index=True)

        # Save the generated samples to the output file
        gen_sample_data_frame.to_csv(out_file, index=False)

        num_samples_generated += len(sample_lines)
