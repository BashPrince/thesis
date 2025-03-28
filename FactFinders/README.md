# FactFinders at CheckThat! 2024 Task 1: Check-worthiness Estimation
This repository contains our approach for the CheckThat! 2024 Task 1: Check-worthiness Estimation. We investigate the application of eight prominent open-source LLMs with fine-tuning and prompt engineering to identify check-worthy statements from political transcriptions. Further, we propose a two-step data pruning approach to automatically identify high-quality training data instances for effective learning. These methods are evaluated on the datasets in English language.

You can find more details about the task and the dataset at the [CLEF2024 CheckThat! Lab](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab.)


## Project Structure

The project is organized into two main directories: `data/` and `src/`, along with a requirements file to set up your environment.

### `data/`

This directory contains datasets used and generated throughout the project pipelineas:

- `CT24_checkworthy_english_train.tsv`
- `CT24_checkworthy_english_dev.tsv`
- `CT24_checkworthy_english_dev-test.tsv`
- `CT24_checkworthy_english_test_gold.tsv`
- `CT24_train_dp_step_1.csv`: Training data after preprocessing through the first pruning step.
- `CT24_train_dp_step_2.csv`: Training data after preprocessing only through the second pruning step.
- `CT24_train_dp_step_1and2.csv`: Training data after preprocessing through both the first and second pruning step.

### `src/`

This directory contains Python scripts that are part of the pipeline:

- `train_evaluate.py`: Main script for training models and evaluating their performance.
- `step1_get_informative_sentences.py`: Script for the first step in the data pruning process to select informative sentences.
- `step2_under_sample.py`: Script for the second step in the data pruning process to balance the dataset.
- `verb_classification.py`: Script dedicated to classifying sentences based on the verbs used.


### `requirements.txt`

A text file containing all necessary Python packages required to run the project scripts. Use the following command to install these dependencies:

```bash
pip install -r requirements.txt
```

## Reference

If you find this work useful, please cite the following information:

```bibtex
@InProceedings{clef-checkthat:2024:task1:FactFinders,
author = {Li, Yufeng and Panchendrarajan, Rrubaa and Zubiaga, Arkaitz},
title = "FactFinders at CheckThat! 2024: Refining Check-worthy Statement Detection with LLMs through Data Pruning",
year = {2024},
booktitle = "Working Notes of CLEF 2024 - Conference and Labs of the Evaluation Forum",
series = {CLEF~'2024},
address = {Grenoble, France},
crossref = "clef2024-workingnotes"
}
```

Repo copied from https://github.com/isyufeng/FactFinders and adapted