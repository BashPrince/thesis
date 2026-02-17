I want to implement a synthetic data augmentation strategy from a paper for text data in a check-worthy claim classification task.

The algorithm takes two samples and combines attributes from them into a new sample described in detail in the following pseudo code:
@pseudo_algorithm.tex

Have a close look at the files in @prompt_templates to gain a better understanding of the task.
For the sentence embeddings mentioned in the pseudo code I want to use `SentenceTransformer("all-MiniLM-L6-v2")`.

We will simulate a low data scenario by sampling small subsets of real data from the file train.csv. This file has the columns "Sentence_id", "Text" containing the claims and "class_label" with values "Yes" and "No" for check-worthy and non check-worthy samples respectively. The "Sentence_id" column can be ignored completely and should not be contained in any of the augmented datasets.