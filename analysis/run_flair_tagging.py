import os
import argparse
from flair.data import Sentence
from flair.nn import Classifier
import pandas as pd
import wandb

parser = argparse.ArgumentParser(description="Flair NER and Sentiment Analysis")
parser.add_argument('--input', type=str, required=True, help='wandb input artifact')
parser.add_argument('--output', type=str, required=True, help='wandb output artifact')
parser.add_argument('--split', type=str, required=True, help='Split of the wandb artifact to use')
args = parser.parse_args()

# Setup wandb
run = wandb.init(
    project="thesis",
    group="tagging",
    job_type="tagging"
)

# Download dataset from wandb
artifact = run.use_artifact(args.input)
artifact_dir = artifact.download()
data_file = os.path.join(artifact_dir, f"{args.split}.csv")
data_frame = pd.read_csv(data_file)

# ------NER------
sentences = [Sentence(text) for text in data_frame["Text"].astype(str)]

# load the NER tagger
ner_tagger = Classifier.load('ner-ontonotes-large')

# run NER tagging
ner_tagger.predict(sentences=sentences, verbose=True, mini_batch_size=64)
ner_labels = [[l.value for l in s.get_labels()] if s.get_labels() else None for s in sentences]
data_frame["ner"] = ner_labels

# ------SENTIMENT------
sentences = [Sentence(text) for text in data_frame["Text"].astype(str)]

# load the sentiment tagger
sentiment_tagger = Classifier.load('sentiment')

# run NER tagging
sentiment_tagger.predict(sentences=sentences, verbose=True, mini_batch_size=64)
sentiment_labels = [s.get_label().value for s in sentences]
data_frame["sentiment"] = sentiment_labels

# save the annotated dataframe
temp_file = "./tagged_data.csv"
data_frame.to_csv(temp_file, index=False)

# Upload to wandb
new_artifact = wandb.Artifact(
    name=args.output,
    type="dataset",
    description=artifact.description+"\nContains NER and sentiment analysis columns (ner, sentiment)")
new_artifact.add_file(temp_file)
run.log_artifact(new_artifact)

# Cleanup
os.remove(temp_file)