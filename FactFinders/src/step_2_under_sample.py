import numpy as np
import pandas as pd
import torch
from imblearn.under_sampling import CondensedNearestNeighbour
from transformers import BertTokenizer, BertModel
from datetime import datetime


def get_bert_embeddings(texts):
    cls_embeddings = []

    batch_size = 64
    data_size = len(texts)
    for i in range(0, data_size, batch_size):
        if i + batch_size < data_size:
            batch_text = texts[i:i+batch_size]
        else:
            batch_text = texts[i:data_size]

        inputs = tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt").to('cuda')
        with torch.no_grad():
           outputs = model(**inputs)
        # Extract the output embeddings corresponding to [CLS] token
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        cls_embeddings.extend(embeddings)
        print("Conversion completed for: ", i+batch_size)
        torch.cuda.empty_cache()
    print(datetime.now().strftime('%Y-%m-%d-%H-%M'), " Embeddings: ", len(cls_embeddings))
    print(datetime.now().strftime('%Y-%m-%d-%H-%M'), " Embedding size: ", len(cls_embeddings[0]))

    return np.array(cls_embeddings)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to('cuda')

df = pd.read_csv("../data/CT24_checkworthy_english_train.tsv", sep='\t', on_bad_lines='skip')
print(datetime.now().strftime('%Y-%m-%d-%H-%M'), " Original dataset size: ", df.shape)

# Convert text in 'Text' column to BERT embeddings
text_embeddings = get_bert_embeddings(df['Text'].tolist())
print(datetime.now().strftime('%Y-%m-%d-%H-%M'), " Embedding generated")

df['y'] = df['class_label'].apply(lambda x: 1 if x == 'Yes' else 0)
sampler = CondensedNearestNeighbour(sampling_strategy='majority', random_state=42)
X, y = sampler.fit_resample(text_embeddings, df['y'].tolist())
print(datetime.now().strftime('%Y-%m-%d-%H-%M'), " Data sampled")

indexes_to_filter = sampler.sample_indices_
print(datetime.now().strftime('%Y-%m-%d-%H-%M'), " Sampled data size: ", len(indexes_to_filter))

sampled_data = df.iloc[indexes_to_filter]
sampled_data.drop(['y'], axis=1)
sampled_data.to_csv("../data/CT24_train_dp_step_2.csv", index=False)
