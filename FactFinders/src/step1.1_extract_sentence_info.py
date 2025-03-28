import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('all')


def is_person_present(NER):
    if len(NER) == 0:
        return False
    else:
        return True


def get_length_without_stop_words(row):
    text = row["Text"]

    tokens = nltk.word_tokenize(text)
    row['Text_Length'] = len(tokens)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    row['Text_Length_without_StopWords'] = len(filtered_tokens)
    return row


def extract_verbs(text):
    tokens = word_tokenize(text.lower())
    tagged_words = pos_tag(tokens)
    return [lemmatizer.lemmatize(word) for word, tag in tagged_words if tag.startswith('VB')]

# read the original training data
train_data = pd.read_csv("../data/CT24_checkworthy_english_train.tsv", sep='\t', on_bad_lines='skip')
train_distribution = train_data['class_label'].value_counts()
print(f"Distribution of labels: \n{train_distribution}")

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# identify the presence of the named entity
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
train_data['NER'] = train_data['Text'].apply(lambda x: nlp(x))
train_data['is_NER_present'] = train_data['NER'].apply(lambda x: is_person_present(x))

# get the length of a sentence without stopwords
stop_words = set(stopwords.words('english'))
train_data = train_data.apply(lambda x: get_length_without_stop_words(x), axis=1)

# extract and lemmatize verbs of sentences
lemmatizer = WordNetLemmatizer()
train_data["verbs"] = train_data["Text"].apply(lambda x: extract_verbs(x))
train_data.to_csv("..data/Training_data_with_stats.csv", index=False)

# collect all verbs of the training data and save into train_verbs.csv
verbs = []
verbs_list = train_data["verbs"].tolist()
for v_list in verbs_list:
    verbs.extend(v_list)
verbs = list(set(verbs))
print("Length of verbs: ", len(verbs))

auxiliary_verbs = [
    # To be
    'am', 'is', 'are', 'was', 'were', 'been', 'being',
    # To have
    'have', 'has', 'had', 'having',
    # To do
    'do', 'does', 'did', 'doing', 'done'
]

verbs_final = [verb for verb in verbs if verb not in auxiliary_verbs]
print("Final verb count: ", len(verbs_final))
df = pd.DataFrame(verbs_final, columns=['Verb'])
df.to_csv("../data/train_verbs.csv", index=False)