import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def is_informative_verb_present(verb_list):
    verb_list = verb_list.replace("'", "")
    verb_list = verb_list.replace("[", "")
    verb_list = verb_list.replace("]", "").strip()
    verb_list = verb_list.split(",")

    for verb in verb_list:
        if verb.strip() in informative_verbs:
            return True

    return False


def is_informative(row):
    if row['class_label'] == 'Yes':
        return True
    elif row['is_informative_verb_present']:
        return True
    elif row['is_NER_present']:
        return True
    elif row['Text_Length_without_StopWords'] >= 8:  # minimum length of the sentence = 8
        return True
    else:
        return False

# filter informative verbs from all verbs
train_data = pd.read_csv("../data/Training_data_with_stats.csv")
verbs = pd.read_csv("../data/train_verbs_types.csv")  # TODO train_verbs_types.csv should we provided in data folder?
informative_verb_ids = [1, 3, 4, 5, 6]
informative_verbs = verbs[verbs['Verb_Type_Index'].isin(informative_verb_ids)]["Verb"].tolist()
print("Number of informative verbs: ", len(informative_verbs))

# select sentences containing informative verbs
train_data["is_informative_verb_present"] = train_data["verbs"].apply(is_informative_verb_present)
verb_distribution = train_data['is_informative_verb_present'].value_counts()
print(f"Distribution of informative verbs: \n{verb_distribution}")
train_data.to_csv("../data/Training_data_with_stats.csv", index=False)


# select informative sentences: the sentence labeled with 'Yes'
# or contains informative verbs or with NER presented or with length >=8
train_data = pd.read_csv("../data/Training_data_with_stats.csv")
train_data["is_informative"] = train_data.apply(lambda x: is_informative(x), axis=1)
filtered_data = train_data[train_data["is_informative"] == True]
print("Filtered data size: ", filtered_data.shape)
informative_distribution = filtered_data['class_label'].value_counts()
print(f"Distribution of labels: \n{informative_distribution}")

filtered_data = filtered_data.drop(
    ["NER", "is_NER_present", "Text_Length", "Text_Length_without_StopWords", "is_informative", "verbs",
     "is_informative_verb_present"], axis=1)
filtered_data.to_csv("../data/CT24_train_dp_step_1.csv", index=False)
