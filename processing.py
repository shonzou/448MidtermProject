import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import Dataset


CSV_FILE = "exampledata.csv" # replace with data.csv when that gets populated
MAX_LEN = 1024

SEED = 0
torch.manual_seed(SEED) # random seed

vocab = {
    "<PAD>": 0, # padding 
    "<UNK>": 1  # unknown word
}

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def encode(text):
    tokens = tokenize(text)

    return [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens]

# CNN requires same dimensions per record so pad with <PAD>
def padded_encode(text):
    ids = encode(text)

    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))

    return ids

class TextDataset(Dataset):
    # initialize dataset
    def __init__(
        self, dataframe, text_column, label_map):
        self.texts = dataframe[text_column].tolist()
        self.labels = [
            label_map[label]
            for label in dataframe["llm_name"]
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = padded_encode(self.texts[idx])
        y = self.labels[idx]

        return torch.tensor(x), torch.tensor(y)

def main():
    df = pd.read_csv(CSV_FILE) 

    print(df.head())
    print(df["llm_name"].value_counts())

    # split by prompt so that the same prompt doesn't get put in two diff sets
    prompt_ids = df["prompt"].unique()

    # split train set (80% train)
    train_ids, temp_ids = train_test_split(
        prompt_ids,
        test_size = 0.2,
        random_state = SEED
    )

    # split val set and test set from remaining set
    # (10% val, 10% test)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size = 0.5,
        random_state = SEED
    )

    train_df = df[df["prompt"].isin(train_ids)]
    val_df = df[df["prompt"].isin(val_ids)]
    test_df = df[df["prompt"].isin(test_ids)]

    # convert tokens to integers for processing
    counter = Counter()

    for text in train_df["output"]:
        counter.update(tokenize(text))

    for word, count in counter.items():
        if count >= 2:
            vocab[word] = len(vocab)

    llm_name = sorted(train_df["llm_name"].unique())

    map_label_id = {
        label: i
        for i, label in enumerate(llm_name)
    }

    # pytorch's Dataset easier interface than regular arrays
    train_dataset = TextDataset(train_df, "output", map_label_id)
    val_dataset = TextDataset(val_df, "output", map_label_id)
    test_dataset = TextDataset(test_df, "output", map_label_id)
