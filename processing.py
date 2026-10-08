import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import Dataset
import torch.nn as nn
import sklearn.metrics as mtr
from torch.utils.data import DataLoader

CSV_FILE = "exampledata.csv" # replace with data.csv when that gets populated

#TODO have these as arguments in CLI instead maybe
SEED = 0; torch.manual_seed(SEED) # random seed
MAX_LEN = 1024

MINIMUM_FREQUENCY = 2 # number of times a word has to appear in training data to get tokenized

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

EMBEDDING_DIM = 128
KERNEL_SIZE = 4

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def encode(text, vocab):
    tokens = tokenize(text)

    return [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens]

# CNN requires same dimensions per record so pad with <PAD>
def padded_encode(text, vocab):
    ids = encode(text, vocab)

    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))

    return ids

# validation
def get_predictions(model, loader, device):
    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            # note: a logit is a final linear layer output
            logits = model(x)
            preds = logits.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())
            actual.extend(y.numpy())

    return predictions, actual

def evaluate_acc(model, loader, device):
    predictions, actual = get_predictions(model, loader, device)
    return mtr.accuracy_score(actual, predictions)

# validation
def evaluate_model(model, test_loader, device, label_map):
    predictions, actual = get_predictions(model, test_loader, device)
    
    # gauge the accuracy of the model
    acc = mtr.accuracy_score(actual, predictions)
    acc_balanced = mtr.balanced_accuracy_score(actual, predictions)
    per_class_report = mtr.classification_report(
        actual,
        predictions,
        target_names=list(label_map.keys())
    )
    print(per_class_report)
    print(f"{'Accuracy:':<20}{acc:.2%}\n{'Balanced Accuracy:':<20}{acc_balanced:.2%}")
    return acc, acc_balanced

# NOTE: for more information on CNN see chapter 6
class CNNClassifier(nn.Module):
    # embedding_dim: how many numbers to represent a token
    def __init__(self, vocab_size, num_classes):
        super().__init__()

        # embedding layer (huge lookup table)
        self.embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            padding_idx = 0 # token 0 <PAD> in vocab is the padding ID
        )

        # 1D convoluition
        self.conv = nn.Conv1d( 
            in_channels = EMBEDDING_DIM,
            out_channels = EMBEDDING_DIM, # number of channels in new sequence
            kernel_size = KERNEL_SIZE # number of tokens processed (sliding window) at a time 
        )

        # this is an activation function type (I grabbed from EE 456)
        self.relu = nn.ReLU() 
        # variant of max pooling (ch 7) grabbing strongest activation for each feature across ENTIRE sequence
        self.pool = nn.AdaptiveMaxPool1d(1) 

        # linear transformation
        self.fc = nn.Linear(
            EMBEDDING_DIM,
            num_classes
        )

    # forward pass 
    def forward(self, features):
        features = self.embedding(features)

        features = features.permute(0, 2, 1)  # conv expects different format so we gotta permute

        features = self.conv(features)
        features = self.relu(features)

        features = self.pool(features)

        features = features.squeeze(-1)       # remove a remaining unneeded dimension

        logits = self.fc(features)            # note: a logit is the final layer of vectors (raw predictions) 
        return logits

#TODO: Implement RNN using LSTM

class TextDataset(Dataset):
    # initialize dataset
    def __init__(
        self, dataframe, text_column, label_map, vocab):
        self.texts = dataframe[text_column].tolist()
        self.labels = [
            label_map[label]
            for label in dataframe["llm_name"]
        ]
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = padded_encode(self.texts[idx], self.vocab)
        y = self.labels[idx]

        return torch.tensor(x), torch.tensor(y)

def load_data():
    df = pd.read_csv(CSV_FILE) 
    
    print(df.head())
    print(df["llm_name"].value_counts())
    return df 

def split_data(df):
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

    return train_df, val_df, test_df

def build_vocab(train_df, text_column):
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    # convert tokens to integers for processing
    counter = Counter()

    for text in train_df[text_column]:
        counter.update(tokenize(text))

    for word, count in counter.items():
        if count >= MINIMUM_FREQUENCY:
            vocab[word] = len(vocab)

    return vocab

def create_label_map(train_df):
    llm_names = sorted(train_df["llm_name"].unique())

    return {
        label: i
        for i, label in enumerate(llm_names)
    }

def create_dataloaders(
    train_df,
    val_df,
    test_df,
    text_column,
    label_map,
    vocab,
):
    # pytorch's Dataset easier interface than regular arrays
    train_dataset = TextDataset(train_df, text_column, label_map, vocab)
    val_dataset = TextDataset(val_df, text_column, label_map, vocab)
    test_dataset = TextDataset(test_df, text_column, label_map, vocab)

    # TODO: train and evaluate classifier using LLM input only for RQ2
    # TODO: train and evaluate classifier using LLM input and output for RQ2

    # wrappers for Datasets
    # give NNs {BATCH_SIZE} examples at a time
    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = BATCH_SIZE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE
    )
    return train_loader, val_loader, test_loader

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    model_path,
):
    criterion = nn.CrossEntropyLoss()
    
    # Adam adaptive learning rate optimizer from ch 9
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE       # learning rate
    )

    model = model.to(device)

    best_val_acc = -1

    #TODO add early stopping within validation
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0  # calculating the loss from training
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()   # clear previous batch's gradients

            predictions = model(inputs) # forward pass

            loss = criterion(predictions, labels)

            loss.backward() # backpropagation
            optimizer.step() # change parameters

            total_loss += loss.item()

        # validating
        val_acc = evaluate_acc(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # save the best found CNN based on found validation accuracy
            torch.save(
                model.state_dict(),
                model_path
            )

        print(
            f"epoch {epoch+1}: "
            f"loss = {total_loss / len(train_loader):.4f}, "
            f"validation acc = {val_acc:.2%}"
        )

    # load the best CNN
    model.load_state_dict(torch.load(model_path, weights_only = True))
    return model

def main():
    df = load_data()

    train_df, val_df, test_df = split_data(df)

    vocab = build_vocab(train_df, "output")

    # TODO: train and evaluate classifier using LLM input only for RQ2
    # TODO: train and evaluate classifier using LLM input and output for RQ2

    label_map = create_label_map(train_df)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df,
        val_df,
        test_df,
        "output",
        label_map,
        vocab
    )

    # small optimization that chooses a NVIDIA GPU if possible (I have a NVIDIA GPU so this
    # benefits me specifically)
    device = torch.device (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CNNClassifier(
        vocab_size = len(vocab),
        num_classes = len(label_map),
    )

    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        model_path = "cnn.pt",
    )

    evaluate_model(
        model,
        test_loader,
        device,
        label_map
    )

    # TODO: compare CNN and RNN performance when RNN is set up
    # TODO: testing between code/programming prompts and non-code prompts