import pandas as pd
import re
import copy
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import TensorDataset
import torch.nn as nn
import sklearn.metrics as mtr
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

CSV_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "data.csv"
)

#TODO have these as arguments in CLI instead maybe
SEED = 0; torch.manual_seed(SEED) # random seed
MAX_LEN = 256   # tokens kept from a response (or from a prompt in prompt-only runs)
PROMPT_LEN = 64 # prompt tokens kept when the prompt and response are combined

VOCAB_SIZE = 20000    # keep only the most common tokens so the embedding table stays small
MINIMUM_FREQUENCY = 2 # number of times a word has to appear in training data to get tokenized

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
KERNEL_SIZE = 6
DROPOUT = 0.5

# our models are small, a few CPU threads are faster than many
torch.set_num_threads(4)

# gradients flowing back through long stretches of padding shrink to "denormal"
# numbers that CPUs handle very slowly (LSTM epochs on short prompts took 5x
# longer), this rounds them to 0
torch.set_flush_denormal(True)

# words, every single punctuation / markdown / emoji character, and newlines
# note: the first version kept words only (r"\b\w+\b") and scored ~15 points
# lower, formatting like "**", "###", "-" bullets and line breaks is a big part
# of an LLM's style so we keep it
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]|\n", text.lower())

# view = which part of the LLM interaction the classifier gets to see (RQ2)
def get_tokens(df, view):
    if view == "output":
        return [tokenize(text)[:MAX_LEN] for text in df["output"]]

    if view == "prompt":
        return [tokenize(text)[:MAX_LEN] for text in df["prompt"]]

    # both: the prompt gets its own budget so a long prompt can't crowd out
    # the response, "both" always sees everything "output" sees
    return [
        tokenize(prompt)[:PROMPT_LEN] + ["<SEP>"] + tokenize(output)[:MAX_LEN]
        for prompt, output in zip(df["prompt"], df["output"])
    ]

def encode(tokens, vocab):
    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in tokens]

    # an empty text still needs one token
    return ids or [vocab["<UNK>"]]

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
    labels = list(label_map.values())

    # gauge the accuracy of the model
    acc = mtr.accuracy_score(actual, predictions)
    acc_balanced = mtr.balanced_accuracy_score(actual, predictions)
    per_class_report = mtr.classification_report(
        actual,
        predictions,
        labels=labels,
        target_names=list(label_map.keys()),
        digits=3,
        zero_division=0
    )
    print(per_class_report)
    print("confusion matrix (rows = true family, columns = predicted)")
    print(mtr.confusion_matrix(actual, predictions, labels=labels))
    print(f"{'Accuracy:':<20}{acc:.2%}\n{'Balanced Accuracy:':<20}{acc_balanced:.2%}")
    return {
        "Accuracy": acc,
        "Accuracy Balanced": acc_balanced
    }

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

        # randomly zero features while training to fight overfitting
        self.dropout = nn.Dropout(DROPOUT)

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

        features = self.dropout(features)

        logits = self.fc(features)            # note: a logit is the final layer of vectors (raw predictions)
        return logits

# bidirectional LSTM: reads the text left->right and right->left, then keeps
# the strongest value of each hidden unit across all positions (max pooling)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            padding_idx = 0 # token 0 <PAD> in vocab is the padding ID
        )

        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_DIM,     # how many numbers the LSTM remembers per direction
            batch_first = True,
            bidirectional = True
        )

        self.dropout = nn.Dropout(DROPOUT)

        # 2 * HIDDEN_DIM because the forward and backward states are joined
        self.fc = nn.Linear(
            2 * HIDDEN_DIM,
            num_classes
        )

    # forward pass
    def forward(self, features):
        features = self.embedding(features)

        features, _ = self.lstm(features)       # one state per token: [batch, sequence_length, 2 * HIDDEN_DIM]

        features = features.max(dim=1).values   # strongest value of each hidden unit: [batch, 2 * HIDDEN_DIM]

        features = self.dropout(features)

        logits = self.fc(features)            # note: a logit is the final layer of vectors (raw predictions)
        return logits

def load_data():
    # keep_default_na=False so a prompt like "NA" or "null" stays text
    # instead of turning into NaN and crashing tokenize()
    df = pd.read_csv(CSV_FILE, keep_default_na=False)

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

def build_vocab(train_df, view):
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    # convert tokens to integers for processing
    counter = Counter()

    for tokens in get_tokens(train_df, view):
        counter.update(tokens)

    for word, count in counter.most_common(VOCAB_SIZE - len(vocab)):
        if count >= MINIMUM_FREQUENCY:
            vocab[word] = len(vocab)

    return vocab

def create_label_map(df):
    llm_names = sorted(df["llm_name"].unique())

    return {
        label: i
        for i, label in enumerate(llm_names)
    }

def make_loader(df, view, vocab, label_map, shuffle=False):
    ids = [
        torch.tensor(encode(tokens, vocab))
        for tokens in get_tokens(df, view)
    ]

    # CNN requires same dimensions per record so pad every text with <PAD> (0)
    x = pad_sequence(ids, batch_first=True)
    y = torch.tensor([
        label_map[label]
        for label in df["llm_name"]
    ])

    # give NNs {BATCH_SIZE} examples at a time
    return DataLoader(
        TensorDataset(x, y),
        batch_size = BATCH_SIZE,
        shuffle = shuffle
    )

def train_model(
    model,
    train_loader,
    val_loader,
    device,
):
    criterion = nn.CrossEntropyLoss()

    # Adam adaptive learning rate optimizer from ch 9
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE       # learning rate
    )

    model = model.to(device)

    best_val_acc = -1
    best_weights = None

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

            # remember the best found model based on found validation accuracy
            best_weights = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch+1}: "
            f"loss = {total_loss / len(train_loader):.4f}, "
            f"validation acc = {val_acc:.2%}"
        )

    # go back to the best epoch, later epochs may have overfit
    model.load_state_dict(best_weights)
    return model

