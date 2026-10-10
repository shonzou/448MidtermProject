from datasets import load_dataset, Dataset
from typing import Any, cast
import pandas as pd
import re
from torch import manual_seed
from collections import Counter
from pathlib import Path

SEED = 0; manual_seed(SEED) # random seed

DATASET_NAME = "lmarena-ai/arena-human-preference-140k"
DATA_FILE = "data/train-00000-of-00007.parquet"

CSV_FILE = Path(__file__).resolve().parent.parent / "data" / "data.csv"
LOG_FILE = Path(__file__).resolve().parent.parent / "log" / "log.txt"
OUTPUT_COLUMNS = ["llm_name", "model", "prompt", "output", "is_code"]
NUM_FAMILIES = 3
FAMILY_PATTERNS = {
    "Claude":   r"claude",
    "GPT":      r"chatgpt|gpt|o\d",
    "Gemini":   r"gemini",
    "Gemma":    r"gemma",
    "Qwen":     r"qwen|qwq",
    "Llama":    r"llama",
    "DeepSeek": r"deepseek",
    "Grok":     r"grok",
    "Mistral":  r"mistral|magistral",
    "Cohere":   r"command",
    "Nova":     r"amazon",
    "MiniMax":  r"minimax",
    "Kimi":     r"kimi",
    "Hunyuan":  r"hunyuan",
}

saved_log = []

def log(line=""):
    print(line)
    saved_log.append(line)

def load_data():
    # casting to avoid type errors when printing length of dataset
    full_data = cast(
        Dataset,
        load_dataset(
            "lmarena-ai/arena-human-preference-140k", 
            data_files = "data/train-00000-of-00007.parquet",
            split = "train",
            verification_mode = "no_checks"),
    )

    full_data = full_data.select_columns(["language", "conv_metadata", "is_code",
                                          "model_a", "model_b", "conversation_a", "conversation_b", "id"])

    log(f"Raw records: {len(full_data)}")

    english_data = []

    # filter out non-english records
    for record in full_data:
        record = cast(dict[str, Any], record)  # type errors again...
        if record["language"] == "en":
            english_data.append(record)
    log(f"English records: {len(english_data)}")

    data = []

    # filter out conversations with more than one turn (one prompt, one answer)
    for record in english_data:
        if record["conv_metadata"]["turns"] == 1:
            data.append(record)

    log(f"Single-Turn records: {len(data)}")
    return data

# helper function to deal with the {"type": "text", "text": [the text]} format that a conversation's messages
# are stored in within the dataset
def get_text(message):
    content = message.get("content", [])

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    text_parts.append(part["text"])

        return "\n".join(text_parts).strip()
    
    return ""

# i.e. turning the one record with two LLM outputs into two records with one LLM output each
def split_records(data):
    new_records = []

    for record in data:
        model: str
        conv: list[dict[str, Any]]
        for model, conv in [
            (record["model_a"], record["conversation_a"]),
            (record["model_b"], record["conversation_b"]),
        ]:
            # "user" 
            prompt = get_text(conv[0])
            # "assistant"
            output = get_text(conv[1])

            # reject empty prompts and outputs
            if not prompt or not output:   
                continue

            new_records.append({
                "model": model,
                "prompt": prompt,
                "output": output,
                "is_code": record["is_code"],
                "id": record["id"], # debugging purposes only
            })
    log(f"Split records: {len(new_records)}")

    return new_records

def remove_duplicates(data):
    df = pd.DataFrame(data)

    temp_len = len(df)

    df = df.drop_duplicates(
        subset = ["model", "prompt", "output"]
    )

    log(f"Removed {temp_len - len(df)} duplicate records")
    return df.to_dict("records")

def model_to_family(model):
    for family, pattern in FAMILY_PATTERNS.items():
        if re.match(pattern, model.lower()):
            return family
    return None 

# grabs number of models belong to each family
def get_family_counts(data):
    family_counts = Counter()
    unmatched_models = Counter()

    for record in data:
        family = model_to_family(record["model"])

        if family is not None:
            family_counts[family] += 1
        else:
            unmatched_models[record["model"]] += 1

    log("Family counts:")
    for family, count in family_counts.most_common():
        log(f"{family:<11}| {count}")

    if unmatched_models:
        log("Unmatched models:")
        for model, count in unmatched_models.most_common():
            log(f"{model:<30}| {count}")

    return family_counts

# filter out anything that's not the top NUM_FAMILIES families in terms of number of instances
def filter_families(data, num_families):
    family_counts = get_family_counts(data)
    selected_families = [family for family, _ in family_counts.most_common(num_families)]
    selected_families = set(selected_families)

    log(f"Selected families {', '.join(selected_families)}")

    filtered_data = []

    for record in data:
        record: dict[str, Any]
        family = model_to_family(record["model"])

        if family in selected_families:
            new_record = record.copy()
            new_record["llm_name"] = family
            filtered_data.append(new_record)

    log(f"Records after filtering out families: {len(filtered_data)}")
    return filtered_data

# check is_code distribution, will be useful in implementing RQ3
def see_code_distr(data):
    df = pd.DataFrame(data)

    counts = pd.crosstab(   # contingency table
        df["llm_name"],
        df["is_code"],
        margins = True  # subtotals
    )

    log(f"Family x is_code distribution:\n{counts.to_string()}")
    return counts

def balance_families(data):
    df = pd.DataFrame(data)

    family_counts = df["llm_name"].value_counts()
    target_count = family_counts.min()

    balanced_df = (
        df.groupby("llm_name", group_keys = False)      # don't need another layer of group IDs
        .sample(n = target_count, random_state = SEED)  # select correct count from each group
        .reset_index(drop = True)                       # reset index values and throw away the old ones
    )

    counts = balanced_df["llm_name"].value_counts().to_string()
    log(f"Family counts after balancing: {counts}")

    return balanced_df.to_dict("records")

def shuffle_data(data):
    df = pd.DataFrame(data)
    df = df.sample(frac = 1, random_state = SEED).reset_index(drop = True)
    return df.to_dict("records")

#TODO: implement balancing data for is_code as well

def build_csv(data, path = CSV_FILE):
    df = pd.DataFrame(data)
    df = df[OUTPUT_COLUMNS]

    df["is_code"] = df["is_code"].astype(int)

    df.to_csv(path, index = False, lineterminator = "\n") # alr bro who decided line_terminator shouldn't have an underscore
    log(f"Saved {len(df)} records to {path}")

def save_logs():
    LOG_FILE.parent.mkdir(parents = True, exist_ok = True)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(saved_log) + "\n")

def main():
    data = load_data()
    data = split_records(data)
    data = remove_duplicates(data)
    data = filter_families(data, NUM_FAMILIES)   
    see_code_distr(data)
    data = balance_families(data)
    see_code_distr(data)
    data = shuffle_data(data)
    build_csv(data)
    save_logs()

if __name__ == "__main__":
    main()