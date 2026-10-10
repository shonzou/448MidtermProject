'''
Builds data/data.csv (llm_name, model, prompt, output, is_code) from LMArena battles.

Source: lmarena-ai/arena-human-preference-140k on Hugging Face. Each row is one
human vote between two models that answered the same user prompt.
License: user prompts are CC-BY-4.0, model outputs fall under each provider's
terms of use.

1 load 1 of the dataset's 7 parquet files (~230 MB, a random sample of all battles)
2 keep english (language = en), single-turn battles (one prompt + one response)
3 split each battle into two data points:
    given original (model_a, model_b, conversation_a, conversation_b),
        = (model_a, prompt, response_a)
        = (model_b, prompt, response_b)
4 drop empty responses and exact duplicates
5 group model versions into families (i.e. gpt-4.1-mini -> GPT) and keep only
  Claude, GPT and Gemini
6 random sample each family down to the same size so classes are balanced

Note: no train/test splits here, processing.py does that
'''
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "lmarena-ai/arena-human-preference-140k"
DATA_FILE = "data/train-00000-of-00007.parquet"
CSV_FILE = Path(__file__).resolve().parent.parent / "data" / "data.csv"

# message content is a list of parts like {"type": "text", "text": ...}
def message_text(message):
    return "\n".join(
        part["text"] for part in message["content"]
        if part["type"] == "text" and part["text"]
    ).strip()

def main():
    # no_checks: the dataset card lists the size of all 7 files, we only load 1
    ds = load_dataset(DATASET_NAME, data_files=DATA_FILE, split="train", verification_mode="no_checks")
    ds = ds.select_columns(["language", "conv_metadata", "is_code",
                            "model_a", "model_b", "conversation_a", "conversation_b"])
    print("battles in the file:", len(ds))

    # single-turn conversations are [user prompt, model response]
    battles = []
    for row in ds:
        if row["language"] == "en" and row["conv_metadata"]["turns"] == 1:
            battles.append({
                "prompt": message_text(row["conversation_a"][0]),
                "is_code": int(row["is_code"]),
                "model_a": row["model_a"],
                "output_a": message_text(row["conversation_a"][1]),
                "model_b": row["model_b"],
                "output_b": message_text(row["conversation_b"][1]),
            })
    battles = pd.DataFrame(battles)
    print("english + single-turn:", len(battles))

        # one row per model response
    df = pd.concat([
        battles[["prompt", "is_code", "model_a", "output_a"]].rename(columns={"model_a": "model", "output_a": "output"}),
        battles[["prompt", "is_code", "model_b", "output_b"]].rename(columns={"model_b": "model", "output_b": "output"}),
    ], ignore_index=True)

    df = df[df["output"] != ""].drop_duplicates(subset=["model", "prompt", "output"])
    print("non-empty, no duplicates:", len(df))

    # temporary: see which models are in the data
    print(df["model"].value_counts().to_string())

if __name__ == "__main__":
    main()