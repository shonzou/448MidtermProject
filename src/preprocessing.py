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

from datasets import load_dataset

DATASET_NAME = "lmarena-ai/arena-human-preference-140k"
DATA_FILE = "data/train-00000-of-00007.parquet"
CSV_FILE = Path(__file__).resolve().parent.parent / "data" / "data.csv"

def main():
    # no_checks: the dataset card lists the size of all 7 files, we only load 1
    ds = load_dataset(DATASET_NAME, data_files=DATA_FILE, split="train", verification_mode="no_checks")
    ds = ds.select_columns(["language", "conv_metadata", "is_code",
                            "model_a", "model_b", "conversation_a", "conversation_b"])
    print("battles in the file:", len(ds))

    # temporary: look at one message to see how the text is stored
    print(ds[0]["conversation_a"][0])

if __name__ == "__main__":
    main()