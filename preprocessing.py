from datasets import load_dataset, Dataset
from typing import Any, cast
import pandas as pd
import csv

load_dataset("lmarena-ai/arena-human-preference-140k")

saved_log = []

def log(line=""):
    print(line)
    saved_log.append(line)

def load_data():
    # casting to avoid type errors when printing length of dataset
    full_data = cast(
        Dataset,
        load_dataset("lmarena-ai/arena-human-preference-140k", split="train[:500]"),
    )

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

def main():
    data = load_data()
    print(data[0]["conv_metadata"])

if __name__ == "__main__":
    main()