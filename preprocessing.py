from datasets import load_dataset, Dataset
from typing import Any, cast
import pandas as pd
import csv

saved_log = []

def log(line=""):
    print(line)
    saved_log.append(line)

def load_data():
    # casting to avoid type errors when printing length of dataset
    full_data = cast(
        Dataset,
        load_dataset("lmarena-ai/arena-human-preference-140k", split="train[:50]"),
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

# i.e. turning the one record with two LLM outputs into two records with one LLM output each
def split_records(data):
    new_records = []

    for record in data:
        conv_a = record["conversation_a"]
        conv_b = record["conversation_b"]

        record_a = {
            "model": record["model_a"],
            "conv": conv_a
        }
        record_b = {
            "model": record["model_b"],
            "conv": conv_b
        }

        new_records.append(record_a); new_records.append(record_b)
    log(f"Split records: {len(new_records)}")

    return new_records

def main():
    data = load_data()
    data = split_records(data)

    print(data[0]["conv"])

    # args = parse_args()

    # battles = load_battles()
    # df = battles_to_rows(battles)
    # df = add_families(df)

    # families = choose_families(df, args.families, args.num_families)
    # log()
    # log(f"selected families: {', '.join(families)}")

    # df = balance_families(df, families, args.per_family, args.seed)

    # for flag in ["is_code", "is_math", "is_creative_writing"]:
    #     df[flag] = df[flag].astype(int)

    # df = df[OUTPUT_COLUMNS]
    # describe_final(df)

    # args.out.parent.mkdir(parents=True, exist_ok=True)
    # # "\n" line endings so the file is identical no matter whose OS made it
    # df.to_csv(args.out, index=False, lineterminator="\n")

    # summary_path = args.out.with_name(f"{args.out.stem}_summary.txt")
    # summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    # print(f"wrote {summary_path}")

if __name__ == "__main__":
    main()