import torch
import processing as p


def run_experiment(name, train_df, val_df, test_df, text_column, label_map, device):
    print(f"\n=====[ {name} ]=====")

    vocab = p.build_vocab(train_df, text_column)

    # TODO: compare CNN and RNN performance when RNN is set up
    # TODO: testing between code/programming prompts and non-code prompts

    # TODO: train and evaluate classifier using LLM input only for RQ2
    # TODO: train and evaluate classifier using LLM input and output for RQ2

    train_loader, val_loader, test_loader = p.create_dataloaders(
        train_df,
        val_df,
        test_df,
        text_column,
        label_map,
        vocab,
    )

    model = p.CNNClassifier(
        vocab_size=len(vocab),
        num_classes=len(label_map),
    )

    model = p.train_model(
        model,
        train_loader,
        val_loader,
        device,
        model_path=f"{name}.pt",
    )

    results = p.evaluate_model(
        model,
        test_loader,
        device,
        label_map,
    )

    return results


def main():
    df = p.load_data()

    # create combined input for RQ2
    df["prompt_and_output"] = (
        df["prompt"].fillna("") + " " + df["output"].fillna("")
    )

    train_df, val_df, test_df = p.split_data(df)

    label_map = p.create_label_map(df)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # RQ1: can we identify LLM from response
    rq1_cnn_res = run_experiment(
        "cnn_output",
        train_df, val_df, test_df,
        "output", label_map, device,
    )

    
    # RQ2: does the USER prompt help identify the LLM
    rq2_cnn_res = run_experiment(
        "cnn_prompt",
        train_df, val_df, test_df,
        "prompt", label_map, device,
    )

    rq2_cnn_both_res = run_experiment(
        "cnn_prompt_and_output",
        train_df, val_df, test_df,
        "prompt_and_output", label_map, device,
    )


if __name__ == "__main__":
    main()