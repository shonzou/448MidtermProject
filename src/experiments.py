'''
Trains a CNN and an LSTM to tell which LLM family (Claude, GPT or Gemini)
wrote a response, then answers the project's research questions.

Run preprocessing.py first to create data/data.csv, then:
    python src/experiments.py

RQ1  response -> family
RQ2  prompt only vs response only vs prompt + response
RQ3  train on non-coding prompts, test on coding prompts (and the reverse)
RQ4  style statistics per family + ablations (retrain with one signal removed)
'''

import torch
import processing as p
from logger import log, save_logs


# trains one model (model_name = "CNN" or "LSTM") and returns it with its vocab
def run_experiment(model_name, train_df, val_df, view, label_map, device):
    log(f"\n=====[ {model_name} on {view}, {len(train_df)} training rows ]=====")
    torch.manual_seed(p.SEED)

    vocab = p.build_vocab(train_df, view)

    train_loader = p.make_loader(train_df, view, vocab, label_map, shuffle=True)
    val_loader = p.make_loader(val_df, view, vocab, label_map)

    if model_name == "CNN":
        model = p.CNNClassifier(
            vocab_size=len(vocab),
            num_classes=len(label_map),
        )
    else:
        model = p.LSTMClassifier(
            vocab_size=len(vocab),
            num_classes=len(label_map),
        )

    model = p.train_model(
        model,
        train_loader,
        val_loader,
        device,
    )

    return model, vocab


def main():
    df = p.load_data()

    train_df, val_df, test_df = p.split_data(df)
    log(f"{len(train_df)} train / {len(val_df)} val / {len(test_df)} test responses")

    label_map = p.create_label_map(df)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    results = []  # (experiment, model, test results) for the summary at the end

    # tests a trained model on test_set and saves the scores for the summary
    def test(experiment, model_name, model, vocab, test_set, view="output"):
        log(f"\n{experiment} ({model_name})")
        res = p.evaluate_model(
            model,
            p.make_loader(test_set, view, vocab, label_map),
            device,
            label_map,
        )
        results.append((experiment, model_name, res))

    # RQ1: can we identify LLM from response (the "output" view)
    # RQ2: does the USER prompt help identify the LLM (prompt only vs output only vs both)
    for view in ["output", "prompt", "both"]:
        for model_name in ["CNN", "LSTM"]:
            model, vocab = run_experiment(model_name, train_df, val_df, view, label_map, device)
            test(f"RQ1/RQ2 {view}", model_name, model, vocab, test_df, view)

    # RQ3: train on one kind of prompt, test on the same kind and on the other kind
    domains = {
        "non-coding": df[df["is_code"] == 0],
        "coding": df[df["is_code"] == 1],
    }
    for train_domain, other_domain in [("non-coding", "coding"), ("coding", "non-coding")]:
        domain_train, domain_val, domain_test = p.split_data(domains[train_domain])

        for model_name in ["CNN", "LSTM"]:
            model, vocab = run_experiment(model_name, domain_train, domain_val, "output", label_map, device)
            test(f"RQ3 train {train_domain}, test {train_domain}", model_name, model, vocab, domain_test)
            test(f"RQ3 train {train_domain}, test {other_domain}", model_name, model, vocab, domains[other_domain])

    # RQ4: what is different about each family's responses?
    log("\n=====[ RQ4 style statistics ]=====")
    log("median for words / words per sentence, otherwise share of responses")
    log(p.style_statistics(df).to_string())

    # retrain the CNN with one kind of signal removed from every response
    for label, remove in p.ABLATIONS.items():
        changed = [
            part.assign(output=part["output"].map(remove))
            for part in (train_df, val_df, test_df)
        ]
        model, vocab = run_experiment("CNN", changed[0], changed[1], "output", label_map, device)
        test(f"RQ4 {label}", "CNN", model, vocab, changed[2])

    log("\n=====[ test accuracy (chance = 33.3%) ]=====")
    log(f"{'experiment':<42}{'model':<7}{'accuracy':<10}balanced accuracy")
    for experiment, model_name, res in results:
        log(f"{experiment:<42}{model_name:<7}{res['Accuracy']:<10.1%}{res['Accuracy Balanced']:.1%}")
    save_logs()


if __name__ == "__main__":
    main()
