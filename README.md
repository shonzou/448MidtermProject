# Identifying LLM Families from Their Responses

CMPSC 448 Midterm Project | Fall 2026 | Professor Wenpeng Yin

**Team:** Jethin Grandhi (Team Lead), Anushka Shrivastava, Shon Zou

In this project, we explore whether an LLM's response contains enough clues to identify the model family that produced it. We trained a Convolutional Neural Network (CNN) and a bidirectional Long Short-Term Memory network (LSTM) to classify responses as coming from Claude, GPT, or Gemini. Both models achieved around 90% accuracy on unseen responses, compared with a 33.3% baseline for random guessing.

For a more detailed discussion of our approach and findings, see [Report.pdf](Report.pdf).

## Results

The table below shows the test accuracy for each experiment. Since the three classes are balanced, random guessing would achieve approximately 33.3% accuracy.

| Experiment | CNN | LSTM |
|---|---|---|
| RQ1: Response only | 90.2% | 91.6% |
| RQ2: Prompt only | 33.5% | 32.6% |
| RQ2: Prompt + response | 90.6% | 89.4% |
| RQ3: Train on non-coding, test on non-coding | 90.9% | 90.0% |
| RQ3: Train on non-coding, test on coding | 89.1% | 88.4% |
| RQ3: Train on coding, test on coding | 90.3% | 89.5% |
| RQ3: Train on coding, test on non-coding | 85.7% | 82.9% |
| RQ4: Model and company names masked | 90.7% | N/A |
| RQ4: Words only (formatting removed) | 75.5% | N/A |
| RQ4: Formatting only (words hidden) | 89.7% | N/A |

A few findings stand out. The models can identify an LLM family from its response, but the prompt alone provides almost no useful signal. The patterns they learn also transfer fairly well between coding and non-coding conversations. Our RQ4 experiments suggest that formatting choices, including Markdown, punctuation, and layout, play a larger role in identification than the actual words used.

## How to Run the Project

The project requires Python 3 (tested with Python 3.12) and can run entirely on a CPU; a GPU is not needed.

From the project directory, run:

```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/experiments.py
```

1. **Preprocessing:** `preprocessing.py` downloads one of the source dataset files from Hugging Face, filters and balances the records, and creates `data/data.csv` with 8,121 responses. Its output is also saved to `log/pre_logs.txt`. Once the file has been downloaded, preprocessing takes about 15 seconds.
2. **Experiments:** `experiments.py` trains 13 models across the four research questions. For each test, it reports precision, recall, and F1-score for each model family, along with a confusion matrix. It then prints the style statistics and an overall results table. The output is also written to `log/log.txt`. A full run takes approximately 15–20 minutes on a laptop CPU.

We use a fixed random seed of `0` throughout the project. Two complete runs on the same machine produced identical results.

## Dataset

We use the [lmarena-ai/arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) dataset from Hugging Face. Each record represents a vote on LMArena, where two anonymous models answer the same user prompt. For this project, we use the first of the dataset's seven files.

The preprocessing steps reduce the source data as follows:

| Preprocessing step | Records remaining |
|---|---:|
| Original votes in the selected file | 19,377 |
| Keep English-language conversations | 10,226 |
| Keep single-turn conversations | 8,759 |
| Separate each vote into two responses and remove empty responses | 17,513 |
| Remove exact duplicates | 17,511 |
| Keep the three largest model families | 9,170 |
| Balance the classes by keeping 2,707 responses per family | 8,121 |

The dataset itself is not included in this repository. The scripts generate the `data/` and `log/` directories when they run. According to the dataset card, user prompts are licensed under CC BY 4.0, while model outputs remain subject to the respective providers' terms of use.

## Method Overview

- **Tokenization:** We lowercase words and treat each punctuation mark, symbol, and line break as a separate token. We keep up to 256 tokens per text and build a vocabulary from the 20,000 most common tokens in the training data.
- **Data split:** We split the data by prompt into 80% training, 10% validation, and 10% testing. This prevents responses to the same prompt from appearing in both the training and test sets.
- **CNN architecture:** A 128-dimensional embedding layer feeds into a 1-D convolution with 128 filters and a six-token window, followed by ReLU, max pooling across the text, 0.5 dropout, and a linear output layer.
- **LSTM architecture:** A 128-dimensional embedding layer feeds into a bidirectional LSTM with 128 hidden units in each direction, followed by max pooling across the text, 0.5 dropout, and a linear output layer.
- **Training:** Both models use cross-entropy loss and the Adam optimizer with a learning rate of 0.001. We train with a batch size of 64 for 10 epochs and retain the weights from the epoch with the highest validation accuracy.

## Repository Structure

| File | Purpose |
|---|---|
| `src/preprocessing.py` | Downloads and prepares the source data, producing `data/data.csv`. |
| `src/processing.py` | Contains the tokenizer, vocabulary, model architectures, training and evaluation code, and the RQ4 style statistics and ablation experiments. Configuration settings, such as sequence length and number of epochs, are defined at the top of the file. |
| `src/experiments.py` | Runs the experiments for RQ1–RQ4 and generates the summary table. |
| `src/logger.py` | Displays experiment output and saves it to the `log/` directory. |
| `requirements.txt` | Lists the Python dependencies needed to run the project. |
