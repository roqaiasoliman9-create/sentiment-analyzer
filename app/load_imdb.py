from datasets import load_dataset
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "imdb_sample.csv"


def main():
    dataset = load_dataset("imdb")
    train_data = dataset["train"]

    negative_texts = []
    positive_texts = []

    for item in train_data:
        if item["label"] == 0 and len(negative_texts) < 500:
            negative_texts.append(item["text"])
        elif item["label"] == 1 and len(positive_texts) < 500:
            positive_texts.append(item["text"])

        if len(negative_texts) == 500 and len(positive_texts) == 500:
            break

    texts = negative_texts + positive_texts
    labels = ["negative"] * 500 + ["positive"] * 500

    df = pd.DataFrame({
        "text": texts,
        "label": labels
    })

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved dataset to:", OUTPUT_PATH)
    print(df["label"].value_counts())
    print(df.head())
    print("\nShape:", df.shape)


if __name__ == "__main__":
    main()