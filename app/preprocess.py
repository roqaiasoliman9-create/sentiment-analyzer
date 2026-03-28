import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    return " ".join(words)


if __name__ == "__main__":
    sample = "I love this product, it's amazing!"
    cleaned = clean_text(sample)
    print("Original:", sample)
    print("Cleaned:", cleaned)