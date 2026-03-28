import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from data_loader import load_data
from preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "imdb_sample.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


def prepare_features():
    df = load_data(DATA_PATH)

    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    return X, y, vectorizer, df


def train_model():
    X, y, vectorizer, df = prepare_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Training Results ===")
    print("Dataset size:", len(df))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, vectorizer


def save_artifacts(model, vectorizer):
    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nModel saved to:", MODEL_PATH)
    print("Vectorizer saved to:", VECTORIZER_PATH)


if __name__ == "__main__":
    model, vectorizer = train_model()
    save_artifacts(model, vectorizer)