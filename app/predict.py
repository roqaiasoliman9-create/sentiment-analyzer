import joblib
from pathlib import Path

from preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_sentiment(text, model, vectorizer):
    cleaned_text = clean_text(text)

    neutral_phrases = [
        "okay",
        "nothing special",
        "average",
        "normal",
        "fine",
        "as expected",
        "ordinary",
        "acceptable",
        "not bad",
        "fair enough"
    ]

    if any(phrase in cleaned_text for phrase in neutral_phrases):
        scores = {
            "negative": 0.10,
            "neutral": 0.80,
            "positive": 0.10
        }
        return "neutral", scores

    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vector)[0]
        classes = model.classes_
        scores = dict(zip(classes, probabilities))
    else:
        scores = {
            "negative": 0.0,
            "neutral": 0.0,
            "positive": 0.0
        }

    return prediction, scores


def interactive_mode():
    model, vectorizer = load_artifacts()

    print("\n=== Sentiment Predictor ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_text = input("Enter text: ").strip()

        if user_text.lower() == "exit":
            print("Goodbye.")
            break

        if not user_text:
            print("Please enter valid text.\n")
            continue

        prediction, scores = predict_sentiment(user_text, model, vectorizer)

        print(f"\nPrediction: {prediction}")
        print("Confidence scores:")
        for label, score in scores.items():
            print(f"  {label}: {score:.4f}")
        print()


if __name__ == "__main__":
    interactive_mode()