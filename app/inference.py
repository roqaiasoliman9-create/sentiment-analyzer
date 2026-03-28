import joblib
from pathlib import Path

from preprocess import clean_text
from groq_helper import predict_with_groq


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def get_rule_based_prediction(cleaned_text: str):
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
        return {
            "prediction": "neutral",
            "scores": scores,
            "source": "rule-based"
        }

    return None


def get_ml_prediction(text: str, model, vectorizer):
    cleaned_text = clean_text(text)
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

    return {
        "prediction": prediction,
        "scores": scores,
        "source": "ml-model"
    }


def get_groq_prediction(text: str):
    groq_prediction = predict_with_groq(text)
    groq_scores = {
        "negative": 0.0,
        "neutral": 0.0,
        "positive": 0.0
    }
    groq_scores[groq_prediction] = 1.0

    return {
        "prediction": groq_prediction,
        "scores": groq_scores,
        "source": "groq-only"
    }


def predict_sentiment(text, model, vectorizer, mode="auto"):
    cleaned_text = clean_text(text)

    if mode == "ml":
        return get_ml_prediction(text, model, vectorizer)

    if mode == "groq":
        return get_groq_prediction(text)

    rule_result = get_rule_based_prediction(cleaned_text)
    if rule_result:
        return rule_result

    ml_result = get_ml_prediction(text, model, vectorizer)
    max_score = max(ml_result["scores"].values()) if ml_result["scores"] else 0.0

    if max_score < 0.60:
        return {
            **get_groq_prediction(text),
            "source": "groq-fallback"
        }

    return ml_result