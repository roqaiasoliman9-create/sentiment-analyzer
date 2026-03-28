import os
from groq import Groq


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    api_key = api_key.strip()

    if not api_key.startswith("gsk_"):
        raise ValueError("GROQ_API_KEY looks invalid.")

    return Groq(api_key=api_key)


def predict_with_groq(text: str) -> str:
    client = get_groq_client()

    prompt = f"""
Classify the sentiment of the following text into exactly one label:
positive, negative, or neutral.

Text: "{text}"

Return only one word:
positive
negative
neutral
""".strip()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a precise sentiment classifier."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    result = response.choices[0].message.content.strip().lower()

    if result not in {"positive", "negative", "neutral"}:
        return "neutral"

    return result


if __name__ == "__main__":
    text = "The product is okay, nothing special"
    print("Key loaded:", os.getenv("GROQ_API_KEY", "")[:10])
    print(predict_with_groq(text))