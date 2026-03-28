from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from train import prepare_features
from sklearn.linear_model import LogisticRegression


def evaluate_model():
    X, y, _, _ = prepare_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=== Evaluation Results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()
