import os
import json
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TEST_FILE = os.path.join(DATA_DIR, "mental_health_combined_test.csv")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "mental_health_model.pkl")
RESULTS_FILE = os.path.join(MODEL_DIR, "evaluation_metrics.json")

MIN_ACCEPTABLE_ACCURACY = 0.70


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = " ".join(text.split())
    return text


def load_test_data():
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    df = pd.read_csv(TEST_FILE)
    df = df[["text", "status"]].copy()
    df.dropna(subset=["text", "status"], inplace=True)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""]

    return df


def check_artifacts():
    required_files = [VECTORIZER_FILE, LABEL_ENCODER_FILE, MODEL_FILE]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required artifact not found: {file_path}")


def main():
    print("Checking saved artifacts...")
    check_artifacts()

    print("Loading saved artifacts...")
    vectorizer = joblib.load(VECTORIZER_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    model = joblib.load(MODEL_FILE)

    print("Loading test data...")
    df = load_test_data()

    X_test_text = df["text"]
    y_test = df["status"]

    X_test = vectorizer.transform(X_test_text)
    y_test_encoded = label_encoder.transform(y_test)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test_encoded, y_pred)
    cm = confusion_matrix(y_test_encoded, y_pred)
    report_dict = classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )
    report_text = classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_
    )

    results = {
        "test_accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("\n===== TEST RESULTS =====")
    print(f"Test Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report_text)
    print(f"\nSaved evaluation metrics to: {RESULTS_FILE}")

    if acc < MIN_ACCEPTABLE_ACCURACY:
        raise ValueError(
            f"Model accuracy {acc:.4f} is below the acceptable threshold of {MIN_ACCEPTABLE_ACCURACY:.2f}"
        )


if __name__ == "__main__":
    main()
