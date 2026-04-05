import os
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TEST_FILE = os.path.join(DATA_DIR, "mental_health_combined_test.csv")


def load_test_data():
    df = pd.read_csv(TEST_FILE)

    df = df[["text", "status"]].copy()
    df.dropna(subset=["text", "status"], inplace=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    return df


def main():
    print("Loading saved artifacts...")
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "mental_health_model.pkl"))

    print("Loading test data...")
    df = load_test_data()

    X_test_text = df["text"]
    y_test = df["status"]

    X_test = vectorizer.transform(X_test_text)
    y_test_encoded = label_encoder.transform(y_test)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test_encoded, y_pred)
    cm = confusion_matrix(y_test_encoded, y_pred)
    report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)

    print("\n===== TEST RESULTS =====")
    print(f"Test Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()