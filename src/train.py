import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_FILE = os.path.join(DATA_DIR, "mental_health_train.csv")


def load_data():
    df = pd.read_csv(TRAIN_FILE)

    # Keep only needed columns
    df = df[["text", "status"]].copy()

    # Drop missing rows
    df.dropna(subset=["text", "status"], inplace=True)

    # Convert text to string
    df["text"] = df["text"].astype(str).str.strip()

    # Remove empty text rows
    df = df[df["text"] != ""]

    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading training data...")
    df = load_data()
    print(f"Training samples: {len(df)}")

    X_text = df["text"]
    y = df["status"]

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X = vectorizer.fit_transform(X_text)

    print("Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y_encoded)

    print("Saving model artifacts...")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(model, os.path.join(MODEL_DIR, "mental_health_model.pkl"))

    # Training set quick check
    y_pred = model.predict(X)
    acc = accuracy_score(y_encoded, y_pred)

    print("\n===== TRAINING COMPLETE =====")
    print(f"Training Accuracy: {acc:.4f}")
    print("\nClasses:", list(label_encoder.classes_))
    print("\nClassification Report (Train):")
    print(classification_report(y_encoded, y_pred, target_names=label_encoder.classes_))


if __name__ == "__main__":
    main()