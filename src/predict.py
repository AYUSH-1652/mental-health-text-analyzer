import os
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def check_artifacts():
    required_files = [
        "tfidf_vectorizer.pkl",
        "label_encoder.pkl",
        "mental_health_model.pkl"
    ]

    for file in required_files:
        path = os.path.join(MODEL_DIR, file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model file: {file}. Run train.py first.")


def load_artifacts():
    check_artifacts()
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "mental_health_model.pkl"))
    return vectorizer, label_encoder, model


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = " ".join(text.split())
    return text


vectorizer, label_encoder, model = load_artifacts()


def predict_text(text):
    text = clean_text(text)

    text_vector = vectorizer.transform([text])
    pred = model.predict(text_vector)[0]
    probs = model.predict_proba(text_vector)[0]

    predicted_label = label_encoder.inverse_transform([pred])[0]

    class_probabilities = {
        label_encoder.classes_[i]: float(probs[i])
        for i in range(len(label_encoder.classes_))
    }

    return predicted_label, class_probabilities


if __name__ == "__main__":
    user_text = input("Enter text: ").strip()

    if not user_text:
        print("Please enter some text.")
    else:
        label, probs = predict_text(user_text)

        print(f"\nPredicted Mental Health Status: {label}")
        print("\nClass Probabilities:")

        for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"{cls}: {prob:.4f}")
