import os
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_artifacts():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "mental_health_model.pkl"))
    return vectorizer, label_encoder, model


def predict_text(text):
    vectorizer, label_encoder, model = load_artifacts()

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