import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "mental_health_model.pkl")

st.set_page_config(
    page_title="Mental Health Analyzer",
    page_icon="🧠",
    layout="centered"
)


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = " ".join(text.split())
    return text


def check_artifacts():
    required_files = [VECTORIZER_FILE, LABEL_ENCODER_FILE, MODEL_FILE]
    missing_files = [file for file in required_files if not os.path.exists(file)]
    return missing_files


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(VECTORIZER_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    model = joblib.load(MODEL_FILE)
    return vectorizer, label_encoder, model


def get_color(label):
    return {
        "Normal": "#16a34a",
        "Anxiety": "#f59e0b",
        "Depression": "#2563eb",
        "Suicidal": "#dc2626",
    }.get(label, "#6b7280")


def get_suggestion(label):
    return {
        "Normal": "Keep maintaining a healthy routine and balance.",
        "Anxiety": "Try breathing exercises and talk to someone you trust.",
        "Depression": "Consider reaching out to a trusted person or professional.",
        "Suicidal": "Please seek immediate help from a trusted person or helpline.",
    }.get(label, "")


def main():
    missing_files = check_artifacts()
    if missing_files:
        st.error("Model artifacts not found.")
        st.code("Please run: python src/train.py")
        st.write("Missing files:")
        for file in missing_files:
            st.write(f"- {os.path.basename(file)}")
        st.stop()

    try:
        vectorizer, label_encoder, model = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
            max-width: 700px;
        }
        textarea {
            border-radius: 12px !important;
        }
        .stButton button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("# 🧠 Mental Health Text Analyzer")
    st.markdown("Enter text and the model will predict the mental health category.")
    st.warning("Educational use only. Not a medical diagnosis.")

    text = st.text_area(
        "Enter text",
        height=180,
        placeholder="Type something here..."
    )

    if st.button("Analyze"):
        text = clean_text(text)

        if not text:
            st.error("Please enter some text.")
            st.stop()

        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        label = label_encoder.inverse_transform([pred])[0]
        color = get_color(label)

        st.markdown(f"""
        <div style="
            border-left: 6px solid {color};
            padding: 18px;
            border-radius: 12px;
            background-color: #f8fafc;
            margin-top: 10px;
        ">
            <h3 style="color:{color}; margin:0;">{label}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.info(get_suggestion(label))

        st.markdown("### Confidence Scores")

        sorted_scores = sorted(
            zip(label_encoder.classes_, probs),
            key=lambda x: x[1],
            reverse=True
        )

        for cls, prob in sorted_scores:
            st.write(f"{cls}: {prob:.2f}")
            st.progress(float(prob))

        if label == "Suicidal":
            st.error("⚠️ This indicates severe distress. Immediate help is recommended.")


if __name__ == "__main__":
    main()
