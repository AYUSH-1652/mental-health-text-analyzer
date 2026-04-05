import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="Mental Health Analyzer",
    page_icon="🧠",
    layout="centered"
)


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "mental_health_model.pkl"))
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
    vectorizer, label_encoder, model = load_artifacts()

    # Clean CSS
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

    # Title
    st.markdown("# 🧠 Mental Health Text Analyzer")

    st.markdown(
        "Enter text and the model will predict the mental health category."
    )

    st.warning("Educational use only. Not a medical diagnosis.")

    # Input
    text = st.text_area(
        "Enter text",
        height=180,
        placeholder="Type something here..."
    )

    if st.button("Analyze"):
        if not text.strip():
            st.error("Please enter some text.")
            return

        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        label = label_encoder.inverse_transform([pred])[0]
        color = get_color(label)

        # Result card
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

        # Suggestion
        st.info(get_suggestion(label))

        # Confidence bars
        st.markdown("### Confidence Scores")

        for i, cls in enumerate(label_encoder.classes_):
            st.write(f"{cls}: {probs[i]:.2f}")
            st.progress(float(probs[i]))

        # Extra warning
        if label == "Suicidal":
            st.error("⚠️ This indicates severe distress. Immediate help is recommended.")


if __name__ == "__main__":
    main()