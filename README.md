# 🧠 Mental Health Text Analyzer

A **Machine Learning + DevOps powered web application** that analyzes user text and predicts mental health categories such as **Anxiety, Depression, Normal, and Suicidal**, along with confidence scores and helpful suggestions.

---

## 🚀 Features

- 📝 Text-based mental health classification
- 📊 Confidence score visualization
- 💡 Context-aware suggestions
- ⚡ Fast TF-IDF + Logistic Regression model
- 🌐 Interactive UI using Streamlit
- 🔁 CI/CD pipeline using GitHub Actions
- 🐳 Dockerized for portability and deployment

---

## 🧠 ML Pipeline

```
Text Input → Cleaning → TF-IDF Vectorization → Logistic Regression → Prediction → UI Output
```

- Feature extraction: **TF-IDF (unigrams + bigrams)**
- Model: **Logistic Regression (class_weight=balanced)**
- Classes:
  - Anxiety
  - Depression
  - Normal
  - Suicidal

---

## ⚙️ DevOps Pipeline

```
Code Push → GitHub Actions (CI) → Docker Build → Push to Docker Hub → Run Anywhere
```

- Automated CI using GitHub Actions  
- Docker image built on every push  
- Image pushed to Docker Hub  
- Application runnable on any system  

---

## 📂 Project Structure

```
mental-health-text-analyzer/
│
├── data/                      # Dataset (train & test)
├── src/
│   ├── train.py              # Model training
│   ├── evaluate.py           # Model evaluation
│   ├── predict.py            # Prediction logic
│   └── app.py                # Streamlit UI
│
├── models/                   # Saved model artifacts
├── .github/workflows/        # CI/CD pipelines
├── Dockerfile                # Docker configuration
├── requirements.txt          # Dependencies
└── README.md
```

---

## 🧪 Model Performance

- Training Accuracy: ~87%
- Test Accuracy: ~74%

---

## 🌐 Live Demo

👉 https://your-app-link.streamlit.app

---

## 🐳 Run using Docker

### Pull image
```bash
docker pull ayush1652/mental-health-text-analyzer:latest
```

### Run container
```bash
docker run -p 8501:8501 ayush1652/mental-health-text-analyzer:latest
```

### Open in browser
```
http://localhost:8501
```

---

## 💻 Run Locally (Without Docker)

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python src/train.py

streamlit run src/app.py
```

---

## 🔁 CI/CD (GitHub Actions)

- Runs automatically on every push
- Builds Docker image
- Pushes image to Docker Hub

👉 Check the **Actions tab** in this repository

---

## 🧠 Use Cases

- Mental health awareness tools
- NLP-based text classification
- ML + DevOps demonstration projects
- Educational purposes

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**.  
It is **not a medical diagnosis system**.

---

## 👨‍💻 Author

**Ayush**  
GitHub: https://github.com/AYUSH-1652

---

## ⭐ Project Highlights

- End-to-end ML pipeline  
- Integrated DevOps (CI/CD)  
- Dockerized deployment  
- Production-ready architecture  
