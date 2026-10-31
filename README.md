# 🛡️ ScamShield AI
### AI-Powered SMS Scam Detection & Cybersecurity Assistant

ScamShield AI is an AI-powered application designed to analyse suspicious SMS messages and provide cybersecurity guidance using Machine Learning, Gemini AI, and Retrieval-Augmented Generation (RAG).

The application combines a machine-learning classifier with AI-generated explanations to help users recognise potential scam indicators and make safer decisions.

## ✨ Features

- **SMS Scam Detection:** Classifies messages as Potential Spam or Likely Legitimate.
- **Machine Learning:** Uses TF-IDF vectorisation and Logistic Regression.
- **Gemini AI Explanations:** Explains possible warning signs in simple English.
- **RAG-Based Guidance:** Retrieves relevant cybersecurity information using FAISS.
- **Interactive Web Interface:** Built with Streamlit.
- **Safety Recommendations:** Encourages users to verify suspicious requests through official channels.

## 🧰 Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Google Gemini API
- LangChain
- FAISS
- Retrieval-Augmented Generation (RAG)

## ⚙️ How It Works

1. The user enters an SMS message.
2. The ML model analyses the message and generates a spam score.
3. FAISS retrieves relevant cybersecurity guidance.
4. Gemini AI uses the message, ML result, and retrieved guidance to generate an explanation.
5. The application displays the classification, score, warning signs, and safety recommendations.

## 📂 Project Structure

```text
ScamShield-AI/
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/harish-tech15/ScamShield-AI.git
cd ScamShield-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Gemini API key

Create a `.env` file only for local development if you choose to load environment variables using a suitable package, or configure `GOOGLE_API_KEY` directly in your operating system's environment.

Never commit API keys or secrets to GitHub.

### 4. Run the Streamlit application

```bash
streamlit run streamlit_app.py
```

## 🔐 Security & Limitations

- This project is an experimental scam-screening tool, not a definitive fraud detector.
- The model's spam score is not a calibrated probability that a message is fraudulent.
- A message classified as legitimate may still be malicious.
- Users should verify unexpected requests through official channels.
- Never share passwords, PINs, OTPs, or banking details with suspicious contacts.
- Gemini API access requires a valid API key.

## 🎯 Project Objective

The objective of ScamShield AI is to demonstrate how Machine Learning, Large Language Models, and Retrieval-Augmented Generation can work together to support SMS scam awareness and cybersecurity education.

## 👨‍💻 Developer

**Harish S.**

GitHub: [harish-tech15](https://github.com/harish-tech15)

---

*Built as an AI and cybersecurity learning project.*
