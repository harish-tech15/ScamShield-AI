import os
import zipfile
import urllib.request

import streamlit as st
import pandas as pd

from google import genai
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ScamShield AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ ScamShield AI")
st.subheader("AI-Powered Scam Detection & Cybersecurity Assistant")

st.markdown(
    """
    Analyse suspicious SMS messages using Machine Learning,
    Gemini AI, and Retrieval-Augmented Generation (RAG).

    ⚠️ Do not enter real passwords, PINs, OTPs, or banking details.
    """
)

# --------------------------------------------------
# API CONFIGURATION
# --------------------------------------------------

def get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("GOOGLE_API_KEY", "")


API_KEY = get_api_key()
GEMINI_MODEL = "gemini-3.8-flash"


# --------------------------------------------------
# LOAD SMS DATASET
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dataset():
    dataset_url = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/00228/smsspamcollection.zip"
    )

    zip_path = "/tmp/sms_spam.zip"
    data_dir = "/tmp/sms_spam_data"
    file_path = os.path.join(data_dir, "SMSSpamCollection")

    if not os.path.exists(file_path):
        os.makedirs(data_dir, exist_ok=True)

        urllib.request.urlretrieve(dataset_url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(data_dir)

    data = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    data = data.dropna(subset=["label", "message"])
    data = data.drop_duplicates(subset=["label", "message"])
    data = data[data["label"].isin(["ham", "spam"])].copy()

    data["target"] = data["label"].map({
        "ham": 0,
        "spam": 1
    })

    return data


# --------------------------------------------------
# TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner="Training Machine Learning model...")
def train_model():
    data = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["target"],
        test_size=0.2,
        random_state=42,
        stratify=data["target"]
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    accuracy = pipeline.score(X_test, y_test)

    return pipeline, accuracy


# --------------------------------------------------
# BUILD RAG KNOWLEDGE BASE
# --------------------------------------------------

@st.cache_resource(show_spinner="Building cybersecurity knowledge base...")
def create_knowledge_base(api_key):
    guidance = [
        """
        Phishing messages may impersonate banks, government agencies,
        delivery companies, or popular brands. Verify suspicious requests
        through official websites or independently verified contact details.
        """,
        """
        Never share your banking PIN, password, or OTP with someone
        who contacts you unexpectedly. Avoid entering sensitive information
        on websites opened through suspicious messages.
        """,
        """
        Unexpected prize, lottery, or reward messages may be scams,
        especially when they demand advance payments or personal details.
        Verify offers independently before responding.
        """,
        """
        Messages threatening immediate account suspension or demanding
        urgent payments should be checked carefully. Urgency alone does
        not prove that a message is fraudulent.
        """,
        """
        If you shared banking credentials with a suspicious website,
        contact your bank using its official contact details. Change
        affected passwords and follow the provider's security advice.
        """,
        """
        Genuine-looking messages can still be fraudulent. Inspect links
        carefully and independently verify unusual requests. Spelling
        mistakes alone cannot reliably identify scams.
        """
    ]

    documents = [
        Document(
            page_content=text.strip(),
            metadata={"source": f"cybersecurity_guide_{i + 1}"}
        )
        for i, text in enumerate(guidance)
    ]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    return FAISS.from_documents(documents, embeddings)


# --------------------------------------------------
# ANALYSE MESSAGE
# --------------------------------------------------

def analyse_message(message, model, vector_store, api_key):
    prediction = int(model.predict([message])[0])
    spam_score = float(model.predict_proba([message])[0][1])

    category = (
        "Potential Spam"
        if prediction == 1
        else "Likely Legitimate"
    )

    relevant_docs = vector_store.similarity_search(message, k=3)

    context = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    prompt = f"""
You are ScamShield AI, a cybersecurity assistant.

Analyse this message using the machine-learning result
and the supplied cybersecurity guidance.

Message:
{message}

ML classification: {category}
Model spam score: {spam_score:.2%}

Retrieved cybersecurity guidance:
{context}

Explain in simple English:
1. Detection result
2. Why the message may be suspicious or legitimate
3. Warning signs
4. Recommended safe actions

The model score is not a calibrated probability of fraud.
Do not claim that a message is definitely safe or fraudulent.
Never ask for passwords, PINs, OTPs, or banking details.
"""

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    explanation = response.text or "No explanation was returned."

    return category, spam_score, explanation


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("🔎 About ScamShield AI")

    st.write(
        "An experimental SMS scam screening tool "
        "combining ML, Gemini AI, and RAG."
    )

    st.info(
        "A low spam score does not guarantee a message is safe. "
        "Always verify unusual requests through official channels."
    )


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if not API_KEY:
    st.error(
        "Gemini API key is missing. Add GOOGLE_API_KEY "
        "under your Streamlit app's Secrets settings."
    )
    st.stop()


message = st.text_area(
    "📩 Enter the SMS or message to analyse",
    placeholder=(
        "Example: Congratulations! You won a prize. "
        "Click this link to claim it..."
    ),
    height=150
)

analyse_button = st.button(
    "🔍 Analyse Message",
    type="primary",
    use_container_width=True
)

if analyse_button:
    if not message.strip():
        st.warning("Please enter a message to analyse.")

    else:
        try:
            with st.spinner("Initialising ScamShield AI..."):
                model, accuracy = train_model()
                vector_store = create_knowledge_base(API_KEY)

            st.success("ScamShield AI is ready!")

            with st.spinner("Analysing your message..."):
                category, score, explanation = analyse_message(
                    message,
                    model,
                    vector_store,
                    API_KEY
                )

            st.divider()
            st.header("🛡️ Analysis Results")

            col1, col2 = st.columns(2)

            with col1:
                if category == "Potential Spam":
                    st.error(f"🚨 {category}")
                else:
                    st.success(f"✅ {category}")

            with col2:
                st.metric("Model Spam Score", f"{score:.2%}")

            st.caption(
                f"Model test accuracy on held-out data: {accuracy:.2%}. "
                "This is not a guarantee of real-world scam detection."
            )

            st.subheader("🤖 Gemini AI Explanation")
            st.markdown(explanation)

            st.warning(
                "Safety reminder: This is an experimental tool. "
                "Verify suspicious messages independently."
            )

        except Exception as error:
            st.error(
                "Analysis failed. Check the app logs and API "
                "configuration, then try again."
            )
            st.exception(error)


st.divider()

st.caption(
    "ScamShield AI | Machine Learning • Gemini AI • RAG • FAISS"
)
