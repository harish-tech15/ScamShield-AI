import os
import zipfile
import urllib.request

import gradio as gr
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
# 1. Configuration
# --------------------------------------------------

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is missing. "
        "Set it as an environment variable or app secret."
    )

gemini_client = genai.Client(api_key=API_KEY)

GEMINI_MODEL = "gemini-3.8-flash"


# --------------------------------------------------
# 2. Download and prepare the SMS dataset
# --------------------------------------------------

def load_dataset():
    dataset_url = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/00228/smsspamcollection.zip"
    )

    zip_path = "sms_spam.zip"
    data_dir = "sms_spam_data"
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
# 3. Train the machine-learning model
# --------------------------------------------------

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

    print(f"Model training completed. Test accuracy: {accuracy:.4f}")

    return pipeline


# --------------------------------------------------
# 4. Build the cybersecurity knowledge base
# --------------------------------------------------

def create_knowledge_base():
    guidance = [
        """
        Phishing messages may impersonate banks, government agencies,
        delivery companies, or popular brands. Verify suspicious requests
        through official websites or independently verified contact details.
        """,
        """
        Never share your banking PIN, password, or OTP with someone
        who contacts you unexpectedly. Do not enter sensitive information
        on websites opened through suspicious messages.
        """,
        """
        Unexpected prize, lottery, or reward messages may be scams,
        particularly when they demand advance payments or personal details.
        Verify the offer independently before responding.
        """,
        """
        Messages threatening immediate account suspension or demanding
        urgent payments should be checked carefully. Urgency alone does
        not prove a message is fraudulent.
        """,
        """
        If you shared banking credentials with a suspicious website,
        contact your bank promptly using its official contact details.
        Change affected passwords and follow the provider's security advice.
        """,
        """
        Genuine-looking messages can still be fraudulent. Inspect links
        carefully without opening them and independently verify unusual
        requests. Spelling mistakes alone cannot reliably identify scams.
        """
    ]

    documents = [
        Document(
            page_content=text.strip(),
            metadata={"source": f"cybersecurity_guide_{index + 1}"}
        )
        for index, text in enumerate(guidance)
    ]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=API_KEY
    )

    return FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )


print("Loading dataset and training model...")
model = train_model()

print("Building cybersecurity knowledge base...")
vector_store = create_knowledge_base()

print("ScamShield AI is ready!")


# --------------------------------------------------
# 5. Analyse a message
# --------------------------------------------------

def scamshield_chat(message):
    if not message or not message.strip():
        return "Please enter a message to analyse."

    # Machine-learning prediction
    prediction = int(model.predict([message])[0])
    spam_score = float(model.predict_proba([message])[0][1])

    category = (
        "Potential Spam"
        if prediction == 1
        else "Likely Legitimate"
    )

    # Retrieve relevant cybersecurity guidance
    relevant_docs = vector_store.similarity_search(
        message,
        k=3
    )

    context = "\n\n".join(
        document.page_content
        for document in relevant_docs
    )

    prompt = f"""
You are ScamShield AI, a cybersecurity assistant.

Analyse the user's message using the machine-learning result and
the supplied cybersecurity guidance.

Message:
{message}

Machine-learning classification: {category}
Model spam score: {spam_score:.2%}

Retrieved cybersecurity guidance:
{context}

Respond in simple English using these sections:

1. Detection result
2. Why the message may be suspicious or legitimate
3. Warning signs to check
4. Recommended safe actions

Do not claim that a message is definitely safe or fraudulent based
only on this classifier. Explain that the score is a model estimate,
not a calibrated probability of fraud.

Never ask the user to reveal passwords, PINs, OTPs, or banking details.
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    explanation = response.text or (
        "No explanation was returned. Please try again."
    )

    return f"""
## 🛡️ ScamShield AI Analysis

**ML classification:** {category}

**Model spam score:** {spam_score:.2%}

---

{explanation}

---

⚠️ **Safety reminder:** This is an experimental tool.
A low spam score does not guarantee that a message is safe.
Verify unusual requests through official channels.
"""


# --------------------------------------------------
# 6. Gradio chatbot interface
# --------------------------------------------------

def respond(message, history):
    if not message or not message.strip():
        return history, ""

    history = history or []

    try:
        answer = scamshield_chat(message)
    except Exception as error:
        print(f"Analysis error: {error}")

        answer = (
            "Sorry, I could not analyse that message right now. "
            "Please check the application logs and API configuration, "
            "then try again."
        )

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]

    return history, ""


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🛡️ ScamShield AI
        ### AI-Powered Scam Detection & Cybersecurity Assistant

        Paste a suspicious SMS or message to analyse it using
        Machine Learning, Gemini AI, and RAG-based guidance.

        ⚠️ Do not enter real passwords, PINs, OTPs, or banking details.
        """
    )

    chatbot = gr.Chatbot(
        label="ScamShield AI Assistant",
        height=450
    )

    message_input = gr.Textbox(
        label="Enter your message",
        placeholder=(
            "Example: Congratulations! You won a prize. "
            "Click this link..."
        ),
        lines=3
    )

    with gr.Row():
        analyse_button = gr.Button(
            "🔍 Analyse Message",
            variant="primary"
        )

        clear_button = gr.Button("🗑️ Clear Chat")

    analyse_button.click(
        fn=respond,
        inputs=[message_input, chatbot],
        outputs=[chatbot, message_input]
    )

    message_input.submit(
        fn=respond,
        inputs=[message_input, chatbot],
        outputs=[chatbot, message_input]
    )

    clear_button.click(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[chatbot, message_input]
    )


if __name__ == "__main__":
    demo.launch()
