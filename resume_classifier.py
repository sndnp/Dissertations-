import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib
import re
import spacy
from PyPDF2 import PdfReader
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
# ------------------------------------------------------------------
# Page design
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Category Predictor",
    page_icon="=�",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg,#f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        padding: 0.5em 1.5em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load model, tokenizer & label encoder
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model &")
def load_model():
    model_dir = "distilbert_resume_classifier"
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    label_enc = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    return tokenizer, model, label_enc

tokenizer, model, label_encoder = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ------------------------------------------------------------------
# Pre-processing
# ------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return "no_content"
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s.!?,]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', text)
    return text if text else "no_content"

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("Resume Category Predictor")
st.markdown("Upload a PDF **resume** **or** paste the text below and let AI guess the super-category in seconds!")

input_text = ""
tab1, tab2 = st.tabs(["= Upload PDF", "= Paste Text"])

with tab1:
    uploaded_file = st.file_uploader("Choose a .pdf resume", type="pdf")
    if uploaded_file:
        reader = PdfReader(uploaded_file)
        input_text = "\n".join(page.extract_text() or "" for page in reader.pages)

with tab2:
    pasted_text = st.text_area("Paste resume text here &", height=250)
    if pasted_text:
        input_text = pasted_text

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if st.button("PREDICT CATEGORY"):
    if not input_text.strip():
        st.error("Please provide resume content first.")
        st.stop()

    with st.spinner("Analyzing resume &"):
        cleaned = clean_text(input_text)
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            pred_id = torch.argmax(logits, dim=1).cpu().item()
            pred_label = label_encoder.inverse_transform([pred_id])[0]

    # Pretty output
    st.success("Prediction complete!")
    st.balloons()
    st.markdown(
        f"<h3 style='text-align:center;color:#2E7D32;'>{pred_label}</h3>",
        unsafe_allow_html=True,
    )
    # ----------  NEW: confidence + word-cloud  ----------
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top3_idx = probs.argsort()[::-1][:3]
    top3_labels = label_encoder.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("   **CONFIDENCE SCORE**")
        chart_data = pd.DataFrame({"Category": top3_labels, "Probability": top3_probs})
        fig, ax = plt.subplots()
        sns.barplot(x="Probability", y="Category", data=chart_data, palette="viridis", ax=ax)
        ax.set_xlabel("Probability")
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.subheader("  **WORLD CLOUD**")
        wc = WordCloud(width=450, height=250, background_color="white",
                       colormap="plasma").generate(cleaned)
        plt.figure(figsize=(5, 3))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    # ----------------------------------------------------

