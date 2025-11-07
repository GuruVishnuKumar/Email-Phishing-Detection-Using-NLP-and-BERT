import streamlit as st
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ----------------------
# Load spaCy NLP Model
# ----------------------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# -----------------------------
# NLP Feature Extraction Logic
# -----------------------------
class NLPFeatures:
    """Extracts features using spaCy NLP"""

    def __init__(self, text: str):
        self.text = text
        self.doc = nlp(text)

    def entity_count(self):
        return len(self.doc.ents)

    def named_entities(self):
        return list(set([ent.label_ for ent in self.doc.ents]))

    def noun_count(self):
        return sum(1 for token in self.doc if token.pos_ == "NOUN")

    def verb_count(self):
        return sum(1 for token in self.doc if token.pos_ == "VERB")

    def suspicious_keywords(self):
        suspicious = ['verify', 'login', 'account', 'urgent', 'password', 'click', 'update', 'bank']
        return any(word in self.text.lower() for word in suspicious)

    def get_features(self):
        return {
            'Important Named Entity Count': self.entity_count(),
            'Named Entity Types': ', '.join(self.named_entities()),
            'Noun Count': self.noun_count(),
            'Verb Count': self.verb_count(),
            'Contains Suspicious Keywords': self.suspicious_keywords()
        }

# ----------------------
# Load DistilBERT Model and Tokenizer
# ----------------------
@st.cache_resource(show_spinner=True)
def load_bert_model():
    model_path = r"C:\Users\Lenovo\Documents\Mini Project\Email Phishing Detection Mini Projects\model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_bert_model()

# ----------------------
# BERT-based Classifier
# ----------------------
def classify_email_bert(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return "Phishing" if pred == 1 else "Legitimate"

# ----------------------
# Streamlit App Layout
# ----------------------
st.set_page_config(page_title="Email Phishing Detection Using NLP and BERT", layout="wide")
st.title("📧 Email Phishing Detection Using NLP and BERT")
st.markdown("Paste any **email content** below to extract **NLP features** and classify it using a **BERT model**.")

email_text = st.text_area("✉️ Paste your email text here:", height=300)

if st.button("🔍 Analyze & Classify"):
    if email_text.strip() == "":
        st.warning("Please enter some email content.")
    else:
        # Classify Using BERT
        try:
            classification = classify_email_bert(email_text)

            st.subheader("🚨 Classification Result")
            if classification == "Phishing":
                st.error("⚠️ This email is likely **Phishing**.")
            else:
                st.success("✅ This email appears to be **Legitimate**.")
        except Exception as e:
            st.error(f"❌ Error while classifying email: {e}")
            
