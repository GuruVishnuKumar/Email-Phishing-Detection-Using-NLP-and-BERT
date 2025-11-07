# 📧 Email Phishing Detection using NLP and BERT  

This project focuses on detecting phishing emails using Natural Language Processing (NLP) and BERT (Bidirectional Encoder Representations from Transformers).  
A Streamlit-based web interface allows users to input email text or upload email files to classify them as **Phishing** or **Legitimate**.  

---

## 🚀 Project Overview  

Phishing attacks are one of the most common forms of cybercrime, tricking users into revealing sensitive information.  
Traditional systems rely on blacklists and keyword matching, which fail to detect new or contextually deceptive phishing emails.  

To address this, this project leverages **NLP** for text processing and **BERT** for contextual word understanding, ensuring accurate classification of phishing emails.  

---

## ⚙️ Workflow  

1. **NLP Preprocessing** – The system first applies NLP techniques to clean the email content, split it into words, and normalize the text to remove noise.  
2. **BERT Embeddings** – The processed words are transformed into embeddings using BERT, which captures the meaning and relationships of words in context.  
3. **BERT Analysis & Classification** – BERT analyzes these embeddings and classifies the email as phishing or legitimate based on learned contextual patterns.  
4. **Streamlit Web UI** – The user can input or upload emails through the Streamlit interface to instantly view the classification result.  

---

## 🧠 Key Features  

- Uses **BERT** for contextual understanding of emails  
- Integrated **NLP preprocessing pipeline** for text cleaning and tokenization  
- **Streamlit-based** user-friendly frontend  
- Detects phishing attacks with improved accuracy compared to traditional methods  

---

## 🧰 Technologies Used  

- **Python 3.x**  
- **NLP (Natural Language Processing)**  
- **BERT (Transformer Model)**  
- **Streamlit**  
- **Pandas, NumPy, Scikit-learn**  

---

## 🖥️ How to Run  

1. **Clone this repository**  
   ```bash
   git clone https://github.com/yourusername/phishing-email-detection.git
   cd phishing-email-detection
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt


3. **Run the Streamlit app**
    ```bash
    streamlit run app.py


4. **Open the browser**
    ```bash
    Visit http://localhost:8501

## 📊 Project Flow Diagram

Email Input → NLP Preprocessing → BERT Embeddings → BERT Classification → Result (Phishing / Legitimate)


👨‍💻 Author

E R Guruvishnukumar.
